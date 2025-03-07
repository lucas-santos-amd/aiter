# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import pytest

@triton.heuristics({
    'EVEN_K':
    lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0, 'GRID_MN':
    lambda args: triton.cdiv(args['M'], args['BLOCK_SIZE_M']) * triton.cdiv(args['N'], args['BLOCK_SIZE_N'])
})
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    a_scale_ptr,
    b_scale_ptr,
    bias_ptr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    GRID_MN: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    NUM_XCDS: tl.constexpr = 8

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # TODO(vgokhale): Add XCD remapping.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    ## pid remapping on xcds
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    # Compute current XCD and local pid within the XCD
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    # Calculate new pid based on the new grouping
    # Note that we need to consider the following two cases:
    # 1. the current pid is on a tall xcd
    # 2. the current pid is on a short xcd
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m > 0)
    tl.assume(pid_n > 0)

    # Create pointers for first block of A and B input matrices
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    if APPLY_SCALE:
        # TODO: test with guard load instead of masks
        offs_a_scale = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
        offs_b_scale = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
        a_scale = tl.load(a_scale_ptr + offs_a_scale, mask=offs_a_scale < M, other=0.0)
        b_scale = tl.load(b_scale_ptr + offs_b_scale, mask=offs_b_scale < N, other=0.0)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # Type conversion to support mixed precision GEMMs where b is lower precision than a
        # b = b.to(a_ptr.type.element_ty)
        accumulator += tl.dot(a, b, input_precision="ieee")

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # Apply scale to recover dynamic range reduced due to lower precision inputs.
    if APPLY_SCALE:
        accumulator = accumulator * a_scale * b_scale

    c = accumulator.to(c_ptr.type.element_ty)

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# Activation function.
# @triton.jit
# def leaky_relu(x):
    # x = x + 1
    # return tl.where(x >= 0, x, 0.01 * x)


# Wrapper for gemm kernel.
def gemm_a8w8(x, w, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    # Check constraints.
    assert x.shape[1] == w.shape[1], "Incompatible dimensions!!!"
    # assert (x.element_size()
            # >= w.element_size()), "Mixed dtype GEMMs are only supported when data type of a is bigger than b!!!"
    # assert (x.is_floating_point() == w.is_floating_point()
            # ), "GEMMs between float and integer type tensors are not supported!!!"
    M, K = x.shape
    # N, K = w.shape
    w = w.t().contiguous()
    K, N = w.shape

    out = torch.empty((M,N), dtype=dtype)


    BLOCK_SIZE_M = 256
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 4
    waves_per_eu = 2
    kpack = 1
    matrix_instr_nonkdim = 16
    num_warps = 8
    num_stages = 2
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )
    matmul_kernel[grid](
        x,
        w,
        out,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        out.stride(0),
        out.stride(1),
        a_scale,
        b_scale,
        bias,
        HAS_BIAS = bias is not None,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
        waves_per_eu=waves_per_eu,
        kpack=kpack,
        matrix_instr_nonkdim=matrix_instr_nonkdim,
        num_warps=num_warps,
        num_stages=num_stages
    )
    return out


def is_cdna4():
    return triton.runtime.driver.active.get_current_target().arch == 'gfx950'


e5m2_type = torch.float8_e5m2 if is_cdna4() else torch.float8_e5m2fnuz
e4m3_type = torch.float8_e4m3fn if is_cdna4() else torch.float8_e4m3fnuz

name_to_torch_types = {
    'int8': torch.int8,
    'int32': torch.int32,
    'fp16': torch.float16,
    'fp32': torch.float32,
    'bf16': torch.bfloat16,
    'fp8e5': e5m2_type,
    'fp8e4': e4m3_type,
}

dtype_max = {
    dtype: (torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)).max
    for dtype in [
        e5m2_type,
        e4m3_type,
        torch.int8,
    ]
}


def dtype_is_8_bit(dtype):
    return (dtype is e5m2_type) or \
           (dtype is e4m3_type) or \
           (dtype is torch.int8)


def gen_input(M, N, dtype, needTrans, seed, device='cuda'):
    torch.manual_seed(seed)

    if needTrans:
        raw_data = torch.randn((N, M), dtype=torch.float32, device='cuda').T
    else:
        raw_data = torch.randn((M, N), dtype=torch.float32, device='cuda')
    scale = None
    if dtype_is_8_bit(dtype):
        max_val = torch.max(torch.abs(raw_data))
        scale = max_val / dtype_max[dtype]
        raw_data = raw_data / scale

    input = raw_data.to(dtype)
    input_f32 = input.to(torch.float32)

    return input, input_f32, scale


def get_x_vals():
    x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in range(1, 9)]

    x_vals += [(4864, 4096, 8192), (9728, 8192, 65536), (4864, 8192, 4160)]

    return x_vals


# Unit tests
#TODO(vgokhale): Test activation.
# yapf: disable
@pytest.mark.parametrize(
    "M, N, K, in_dtype_a, in_dtype_b, out_dtype, col_a, col_b",
    [(*shape, in_dtype_a, in_dtype_b, out_dtype, col_a, col_b)
     for shape in get_x_vals()
     for in_dtype_a, in_dtype_b, out_dtype in [
        ('fp16', 'fp16', 'fp16'),   ('bf16', 'bf16', 'bf16'),   ('fp32', 'fp32', 'fp32'),
        ('fp8e4', 'fp8e4', 'fp16'), ('fp8e5', 'fp8e5', 'fp16'), ('fp16', 'fp8e4', 'fp16'),
        ('fp16', 'fp8e5', 'fp16'),  ('bf16', 'fp8e4', 'bf16'),  ('bf16', 'fp8e5', 'bf16'),
        ('int8', 'int8', 'int8'),   ('int8', 'int8', 'int32')]
     # Defines if a matrix is row or column major.
     for col_a in [True, False]
     for col_b in [True, False]])
# yapf: enable
def test_correctness(M, N, K, col_a, col_b, in_dtype_a, in_dtype_b, out_dtype):
    torch_in_dtype_a = name_to_torch_types[in_dtype_a]
    torch_in_dtype_b = name_to_torch_types[in_dtype_b]
    a, a_fp32, a_scale = gen_input(M, K, torch_in_dtype_a, col_a, 1, device='cuda')
    b, b_fp32, b_scale = gen_input(K, N, torch_in_dtype_b, col_b, 2, device='cuda')
    torch_out_dtype = name_to_torch_types[out_dtype]
    c = torch.empty((M, N), device=a.device, dtype=torch_out_dtype)
    # For 8-bit, we have scaled to the dynamic range of the data type.
    # This requires us to compute in fp32 because for e5m2, the range is same as fp16 (e5m10).
    # If we use fp16 it is possible to return infs from the torch.matmul call.
    if dtype_is_8_bit(torch_in_dtype_a) or dtype_is_8_bit(torch_in_dtype_b):
        matmul(a, b, c, a_scale, b_scale, scale_a8_b8=True, activation="")
        torch_output = torch.matmul(a_fp32, b_fp32)
        # Set a_scale to 1.0 if it is not set
        torch_output = torch_output * (a_scale or 1.0) * b_scale
    # For other dtypes, use the same torch matmul as the dtype.
    else:
        matmul(a, b, c, a_scale=None, b_scale=None, scale_a8_b8=False, activation="")
        torch_output = torch.matmul(a.to(torch_in_dtype_a), b.to(torch_in_dtype_b))
    if out_dtype == 'int8':
        torch.testing.assert_close(c.to(torch.float32),
                                   torch_output.to(torch.int8).to(torch.float32), atol=1e-3, rtol=1e-2)
    else:
        torch.testing.assert_close(c, torch_output.to(torch_out_dtype), atol=5e-3, rtol=1e-2)


def get_type(provider):
    res = re.findall(r'\(.*?\)', provider)
    return res[0][1:-1].split('/', 1)


