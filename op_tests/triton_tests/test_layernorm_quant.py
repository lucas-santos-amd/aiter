import triton
import torch
import torch.nn.functional as F
import pytest
import aiter
from aiter.ops.triton.norm import (
    layernorm2d_fwd_with_dynamicquant,
    layernorm2d_fwd_with_smoothquant,
)

import random

seed = random.randint(0, 2**32 - 1)
# seed = 0

import triton.language as tl


@triton.jit
def broadcast_div_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    N,
    stride_am,
    stride_an,
    stride_cm,
    stride_cn,
    stride_bm,
    BLOCK_N: tl.constexpr,
):

    pid_m = tl.program_id(0)  # bloco processando a linha m

    # Offset do inicio da linha m
    row_offset_A = A_ptr + pid_m * stride_am
    row_offset_C = C_ptr + pid_m * stride_cm

    # B e 1D: shape (M,)
    b = tl.load(B_ptr + pid_m * stride_bm)

    # Processar a linha em blocos de tamanho BLOCK_N
    for col_start in range(0, N, BLOCK_N):
        offsets_n = col_start + tl.arange(0, BLOCK_N)
        mask = offsets_n < N

        # Pointers para A e C
        A_ptrs = row_offset_A + offsets_n * stride_an
        C_ptrs = row_offset_C + offsets_n * stride_cn

        # Load A
        A = tl.load(A_ptrs, mask=mask)

        # b_recip = 1 / b
        # C = A * b_recip
        C = A / b

        # Store result
        tl.store(C_ptrs, C.to(C_ptr.type.element_ty), mask=mask)


def broadcast_div(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # assert A.ndim == 2 and B.ndim == 1, "A must be 2D and B must be 1D"
    # assert A.shape[0] == B.shape[0], "A and B must have the same first dimension"
    # assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA device."

    M, N = A.shape
    C = torch.empty_like(A)
    C = torch.empty(A.shape, dtype=torch.int8, device="cuda")

    # BLOCK_N = 128
    BLOCK_N = triton.next_power_of_2(N)

    grid = (M,)

    broadcast_div_kernel[grid](
        A,
        B,
        C,
        N,
        A.stride(0),
        A.stride(1),
        C.stride(0),
        C.stride(1),
        B.stride(0),
        BLOCK_N=BLOCK_N,
    )

    return C


def get_dtype_max(dtype):
    try:
        dtypeMax = torch.finfo(dtype).max
    except:
        dtypeMax = torch.iinfo(dtype).max
    return dtypeMax


def pertoken_quant(
    x,
    scale=None,
    x_scale=None,  # smooth_scale
    scale_dtype=torch.float32,
    quant_dtype=torch.int8,
    dtypeMax=None,
):
    x = x.to(torch.float32)
    if x_scale is None:
        hidden_states = x
    else:
        # smooth quant
        hidden_states = x * x_scale
    # [m, 1]
    per_token_amax, _ = torch.max(input=torch.abs(hidden_states), dim=-1, keepdim=True)

    if dtypeMax is None:
        dtypeMax = get_dtype_max(quant_dtype)

    per_token_scale = scale
    if scale is None:
        per_token_scale = per_token_amax / dtypeMax
        per_token_scale[per_token_scale == 0] = 1

    # quant hidden_states
    y = (hidden_states / per_token_scale).to(dtype=quant_dtype)
    # y = (hidden_states / per_token_scale)

    # y = torch.round(hidden_states / per_token_scale)
    # print((y > 127).sum(), (y < -128).sum())
    # y = torch.clamp(y, -128, 127).to(torch.int8)
    # y = torch.clamp(y, -128, 127)
    y_scale = per_token_scale.to(scale_dtype)
    return y, y_scale, per_token_amax


# def torch_layernorm(x, g, b, out_dtype=torch.float16, epsilon=1e-6):
# M, N = x.shape
# x_f32 = x.float()
# g_f32 = g.float()
# b_f32 = b.float()

# mean = torch.mean(x_f32, dim=-1, keepdim=True)  # shape: (M, 1)
# var = torch.var(x_f32, dim=-1, unbiased=False, keepdim=True)  # shape: (M, 1)
# # inv_std = 1.0 / torch.sqrt(var + epsilon)  # shape: (M, 1)
# inv_std = torch.rsqrt(var + epsilon)  # shape: (M, 1)

# norm_x = (x_f32 - mean) * inv_std  # shape: (M, N)
# out = norm_x * g_f32 + b_f32  # broadcast g and b from (N,) to (M, N)

# return out.to(out_dtype)


def run_torch(
    input, weight, bias, eps, residual=None, x_scale=None, y_scale_dtype=None
):
    if residual is None:
        residual_out = None
        output = F.layer_norm(
            input=input,
            normalized_shape=(input.shape[-1],),
            weight=weight,
            bias=bias,
            eps=eps,
        )
        # output = torch_layernorm(input, weight, bias, out_dtype=torch.float16, epsilon=eps)
    else:
        residual_out = input + residual
        output = F.layer_norm(
            input=residual_out,
            normalized_shape=(input.shape[-1],),
            weight=weight,
            bias=bias,
            eps=eps,
        )
    aux = output
    if y_scale_dtype is None:
        y_scale = None
    else:
        # output, y_scale = aiter.pertoken_quant(
        # output, x_scale=x_scale, quant_dtype=torch.int8
        # )
        output, y_scale, row_max = pertoken_quant(
            output, x_scale=x_scale, quant_dtype=torch.int8
        )
    # return output, residual_out, y_scale
    return output, residual_out, y_scale, aux, row_max


def run_triton(
    input, weight, bias, eps, residual=None, x_scale=None, y_scale_dtype=None
):
    aux = None
    if x_scale is None:
        y_scale = torch.empty(input.shape[0], 1, dtype=y_scale_dtype, device="cuda")
        output = torch.empty(input.shape, dtype=torch.int8, device="cuda")
        # output = torch.empty(input.shape, dtype=torch.float32, device="cuda")
        if residual is None:
            residual_out = None
            _, aux = layernorm2d_fwd_with_dynamicquant(
                output, input, y_scale, weight, bias, eps
            )
        elif residual is not None:
            residual_out = torch.empty_like(input)
            layernorm2d_fwd_with_add_dynamicquant(
                output, input, residual, residual_out, y_scale, weight, bias, eps
            )
    else:
        y_scale = torch.empty(input.shape[0], 1, dtype=y_scale_dtype, device="cuda")
        output = torch.empty(input.shape, dtype=torch.int8, device="cuda")
        if residual is None:
            residual_out = None
            layernorm2d_fwd_with_smoothquant(
                output, input, x_scale, y_scale, weight, bias, eps
            )
        elif residual is not None:
            residual_out = torch.empty_like(input)
            layernorm2d_fwd_with_add_smoothquant(
                output,
                input,
                residual,
                residual_out,
                x_scale,
                y_scale,
                weight,
                bias,
                eps,
            )

    return output, residual_out, y_scale, aux


def get_vals():

    vals = [
        (1823, 781),
        # (2, 128),
        # (1, 4),
        # (128, 2),
        # (1, 128),
        # (8192, 8192),
        # (4096, 8192),
        # (359, 1),
        # (1, 359),
        # (1, 131072),
        # (1, 89999),
        # (10000, 10000),
        # (3, 7),
    ]

    # Test cases for the CK unit tests
    # vals += [
    # (m, n)
    # for m in [1, 2, 4, 8, 16, 32, 64, 128, 256]
    # for n in [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    # ]
    return vals


# pytest
# @pytest.mark.parametrize("dtype_str", ["fp32", "fp16", "bf16"])
# @pytest.mark.parametrize("scale_dtype_str", ["fp32"])
# @pytest.mark.parametrize(
# "M, N",
# [(shape) for shape in get_vals()],
# )
# def test_layernorm_smoothquant(M, N, dtype_str, scale_dtype_str, eps=1e-5):
# arg_to_torch_dtype = {
# "fp16": torch.float16,
# "bf16": torch.bfloat16,
# "fp32": torch.float32,
# }
# dtype = arg_to_torch_dtype[dtype_str]
# scale_dtype = arg_to_torch_dtype[scale_dtype_str]
# # torch.manual_seed(0)
# torch.manual_seed(seed)

# x = torch.randn(M, N, device="cuda", dtype=dtype)
# w_shape = (N,)
# b = torch.rand(w_shape, device="cuda", dtype=dtype)
# w = torch.rand(w_shape, device="cuda", dtype=dtype)
# x_scale = torch.rand(w_shape, device="cuda", dtype=scale_dtype)

# y_torch, _, y_scale_torch = run_torch(
# x, w, b, eps, x_scale=x_scale, y_scale_dtype=scale_dtype
# )
# y_triton, _, y_scale_triton = run_triton(
# x, w, b, eps, x_scale=x_scale, y_scale_dtype=scale_dtype
# )

# xq_dequant = y_triton.to(torch.int32) * y_scale_triton
# xq_dequant = xq_dequant.to(dtype)
# ref_xq_dequant = y_torch.to(torch.int32) * y_scale_torch
# ref_xq_dequant = xq_dequant.to(dtype)

# if dtype == torch.float32:
# atol = 1e-5
# rtol = 1e-5
# else:
# atol = 1e-2
# rtol = 1e-2

# triton.testing.assert_close(xq_dequant, ref_xq_dequant, atol=atol, rtol=rtol)
# triton.testing.assert_close(y_triton, y_torch, atol=1, rtol=0)
# triton.testing.assert_close(y_scale_triton, y_scale_torch, atol=1e-3, rtol=1e-3)


# @pytest.mark.parametrize("dtype_str", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("dtype_str", ["fp32"])
@pytest.mark.parametrize("scale_dtype_str", ["fp32"])
@pytest.mark.parametrize(
    "M, N",
    [(shape) for shape in get_vals()],
)
def test_layernorm_dynamicquant(M, N, dtype_str, scale_dtype_str, eps=1e-3):
    arg_to_torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = arg_to_torch_dtype[dtype_str]
    scale_dtype = arg_to_torch_dtype[scale_dtype_str]
    # torch.manual_seed(0)
    torch.manual_seed(seed)

    x = torch.randn(M, N, device="cuda", dtype=dtype)
    # x = torch.tensor([[0.0001, 10000, 0.0001, 10000],
    # [0.0001, 10000, 0.0001, 10000],
    # [0.0001, 10000, 0.0001, 10000],
    # [0.0001, 10000, 0.0001, 10000]], device="cuda", dtype=dtype)
    w_shape = (N,)
    b = torch.rand(w_shape, device="cuda", dtype=dtype)
    w = torch.rand(w_shape, device="cuda", dtype=dtype)

    # forward pass
    # y_torch, _, y_scale_torch = run_torch(x, w, b, eps, y_scale_dtype=scale_dtype)
    y_torch, _, y_scale_torch, aux_torch, row_max_torch = run_torch(
        x, w, b, eps, y_scale_dtype=scale_dtype
    )
    # y_triton, _, y_scale_triton = run_triton(x, w, b, eps, y_scale_dtype=scale_dtype)
    y_triton, _, y_scale_triton, aux_triton = run_triton(
        x, w, b, eps, y_scale_dtype=scale_dtype
    )

    # aux_torch = F.layer_norm(
    # input=x,
    # normalized_shape=(x.shape[-1],),
    # weight=w,
    # bias=b,
    # eps=eps,
    # )
    # y_triton, y_scale_triton, row_max_triton = pertoken_quant(aux_triton, scale=y_scale_triton, x_scale=None)

    xq_dequant = y_triton.to(torch.int32) * y_scale_triton
    xq_dequant = xq_dequant.to(dtype)
    ref_xq_dequant = y_torch.to(torch.int32) * y_scale_torch
    ref_xq_dequant = xq_dequant.to(dtype)

    if dtype == torch.float32:
        atol = 1e-5
        rtol = 1e-5
    else:
        atol = 1e-2
        rtol = 1e-2

    # y_triton_2 = broadcast_div(aux_triton, y_scale_triton)

    # torch.set_printoptions(precision=6)
    # threshold = 120
    # print("\nLayerNorm outputs where the final result didn't match:")
    # print("Triton:", aux_triton[torch.abs(y_triton - y_torch) > threshold])
    # print("Torch:", aux_torch[torch.abs(y_triton - y_torch) > threshold])
    # print("\nScales where the final result didn't match:")
    # print(
    # "Triton:",
    # y_scale_torch[(torch.abs(y_triton - y_torch) > threshold).any(dim=1)].view(-1),
    # )
    # print(
    # "Torch:",
    # y_scale_triton[(torch.abs(y_triton - y_torch) > threshold).any(dim=1)].view(-1),
    # )
    # print("\nQuantized outputs that didn't match:")
    # print("Triton:", y_triton[torch.abs(y_triton - y_torch) > threshold])
    # print("Torch:", y_torch[torch.abs(y_triton - y_torch) > threshold])

    # index_cond = (torch.abs(y_triton - y_torch) > threshold)
    # indexes = torch.nonzero(index_cond)
    # print("\nIndexes ", indexes)
    # triton.testing.assert_close(xq_dequant, ref_xq_dequant, atol=atol, rtol=rtol)
    triton.testing.assert_close(y_triton, y_torch, atol=1, rtol=0)
    # triton.testing.assert_close(y_triton_2, y_torch, atol=1, rtol=0)
    # triton.testing.assert_close(y_scale_triton, y_scale_torch, atol=1e-3, rtol=1e-3)
    # triton.testing.assert_close(aux_torch, aux_triton, atol=atol, rtol=rtol)
