# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
import torch
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


@triton.jit
def _static_per_tensor_quant_fp8_i8_kernel(
    qx_ptr,
    x_in_ptr,
    scale_in_ptr,
    cols: int,
    x_in_stride_r: int,
    NUM_COL_POW2: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    tl.assume(pid > 0)
    tl.assume(x_in_stride_r > 0)

    offs = pid * x_in_stride_r + tl.arange(0, NUM_COL_POW2)
    mask = tl.arange(0, NUM_COL_POW2) < cols
    x = tl.load(x_in_ptr + offs, mask=mask, cache_modifier=".cg")

    scale = tl.load(scale_in_ptr)
    scale_recip = 1 / scale

    qx = (x * scale_recip).to(qx_ptr.dtype.element_ty)

    tl.store(qx_ptr + offs, qx, mask=mask)


def static_per_tensor_quant_fp8_i8(
    qx: torch.Tensor, x_in: torch.Tensor, scale_in: torch.Tensor
):
    """
    Quantizes tensor using the provided scale to int8 or fp8

    Parameters:
    - qx: Output tensor of same shape as x_in. Must be fp8 or int8 dtype and allocated by the caller
    - x_in: Input tensor of shape (M, N).
    - scale_in: Input Scale tensor of shape (1,) and dtype fp32

    Returns:
    - qx: Quantized output values.
    """
    _LOGGER.info(f"STAIC_PER_TENSOR_QUANT_FP8_I8: x={tuple(x_in.shape)}")
    assert scale_in.numel() == 1  # only single scale value
    rows = x_in.shape[0]
    cols = x_in.shape[1]
    NUM_COL_POW2 = triton.next_power_of_2(cols)
    grid = lambda meta: (rows,)  # noqa: E731
    _static_per_tensor_quant_fp8_i8_kernel[grid](
        qx, x_in, scale_in, cols, x_in.stride(0), NUM_COL_POW2=NUM_COL_POW2
    )

    return qx


@triton.jit
def _dynamic_per_tensor_quant_fp8_i8_kernel(
    x_in_ptr,
    scale_out_ptr,
    cols: int,
    x_in_stride_r: int,
    NUM_COL_POW2: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    tl.assume(pid > 0)
    tl.assume(x_in_stride_r > 0)

    offs = pid * x_in_stride_r + tl.arange(0, NUM_COL_POW2)
    mask = tl.arange(0, NUM_COL_POW2) < cols
    x = tl.load(x_in_ptr + offs, mask=mask, cache_modifier=".cg")

    m = tl.max(tl.abs(x))
    tl.atomic_max(scale_out_ptr, m / DTYPE_MAX, sem="relaxed")


def dynamic_per_tensor_quant_fp8_i8(
    qx: torch.Tensor, x_in: torch.Tensor, scale_out: torch.Tensor
):
    """
    Calculate per tensor scale and then uses the scale to quantize input tensor to fp8 or int8

    Parameters:
    - x_in: Input tensor of shape (M, N).
    - qx: Output tensor of same shape as x_in. Must be fp8 or int8 dtype and allocated by the caller
    - scale_out: Output scale tensor of shape (1,), dtype fp32 and allocated by the caller

    Returns:
    - qx: Quantized output values of shape (M, N) with dtype fp8 or int8
    - scale_out: Single scale value of shape (1,)
    """
    _LOGGER.info(f"DYNAMIC_PER_TENSOR_QUANT_FP8_I8: x={tuple(x_in.shape)}")
    rows = x_in.shape[0]
    cols = x_in.shape[1]
    NUM_COL_POW2 = triton.next_power_of_2(cols)
    grid = lambda meta: (rows,)  # noqa: E731
    _dynamic_per_tensor_quant_fp8_i8_kernel[grid](
        x_in,
        scale_out,
        cols,
        x_in.stride(0),
        NUM_COL_POW2=NUM_COL_POW2,
        DTYPE_MAX=(
            torch.finfo(qx.dtype).max
            if torch.is_floating_point(qx)
            else torch.iinfo(qx.dtype).max
        ),
    )

    _static_per_tensor_quant_fp8_i8_kernel[grid](
        qx, x_in, scale_out, cols, x_in.stride(0), NUM_COL_POW2=NUM_COL_POW2
    )

    return qx, scale_out


@triton.jit
def _dynamic_per_token_quant_fp8_i8_kernel(
    qx_ptr,
    scale_out_ptr,
    x_in_ptr,
    cols: int,
    x_in_stride_r: int,
    NUM_COL_POW2: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    tl.assume(pid > 0)
    tl.assume(x_in_stride_r > 0)

    offs = pid * x_in_stride_r + tl.arange(0, NUM_COL_POW2)
    mask = tl.arange(0, NUM_COL_POW2) < cols
    x = tl.load(x_in_ptr + offs, mask=mask, cache_modifier=".cg")

    m = tl.max(tl.abs(x), axis=-1)
    scale_out = m.to(tl.float32) / DTYPE_MAX
    scale_recip = 1 / scale_out

    qx = x * scale_recip
    qx = qx.to(qx_ptr.dtype.element_ty)

    scale_offs = pid
    tl.store(scale_out_ptr + scale_offs, scale_out)

    tl.store(qx_ptr + offs, qx, mask=mask, cache_modifier=".cs")


def dynamic_per_token_quant_fp8_i8(
    qx: torch.Tensor,
    x_in: torch.Tensor,
    scale_out: torch.Tensor,
):
    """
    Quantizes tensor using the provided scale

    Parameters:
    - x_in: Input tensor of shape (M, N).
    - dtype_max: Optional parameter which specifies the max value of the dtype of x_in.
    - qx: Output tensor of same shape as x_in. Must be fp8 dtype and allocated by the caller
    - scale_out: Output scale tensor of shape (M,) dtype fp32 and allocated by the caller

    Returns:
    - qx: Quantized output values.
    - scale_out: Scale tensor of shape (M, )
    """
    _LOGGER.info(f"DYNAMIC_PER_TOKEN_QUANT_FP8_I8: x={tuple(x_in.shape)}")
    rows = x_in.shape[0]
    cols = x_in.shape[1]
    NUM_COL_POW2 = triton.next_power_of_2(cols)
    grid = lambda meta: (rows,)  # noqa: E731
    _dynamic_per_token_quant_fp8_i8_kernel[grid](
        qx,
        scale_out,
        x_in,
        cols,
        x_in.stride(0),
        NUM_COL_POW2=NUM_COL_POW2,
        DTYPE_MAX=(
            torch.finfo(qx.dtype).max
            if torch.is_floating_point(qx)
            else torch.iinfo(qx.dtype).max
        ),
    )

    return qx, scale_out


@triton.jit
def _mxfp4_quant_op(
    x,
    BLOCK_SIZE_N,
    BLOCK_SIZE_M,
    MXFP4_QUANT_BLOCK_SIZE,
):
    """
    Converts given x (in fp32) to mxfp4 format.
    x: [BLOCK_SIZE_M, BLOCK_SIZE_N], fp32

    """
    EXP_BIAS_FP32: tl.constexpr = 127
    EXP_BIAS_FP4: tl.constexpr = 1
    EBITS_F32: tl.constexpr = 8
    EBITS_FP4: tl.constexpr = 2
    MBITS_F32: tl.constexpr = 23
    MBITS_FP4: tl.constexpr = 1

    max_normal: tl.constexpr = 6
    min_normal: tl.constexpr = 1

    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    x = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE)
    # Calculate scale
    amax = tl.max(tl.abs(x), axis=-1, keep_dims=True)
    amax = amax.to(tl.int32, bitcast=True)
    amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(tl.float32, bitcast=True)
    scale_e8m0_unbiased = tl.log2(amax).floor() - 2
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)

    # blockscale_e8m0
    bs_e8m0 = scale_e8m0_unbiased.to(tl.uint8) + 127  # in fp32, we have 2&(e - 127)

    quant_scale = tl.exp2(-scale_e8m0_unbiased)

    # Compute quantized x
    qx = x * quant_scale

    # Convert quantized fp32 tensor to uint32 before converting to mxfp4 format
    # Note: MXFP4  S:1-bit, E:2-bit, M:1-bit
    #   Zeros: S000 -> +/-0
    #   Denormal Numbers: S001 -> +/- 0.5
    #   Normal Numbers:
    #           S010 -> +/- 1.0
    #           S011 -> +/- 1.5
    #           S100 -> +/- 2.0
    #           S101 -> +/- 3.0
    #           S110 -> +/- 4.0
    #           S111 -> +/- 6.0
    qx = qx.to(tl.uint32, bitcast=True)

    # Extract sign
    s = qx & 0x80000000
    # Set everything to positive, will add sign back at the end
    qx = qx ^ s

    qx_fp32 = qx.to(tl.float32, bitcast=True)
    tl.device_print("qx_fp32", qx_fp32)
    saturate_mask = qx_fp32 >= max_normal
    denormal_mask = (not saturate_mask) & (qx_fp32 < min_normal)
    normal_mask = not (saturate_mask | denormal_mask)

    # Denormal numbers
    denorm_exp: tl.constexpr = (
        (EXP_BIAS_FP32 - EXP_BIAS_FP4) + (MBITS_F32 - MBITS_FP4) + 1
    )
    denorm_mask_int: tl.constexpr = denorm_exp << MBITS_F32
    denorm_mask_float: tl.constexpr = tl.cast(denorm_mask_int, tl.float32, bitcast=True)

    denormal_x = qx_fp32 + denorm_mask_float
    denormal_x = denormal_x.to(tl.uint32, bitcast=True)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(tl.uint8)

    # Normal numbers
    normal_x = qx
    # resulting mantissa is odd
    mant_odd = (normal_x >> (MBITS_F32 - MBITS_FP4)) & 1
    # update exponent, rounding bias part 1
    val_to_add = ((EXP_BIAS_FP4 - EXP_BIAS_FP32) << MBITS_F32) + (1 << 21) - 1
    normal_x += val_to_add
    # rounding bias part 2
    normal_x += mant_odd
    # take the bits!
    normal_x = normal_x >> (MBITS_F32 - MBITS_FP4)
    normal_x = normal_x.to(tl.uint8)

    # Merge results
    e2m1_value = tl.full(qx.type.get_block_shapes(), 0x7, dtype=tl.uint8)
    e2m1_value = tl.where(normal_mask, normal_x, e2m1_value)
    e2m1_value = tl.where(denormal_mask, denormal_x, e2m1_value)
    # add sign back
    sign_lp = s >> (MBITS_F32 + EBITS_F32 - MBITS_FP4 - EBITS_FP4)
    sign_lp = sign_lp.to(tl.uint8)
    e2m1_value = e2m1_value | sign_lp

    # print(f"{e2m1_value}")
    # last_val = e2m1_value.handle.data[MXFP4_QUANT_BLOCK_SIZE-1]
    last_val = e2m1_value
    tl.device_print("last value = {}", last_val)

    # print(e2m1_value)
    # print("teste")

    e2m1_value = tl.reshape(
        e2m1_value, [BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE // 2, 2]
    )
    evens, odds = tl.split(e2m1_value)
    x_fp4 = evens | (odds << 4)
    x_fp4 = x_fp4.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2)

    return x_fp4, bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS)


@triton.heuristics(
    {
        "EVEN_M_N": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0
        and args["N"] % (args["BLOCK_SIZE_N"] * args["NUM_ITER"]) == 0,
    }
)
@triton.jit
def _dynamic_mxfp4_quant_kernel(
    x_ptr,
    x_fp4_ptr,
    bs_ptr,
    stride_x_m_in,
    stride_x_n_in,
    stride_x_fp4_m_in,
    stride_x_fp4_n_in,
    stride_bs_m_in,
    stride_bs_n_in,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_ITER: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr,
    EVEN_M_N: tl.constexpr,
    SCALING_MODE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    start_n = tl.program_id(1) * NUM_ITER
    # cast strides to int64, in case M*N > max int32
    stride_x_m = tl.cast(stride_x_m_in, tl.int64)
    stride_x_n = tl.cast(stride_x_n_in, tl.int64)
    stride_x_fp4_m = tl.cast(stride_x_fp4_m_in, tl.int64)
    stride_x_fp4_n = tl.cast(stride_x_fp4_n_in, tl.int64)
    stride_bs_m = tl.cast(stride_bs_m_in, tl.int64)
    stride_bs_n = tl.cast(stride_bs_n_in, tl.int64)

    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE

    for pid_n in tl.range(start_n, min(start_n + NUM_ITER, N), num_stages=NUM_STAGES):
        x_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        x_offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        x_offs = x_offs_m[:, None] * stride_x_m + x_offs_n[None, :] * stride_x_n

        if EVEN_M_N:
            x = tl.load(x_ptr + x_offs, cache_modifier=".cg").to(tl.float32)
        else:
            x_mask = (x_offs_m < M)[:, None] & (x_offs_n < N)[None, :]
            x = tl.load(x_ptr + x_offs, mask=x_mask, cache_modifier=".cg").to(
                tl.float32
            )

        out_tensor, bs_e8m0 = _mxfp4_quant_op(
            x, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE
        )

        out_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        out_offs_n = pid_n * BLOCK_SIZE_N // 2 + tl.arange(0, BLOCK_SIZE_N // 2)
        out_offs = (
            out_offs_m[:, None] * stride_x_fp4_m + out_offs_n[None, :] * stride_x_fp4_n
        )

        if EVEN_M_N:
            tl.store(x_fp4_ptr + out_offs, out_tensor)
        else:
            out_mask = (out_offs_m < M)[:, None] & (out_offs_n < (N // 2))[None, :]
            tl.store(x_fp4_ptr + out_offs, out_tensor, mask=out_mask)

        bs_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        bs_offs_n = pid_n * NUM_QUANT_BLOCKS + tl.arange(0, NUM_QUANT_BLOCKS)
        bs_offs = bs_offs_m[:, None] * stride_bs_m + bs_offs_n[None, :] * stride_bs_n
        if EVEN_M_N:
            tl.store(bs_ptr + bs_offs, bs_e8m0)
        else:
            bs_mask = (bs_offs_m < M)[:, None] & (
                bs_offs_n < (N + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE
            )[None, :]
            tl.store(
                bs_ptr + bs_offs,
                bs_e8m0,
                mask=bs_mask,
            )


def dynamic_mxfp4_quant(
    x: torch.Tensor, scaling_mode: str = "even"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to MX FP4 format.

    Args:
        x: The input tensor, typically fp16 or bf16.
        scaling_mode: The method to calculate MX block scaling.
            - "even" (default): `even_round` in `quark.torch.quantization.utils`.
            - etc.
    Returns:
        A tuple of (x_fp4, blockscale_e8m0).
    """
    _LOGGER.info(f"DYNAMIC_MXFP4_QUANT: x={tuple(x.shape)}")
    # Assume x is 2D-Tensor for now
    M, N = x.shape

    assert (N // 2) % 2 == 0

    # This is fixed by spec for MXFP4. Do not tune this.
    MXFP4_QUANT_BLOCK_SIZE = 32
    x_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
    blockscale_e8m0 = torch.empty(
        ((N + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE, M),
        dtype=torch.uint8,
        device=x.device,
    ).T

    # for large N values
    if M <= 32:
        NUM_ITER = 1
        BLOCK_SIZE_M = triton.next_power_of_2(M)
        BLOCK_SIZE_N = 32
        NUM_WARPS = 1
        NUM_STAGES = 1
    else:
        NUM_ITER = 4
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        NUM_WARPS = 4
        NUM_STAGES = 2

        if N <= 16384:
            BLOCK_SIZE_M = 32
            BLOCK_SIZE_N = 128

    # for small N values
    if N <= 1024:
        NUM_ITER = 1
        NUM_STAGES = 1
        NUM_WARPS = 4
        BLOCK_SIZE_N = min(256, triton.next_power_of_2(N))
        # BLOCK_SIZE_N needs to be multiple of 32
        BLOCK_SIZE_N = max(32, BLOCK_SIZE_N)
        BLOCK_SIZE_M = min(8, triton.next_power_of_2(M))

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N * NUM_ITER),
    )

    _dynamic_mxfp4_quant_kernel[grid](
        x,
        x_fp4,
        blockscale_e8m0,
        *x.stride(),
        *x_fp4.stride(),
        *blockscale_e8m0.stride(),
        M=M,
        N=N,
        MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
        SCALING_MODE=0,
        NUM_ITER=NUM_ITER,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        NUM_STAGES=NUM_STAGES,
        num_warps=NUM_WARPS,
        waves_per_eu=0,
        num_stages=1,
    )

    return (x_fp4, blockscale_e8m0)
