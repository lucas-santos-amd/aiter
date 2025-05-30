import triton
import triton.language as tl
import torch
import torch.nn.functional as F
import pytest
from typing import Optional


def get_dtype_max(dtype):
    if torch.is_floating_point(torch.tensor([], dtype=dtype)):
        return torch.finfo(dtype).max
    else:
        return torch.iinfo(dtype).max


@triton.jit
def _triton_per_token_quant(
    x,
    y_scale_ptr,
    row_max,
    row_idx,
    DTYPE_MAX: tl.constexpr,
):
    """
    #TODO: Add Doc
    """

    scale_out = row_max / DTYPE_MAX
    scale_out = tl.where(scale_out == 0, 1.0, scale_out)

    scale_recip = 1 / scale_out
    qx = x * scale_recip

    tl.store(y_scale_ptr + row_idx, scale_out.to(y_scale_ptr.type.element_ty))
    return qx


@triton.jit
def _quant_layernorm_kernel(
    # Pointers to matrices
    x_ptr,
    y_ptr,
    w_ptr,
    b_ptr,
    x_scale_ptr,
    y_scale_ptr,
    # Auxiliary tensor to store intermediate data
    aux_ptr,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `x_row_stride` is
    # how much to increase `x_ptr` by to get the element one row down.
    x_row_stride,
    y_row_stride,
    aux_row_stride,
    # Matrix dimensions
    n_rows,
    n_cols,
    # Epsilon to avoid division by zero
    eps,
    # Dtype max for quantization
    DTYPE_MAX: tl.constexpr,
    # Meta-parameters
    IS_SMOOTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call layer_norm function
    below

    Applies Layer Normalization over a mini-batch of inputs and quantizes the result.

    Key parameters:
    - X: The input tensor to be normalized with shape (M, N).
    - Y: The output tensor with the same shape as the input one.
    - W: The learnable weights tensor with shape (N, ).
    - B: The learnable bias tensor with shape (N, ).
    - X_scale: The tensor to be multiplied by the LayerNorm output if IS_SMOOTH is true, with shape (n_cols, ).
    - Y_scale: The tensor where the scale for each row will be stored with shape (n_rows, ).
    """
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    x_ptr_start = x_ptr + (row * x_row_stride)
    y_ptr_start = y_ptr + (row * y_row_stride)
    aux_ptr_start = aux_ptr + (row * aux_row_stride)

    loop_num = tl.cdiv(n_cols, BLOCK_SIZE) - 1

    # Calculate mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    loop_num_l = loop_num
    for b in range(0, loop_num_l):
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x_block = tl.load(x_ptr_start + col_offsets).to(tl.float32)  # Unmasked loads
        _mean += x_block

    # For last iteration, do masked load
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_block = tl.load(
        x_ptr_start + col_offsets, mask=col_offsets < n_cols, other=0.0
    ).to(tl.float32)
    _mean += x_block
    mean = tl.sum(_mean, axis=0) / n_cols

    # Calculate variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    loop_num_l = loop_num
    for b in range(0, loop_num_l):
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x_block = tl.load(x_ptr_start + col_offsets).to(tl.float32)  # Unmasked loads
        x_block = x_block - mean
        _var += x_block * x_block

    # For last iteration, do masked load
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_block = tl.load(
        x_ptr_start + col_offsets, mask=col_offsets < n_cols, other=0.0
    ).to(tl.float32)
    x_block = tl.where(col_offsets < n_cols, x_block - mean, 0.0)
    _var += x_block * x_block

    var = tl.sum(_var, axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)

    row_max: tl.float32 = 0.0

    # Normalize and write output temporarily as fp32
    loop_num_l = loop_num
    for b in range(0, loop_num_l):
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        w_block = tl.load(w_ptr + col_offsets)
        b_block = tl.load(b_ptr + col_offsets)
        x_block = tl.load(x_ptr_start + col_offsets).to(tl.float32)
        y_block = (x_block - mean) * rstd
        y_block = y_block * w_block + b_block

        if IS_SMOOTH:
            x_scale_ptrs = x_scale_ptr + col_offsets
            x_scale = tl.load(x_scale_ptrs)
            y_block *= x_scale

        blk_max = tl.max(tl.abs(y_block), axis=-1)
        row_max = max(row_max, blk_max)

        aux_ptrs = aux_ptr_start + col_offsets
        tl.store(aux_ptrs, y_block)

    # For last iteration, do masked load
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    w_block = tl.load(w_ptr + col_offsets, mask=mask, other=0.0)
    b_block = tl.load(b_ptr + col_offsets, mask=mask, other=0.0)
    x_block = tl.load(x_ptr_start + col_offsets, mask=mask, other=0.0).to(tl.float32)
    y_block = tl.where(col_offsets < n_cols, (x_block - mean) * rstd, 0.0)
    y_block = y_block * w_block + b_block

    if IS_SMOOTH:
        x_scale_ptrs = x_scale_ptr + col_offsets
        x_scale = tl.load(x_scale_ptrs, mask=mask, other=0.0)
        y_block *= x_scale

    blk_max = tl.max(tl.abs(y_block), axis=-1)
    row_max = max(row_max, blk_max)

    tl.store(aux_ptr_start + col_offsets, y_block, mask=mask)

    # Apply quantization and write output
    loop_num_l = loop_num
    for b in range(0, loop_num_l):
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        aux_block = tl.load(aux_ptr_start + col_offsets)  # Unmasked loads

        y_block = _triton_per_token_quant(
            aux_block, y_scale_ptr, row_max, row, DTYPE_MAX
        )
        tl.store(y_ptr_start + col_offsets, y_block.to(y_ptr.type.element_ty))

    # For last iteration, do masked load and store
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    aux_block = tl.load(aux_ptr_start + col_offsets, mask=mask, other=0.0)

    y_block = _triton_per_token_quant(aux_block, y_scale_ptr, row_max, row, DTYPE_MAX)

    tl.store(y_ptr_start + col_offsets, y_block.to(y_ptr.type.element_ty), mask=mask)


def layernorm2d_fwd_with_dynamicquant(
    out: torch.Tensor,
    input: torch.Tensor,
    yscale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    epsilon: float = 1e-5,
    x_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    M, N = input.shape

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // input.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))

    xscale = None
    IS_SMOOTH = False
    DTYPE_MAX = get_dtype_max(out.dtype)

    # Auxiliary tensor to store the RMSNorm output as fp32 before applying the quantization when using the blocked approach
    aux = torch.empty(M, N, dtype=torch.float32, device=input.device)

    _quant_layernorm_kernel[(M,)](
        input,
        out,
        weight,
        bias,
        xscale,
        yscale,
        aux,
        input.stride(0),
        out.stride(0),
        aux.stride(0),
        M,
        N,
        epsilon,
        DTYPE_MAX,
        IS_SMOOTH,
        BLOCK_SIZE,
    )

    return out, aux.to(input.dtype)


@triton.jit
def standalone_triton_quant_kernel(
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
    pid_m = tl.program_id(0)
    row_offset_A = A_ptr + pid_m * stride_am
    row_offset_C = C_ptr + pid_m * stride_cm
    b = tl.load(B_ptr + pid_m * stride_bm)
    for col_start in range(0, N, BLOCK_N):
        offsets_n = col_start + tl.arange(0, BLOCK_N)
        mask = offsets_n < N
        A_ptrs = row_offset_A + offsets_n * stride_an
        C_ptrs = row_offset_C + offsets_n * stride_cn
        A = tl.load(A_ptrs, mask=mask)

        b_recip = 1 / b
        C = A * b_recip

        tl.store(C_ptrs, C.to(C_ptr.type.element_ty), mask=mask)


def standalone_triton_quant(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, N = A.shape
    C = torch.empty_like(A)
    C = torch.empty(A.shape, dtype=torch.int8, device="cuda")

    BLOCK_N = triton.next_power_of_2(N)

    grid = (M,)

    standalone_triton_quant_kernel[grid](
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


def torch_pertoken_quant(
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

    y_scale = per_token_scale.to(scale_dtype)
    return y, y_scale


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
        output, y_scale = torch_pertoken_quant(
            output, x_scale=x_scale, quant_dtype=torch.int8
        )
    # return output, residual_out, y_scale
    return output, residual_out, y_scale, aux


def run_triton(
    input, weight, bias, eps, residual=None, x_scale=None, y_scale_dtype=None
):
    aux = None
    if x_scale is None:
        y_scale = torch.empty(input.shape[0], 1, dtype=y_scale_dtype, device="cuda")
        output = torch.empty(input.shape, dtype=torch.int8, device="cuda")
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
        (1823, 781),  # -> FAIL
        # (8192, 8192), # -> FAIL
        # (4096, 8192), # -> FAIL
        # (2, 128),
        # (1, 4),
        # (128, 2),
        # (1, 128),
        # (359, 1),
        # (1, 359),
        # (1, 131072),
        # (1, 89999),
        # (10000, 10000),
    ]

    # Test cases for the CK unit tests, everything works for these
    # vals += [
    # (m, n)
    # for m in [1, 2, 4, 8, 16, 32, 64, 128, 256]
    # for n in [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    # ]
    return vals


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
    torch.manual_seed(0)

    x = torch.randn(M, N, device="cuda", dtype=dtype)

    w_shape = (N,)
    b = torch.rand(w_shape, device="cuda", dtype=dtype)
    w = torch.rand(w_shape, device="cuda", dtype=dtype)

    # forward pass
    # y_torch, _, y_scale_torch = run_torch(x, w, b, eps, y_scale_dtype=scale_dtype)
    y_torch, _, y_scale_torch, x_normed_torch = run_torch(
        x, w, b, eps, y_scale_dtype=scale_dtype
    )
    # y_triton, _, y_scale_triton = run_triton(x, w, b, eps, y_scale_dtype=scale_dtype)
    y_triton, _, y_scale_triton, x_normed_triton = run_triton(
        x, w, b, eps, y_scale_dtype=scale_dtype
    )

    # Passing the triton layernorm output and scale to the torch quant function yields the correct result
    # Just uncomment the line below to verify
    # y_triton, y_scale_triton = torch_pertoken_quant(x_normed_triton, scale=y_scale_triton, x_scale=None)

    # Passing the triton layernorm output and scale to a separate triton quant kernel also returns the correct result
    y_triton_standalone = standalone_triton_quant(x_normed_triton, y_scale_triton)

    if dtype == torch.float32:
        atol = 1e-5
        rtol = 1e-5
    else:
        atol = 1e-2
        rtol = 1e-2

    # This section prints the values for the layernorm outputs, scales and quantized results where
    # the absolute difference between the triton and torch quantized output is greater than a set threshold,
    # showing that the values of the layernorm output and scales are equivalent in the same indexes where
    # the quantized results differ.
    torch.set_printoptions(precision=5)
    threshold = 120
    condition = torch.abs(y_triton - y_torch) > threshold
    print(
        f"\nLayerNorm outputs where the quantized output error is greater than {threshold}:"
    )
    print("Triton:", x_normed_triton[condition])
    print("Torch:", x_normed_torch[condition])
    print(f"\nScales where the quantized output error is greater than {threshold}:")
    print(
        "Triton:",
        y_scale_torch[(condition).any(dim=1)].view(-1),
    )
    print(
        "Torch:",
        y_scale_triton[(condition).any(dim=1)].view(-1),
    )
    print(f"\nQuantized outputs where the error is greater than {threshold}:")
    print("Triton:", y_triton[condition])
    print("Torch:", y_torch[condition])

    # Compares the scales (PASS)
    triton.testing.assert_close(y_scale_triton, y_scale_torch, atol=1e-3, rtol=1e-3)
    # Compares the LayerNorm outputs (PASS)
    triton.testing.assert_close(x_normed_torch, x_normed_triton, atol=atol, rtol=rtol)
    # Compares the standalone quant triton kernel (PASS)
    triton.testing.assert_close(y_triton_standalone, y_torch, atol=1, rtol=0)
    # Compares the layernorm fused with quant triton kernel (FAIL)
    triton.testing.assert_close(y_triton, y_torch, atol=1, rtol=0)
