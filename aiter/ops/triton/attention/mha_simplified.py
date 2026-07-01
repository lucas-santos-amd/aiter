# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
import triton
from packaging.version import Version

from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.device_info import get_num_xcds
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton._gluon_kernels.gfx950.attention.mha import _attn_fwd, _get_config

_LOGGER = AiterTritonLogger()

_USE_INT64_STRIDES = True
_GLUON_SUPPORTED_ARCHS = ("gfx950",)
_TRITON_GE_36 = Version(triton.__version__) >= Version("3.6.0")


def _is_gluon_available() -> bool:
    """True when the naive gfx950 Gluon MHA kernel can run on this device."""
    if not _TRITON_GE_36:
        return False
    try:
        arch = get_arch() or ""
        if not any(supported in arch for supported in _GLUON_SUPPORTED_ARCHS):
            return False
        from triton.experimental import gluon  # noqa: F401

        return True
    except Exception:
        return False


def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    max_seqlen_q: int,
    max_seqlen_k: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
) -> torch.Tensor:
    # Triton cannot specialize on numpy scalar types; ensure native Python int
    max_seqlen_q = int(max_seqlen_q)
    max_seqlen_k = int(max_seqlen_k)

    is_varlen = cu_seqlens_q is not None

    o = torch.zeros((q.shape[:-1] + v.shape[-1:]), dtype=q.dtype, device=q.device)

    if is_varlen:
        # Layout is thd: q/k/v are [total_tokens, num_head, head_dim].
        batch, seqlen_q, num_q_heads = len(cu_seqlens_q) - 1, max_seqlen_q, q.shape[1]
        num_k_heads = k.shape[1]
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
    else:
        # Layout is bshd: q/k/v are [batch, seq_len, num_head, head_dim].
        batch, seqlen_q, num_q_heads = (int(x) for x in q.shape[:-1])
        num_k_heads = k.shape[2]
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
        o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))

    v_head_dim = v.shape[-1]
    # padding for head_dim. Power of 2 or 16
    BLOCK_DMODEL_POW2 = max(triton.next_power_of_2(v_head_dim), 16)

    if config is None:
        config = _get_config(q.dtype)

    grid = lambda META: (  # noqa: E731
        batch * num_q_heads * triton.cdiv(seqlen_q, META["BLOCK_M"]),
    )

    _attn_fwd[grid](
        q,
        k,
        v,
        o,
        *q_strides,
        *k_strides,
        *v_strides,
        *o_strides,
        softmax_scale,
        cu_seqlens_q,
        cu_seqlens_k,
        SEQLEN_Q=max_seqlen_q,
        SEQLEN_K=max_seqlen_k,
        IS_CAUSAL=causal,
        NUM_Q_HEADS=num_q_heads,
        NUM_K_HEADS=num_k_heads,
        BLOCK_DMODEL=v_head_dim,
        BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
        VARLEN=is_varlen,
        BATCH=batch,
        NUM_XCD=get_num_xcds(),
        USE_INT64_STRIDES=_USE_INT64_STRIDES,
        # Soundness precondition: only set when every Q/K/V head-axis stride is
        # a multiple of 8 elements. *_strides[1] are the head-axis strides in
        # both thd and bshd layouts (see strides assembly above).
        HEAD_STRIDE_ALIGNED_8=(
            q_strides[1] % 8 == 0
            and k_strides[1] % 8 == 0
            and v_strides[1] % 8 == 0
        ),
        **config,
    )

    return o


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    config: Optional[dict] = None,
    backend: Optional[str] = None,
) -> torch.Tensor:
    """Forward-only flash attention for the simplified gfx950 kernel.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV
    with fewer heads than Q. The number of Q heads must be divisible by the
    number of KV heads.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        softmax_scale: scaling of QK^T before softmax. Defaults to 1 / sqrt(headdim).
        causal: whether to apply a (bottom-right aligned) causal mask.
        config: [triton only] kernel tuning parameters.
        backend: ``"triton"``, ``"gluon"``, or ``None`` (defaults to ``"triton"``).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
    """
    if backend is None:
        backend = "triton"
    backend = backend.lower()
    assert backend in (
        "triton",
        "gluon",
    ), f"Unknown backend '{backend}', must be 'triton' or 'gluon'"

    _LOGGER.info(
        f"FLASH_ATTN [{backend}]:  q={tuple(q.shape)}  k={tuple(k.shape)}  v={tuple(v.shape)}"
    )
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if backend == "gluon":
        assert (
            _is_gluon_available()
        ), f"Gluon backend requires one of {_GLUON_SUPPORTED_ARCHS}, got '{get_arch()}'"
        from aiter.ops.triton._gluon_kernels.gfx950.attention.mha_gluon import (
            flash_attn_fwd,
        )

        return flash_attn_fwd(q, k, v, causal=causal, sm_scale=softmax_scale)

    head_size_og = q.size(3)
    if head_size_og % 8 != 0:
        q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
        k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
        v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
    out_padded = _flash_attn_forward(
        q,
        k,
        v,
        softmax_scale,
        causal,
        max_seqlen_q=q.shape[1],
        max_seqlen_k=k.shape[1],
        config=config,
    )
    return out_padded[..., :head_size_og]


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    config: Optional[dict] = None,
    backend: Optional[str] = None,
) -> torch.Tensor:
    """Forward-only varlen flash attention for the simplified gfx950 kernel.

    Arguments:
        q: (total_q, nheads, headdim), total query tokens across the batch.
        k: (total_k, nheads_k, headdim), total key tokens across the batch.
        v: (total_k, nheads_k, headdim), total key tokens across the batch.
        cu_seqlens_q: (batch_size + 1,), int32 cumulative query sequence lengths.
        cu_seqlens_k: (batch_size + 1,), int32 cumulative key sequence lengths.
        max_seqlen_q: maximum query sequence length in the batch.
        max_seqlen_k: maximum key sequence length in the batch.
        softmax_scale: scaling of QK^T before softmax. Defaults to 1 / sqrt(headdim).
        causal: whether to apply a (bottom-right aligned) causal mask.
        config: [triton only] kernel tuning parameters.
        backend: ``"triton"``, ``"gluon"``, or ``None`` (defaults to ``"triton"``).
    Return:
        out: (total_q, nheads, headdim).
    """
    if backend is None:
        backend = "triton"
    backend = backend.lower()
    assert backend in (
        "triton",
        "gluon",
    ), f"Unknown backend '{backend}', must be 'triton' or 'gluon'"

    _LOGGER.info(
        f"FLASH_ATTN_VARLEN [{backend}]:  q={tuple(q.shape)}  k={tuple(k.shape)}  v={tuple(v.shape)}"
    )
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if backend == "gluon":
        assert (
            _is_gluon_available()
        ), f"Gluon backend requires one of {_GLUON_SUPPORTED_ARCHS}, got '{get_arch()}'"
        from aiter.ops.triton._gluon_kernels.gfx950.attention.mha_gluon import (
            flash_attn_varlen_fwd,
        )

        return flash_attn_varlen_fwd(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=causal,
            sm_scale=softmax_scale,
        )

    head_size_og = q.size(2)
    if head_size_og % 8 != 0:
        q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
        k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
        v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
    out_padded = _flash_attn_forward(
        q,
        k,
        v,
        softmax_scale,
        causal,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        config=config,
    )
    return out_padded[..., :head_size_og]
