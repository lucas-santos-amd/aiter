# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
import logging
import triton
from aiter.ops.triton.attention.mha_simplified import (
    _is_gluon_available,
    flash_attn_func,
    flash_attn_varlen_func,
)
from aiter.test_mha_common import (
    attention_ref,
    generate_random_padding_mask,
    generate_qkv,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
DEBUG_MODE = False


def _skip_if_gluon_unsupported(backend: str, head_sz: int, num_k_heads: int):
    if backend != "gluon":
        return
    if not _is_gluon_available():
        pytest.skip("Gluon MHA backend is not available on this architecture")
    # Known-bugged cases: for a padded head dim (head_sz whose value differs from
    # the kernel's padded tile, e.g. head_sz=8 -> padded to 32) the gluon kernel
    # stages K/V through a masked async global->LDS copy that fails to legalize on
    # gfx950 when the key-sequence stride (num_k_heads * head_sz) is not a multiple
    # of 16 elements -- e.g. head_sz=8 with a small number of KV heads
    # (num_k_heads=1). Skip these until the kernel handles padded heads.
    padded_head = head_sz != max(triton.next_power_of_2(head_sz), 32)
    if padded_head and (num_k_heads * head_sz) % 16 != 0:
        pytest.skip(
            "gluon MHA: padded head_dim with a non-16-element-aligned KV stride "
            f"(head_sz={head_sz}, num_k_heads={num_k_heads}) is currently broken"
        )


def _test_mha_impl(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    CAUSAL: bool,
    backend: str,
    dtype=torch.bfloat16,
):
    _skip_if_gluon_unsupported(backend, HEAD_SZ, NUM_K_HEADS)

    torch.manual_seed(20)
    torch.cuda.empty_cache()
    q = torch.randn((BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    k = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    v = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)

    kernel_out = flash_attn_func(q, k, v, causal=CAUSAL, backend=backend)
    if DEBUG_MODE:
        print(f"kernel_out.shape={kernel_out.shape}, kernel_out={kernel_out}")

    torch_out = attention_ref(q, k, v, causal=CAUSAL)
    torch_out, attention_scores, _ = torch_out
    if DEBUG_MODE:
        print(f"torch_out.shape={torch_out.shape}, torch_out={torch_out}")

    torch.testing.assert_close(kernel_out, torch_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("BATCH", [1, 30, 50])
@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(1, 1), (128, 128), (32, 16), (64, 128), (2048, 2048)],
)
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(1, 1), (8, 8), (48, 8)])
@pytest.mark.parametrize("HEAD_SZ", [64, 128])
@pytest.mark.parametrize("CAUSAL", [(True), (False)])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
def test_mha(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    CAUSAL: bool,
    backend: str,
    dtype=torch.bfloat16,
):
    _test_mha_impl(
        BATCH,
        SEQLEN_Q,
        SEQLEN_K,
        NUM_Q_HEADS,
        NUM_K_HEADS,
        HEAD_SZ,
        CAUSAL=CAUSAL,
        backend=backend,
        dtype=dtype,
    )


def _test_mha_varlen_impl(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    CAUSAL: bool,
    backend: str,
    dtype=torch.bfloat16,
):
    _skip_if_gluon_unsupported(backend, HEAD_SZ, NUM_K_HEADS)

    torch.set_printoptions(threshold=10000)
    torch.cuda.empty_cache()
    torch.manual_seed(20)

    q = torch.randn((BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    k = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    v = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    query_padding_mask = generate_random_padding_mask(
        SEQLEN_Q, BATCH, "cuda", mode="random"
    )
    key_padding_mask = generate_random_padding_mask(
        SEQLEN_K, BATCH, "cuda", mode="random"
    )
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        _,
        _,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)

    triton_out = flash_attn_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal=CAUSAL,
        backend=backend,
    )
    triton_out = output_pad_fn(triton_out)
    if DEBUG_MODE:
        print(f"triton_out.shape={triton_out.shape}, triton_out={triton_out}")

    torch_out = attention_ref(
        q,
        k,
        v,
        query_padding_mask=query_padding_mask,
        key_padding_mask=key_padding_mask,
        causal=CAUSAL,
    )
    torch_out, attention_scores, _ = torch_out
    if DEBUG_MODE:
        print(f"torch_out.shape={torch_out.shape}, torch_out={torch_out}")

    torch.testing.assert_close(
        triton_out, torch_out.to(triton_out.dtype), atol=1e-1, rtol=1e-1
    )


@pytest.mark.parametrize("BATCH", [1, 4, 30, 50])
@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(1, 1), (128, 128), (32, 16), (64, 128), (2048, 2048)],
)
@pytest.mark.parametrize(
    "NUM_Q_HEADS, NUM_K_HEADS", [(1, 1), (16, 16), (2, 1), (48, 8)]
)
@pytest.mark.parametrize("HEAD_SZ", [8, 32, 128])
@pytest.mark.parametrize("CAUSAL", [(True), (False)])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
def test_mha_varlen(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    CAUSAL: bool,
    backend: str,
    dtype=torch.bfloat16,
):
    _test_mha_varlen_impl(
        BATCH,
        SEQLEN_Q,
        SEQLEN_K,
        NUM_Q_HEADS,
        NUM_K_HEADS,
        HEAD_SZ,
        CAUSAL=CAUSAL,
        backend=backend,
        dtype=dtype,
    )
