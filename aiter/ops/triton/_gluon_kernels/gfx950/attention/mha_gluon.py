##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
##############################################################################

# Naive Gluon flash-attention forward kernel for gfx950.
#
# This mirrors the style of the naive Gluon GEMM kernel in ``matmul_kernel.py``:
# everything is spelled out inline (explicit MFMA / DotOperand / Blocked
# layouts, explicit ``gl.load`` / ``gl.convert_layout`` / ``gl.amd.cdna4.mfma``)
# with no autotuning, no async pipelining and no shared-memory staging. The goal
# is readability, not peak performance.
#
# One program computes a single (batch, q_head, query-block) tile of the output
# by looping over the key/value blocks and running an online softmax.
#
# Varlen (ragged batch) inputs are supported via ``cu_seqlens_q`` / ``cu_seqlens_k``
# (cumulative sequence-length offsets, batch+1 entries each) and the ``VARLEN``
# flag, mirroring the Triton kernel in ``mha.py``. In varlen mode Q/K/V are packed
# as ``[total_tokens, num_heads, head_dim]`` (no batch dimension), the batch stride
# is 0, and each program looks up its sequence start/length from the cu_seqlens
# tensors. ``SEQLEN_Q`` / ``SEQLEN_K`` then carry the *max* sequence lengths (used
# only to size the launch grid).

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _attn_fwd_naive(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    sm_scale,
    cu_seqlens_q,
    cu_seqlens_k,
    SEQLEN_Q,
    SEQLEN_K,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,  #
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,  #
    NUM_Q_HEADS: gl.constexpr,
    NUM_K_HEADS: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    VARLEN: gl.constexpr,
    BATCH,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
):
    RCP_LN2: gl.constexpr = 1.4426950408889634

    # --- program -> (batch, q_head, query block) -------------------------------
    # SEQLEN_Q is the max query length, so NUM_BLOCKS_M matches the launch grid in
    # both fixed-length and varlen mode.
    NUM_BLOCKS_M = gl.cdiv(SEQLEN_Q, BLOCK_M)
    pid = gl.program_id(axis=0)
    off_q_head = pid % NUM_Q_HEADS
    start_m = (pid // NUM_Q_HEADS) % NUM_BLOCKS_M
    off_z = pid // (NUM_Q_HEADS * NUM_BLOCKS_M) % BATCH

    # --- per-batch sequence bounds ---------------------------------------------
    # In varlen mode the actual lengths come from cu_seqlens; the batch axis is
    # collapsed (stride_*z == 0) and the row offset into the packed tensor is the
    # cumulative start. In fixed mode we fall back to the SEQLEN_Q/SEQLEN_K args.
    if VARLEN:
        cu_seqlens_q_start = gl.load(cu_seqlens_q + off_z)
        seqlen_q = gl.load(cu_seqlens_q + off_z + 1) - cu_seqlens_q_start
        # This query block is entirely past the end of this batch's sequence.
        if start_m * BLOCK_M >= seqlen_q:
            return
        cu_seqlens_k_start = gl.load(cu_seqlens_k + off_z)
        seqlen_k = gl.load(cu_seqlens_k + off_z + 1) - cu_seqlens_k_start
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = SEQLEN_Q
        seqlen_k = SEQLEN_K

    grp_sz: gl.constexpr = NUM_Q_HEADS // NUM_K_HEADS
    off_k_head = off_q_head // grp_sz

    # --- layouts ----------------------------------------------------------------
    # A single MFMA layout is reused for both matmuls (QK^T and P@V). For bf16/fp16
    # both dots share the same 16x16x32 instruction shape, so the result
    # distribution is identical and softmax state can live in its slice layouts.
    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2]
    )
    dotQ: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotK: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)
    dotP: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotV: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

    # Coalesced global-load layouts (converted to the dot-operand layouts after
    # loading), in the same spirit as the GEMM kernel's gLoadLayoutA/B.
    qLoadLayout: gl.constexpr = gl.BlockedLayout(
        [1, 8],
        [512 // BLOCK_DMODEL, BLOCK_DMODEL // 8],
        [4, 1],
        [1, 0],
    )
    # K is read transposed as [BLOCK_DMODEL, BLOCK_N] (head dim contiguous).
    kLoadLayout: gl.constexpr = gl.BlockedLayout(
        [8, 1],
        [BLOCK_DMODEL // 8, 512 // BLOCK_DMODEL],
        [1, 4],
        [0, 1],
    )
    vLoadLayout: gl.constexpr = gl.BlockedLayout(
        [1, 8],
        [512 // BLOCK_DMODEL, BLOCK_DMODEL // 8],
        [4, 1],
        [1, 0],
    )

    # --- load Q (stays resident for the whole key loop) -------------------------
    offs_qm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, qLoadLayout))
    offs_qd = gl.arange(0, BLOCK_DMODEL, layout=gl.SliceLayout(0, qLoadLayout))
    q_base = (
        q_ptr + off_z * stride_qz + off_q_head * stride_qh + cu_seqlens_q_start * stride_qm
    )
    q_ptrs = (
        q_base
        + (start_m * BLOCK_M + offs_qm)[:, None] * stride_qm
        + offs_qd[None, :] * stride_qk
    )
    q_mask = (start_m * BLOCK_M + offs_qm)[:, None] < seqlen_q
    q = gl.load(q_ptrs, mask=q_mask, other=0.0)
    q = gl.convert_layout(q, layout=dotQ)

    # --- key/value pointer setup ------------------------------------------------
    offs_kd = gl.arange(0, BLOCK_DMODEL, layout=gl.SliceLayout(1, kLoadLayout))
    offs_kn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, kLoadLayout))
    k_base = (
        k_ptr + off_z * stride_kz + off_k_head * stride_kh + cu_seqlens_k_start * stride_kn
    )
    k_ptrs = k_base + offs_kd[:, None] * stride_kk + offs_kn[None, :] * stride_kn

    offs_vn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, vLoadLayout))
    offs_vd = gl.arange(0, BLOCK_DMODEL, layout=gl.SliceLayout(0, vLoadLayout))
    v_base = (
        v_ptr + off_z * stride_vz + off_k_head * stride_vh + cu_seqlens_k_start * stride_vn
    )
    v_ptrs = v_base + offs_vn[:, None] * stride_vn + offs_vd[None, :] * stride_vk

    # --- online-softmax state ---------------------------------------------------
    # l_i is seeded to 1.0: on the first iteration alpha == exp2(-inf - m) == 0,
    # which zeroes the seed, so no special-casing of the first block is needed.
    m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=gl.SliceLayout(1, mfmaLayout))
    l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=gl.SliceLayout(1, mfmaLayout))
    acc = gl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=gl.float32, layout=mfmaLayout)

    qk_scale = sm_scale * RCP_LN2

    # Query / key positions used for masking, in the MFMA result layout.
    offs_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mfmaLayout))
    offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfmaLayout))

    # Number of key blocks; for causal we stop at the (bottom-right aligned) diagonal.
    n_blocks = gl.cdiv(seqlen_k, BLOCK_N)
    if IS_CAUSAL:
        n_blocks_causal = gl.cdiv(
            (start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N
        )
        n_blocks = min(n_blocks, n_blocks_causal)

    for start_n in range(0, n_blocks * BLOCK_N, BLOCK_N):
        # qk-space mask (MFMA layout): boundary + (optionally) causal.
        key_pos = start_n + offs_n
        mask = key_pos[None, :] < seqlen_k
        if IS_CAUSAL:
            mask = mask & (key_pos[None, :] <= (offs_m[:, None] + (seqlen_k - seqlen_q)))

        # Load-space boundary masks must live in the load layouts (not the MFMA
        # layout) so they line up with the pointer tensors.
        gk = gl.load(k_ptrs, mask=(start_n + offs_kn)[None, :] < seqlen_k, other=0.0)
        gv = gl.load(v_ptrs, mask=(start_n + offs_vn)[:, None] < seqlen_k, other=0.0)
        k = gl.convert_layout(gk, layout=dotK)

        # -- QK^T --
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mfmaLayout)
        qk = gl.amd.cdna4.mfma(q, k, qk)
        qk = qk * qk_scale
        qk = gl.where(mask, qk, float("-inf"))

        # -- online softmax --
        # Fully-masked rows leave m_ij == -inf and produce NaNs here; those rows
        # are zeroed out by index in the epilogue (see mha.py for the same scheme).
        m_ij = gl.maximum(m_i, gl.max(qk, 1))
        p = gl.exp2(qk - m_ij[:, None])
        alpha = gl.exp2(m_i - m_ij)
        l_ij = gl.sum(p, 1)

        acc = acc * alpha[:, None]
        v = gl.convert_layout(gv, layout=dotV)
        p = gl.convert_layout(p.to(v_ptr.dtype.element_ty), layout=dotP)
        acc = gl.amd.cdna4.mfma(p, v, acc)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    # --- epilogue: normalize and write back -------------------------------------
    acc = acc / l_i[:, None]
    out = acc.to(o_ptr.dtype.element_ty)

    offs_om = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mfmaLayout))
    offs_od = gl.arange(0, BLOCK_DMODEL, layout=gl.SliceLayout(0, mfmaLayout))
    o_base = (
        o_ptr + off_z * stride_oz + off_q_head * stride_oh + cu_seqlens_q_start * stride_om
    )
    o_ptrs = o_base + offs_om[:, None] * stride_om + offs_od[None, :] * stride_on

    # Rows above the bottom-right causal boundary (only possible when
    # seqlen_q > seqlen_k) attend to no keys, so their softmax row is all -inf and
    # the normalization above yields NaN. Mirror mha.py / the reference by zeroing
    # those rows out by index rather than guarding the division.
    if IS_CAUSAL:
        out = gl.where(offs_om[:, None] >= (seqlen_q - seqlen_k), out, 0.0)

    o_mask = offs_om[:, None] < seqlen_q
    gl.store(o_ptrs, out, mask=o_mask)


def _validate_and_launch(
    q,
    k,
    v,
    o,
    sm_scale,
    causal,
    seqlen_q,
    seqlen_k,
    num_q_heads,
    num_k_heads,
    head_dim,
    batch,
    q_strides,
    k_strides,
    v_strides,
    o_strides,
    cu_seqlens_q,
    cu_seqlens_k,
    varlen,
    BLOCK_M,
    BLOCK_N,
):
    """Shared validation + launch for the fixed-length and varlen wrappers.

    ``*_strides`` are 4-tuples in (batch, head, seq, head_dim) order. ``seqlen_q``
    is the max query length (used to size the grid); in varlen mode the per-batch
    lengths are read from ``cu_seqlens_*`` inside the kernel.
    """
    assert q.shape[-1] == k.shape[-1] == v.shape[-1], "head_dim mismatch"
    assert num_q_heads % num_k_heads == 0, "num_q_heads must be divisible by num_k_heads"
    # MFMA 16x16x32 needs the contraction dims (head_dim, BLOCK_N) to be
    # multiples of 32, and the blocked load layouts need head_dim | 512.
    assert head_dim in (32, 64, 128, 256), "naive kernel supports head_dim in {32,64,128,256}"
    assert BLOCK_N % 32 == 0, "BLOCK_N must be a multiple of 32"

    grid = (batch * num_q_heads * triton.cdiv(seqlen_q, BLOCK_M), 1)

    _attn_fwd_naive[grid](
        q,
        k,
        v,
        o,
        sm_scale,
        cu_seqlens_q,
        cu_seqlens_k,
        seqlen_q,
        seqlen_k,
        *q_strides,  #
        *k_strides,  #
        *v_strides,  #
        *o_strides,  #
        NUM_Q_HEADS=num_q_heads,
        NUM_K_HEADS=num_k_heads,
        IS_CAUSAL=causal,
        VARLEN=varlen,
        BATCH=batch,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=head_dim,
        num_warps=4,
    )
    return o


def flash_attn_fwd(
    q,
    k,
    v,
    causal=False,
    sm_scale=None,
    o=None,
    BLOCK_M=128,
    BLOCK_N=64,
):
    """Naive Gluon flash-attention forward (fixed-length / padded batch).

    Arguments:
        q: (batch, seqlen_q, num_q_heads, head_dim)
        k: (batch, seqlen_k, num_k_heads, head_dim)
        v: (batch, seqlen_k, num_k_heads, head_dim)
        causal: whether to apply a (bottom-right aligned) causal mask.
        sm_scale: QK^T scale. Defaults to 1 / sqrt(head_dim).
    Return:
        o: (batch, seqlen_q, num_q_heads, head_dim)
    """
    batch, seqlen_q, num_q_heads, head_dim = q.shape
    _, seqlen_k, num_k_heads, _ = k.shape

    if sm_scale is None:
        sm_scale = head_dim ** (-0.5)
    if o is None:
        o = torch.empty_like(q)

    # (batch, head, seq, head_dim) stride order expected by the kernel.
    q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
    k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
    v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
    o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))

    return _validate_and_launch(
        q,
        k,
        v,
        o,
        sm_scale,
        causal,
        seqlen_q,
        seqlen_k,
        num_q_heads,
        num_k_heads,
        head_dim,
        batch,
        q_strides,
        k_strides,
        v_strides,
        o_strides,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        varlen=False,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )


def flash_attn_varlen_fwd(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    causal=False,
    sm_scale=None,
    o=None,
    BLOCK_M=128,
    BLOCK_N=64,
):
    """Naive Gluon flash-attention forward for ragged (varlen) batches.

    Q/K/V are packed token sequences without a batch dimension; the per-sequence
    boundaries are given by the cumulative-length tensors.

    Arguments:
        q: (total_q, num_q_heads, head_dim), all query tokens across the batch.
        k: (total_k, num_k_heads, head_dim), all key tokens across the batch.
        v: (total_k, num_k_heads, head_dim), all value tokens across the batch.
        cu_seqlens_q: (batch + 1,) int32 cumulative query sequence lengths.
        cu_seqlens_k: (batch + 1,) int32 cumulative key sequence lengths.
        max_seqlen_q: maximum query sequence length in the batch.
        max_seqlen_k: maximum key sequence length in the batch.
        causal: whether to apply a (bottom-right aligned) causal mask.
        sm_scale: QK^T scale. Defaults to 1 / sqrt(head_dim).
    Return:
        o: (total_q, num_q_heads, head_dim)
    """
    _, num_q_heads, head_dim = q.shape
    _, num_k_heads, _ = k.shape
    batch = cu_seqlens_q.numel() - 1
    max_seqlen_q = int(max_seqlen_q)
    max_seqlen_k = int(max_seqlen_k)

    if sm_scale is None:
        sm_scale = head_dim ** (-0.5)
    if o is None:
        o = torch.empty_like(q)

    # Packed [total_tokens, head, head_dim] layout: the batch axis is collapsed,
    # so its stride is 0 and the per-sequence row offset comes from cu_seqlens.
    q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
    k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
    v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
    o_strides = (0, o.stride(1), o.stride(0), o.stride(2))

    return _validate_and_launch(
        q,
        k,
        v,
        o,
        sm_scale,
        causal,
        max_seqlen_q,
        max_seqlen_k,
        num_q_heads,
        num_k_heads,
        head_dim,
        batch,
        q_strides,
        k_strides,
        v_strides,
        o_strides,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        varlen=True,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
