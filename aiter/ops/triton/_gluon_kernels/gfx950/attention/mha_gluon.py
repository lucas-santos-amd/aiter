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

# Gluon flash-attention forward kernel for gfx950.
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

from aiter.ops.triton.utils.device_info import get_num_xcds


@gluon.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: gl.constexpr = 8):
    """Gluon copy of ``aiter...pid_preprocessing.remap_xcd``.

    Re-groups the linear program id so consecutive ids land on the same XCD,
    improving L2/cache locality. ``GRID_MN`` is intentionally a runtime value
    (not constexpr) so ``tall_xcds`` is a tensor scalar with a ``.type``.
    """
    # Number of pids per XCD in the new arrangement.
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN is not divisible by NUM_XCDS, some xcds get pids_per_xcd pids
    # and the rest get pids_per_xcd - 1; tall_xcds counts the former.
    tall_xcds = GRID_MN % NUM_XCDS
    if tall_xcds == 0:
        tall_xcds = gl.cast(NUM_XCDS, tall_xcds.type)
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )
    return pid


@gluon.constexpr_function
def _make_kv_shared_layouts(head_dim_pow2, elem_bytes, k_width=8, non_k_dim=16, banks=64):
    """Swizzled LDS layouts for the async K/V staging tiles (gfx950).

    Returns (kSharedLayout, vSharedLayout):
    K is stored [head, N] (order [0, 1]), V is [N, head] (order
    [1, 0]); both are contiguous along the head dim.
    """
    bank_line_bytes = banks * 4
    bank_line_elems = bank_line_bytes // elem_bytes
    read_vec_bytes = min(k_width * elem_bytes, 16)
    num_threads_same_cycle = bank_line_bytes // read_vec_bytes
    per_phase = (bank_line_elems + head_dim_pow2 - 1) // head_dim_pow2
    swizzle_vec = k_width * max(1, per_phase // 2)
    max_phase = min(
        min(non_k_dim, num_threads_same_cycle) // per_phase,
        bank_line_elems // swizzle_vec,
    )
    k_shared = gl.SwizzledSharedLayout(swizzle_vec, per_phase, max_phase, order=[0, 1])
    v_shared = gl.SwizzledSharedLayout(swizzle_vec, per_phase, max_phase, order=[1, 0])
    return k_shared, v_shared


@gluon.jit
def _async_load_fn(smem, base, offsets):
    """Unmasked async global->LDS copy (AMD ``buffer_load_to_shared``).

    Issues an async copy from scalar base ``base`` + constant int32 ``offsets``
    into the shared-memory tile ``smem``; the caller owns the matching
    ``commit_group`` / ``wait_group`` before reading ``smem``.

    Only used for fully in-range tiles. The scalar-base ``buffer_load_to_shared``
    masked path was unreliable for partial tiles here, so tiles that need per-lane
    masking (boundary key columns or head-dim padding) go through the pointer-
    tensor ``global_load_to_shared`` path in ``_async_load_masked_fn`` instead.
    """
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smem, base, offsets)


@gluon.jit
def _async_load_masked_fn(
    smem, base, offsets, offset_first, offset_second, boundary_first, boundary_second
):
    """Masked async global->LDS copy (AMD ``global_load_to_shared``).

    Builds the 64-bit pointer tensor ``base + offsets`` and async-copies it into
    the shared tile ``smem`` with a per-axis mask, filling masked lanes with
    ``other=0.0`` so out-of-range key columns / head-dim padding read 0 (required
    so they contribute 0 to the QK / P@V matmuls). Used for boundary and
    padded-head tiles so K/V always stage through LDS. Unlike the scalar-base
    ``buffer_load_to_shared`` fast path, ``global_load_to_shared`` takes a full
    pointer tensor, which is what supports masked async-to-LDS here. Each axis is
    masked only when its offset tensor is not ``None`` (a compile-time decision).
    The caller owns the matching ``commit_group`` / ``wait_group``.
    """
    ptrs = base + offsets
    if offset_first is not None and offset_second is not None:
        mask = (offset_first[:, None] < boundary_first) & (
            offset_second[None, :] < boundary_second
        )
        gl.amd.cdna4.async_copy.global_load_to_shared(smem, ptrs, mask=mask, other=0.0)
    elif offset_first is not None:
        mask = offset_first[:, None] < boundary_first
        gl.amd.cdna4.async_copy.global_load_to_shared(smem, ptrs, mask=mask, other=0.0)
    elif offset_second is not None:
        mask = offset_second[None, :] < boundary_second
        gl.amd.cdna4.async_copy.global_load_to_shared(smem, ptrs, mask=mask, other=0.0)
    else:
        gl.amd.cdna4.async_copy.global_load_to_shared(smem, ptrs)


@gluon.jit
def _issue_kv_copy(
    smemK_buf,
    smemV_buf,
    k_base,
    v_base,
    k_offsets,
    v_offsets,
    copy_start_n,
    seqlen_k,
    K_LOAD_LAYOUT: gl.constexpr,
    V_LOAD_LAYOUT: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DMODEL_POW2: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
    MASK_STEPS: gl.constexpr,
    PADDED_HEAD: gl.constexpr,
):
    """Issue the async global->LDS copies of one K and one V key block.

    Mirrors the load logic that used to live inline in ``_attn_fwd_inner`` but
    targets explicit shared-memory buffer slots (``smemK_buf`` / ``smemV_buf``,
    already selected via ``smem.index(...)``) so the caller can double-buffer and
    prefetch the next block while the current one is being consumed. K is staged
    as ``[BLOCK_DMODEL_POW2, BLOCK_N]`` (d on axis 0, n on axis 1); V as
    ``[BLOCK_N, BLOCK_DMODEL_POW2]``. Fully in-range tiles take the fast
    scalar-base ``buffer_load_to_shared`` path; boundary / padded-head tiles use
    the pointer-tensor ``global_load_to_shared`` path with ``other=0`` so masked
    lanes stage as 0. ``copy_start_n`` is the key-block start of the tile being
    copied (used only for the boundary n-axis mask). The caller owns the matching
    ``commit_group`` / ``wait_group``.
    """
    if USE_BUFFER_LOAD:
        _async_load_fn(smemK_buf, k_base, k_offsets)
        _async_load_fn(smemV_buf, v_base, v_offsets)
    else:
        # Load-space offsets line up with the pointer tensors. The n-axis is
        # masked only on boundary (masked) blocks; the head-dim axis only when
        # the head is padded (None => that axis is unmasked).
        offs_kn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, K_LOAD_LAYOUT))
        offs_vn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, V_LOAD_LAYOUT))
        offs_kd = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(1, K_LOAD_LAYOUT))
        offs_vd = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(0, V_LOAD_LAYOUT))
        if MASK_STEPS:
            k_offs_n = copy_start_n + offs_kn
            v_offs_n = copy_start_n + offs_vn
        else:
            k_offs_n = None
            v_offs_n = None
        if PADDED_HEAD:
            k_offs_d = offs_kd
            v_offs_d = offs_vd
        else:
            k_offs_d = None
            v_offs_d = None
        _async_load_masked_fn(
            smemK_buf, k_base, k_offsets, k_offs_d, k_offs_n, BLOCK_DMODEL, seqlen_k
        )
        _async_load_masked_fn(
            smemV_buf, v_base, v_offsets, v_offs_n, v_offs_d, seqlen_k, BLOCK_DMODEL
        )


@gluon.jit
def _attn_qk(
    q,
    k,
    start_n,
    seqlen_q,
    seqlen_k,
    block_max,
    n_extra_tokens,
    offs_m,
    offs_n,
    qk_scale,
    MFMA_LAYOUT: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    MASK_STEPS: gl.constexpr,
):
    """QK^T (+ scale + boundary/causal mask) for one already-staged key block.

    ``k`` has already been read out of LDS into its MFMA dot-operand layout.
    Returns the masked ``qk`` scores in ``MFMA_LAYOUT`` (float32). This is the
    first half of one online-softmax step, split out so the caller can pipeline
    the QK^T of block ``i+1`` ahead of the softmax / P@V of block ``i`` -- the
    independent QK MFMA then fills the issue gaps left by the softmax VALU chain,
    mimicking what mha.py's auto-pipeliner does across loop iterations. The masks
    are compiled out entirely when ``MASK_STEPS`` / ``IS_CAUSAL`` are ``False``.
    """
    qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=MFMA_LAYOUT)
    qk = gl.amd.cdna4.mfma(q, k, qk)
    qk = qk * qk_scale

    # qk-space mask (MFMA layout): boundary + (optionally) causal. Skipped
    # entirely for full blocks.
    if MASK_STEPS or IS_CAUSAL:
        mask = gl.full([BLOCK_M, BLOCK_N], True, dtype=gl.int1, layout=MFMA_LAYOUT)
        if MASK_STEPS:
            # Only the last visible block can be partial, and only when seqlen_k
            # is not a multiple of BLOCK_N. mask_partial is computed
            # unconditionally and selected via bound_cond so the body stays
            # branch-free (mirrors mha.py).
            bound_cond = (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0)
            size_n = start_n + offs_n
            mask_partial = size_n[None, :] < seqlen_k
            mask = gl.where(bound_cond, mask_partial, mask)
        if IS_CAUSAL:
            causal_boundary = start_n + offs_n + (seqlen_q - seqlen_k)
            causal_mask = offs_m[:, None] >= causal_boundary[None, :]
            mask = mask & causal_mask
        qk = gl.where(mask, qk, float("-inf"))

    return qk


@gluon.jit
def _attn_softmax_pv(acc, l_i, m_i, qk, v, dotP: gl.constexpr):
    """Online-softmax rescale + P@V accumulation for one key block.

    ``qk`` holds the (already masked) QK^T scores for this block and ``v`` its
    value tile in the MFMA dot-operand layout. Returns the updated
    ``(acc, l_i, m_i)``. Second half of one online-softmax step (see ``_attn_qk``).
    """
    # Fully-masked rows leave m_ij == -inf and produce NaNs here; those rows are
    # zeroed out by index in the epilogue (see mha.py for the same scheme).
    m_ij = gl.maximum(m_i, gl.max(qk, 1))
    p = gl.exp2(qk - m_ij[:, None])
    alpha = gl.exp2(m_i - m_ij)
    l_ij = gl.sum(p, 1)

    acc = acc * alpha[:, None]

    p = gl.convert_layout(p.to(v.dtype), layout=dotP)
    acc = gl.amd.cdna4.mfma(p, v, acc)

    l_i = l_i * alpha + l_ij
    m_i = m_ij

    return acc, l_i, m_i


@gluon.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    k_base,
    k_offsets,
    v_base,
    v_offsets,
    smemK,
    smemV,
    stride_kn,
    stride_vn,
    seqlen_q,
    seqlen_k,
    block_min,
    block_max,
    n_extra_tokens,
    offs_m,
    qk_scale,
    MFMA_LAYOUT: gl.constexpr,
    K_LOAD_LAYOUT: gl.constexpr,
    V_LOAD_LAYOUT: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DMODEL_POW2: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    MASK_STEPS: gl.constexpr,
    PRELOAD_V: gl.constexpr,
    NUM_KV_BUFFERS: gl.constexpr,
):
    """QK-ahead software-pipelined online-softmax loop over ``[block_min, block_max)``.

    ``q`` arrives already in the ``dotQ`` layout. ``k_base`` / ``v_base`` are scalar
    base pointers positioned at ``block_min`` (advanced by one key block per copy
    issued) and ``k_offsets`` / ``v_offsets`` are the constant int32 offset tiles
    consumed by the buffer loads. ``smemK`` / ``smemV`` are ``NUM_KV_BUFFERS``-way
    shared-memory tiles (block ``j`` lives in buffer ``j % NUM_KV_BUFFERS``).

    Three overlapping pipelines run concurrently, mirroring what mha.py's
    auto-pipeliner buys for free:

    * **global prefetch** -- the async copy of block ``i+2`` (issued at the top of
      iteration ``i``) is in flight while block ``i`` is being consumed;
    * **local prefetch** -- block ``i+1`` is read LDS->register just before it is
      used;
    * **compute pipeline** -- the QK^T MFMA of block ``i+1`` is issued *before* the
      softmax + P@V of block ``i`` (``qk``/``v`` for the current block are carried
      across iterations). The independent QK MFMA overlaps the softmax VALU chain,
      hiding the softmax latency the two dependent matmuls would otherwise expose.

    Keeping blocks ``i`` (being consumed), ``i+1`` (read into registers) and
    ``i+2`` (async copy in flight) simultaneously resident requires
    ``NUM_KV_BUFFERS >= 3``. The boundary / causal masks are compiled out entirely
    when MASK_STEPS / IS_CAUSAL are False (the full-block case). n_iter >= 1 (the
    caller only enters with at least one block); the pipeline degrades gracefully
    at n_iter == 1/2. Each pass drains its own copies before returning, so no
    async copy leaks across the full/masked handoff.
    """
    PADDED_HEAD: gl.constexpr = BLOCK_DMODEL != BLOCK_DMODEL_POW2
    # Fully in-range tiles take the fast scalar-base ``buffer_load_to_shared``
    # path; tiles that need per-lane masking (boundary key columns or head-dim
    # padding) use the pointer-tensor ``global_load_to_shared`` path with other=0.
    USE_BUFFER_LOAD: gl.constexpr = (not MASK_STEPS) and (not PADDED_HEAD)

    # Dot-operand layouts are derived from the shared MFMA layout.
    dotK: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MFMA_LAYOUT, k_width=8)
    dotP: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MFMA_LAYOUT, k_width=8)
    dotV: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MFMA_LAYOUT, k_width=8)

    # qk-space key offsets (MFMA layout) for the boundary / causal masks.
    offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, MFMA_LAYOUT))

    n_iter = (block_max - block_min) // BLOCK_N

    # prologue: stage block 0 (and block 1 if present) into LDS, read block 0 into
    # registers and compute its QK^T so the loop starts one QK ahead.
    _issue_kv_copy(
        smemK.index(0),
        smemV.index(0),
        k_base,
        v_base,
        k_offsets,
        v_offsets,
        block_min,
        seqlen_k,
        K_LOAD_LAYOUT,
        V_LOAD_LAYOUT,
        BLOCK_N,
        BLOCK_DMODEL,
        BLOCK_DMODEL_POW2,
        USE_BUFFER_LOAD,
        MASK_STEPS,
        PADDED_HEAD,
    )
    gl.amd.cdna4.async_copy.commit_group()
    k_base += BLOCK_N * stride_kn
    v_base += BLOCK_N * stride_vn

    if n_iter > 1:
        _issue_kv_copy(
            smemK.index(1),
            smemV.index(1),
            k_base,
            v_base,
            k_offsets,
            v_offsets,
            block_min + BLOCK_N,
            seqlen_k,
            K_LOAD_LAYOUT,
            V_LOAD_LAYOUT,
            BLOCK_N,
            BLOCK_DMODEL,
            BLOCK_DMODEL_POW2,
            USE_BUFFER_LOAD,
            MASK_STEPS,
            PADDED_HEAD,
        )
        gl.amd.cdna4.async_copy.commit_group()
        k_base += BLOCK_N * stride_kn
        v_base += BLOCK_N * stride_vn
        # block 0 ready, block 1 copy stays in flight (overlaps the QK^T below).
        gl.amd.cdna4.async_copy.wait_group(1)
    else:
        gl.amd.cdna4.async_copy.wait_group(0)

    k = smemK.index(0).load(dotK)
    v_cur = smemV.index(0).load(dotV)
    qk_cur = _attn_qk(
        q,
        k,
        block_min,
        seqlen_q,
        seqlen_k,
        block_max,
        n_extra_tokens,
        offs_m,
        offs_n,
        qk_scale,
        MFMA_LAYOUT=MFMA_LAYOUT,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=IS_CAUSAL,
        MASK_STEPS=MASK_STEPS,
    )

    # steady state: iteration i consumes block i (softmax + P@V from the carried
    # qk_cur / v_cur) while (a) the async copy of block i+2 is in flight, (b) block
    # i+1 is read LDS->register, and (c) its QK^T MFMA is issued ahead of the
    # softmax so the two overlap.
    for i in range(0, n_iter):
        if i + 1 < n_iter:
            # (a) Prefetch block i+2 into its buffer, which last held block i-1
            # (already fully consumed), so the async write is race-free. Issued
            # first so the DMA overlaps the wait / compute below.
            if i + 2 < n_iter:
                g_idx = (i + 2) % NUM_KV_BUFFERS
                _issue_kv_copy(
                    smemK.index(g_idx),
                    smemV.index(g_idx),
                    k_base,
                    v_base,
                    k_offsets,
                    v_offsets,
                    block_min + (i + 2) * BLOCK_N,
                    seqlen_k,
                    K_LOAD_LAYOUT,
                    V_LOAD_LAYOUT,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    BLOCK_DMODEL_POW2,
                    USE_BUFFER_LOAD,
                    MASK_STEPS,
                    PADDED_HEAD,
                )
                gl.amd.cdna4.async_copy.commit_group()
                k_base += BLOCK_N * stride_kn
                v_base += BLOCK_N * stride_vn
                # block i+1 copy + block i+2 copy in flight -> retire the oldest
                # (block i+1) so its ds_read below is safe.
                gl.amd.cdna4.async_copy.wait_group(1)
            else:
                # Tail: no more copies to issue, only block i+1's copy is left.
                gl.amd.cdna4.async_copy.wait_group(0)

            # (b) local prefetch of block i+1 into registers.
            l_idx = (i + 1) % NUM_KV_BUFFERS
            k = smemK.index(l_idx).load(dotK)
            v_next = smemV.index(l_idx).load(dotV)

            # (c) QK^T of block i+1, issued ahead of the softmax + P@V of block i.
            qk_next = _attn_qk(
                q,
                k,
                block_min + (i + 1) * BLOCK_N,
                seqlen_q,
                seqlen_k,
                block_max,
                n_extra_tokens,
                offs_m,
                offs_n,
                qk_scale,
                MFMA_LAYOUT=MFMA_LAYOUT,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                IS_CAUSAL=IS_CAUSAL,
                MASK_STEPS=MASK_STEPS,
            )

            # softmax + P@V for block i; its softmax VALU overlaps the block i+1
            # QK MFMA issued just above. Carry block i+1's scores/values as the new
            # "current" state. The carry lives in this branch (not a second
            # ``if i + 1 < n_iter``) so qk_next / v_next stay in scope for the JIT.
            acc, l_i, m_i = _attn_softmax_pv(acc, l_i, m_i, qk_cur, v_cur, dotP)
            qk_cur = qk_next
            v_cur = v_next
        else:
            # Final block: nothing left to prefetch, just drain the softmax + P@V.
            acc, l_i, m_i = _attn_softmax_pv(acc, l_i, m_i, qk_cur, v_cur, dotP)

    return acc, l_i, m_i


@gluon.jit
def _attn_fwd(
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
    BLOCK_DMODEL_POW2: gl.constexpr,
    PRELOAD_V: gl.constexpr,
    NUM_XCD: gl.constexpr,
    HEAD_STRIDE_ALIGNED_8: gl.constexpr = False,
):
    RCP_LN2: gl.constexpr = 1.4426950408889634
    PADDED_HEAD: gl.constexpr = BLOCK_DMODEL != BLOCK_DMODEL_POW2

    # --- program -> (batch, q_head, query block) -------------------------------
    # SEQLEN_Q is the max query length, so NUM_BLOCKS_M matches the launch grid in
    # both fixed-length and varlen mode.
    NUM_BLOCKS_M = gl.cdiv(SEQLEN_Q, BLOCK_M)
    pid = gl.program_id(axis=0)
    off_q_head = pid % NUM_Q_HEADS
    # Re-map the q-head index across XCDs for better cache locality.
    off_q_head = remap_xcd(off_q_head, NUM_Q_HEADS, NUM_XCD)
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
    # The dot/load layouts for K/V are recomputed inside _attn_fwd.
    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2]
    )
    dotQ: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)

    # Coalesced global-load layouts (converted to the dot-operand layouts after
    # loading), in the same spirit as the GEMM kernel's gLoadLayoutA/B. These are
    # sized to the padded head dim (BLOCK_DMODEL_POW2); the real head dim is only
    # used for the head-padding load/store masks.
    qLoadLayout: gl.constexpr = gl.BlockedLayout(
        [1, 8],
        [512 // BLOCK_DMODEL_POW2, BLOCK_DMODEL_POW2 // 8],
        [4, 1],
        [1, 0],
    )
    # K is read transposed as [BLOCK_DMODEL_POW2, BLOCK_N] (head dim contiguous).
    kLoadLayout: gl.constexpr = gl.BlockedLayout(
        [8, 1],
        [BLOCK_DMODEL_POW2 // 8, 512 // BLOCK_DMODEL_POW2],
        [1, 4],
        [0, 1],
    )
    vLoadLayout: gl.constexpr = gl.BlockedLayout(
        [1, 8],
        [512 // BLOCK_DMODEL_POW2, BLOCK_DMODEL_POW2 // 8],
        [4, 1],
        [1, 0],
    )

    # Swizzled shared layouts for the async global->LDS staging of K and V
    _KV_SHARED: gl.constexpr = _make_kv_shared_layouts(
        BLOCK_DMODEL_POW2, k_ptr.dtype.element_ty.primitive_bitwidth // 8
    )
    kSharedLayout: gl.constexpr = _KV_SHARED[0]
    vSharedLayout: gl.constexpr = _KV_SHARED[1]

    # --- head-axis byte offsets -------------------------------------------------
    # When the caller guarantees Q/K/V head strides are multiples of 8 elements
    # (HEAD_STRIDE_ALIGNED_8), the head-axis offset is 16-byte aligned. Hinting
    # the multiple lets AxisInfo widen the global loads.
    qh_off = off_q_head * stride_qh
    kh_off = off_k_head * stride_kh
    vh_off = off_k_head * stride_vh
    if HEAD_STRIDE_ALIGNED_8:
        qh_off = gl.multiple_of(qh_off, 8)
        kh_off = gl.multiple_of(kh_off, 8)
        vh_off = gl.multiple_of(vh_off, 8)

    # --- load Q (stays resident for the whole key loop) -------------------------
    # The constant per-program part of the address (batch / head / sequence-start
    # and this program's query-block offset) is folded into the scalar base; the
    # int32 offset tile carries only the in-tile (row, head-dim) offsets.
    offs_qm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, qLoadLayout))
    offs_qd = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(0, qLoadLayout))
    q_base = (
        q_ptr
        + off_z * stride_qz
        + qh_off
        + cu_seqlens_q_start * stride_qm
        + start_m * BLOCK_M * stride_qm
    )
    q_offsets = (offs_qm[:, None] * stride_qm + offs_qd[None, :] * stride_qk).to(gl.int32)
    q_mask = (start_m * BLOCK_M + offs_qm)[:, None] < seqlen_q
    if PADDED_HEAD:
        q_mask = q_mask & (offs_qd[None, :] < BLOCK_DMODEL)
    # Cache Q at .cg when a single Q block spans at least one full head, so the
    # repeated head-major reads stay resident.
    if BLOCK_M >= NUM_Q_HEADS:
        q_cache_mod: gl.constexpr = ".cg"
    else:
        q_cache_mod: gl.constexpr = ""
    q = gl.amd.cdna4.buffer_load(
        ptr=q_base, offsets=q_offsets, mask=q_mask, other=0.0, cache=q_cache_mod
    )
    q = gl.convert_layout(q, layout=dotQ)

    # --- key/value pointer setup ------------------------------------------------
    # Same base/offset split as Q: the scalar base lands at key block 0 of this
    # (batch, head) and is advanced one key block per loop iteration inside
    # _attn_fwd, while the int32 offset tiles stay constant.
    offs_kd = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(1, kLoadLayout))
    offs_kn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, kLoadLayout))
    k_base = k_ptr + off_z * stride_kz + kh_off + cu_seqlens_k_start * stride_kn
    k_offsets = (offs_kd[:, None] * stride_kk + offs_kn[None, :] * stride_kn).to(gl.int32)

    offs_vn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, vLoadLayout))
    offs_vd = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(0, vLoadLayout))
    v_base = v_ptr + off_z * stride_vz + vh_off + cu_seqlens_k_start * stride_vn
    v_offsets = (offs_vn[:, None] * stride_vn + offs_vd[None, :] * stride_vk).to(gl.int32)

    # --- shared-memory tiles for the async K/V staging --------------------------
    # Triple-buffered (leading dim = NUM_KV_BUFFERS) so _attn_fwd_inner can keep
    # block i (being consumed), block i+1 (read LDS->register) and block i+2 (async
    # copy in flight) simultaneously resident. This is what lets the QK^T of block
    # i+1 be issued ahead of the softmax + P@V of block i (compute pipeline) on top
    # of the global/local prefetch. Allocated once and reused across both inner-loop
    # passes (each pass drains its copies before returning). CDNA4 has 160KB of LDS,
    # so three K/V buffers fit comfortably.
    NUM_KV_BUFFERS: gl.constexpr = 3
    smemK = gl.allocate_shared_memory(
        k_ptr.dtype.element_ty, [NUM_KV_BUFFERS, BLOCK_DMODEL_POW2, BLOCK_N], kSharedLayout
    )
    smemV = gl.allocate_shared_memory(
        v_ptr.dtype.element_ty, [NUM_KV_BUFFERS, BLOCK_N, BLOCK_DMODEL_POW2], vSharedLayout
    )

    # --- online-softmax state ---------------------------------------------------
    # l_i is seeded to 1.0: on the first iteration alpha == exp2(-inf - m) == 0,
    # which zeroes the seed, so no special-casing of the first block is needed.
    m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=gl.SliceLayout(1, mfmaLayout))
    l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=gl.SliceLayout(1, mfmaLayout))
    acc = gl.zeros([BLOCK_M, BLOCK_DMODEL_POW2], dtype=gl.float32, layout=mfmaLayout)

    qk_scale = sm_scale * RCP_LN2

    # Query positions used for the causal mask, in the MFMA result layout.
    offs_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mfmaLayout))

    # --- classify key blocks: full (mask-free) vs masked (boundary/causal) ------
    # Mirrors mha.py: full blocks are fully inside seqlen_k and (for causal) fully
    # below the diagonal, so they run without any masking. Only the trailing
    # boundary block and the causal-diagonal blocks need masking.
    n_blocks = gl.cdiv(seqlen_k, BLOCK_N)
    if IS_CAUSAL:
        n_blocks_causal = gl.cdiv(
            (start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N
        )
        n_blocks = min(n_blocks, n_blocks_causal)

        # If there are no visible blocks left, this query block lies entirely
        # above the (bottom-right aligned) causal diagonal and attends to no
        # keys. Its output rows are all zero, so write them out and bail before
        # touching the matrix units (mirrors the early-out in mha.py).
        if n_blocks <= 0:
            offs_od = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(0, mfmaLayout))
            offs_rm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mfmaLayout))
            o_base = (
                o_ptr
                + off_z * stride_oz
                + off_q_head * stride_oh
                + cu_seqlens_q_start * stride_om
                + start_m * BLOCK_M * stride_om
            )
            o_offsets = (
                offs_rm[:, None] * stride_om + offs_od[None, :] * stride_on
            ).to(gl.int32)
            zeros = gl.zeros(
                [BLOCK_M, BLOCK_DMODEL_POW2],
                dtype=o_ptr.dtype.element_ty,
                layout=mfmaLayout,
            )
            o_mask = offs_m[:, None] < seqlen_q
            if PADDED_HEAD:
                o_mask = o_mask & (offs_od[None, :] < BLOCK_DMODEL)
            gl.amd.cdna4.buffer_store(zeros, ptr=o_base, offsets=o_offsets, mask=o_mask)
            return

    # A trailing partial key block (seqlen_k not a multiple of BLOCK_N) must be
    # masked on its boundary.
    n_extra_tokens = 0
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N
    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = (not padded_block_k) and (seqlen_q % BLOCK_M == 0)

    if IS_CAUSAL:
        # At least BLOCK_M // BLOCK_N diagonal blocks, plus one more when the
        # seqlens are not aligned.
        masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        masked_blocks = padded_block_k

    masked_blocks = min(masked_blocks, n_blocks)
    n_full_blocks = n_blocks - masked_blocks
    block_min = 0
    block_max = n_blocks * BLOCK_N

    # Full blocks: no boundary mask, no causal mask.
    if n_full_blocks > 0:
        block_max = n_full_blocks * BLOCK_N
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            k_base,
            k_offsets,
            v_base,
            v_offsets,
            smemK,
            smemV,
            stride_kn,
            stride_vn,
            seqlen_q,
            seqlen_k,
            block_min,
            block_max,
            0,
            offs_m,
            qk_scale,
            MFMA_LAYOUT=mfmaLayout,
            K_LOAD_LAYOUT=kLoadLayout,
            V_LOAD_LAYOUT=vLoadLayout,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
            IS_CAUSAL=False,
            MASK_STEPS=False,
            PRELOAD_V=PRELOAD_V,
            NUM_KV_BUFFERS=NUM_KV_BUFFERS,
        )
        block_min = block_max
        block_max = n_blocks * BLOCK_N

    # Remaining blocks carry the boundary / causal masking.
    if masked_blocks > 0:
        k_base += n_full_blocks * BLOCK_N * stride_kn
        v_base += n_full_blocks * BLOCK_N * stride_vn
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            k_base,
            k_offsets,
            v_base,
            v_offsets,
            smemK,
            smemV,
            stride_kn,
            stride_vn,
            seqlen_q,
            seqlen_k,
            block_min,
            block_max,
            n_extra_tokens,
            offs_m,
            qk_scale,
            MFMA_LAYOUT=mfmaLayout,
            K_LOAD_LAYOUT=kLoadLayout,
            V_LOAD_LAYOUT=vLoadLayout,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
            IS_CAUSAL=IS_CAUSAL,
            MASK_STEPS=True,
            PRELOAD_V=PRELOAD_V,
            NUM_KV_BUFFERS=NUM_KV_BUFFERS,
        )

    # --- epilogue: normalize and write back -------------------------------------
    acc = acc / l_i[:, None]

    offs_rm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mfmaLayout))
    offs_om = start_m * BLOCK_M + offs_rm
    offs_od = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(0, mfmaLayout))

    # If seqlen_q > seqlen_k and the delta is not a multiple of BLOCK_M, this
    # block straddles the bottom-right causal boundary: rows above it attend to no
    # keys, so softmax over an all-(-inf) row yields NaN. Zero those rows by index
    # (only when the boundary actually falls inside this block), mirroring mha.py.
    end_m_idx = (start_m + 1) * BLOCK_M
    start_m_idx = start_m * BLOCK_M
    causal_start_idx = seqlen_q - seqlen_k
    if IS_CAUSAL:
        if (causal_start_idx > start_m_idx) and (causal_start_idx < end_m_idx):
            out_mask_boundary = gl.full(
                [BLOCK_DMODEL_POW2],
                causal_start_idx,
                dtype=gl.int32,
                layout=gl.SliceLayout(0, mfmaLayout),
            )
            mask_m_offsets = start_m_idx + gl.arange(
                0, BLOCK_M, layout=gl.SliceLayout(1, mfmaLayout)
            )
            out_ptrs_mask = mask_m_offsets[:, None] >= out_mask_boundary[None, :]
            acc = gl.where(out_ptrs_mask, acc, 0.0)

    out = acc.to(o_ptr.dtype.element_ty)

    o_base = (
        o_ptr
        + off_z * stride_oz
        + off_q_head * stride_oh
        + cu_seqlens_q_start * stride_om
        + start_m * BLOCK_M * stride_om
    )
    o_offsets = (offs_rm[:, None] * stride_om + offs_od[None, :] * stride_on).to(gl.int32)

    # Build the store mask exactly as mha.py: start all-true, AND in the query-row
    # bound only when this block overflows seqlen_q, and the head-dim bound only
    # when the head is padded. The [BLOCK_M, 1] base broadcasts over the d-axis.
    overflow_size = end_m_idx - seqlen_q
    out_mask = gl.full([BLOCK_M, 1], True, dtype=gl.int1, layout=mfmaLayout)
    if overflow_size > 0:
        out_mask = out_mask & (offs_om[:, None] < seqlen_q)
    if PADDED_HEAD:
        out_mask = out_mask & (offs_od[None, :] < BLOCK_DMODEL)
    gl.amd.cdna4.buffer_store(out, ptr=o_base, offsets=o_offsets, mask=out_mask)


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
    # The real head dim may be any value; it is padded up to the next power of 2
    # for the layouts/MFMA. MFMA 16x16x32 needs the padded head-dim contraction to
    # be a multiple of 32, and the blocked load layouts need it to divide 512.
    BLOCK_DMODEL_POW2 = max(triton.next_power_of_2(head_dim), 32)
    assert BLOCK_N % 32 == 0, "BLOCK_N must be a multiple of 32"

    # Soundness precondition for the widened head-axis loads: only hint 8-element
    # alignment when every Q/K/V head-axis stride is a multiple of 8 elements.
    # *_strides[1] are the head-axis strides in both bshd and thd layouts.
    head_stride_aligned_8 = (
        q_strides[1] % 8 == 0 and k_strides[1] % 8 == 0 and v_strides[1] % 8 == 0
    )

    grid = (batch * num_q_heads * triton.cdiv(seqlen_q, BLOCK_M), 1)

    _attn_fwd[grid](
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
        BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
        PRELOAD_V=True,
        NUM_XCD=get_num_xcds(),
        HEAD_STRIDE_ALIGNED_8=head_stride_aligned_8,
        num_warps=4,
        waves_per_eu=2,
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
    """Gluon flash-attention forward (fixed-length / padded batch).

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
    """Gluon flash-attention forward for ragged (varlen) batches.

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
