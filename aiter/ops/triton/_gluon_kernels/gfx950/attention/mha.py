# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import json
import torch
import triton
import triton.language as tl

from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton.pid_preprocessing import remap_xcd
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@triton.jit
def _cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def _load_fn(ptrs, offset_first, offset_second, boundary_first, boundary_second):
    if offset_first is not None and offset_second is not None:
        mask = (offset_first[:, None] < boundary_first) & (
            offset_second[None, :] < boundary_second
        )
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_first is not None:
        mask = offset_first[:, None] < boundary_first
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_second is not None:
        mask = offset_second[None, :] < boundary_second
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    else:
        tensor = tl.load(ptrs)
    return tensor


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    k_ptrs,
    v_ptrs,
    stride_kn,
    stride_vk,
    start_m,
    seqlen_k,
    seqlen_q,
    block_min,
    block_max,
    offs_n_causal,
    masked_blocks,
    n_extra_tokens,
    OFFS_M: tl.constexpr,
    OFFS_N: tl.constexpr,
    PRELOAD_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_POW2: tl.constexpr,
    SM_SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    MASK_STEPS: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    ENABLE_PIPELINING: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634

    # loop over k, v, and update accumulator

    num_stages: tl.constexpr = (
        None if ENABLE_PIPELINING else 1
    )  # Set num_stages==1 if we want to disable pipelining
    for start_n in tl.range(block_min, block_max, BLOCK_N, num_stages=num_stages):
        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        if MASK_STEPS:
            k_offs_n = start_n + tl.arange(0, BLOCK_N)
        else:
            k_offs_n = None
        k_offs_k = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL_POW2)
        k = _load_fn(k_ptrs, k_offs_k, k_offs_n, BLOCK_DMODEL, seqlen_k)
        if PRELOAD_V:
            v = _load_fn(v_ptrs, k_offs_n, k_offs_k, seqlen_k, BLOCK_DMODEL)

        # We start from end of seqlen_k so only the first iteration would need
        # to be checked for padding if it is not a multiple of block_n
        # TODO: This can be optimized to only be true for the padded block.
        mask = tl.full([BLOCK_M, BLOCK_N], True, dtype=tl.int1)
        if MASK_STEPS:
            # If this is the last block / iteration, we want to
            # mask if the sequence length is not a multiple of block size
            # a solution is to always do BLOCK_M // BLOCK_N + 1 steps if not is_modulo_mn.
            # last step might get wasted but that is okay. check if this masking works For
            # that case.

            # remove the old if condition
            # if (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0):
            # Though this will unconditionally compute mask_partial at runtime,
            # the causal for loop does not have the if-else block any more, which
            # helps instruction scheduling and register pressure.
            bound_cond = (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0)
            size_n = start_n + OFFS_N[None, :]
            mask_partial = size_n < seqlen_k
            mask = tl.where(bound_cond, mask_partial, mask)

        qk_scale = SM_SCALE * RCP_LN2
        # -- compute qk ----
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk)
        qk = qk * qk_scale
        if IS_CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = OFFS_M[:, None] >= causal_boundary[None, :]
            mask = mask & causal_mask

        qk = tl.where(mask, qk, float("-inf"))

        # get max scores so far
        m_ij = tl.maximum(m_i, tl.max(qk, 1))

        # Compute scaled QK and softmax probabilities
        p = tl.math.exp2(qk - m_ij[:, None])

        l_ij = tl.sum(p, 1)

        # -- update output accumulator --
        # alpha is an adjustment factor for acc and li as we loop and find new maxes
        # store the diff in maxes to adjust acc and li as we discover new maxes
        alpha = tl.math.exp2(m_i - m_ij)

        acc = acc * alpha[:, None]
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij
        if not PRELOAD_V:
            v = _load_fn(v_ptrs, k_offs_n, k_offs_k, seqlen_k, BLOCK_DMODEL)

        acc = tl.dot(p.to(v.type.element_ty), v, acc=acc)

        k_ptrs += BLOCK_N * stride_kn

        v_ptrs += BLOCK_N * stride_vk

    return acc, l_i, m_i


_attn_fwd_repr = make_kernel_repr(
    "_attn_fwd",
    [
        "IS_CAUSAL",
        "NUM_Q_HEADS",
        "NUM_K_HEADS",
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_DMODEL",
        "VARLEN",
        "NUM_XCD",
        "USE_INT64_STRIDES",    
    ],
)


@triton.jit(repr=_attn_fwd_repr)
def _attn_fwd(
    q_ptr: torch.Tensor,
    k_ptr: torch.Tensor,
    v_ptr: torch.Tensor,
    out_ptr: torch.Tensor,

    stride_qz_in,
    stride_qh_in,
    stride_qm_in,
    stride_qk_in,
    stride_kz_in,
    stride_kh_in,
    stride_kn_in,
    stride_kk_in,
    stride_vz_in,
    stride_vh_in,
    stride_vn_in,
    stride_vk_in,
    stride_oz_in,
    stride_oh_in,
    stride_om_in,
    stride_on_in,
    sm_scale,
    cu_seqlens_q,
    cu_seqlens_k,
    SEQLEN_Q,
    SEQLEN_K,
    IS_CAUSAL: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    PRELOAD_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_POW2: tl.constexpr,
    VARLEN: tl.constexpr,
    BATCH,
    NUM_XCD: tl.constexpr,
    USE_INT64_STRIDES: tl.constexpr,
    HEAD_STRIDE_ALIGNED_8: tl.constexpr = False,
):
    NUM_BLOCKS = (SEQLEN_Q + BLOCK_M - 1) // BLOCK_M
    # calculate offsets
    wid = tl.program_id(
        0
    )  # workgroup id ranging: 0,1,2,...., (BATCH * NUM_Q_HEADS * NUM_BLOCKS - 1)
    # num blocks along seqlen

    off_q_head = wid % NUM_Q_HEADS
    off_q_head = remap_xcd(off_q_head, NUM_Q_HEADS, NUM_XCD)
    start_m = (wid // NUM_Q_HEADS) % NUM_BLOCKS
    off_z = (wid // (NUM_BLOCKS * NUM_Q_HEADS)) % BATCH

    # offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL_POW2)

    # NOTE:
    # Workaround for int64 strides, In the absence of strides being int64, parts of the offset
    # computation is done in 32 bit and overflows resulting in segfaults
    # If input strides are defined as int64, it disables vectorized loads which drops perf
    # If we define new strides as stride_x = stride_x_in.to(tl.int64), that does not work
    # because strides are tl.constexpr and cannot be upcasted
    # If we define new strides as stride_x: tl.int64 = stride_x_in, segfault remains
    # The permanent solution is to enable upcasting of tl.constexpr
    # In the meantime, the following workaround provides correctness and does not drop perf
    if USE_INT64_STRIDES:
        stride_qz = tl.cast(stride_qz_in, tl.int64)
        stride_qh = tl.cast(stride_qh_in, tl.int64)
        stride_qm = tl.cast(stride_qm_in, tl.int64)
        stride_qk = tl.cast(stride_qk_in, tl.int64)
        stride_kz = tl.cast(stride_kz_in, tl.int64)
        stride_kh = tl.cast(stride_kh_in, tl.int64)
        stride_kn = tl.cast(stride_kn_in, tl.int64)
        stride_kk = tl.cast(stride_kk_in, tl.int64)
        stride_vz = tl.cast(stride_vz_in, tl.int64)
        stride_vh = tl.cast(stride_vh_in, tl.int64)
        stride_vn = tl.cast(stride_vn_in, tl.int64)
        stride_vk = tl.cast(stride_vk_in, tl.int64)
        stride_oz = tl.cast(stride_oz_in, tl.int64)
        stride_oh = tl.cast(stride_oh_in, tl.int64)
        stride_om = tl.cast(stride_om_in, tl.int64)
        stride_on = tl.cast(stride_on_in, tl.int64)
    else:
        stride_qz = stride_qz_in
        stride_qm = stride_qm_in
        stride_qk = stride_qk_in
        stride_qh = stride_qh_in
        stride_kz = stride_kz_in
        stride_kh = stride_kh_in
        stride_kn = stride_kn_in
        stride_kk = stride_kk_in
        stride_vz = stride_vz_in
        stride_vh = stride_vh_in
        stride_vn = stride_vn_in
        stride_vk = stride_vk_in
        stride_oz = stride_oz_in
        stride_oh = stride_oh_in
        stride_om = stride_om_in
        stride_on = stride_on_in

    tl.assume(stride_qz_in >= 0)
    tl.assume(stride_qh_in >= 0)
    tl.assume(stride_qm_in >= 0)
    tl.assume(stride_qk_in >= 0)
    tl.assume(stride_kz_in >= 0)
    tl.assume(stride_kh_in >= 0)
    tl.assume(stride_kn_in >= 0)
    tl.assume(stride_kk_in >= 0)
    tl.assume(stride_vz_in >= 0)
    tl.assume(stride_vh_in >= 0)
    tl.assume(stride_vn_in >= 0)
    tl.assume(stride_vk_in >= 0)

    if VARLEN:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)

        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        # We have a one-size-fits-all grid in id(0). Some seqlens might be too
        # small for all start_m so for those we return early.
        if start_m * BLOCK_M > seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = SEQLEN_Q
        seqlen_k = SEQLEN_K

    n_blocks = _cdiv_fn(seqlen_k, BLOCK_N)

    # Now we compute whether we need to exit early due to causal masking.
    # This is because for seqlen_q > seqlen_k, M rows of the attn scores
    # are completely masked, resulting in 0s written to the output.
    # We don't need to do any GEMMs in this case.
    # This block of code determines what N is, and if this WG is operating
    # on those M rows.
    if IS_CAUSAL:
        # If seqlen_q == seqlen_k, the attn scores are a square matrix.
        # If seqlen_q != seqlen_k, attn scores are rectangular which means
        # the causal mask boundary is bottom right aligned, and ends at either
        # the top edge (seqlen_q < seqlen_k) or left edge.

        # This captures the decrease in n_blocks if we have a rectangular attn matrix
        n_blocks_seqlen = _cdiv_fn(
            (start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N
        )

        # This is what adjusts the block_max for the current WG, only
        # if IS_CAUSAL. Otherwise we want to always iterate through all n_blocks
        n_blocks = min(n_blocks, n_blocks_seqlen)

        # If we have no blocks after adjusting for seqlen deltas, this WG is part of
        # the blocks that are all 0. We exit early.
        if n_blocks <= 0:
            offs_out = (
                off_z * stride_oz
                + off_q_head * stride_oh
                + cu_seqlens_q_start * stride_om
                + offs_m[:, None] * stride_om
                + offs_d[None, :] * stride_on
            )
            acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_POW2], dtype=out_ptr.type.element_ty)
            out_mask = (offs_m[:, None] < seqlen_q) & (offs_d[None, :] < BLOCK_DMODEL)
            tl.store(out_ptr + offs_out, acc, mask=out_mask)

            return

    grp_sz: tl.constexpr = NUM_Q_HEADS // NUM_K_HEADS
    if grp_sz != 1:  # Grouped Query Attention
        off_k_head = off_q_head // grp_sz
    else:
        off_k_head = off_q_head

    # q,k,v offsets
    # When the caller guarantees that the head-axis strides of Q/K/V are
    # multiples of 8 elements (set via HEAD_STRIDE_ALIGNED_8), the head-axis
    # byte offset is 16-byte aligned. Auto-specialization only fires at the
    # 16-element threshold, so hint the smaller multiple explicitly to let
    # AxisInfo widen the global load.
    qh_off = off_q_head * stride_qh
    kh_off = off_k_head * stride_kh
    vh_off = off_k_head * stride_vh
    if HEAD_STRIDE_ALIGNED_8:
        qh_off = tl.multiple_of(qh_off, 8)
        kh_off = tl.multiple_of(kh_off, 8)
        vh_off = tl.multiple_of(vh_off, 8)

    q_offs = (
        off_z * stride_qz
        + qh_off
        + cu_seqlens_q_start * stride_qm
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qk
    )
    q_ptrs = q_ptr + q_offs

    k_offs = (
        off_z * stride_kz
        + kh_off
        + cu_seqlens_k_start * stride_kn
        + offs_d[:, None] * stride_kk
        + offs_n[None, :] * stride_kn
    )
    k_ptrs = k_ptr + k_offs

    v_offs = (
        off_z * stride_vz
        + vh_off
        + cu_seqlens_k_start * stride_vn
        + offs_n[:, None] * stride_vn
        + offs_d[None, :] * stride_vk
    )
    v_ptrs = v_ptr + v_offs

    m_i_value = float("-inf")

    m_i = tl.full([BLOCK_M], m_i_value, dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_POW2], dtype=tl.float32)
    if BLOCK_DMODEL == BLOCK_DMODEL_POW2:
        q_mask = offs_m[:, None] < seqlen_q
    else:
        q_mask = (offs_m[:, None] < seqlen_q) & (offs_d[None, :] < BLOCK_DMODEL)

    if BLOCK_M >= NUM_Q_HEADS:
        q_cache_mod: tl.constexpr = ".cg"
    else:
        q_cache_mod: tl.constexpr = ""

    q = tl.load(q_ptrs, mask=q_mask, other=0.0, cache_modifier=q_cache_mod)

    n_extra_tokens = 0
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N

    # if CAUSAL, then determine masked_blocks and full blocks
    # Here we compute how many full and masked blocks we have.
    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = not padded_block_k and (seqlen_q % BLOCK_M == 0)

    if IS_CAUSAL:
        # There are always at least BLOCK_M // BLOCK_N masked blocks.
        # Additionally there might be one more due to dissimilar seqlens.
        masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        # Padding on Q does not need to be masked in the FA loop.
        masked_blocks = padded_block_k
    # if IS_CAUSAL, not is_modulo_mn does not always result in an additional block.
    # In this case we might exceed n_blocks so pick the min.
    visible_blocks = n_blocks
    masked_blocks = min(masked_blocks, visible_blocks)
    n_full_blocks = visible_blocks - masked_blocks
    block_min = 0
    block_max = n_blocks * BLOCK_N
    # Compute for full blocks. Here we set causal to false regardless of its actual
    # value because there is no masking. Similarly we do not need padding.
    if n_full_blocks > 0:
        block_max = block_min + n_full_blocks * BLOCK_N
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            k_ptrs,
            v_ptrs,
            stride_kn,
            stride_vn,
            start_m,
            seqlen_k,
            seqlen_q,
            block_min,
            block_max,
            0,
            0,
            0,
            offs_m,
            offs_n,
            PRELOAD_V,
            BLOCK_M,
            BLOCK_N,
            BLOCK_DMODEL,
            BLOCK_DMODEL_POW2,
            sm_scale,
            False,
            MASK_STEPS=False,
            PADDED_HEAD=BLOCK_DMODEL != BLOCK_DMODEL_POW2,
            ENABLE_PIPELINING=True,
        )
        block_min = block_max
        block_max = n_blocks * BLOCK_N

    # Remaining blocks, if any, are full / not masked.
    if masked_blocks > 0:
        if IS_CAUSAL:
            offs_n_causal = offs_n + (seqlen_q - seqlen_k)
        else:
            offs_n_causal = 0
        k_ptrs += n_full_blocks * BLOCK_N * stride_kn
        v_ptrs += n_full_blocks * BLOCK_N * stride_vn
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            k_ptrs,
            v_ptrs,
            stride_kn,
            stride_vn,
            start_m,
            seqlen_k,
            seqlen_q,
            block_min,
            block_max,
            offs_n_causal,
            masked_blocks,
            n_extra_tokens,
            offs_m,
            offs_n,
            PRELOAD_V,
            BLOCK_M,
            BLOCK_N,
            BLOCK_DMODEL,
            BLOCK_DMODEL_POW2,
            sm_scale,
            IS_CAUSAL,
            MASK_STEPS=True,
            PADDED_HEAD=BLOCK_DMODEL != BLOCK_DMODEL_POW2,
            ENABLE_PIPELINING=False,
        )
    # epilogue
    # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    # If seqlen_q > seqlen_k but the delta is not a multiple of BLOCK_M,
    # then we have one block with a row of all NaNs which come from computing
    # softmax over a row of all -infs (-inf - inf = NaN). We check for that here
    # and store 0s where there are NaNs as these rows should've been zeroed out.
    end_m_idx = (start_m + 1) * BLOCK_M
    start_m_idx = start_m * BLOCK_M
    causal_start_idx = seqlen_q - seqlen_k
    if IS_CAUSAL:
        if causal_start_idx > start_m_idx and causal_start_idx < end_m_idx:
            out_mask_boundary = tl.full(
                (BLOCK_DMODEL_POW2,), causal_start_idx, dtype=tl.int32
            )
            mask_m_offsets = start_m_idx + tl.arange(0, BLOCK_M)
            out_ptrs_mask = mask_m_offsets[:, None] >= out_mask_boundary[None, :]
            z = 0.0
            acc = tl.where(out_ptrs_mask, acc, z.to(acc.type.element_ty))

    overflow_size = end_m_idx - seqlen_q

    # write back O
    offs_out = (
        off_z * stride_oz
        + off_q_head * stride_oh
        + cu_seqlens_q_start * stride_om
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_on
    )
    out_mask = tl.full([BLOCK_M, 1], 1, dtype=tl.int1)
    if overflow_size > 0:
        out_mask = out_mask & (offs_m[:, None] < seqlen_q)
    if BLOCK_DMODEL != BLOCK_DMODEL_POW2:
        out_mask = out_mask & (offs_d[None, :] < BLOCK_DMODEL)
    op = acc.to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs_out, op, mask=out_mask)


@functools.lru_cache(maxsize=1024)
def _get_config(
    dtype: torch.dtype,
):
    assert isinstance(dtype, torch.dtype)
    if not hasattr(_get_config, "_config_dict"):
        dev = arch_info.get_arch()
        _get_config._config_dict = {}
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/{dev}-MHA-DEFAULT.json"
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict["default"] = config
    fwd_cfg = _get_config._config_dict["default"]["fwd"]

    return fwd_cfg["default"]
