// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <hip/hip_bf16.h>
#include "hip_compat.h"

#include "dtype_fp8.cuh"
#include "quant_utils.cuh"

#include <ck_tile/ops/fmha/block/block_masking.hpp>
#include <ck_tile/ops/fmha/block/variants.hpp>

#if defined(NDEBUG)
#undef NDEBUG
#include <assert.h>
#define UNREACHABLE_CODE assert(false);
#define NDEBUG
#else
#define UNREACHABLE_CODE assert(false);
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b)-1) / (b))


using floatx4   = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using float16x4 = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
typedef float16x4 _Half4;
using float16x2 = __attribute__((__vector_size__(2 * sizeof(_Float16)))) _Float16;
typedef float16x2 _Half2;
typedef struct _Half8
{
    _Half4 xy[2];
} _Half8;

using bit16_t = uint16_t;
using bit16x4 = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;
typedef bit16x4 _B16x4;
typedef struct _B16x8
{
    _B16x4 xy[2];
} _B16x8;

using bit16x8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) uint16_t;
typedef bit16x8 _B16x8_2;

using _B8x8  = uint2;
using _B8x4  = int32_t; // used in builtins
using bit8_t = uint8_t;

typedef struct _B8x16
{
    _B8x8 xy[2];
} _B8x16;

////// Non temporal loads ///////
template <typename T>
__device__ __forceinline__ T loadnt(T* addr)
{
    return __builtin_nontemporal_load(addr);
}

__device__ __forceinline__ _B16x8 load_ntmprl_16Byte(const _B16x8* addr)
{
    auto addr_alias = reinterpret_cast<const float*>(addr);
    auto dat0       = loadnt(addr_alias);
    auto dat1       = loadnt(addr_alias + 1);
    auto dat2       = loadnt(addr_alias + 2);
    auto dat3       = loadnt(addr_alias + 3);
    auto res        = make_float4(dat0, dat1, dat2, dat3);
    return *reinterpret_cast<_B16x8*>(&res);
}

#if defined(__gfx950__)
template <typename T, int absz, int cbid, int blgp>
__device__ __forceinline__ floatx4 gcn_mfma16x16x32_instr(const _B16x8& inpA,
                                                          const _B16x8& inpB,
                                                          const floatx4& inpC)
{
    _B16x8_2 tmpA = __builtin_shufflevector(inpA.xy[0], inpA.xy[1], 0, 1, 2, 3, 4, 5, 6, 7);
    _B16x8_2 tmpB = __builtin_shufflevector(inpB.xy[0], inpB.xy[1], 0, 1, 2, 3, 4, 5, 6, 7);

    if constexpr(std::is_same<T, _Float16>::value)
    {
        return __builtin_amdgcn_mfma_f32_16x16x32_f16(tmpA, tmpB, inpC, absz, cbid, blgp);
    }
    else if constexpr(std::is_same<T, __hip_bfloat16>::value)
    {
        return __builtin_amdgcn_mfma_f32_16x16x32_bf16(tmpA, tmpB, inpC, absz, cbid, blgp);
    }
    else
    {
        static_assert(false, "unsupported 16b dtype");
    }
}
#else
template <typename T, int absz, int cbid, int blgp>
__device__ __forceinline__ floatx4 gcn_mfma16x16x16_instr(const _B16x4& inpA,
                                                          const _B16x4& inpB,
                                                          const floatx4& inpC)
{
    if constexpr(std::is_same<T, _Float16>::value)
    {
        return __builtin_amdgcn_mfma_f32_16x16x16f16(inpA, inpB, inpC, absz, cbid, blgp);
    }
    else if constexpr(std::is_same<T, __hip_bfloat16>::value)
    {
        return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(inpA, inpB, inpC, absz, cbid, blgp);
    }
    else
    {
        static_assert(false, "unsupported 16b dtype");
    }
}
#endif

template <typename T>
__device__ __forceinline__ float to_float(const T& inp)
{
    if constexpr(std::is_same<T, _Float16>::value)
    {
        return (float)inp;
    }
    else if constexpr(std::is_same<T, __hip_bfloat16>::value)
    {
        return __bfloat162float(inp);
    }
    else
    {
        static_assert(false, "unsupported 16b dtype");
    }
}

template <typename T>
__device__ __forceinline__ float to_float_b16(const bit16_t& inp)
{
    union tmpcvt
    {
        bit16_t u;
        _Float16 f;
        __hip_bfloat16 b;
    } t16;
    t16.u = inp;
    if constexpr(std::is_same<T, _Float16>::value)
    {
        return (float)t16.f;
    }
    else if constexpr(std::is_same<T, __hip_bfloat16>::value)
    {
        return __bfloat162float(t16.b);
    }
    else
    {
        static_assert(false, "unsupported 16b dtype");
    }
}

template <typename T>
__device__ __forceinline__ T from_float(const float& inp)
{
    if constexpr(std::is_same<T, _Float16>::value)
    {
        return (_Float16)inp;
    }
    else if constexpr(std::is_same<T, __hip_bfloat16>::value)
    {
        return __float2bfloat16(inp);
    }
    else
    {
        static_assert(false, "unsupported 16b dtype");
    }
}

template <typename T>
__device__ __forceinline__ _B16x4 from_floatx4(const floatx4& inp)
{
    union tmpcvt
    {
        uint16_t u;
        _Float16 f;
        __hip_bfloat16 b;
    } t16;
    _B16x4 ret;
    if constexpr(std::is_same<T, _Float16>::value)
    {
        union h2cvt
        {
            __half2 h2[2];
            _B16x4 b16x4;
        } u;
        u.h2[0] = __float22half2_rn(make_float2(inp[0], inp[1]));
        u.h2[1] = __float22half2_rn(make_float2(inp[2], inp[3]));
        return u.b16x4;
    }
    else if constexpr(std::is_same<T, __hip_bfloat16>::value)
    {
        for(int i = 0; i < 4; i++)
        {
            union fcvt
            {
                uint32_t u32;
                float f32;
            } u;
            u.f32 = inp[i];
            u.u32 += 0x7fff + ((u.u32 >> 16) & 1); // BF16 RNE with no nan/inf check
            ret[i] = uint16_t(u.u32 >> 16);
        }
        return ret;
    }
    else
    {
        static_assert(false, "unsupported 16b dtype");
    }
}

template <typename T>
__device__ __forceinline__ _B16x4 addx4(const _B16x4& inp1, const _B16x4& inp2)
{
    union tmpcvt
    {
        uint16_t u;
        _Float16 f;
        __hip_bfloat16 b;
    } t1, t2, res;
    _B16x4 ret;
    if constexpr(std::is_same<T, _Float16>::value)
    {
        union h2cvt
        {
            _B16x4 b16x4;
            __half2 h2[2];
        } u1, u2, s;
        u1.b16x4 = inp1;
        u2.b16x4 = inp2;
        s.h2[0]  = u1.h2[0] + u2.h2[0];
        s.h2[1]  = u1.h2[1] + u2.h2[1];
        return s.b16x4;
    }
    else if constexpr(std::is_same<T, __hip_bfloat16>::value)
    {
        for(int i = 0; i < 4; i++)
        {
            union fcvt
            {
                float f32;
                uint32_t i32;
            } u1, u2, s;
            u1.i32 = uint32_t(inp1[i]) << 16;
            u2.i32 = uint32_t(inp2[i]) << 16;
            s.f32  = u1.f32 + u2.f32;
            ret[i] = uint16_t(s.i32 >> 16);
        }
        return ret;
    }
    else
    {
        static_assert(false, "unsupported 16b dtype");
    }
}


__device__ __forceinline__ floatx4 to_float_fp8x4(const _B8x4& inp)
{
#if defined(__gfx90a__)
    float4 f32x4 =
        vllm::fp8::vec_conversion<float4, uint32_t>(*reinterpret_cast<const uint32_t*>(&inp));
    return *reinterpret_cast<floatx4*>(&f32x4);
#else // MI3xx+ optimized builtins
    const auto f0 = __builtin_amdgcn_cvt_pk_f32_fp8(inp, false);
    const auto f1 = __builtin_amdgcn_cvt_pk_f32_fp8(inp, true);
    floatx4 ret;
    ret[0] = f0[0];
    ret[1] = f0[1];
    ret[2] = f1[0];
    ret[3] = f1[1];
    return ret;
#endif
}

template <typename T>
__device__ __forceinline__ _B16x4 from_floatx4_rtz(const floatx4& inp)
{
    _B16x4 ret;
    if constexpr(std::is_same<T, _Float16>::value)
    {
        union h2cvt
        {
            _Half2 h2[2];
            _B16x4 b16x4;
        } u;
        u.h2[0] = __builtin_amdgcn_cvt_pkrtz(inp[0], inp[1]);
        u.h2[1] = __builtin_amdgcn_cvt_pkrtz(inp[2], inp[3]);
        return u.b16x4;
    }
    else if constexpr(std::is_same<T, __hip_bfloat16>::value)
    {
        for(int i = 0; i < 4; i++)
        {
            union fcvt
            {
                uint32_t i32;
                float f32;
            } u;
            u.f32  = inp[i];
            ret[i] = uint16_t(u.i32 >> 16);
        }
        return ret;
    }
    else
    {
        static_assert(false, "unsupported 16b dtype");
    }
}

template <typename T>
__device__ __forceinline__ _B16x8 convert_b8x8_custom(const _B8x8 input)
{
    union
    {
        _B8x8 b8x8;
        _B8x4 b8x4[2];
    } tmp;
    tmp.b8x8 = input;
    _B16x8 ret;
    for(int i = 0; i < 2; i++)
    {
        ret.xy[i] = from_floatx4_rtz<T>(to_float_fp8x4(tmp.b8x4[i]));
    }
    return ret;
}

template <typename scalar_t,
          typename cache_t,
          vllm::Fp8KVCacheDataType KV_DTYPE,
          int BLOCK_SIZE,
          int HEAD_SIZE,
          int NUM_THREADS,
          bool ALIBI_ENABLED,
          int GQA_RATIO,
          typename AttentionVariant>
__device__ void _paged_attention_kernel(
    const int* block_table_seq,
    const int64_t query_loc,
    int context_len,
    const int partition_start_token_idx,
    const scalar_t* q,
    const cache_t* k_cache,
    const cache_t* v_cache,     
    const float scale,
    const float* __restrict__ alibi_slopes,    // [num_heads]
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    const int kv_seq_stride,
    float* __restrict__ exp_sums,   // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits, // [num_seqs, num_heads,
                                    // max_num_partitions]
    scalar_t* __restrict__ out,     // [num_seqs, num_heads, max_num_partitions,
                                    // head_size]
    float logits_soft_cap,
    float logits_soft_cap_rcp,
    const float* k_scale_ptr,
    const float* v_scale_ptr,
    const AttentionVariant* variant)
{
    const int seq_idx       = blockIdx.x;
    const int partition_idx = blockIdx.y;
    constexpr int T_PAR_SIZE = 256; 
    constexpr int NWARPS = NUM_THREADS / WARP_SIZE;
    const int warpid     = threadIdx.x / WARP_SIZE;
    const int laneid     = threadIdx.x % WARP_SIZE;
    const int lane4id    = laneid % 4;
    const int lane16id   = laneid % 16;
    const int rowid      = laneid / 16;

    const int max_num_partitions = gridDim.y;
    constexpr int GQA_RATIO4 = DIVIDE_ROUND_UP(GQA_RATIO, 4);

    __shared__ float shared_qk_max[NWARPS][16 + 1];
    __shared__ float shared_exp_sum[NWARPS][16 + 1];
    // shared_logits is used for multiple purposes
    __shared__ _B16x4 shared_logits[NWARPS][4][16][4];

    // for QK mfma16x16, layout is QHead/Tokenx16 across every 16 lanes, 16 Bytes
    // HeadElements in each lane, 4x16B HeadElements across 4 rows of warp
    constexpr int ROWS_PER_WARP = WARP_SIZE / 16; // rows refers to 16 lanes; refer dpp terminology
    constexpr int CONTIGUOUS_KV_ELEMS_16B_LOAD =
        16 / sizeof(cache_t); // 8 for 16 bit cache type, 16 for 8 bit types
    constexpr int QKHE_PER_FETCH =
        CONTIGUOUS_KV_ELEMS_16B_LOAD *
        ROWS_PER_WARP; // each fetch across a warp fetches these many elements
    constexpr int QK_SIZE_RATIO =
        sizeof(scalar_t) / sizeof(cache_t);              // 1 for 16bit types, 2 for 8bit types
    constexpr int QKHELOOP = HEAD_SIZE / QKHE_PER_FETCH; // 4xQKHE_16B across warp

    _B16x8 Qlocal[QKHELOOP][QK_SIZE_RATIO]; // note that 16 contiguous elements of Q should
                                            // be fetched per lane for 8 bit cache types :
                                            // QK_SIZE_RATIO changes for this

    constexpr int CONTIGUOUS_SCALAR_ELEMS_16B = 16 / sizeof(scalar_t);

    constexpr int TOKENS_PER_WARP =
        T_PAR_SIZE / NWARPS; // sub partition of tokens per warp for qk calculation
    constexpr int TLOOP = TOKENS_PER_WARP / 16; // each mfma16x16x16 instruction processes 16 tokens

    _B16x8 Klocal[TLOOP][QKHELOOP]; // can be interpreted as B8x16 for 8 bit types

    const int wg_start_head_idx    = blockIdx.z * GQA_RATIO;
    const int wg_start_kv_head_idx = blockIdx.z;
    const int total_num_heads      = gridDim.z * GQA_RATIO;

    /// NOTICE: We don't support mask for this kernel, so just use a placeholder type/object here.
    using Mask = ck_tile::SimplifiedGenericAttentionMask</*IsMasking=*/false>;
    const Mask mask{/*seqlen_q=*/1, /*seqlen_k=*/context_len};

    const auto variant_params = [&] {
        if constexpr(AttentionVariant::use_logits_soft_cap)
        {
            return ck_tile::LogitsSoftCapParams<Mask, AttentionVariant::use_exp2>{
                mask, scale, logits_soft_cap, logits_soft_cap_rcp};
        }
        else
        {
            return ck_tile::StandardAttentionParams<Mask>{mask, scale};
        }
    }();

    // for QK mfma, tokens in multiples of TOKENS_PER_WARP are spread across warps
    // each mfma takes QH16xT16x16HE across warp
    // repeat mfmas across QKHELOOP dimension
    // output layout from QKmfma : QH16xT4x4 16 qheads across 16 lanes, 16 tokens
    // across 4 rows x 4 tokens per lane

    const int num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);
    const int last_ctx_block     = num_context_blocks - 1;

    int kphysical_block_number[TLOOP];

    // fetch k physical block numbers
    for(int token_depth = 0; token_depth < TLOOP; token_depth++)
    {
        const int klocal_token_idx  = TOKENS_PER_WARP * warpid + token_depth * 16 + lane16id;
        const int kglobal_token_idx = partition_start_token_idx + klocal_token_idx;
        const int kblock_idx =
            (kglobal_token_idx < context_len) ? kglobal_token_idx / BLOCK_SIZE : last_ctx_block;
        kphysical_block_number[token_depth] = block_table_seq[kblock_idx];
    }

    // fetch Q in shared across warps and then write to registers
    const int local_qhead_idx  = 4 * warpid + rowid;
    const int global_qhead_idx = wg_start_head_idx + local_qhead_idx;

    const scalar_t* q_ptr      = q + query_loc * q_stride + global_qhead_idx * HEAD_SIZE;

    const int qhead_element = lane16id * CONTIGUOUS_SCALAR_ELEMS_16B;
    if((local_qhead_idx < GQA_RATIO) && (qhead_element < HEAD_SIZE))
    {
        const scalar_t* q_fetch_ptr   = q_ptr + qhead_element;
        const _B16x8* q_fetch_ptr_16B = reinterpret_cast<const _B16x8*>(q_fetch_ptr);
        _B16x8 tmp                    = *q_fetch_ptr_16B;
        if constexpr(KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto)
        {
            const int offset1 =
                lane16id / 4; // 16 contiguous chunks of head elems are spread across 4x4lanes
            shared_logits[offset1][lane4id][local_qhead_idx][0] = tmp.xy[0];
            shared_logits[offset1][lane4id][local_qhead_idx][1] = tmp.xy[1];
        }
        else
        {
            for(int i = 0; i < 2; i++)
            {
                const int head_elem = lane16id * 2 + i; // element id in _B16x4 terms
                const int offset3   = head_elem % 4;
                const int offset2   = (head_elem / 4) % 4;
                const int offset1   = head_elem / 4 / 4;
                shared_logits[offset1][offset2][local_qhead_idx][offset3] = tmp.xy[i];
            }
        }
    }
    __syncthreads();
    for(int qkhe_depth = 0; qkhe_depth < QKHELOOP; qkhe_depth++)
    {
        for(int qkratio = 0; qkratio < QK_SIZE_RATIO; qkratio++)
        {
            for(int i = 0; i < 2; i++)
            {
                Qlocal[qkhe_depth][qkratio].xy[i] =
                    shared_logits[qkhe_depth][rowid][lane16id % GQA_RATIO][2 * qkratio + i];
            }
        }
    }

    // set to true to enable non temporal kv loads: has some benefit in very high
    // batch size cases
    constexpr bool NT_KV_LOAD = false;

    constexpr int KX     = 16 / sizeof(cache_t); // vLLM defines x as 16 Bytes of kv cache elements
    const cache_t* k_ptr = k_cache + wg_start_kv_head_idx * kv_head_stride;

    const int row_head_elem = rowid * CONTIGUOUS_KV_ELEMS_16B_LOAD;
    // fetch K values
    for(int token_depth = 0; token_depth < TLOOP; token_depth++)
    {
        const int64_t kblock_number = static_cast<int64_t>(kphysical_block_number[token_depth]);
        const cache_t* k_ptr2       = k_ptr + kblock_number * kv_block_stride;
        const int klocal_token_idx  = TOKENS_PER_WARP * warpid + token_depth * 16 + lane16id;
        const int kglobal_token_idx = partition_start_token_idx + klocal_token_idx;
        const int kphysical_block_offset = klocal_token_idx % BLOCK_SIZE;
        const cache_t* k_ptr3            = k_ptr2 + kphysical_block_offset * kv_seq_stride;

        for(int qkhe_depth = 0; qkhe_depth < QKHELOOP; qkhe_depth++)
        {
            const int head_elem           = row_head_elem + qkhe_depth * QKHE_PER_FETCH;
            const int offset1             = head_elem / KX;
            const int offset2             = head_elem % KX;
            const cache_t* k_fetch_ptr    = k_ptr3 + offset1 * KX + offset2;
            const _B16x8* k_fetch_ptr_16B = reinterpret_cast<const _B16x8*>(k_fetch_ptr);
            if constexpr(NT_KV_LOAD)
            {
                Klocal[token_depth][qkhe_depth] = load_ntmprl_16Byte(k_fetch_ptr_16B);
            }
            else
            {
                Klocal[token_depth][qkhe_depth] = *k_fetch_ptr_16B;
            }
        }
    }

    float alibi_slope;
    if constexpr(ALIBI_ENABLED)
    {
        const int alibi_head_idx = wg_start_head_idx + lane16id;
        alibi_slope              = (lane16id < GQA_RATIO) ? alibi_slopes[alibi_head_idx] : 0.f;
    }

    constexpr int n_thread_per_warp  = (NWARPS * 16) / CONTIGUOUS_KV_ELEMS_16B_LOAD; // 8
    constexpr int k_thread_per_warp  = WARP_SIZE / n_thread_per_warp;                // 8
    constexpr int n_thread_per_block = n_thread_per_warp;                            // 8
    constexpr int k_thread_per_block = NWARPS * k_thread_per_warp;                   // 32
    constexpr int k_repeat           = TOKENS_PER_WARP / k_thread_per_block;         // 2
    static_assert(BLOCK_SIZE <= k_thread_per_block);

    constexpr int VTOKENS_PER_LANE =
        TOKENS_PER_WARP / ROWS_PER_WARP;       // 64/4 = 16 contiguous vtokens per lane
    constexpr int VBLOCKS_PER_LANE = k_repeat; // assumes block size <= 32
    constexpr int VTLOOP           = NWARPS;   // corresponds to tokens across warps
    constexpr int VTLANELOOP =
        DIVIDE_ROUND_UP(VTOKENS_PER_LANE,
                        CONTIGUOUS_KV_ELEMS_16B_LOAD); // optimized for 16B fetches; assumes
                                                       // minimum block size is 16
    constexpr int VHELOOP = HEAD_SIZE / 16 / NWARPS;   // head_size distributed across warps; each
                                                       // mfma instr works on 16 head elements

    int vphysical_block_number[VTLOOP][VBLOCKS_PER_LANE];

    // fetch v physical block numbers
    for(int vtoken_depth = 0; vtoken_depth < VTLOOP; vtoken_depth++)
    {
        for(int vblock_depth = 0; vblock_depth < VBLOCKS_PER_LANE; vblock_depth++)
        {
            const int vlocal_token_idx = vtoken_depth * TOKENS_PER_WARP +
                                         vblock_depth * k_thread_per_block +
                                         threadIdx.x / n_thread_per_block;
            const int vglobal_token_idx = partition_start_token_idx + vlocal_token_idx;
            const int vblock_idx =
                (vglobal_token_idx < context_len) ? vglobal_token_idx / BLOCK_SIZE : last_ctx_block;
            vphysical_block_number[vtoken_depth][vblock_depth] = block_table_seq[vblock_idx];
        }
    }

    _B16x8 Vlocal[VTLOOP][VHELOOP][VTLANELOOP]; // this can be interpreted as B8x16 too
    __shared__ unsigned char vlds_ptr[TOKENS_PER_WARP * n_thread_per_block * 16];
    static_assert(VBLOCKS_PER_LANE == VTLANELOOP,
                  "make sure we can keep un-shuffled data in Vlocal as well");

    const cache_t* v_ptr = v_cache + wg_start_kv_head_idx * kv_head_stride +
                           ((threadIdx.x / n_thread_per_block) % BLOCK_SIZE) * kv_seq_stride;

    // v fetches are 16head elems across lanes x 16 tokens per lane
    for(int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++)
    {
        for(int vtoken_depth = 0; vtoken_depth < VTLOOP; vtoken_depth++)
        {
            for(int vblock_depth = 0; vblock_depth < VBLOCKS_PER_LANE; vblock_depth++)
            {
                const int vlds_col_idx = laneid % n_thread_per_block;
                const int vhead_elem =
                    vhe_depth * NWARPS * 16 + vlds_col_idx * CONTIGUOUS_KV_ELEMS_16B_LOAD;
                const cache_t* v_ptr2 = v_ptr + vhead_elem;

                const int64_t vblock_number =
                    static_cast<int64_t>(vphysical_block_number[vtoken_depth][vblock_depth]);
                const cache_t* v_fetch_ptr = v_ptr2 + (vblock_number * kv_block_stride);

                Vlocal[vtoken_depth][vhe_depth][vblock_depth] =
                    *reinterpret_cast<const _B16x8*>(v_fetch_ptr);
            }
        }
    }

    // calculate post qk mfma scale
    float scale2 = scale;
    if constexpr(KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto)
    {
        // multiply by k_scale if fp8 kv cache
        scale2 *= *k_scale_ptr;
    }

    floatx4 dout[TLOOP];
    // qk mfma
    for(int token_depth = 0; token_depth < TLOOP; token_depth++)
    {
        dout[token_depth] = {0};
        for(int qkhe_depth = 0; qkhe_depth < QKHELOOP; qkhe_depth++)
        {
            if constexpr(KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto)
            {
                for(int qkratio = 0; qkratio < QK_SIZE_RATIO; qkratio++)
                {
#if defined(__gfx950__)
                    dout[token_depth] = gcn_mfma16x16x32_instr<scalar_t, 0, 0, 0>(
                        Klocal[token_depth][qkhe_depth],
                        Qlocal[qkhe_depth][qkratio],
                        dout[token_depth]);
#else
                    for(int i = 0; i < 2; i++)
                    {
                        dout[token_depth] = gcn_mfma16x16x16_instr<scalar_t, 0, 0, 0>(
                            Klocal[token_depth][qkhe_depth].xy[i],
                            Qlocal[qkhe_depth][qkratio].xy[i],
                            dout[token_depth]);
                    }
#endif
                }
            }
            else
            { // kv cache dtype fp8
                auto Ktmp       = Klocal[token_depth][qkhe_depth];
                _B8x16 Ktmp8x16 = *reinterpret_cast<_B8x16*>(&Ktmp);
                for(int qkratio = 0; qkratio < QK_SIZE_RATIO; qkratio++)
                {
                    _B8x8 Ktmp8x8    = Ktmp8x16.xy[qkratio];
                    _B16x8 Klocaltmp = convert_b8x8_custom<scalar_t>(Ktmp8x8);
#if defined(__gfx950__)
                    dout[token_depth] = gcn_mfma16x16x32_instr<scalar_t, 0, 0, 0>(
                        Klocaltmp,
                        Qlocal[qkhe_depth][qkratio],
                        dout[token_depth]);
#else
                    for(int i = 0; i < 2; i++)
                    {
                        dout[token_depth] = gcn_mfma16x16x16_instr<scalar_t, 0, 0, 0>(
                            Klocaltmp.xy[i], Qlocal[qkhe_depth][qkratio].xy[i], dout[token_depth]);
                    }
#endif
                }
            }
        }
        for(int i = 0; i < 4; i++)
        {
            dout[token_depth][i] = variant->QueryTransform(variant_params, dout[token_depth][i]);
        }
    }

    const int qkout_token_idx = partition_start_token_idx + TOKENS_PER_WARP * warpid + rowid * 4;

    // apply alibi
    if constexpr(ALIBI_ENABLED)
    {
        for(int token_depth = 0; token_depth < TLOOP; token_depth++)
        {
            const int local_token_idx = qkout_token_idx + token_depth * 16;
            const int alibi_offset    = local_token_idx - context_len + 1;
            for(int i = 0; i < 4; i++)
            {
                dout[token_depth][i] += alibi_slope * (alibi_offset + i);
            }
        }
    }
    // apply soft-capping to logits
    for(int token_depth = 0; token_depth < TLOOP; token_depth++)
    {
        for(int i = 0; i < 4; i++)
        {
            dout[token_depth][i] =
                variant->LogitsTransform(variant_params,
                                        dout[token_depth][i],
                                        /*batch_idx=*/blockIdx.x,
                                        /*qo_head_idx=*/wg_start_head_idx + lane16id,
                                        /*kv_head_idx=*/blockIdx.z);
        }
    }

    // calculate qk_max and exp_sum per warp and write to shared memory
    float qk_max  = -FLT_MAX;
    float exp_sum = 0.0f;

    for(int token_depth = 0; token_depth < TLOOP; token_depth++)
    {
        const int local_token_idx = qkout_token_idx + token_depth * 16;
        for(int i = 0; i < 4; i++)
        {
            const float tmp = (local_token_idx + i < context_len) ? dout[token_depth][i] : -FLT_MAX;
            qk_max          = fmaxf(qk_max, tmp);
        }
    }

    for(int mask = WARP_SIZE / 2; mask >= 16; mask /= 2)
    {
        qk_max = fmaxf(qk_max, __shfl_xor(qk_max, mask));
    }

    for(int token_depth = 0; token_depth < TLOOP; token_depth++)
    {
        const int local_token_idx = qkout_token_idx + token_depth * 16;
        for(int i = 0; i < 4; i++)
        {
            const float tmp =
                (local_token_idx + i < context_len) ? __expf(dout[token_depth][i] - qk_max) : 0.0f;
            dout[token_depth][i] = tmp;
            exp_sum += tmp;
        }
    }

    for(int mask = WARP_SIZE / 2; mask >= 16; mask /= 2)
    {
        exp_sum += __shfl_xor(exp_sum, mask);
    }

    __syncthreads(); // sync before writing to shared mem

    float* shared_mem = reinterpret_cast<float*>(shared_logits);
    if(laneid < 16)
    {
        const int qk_max_offset    = warpid * 16 + lane16id;
        shared_mem[qk_max_offset]  = qk_max;
        const int exp_sum_offset   = NWARPS * 16 + qk_max_offset;
        shared_mem[exp_sum_offset] = exp_sum;
    }

    __syncthreads();

    // calculate partition qk_max and exp_sum
    float partition_qk_max = -FLT_MAX;
    float warp_qk_max_exp[NWARPS];
    float partition_exp_sum = 0.0f;

    for(int w = 0; w < NWARPS; w++)
    {
        warp_qk_max_exp[w] = shared_mem[w * 16 + lane16id];
        partition_qk_max   = fmaxf(partition_qk_max, warp_qk_max_exp[w]);
    }

    for(int w = 0; w < NWARPS; w++)
    {
        warp_qk_max_exp[w] = __expf(warp_qk_max_exp[w] - partition_qk_max);
        partition_exp_sum += shared_mem[NWARPS * 16 + w * 16 + lane16id] * warp_qk_max_exp[w];
    }

    const float inv_sum_scale =
        __fdividef(1.f, partition_exp_sum + 1e-6f) * warp_qk_max_exp[warpid];

    __syncthreads();

    // write logits to shared mem
    for(int token_depth = 0; token_depth < TLOOP; token_depth++)
    {
        dout[token_depth] *= inv_sum_scale;
        // use rtz conversion for performance, with no visible impact on accuracy
        shared_logits[warpid][token_depth][lane16id][rowid] =
            from_floatx4_rtz<scalar_t>(dout[token_depth]);
    }
    // write out partition max_logits and exp_sum
    if(threadIdx.x < GQA_RATIO)
    {
        const int qhead_idx = lane16id;
        const int offset    = seq_idx * total_num_heads * max_num_partitions +
                           (wg_start_head_idx + qhead_idx) * max_num_partitions + partition_idx;
        max_logits[offset] = partition_qk_max;
        exp_sums[offset]   = partition_exp_sum;
    }

    __syncthreads();

    constexpr int ELEMS8_ELEMS4_RATIO  = 8 / 4;
    constexpr int ELEMS16_ELEMS8_RATIO = 16 / 8;

    _B16x4 outelems[VHELOOP];
    // Softmax V mfma
    // v layout: 16he across lanes x 16 tokens per lane
    for(int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++)
    {
        floatx4 tmp_out = {0};

        for(int vtoken_depth = 0; vtoken_depth < VTLOOP; vtoken_depth++)
        {
            // 1. store data into LDS
            for(int vblock_depth = 0; vblock_depth < VBLOCKS_PER_LANE; vblock_depth++)
            {
                const int vlds_col_idx = laneid % n_thread_per_block;
                const int vlocal_token_idx =
                    vblock_depth * k_thread_per_block + threadIdx.x / n_thread_per_block;
                *reinterpret_cast<_B16x8*>(vlds_ptr +
                                           (/*row=*/vlocal_token_idx * n_thread_per_block +
                                            /*col=*/vlds_col_idx) *
                                               16) = Vlocal[vtoken_depth][vhe_depth][vblock_depth];
            }
            __syncthreads();

            // 2. load data from LDS (transposed), then do multification
            if constexpr(KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto)
            {
                for(int vfetch_depth = 0; vfetch_depth < VTLANELOOP; vfetch_depth++)
                {
                    {
                        const int vlocal_head_elem = warpid * 16 + lane16id;

                        const int vlds_col_idx  = vlocal_head_elem / CONTIGUOUS_KV_ELEMS_16B_LOAD;
                        const int vlds_elem_idx = vlocal_head_elem % CONTIGUOUS_KV_ELEMS_16B_LOAD;

                        const int vlocal_token_idx =
                            rowid * VTOKENS_PER_LANE + vfetch_depth * CONTIGUOUS_KV_ELEMS_16B_LOAD;

                        // read data points individually and save them into array
                        cache_t elems[CONTIGUOUS_KV_ELEMS_16B_LOAD];
                        for(int d2 = 0; d2 < CONTIGUOUS_KV_ELEMS_16B_LOAD; ++d2)
                        {
                            const cache_t* fetched_elems = reinterpret_cast<const cache_t*>(
                                vlds_ptr + (/*row=*/(vlocal_token_idx + d2) * n_thread_per_block +
                                            /*col=*/vlds_col_idx) *
                                               16);

                            elems[d2] = fetched_elems[vlds_elem_idx];
                        }

                        // copy all the read data points together
                        Vlocal[vtoken_depth][vhe_depth][vfetch_depth] =
                            *reinterpret_cast<const _B16x8*>(elems);
                    }
#if defined(__gfx950__)
		    assert(ELEMS8_ELEMS4_RATIO == 2);
                    _B16x8 tmp_in;
                    for(int i = 0; i < 2; i++)
                    {
                        const int offset = rowid * VTLANELOOP * ELEMS8_ELEMS4_RATIO +
                            vfetch_depth * ELEMS8_ELEMS4_RATIO + i;
                        const int offset1 = offset % ROWS_PER_WARP;
                        const int offset2 = offset / ROWS_PER_WARP;
                        tmp_in.xy[i] = shared_logits[vtoken_depth][offset2][lane16id][offset1];
                    }
                    // output format is 16 qheads across 16 lanes, 16 head elems spread
                    // across 4 rows
                    tmp_out = gcn_mfma16x16x32_instr<scalar_t, 0, 0, 0>(
                        Vlocal[vtoken_depth][vhe_depth][vfetch_depth],
                        tmp_in,
                        tmp_out);
#else
                    for(int i = 0; i < ELEMS8_ELEMS4_RATIO; i++)
                    {
                        const int offset = rowid * VTLANELOOP * ELEMS8_ELEMS4_RATIO +
                                           vfetch_depth * ELEMS8_ELEMS4_RATIO + i;
                        const int offset1 = offset % ROWS_PER_WARP;
                        const int offset2 = offset / ROWS_PER_WARP;
                        // output format is 16 qheads across 16 lanes, 16 head elems spread
                        // across 4 rows
                        tmp_out = gcn_mfma16x16x16_instr<scalar_t, 0, 0, 0>(
                            Vlocal[vtoken_depth][vhe_depth][vfetch_depth].xy[i],
                            shared_logits[vtoken_depth][offset2][lane16id][offset1],
                            tmp_out);
                    }
#endif
                }
                // KV cache fp8
            }
            else
            {
                for(int vfetch_depth = 0; vfetch_depth < VTLANELOOP; vfetch_depth++)
                {
                    _B16x8 Vtmp = Vlocal[vtoken_depth][vhe_depth][vfetch_depth];
                    // reinterpret V format as 16 elements of 8bits
                    _B8x16 Vtmp8x16 = *reinterpret_cast<_B8x16*>(&Vtmp);
                    for(int j = 0; j < ELEMS16_ELEMS8_RATIO; j++)
                    {
                        _B8x8 Vtmp8x8    = Vtmp8x16.xy[j];
                        _B16x8 Vlocaltmp = convert_b8x8_custom<scalar_t>(Vtmp8x8);
#if defined(__gfx950__)
                        assert(ELEMS8_ELEMS4_RATIO == 2);
                        _B16x8 tmp_in;
                        for(int i = 0; i < 2; i++)
                        {
                            const int offset = rowid * ELEMS16_ELEMS8_RATIO * ELEMS8_ELEMS4_RATIO +
                                               j * ELEMS8_ELEMS4_RATIO + i;
                            const int offset1 = offset % ROWS_PER_WARP;
                            const int offset2 = offset / ROWS_PER_WARP;
			    tmp_in.xy[i] = shared_logits[vtoken_depth][offset2][lane16id][offset1];
                        }
                        // output format is 16 qheads across 16 lanes, 16 head elems
                        // spread across 4 rows
                        tmp_out = gcn_mfma16x16x32_instr<scalar_t, 0, 0, 0>(
                            Vlocaltmp,
                            tmp_in,
                            tmp_out);
#else
                        for(int i = 0; i < ELEMS8_ELEMS4_RATIO; i++)
                        {
                            const int offset = rowid * ELEMS16_ELEMS8_RATIO * ELEMS8_ELEMS4_RATIO +
                                               j * ELEMS8_ELEMS4_RATIO + i;
                            const int offset1 = offset % ROWS_PER_WARP;
                            const int offset2 = offset / ROWS_PER_WARP;
                            // output format is 16 qheads across 16 lanes, 16 head elems
                            // spread across 4 rows
                            tmp_out = gcn_mfma16x16x16_instr<scalar_t, 0, 0, 0>(
                                Vlocaltmp.xy[i],
                                shared_logits[vtoken_depth][offset2][lane16id][offset1],
                                tmp_out);
                        }
#endif
                    }
                }
            }
            __syncthreads();
        }
        // apply post Softmax V mfma v_scale
        if constexpr(KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto)
        {
            tmp_out *= *v_scale_ptr;
        }
        outelems[vhe_depth] = from_floatx4<scalar_t>(tmp_out);
    }

    __syncthreads();

    // store Softmax-V mfma output to shared mem
    for(int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++)
    {
        // lane16 id head dimension; rowid head element dimension
        shared_logits[warpid][vhe_depth][lane16id][rowid] = outelems[vhe_depth];
    }

    __syncthreads();

    // write to tmp_out with coalesced writes after reading from shared mem
    if(warpid == 0)
    {
        _B16x8 vout[GQA_RATIO4];
        // each lane writes out 16Bytes of tmp_out along head elem dimension
        const int head_elem_idx = lane16id * 8;
        if(head_elem_idx < HEAD_SIZE)
        {
            for(int h = 0; h < GQA_RATIO4; h++)
            {
                const int local_head_idx = 4 * h + rowid;
                const int offset1        = (head_elem_idx / 16) % 4;
                const int offset2        = head_elem_idx / 16 / NWARPS;
                const int offset3        = (head_elem_idx / 4) % 4;
                for(int i = 0; i < 2; i++)
                {
                    vout[h].xy[i] = shared_logits[offset1][offset2][local_head_idx][offset3 + i];
                }
            }

            const int hsz_maxp_mult = HEAD_SIZE * max_num_partitions;
            scalar_t* out_ptr =
                out + seq_idx * total_num_heads * hsz_maxp_mult + partition_idx * HEAD_SIZE;
            for(int h = 0; h < GQA_RATIO4; h++)
            {
                const int local_head_idx = 4 * h + rowid;
                if(local_head_idx < GQA_RATIO)
                {
                    const int out_head_idx = wg_start_head_idx + local_head_idx;
                    scalar_t* out_ptr2     = out_ptr + out_head_idx * hsz_maxp_mult;
                    scalar_t* out_ptr3     = out_ptr2 + head_elem_idx;
                    _B16x8* out_ptr_B16x8  = reinterpret_cast<_B16x8*>(out_ptr3);
                    *out_ptr_B16x8         = vout[h];
                }
            }
        }
    }
}

template <typename scalar_t,
          typename OUTT,
          int HEAD_SIZE,
          int NUM_THREADS,
          int PARTITION_SIZE,
          int NPAR_LOOPS>
__device__ void _paged_attention_ll4mi_reduce_kernel(
    const int64_t query_loc,
    int context_len,
    OUTT* __restrict__ out,                    // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums,        // [num_seqs, num_heads,
                                               // max_num_partitions]
    const float* __restrict__ max_logits,      // [num_seqs, num_heads,
                                               // max_num_partitions]
    const scalar_t* __restrict__ tmp_out,      // [num_seqs, num_heads,
                                               // max_num_partitions, head_size]
    const int max_num_partitions,
    const float* __restrict__ fp8_out_scale_ptr
){
    const int num_heads = gridDim.x;
    const int head_idx  = blockIdx.x;
    const int seq_idx   = blockIdx.y;

    const int num_partitions = DIVIDE_ROUND_UP(context_len, PARTITION_SIZE);
    constexpr int NUM_WARPS  = NUM_THREADS / WARP_SIZE;
    const int warpid         = threadIdx.x / WARP_SIZE;
    const int laneid         = threadIdx.x % WARP_SIZE;

    __shared__ float shared_global_exp_sum;
    // max num partitions supported is warp_size * NPAR_LOOPS
    __shared__ float shared_exp_sums[NPAR_LOOPS * WARP_SIZE];

    if(warpid == 0)
    {
        const float* max_logits_ptr =
            max_logits + seq_idx * num_heads * max_num_partitions + head_idx * max_num_partitions;

        // valid partition is the last valid partition in case threadid > num
        // partitions
        int valid_partition[NPAR_LOOPS];
        float reg_max_logit[NPAR_LOOPS];
        const int last_valid_partition = num_partitions - 1;

#pragma unroll
        for(int i = 0; i < NPAR_LOOPS; i++)
        {
            const int partition_no = i * WARP_SIZE + threadIdx.x;
            valid_partition[i] =
                (partition_no < num_partitions) ? partition_no : last_valid_partition;
        }
#pragma unroll
        for(int i = 0; i < NPAR_LOOPS; i++)
        {
            reg_max_logit[i] = max_logits_ptr[valid_partition[i]];
        }
        float max_logit = reg_max_logit[0];
#pragma unroll
        for(int i = 1; i < NPAR_LOOPS; i++)
        {
            max_logit = fmaxf(max_logit, reg_max_logit[i]);
        }

#pragma unroll
        for(int mask = WARP_SIZE / 2; mask >= 1; mask /= 2)
        {
            max_logit = fmaxf(max_logit, __shfl_xor(max_logit, mask));
        }

        const float* exp_sums_ptr =
            exp_sums + seq_idx * num_heads * max_num_partitions + head_idx * max_num_partitions;

        float rescaled_exp_sum[NPAR_LOOPS];
#pragma unroll
        for(int i = 0; i < NPAR_LOOPS; i++)
        {
            rescaled_exp_sum[i] = exp_sums_ptr[valid_partition[i]];
        }
#pragma unroll
        for(int i = 0; i < NPAR_LOOPS; i++)
        {
            const int partition_no = i * WARP_SIZE + threadIdx.x;
            rescaled_exp_sum[i] *=
                (partition_no < num_partitions) ? expf(reg_max_logit[i] - max_logit) : 0.0f;
        }
        float global_exp_sum = rescaled_exp_sum[0];
#pragma unroll
        for(int i = 1; i < NPAR_LOOPS; i++)
        {
            global_exp_sum += rescaled_exp_sum[i];
        }
#pragma unroll
        for(int i = 0; i < NPAR_LOOPS; i++)
        {
            const int partition_no        = i * WARP_SIZE + threadIdx.x;
            shared_exp_sums[partition_no] = rescaled_exp_sum[i];
        }

#pragma unroll
        for(int mask = WARP_SIZE / 2; mask >= 1; mask /= 2)
        {
            global_exp_sum += __shfl_xor(global_exp_sum, mask);
        }
        if(threadIdx.x == 0)
        {
            shared_global_exp_sum = global_exp_sum;
        }
    } // warpid == 0
    const scalar_t* tmp_out_ptr = tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
                                  head_idx * max_num_partitions * HEAD_SIZE + threadIdx.x;
    constexpr int MAX_NPAR = 64;
    scalar_t tmps[MAX_NPAR];
    const float dzero = 0.0f;
#pragma unroll
    for(int j = 0; j < MAX_NPAR; j++)
    {
        tmps[j] = from_float<scalar_t>(dzero);
    }
    const int last_partition_offset = (num_partitions - 1) * HEAD_SIZE;
    const int num_partition_offset  = (num_partitions)*HEAD_SIZE;
    int idx                         = 0;

    constexpr int JCHUNK = 16;

#pragma unroll
    for(int j = 0; j < JCHUNK * HEAD_SIZE; j += HEAD_SIZE)
    {
        // lastj is last valid partition
        const int lastj_offset = (j < num_partition_offset) ? j : last_partition_offset;
        tmps[idx]              = tmp_out_ptr[lastj_offset];
        idx++;
    }
    __syncthreads();

    if(num_partitions > JCHUNK)
    {
#pragma unroll
        for(int j = JCHUNK * HEAD_SIZE; j < 2 * JCHUNK * HEAD_SIZE; j += HEAD_SIZE)
        {
            const int lastj_offset = (j < num_partition_offset) ? j : last_partition_offset;
            tmps[idx]              = tmp_out_ptr[lastj_offset];
            idx++;
        }

        if(num_partitions > 2 * JCHUNK)
        {
#pragma unroll
            for(int j = 2 * JCHUNK * HEAD_SIZE; j < MAX_NPAR * HEAD_SIZE; j += HEAD_SIZE)
            {
                const int lastj_offset = (j < num_partition_offset) ? j : last_partition_offset;
                tmps[idx]              = tmp_out_ptr[lastj_offset];
                idx++;
            }
        }
    } // num_partitions > JCHUNK

    // Aggregate tmp_out to out.
    float acc = 0.0f;
#pragma unroll
    for(int j = 0; j < JCHUNK; j++)
    {
        acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j];
    }
    if(num_partitions > JCHUNK)
    {
#pragma unroll
        for(int j = JCHUNK; j < 2 * JCHUNK; j++)
        {
            acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j];
        }
        if(num_partitions > 2 * JCHUNK)
        {
#pragma unroll
            for(int j = 2 * JCHUNK; j < MAX_NPAR; j++)
            {
                acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j];
            }
        }
    }

    for(int p = 1; p < NPAR_LOOPS; p++)
    {
        if(num_partitions > p * MAX_NPAR)
        {
            idx = 0;
#pragma unroll
            for(int j = p * MAX_NPAR * HEAD_SIZE; j < (p + 1) * MAX_NPAR * HEAD_SIZE;
                j += HEAD_SIZE)
            {
                // lastj is last valid partition
                const int lastj_offset = (j < num_partition_offset) ? j : last_partition_offset;
                tmps[idx]              = tmp_out_ptr[lastj_offset];
                idx++;
            }

#pragma unroll
            for(int j = 0; j < MAX_NPAR; j++)
            {
                acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j + p * MAX_NPAR];
            }
        }
    }

    const float inv_global_exp_sum = __fdividef(1.0f, shared_global_exp_sum + 1e-6f);
    const float out_scale = (fp8_out_scale_ptr != nullptr) ? 1.0f / (*fp8_out_scale_ptr) : 1.0f;
    acc *= inv_global_exp_sum;
    acc *= out_scale;
    OUTT* out_ptr = out + query_loc * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
    if constexpr(std::is_same<OUTT, bit8_t>::value)
    {
        out_ptr[threadIdx.x] = hip_fp8(acc).data;
    }
    else
    {
        out_ptr[threadIdx.x] = from_float<scalar_t>(acc);
    }
}
