/*
 * Adapted from https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.1/cpp/tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.cu
 * Copyright (c) 2024, The vLLM team.
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "hip_compat.h"
#include "dispatch_utils.h"

#ifndef USE_ROCM
#include <cub/util_type.cuh>
#include <cub/cub.cuh>
#else
#include <hipcub/util_type.hpp>
#include <hipcub/hipcub.hpp>
#endif

namespace vllm
{
#define EL_TYPE c10::Half
#define BIG_TILE_SIZE 64
// pad LDS row by dword
#define LDS_PAD (4 / sizeof(EL_TYPE))
  constexpr uint32_t element_size = sizeof(EL_TYPE); // in bytes
  constexpr uint32_t elements_in_16B = 16 / element_size;

  union BLOCK_16B
  {
    EL_TYPE e[elements_in_16B];
    __uint128_t ow;
  };

  template <class _T, int _WG>
  __global__ void add_tn_big_tile_kernel(const void *__restrict a, const void *__restrict b, void *__restrict c, const int N, const int K, int stride0, int stride2)
  {
    // Round up processing to next full tile
    const uint32_t n_tiles = (N + BIG_TILE_SIZE - 1) / BIG_TILE_SIZE;
    const uint32_t k_tiles = (K + BIG_TILE_SIZE - 1) / BIG_TILE_SIZE;
    const uint32_t nk_tiles = n_tiles * k_tiles;
    const uint32_t m = blockIdx.x / nk_tiles;
    const uint64_t stride_n = N * sizeof(_T);
    const uint64_t stride_k = K * sizeof(_T);
    const uint64_t stride_nk = N * K * sizeof(_T);

    // Walk destination tiles continuously for cache coherency
    constexpr uint32_t XCD = 8;
    constexpr uint32_t SEQ = 8;
    constexpr uint32_t sblk = XCD * SEQ;
    const uint32_t max_swizzle = (nk_tiles / sblk) * sblk;
    uint32_t tIdx = blockIdx.x % nk_tiles;
    tIdx = tIdx > max_swizzle ? tIdx : (tIdx / sblk) * sblk + (tIdx % sblk) / SEQ + (tIdx % SEQ) * XCD;
    uint32_t ti = tIdx / k_tiles;
    uint32_t tj = tIdx % k_tiles;

    __shared__ _T sa[BIG_TILE_SIZE][BIG_TILE_SIZE + LDS_PAD];

    // Detect partial tiles
    uint32_t max_part_n = (ti == (n_tiles - 1) && (N % BIG_TILE_SIZE) != 0) ? (N % BIG_TILE_SIZE) : BIG_TILE_SIZE;
    uint32_t max_part_k = (tj == (k_tiles - 1) && (K % BIG_TILE_SIZE) != 0) ? (K % BIG_TILE_SIZE) : BIG_TILE_SIZE;

    if (max_part_n == BIG_TILE_SIZE && max_part_k == BIG_TILE_SIZE)
    {
      // Copy full tile with large loads
      constexpr uint32_t row_bytes = BIG_TILE_SIZE * sizeof(_T);
      constexpr uint32_t vmem_per_row = row_bytes / sizeof(__uint128_t);
      constexpr uint32_t rows_per_wg = _WG / vmem_per_row;
      constexpr uint32_t vmem_per_thread = BIG_TILE_SIZE / rows_per_wg;
      // Make sure WG isn't too large
      static_assert(vmem_per_thread >= 1);

      const uint8_t *pat = (const uint8_t *)a + tj * BIG_TILE_SIZE * stride2 + ti * row_bytes + m * stride0;
#pragma unroll
      for (uint32_t t = 0; t < vmem_per_thread; t++)
      {
        uint32_t col = threadIdx.x % vmem_per_row;
        uint32_t row = threadIdx.x / vmem_per_row + t * rows_per_wg;
        uint64_t offset = row * stride2 + col * sizeof(__uint128_t);
        const __uint128_t *pfa = (const __uint128_t *)(pat + offset);
        BLOCK_16B d;
        d.ow = *pfa;
#pragma unroll
        for (uint32_t i = 0; i < elements_in_16B; i++)
        {
          sa[row][col * elements_in_16B + i] = d.e[i];
        }
      }
      __syncthreads();

      const uint8_t *pb = (const uint8_t *)b + ti * BIG_TILE_SIZE * stride_k + tj * row_bytes + m * stride_nk;
      const uint8_t *pc = (const uint8_t *)c + ti * BIG_TILE_SIZE * stride_k + tj * row_bytes + m * stride_nk;
#pragma unroll
      for (uint32_t t = 0; t < vmem_per_thread; t++)
      {
        uint32_t col = threadIdx.x % vmem_per_row;
        uint32_t row = threadIdx.x / vmem_per_row + t * rows_per_wg;
        uint64_t offset = row * stride_k + col * sizeof(__uint128_t);
        BLOCK_16B d;
        const __uint128_t *pfb = (const __uint128_t *)(pb + offset);
        d.ow = *pfb;
// Transpose tile on read from LDS
#pragma unroll
        for (uint32_t i = 0; i < elements_in_16B; i++)
        {
          d.e[i] += sa[col * elements_in_16B + i][row];
        }
        __uint128_t *pfc = (__uint128_t *)(pc + offset);
        *pfc = d.ow;
      }
    }
    else
    {
      // Copy partial tiles with element accesses
      constexpr uint32_t row_bytes = BIG_TILE_SIZE * sizeof(_T);
      constexpr uint32_t vmem_per_row = BIG_TILE_SIZE;
      constexpr uint32_t rows_per_wg = _WG / vmem_per_row;
      constexpr uint32_t vmem_per_thread = BIG_TILE_SIZE / rows_per_wg;
      // Make sure WG isn't too large
      static_assert(vmem_per_thread >= 1);

      const uint8_t *pat = (const uint8_t *)a + tj * BIG_TILE_SIZE * stride2 + ti * row_bytes + m * stride0;
#pragma unroll
      for (uint32_t t = 0; t < vmem_per_thread; t++)
      {
        uint32_t col = threadIdx.x % vmem_per_row;
        uint32_t row = threadIdx.x / vmem_per_row + t * rows_per_wg;
        uint64_t offset = (col < max_part_n && row < max_part_k) ? row * stride2 + col * 2 : 0;
        const _T *pfa = (const _T *)(pat + offset);
        sa[row][col] = *pfa;
      }
      __syncthreads();

      const uint8_t *pb = (const uint8_t *)b + ti * BIG_TILE_SIZE * stride_k + tj * row_bytes + m * stride_nk;
      const uint8_t *pc = (const uint8_t *)c + ti * BIG_TILE_SIZE * stride_k + tj * row_bytes + m * stride_nk;
#pragma unroll
      for (uint32_t t = 0; t < vmem_per_thread; t++)
      {
        uint32_t col = threadIdx.x % vmem_per_row;
        uint32_t row = threadIdx.x / vmem_per_row + t * rows_per_wg;
        if (col < max_part_k && row < max_part_n)
        {
          uint64_t offset = row * stride_k + col * 2;
          const _T *pfb = (const _T *)(pb + offset);
          _T *pfc = (_T *)(pc + offset);
          *pfc = sa[col][row] + *pfb;
        }
      }
    }
  }
}

void transpose_add(
    torch::Tensor &output,
    torch::Tensor &input0,
    torch::Tensor &input1)
{
  int M = input0.size(0);
  int N = input0.size(1);
  int K = input0.size(2);

  int big_tile_wg = M * ((N + BIG_TILE_SIZE - 1) / BIG_TILE_SIZE) * ((K + BIG_TILE_SIZE - 1) / BIG_TILE_SIZE);
  const dim3 grid_dim(big_tile_wg, 1, 1);
  const dim3 block_dim(256, 1, 1);
  EL_TYPE *buf_a = reinterpret_cast<EL_TYPE *>(input0.data_ptr());
  EL_TYPE *buf_b = reinterpret_cast<EL_TYPE *>(input1.data_ptr());
  EL_TYPE *buf_c = reinterpret_cast<EL_TYPE *>(output.data_ptr());

  int stride0 = sizeof(EL_TYPE);
  int stride2 = sizeof(EL_TYPE);

  bool is_conti = input0.is_contiguous() != input1.is_contiguous();
  bool is_conti_0 = is_conti && !input0.is_contiguous();
  bool is_conti_1 = is_conti && !input1.is_contiguous();
  if (is_conti_0)
  {
    stride0 *= input0.stride(0);
    stride2 *= input0.stride(2);
  }
  else if (is_conti_1)
  {
    stride0 *= input1.stride(0);
    stride2 *= input1.stride(2);
    EL_TYPE *buf_a = reinterpret_cast<EL_TYPE *>(input1.data_ptr());
    EL_TYPE *buf_b = reinterpret_cast<EL_TYPE *>(input0.data_ptr());
  }

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  vllm::add_tn_big_tile_kernel<EL_TYPE, 256><<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_b, buf_c, N, K, stride0, stride2);
}
