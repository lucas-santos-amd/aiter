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
#include <torch/torch.h>

#ifdef USE_ROCM
#include <hip/hip_bf16.h>
typedef __hip_bfloat16 nv_bfloat16;
#else
#include <cuda_bf16.h>
#endif
#include <cuda_fp16.h>

namespace vllm
{
#define BIG_TILE_SIZE 64

  template <class _T, int _WG>
  __global__ void add_tn_big_tile_kernel(const void *__restrict a, const void *__restrict b, void *__restrict c, const int N, const int K, int stride0, int stride2)
  {
    // pad LDS row by dword
    constexpr uint32_t LDS_PAD = 4 / sizeof(_T);
    constexpr uint32_t element_size = sizeof(_T); // in bytes
    constexpr uint32_t elements_in_16B = 16 / element_size;

    union BLOCK_16B
    {
      _T e[elements_in_16B];
      __uint128_t ow;
    };

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
        uint64_t offset = (col < max_part_n && row < max_part_k) ? row * stride2 + col * sizeof(_T) : 0;
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
          uint64_t offset = row * stride_k + col * sizeof(_T);
          const _T *pfb = (const _T *)(pb + offset);
          _T *pfc = (_T *)(pc + offset);
          *pfc = sa[col][row] + *pfb;
        }
      }
    }
  }
}

torch::Tensor transpose_add(torch::Tensor &input, torch::Tensor &other)
{
  int dim = input.dim();
  bool is_support = input.is_contiguous() != other.is_contiguous();
  is_support &= input.dim() == other.dim();
  is_support &= dim == 3;
  if (dim == 3)
  {
    is_support &= input.size(0) == other.size(0);
    is_support &= input.size(1) == other.size(1);
    is_support &= input.size(2) == other.size(2);
  }

  if (is_support)
  {
    int M = input.size(0);
    int N = input.size(1);
    int K = input.size(2);

    int big_tile_wg = M * ((N + BIG_TILE_SIZE - 1) / BIG_TILE_SIZE) * ((K + BIG_TILE_SIZE - 1) / BIG_TILE_SIZE);
    const dim3 grid_dim(big_tile_wg, 1, 1);
    const dim3 block_dim(256, 1, 1);

    auto options =
        torch::TensorOptions().dtype(input.dtype()).device("cuda");
    auto output =
        torch::empty(input.sizes(), options);
    void *buf_c = reinterpret_cast<void *>(output.data_ptr());

    int stride0 = 1;
    int stride2 = 1;
    void *buf_a = reinterpret_cast<void *>(input.data_ptr());
    void *buf_b = reinterpret_cast<void *>(other.data_ptr());

    if (!input.is_contiguous())
    {
      stride0 *= input.stride(0);
      stride2 *= input.stride(2);
    }
    else if (!other.is_contiguous())
    {
      stride0 *= other.stride(0);
      stride2 *= other.stride(2);
      void *buf_a = reinterpret_cast<void *>(other.data_ptr());
      void *buf_b = reinterpret_cast<void *>(input.data_ptr());
    }

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    switch (input.scalar_type())
    {
    case at::ScalarType::Float:
    {
      stride0 *= sizeof(float);
      stride2 *= sizeof(float);
      vllm::add_tn_big_tile_kernel<float, 256><<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_b, buf_c, N, K, stride0, stride2);
      break;
    }
    case at::ScalarType::Half:
    {
      stride0 *= sizeof(half);
      stride2 *= sizeof(half);
      vllm::add_tn_big_tile_kernel<half, 256><<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_b, buf_c, N, K, stride0, stride2);
      break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16:
    {
      stride0 *= sizeof(nv_bfloat16);
      stride2 *= sizeof(nv_bfloat16);
      vllm::add_tn_big_tile_kernel<nv_bfloat16, 256><<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_b, buf_c, N, K, stride0, stride2);
      break;
    }
#endif
    default:
      output = at::add(input, other);
    }
    return output;
  }
  else
  {
    return at::add(input, other);
  }
}
