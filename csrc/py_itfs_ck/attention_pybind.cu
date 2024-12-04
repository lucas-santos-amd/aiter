/*
 * Copyright (c) 2024 Advanced Micro Devices, Inc.  All rights reserved.
 *
 * @Script: attention_pybind.cu
 * @Author: valarLip
 * @Email: lingpeng.jin@amd.com
 * @Create At: 2024-12-04 21:32:40
 * @Last Modified By: valarLip
 * @Last Modified At: 2024-12-04 21:50:56
 * @Description: This is description.
 */

#include <torch/extension.h>

torch::Tensor pa_fwd_naive(torch::Tensor &Q, //   [num_seqs, num_heads, head_size]
                           torch::Tensor &K, //   [num_blocks, num_kv_heads, head_size/x, block_size, x]
                                             // or[num_batch, seqlen, num_kv_heads, head_size]
                           torch::Tensor &V, //   [num_blocks, num_kv_heads, head_size, block_size]
                                             // or[num_batch*seqlen, num_kv_heads, head_size]
                           torch::Tensor &block_tables,
                           torch::Tensor &context_lens,
                           const int max_seq_len,
                           const int num_kv_heads,
                           const float scale_s,
                           const float scale_k,
                           const float scale_v,
                           const int block_size
                           // above are input
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("pa_fwd_naive", &pa_fwd_naive, "pa_fwd_naive");
}
