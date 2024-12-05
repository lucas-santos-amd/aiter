#pragma once

#include <torch/extension.h>

void wvSpltK(at::Tensor &in_a, at::Tensor &in_b, at::Tensor &out_c,
             const int64_t N_in, const int64_t CuCount);

void LLMM1(
    at::Tensor &in_a, at::Tensor &in_b, at::Tensor &out_c,
    const int64_t rows_per_block);
