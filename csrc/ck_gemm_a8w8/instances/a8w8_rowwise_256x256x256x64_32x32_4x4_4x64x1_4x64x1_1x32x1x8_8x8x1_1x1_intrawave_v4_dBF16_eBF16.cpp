// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4.cuh"

template torch::Tensor
a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<B16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias);
