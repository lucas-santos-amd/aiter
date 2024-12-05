#pragma once
#include <torch/extension.h>

torch::Tensor transpose_add(torch::Tensor &input0, torch::Tensor &input1);
torch::Tensor transpose_mul(torch::Tensor &input0, torch::Tensor &input1);
torch::Tensor transpose_sub(torch::Tensor &input0, torch::Tensor &input1);
torch::Tensor transpose_div(torch::Tensor &input0, torch::Tensor &input1);