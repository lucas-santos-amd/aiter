#pragma once

#include <torch/extension.h>

void smoothquant_fwd(torch::Tensor &out,      // [m ,n]
                     torch::Tensor &input,    // [m ,n]
                     torch::Tensor &x_scale,  // [1 ,n]
                     torch::Tensor &y_scale); // [m ,1]

void moe_smoothquant_fwd(torch::Tensor &out,      // [topk * tokens, hidden_size]
                         torch::Tensor &input,    // [tokens, hidden_size]
                         torch::Tensor &x_scale,  // [experts, hidden_size]
                         torch::Tensor &topk_ids, // [tokens, topk]
                         torch::Tensor &y_scale); // [topk * tokens,  1]