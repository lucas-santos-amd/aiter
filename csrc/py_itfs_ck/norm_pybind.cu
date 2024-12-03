/*
 * Copyright (c) 2024 Advanced Micro Devices, Inc.  All rights reserved.
 *
 * @Script: norm_pybind.cu
 * @Author: valarLip
 * @Email: lingpeng.jin@amd.com
 * @Create At: 2024-12-02 16:00:01
 * @Last Modified By: valarLip
 * @Last Modified At: 2024-12-02 16:07:05
 * @Description: This is description.
 */

#include <torch/extension.h>

// void layernorm2d(torch::Tensor &out, torch::Tensor &input, torch::Tensor &weight, torch::Tensor &bias, double epsilon);
torch::Tensor layernorm2d(torch::Tensor &input, torch::Tensor &weight, torch::Tensor &bias, double epsilon);
void layernorm2d_with_add(torch::Tensor &out, torch::Tensor &input, torch::Tensor &residual_in, torch::Tensor &residual_out, torch::Tensor &weight, torch::Tensor &bias, double epsilon);
void layernorm2d_with_smoothquant(torch::Tensor &out,    // [m ,n]
                                  torch::Tensor &input,  // [m ,n]
                                  torch::Tensor &xscale, // [1 ,n]
                                  torch::Tensor &yscale, // [m ,1]
                                  torch::Tensor &weight, // [1 ,n]
                                  torch::Tensor &bias,   // [1 ,n]
                                  double epsilon);
void layernorm2d_with_add_smoothquant(torch::Tensor &out,          // [m ,n]
                                      torch::Tensor &input,        // [m ,n]
                                      torch::Tensor &residual_in,  // [m ,n]
                                      torch::Tensor &residual_out, // [m ,n]
                                      torch::Tensor &xscale,       // [1 ,n]
                                      torch::Tensor &yscale,       // [m ,1]
                                      torch::Tensor &weight,       // [1 ,n]
                                      torch::Tensor &bias,         // [1 ,n]
                                      double epsilon);
void layernorm2d_with_dynamicquant(torch::Tensor &out,    // [m ,n]
                                   torch::Tensor &input,  // [m ,n]
                                   torch::Tensor &yscale, // [m ,1]
                                   torch::Tensor &weight, // [1 ,n]
                                   torch::Tensor &bias,   // [1 ,n]
                                   double epsilon);
void layernorm2d_with_add_dynamicquant(torch::Tensor &out,          // [m ,n]
                                       torch::Tensor &input,        // [m ,n]
                                       torch::Tensor &residual_in,  // [m ,n]
                                       torch::Tensor &residual_out, // [m ,n]
                                       torch::Tensor &yscale,       // [m ,1]
                                       torch::Tensor &weight,       // [1 ,n]
                                       torch::Tensor &bias,         // [1 ,n]
                                       double epsilon);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("layernorm2d_fwd", &layernorm2d);
    m.def("layernorm2d_fwd_with_add", &layernorm2d_with_add);
    m.def("layernorm2d_fwd_with_smoothquant", &layernorm2d_with_smoothquant);
    m.def("layernorm2d_fwd_with_add_smoothquant", &layernorm2d_with_add_smoothquant);
    m.def("layernorm2d_fwd_with_dynamicquant", &layernorm2d_with_dynamicquant);
    m.def("layernorm2d_fwd_with_add_dynamicquant", &layernorm2d_with_add_dynamicquant);
}