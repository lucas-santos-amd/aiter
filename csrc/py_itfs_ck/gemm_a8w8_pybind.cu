#include <torch/extension.h>

torch::Tensor gemm_a8w8(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gemm_a8w8", &gemm_a8w8, "gemm_a8w8", py::arg("XQ"), py::arg("WQ"),
          py::arg("x_scale"), py::arg("w_scale"), py::arg("Out"), py::arg("bias") = std::nullopt);
}
