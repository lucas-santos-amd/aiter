#include "transpose_operator.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("transpose_add", &transpose_add, "apply for add with transpose.");
    m.def("transpose_mul", &transpose_mul, "apply for mul with transpose.");
    m.def("transpose_sub", &transpose_sub, "apply for sub with transpose.");
    m.def("transpose_div", &transpose_div, "apply for div with transpose.");
}