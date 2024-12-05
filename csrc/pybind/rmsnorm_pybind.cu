#include "rmsnorm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("rms_norm", &rms_norm, "Apply Root Mean Square (RMS) Normalization to the input tensor.");
    m.def("fused_add_rms_norm", &fused_add_rms_norm, "In-place fused Add and RMS Normalization");
}