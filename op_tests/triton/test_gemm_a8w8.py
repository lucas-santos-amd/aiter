import torch
import triton
import triton.language as tl
import pytest
from aiter.ops.triton.gemm_a8w8 import gemm_a8w8
import torch.nn.functional as F


def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    x = F.linear(x.to(torch.float32), weight.to(torch.float32))
    scale = torch.matmul(x_scale, w_scale)
    out = torch.mul(x, scale)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)


def run_triton(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    return gemm_a8w8(x, weight, x_scale, w_scale, bias)


def is_cdna4():
    return triton.runtime.driver.active.get_current_target().arch == "gfx950"


e5m2_type = torch.float8_e5m2 if is_cdna4() else torch.float8_e5m2fnuz
e4m3_type = torch.float8_e4m3fn if is_cdna4() else torch.float8_e4m3fnuz

name_to_torch_types = {
    "int8": torch.int8,
    "int32": torch.int32,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp8e5": e5m2_type,
    "fp8e4": e4m3_type,
}


def get_x_vals():

    x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in range(1, 9)]
    x_vals += [(4864, 4096, 8192), (9728, 8192, 65536), (4864, 8192, 4160)]

    return x_vals


@pytest.mark.parametrize(
    "dtype, m, n, k", [(dtype, *shape) for shape in get_x_vals() for dtype in ["bf16"]]
)
def test_gemm(dtype, m, n, k):
    dim = (m, n, k)
    dtype = name_to_torch_types[dtype]
    x = torch.randint(-20, 20, (m, k), dtype=torch.int8).cuda()
    weight = torch.randint(-20, 20, (n, k), dtype=torch.int8).cuda()
    x_scale = torch.rand([m, 1], dtype=torch.float32).cuda() + 1e-6
    w_scale = torch.rand([1, n], dtype=torch.float32).cuda() + 1e-6
    bias = torch.rand([1, n], dtype=dtype).cuda() * 10

    a = run_torch(x, weight, x_scale, w_scale, bias, dtype)
    b = run_triton(x, weight, x_scale, w_scale, bias, dtype)

    triton.testing.assert_close(a, b, atol=0.01, rtol=1e-2)
