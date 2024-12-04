from test_common import checkAllclose, perftest
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import ater


@perftest()
def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    x = F.linear(x.to(torch.float32), weight.to(torch.float32))
    scale = torch.matmul(x_scale, w_scale)
    out = torch.mul(x, scale)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)


@perftest()
def run_gemm_b(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    return ater.gemm_a8w8_bias(x, weight, x_scale, w_scale, bias, dtype = dtype)


def test_gemm(dtype, m, n, k):
    dim = (m, n, k)
    x = torch.randint(-20, 20, (m, k), dtype=torch.int8).cuda()
    weight = torch.randint(-20, 20, (n, k), dtype=torch.int8).cuda()
    x_scale = torch.rand([m, 1], dtype=torch.float32).cuda()
    w_scale = torch.rand([1, n], dtype=torch.float32).cuda()
    bias = torch.rand([1,n],dtype = dtype).cuda()
    (a, *_), avg_a = run_torch(x, weight, x_scale, w_scale, bias, dtype)
    (b, *_), avg_b = run_gemm_b(x, weight, x_scale, w_scale, bias, dtype)

    msg = f"[perf] dim: {str(dim):<20} dtype: {dtype}, torch avg: {avg_a:<8.2f} us, B avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    checkAllclose(a, b, msg=msg, rtol=1e-3, atol=1000)


for dtype in [torch.bfloat16]:
    # qkv_proj
    for (m, n, k) in [(4096, 1280, 8192),
                      (128, 1280, 8192)
                      ]:
        test_gemm(dtype, m, n, k)
    # attn_out
    for (m, n, k) in [(4096, 8192, 1024),
                      (128, 8192, 1024)]:
        test_gemm(dtype, m, n, k)
