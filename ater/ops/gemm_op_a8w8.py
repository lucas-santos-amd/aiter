import os
import torch
from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR
MD_NAME = 'module_gemm_a8w8'

@compile_ops(srcs=[f'{ATER_CSRC_DIR}/py_itfs_ck/gemm_a8w8_pybind.cu',
                   f'{ATER_CSRC_DIR}/ck_gemm_a8w8',
                   f'{ATER_CSRC_DIR}/ck_gemm_a8w8/instances',
                   f'{ATER_CSRC_DIR}/ck_gemm_a8w8/impl'],
             verbose=int(os.environ.get('ATER_LOG_MORE', 0))>0,
             md_name=MD_NAME,
             fc_name='gemm_a8w8',)
def gemm_a8w8(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
    bias : Optional[Tensor] = None,):...

def gemm_a8w8_bias(
        XQ: Tensor,
        WQ: Tensor,
        x_scale: Tensor,
        w_scale: Tensor,
        bias : Optional[Tensor] = None,
        dtype=torch.bfloat16):
    assert dtype in [torch.bfloat16, torch.float16], \
        f"Output {dtype=} is currently not supported in gemm_a8w8"
    m = XQ.shape[0]
    n = WQ.shape[0]
    Y = torch.empty(m, n, dtype=dtype, device="cuda")
    gemm_a8w8(XQ, WQ, x_scale, w_scale, Y, bias)
    return Y