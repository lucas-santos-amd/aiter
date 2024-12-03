# Copyright (c) 2024 Advanced Micro Devices, Inc.  All rights reserved.
#
# -*- coding:utf-8 -*-
# @Script: norm.py
# @Author: valarLip
# @Email: lingpeng.jin@amd.com
# @Create At: 2024-11-29 16:30:04
# @Last Modified By: valarLip
# @Last Modified At: 2024-12-03 12:46:42
# @Description: This is description.

from torch import Tensor
from typing import List, Optional
from jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR
import torch.nn.functional as F
MD_NAME = 'norm_module'


@compile_ops(srcs=[f'{ATER_CSRC_DIR}/py_itfs_ck/norm_kernels.cu',
                   f'{ATER_CSRC_DIR}/py_itfs_ck/norm_pybind.cu'],
             extra_include=[f'{CK_DIR}/example/ck_tile/02_layernorm2d'],
             blob_gen_cmd=f'{CK_DIR}/example/ck_tile/02_layernorm2d/generate.py --api fwd --gen_blobs --working_path {{}}',
             #  verbose=True,
             md_name=MD_NAME,
             fc_name='layernorm2d_fwd',)
def layer_norm(
    input: Tensor,
    # normalized_shape: List[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5) -> Tensor: ...


if __name__ == "__main__":
    import torch
    dtype = torch.bfloat16
    m, n = 128, 1024
    dim = (m, n)
    eps = 1e-5
    input = torch.randn(dim, dtype=dtype, device="cuda")
    weight = torch.randn(n, dtype=dtype, device="cuda")
    bias = torch.randn(n, dtype=dtype, device="cuda")
    output = layer_norm(input,
                        weight,
                        bias,
                        eps)
    output2 = F.layer_norm(
        input=input,
        normalized_shape=(input.shape[-1],),
        weight=weight,
        bias=bias,
        eps=eps
    )
    print(f'{output=}')
    print(f'{output2=}')
    torch.allclose(output, output2)
