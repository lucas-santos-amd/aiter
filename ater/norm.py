# Copyright (c) 2024 Advanced Micro Devices, Inc.  All rights reserved.
#
# -*- coding:utf-8 -*-
# @Script: norm.py
# @Author: valarLip
# @Email: lingpeng.jin@amd.com
# @Create At: 2024-11-29 16:30:04
# @Last Modified By: valarLip
# @Last Modified At: 2024-12-03 18:43:04
# @Description: This is description.

from torch import Tensor
from typing import List, Optional
from .jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR
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
