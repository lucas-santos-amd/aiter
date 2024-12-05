# Copyright (c) 2024 Advanced Micro Devices, Inc.  All rights reserved.
#
# -*- coding:utf-8 -*-
# @Script: attention.py
# @Author: valarLip
# @Email: lingpeng.jin@amd.com
# @Create At: 2024-12-04 21:35:47
# @Last Modified By: valarLip
# @Last Modified At: 2024-12-04 21:43:26
# @Description: This is description.

import os
import torch
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR
MD_NAME = 'module_attention'


@compile_ops(srcs=[f'{ATER_CSRC_DIR}/py_itfs_ck/attention_kernels.cu',
                   f'{ATER_CSRC_DIR}/pybind/attention_ck_pybind.cu'],
             md_name=MD_NAME)
def pa_fwd_naive(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    kv_cache_dtype: str,
    num_kv_heads: int,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
    k_scale: float,
    v_scale: float) -> torch.Tensor: ...
