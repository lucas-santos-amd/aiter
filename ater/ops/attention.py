# Copyright (c) 2024 Advanced Micro Devices, Inc.  All rights reserved.
#
# -*- coding:utf-8 -*-
# @Script: attention.py
# @Author: valarLip
# @Email: lingpeng.jin@amd.com
# @Create At: 2024-12-04 21:35:47
# @Last Modified By: valarLip
# @Last Modified At: 2024-12-10 14:43:14
# @Description: This is description.

import os
import torch
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR, ATER_ROOT_DIR
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


@compile_ops(srcs=[f'{ATER_CSRC_DIR}/py_itfs_cu/asm_pa.cpp',
                   f'{ATER_CSRC_DIR}/pybind/attention_asm_pybind.cu'],
             flags_extra_hip=[f'-DATER_ASM_DIR=\\"{ATER_ROOT_DIR}/hsa/\\"'],
             md_name=f"{MD_NAME}_asm")
def pa_fwd_asm(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor) -> torch.Tensor: ...


MD_NAME = "module_pa"


@compile_ops(
    srcs=[
        f"{ATER_CSRC_DIR}/pybind/attention_pybind.cu",
        f"{ATER_CSRC_DIR}/kernels/attention.cu",
    ],
    flags_extra_hip=['-DENABLE_FP8'],
    md_name=MD_NAME,
)
def paged_attention_rocm(
    out: torch.Tensor,
    exp_sums: torch.Tensor,
    block_mapping: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    max_context_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
    fp8_out_scale: Optional[torch.Tensor],
    partition_size: int,
): ...
