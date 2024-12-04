import torch
import os
from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR
import torch.nn.functional as F
MD_NAME = 'module_smoothquant'


@compile_ops(srcs=[f'{ATER_CSRC_DIR}/py_itfs_ck/smoothquant_kernels.cu',
                   f'{ATER_CSRC_DIR}/py_itfs_ck/smoothquant_pybind.cu',
                   f'{CK_DIR}/example/ck_tile/12_smoothquant/instances',
                   f'{CK_DIR}/example/ck_tile/14_moe_smoothquant/instances'],
             extra_include=[f'{CK_DIR}/example/ck_tile/12_smoothquant',
                            f'{CK_DIR}/example/ck_tile/14_moe_smoothquant'],
             verbose=int(os.environ.get('ATER_LOG_MORE', 0)) > 0,
             md_name=MD_NAME,
             fc_name='smoothquant_fwd',)
def smoothquant_fwd(
    input: Tensor,
    out: Tensor,
    x_scale: Tensor,
    y_scale: Tensor): ...


def smoothquant_fwd_native(hidden_states_input, y_scale_dtype, x_scale=None):
    # assume output int8, hidden_states is [m, n] shape and in fp16/bf16
    if x_scale is None:
        hidden_states = hidden_states_input
    else:
        # smooth quant
        hidden_states = hidden_states_input.to(x_scale) * x_scale
    # [m, 1]
    per_token_amax, _ = torch.max(
        input=torch.abs(hidden_states),
        dim=-1,
        keepdim=True
    )
    per_token_scale = per_token_amax.to(dtype=torch.float32) / 127.0

    # quant hidden_states
    hidden_states = (hidden_states / per_token_scale).to(dtype=torch.int8)

    return hidden_states, per_token_scale.to(y_scale_dtype)


@compile_ops(srcs=[f'{ATER_CSRC_DIR}/py_itfs_ck/smoothquant_kernels.cu',
                   f'{ATER_CSRC_DIR}/py_itfs_ck/smoothquant_pybind.cu',
                   f'{CK_DIR}/example/ck_tile/12_smoothquant/instances',
                   f'{CK_DIR}/example/ck_tile/14_moe_smoothquant/instances'],
             extra_include=[f'{CK_DIR}/example/ck_tile/12_smoothquant',
                            f'{CK_DIR}/example/ck_tile/14_moe_smoothquant'],
             verbose=int(os.environ.get('ATER_LOG_MORE', 0)) > 0,
             md_name=MD_NAME,
             fc_name='moe_smoothquant_fwd',)
def moe_smoothquant_fwd(
    input: Tensor,
    out: Tensor,
    x_scale: Tensor,
    topk_ids: Tensor,
    y_scale: Tensor): ...
