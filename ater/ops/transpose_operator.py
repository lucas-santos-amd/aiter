from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR
import torch.nn.functional as F

MD_NAME = "transpose_operator_module"

compile_ops_ = {
    "srcs": [
        f"{ATER_CSRC_DIR}/py_itfs_ck/transpose_operator_pybind.cu",
        f"{ATER_CSRC_DIR}/kernels/transpose_operator.cu",
    ],
    "md_name": MD_NAME,
}


@compile_ops(**compile_ops_)
def transpose_add(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops(**compile_ops_)
def transpose_sub(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops(**compile_ops_)
def transpose_mul(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops(**compile_ops_)
def transpose_div(input: Tensor, other: Tensor) -> Tensor: ...
