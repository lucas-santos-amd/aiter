import torch
from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR
import torch.nn.functional as F

MD_NAME = "module_smoothquant"

compile_ops_ = {
    "srcs": [
        f"{ATER_CSRC_DIR}/py_itfs_ck/smoothquant_kernels.cu",
        f"{ATER_CSRC_DIR}/pybind/smoothquant_pybind.cu",
        f"{CK_DIR}/example/ck_tile/12_smoothquant/instances",
        f"{CK_DIR}/example/ck_tile/14_moe_smoothquant/instances",
    ],
    "extra_include": [
        f"{CK_DIR}/example/ck_tile/12_smoothquant",
        f"{CK_DIR}/example/ck_tile/14_moe_smoothquant",
    ],
    "md_name": MD_NAME,
}


@compile_ops(**compile_ops_)
def smoothquant_fwd(input: Tensor, out: Tensor,
                    x_scale: Tensor, y_scale: Tensor): ...


@compile_ops(**compile_ops_)
def moe_smoothquant_fwd(
    input: Tensor, out: Tensor, x_scale: Tensor, topk_ids: Tensor, y_scale: Tensor
): ...
