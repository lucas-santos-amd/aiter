from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR, ATER_ROOT_DIR
import torch.nn.functional as F

MD_NAME = "module_pos_encoding"

compile_ops_ = {
    "srcs": [
        f"{ATER_CSRC_DIR}/pybind/pos_encoding_pybind.cu",
        f"{ATER_CSRC_DIR}/kernels/pos_encoding_kernels.cu",
    ],
    "md_name": MD_NAME,
}


@compile_ops(**compile_ops_)
def rotary_embedding(
    positions: Tensor,
    query: Tensor,
    key: Tensor,
    head_size: int,
    cos_sin_cache: Tensor,
    is_neox: bool,
): ...


@compile_ops(**compile_ops_)
def batched_rotary_embedding(
    positions: Tensor,
    query: Tensor,
    key: Tensor,
    head_size: int,
    cos_sin_cache: Tensor,
    is_neox: bool,
    rot_dim: int,
    cos_sin_cache_offsets: Tensor,
): ...
