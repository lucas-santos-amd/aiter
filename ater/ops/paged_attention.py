from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR
import torch.nn.functional as F

MD_NAME = "module_pa"


@compile_ops(
    srcs=[
        f"{ATER_CSRC_DIR}/pybind/attention_pybind.cu",
        f"{ATER_CSRC_DIR}/kernels/attention.cu",
    ],
    md_name=MD_NAME,
)
def paged_attention_rocm(
    out: Tensor,
    exp_sums: Tensor,
    block_mapping: Tensor,
    max_logits: Tensor,
    tmp_out: Tensor,
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: Tensor,
    context_lens: Tensor,
    block_size: int,
    max_context_len: int,
    alibi_slopes: Optional[Tensor],
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
    fp8_out_scale: Optional[Tensor],
    partition_size: int,
): ...
