# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the naive Gluon flash-attention forward kernel against the Triton one.

The Gluon forward kernel lives under
``aiter/ops/triton/_gluon_kernels/gfx950/attention/mha_gluon.py`` and is reached
through ``aiter.ops.triton.attention.mha.flash_attn_func`` via its ``backend``
switch (``"triton"`` -> the default Triton kernel, ``"gluon"`` ->
``mha_gluon.py::flash_attn_fwd``).

For each shape this script times both backends with ``triton.testing.do_bench``,
optionally checks the Gluon output against the Triton output, and prints latency,
achieved TFLOPS and the gluon/triton speedup. Only the forward pass (fixed-length
and varlen) is supported, mirroring what the Gluon kernel implements.

Examples:
    # Default model sweep (bf16, causal + non-causal), fixed-length forward:
    python op_tests/op_benchmarks/triton/bench_mha_gluon.py

    # A single custom shape:
    python op_tests/op_benchmarks/triton/bench_mha_gluon.py -b 2 -hq 32 -hk 8 -sq 4096 -sk 4096 -d 128 -causal True

    # Varlen forward, fp16, no correctness check:
    python op_tests/op_benchmarks/triton/bench_mha_gluon.py -fn fwd_varlen --dtype fp16 -no_check
"""

import argparse
import itertools
import sys

import torch
import triton

from aiter.ops.triton.attention.mha import (
    flash_attn_func,
    flash_attn_varlen_func,
    _is_gluon_available,
    _GLUON_SUPPORTED_HEAD_DIMS,
)


VALID_FUNCTIONS = ("fwd", "fwd_varlen")

_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _str2bool(v):
    if isinstance(v, bool) or v is None:
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def _count_valid_attention_elements(seqlen_q, seqlen_k, causal):
    """Number of (q, k) pairs that contribute to the output (for FLOPs)."""
    if not causal:
        return seqlen_q * seqlen_k
    shift = seqlen_k - seqlen_q
    total = 0
    for q_idx in range(seqlen_q):
        right = min(seqlen_k - 1, q_idx + shift)
        if right >= 0:
            total += right + 1
    return total


def _make_cu_seqlens(batch, seqlen, device, equal_seqlens):
    """Build cumulative seqlens (and the per-seq lengths) for a varlen batch."""
    if equal_seqlens:
        lengths = torch.full((batch,), seqlen, dtype=torch.int32, device=device)
    else:
        # Random lengths in [1, seqlen], reproducible via the global manual seed.
        lengths = torch.randint(
            1, seqlen + 1, (batch,), dtype=torch.int32, device=device
        )
    cu = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(lengths, dim=0)
    return cu, lengths


def _build_inputs(cfg, device):
    """Return (call_triton, call_gluon, total_flops) closures for one config.

    Each ``call_*`` closure runs the corresponding backend once and returns the
    output tensor, so the same closures drive both correctness and timing.
    """
    batch, hq, hk = cfg["batch"], cfg["hq"], cfg["hk"]
    sq, sk, d = cfg["sq"], cfg["sk"], cfg["d"]
    causal, dtype = cfg["causal"], cfg["dtype"]

    if cfg["function"] == "fwd":
        q = torch.randn((batch, sq, hq, d), device=device, dtype=dtype)
        k = torch.randn((batch, sk, hk, d), device=device, dtype=dtype)
        v = torch.randn((batch, sk, hk, d), device=device, dtype=dtype)

        def call_triton():
            return flash_attn_func(q, k, v, causal=causal, backend="triton")

        def call_gluon():
            return flash_attn_func(q, k, v, causal=causal, backend="gluon")

        flops = (
            2.0
            * 2.0
            * batch
            * hq
            * _count_valid_attention_elements(sq, sk, causal)
            * d
        )
        return call_triton, call_gluon, flops

    # Varlen: pack [total_tokens, heads, head_dim] without a batch dim.
    cu_q, len_q = _make_cu_seqlens(batch, sq, device, cfg["equal_seqlens"])
    cu_k, len_k = _make_cu_seqlens(batch, sk, device, cfg["equal_seqlens"])
    total_q = int(cu_q[-1].item())
    total_k = int(cu_k[-1].item())
    max_sq = int(len_q.max().item())
    max_sk = int(len_k.max().item())

    q = torch.randn((total_q, hq, d), device=device, dtype=dtype)
    k = torch.randn((total_k, hk, d), device=device, dtype=dtype)
    v = torch.randn((total_k, hk, d), device=device, dtype=dtype)

    def call_triton():
        return flash_attn_varlen_func(
            q, k, v, cu_q, cu_k, max_sq, max_sk, causal=causal, backend="triton"
        )

    def call_gluon():
        return flash_attn_varlen_func(
            q, k, v, cu_q, cu_k, max_sq, max_sk, causal=causal, backend="gluon"
        )

    flops = 0.0
    for i in range(batch):
        flops += (
            2.0
            * 2.0
            * hq
            * _count_valid_attention_elements(
                int(len_q[i].item()), int(len_k[i].item()), causal
            )
            * d
        )
    return call_triton, call_gluon, flops


def _default_configs():
    """A small, representative sweep across heads / seqlens / head dims."""
    shapes = [
        # (batch, hq, hk, sq, sk, d)
        (4, 32, 32, 1024, 1024, 128),
        (2, 32, 8, 2048, 2048, 128),
        (1, 16, 16, 4096, 4096, 128),
        (2, 32, 8, 4096, 4096, 64),
        (1, 64, 8, 8192, 8192, 128),
    ]
    return shapes


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Benchmark naive Gluon MHA fwd vs Triton MHA fwd (gfx950)."
    )
    p.add_argument(
        "-fn",
        type=str,
        default="fwd",
        choices=VALID_FUNCTIONS,
        help=f"Function to benchmark: {VALID_FUNCTIONS}.",
    )
    p.add_argument("-b", type=int, default=0, help="batch (custom shape)")
    p.add_argument("-hq", type=int, default=0, help="num query heads (custom shape)")
    p.add_argument("-hk", type=int, default=0, help="num kv heads (custom shape)")
    p.add_argument("-sq", type=int, default=0, help="query seqlen (custom shape)")
    p.add_argument("-sk", type=int, default=0, help="key seqlen (custom shape)")
    p.add_argument("-d", type=int, default=0, help="head dim (custom shape)")
    p.add_argument(
        "-causal",
        type=_str2bool,
        default=None,
        help="Causal mask. Omit to sweep both False and True.",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=sorted(_DTYPE_MAP),
        help="Compute dtype.",
    )
    p.add_argument(
        "-equal_seqlens",
        action="store_true",
        default=False,
        help="Varlen: use equal sequence lengths instead of random ones.",
    )
    p.add_argument(
        "-no_check",
        action="store_true",
        default=False,
        help="Skip the gluon-vs-triton correctness check.",
    )
    p.add_argument(
        "-atol", type=float, default=1e-2, help="Absolute tolerance for the check."
    )
    p.add_argument(
        "-rtol", type=float, default=1e-2, help="Relative tolerance for the check."
    )
    return p.parse_args(argv)


def _build_config_list(args):
    dtype = _DTYPE_MAP[args.dtype]
    custom = bool(args.b or args.hq or args.sq or args.d)
    if custom:
        assert (
            args.b and args.hq and args.sq and args.d
        ), "Custom shape requires at least -b, -hq, -sq and -d."
        hk = args.hk if args.hk else args.hq
        sk = args.sk if args.sk else args.sq
        shapes = [(args.b, args.hq, hk, args.sq, sk, args.d)]
    else:
        shapes = _default_configs()

    causals = [args.causal] if args.causal is not None else [False, True]

    configs = []
    for (b, hq, hk, sq, sk, d), causal in itertools.product(shapes, causals):
        configs.append(
            dict(
                batch=b,
                hq=hq,
                hk=hk,
                sq=sq,
                sk=sk,
                d=d,
                causal=causal,
                dtype=dtype,
                function=args.fn,
                equal_seqlens=args.equal_seqlens,
            )
        )
    return configs


def _fmt_cfg(cfg):
    return (
        f"B={cfg['batch']} HQ={cfg['hq']} HK={cfg['hk']} "
        f"sq={cfg['sq']} sk={cfg['sk']} d={cfg['d']} "
        f"causal={cfg['causal']} {cfg['function']}"
    )


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if not torch.cuda.is_available():
        print("[ERROR] No CUDA/HIP device available.", flush=True)
        return
    if not _is_gluon_available():
        print(
            "[ERROR] Gluon backend is unavailable on this device "
            "(requires gfx950 + Triton >= 3.6). Nothing to compare.",
            flush=True,
        )
        return

    torch.manual_seed(20)
    device = "cuda"
    configs = _build_config_list(args)

    header = (
        f"{'config':<62}{'triton(ms)':>12}{'gluon(ms)':>12}"
        f"{'speedup':>9}{'tri TFLOPS':>12}{'glu TFLOPS':>12}{'check':>8}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for cfg in configs:
        if cfg["d"] not in _GLUON_SUPPORTED_HEAD_DIMS:
            print(
                f"{_fmt_cfg(cfg):<62}  [SKIP] head dim {cfg['d']} unsupported by Gluon",
                flush=True,
            )
            continue
        try:
            call_triton, call_gluon, flops = _build_inputs(cfg, device)

            check = "skip"
            if not args.no_check:
                out_tri = call_triton()
                out_glu = call_gluon()
                ok = torch.allclose(
                    out_glu.float(), out_tri.float(), atol=args.atol, rtol=args.rtol
                )
                check = "pass" if ok else "FAIL"

            ms_tri = triton.testing.do_bench(call_triton)
            ms_glu = triton.testing.do_bench(call_gluon)

            speedup = ms_tri / ms_glu if ms_glu > 0 else float("nan")
            tflops_tri = flops / ms_tri * 1e-9
            tflops_glu = flops / ms_glu * 1e-9

            print(
                f"{_fmt_cfg(cfg):<62}{ms_tri:>12.4f}{ms_glu:>12.4f}"
                f"{speedup:>9.2f}{tflops_tri:>12.1f}{tflops_glu:>12.1f}{check:>8}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"{_fmt_cfg(cfg):<62}  [SKIP] {e}", flush=True)
        finally:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
