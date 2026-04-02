#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Compare ``test_mha_backward`` snapshot tensors saved on two GPUs (or two runs).

Snapshots are written when ``AITER_MHA_BWD_SNAPSHOT_DIR`` is set while running
``pytest`` (see ``test_mha.py``). Each run creates
``$AITER_MHA_BWD_SNAPSHOT_DIR/<arch>/*.pt``.

Example (mask from gfx950 in dir_b — typical failing GPU vs reference):

  python op_tests/triton_tests/attention/compare_mha_backward_snapshots.py \\
      /tmp/mha_snaps/gfx942 /tmp/mha_snaps/gfx950 \\
      --mask-source b --mask-threshold 0.3

For each matching ``.pt`` pair, positions are selected only where
``|triton_out - torch_out| > mask-threshold`` **on the mask source run**
(e.g. gfx950). Cross-GPU columns report max absolute difference **restricted
to those indices**:
  - |triton_A - triton_B| at masked positions
  - |torch_A - torch_B| at masked positions

Use ``--verbose`` to print a line for every snapshot (including zero masked
count). By default, only rows with at least one masked position are printed.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _masked_max_abs(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    """Max |(a-b)| over elements where mask is True. NaN if mask is empty."""
    if not mask.any():
        return float("nan")
    return (a - b).abs()[mask].max().item()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "dir_a",
        type=Path,
        help="First snapshot directory (e.g. .../gfx942) containing .pt files",
    )
    p.add_argument(
        "dir_b",
        type=Path,
        help="Second snapshot directory (e.g. .../gfx950) containing .pt files",
    )
    p.add_argument(
        "--mask-source",
        choices=("a", "b"),
        default="b",
        help="Which run defines failing positions via |triton_out - torch_out| (default: b, e.g. gfx950)",
    )
    p.add_argument(
        "--mask-threshold",
        type=float,
        default=0.3,
        help="Element is 'failing' on mask source if |triton - torch| exceeds this (default: 0.3)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print one line per snapshot even when masked count is 0",
    )
    p.add_argument(
        "--fail-on-diff",
        action="store_true",
        help="Exit with code 1 if any masked max-abs cross-GPU diff is non-zero",
    )
    args = p.parse_args(argv)

    if not args.dir_a.is_dir() or not args.dir_b.is_dir():
        print("Both arguments must be existing directories.", file=sys.stderr)
        return 2

    idx_a = {f.name: f for f in args.dir_a.glob("*.pt")}
    idx_b = {f.name: f for f in args.dir_b.glob("*.pt")}
    names = sorted(set(idx_a) & set(idx_b))
    only_a = sorted(set(idx_a) - set(idx_b))
    only_b = sorted(set(idx_b) - set(idx_a))

    if only_a:
        print(f"Only in dir_a ({len(only_a)} files):", *only_a[:10], sep="\n  ")
        if len(only_a) > 10:
            print(f"  ... and {len(only_a) - 10} more")
    if only_b:
        print(f"Only in dir_b ({len(only_b)} files):", *only_b[:10], sep="\n  ")
        if len(only_b) > 10:
            print(f"  ... and {len(only_b) - 10} more")

    if not names:
        print("No matching .pt basenames between directories.", file=sys.stderr)
        return 3

    src_label = "A" if args.mask_source == "a" else "B"
    print(
        f"Comparing {len(names)} snapshot(s)\n"
        f"  A: {args.dir_a.resolve()}\n"
        f"  B: {args.dir_b.resolve()}\n"
        f"Mask: |triton - torch| > {args.mask_threshold} on side {src_label}\n"
    )
    col_w = 100
    header = (
        f"{'name':<{col_w}} {'n_mask':>8} "
        f"{'|d triton|@mask':>16} {'|d torch|@mask':>16}"
    )
    print(header)
    print("-" * len(header))

    worst_triton: float | None = None
    worst_torch: float | None = None
    worst_triton_name = ""
    worst_torch_name = ""
    any_fail_diff = False

    for name in names:
        da = _load(idx_a[name])
        db = _load(idx_b[name])
        ta, tb = da["triton_out"], db["triton_out"]
        oa, ob = da["torch_out"], db["torch_out"]

        if ta.shape != tb.shape:
            print(
                f"{name[:col_w]:<{col_w}} triton SHAPE MISMATCH "
                f"{tuple(ta.shape)} vs {tuple(tb.shape)}"
            )
            worst_triton = float("inf")
            any_fail_diff = True
            continue
        if oa.shape != ob.shape:
            print(
                f"{name[:col_w]:<{col_w}} torch SHAPE MISMATCH "
                f"{tuple(oa.shape)} vs {tuple(ob.shape)}"
            )
            worst_torch = float("inf")
            any_fail_diff = True
            continue

        d_src = da if args.mask_source == "a" else db
        t_src = d_src["triton_out"]
        o_src = d_src["torch_out"]
        if t_src.shape != o_src.shape:
            print(
                f"{name[:col_w]:<{col_w}} mask-source triton/torch SHAPE MISMATCH"
            )
            any_fail_diff = True
            continue

        mask = (t_src - o_src).abs() > args.mask_threshold
        n_mask = int(mask.sum().item())

        if n_mask == 0:
            dt_m = float("nan")
            do_m = float("nan")
        else:
            dt_m = _masked_max_abs(ta, tb, mask)
            do_m = _masked_max_abs(oa, ob, mask)
            if not math.isnan(dt_m) and (
                worst_triton is None or dt_m > worst_triton
            ):
                worst_triton, worst_triton_name = dt_m, name
            if not math.isnan(do_m) and (worst_torch is None or do_m > worst_torch):
                worst_torch, worst_torch_name = do_m, name
            if (not math.isnan(dt_m) and dt_m > 0.0) or (
                not math.isnan(do_m) and do_m > 0.0
            ):
                any_fail_diff = True

        should_print = args.verbose or n_mask > 0
        if should_print:
            short = name[:col_w] if len(name) <= col_w else name[: col_w - 3] + "..."
            dt_s = f"{dt_m:.6g}" if not math.isnan(dt_m) else "n/a"
            do_s = f"{do_m:.6g}" if not math.isnan(do_m) else "n/a"
            print(f"{short:<{col_w}} {n_mask:8d} {dt_s:>16} {do_s:>16}")

    print("-" * len(header))
    wt = f"{worst_triton:.6g}" if worst_triton is not None else "n/a"
    wo = f"{worst_torch:.6g}" if worst_torch is not None else "n/a"
    print(f"Worst |triton_A - triton_B| @mask: {wt}  ({worst_triton_name})")
    print(f"Worst |torch_A - torch_B| @mask: {wo}  ({worst_torch_name})")

    if args.fail_on_diff and any_fail_diff:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
