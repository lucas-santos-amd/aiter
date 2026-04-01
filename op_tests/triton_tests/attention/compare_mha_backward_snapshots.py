#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Compare ``test_mha_backward`` snapshot tensors saved on two GPUs (or two runs).

Snapshots are written when ``AITER_MHA_BWD_SNAPSHOT_DIR`` is set while running
``pytest`` (see ``test_mha.py``). Each run creates
``$AITER_MHA_BWD_SNAPSHOT_DIR/<arch>/*.pt``.

Example:

  # On machine A (e.g. gfx942)
  export AITER_MHA_BWD_SNAPSHOT_DIR=/tmp/mha_snaps_a
  pytest op_tests/triton_tests/attention/test_mha.py::test_mha_backward -k "..." -q

  # Copy /tmp/mha_snaps_a to a shared folder, then on machine B (e.g. gfx950)
  export AITER_MHA_BWD_SNAPSHOT_DIR=/tmp/mha_snaps_b
  pytest ... # same -k selection

  python op_tests/triton_tests/attention/compare_mha_backward_snapshots.py \\
      /tmp/mha_snaps_a/gfx942 /tmp/mha_snaps_b/gfx950

Reports per matching snapshot file:
  - max |triton_A - triton_B|
  - max |torch_A - torch_B|
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


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
        "--fail-on-diff",
        action="store_true",
        help="Exit with code 1 if any triton or torch max-abs diff is non-zero",
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

    print(
        f"Comparing {len(names)} snapshot(s)\n"
        f"  A: {args.dir_a.resolve()}\n"
        f"  B: {args.dir_b.resolve()}\n"
    )
    header = f"{'name':<56} {'|d triton|':>12} {'|d torch|':>12}"
    print(header)
    print("-" * len(header))

    worst_triton = 0.0
    worst_torch = 0.0
    worst_triton_name = ""
    worst_torch_name = ""

    for name in names:
        da = _load(idx_a[name])
        db = _load(idx_b[name])
        ta, tb = da["triton_out"], db["triton_out"]
        oa, ob = da["torch_out"], db["torch_out"]
        if ta.shape != tb.shape:
            print(f"{name[:56]:<56} triton SHAPE MISMATCH {tuple(ta.shape)} vs {tuple(tb.shape)}")
            worst_triton = float("inf")
            continue
        if oa.shape != ob.shape:
            print(f"{name[:56]:<56} torch SHAPE MISMATCH {tuple(oa.shape)} vs {tuple(ob.shape)}")
            worst_torch = float("inf")
            continue
        dt = _max_abs(ta, tb)
        do = _max_abs(oa, ob)
        if dt > worst_triton:
            worst_triton, worst_triton_name = dt, name
        if do > worst_torch:
            worst_torch, worst_torch_name = do, name
        short = name[:56] if len(name) <= 56 else name[:53] + "..."
        print(f"{short:<56} {dt:12.6g} {do:12.6g}")

    print("-" * len(header))
    print(f"Worst |d triton|: {worst_triton:.6g}  ({worst_triton_name})")
    print(f"Worst |d torch|: {worst_torch:.6g}  ({worst_torch_name})")

    if args.fail_on_diff and (worst_triton > 0.0 or worst_torch > 0.0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
