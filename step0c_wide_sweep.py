"""Step 0c: wider, staged Triton sweep of the B pair (gemm_a16w16 skip_reduce + reduce/norm).

Step 0 only varied tiles, NUM_KSPLIT and num_warps. Here, at each M (default 64):
  stage 1  seeds: the fastest B pairs from step0_results.jsonl (or step 0's grid if absent)
  stage 2  per seed: num_stages x waves_per_eu x cache_modifier (x GROUP_SIZE_M if >1 M tile)
  stage 3  per best-so-far config: NUM_KSPLIT x BLOCK_SIZE_K x num_warps
  stage 4  per best-so-far config: BLOCK_SIZE_N and BLOCK_SIZE_M neighbours
  final    A and the top B configs re-timed with more replays, kernels profiled
Each stage starts from the fastest configs of all earlier stages. Values come from the tuner's
SEARCH_SPACE (utils/_triton/tuning/space.py), plus num_stages 3 and 4 (the default bucket uses
3; the tuner's list stops at 2). Same timing as
step 0 (CUDA graphs over rotating cold weight copies). Reuses step0_splitk_reduce_rmsnorm_bench.py
from the same folder.

    python step0c_wide_sweep.py --M 64
"""

import argparse
import json
import os
import sys
import time

import torch

import step0_splitk_reduce_rmsnorm_bench as s0
from aiter.ops.triton.utils.gemm_config_utils import (
    compute_splitk_params,
    get_gemm_config,
)

STAGE2 = {
    "num_stages": (1, 2, 3, 4),
    "waves_per_eu": (1, 2, 4, 6, 8),
    "cache_modifier": (".cg", None),
}
STAGE3 = {
    "NUM_KSPLIT": (3, 4, 7, 8, 14, 16, 28),
    "BLOCK_SIZE_K": (64, 128, 256, 512),
    "num_warps": (2, 4, 8),
}


def describe(c):
    return (
        f"{c['BLOCK_SIZE_M']}x{c['BLOCK_SIZE_N']}x{c['BLOCK_SIZE_K']} S{c['NUM_KSPLIT']}"
        f" w{c['num_warps']} st{c['num_stages']} wpe{c['waves_per_eu']}"
        f" cm{c.get('cache_modifier')} g{c['GROUP_SIZE_M']}"
    )


def raw(config):
    """The tunable keys only (drop what compute_splitk_params derives)."""
    return {k: v for k, v in config.items() if k != "SPLITK_BLOCK_SIZE"}


class Sweep:
    def __init__(self, M, ws, wq, wkv, reps, log):
        self.M, self.wq, self.wkv, self.reps, self.log = M, wq, wkv, reps, log
        self.arg_sets = [
            (torch.randn(M, s0.K, device=s0.DEV, dtype=s0.DTYPE), w) for w in ws
        ]
        self.timed = {}

    def time(self, config, stage):
        config = compute_splitk_params(dict(raw(config)), s0.K)
        key = json.dumps(raw(config), sort_keys=True)
        if key in self.timed:
            return self.timed[key]
        record = {"M": self.M, "stage": stage, "config": raw(config)}
        try:
            record["b_pair_us"] = s0.graph_time_us(
                lambda x, w, c=config: s0.path_b(x, w, self.wq, self.wkv, c),
                self.arg_sets,
                self.reps,
            )
        except Exception as e:  # noqa: BLE001
            record["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        self.timed[key] = record
        self.log(record)
        return record

    def run(self, configs, stage):
        t0 = time.time()
        records = [self.time(c, stage) for c in configs]
        ok = sorted(
            (r for r in records if "error" not in r), key=lambda r: r["b_pair_us"]
        )
        failed = len(records) - len(ok)
        print(
            f"  stage {stage}: {len(records)} configs, {failed} failed, "
            f"{time.time() - t0:.0f} s; best {ok[0]['b_pair_us']:.2f} us "
            f"[{describe(ok[0]['config'])}]"
            if ok
            else f"  stage {stage}: all failed"
        )
        return self.best()

    def best(self):
        """Every config timed so far, fastest first: each stage builds on the overall best."""
        return sorted(
            (r for r in self.timed.values() if "error" not in r),
            key=lambda r: r["b_pair_us"],
        )


def seeds_from_jsonl(path, M, top):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("path") == "BC" and r.get("M") == M and "b_pair_us" in r:
                rows.append(r)
    rows.sort(key=lambda r: r["b_pair_us"])
    seen, out = set(), []
    for r in rows:
        key = json.dumps(raw(r["config"]), sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(raw(r["config"]))
        if len(out) == top:
            break
    return out


def stage1_grid(M):
    base, _ = get_gemm_config("GEMM-A16W16", M, s0.N, s0.K, backend="triton")
    out = []
    for bm in (b for b in (16, 32, 64) if b <= max(16, M)):
        for bn in (16, 32, 64, 128):
            for bk in (128, 256):
                for split in s0.SPLITS:
                    for nw in (4, 8):
                        c = dict(base)
                        c.update(
                            BLOCK_SIZE_M=bm,
                            BLOCK_SIZE_N=bn,
                            BLOCK_SIZE_K=bk,
                            NUM_KSPLIT=split,
                            num_warps=nw,
                        )
                        out.append(c)
    return out


def vary(config, axes):
    out = [dict(config)]
    for name, values in axes.items():
        out = [dict(c, **{name: v}) for c in out for v in values]
    return out


def sweep_m(M, ws, wq, wkv, args, log):
    print(f"\nM={M}")
    sweep = Sweep(M, ws, wq, wkv, args.reps, log)
    seeds = seeds_from_jsonl(args.seeds, M, args.top)
    if seeds:
        print(f"  {len(seeds)} seeds from {args.seeds}")
        ranked = sweep.run(seeds, 1)
    else:
        print(f"  no seeds in {args.seeds}: running step 0's grid as stage 1")
        ranked = sweep.run(stage1_grid(M), 1)[: args.top]

    stage2 = dict(STAGE2)
    multi_m_tile = any(
        -(-M // r["config"]["BLOCK_SIZE_M"]) > 1 for r in ranked[: args.top]
    )
    if multi_m_tile:
        stage2["GROUP_SIZE_M"] = (1, 4)
    ranked = sweep.run(
        [c for r in ranked[: args.top] for c in vary(r["config"], stage2)], 2
    )
    ranked = sweep.run(
        [c for r in ranked[: max(1, args.top // 2)] for c in vary(r["config"], STAGE3)],
        3,
    )
    neighbours = []
    for r in ranked[:2]:
        c = r["config"]
        for bn in {
            max(16, c["BLOCK_SIZE_N"] // 2),
            c["BLOCK_SIZE_N"],
            c["BLOCK_SIZE_N"] * 2,
        }:
            for bm in {
                max(16, c["BLOCK_SIZE_M"] // 2),
                c["BLOCK_SIZE_M"],
                min(c["BLOCK_SIZE_M"] * 2, max(16, M)),
            }:
                neighbours.append(dict(c, BLOCK_SIZE_N=bn, BLOCK_SIZE_M=bm))
    ranked = sweep.run(neighbours, 4)

    everything = sorted(
        (r for r in sweep.timed.values() if "error" not in r),
        key=lambda r: r["b_pair_us"],
    )
    finals = [compute_splitk_params(dict(r["config"]), s0.K) for r in everything[:5]]
    gemm_a = s0.make_gemm_a(args.vllm_gemm)
    reps = args.final_reps
    a_pair = s0.graph_time_us(
        lambda x, w: s0.path_a(gemm_a, x, w, wq, wkv), sweep.arg_sets, reps
    )
    print(f"  final ({reps} replays): A pair {a_pair:.2f} us")
    best = None
    for c in finals:
        pair = s0.graph_time_us(
            lambda x, w, c=c: s0.path_b(x, w, wq, wkv, c), sweep.arg_sets, reps
        )
        gemm = s0.graph_time_us(
            lambda x, w, c=c: s0.gemm_partials(x, w, c), sweep.arg_sets, reps
        )
        print(
            f"    B pair {pair:6.2f} us (gemm {gemm:5.2f})  A - B {a_pair - pair:+5.2f}"
            f"  [{describe(c)}]"
        )
        log(
            {
                "M": M,
                "stage": "final",
                "config": raw(c),
                "b_pair_us": pair,
                "b_gemm_us": gemm,
                "a_pair_us": a_pair,
            }
        )
        if best is None or pair < best[0]:
            best = (pair, gemm, c)
    x0, w0 = sweep.arg_sets[0]
    err = s0.rel_err(s0.path_b(x0, w0, wq, wkv, best[2]), s0.reference(x0, w0, wq, wkv))
    print(f"  best B rel err vs fp32 reference: {err:.2e}")
    try:
        for name, (calls, us) in s0.kernel_device_times(
            lambda x, w: s0.path_b(x, w, wq, wkv, best[2]), (x0, w0)
        ).items():
            print(f"    B {calls:4.1f} x {us:7.2f}  {name[:110]}")
    except Exception as e:  # noqa: BLE001
        print(f"    profiler failed ({type(e).__name__}: {e})")
    return {
        "M": M,
        "a_pair_us": a_pair,
        "b_pair_us": best[0],
        "b_gemm_us": best[1],
        "config": describe(best[2]),
        "tried": len(sweep.timed),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--M", type=int, nargs="+", default=[64])
    parser.add_argument("--seeds", default="step0_results.jsonl")
    parser.add_argument(
        "--top", type=int, default=8, help="configs carried between stages"
    )
    parser.add_argument("--copies", type=int, default=16)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--final-reps", type=int, default=100)
    parser.add_argument("--vllm-gemm", action="store_true")
    parser.add_argument("--out", default="step0c_results.jsonl")
    args = parser.parse_args()

    torch.manual_seed(0)
    ws = [
        torch.randn(s0.N, s0.K, device=s0.DEV, dtype=s0.DTYPE) * s0.K**-0.5
        for _ in range(args.copies)
    ]
    wq = 1 + 0.1 * torch.randn(s0.N_Q, device=s0.DEV, dtype=s0.DTYPE)
    wkv = 1 + 0.1 * torch.randn(s0.N_KV, device=s0.DEV, dtype=s0.DTYPE)
    print(
        f"device {torch.cuda.get_device_name()}; read ceiling for 30.3 MB cold: 6.56 us (step 0b)"
    )
    with open(args.out, "w", encoding="utf-8") as out:

        def log(record):
            out.write(json.dumps(record) + "\n")
            out.flush()

        results = [sweep_m(M, ws, wq, wkv, args, log) for M in args.M]

    print("\n   M |  A pair | B pair | A - B | B gemm | tried | B config")
    for r in results:
        print(
            f"{r['M']:4d} | {r['a_pair_us']:7.2f} | {r['b_pair_us']:6.2f} | "
            f"{r['a_pair_us'] - r['b_pair_us']:+5.2f} | {r['b_gemm_us']:6.2f} | "
            f"{r['tried']:5d} | {r['config']}"
        )
    for r in results:
        if r["M"] == 64:
            gain = r["a_pair_us"] - r["b_pair_us"]
            print(
                f"\nM=64: {'GO' if gain >= 2.0 else 'NO-GO'} (A - B = {gain:+.2f} us, "
                "threshold 2 us)"
            )
    print(f"records: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
