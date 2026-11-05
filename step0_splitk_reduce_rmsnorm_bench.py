"""Step 0 go / no-go: split-K qkv_a GEMM + fused reduce / dual RMSNorm vs today's decode path.

Shape: Kimi-K2.6 / DeepSeek-V3 MLA qkv_a, N = 1536 (q_c) + 512 (kv_c) + 64 (k_pe), K = 7168, bf16.

  A  today:    tgemm.mm (FlyDSL hgemm, in-kernel split-K atomics) + HIP fused_qk_rmsnorm
  B  proposal: gemm_a16w16(skip_reduce=True) + fused_reduce_rms_fp8_group_quant with the
               unquantized q output (pessimistic: it also pays for an fp8 quant of q)
  C  ref:      gemm_a16w16 with its own split-K reduce kernel (what the regular tuning entry times)
  D  reduce kernel alone, over split counts

Timing: CUDA graphs of --copies calls on rotating weight copies (cold weights, as across the 61
layers), median over --reps replays, per call in us. A profiler pass then prints the kernels that
ran and their device times for A and the best B.

Run on the box, M=64 first:
    python step0_splitk_reduce_rmsnorm_bench.py --M 64
    python step0_splitk_reduce_rmsnorm_bench.py            # all M, D, writes step0_results.jsonl
"""

import argparse
import json
import statistics
import sys
import time

import torch

from aiter.ops.fused_qk_norm_rope_cache_quant import _fused_qk_rmsnorm
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.quant.fused_fp8_quant import fused_reduce_rms_fp8_group_quant
from aiter.ops.triton.utils.gemm_config_utils import (
    compute_splitk_params,
    get_gemm_config,
)
from aiter.tuned_gemm import tgemm

N_Q, N_KV, N_PE = 1536, 512, 64
N = N_Q + N_KV + N_PE
K = 7168
EPS = 1e-6
DTYPE = torch.bfloat16
SPLITS = (1, 2, 4, 7, 8, 14, 16)
DEV = "cuda"


def graph_time_us(fn, arg_sets, reps):
    """Median time per call of fn over arg_sets, captured in one CUDA graph."""
    for args in arg_sets:
        fn(*args)
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for args in arg_sets:
            fn(*args)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for args in arg_sets:
            fn(*args)
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    times = []
    for _ in range(reps):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0 / len(arg_sets))
    del graph
    return statistics.median(times)


def kernel_device_times(fn, args, iters=20):
    """Average device time (us) of each kernel fn launches, from the torch profiler."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        fn(*args)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn(*args)
        torch.cuda.synchronize()
    durations = {}
    for event in prof.events():
        if event.device_type == DeviceType.CUDA:
            durations.setdefault(event.name, []).append(event.time_range.elapsed_us())
    return {name: (len(v) / iters, statistics.mean(v)) for name, v in durations.items()}


def triton_resources(jit_fn):
    """(name, registers, spills) of every compiled specialization of a Triton kernel."""
    rows = []
    for entry in getattr(jit_fn, "device_caches", {}).values():
        cache = entry[0] if isinstance(entry, tuple) else entry
        for compiled in cache.values():
            try:
                compiled._init_handles()
            except Exception:  # noqa: BLE001
                pass
            name = getattr(compiled, "name", None) or getattr(
                compiled.metadata, "name", "?"
            )
            rows.append(
                (
                    name,
                    getattr(compiled, "n_regs", None),
                    getattr(compiled, "n_spills", None),
                )
            )
    return rows


def rms_ref(t, g):
    return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + EPS) * g.float()


def reference(x, w, wq, wkv):
    y = x.float() @ w.float().T
    q, kv, pe = y.split([N_Q, N_KV, N_PE], dim=-1)
    return rms_ref(q, wq), rms_ref(kv, wkv), pe


def rel_err(outs, refs):
    return max(
        ((o.float() - r).abs().max() / r.abs().max().clamp_min(1e-6)).item()
        for o, r in zip(outs, refs)
    )


def make_gemm_a(use_vllm_gemm):
    if use_vllm_gemm:
        from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl

        return lambda x, w: rocm_unquantized_gemm_impl(x, w, None)
    return lambda x, w: tgemm.mm(x, w)


def path_a(gemm, x, w, wq, wkv):
    q, kv, pe = gemm(x, w).split([N_Q, N_KV, N_PE], dim=-1)
    qn, kvn = _fused_qk_rmsnorm(None, q, wq, EPS, None, kv, wkv, EPS)
    return qn, kvn, pe


def norm_a(y, wq, wkv):
    q, kv, _ = y.split([N_Q, N_KV, N_PE], dim=-1)
    return _fused_qk_rmsnorm(None, q, wq, EPS, None, kv, wkv, EPS)


def gemm_partials(x, w, config):
    return gemm_a16w16(
        x, w, dtype=DTYPE, config=config, skip_reduce=True, backend="triton"
    )


def gemm_reduced(x, w, config):
    return gemm_a16w16(x, w, dtype=DTYPE, config=config, backend="triton")


def reduce_norm(p, wq, wkv):
    if p.dim() == 3:
        q, kv, pe = p[..., :N_Q], p[..., N_Q : N_Q + N_KV], p[..., N_Q + N_KV :]
        _, qn, kvn, _, pe_out = fused_reduce_rms_fp8_group_quant(
            q, wq, EPS, kv, wkv, EPS, pe, output_unquantized_inp1=True, dtype=DTYPE
        )
        return qn, kvn, pe_out
    q, kv, pe = p.split([N_Q, N_KV, N_PE], dim=-1)
    _, qn, kvn, _, _ = fused_reduce_rms_fp8_group_quant(
        q, wq, EPS, kv, wkv, EPS, None, output_unquantized_inp1=True, dtype=DTYPE
    )
    return qn, kvn, pe


def path_b(x, w, wq, wkv, config):
    return reduce_norm(gemm_partials(x, w, config), wq, wkv)


def candidates(M, quick):
    base, is_tuned = get_gemm_config("GEMM-A16W16", M, N, K, backend="triton")
    block_ms = [b for b in (16, 32, 64) if b <= max(16, M)]
    block_ns = (32, 64, 128) if quick else (16, 32, 64, 128)
    block_ks = (128, 256)
    warps = (4,) if quick else (4, 8)
    configs = {"default": compute_splitk_params(dict(base), K)}
    for bm in block_ms:
        for bn in block_ns:
            for bk in block_ks:
                for split in SPLITS:
                    for nw in warps:
                        config = dict(base)
                        config.update(
                            BLOCK_SIZE_M=bm,
                            BLOCK_SIZE_N=bn,
                            BLOCK_SIZE_K=bk,
                            NUM_KSPLIT=split,
                            num_warps=nw,
                        )
                        config = compute_splitk_params(config, K)
                        key = json.dumps(config, sort_keys=True)
                        configs.setdefault(key, config)
    return list(configs.values()), is_tuned


def short(config):
    return (
        f"{config['BLOCK_SIZE_M']}x{config['BLOCK_SIZE_N']}x{config['BLOCK_SIZE_K']}"
        f" S{config['NUM_KSPLIT']} w{config['num_warps']}"
    )


def run_m(M, ws, wq, wkv, args, log):
    xs = [torch.randn(M, K, device=DEV, dtype=DTYPE) for _ in ws]
    arg_sets = list(zip(xs, ws))
    gemm_a = make_gemm_a(args.vllm_gemm)
    refs = reference(xs[0], ws[0], wq, wkv)

    a_pair = graph_time_us(
        lambda x, w: path_a(gemm_a, x, w, wq, wkv), arg_sets, args.reps
    )
    a_gemm = graph_time_us(gemm_a, arg_sets, args.reps)
    ys = [gemm_a(x, w) for x, w in arg_sets]
    a_norm = graph_time_us(lambda y: norm_a(y, wq, wkv), [(y,) for y in ys], args.reps)
    a_err = rel_err(path_a(gemm_a, xs[0], ws[0], wq, wkv), refs)
    log(
        {
            "M": M,
            "path": "A",
            "pair_us": a_pair,
            "gemm_us": a_gemm,
            "norm_us": a_norm,
            "rel_err": a_err,
        }
    )

    configs, is_tuned = candidates(M, args.quick)
    print(
        f"\nM={M}: A pair {a_pair:.2f} us (gemm {a_gemm:.2f} + norm {a_norm:.2f}), "
        f"rel err {a_err:.2e}; {len(configs)} B/C candidates, GEMM-A16W16 is_tuned={is_tuned}"
    )
    rows = []
    t0 = time.time()
    for i, config in enumerate(configs):
        row = {"M": M, "config": config}
        try:
            row["b_pair_us"] = graph_time_us(
                lambda x, w, c=config: path_b(x, w, wq, wkv, c), arg_sets, args.reps
            )
            row["b_gemm_us"] = graph_time_us(
                lambda x, w, c=config: gemm_partials(x, w, c), arg_sets, args.reps
            )
            row["c_gemm_us"] = graph_time_us(
                lambda x, w, c=config: gemm_reduced(x, w, c), arg_sets, args.reps
            )
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        rows.append(row)
        log({"path": "BC", **row})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(configs)} candidates, {time.time() - t0:.0f} s")
    ok = [r for r in rows if "error" not in r]
    if not ok:
        print("  every candidate failed; first error:", rows[0].get("error"))
        return None
    best_b = min(ok, key=lambda r: r["b_pair_us"])
    best_b_gemm = min(ok, key=lambda r: r["b_gemm_us"])
    best_c = min(ok, key=lambda r: r["c_gemm_us"])
    b_err = rel_err(path_b(xs[0], ws[0], wq, wkv, best_b["config"]), refs)
    print(
        f"  B best pair  {best_b['b_pair_us']:.2f} us [{short(best_b['config'])}]"
        f" (its gemm {best_b['b_gemm_us']:.2f}), rel err {b_err:.2e}"
    )
    print(
        f"  B best gemm  {best_b_gemm['b_gemm_us']:.2f} us [{short(best_b_gemm['config'])}]"
    )
    print(
        f"  C best gemm+own reduce {best_c['c_gemm_us']:.2f} us [{short(best_c['config'])}];"
        f" that config as the B pair: {best_c['b_pair_us']:.2f} us"
    )
    print(f"  A - B = {a_pair - best_b['b_pair_us']:+.2f} us per layer")
    failed = len(rows) - len(ok)
    if failed:
        print(f"  {failed} candidates failed (see jsonl)")

    print("  kernels (profiler, calls per iteration x avg device us):")
    for label, fn, fargs in (
        ("A", lambda x, w: path_a(gemm_a, x, w, wq, wkv), arg_sets[0]),
        ("B", lambda x, w: path_b(x, w, wq, wkv, best_b["config"]), arg_sets[0]),
    ):
        try:
            kernels = kernel_device_times(fn, fargs)
        except Exception as e:  # noqa: BLE001
            print(f"    {label}: profiler failed ({type(e).__name__}: {e})")
            continue
        for name, (calls, us) in kernels.items():
            print(f"    {label} {calls:4.1f} x {us:7.2f}  {name[:110]}")
    summary = {
        "M": M,
        "a_pair_us": a_pair,
        "a_gemm_us": a_gemm,
        "a_norm_us": a_norm,
        "b_pair_us": best_b["b_pair_us"],
        "b_gemm_us": best_b["b_gemm_us"],
        "b_config": best_b["config"],
        "c_gemm_us": best_c["c_gemm_us"],
        "c_config": best_c["config"],
        "c_as_b_pair_us": best_c["b_pair_us"],
        "a_err": a_err,
        "b_err": b_err,
    }
    log({"path": "summary", **summary})
    return summary


def run_reduce_only(wq, wkv, args, log):
    from aiter.ops.triton._triton_kernels.quant.fused_fp8_quant import (
        _fused_reduce_rms_fp8_group_quant_kernel,
    )

    print("\nD: reduce kernel alone (partials fp32 [S, M, 2112])")
    for M in (64, 256):
        for split in SPLITS:
            if split == 1:
                parts = [torch.randn(M, N, device=DEV, dtype=DTYPE) for _ in range(8)]
            else:
                parts = [
                    torch.randn(split, M, N, device=DEV, dtype=torch.float32)
                    for _ in range(8)
                ]
            us = graph_time_us(
                lambda p: reduce_norm(p, wq, wkv), [(p,) for p in parts], args.reps
            )
            print(f"  M={M:4d} S={split:2d}: {us:6.2f} us")
            log({"path": "D", "M": M, "S": split, "us": us})
    print("  compiled specializations (name, registers, spills):")
    try:
        resources = triton_resources(_fused_reduce_rms_fp8_group_quant_kernel)
    except Exception as e:  # noqa: BLE001
        print(f"    not readable on this Triton ({type(e).__name__}: {e})")
        resources = []
    for name, regs, spills in resources:
        print(f"    regs={regs} spills={spills}  {name[:120]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--M", type=int, nargs="+", default=[1, 8, 16, 32, 64, 128, 256]
    )
    parser.add_argument("--copies", type=int, default=16, help="rotating weight copies")
    parser.add_argument("--reps", type=int, default=20, help="graph replays per timing")
    parser.add_argument("--quick", action="store_true", help="smaller candidate grid")
    parser.add_argument("--skip-reduce-only", action="store_true", help="skip part D")
    parser.add_argument(
        "--vllm-gemm",
        action="store_true",
        help="A's GEMM through vLLM's rocm_unquantized_gemm_impl",
    )
    parser.add_argument("--out", default="step0_results.jsonl")
    args = parser.parse_args()

    torch.manual_seed(0)
    ws = [
        torch.randn(N, K, device=DEV, dtype=DTYPE) * K**-0.5 for _ in range(args.copies)
    ]
    wq = 1 + 0.1 * torch.randn(N_Q, device=DEV, dtype=DTYPE)
    wkv = 1 + 0.1 * torch.randn(N_KV, device=DEV, dtype=DTYPE)
    print(
        f"device {torch.cuda.get_device_name()}, weights {args.copies} x "
        f"{N * K * 2 / 2**20:.1f} MiB, torch {torch.__version__}"
    )

    with open(args.out, "w", encoding="utf-8") as out:

        def log(record):
            out.write(json.dumps(record) + "\n")
            out.flush()

        summaries = [s for s in (run_m(M, ws, wq, wkv, args, log) for M in args.M) if s]
        if not args.skip_reduce_only:
            run_reduce_only(wq, wkv, args, log)

    print("\n   M |  A pair | B pair | A - B | B gemm | C gemm | C cfg as B | B config")
    for s in summaries:
        print(
            f"{s['M']:4d} | {s['a_pair_us']:7.2f} | {s['b_pair_us']:6.2f} | "
            f"{s['a_pair_us'] - s['b_pair_us']:+5.2f} | {s['b_gemm_us']:6.2f} | "
            f"{s['c_gemm_us']:6.2f} | {s['c_as_b_pair_us']:10.2f} | {short(s['b_config'])}"
        )
    m64 = next((s for s in summaries if s["M"] == 64), None)
    if m64:
        gain = m64["a_pair_us"] - m64["b_pair_us"]
        print(
            f"\nM=64: {'GO' if gain >= 2.0 else 'NO-GO'} (A - B = {gain:+.2f} us, threshold 2 us)"
        )
    print(f"records: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
