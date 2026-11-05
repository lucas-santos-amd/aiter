"""Step 0b: practical HBM read ceiling for the qkv_a weights (2112 x 7168 bf16, 30.3 MB, cold).

A GEMM at decode M streams its weights once, so the fastest plain read of the same bytes is the
real floor for it. A Triton streaming-read kernel (grid-stride loop, one partial sum per program)
is swept over grid size, block, num_warps and cache modifier; each config is timed in a CUDA
graph over rotating copies, as in step0_splitk_reduce_rmsnorm_bench.py.

References: torch copy_ (read + write), and hipBLASLt F.linear at M = 1 and 64 for the GEMM itself.

    python step0b_hbm_read_ceiling.py
"""

import argparse
import statistics
import sys

import torch
import triton
import triton.language as tl

N, K = 2112, 7168
DTYPE = torch.bfloat16
DEV = "cuda"


@triton.jit
def _stream_read_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr, CACHE: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(pid * BLOCK, n, num_programs * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        acc += tl.load(x_ptr + offs, mask=offs < n, other=0.0, cache_modifier=CACHE).to(
            tl.float32
        )
    tl.store(out_ptr + pid, tl.sum(acc))


def graph_time_us(fn, arg_sets, reps):
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


def sweep_read(numel, copies, reps):
    bufs = [torch.randn(numel, device=DEV, dtype=DTYPE) for _ in range(copies)]
    out = torch.empty(16384, device=DEV, dtype=torch.float32)
    results = []
    for block in (1024, 2048, 4096, 8192):
        for warps in (4, 8):
            for cache in ("", ".cg"):
                for grid in (256, 512, 1024, 2048, 4096, 8192):
                    grid = min(grid, triton.cdiv(numel, block))

                    def read(x, b=block, w=warps, c=cache, g=grid):
                        _stream_read_kernel[(g,)](
                            x, out, numel, BLOCK=b, CACHE=c, num_warps=w
                        )

                    us = graph_time_us(read, [(x,) for x in bufs], reps)
                    results.append((us, block, warps, cache or "default", grid))
    del bufs
    return sorted(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--copies", type=int, default=16)
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args()
    torch.manual_seed(0)
    print(
        f"device {torch.cuda.get_device_name()}, torch {torch.__version__}, "
        f"triton {triton.__version__}"
    )

    weight_numel = N * K
    weight_mb = weight_numel * 2 / 1e6
    big_numel = 512 * 2**20  # 1 GiB of bf16

    for label, numel, copies in (
        (f"qkv_a weights ({weight_mb:.1f} MB)", weight_numel, args.copies),
        ("1 GiB buffer", big_numel, 2),
    ):
        results = sweep_read(numel, copies, args.reps)
        nbytes = numel * 2
        print(f"\n{label}, {copies} rotating copies: 5 fastest read configs")
        for us, block, warps, cache, grid in results[:5]:
            print(
                f"  {us:8.2f} us  {nbytes / us / 1e6:6.2f} TB/s  "
                f"BLOCK={block} warps={warps} cache={cache} grid={grid}"
            )

    srcs = [
        torch.randn(weight_numel, device=DEV, dtype=DTYPE) for _ in range(args.copies)
    ]
    dsts = [torch.empty_like(s) for s in srcs]
    us = graph_time_us(lambda d, s: d.copy_(s), list(zip(dsts, srcs)), args.reps)
    print(
        f"\ncopy_ of {weight_mb:.1f} MB: {us:.2f} us, "
        f"{2 * weight_numel * 2 / us / 1e6:.2f} TB/s (read + write)"
    )
    del srcs, dsts

    ws = [torch.randn(N, K, device=DEV, dtype=DTYPE) for _ in range(args.copies)]
    print("\nhipBLASLt F.linear (cold weights), for reference:")
    for M in (1, 64):
        xs = [torch.randn(M, K, device=DEV, dtype=DTYPE) for _ in ws]
        us = graph_time_us(torch.nn.functional.linear, list(zip(xs, ws)), args.reps)
        print(
            f"  M={M:3d}: {us:.2f} us, {weight_numel * 2 / us / 1e6:.2f} TB/s of weights"
        )
    print(
        "\nRoofline on the sheet: 30.3 MB / 6 TB/s = 5.05 us (+ KLL); "
        "step 0: Triton partials GEMM 7.25 us (M=1), 9.19 us (M=64)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
