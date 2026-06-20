import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing
from pathlib import Path

ROWS = 8192
COLS = 8192

DATA_DIR = Path(__file__).parent.parent / "data"

# Dims saved to .bin files (must match softmax_sm120.cu defaults)
REF_ROWS, REF_COLS = 8192, 1024

# ── Triton kernel (autotuned num_warps) ───────────────────────────────────────
# do_bench uses torch.cuda.Event internally — all reported times are pure GPU
# execution time, not CPU-side Python overhead.

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=['n_cols'],
)
@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_start_ptr = input_ptr + row * input_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    x = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    x = x - tl.max(x, axis=0)
    numerator = tl.exp(x)

    out_row_start_ptr = output_ptr + row * output_row_stride
    tl.store(out_row_start_ptr + col_offsets, numerator / tl.sum(numerator, axis=0), mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    out = torch.empty_like(x)
    softmax_kernel[(rows,)](
        x, out,
        x.stride(0), out.stride(0),
        cols,
        BLOCK_SIZE=triton.next_power_of_2(cols),
    )
    return out


# ── Benchmark ─────────────────────────────────────────────────────────────────

def bench(fn, warmup=25, reps=100):
    # do_bench uses torch.cuda.Event — measures pure GPU time, not CPU overhead.
    # L2 cache is flushed before every rep via a 256 MB cache.zero_() (fast_flush=True
    # default), matching the cudaMemset flush in kernelBench.cuh.
    # quantiles=[0.5, 0.05, 0.95] -> returns (median, p5, p95)
    ms_med, ms_p5, ms_p95 = triton.testing.do_bench(
        fn, warmup=warmup, rep=reps, quantiles=[0.5, 0.05, 0.95]
    )
    return ms_med, ms_p5, ms_p95


def main():
    # uniform [0, 1] matches initVec() in kernelUtils.cuh
    x = torch.rand(ROWS, COLS, device='cuda', dtype=torch.float32)

    # ── compile PyTorch's softmax so Inductor can fuse/optimize it ────────────
    compiled_softmax = torch.compile(lambda t: F.softmax(t, dim=-1))
    _ = compiled_softmax(x)           # trigger JIT compilation before timing
    torch.cuda.synchronize()

    # ── correctness ───────────────────────────────────────────────────────────
    ref     = F.softmax(x, dim=-1)
    out_tri = triton_softmax(x)
    out_cmp = compiled_softmax(x)
    print(f"Triton   vs F.softmax  max abs err: {(ref - out_tri).abs().max().item():.2e}")
    print(f"Compiled vs F.softmax  max abs err: {(ref - out_cmp).abs().max().item():.2e}\n")

    # ── benchmark ─────────────────────────────────────────────────────────────
    results = {
        "PyTorch  F.softmax":        bench(lambda: F.softmax(x, dim=-1)),
        "PyTorch  compiled":         bench(lambda: compiled_softmax(x)),
        "Triton   softmax":          bench(lambda: triton_softmax(x)),
    }

    nbytes = x.numel() * x.element_size()
    # Triton kernel is single-pass: loads the row once, stores once → 2× NBytes.
    # Our CUDA kernels (V4–V6) are two-pass: load twice, store once → 3× NBytes.
    # GB/s is NOT directly comparable across the two files — use time (ms) for that.
    bw = lambda ms: 2 * nbytes / (ms * 1e-3) / 1e9

    print("For the dimension: {%d}, {%d}\n", ROWS, COLS)
    header = f"{'Kernel':<28} {'median ms':>10}  {'p5 ms':>8}  {'p95 ms':>8}  {'GB/s (median)':>14}"
    print(header)
    print("-" * len(header))
    for name, (med, p5, p95) in results.items():
        print(f"{name:<28} {med:>10.4f}  {p5:>8.4f}  {p95:>8.4f}  {bw(med):>14.2f}")


def save_reference():
    torch.manual_seed(0)
    x = torch.rand(REF_ROWS, REF_COLS, device='cuda', dtype=torch.float32)
    ref_out = F.softmax(x, dim=-1)
    DATA_DIR.mkdir(exist_ok=True)
    x.cpu().numpy().tofile(DATA_DIR / "softmax_inp.bin")
    ref_out.cpu().numpy().tofile(DATA_DIR / "softmax_out.bin")
    print(f"Saved softmax reference data ({REF_ROWS}×{REF_COLS}) to {DATA_DIR}/\n")


if __name__ == '__main__':
    save_reference()
    main()
