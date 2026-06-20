import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing
from pathlib import Path

M    = 4096   # rows
N    = 4096   # output cols  (input is M x 2N — first half gate, second half value)

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Triton kernel ──────────────────────────────────────────────────────────────
# Each program handles one row. Loads a and b in one vectorised pass,
# applies swish(a)*b, stores output.
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=['N'],
)
@triton.jit
def swiglu_kernel(
    X_ptr, Y_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    row  = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    a_ptr = X_ptr + row * 2 * N + cols
    b_ptr = X_ptr + row * 2 * N + N + cols

    a = tl.load(a_ptr, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr, mask=mask, other=0.0).to(tl.float32)

    swish_a = a * tl.sigmoid(a)
    y = swish_a * b

    tl.store(Y_ptr + row * N + cols, y, mask=mask)


def triton_swiglu(x: torch.Tensor):
    m, two_n = x.shape
    n = two_n // 2
    y = torch.empty(m, n, device=x.device, dtype=x.dtype)
    swiglu_kernel[(m,)](x, y, m, n, BLOCK_SIZE=triton.next_power_of_2(n))
    return y


def pt_swiglu(x: torch.Tensor):
    n = x.shape[1] // 2
    a, b = x[:, :n], x[:, n:]
    return F.silu(a) * b


def bench(fn, warmup=25, reps=100):
    ms_med, ms_p5, ms_p95 = triton.testing.do_bench(
        fn, warmup=warmup, rep=reps, quantiles=[0.5, 0.05, 0.95]
    )
    return ms_med, ms_p5, ms_p95


def main():
    torch.manual_seed(0)
    x = torch.randn(M, 2 * N, device='cuda', dtype=torch.float32)

    # ── PyTorch reference ─────────────────────────────────────────────────────
    ref_out = pt_swiglu(x)

    # ── correctness: Triton vs PyTorch ────────────────────────────────────────
    out_tri = triton_swiglu(x)
    print(f"Triton vs PyTorch  out  max abs err: {(ref_out - out_tri).abs().max().item():.2e}\n")

    # ── save reference data for CUDA correctness check ────────────────────────
    DATA_DIR.mkdir(exist_ok=True)
    x.cpu().numpy().tofile(DATA_DIR / "swiglu_inp.bin")
    ref_out.cpu().numpy().tofile(DATA_DIR / "swiglu_out.bin")
    print(f"Saved reference data to {DATA_DIR}/\n")

    # ── benchmark ─────────────────────────────────────────────────────────────
    compiled_swiglu = torch.compile(pt_swiglu)
    _ = compiled_swiglu(x)
    torch.cuda.synchronize()

    results = {
        "PyTorch  swiglu":   bench(lambda: pt_swiglu(x)),
        "PyTorch  compiled": bench(lambda: compiled_swiglu(x)),
        "Triton   swiglu":   bench(lambda: triton_swiglu(x)),
    }

    # bandwidth: read 2N floats per row (inp) + write N floats per row (out)
    bw_bytes = M * (2 * N + N) * 4
    bw = lambda ms: bw_bytes / (ms * 1e-3) / 1e9

    print(f"Dimensions: M={M}, N={N}  (input: {M}x{2*N}, output: {M}x{N})\n")
    header = f"{'Kernel':<30} {'median ms':>10}  {'p5 ms':>8}  {'p95 ms':>8}  {'GB/s (median)':>14}"
    print(header)
    print("-" * len(header))
    for name, (med, p5, p95) in results.items():
        print(f"{name:<30} {med:>10.4f}  {p5:>8.4f}  {p95:>8.4f}  {bw(med):>14.2f}")


if __name__ == '__main__':
    main()
