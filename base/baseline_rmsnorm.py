import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing
from pathlib import Path

BATCH = 24567 #8192
C     = 24567 #1024
EPS   = 1e-5

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Triton kernel ──────────────────────────────────────────────────────────────
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
def rmsnorm_fwd_kernel(
    X_ptr, G_ptr,
    Y_ptr, Rstd_ptr,
    row_stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row   = tl.program_id(0)
    X_ptr += row * row_stride
    Y_ptr += row * row_stride

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x    = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rms  = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(rms + eps)

    tl.store(Rstd_ptr + row, rstd)

    g = tl.load(G_ptr + cols, mask=mask)
    y = x * rstd * g
    tl.store(Y_ptr + cols, y, mask=mask)


def triton_rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float = EPS):
    batch, n = x.shape
    y    = torch.empty_like(x)
    rstd = torch.empty(batch, device=x.device, dtype=x.dtype)
    rmsnorm_fwd_kernel[(batch,)](
        x, gamma, y, rstd,
        x.stride(0), n, eps,
        BLOCK_SIZE=triton.next_power_of_2(n),
    )
    return y, rstd


def bench(fn, warmup=25, reps=100):
    ms_med, ms_p5, ms_p95 = triton.testing.do_bench(
        fn, warmup=warmup, rep=reps, quantiles=[0.5, 0.05, 0.95]
    )
    return ms_med, ms_p5, ms_p95


def main():
    torch.manual_seed(0)
    x     = torch.randn(BATCH, C, device='cuda', dtype=torch.float32)
    gamma = torch.ones(C,        device='cuda', dtype=torch.float32)

    # ── PyTorch reference ─────────────────────────────────────────────────────
    ref_out  = F.rms_norm(x, (C,), gamma, eps=EPS)

    # rstd not exposed by F.rms_norm — compute manually (matches kernel definition)
    rms      = (x * x).mean(dim=-1)              # (N,)  mean of squares per row
    ref_rstd = 1.0 / torch.sqrt(rms + EPS)       # (N,)

    # ── correctness: Triton vs PyTorch ────────────────────────────────────────
    out_tri, rstd_tri = triton_rmsnorm(x, gamma)
    print(f"Triton vs F.rms_norm  out  max abs err: {(ref_out - out_tri).abs().max().item():.2e}")
    print(f"Triton vs F.rms_norm  rstd max abs err: {(ref_rstd - rstd_tri).abs().max().item():.2e}\n")

    # ── save reference data for CUDA correctness check ────────────────────────
    DATA_DIR.mkdir(exist_ok=True)
    x.cpu().numpy().tofile(DATA_DIR / "rmsnorm_inp.bin")
    gamma.cpu().numpy().tofile(DATA_DIR / "rmsnorm_gamma.bin")
    ref_out.cpu().numpy().tofile(DATA_DIR / "rmsnorm_out.bin")
    rms.cpu().numpy().tofile(DATA_DIR / "rmsnorm_mean.bin")
    ref_rstd.cpu().numpy().tofile(DATA_DIR / "rmsnorm_rstd.bin")
    print(f"Saved reference data to {DATA_DIR}/\n")

    # ── benchmark ─────────────────────────────────────────────────────────────
    compiled_rms = torch.compile(lambda x, g: F.rms_norm(x, (C,), g, eps=EPS))
    _ = compiled_rms(x, gamma)
    torch.cuda.synchronize()

    results = {
        "PyTorch  F.rms_norm": bench(lambda: F.rms_norm(x, (C,), gamma, eps=EPS)),
        "PyTorch  compiled":   bench(lambda: compiled_rms(x, gamma)),
        "Triton   rmsnorm":    bench(lambda: triton_rmsnorm(x, gamma)),
    }

    # RMSNorm single-pass bandwidth: 1 read X + 1 write Y (dominant)
    bw_bytes = x.numel() * x.element_size() * 2
    bw = lambda ms: bw_bytes / (ms * 1e-3) / 1e9

    print(f"Dimensions: BATCH={BATCH}, C={C}\n")
    header = f"{'Kernel':<30} {'median ms':>10}  {'p5 ms':>8}  {'p95 ms':>8}  {'GB/s (median)':>14}"
    print(header)
    print("-" * len(header))
    for name, (med, p5, p95) in results.items():
        print(f"{name:<30} {med:>10.4f}  {p5:>8.4f}  {p95:>8.4f}  {bw(med):>14.2f}")


if __name__ == '__main__':
    main()
