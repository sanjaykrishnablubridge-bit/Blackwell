import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing
from pathlib import Path

BATCH = 8192
C     = 1024
EPS   = 1e-5

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Triton kernel ──────────────────────────────────────────────────────────────
#
# Single HBM pass: X is loaded once into registers. Mean is computed, then
# variance is computed from those same registers — no second trip to HBM.
# CUDA V1 reads inp 3× (mean pass, variance pass, output pass).
# CUDA V2 reads inp 3× from HBM too (each stride-loop re-fetches from L2/HBM).
# Triton: 1 read (X) + 1 write (Y) + small reads (W, B) + small writes (Mean, Rstd).
# Compare time (ms), not GB/s, across Python and CUDA files.

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
def layernorm_fwd_kernel(
    X_ptr, W_ptr, B_ptr,
    Y_ptr, Mean_ptr, Rstd_ptr,
    row_stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X_ptr += row * row_stride
    Y_ptr += row * row_stride

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # single load — X lives in registers for both passes below
    x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # pass 1 (in register): mean
    mean = tl.sum(x, axis=0) / N

    # pass 2 (in register): variance
    xc   = tl.where(mask, x - mean, 0.0)
    var  = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(Mean_ptr + row, mean)
    tl.store(Rstd_ptr + row, rstd)

    # affine transform and output
    w = tl.load(W_ptr + cols, mask=mask)
    b = tl.load(B_ptr + cols, mask=mask)
    y = xc * rstd * w + b
    tl.store(Y_ptr + cols, y, mask=mask)


def triton_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
):
    batch, n = x.shape
    y    = torch.empty_like(x)
    mean = torch.empty(batch, device=x.device, dtype=x.dtype)
    rstd = torch.empty(batch, device=x.device, dtype=x.dtype)
    layernorm_fwd_kernel[(batch,)](
        x, weight, bias,
        y, mean, rstd,
        x.stride(0),
        n,
        eps,
        BLOCK_SIZE=triton.next_power_of_2(n),
    )
    return y, mean, rstd


# ── Benchmark helper ───────────────────────────────────────────────────────────

def bench(fn, warmup=25, reps=100):
    ms_med, ms_p5, ms_p95 = triton.testing.do_bench(
        fn, warmup=warmup, rep=reps, quantiles=[0.5, 0.05, 0.95]
    )
    return ms_med, ms_p5, ms_p95


def main():
    x      = torch.randn(BATCH, C, device='cuda', dtype=torch.float32)
    weight = torch.ones(C,       device='cuda', dtype=torch.float32)
    bias   = torch.zeros(C,      device='cuda', dtype=torch.float32)

    # compile PyTorch layer_norm so Inductor can fuse/optimize
    compiled_ln = torch.compile(lambda x, w, b: F.layer_norm(x, (C,), w, b))
    _ = compiled_ln(x, weight, bias)
    torch.cuda.synchronize()

    # ── correctness ───────────────────────────────────────────────────────────
    ref         = F.layer_norm(x, (C,), weight, bias)
    out_tri, *_ = triton_layernorm(x, weight, bias)
    out_cmp     = compiled_ln(x, weight, bias)

    print(f"Triton   vs F.layer_norm  max abs err: {(ref - out_tri).abs().max().item():.2e}")
    print(f"Compiled vs F.layer_norm  max abs err: {(ref - out_cmp).abs().max().item():.2e}\n")

    # ── benchmark ─────────────────────────────────────────────────────────────
    results = {
        "PyTorch  F.layer_norm":  bench(lambda: F.layer_norm(x, (C,), weight, bias)),
        "PyTorch  compiled":      bench(lambda: compiled_ln(x, weight, bias)),
        "Triton   layernorm":     bench(lambda: triton_layernorm(x, weight, bias)),
    }

    # Bandwidth: Triton is single-pass (1 read X + 1 write Y dominant).
    # CUDA V1/V2 are multi-pass — use time (ms) for cross-file comparison.
    x_bytes    = x.numel()      * x.element_size()
    y_bytes    = x.numel()      * x.element_size()
    w_bytes    = weight.numel() * weight.element_size()
    b_bytes    = bias.numel()   * bias.element_size()
    total_bytes = x_bytes + y_bytes + w_bytes + b_bytes   # single-pass estimate
    bw = lambda ms: total_bytes / (ms * 1e-3) / 1e9

    print(f"Dimensions: BATCH={BATCH}, C={C}\n")
    header = f"{'Kernel':<30} {'median ms':>10}  {'p5 ms':>8}  {'p95 ms':>8}  {'GB/s (median)':>14}"
    print(header)
    print("-" * len(header))
    for name, (med, p5, p95) in results.items():
        print(f"{name:<30} {med:>10.4f}  {p5:>8.4f}  {p95:>8.4f}  {bw(med):>14.2f}")


def save_reference():
    torch.manual_seed(0)
    x      = torch.randn(BATCH, C, device='cuda', dtype=torch.float32)
    weight = torch.ones(C,        device='cuda', dtype=torch.float32)
    bias   = torch.zeros(C,       device='cuda', dtype=torch.float32)
    ref_out = F.layer_norm(x, (C,), weight, bias, eps=EPS)
    mean    = x.mean(dim=-1)
    xc      = x - mean.unsqueeze(-1)
    var     = (xc * xc).mean(dim=-1)
    rstd    = 1.0 / torch.sqrt(var + EPS)
    DATA_DIR.mkdir(exist_ok=True)
    x.cpu().numpy().tofile(DATA_DIR / "layernorm_inp.bin")
    weight.cpu().numpy().tofile(DATA_DIR / "layernorm_weight.bin")
    bias.cpu().numpy().tofile(DATA_DIR / "layernorm_bias.bin")
    ref_out.cpu().numpy().tofile(DATA_DIR / "layernorm_out.bin")
    mean.cpu().numpy().tofile(DATA_DIR / "layernorm_mean.bin")
    rstd.cpu().numpy().tofile(DATA_DIR / "layernorm_rstd.bin")
    print(f"Saved layernorm reference data ({BATCH}×{C}) to {DATA_DIR}/\n")


if __name__ == '__main__':
    save_reference()
    main()
