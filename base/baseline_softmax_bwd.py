import torch
import torch.nn.functional as F
import triton
import triton.testing
from pathlib import Path

ROWS     = 8192   # matches softmax_bwd_sm120.cu
COLS     = 1024
DATA_DIR = Path(__file__).parent.parent / "data"


def bench(fn, warmup=25, reps=100):
    ms_med, ms_p5, ms_p95 = triton.testing.do_bench(
        fn, warmup=warmup, rep=reps, quantiles=[0.5, 0.05, 0.95]
    )
    return ms_med, ms_p5, ms_p95


def save_reference():
    torch.manual_seed(0)
    x        = torch.randn(ROWS, COLS, device='cuda', dtype=torch.float32, requires_grad=True)
    grad_out = torch.rand(ROWS, COLS,  device='cuda', dtype=torch.float32)

    softmax_out = F.softmax(x, dim=-1)
    # Use PyTorch autograd — this is the exact backward PyTorch runs, not a manual reconstruction
    softmax_out.backward(grad_out)
    grad_in = x.grad.clone()

    DATA_DIR.mkdir(exist_ok=True)
    grad_out.cpu().numpy().tofile(DATA_DIR / "softmax_bwd_grad_out.bin")
    softmax_out.detach().cpu().numpy().tofile(DATA_DIR / "softmax_bwd_softmax_out.bin")
    grad_in.cpu().numpy().tofile(DATA_DIR / "softmax_bwd_ref.bin")
    print(f"Saved softmax-backward reference data ({ROWS}×{COLS}) to {DATA_DIR}/\n")


def main():
    torch.manual_seed(0)
    x        = torch.randn(ROWS, COLS, device='cuda', dtype=torch.float32, requires_grad=True)
    grad_out = torch.rand(ROWS, COLS,  device='cuda', dtype=torch.float32)

    # ── Correctness: autograd vs manual formula ───────────────────────────────
    softmax_out = F.softmax(x, dim=-1)
    softmax_out.backward(grad_out)
    grad_in_autograd = x.grad.clone()

    y = softmax_out.detach()
    dot         = (grad_out * y).sum(dim=-1, keepdim=True)
    grad_in_man = y * (grad_out - dot)

    print(f"Autograd vs manual formula  max abs err: {(grad_in_autograd - grad_in_man).abs().max().item():.2e}\n")

    # ── Benchmarks ────────────────────────────────────────────────────────────
    # fwd+bwd: matches what a training loop does end-to-end
    x_fwd = x.detach().requires_grad_(True)
    fwd_bwd_fn = lambda: F.softmax(x_fwd, dim=-1).backward(grad_out)

    # bwd only: pre-computed softmax_out — apples-to-apples with CUDA bwd kernels
    x_bwd = x.detach().requires_grad_(True)
    y_bwd = F.softmax(x_bwd, dim=-1)   # pre-computed outside the timed loop
    bwd_only_fn = lambda: torch.autograd.grad(y_bwd, x_bwd, grad_out, retain_graph=True)

    # warmup
    fwd_bwd_fn(); bwd_only_fn()
    torch.cuda.synchronize()

    results = {
        "PyTorch  fwd + bwd":  bench(fwd_bwd_fn),
        "PyTorch  bwd only":   bench(bwd_only_fn),
    }

    bw_bytes = x.numel() * x.element_size() * 4   # grad_out (r) + softmax_out (r×2) + grad_in (w)
    bw = lambda ms: bw_bytes / (ms * 1e-3) / 1e9

    print(f"ROWS={ROWS}, COLS={COLS}\n")
    header = f"{'Kernel':<30} {'median ms':>10}  {'p5 ms':>8}  {'p95 ms':>8}  {'GB/s (median)':>14}"
    print(header)
    print("-" * len(header))
    for name, (med, p5, p95) in results.items():
        print(f"{name:<30} {med:>10.4f}  {p5:>8.4f}  {p95:>8.4f}  {bw(med):>14.2f}")


if __name__ == '__main__':
    save_reference()
    main()
