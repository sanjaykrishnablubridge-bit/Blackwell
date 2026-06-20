import torch
import torch.nn.functional as F
import triton
import triton.testing
from pathlib import Path

N        = 1 << 24   # 16 M elements — matches gelu_ptx_sm120.cu
DATA_DIR = Path(__file__).parent.parent / "data"


def bench(fn, warmup=25, reps=100):
    ms_med, ms_p5, ms_p95 = triton.testing.do_bench(
        fn, warmup=warmup, rep=reps, quantiles=[0.5, 0.05, 0.95]
    )
    return ms_med, ms_p5, ms_p95


def save_reference():
    torch.manual_seed(0)
    x       = torch.rand(N, device='cuda', dtype=torch.float32)
    ref_out = F.gelu(x, approximate='tanh')
    DATA_DIR.mkdir(exist_ok=True)
    x.cpu().numpy().tofile(DATA_DIR / "gelu_inp.bin")
    ref_out.cpu().numpy().tofile(DATA_DIR / "gelu_out.bin")
    print(f"Saved GELU reference data (N={N}) to {DATA_DIR}/\n")


def main():
    torch.manual_seed(0)
    x = torch.rand(N, device='cuda', dtype=torch.float32)

    ref     = F.gelu(x, approximate='tanh')
    compiled = torch.compile(lambda t: F.gelu(t, approximate='tanh'))
    _ = compiled(x)
    torch.cuda.synchronize()

    out_cmp = compiled(x)
    print(f"Compiled vs F.gelu(tanh)  max abs err: {(ref - out_cmp).abs().max().item():.2e}\n")

    results = {
        "PyTorch  F.gelu(tanh)": bench(lambda: F.gelu(x, approximate='tanh')),
        "PyTorch  compiled":     bench(lambda: compiled(x)),
    }

    bw_bytes = x.numel() * x.element_size() * 2
    bw = lambda ms: bw_bytes / (ms * 1e-3) / 1e9

    print(f"N = {N} ({N * 4 / 1e6:.0f} MB)\n")
    header = f"{'Kernel':<30} {'median ms':>10}  {'p5 ms':>8}  {'p95 ms':>8}  {'GB/s (median)':>14}"
    print(header)
    print("-" * len(header))
    for name, (med, p5, p95) in results.items():
        print(f"{name:<30} {med:>10.4f}  {p5:>8.4f}  {p95:>8.4f}  {bw(med):>14.2f}")


if __name__ == '__main__':
    save_reference()
    main()
