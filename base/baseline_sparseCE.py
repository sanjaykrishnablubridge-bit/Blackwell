import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing
from pathlib import Path

BATCH = 8192
VOCAB = 50304

DATA_DIR = Path(__file__).parent.parent / "data"

# Dims saved to .bin files (must match sparseCE_sm120.cu defaults)
REF_BATCH = 128

# ── Triton kernel (autotuned num_warps) ───────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=['vocab_size'],
)
@triton.jit
def sparse_ce_kernel(
    logits_ptr, targets_ptr, losses_ptr,
    logits_row_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_start = logits_ptr + row * logits_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < vocab_size

    # load row; out-of-bounds lanes get -inf so they don't affect max or sum
    x = tl.load(row_start + col_offsets, mask=mask, other=-float('inf'))

    x_max = tl.max(x, axis=0)

    # mask OOB lanes to -inf so exp(-inf)=0 contributes nothing to the sum
    x_shifted = tl.where(mask, x - x_max, -float('inf'))
    sum_exp = tl.sum(tl.exp(x_shifted), axis=0)

    t = tl.load(targets_ptr + row)
    logit_t = tl.load(row_start + t)

    tl.store(losses_ptr + row, tl.log(sum_exp) + x_max - logit_t)


def triton_sparse_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    batch, vocab = logits.shape
    losses = torch.empty(batch, device=logits.device, dtype=logits.dtype)
    sparse_ce_kernel[(batch,)](
        logits, targets, losses,
        logits.stride(0),
        vocab,
        BLOCK_SIZE=triton.next_power_of_2(vocab),
    )
    return losses


# ── Benchmark ─────────────────────────────────────────────────────────────────

def bench(fn, warmup=25, reps=100):
    ms_med, ms_p5, ms_p95 = triton.testing.do_bench(
        fn, warmup=warmup, rep=reps, quantiles=[0.5, 0.05, 0.95]
    )
    return ms_med, ms_p5, ms_p95


def main():
    logits  = torch.rand(BATCH, VOCAB, device='cuda', dtype=torch.float32)
    targets = torch.randint(0, VOCAB, (BATCH,), device='cuda', dtype=torch.int64)

    # ── compile PyTorch CE so Inductor can fuse/optimize it ──────────────────
    compiled_ce = torch.compile(lambda l, t: F.cross_entropy(l, t, reduction='none'))
    _ = compiled_ce(logits, targets)
    torch.cuda.synchronize()

    # ── correctness ───────────────────────────────────────────────────────────
    ref     = F.cross_entropy(logits, targets, reduction='none')
    out_tri = triton_sparse_ce(logits, targets)
    out_cmp = compiled_ce(logits, targets)
    print(f"Triton   vs F.cross_entropy  max abs err: {(ref - out_tri).abs().max().item():.2e}")
    print(f"Compiled vs F.cross_entropy  max abs err: {(ref - out_cmp).abs().max().item():.2e}\n")

    # ── benchmark ─────────────────────────────────────────────────────────────
    results = {
        "PyTorch  F.cross_entropy": bench(lambda: F.cross_entropy(logits, targets, reduction='none')),
        "PyTorch  compiled":        bench(lambda: compiled_ce(logits, targets)),
        "Triton   sparse CE":       bench(lambda: triton_sparse_ce(logits, targets)),
    }

    # Triton kernel is single-pass: loads logits once, stores losses once.
    # Our CUDA kernels (V1/V2) are two-pass: read logits twice (find-max + sum-exp).
    # GB/s here uses single-pass bytes — NOT directly comparable to sparseCE.cu GB/s.
    # Use time (ms) for cross-file comparisons.
    logits_bytes  = logits.numel()  * logits.element_size()
    targets_bytes = targets.numel() * targets.element_size()
    losses_bytes  = BATCH * logits.element_size()
    total_bytes   = logits_bytes + targets_bytes + losses_bytes
    bw = lambda ms: total_bytes / (ms * 1e-3) / 1e9

    print(f"For the dimension: BATCH={BATCH}, VOCAB={VOCAB}\n")
    header = f"{'Kernel':<30} {'median ms':>10}  {'p5 ms':>8}  {'p95 ms':>8}  {'GB/s (median)':>14}"
    print(header)
    print("-" * len(header))
    for name, (med, p5, p95) in results.items():
        print(f"{name:<30} {med:>10.4f}  {p5:>8.4f}  {p95:>8.4f}  {bw(med):>14.2f}")


def save_reference():
    torch.manual_seed(42)
    logits  = torch.rand(REF_BATCH, VOCAB, device='cuda', dtype=torch.float32)
    targets = torch.randint(0, VOCAB, (REF_BATCH,), device='cuda', dtype=torch.int64)
    losses  = F.cross_entropy(logits, targets, reduction='none')
    DATA_DIR.mkdir(exist_ok=True)
    logits.cpu().numpy().tofile(DATA_DIR / "sparsece_logits.bin")
    targets.float().cpu().numpy().tofile(DATA_DIR / "sparsece_targets.bin")
    losses.cpu().numpy().tofile(DATA_DIR / "sparsece_losses.bin")
    print(f"Saved sparseCE reference data (BATCH={REF_BATCH}, VOCAB={VOCAB}) to {DATA_DIR}/\n")


if __name__ == '__main__':
    save_reference()
    main()
