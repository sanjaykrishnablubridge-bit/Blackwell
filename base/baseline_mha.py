"""
Reference generator + precision/latency comparison for the SM120 MHA kernel.

The high-level F.scaled_dot_product_attention silently falls back to MATH on
Blackwell + FP32 (warning: "Memory Efficient attention has been runtime
disabled").  We bypass that and call torch.ops.aten._efficient_attention_forward
directly — that's the actual cutlass FMHA kernel whose SASS we analyzed.

Saves:
  - mha_q.bin / mha_k.bin / mha_v.bin    in [B, H, S, D]
  - mha_out.bin                          cutlass FMHA output, [B, H, S, D]
  - mha_out_fp64.bin                     FP64 ground truth, [B, H, S, D]

Dimensions match script.cpp's GPT-2 124M config: B=16, H=12, S=1024, D=64.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import math
import triton.testing

B, H, S, D = 16, 12, 1024, 64

DATA_DIR = Path(__file__).parent.parent / "data"


def fp64_attention(q, k, v, scale):
    """Ground truth in double precision. Input layout [B, H, S, D]."""
    q64 = q.double(); k64 = k.double(); v64 = v.double()
    scores  = torch.matmul(q64, k64.transpose(-2, -1)) * scale
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v64).float()


def cutlass_fmha(q, k, v, scale):
    """Direct call to PyTorch's _efficient_attention_forward (cutlass FMHA).

    Layout: this op wants [B, S, H, D], not [B, H, S, D].  We transpose in/out.
    """
    q_bshd = q.transpose(1, 2).contiguous()
    k_bshd = k.transpose(1, 2).contiguous()
    v_bshd = v.transpose(1, 2).contiguous()
    res = torch.ops.aten._efficient_attention_forward(
        q_bshd, k_bshd, v_bshd,
        None, None, None, None, None,   # bias, cu_seqlens_*, max_seqlen_*
        0.0,                             # dropout_p
        0,                               # custom_mask_type (0 = no mask)
        False,                           # compute_log_sumexp
        scale=scale,
    )
    out_bshd = res[0]                                # [B, S, H, D]
    return out_bshd.transpose(1, 2).contiguous()     # back to [B, H, S, D]


def report(name, ref, got):
    diff = (ref.double() - got.double()).abs()
    rel  = diff / ref.double().abs().clamp_min(1e-5)
    print(f"  {name:<40}  max_abs={diff.max().item():.3e}  "
          f"mean_abs={diff.mean().item():.3e}  max_rel={rel.max().item():.3e}")


def main():
    DATA_DIR.mkdir(exist_ok=True)

    torch.manual_seed(42)
    q = torch.rand(B, H, S, D, device="cuda", dtype=torch.float32) - 0.5
    k = torch.rand(B, H, S, D, device="cuda", dtype=torch.float32) - 0.5
    v = torch.rand(B, H, S, D, device="cuda", dtype=torch.float32) - 0.5
    scale = 1.0 / math.sqrt(D)

    # Confirm which backend PyTorch's auto-dispatch picks
    backends = {0: 'math', 1: 'flash', 2: 'mem-eff', 3: 'cudnn'}
    pick = torch._fused_sdp_choice(q, k, v)
    print(f"PyTorch auto-dispatch picks: {pick} ({backends.get(pick, '?')})")

    # === Compute three outputs ===
    out_fp64    = fp64_attention(q, k, v, scale)
    out_cutlass = cutlass_fmha(q, k, v, scale)
    out_sdpa    = F.scaled_dot_product_attention(q, k, v, is_causal=False)

    print(f"\n=== Precision vs FP64 ground truth  ({B}x{H}x{S}x{D}) ===")
    report("cutlass FMHA  (mem-eff, direct call)", out_fp64, out_cutlass)
    report("F.scaled_dot_product_attention      ", out_fp64, out_sdpa)
    report("cutlass FMHA  vs  F.SDPA            ", out_cutlass, out_sdpa)

    # Save Q/K/V plus BOTH refs.  mha_out.bin is the cutlass FMHA output
    # (the real apples-to-apples target for our kernel).
    q.contiguous().cpu().numpy().tofile(DATA_DIR / "mha_q.bin")
    k.contiguous().cpu().numpy().tofile(DATA_DIR / "mha_k.bin")
    v.contiguous().cpu().numpy().tofile(DATA_DIR / "mha_v.bin")
    out_cutlass.contiguous().cpu().numpy().tofile(DATA_DIR / "mha_out.bin")
    out_fp64.contiguous().cpu().numpy().tofile(DATA_DIR / "mha_out_fp64.bin")

    # === Latency ===
    print(f"\n=== Latency  ({B}x{H}x{S}x{D}, FP32 inputs) ===")
    fns = {
        "cutlass FMHA (direct)            ": lambda: cutlass_fmha(q, k, v, scale),
        "F.scaled_dot_product_attention   ": lambda: F.scaled_dot_product_attention(q, k, v, is_causal=False),
    }
    for name, fn in fns.items():
        med, p5, p95 = triton.testing.do_bench(fn, warmup=25, rep=100,
                                                quantiles=[0.5, 0.05, 0.95])
        print(f"  {name}  median={med:7.3f} ms   p5={p5:6.3f}   p95={p95:6.3f}")


if __name__ == "__main__":
    main()
