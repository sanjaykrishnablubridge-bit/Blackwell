"""
Reference generator + precision/latency comparison for the CAUSAL GQA kernel
(src/SM103/attention/GQA_sm103_causal.cu).

Same setup as baseline_gqa.py (Grouped-Query Attention: Hq query heads share Hkv
key/value heads, G = Hq/Hkv query heads per kv head), but with a causal mask:
query row i may only attend to key positions j <= i.

Writes to SEPARATE gqa_causal_*.bin files so the non-causal gqa_*.bin reference
used by GQA_sm103.cu is left untouched:
  - gqa_causal_q.bin           [B, Hq,  S, D]
  - gqa_causal_k.bin / _v.bin  [B, Hkv, S, D]
  - gqa_causal_o.bin           causal bf16 SDPA output (apples-to-apples target)
  - gqa_causal_lse.bin         fp64 ground-truth log-sum-exp of the (masked) logits

All saved as float32 (bf16-rounded inputs widened back to fp32) so the CUDA side
reuses loadBin() and both sides start from identical input bits.

Dimensions MUST match GQA_sm103_causal.cu's main(): B=8, Hq=12, Hkv=4, S=4096, D=64
(matching GQA_sm103.cu's current config, NOT this file's un-causal sibling's stale
B=16/SM120 docstring).
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import math

B, Hq, Hkv, S, D = 8, 12, 4, 4096, 64
G = Hq // Hkv

DATA_DIR = Path(__file__).parent.parent / "data"


def fp64_reference_causal(q, k, v, scale):
    """Double-precision causal ground truth + log-sum-exp, one head at a time.

    The full [B, Hq, S, S] score tensor would be huge in fp64, so we loop over
    (b, hq) and keep only one [S, S] block live at a time. Inputs are
    [B, Hq, S, D] (q) and [B, Hkv, S, D] (k, v); GQA routing is hkv = hq // G.
    """
    out = torch.empty(B, Hq, S, D, dtype=torch.float32)
    lse = torch.empty(B, Hq, S,    dtype=torch.float32)
    causal_mask = torch.triu(
        torch.ones(S, S, dtype=torch.bool, device=q.device), diagonal=1
    )  # True where j > i (disallowed)
    for b in range(B):
        for hq in range(Hq):
            hkv = hq // G
            q64 = q[b, hq ].double()                       # [S, D]
            k64 = k[b, hkv].double()                       # [S, D]
            v64 = v[b, hkv].double()                       # [S, D]
            scores = (q64 @ k64.transpose(-2, -1)) * scale  # [S, S]
            scores = scores.masked_fill(causal_mask, float("-inf"))
            lse[b, hq] = torch.logsumexp(scores, dim=-1).float().cpu()
            w = torch.softmax(scores, dim=-1)
            out[b, hq] = (w @ v64).float().cpu()
    return out, lse


def sdpa_gqa_causal(q, k, v, scale):
    """PyTorch causal GQA via scaled_dot_product_attention (enable_gqa expands KV heads)."""
    return F.scaled_dot_product_attention(
        q, k, v, scale=scale, enable_gqa=True, is_causal=True
    )


def report(name, ref, got):
    diff = (ref.double() - got.double()).abs()
    rel  = diff / ref.double().abs().clamp_min(1e-5)
    print(f"  {name:<40}  max_abs={diff.max().item():.3e}  "
          f"mean_abs={diff.mean().item():.3e}  max_rel={rel.max().item():.3e}")


def save_reference():
    DATA_DIR.mkdir(exist_ok=True)
    torch.manual_seed(42)

    # Generate fp32 in [0,1) to match the CUDA initPtr range, then round to bf16
    # so both sides begin from identical input bits.
    q = torch.rand(B, Hq,  S, D, device="cuda", dtype=torch.float32).bfloat16()
    k = torch.rand(B, Hkv, S, D, device="cuda", dtype=torch.float32).bfloat16()
    v = torch.rand(B, Hkv, S, D, device="cuda", dtype=torch.float32).bfloat16()
    scale = 1.0 / math.sqrt(D)

    out_bf16 = sdpa_gqa_causal(q, k, v, scale)                    # bf16 reference (matching precision)
    out_fp64, lse_fp64 = fp64_reference_causal(q, k, v, scale)    # fp64 ground truth

    print(f"=== Causal precision vs FP64 ground truth  ({B}x{Hq}x{S}x{D}, GQA G={G}) ===")
    report("SDPA bf16 causal (enable_gqa)", out_fp64, out_bf16.float().cpu())

    # Save as float32 (bf16 widened) so the CUDA side reuses loadBin().
    q.float().cpu().numpy().tofile(DATA_DIR / "gqa_causal_q.bin")
    k.float().cpu().numpy().tofile(DATA_DIR / "gqa_causal_k.bin")
    v.float().cpu().numpy().tofile(DATA_DIR / "gqa_causal_v.bin")
    out_bf16.float().cpu().numpy().tofile(DATA_DIR / "gqa_causal_o.bin")
    lse_fp64.cpu().numpy().tofile(DATA_DIR / "gqa_causal_lse.bin")
    print(f"Saved causal GQA reference data (B={B}, Hq={Hq}, Hkv={Hkv}, S={S}, D={D}) "
          f"to {DATA_DIR}/\n")


def main():
    torch.manual_seed(42)
    q = torch.rand(B, Hq,  S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.rand(B, Hkv, S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.rand(B, Hkv, S, D, device="cuda", dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(D)

    # compile so Inductor can fuse/pick the best backend
    compiled = torch.compile(lambda q, k, v: sdpa_gqa_causal(q, k, v, scale))
    _ = compiled(q, k, v)
    torch.cuda.synchronize()

    print(f"=== Latency  ({B}x{Hq}x{S}x{D}, GQA G={G}, bf16, CAUSAL) ===")
    fns = {
        "SDPA bf16 causal (enable_gqa)": lambda: sdpa_gqa_causal(q, k, v, scale),
        "SDPA bf16 causal (compiled)  ": lambda: compiled(q, k, v),
    }
    # Causal attention FLOPs: roughly HALF of the full 4*B*Hq*S*S*D (upper-triangular
    # half of the score matrix is skipped); report the exact triangular count so the
    # TFLOP/s figure is comparable to the CUDA side's causal tile-count accounting.
    flops = 2 * B * Hq * S * S * D  # 4*.../2, i.e. only the lower-triangular half
    header = f"{'Kernel':<30} {'median ms':>10}  {'p5 ms':>8}  {'p95 ms':>8}  {'TFLOP/s':>10}"
    print(header)
    print("-" * len(header))
    import triton.testing
    for name, fn in fns.items():
        med, p5, p95 = triton.testing.do_bench(fn, warmup=25, rep=100,
                                                quantiles=[0.5, 0.05, 0.95])
        tflops = flops / (med * 1e-3) / 1e12
        print(f"{name:<30} {med:>10.4f}  {p5:>8.4f}  {p95:>8.4f}  {tflops:>10.2f}")


if __name__ == "__main__":
    save_reference()
    main()
