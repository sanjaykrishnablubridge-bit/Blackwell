"""
Reference generator for the split-seqlen GQA kernel (src/SM120/attention/adhitya_GQA.cu).

Same as baseline_gqa.py, but with SEPARATE sequence lengths for Q and K/V:
  Sq  : query  sequence length  -> Q / O / LSE are [B, Hq,  Sq,  D]
  Skv : key/value sequence length -> K / V       are [B, Hkv, Skv, D]
(cross-attention / KV-cache style, where the query and kv lengths differ).

Writes to distinct gqa_split_*.bin files so the equal-length gqa_*.bin reference
used by GQA_sm120.cu is left untouched:
  - gqa_split_q.bin              [B, Hq,  Sq,  D]
  - gqa_split_k.bin / _v.bin     [B, Hkv, Skv, D]
  - gqa_split_o.bin              bf16 SDPA output (apples-to-apples target)
  - gqa_split_lse.bin            fp64 ground-truth log-sum-exp of the logits, [B, Hq, Sq]

All saved as float32 (bf16-rounded inputs widened back to fp32) so the CUDA side
reuses loadBin() and both sides start from identical input bits.

Dimensions MUST match adhitya_GQA.cu's main(): B=16, Hq=12, Hkv=4, Sq=4096, Skv=2048, D=64.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import math

B, Hq, Hkv, Sq, Skv, D = 16, 12, 4, 4096, 2048, 64
G = Hq // Hkv

DATA_DIR = Path(__file__).parent.parent / "data"


def fp64_reference(q, k, v, scale):
    """Double-precision ground truth + log-sum-exp, one (b, hq) head at a time."""
    out = torch.empty(B, Hq, Sq, D, dtype=torch.float32)
    lse = torch.empty(B, Hq, Sq,    dtype=torch.float32)
    for b in range(B):
        for hq in range(Hq):
            hkv = hq // G
            q64 = q[b, hq ].double()                          # [Sq,  D]
            k64 = k[b, hkv].double()                          # [Skv, D]
            v64 = v[b, hkv].double()                          # [Skv, D]
            scores = (q64 @ k64.transpose(-2, -1)) * scale    # [Sq, Skv]
            lse[b, hq] = torch.logsumexp(scores, dim=-1).float().cpu()
            w = torch.softmax(scores, dim=-1)
            out[b, hq] = (w @ v64).float().cpu()
    return out, lse


def sdpa_gqa(q, k, v, scale):
    """PyTorch GQA via SDPA (enable_gqa expands KV heads; non-causal cross-attn)."""
    return F.scaled_dot_product_attention(q, k, v, scale=scale, enable_gqa=True)


def report(name, ref, got):
    diff = (ref.double() - got.double()).abs()
    rel  = diff / ref.double().abs().clamp_min(1e-5)
    print(f"  {name:<40}  max_abs={diff.max().item():.3e}  "
          f"mean_abs={diff.mean().item():.3e}  max_rel={rel.max().item():.3e}")


def save_reference():
    DATA_DIR.mkdir(exist_ok=True)
    torch.manual_seed(42)

    # fp32 in [0,1) to match the CUDA initPtr range, rounded to bf16.
    q = torch.rand(B, Hq,  Sq,  D, device="cuda", dtype=torch.float32).bfloat16()
    k = torch.rand(B, Hkv, Skv, D, device="cuda", dtype=torch.float32).bfloat16()
    v = torch.rand(B, Hkv, Skv, D, device="cuda", dtype=torch.float32).bfloat16()
    scale = 1.0 / math.sqrt(D)

    out_bf16 = sdpa_gqa(q, k, v, scale)                  # bf16 reference (matching precision)
    out_fp64, lse_fp64 = fp64_reference(q, k, v, scale)  # fp64 ground truth

    print(f"=== Precision vs FP64 ground truth  ({B}x{Hq}, Sq={Sq}, Skv={Skv}, D={D}, GQA G={G}) ===")
    report("SDPA bf16 (enable_gqa)", out_fp64, out_bf16.float().cpu())

    q.float().cpu().numpy().tofile(DATA_DIR / "gqa_split_q.bin")
    k.float().cpu().numpy().tofile(DATA_DIR / "gqa_split_k.bin")
    v.float().cpu().numpy().tofile(DATA_DIR / "gqa_split_v.bin")
    out_bf16.float().cpu().numpy().tofile(DATA_DIR / "gqa_split_o.bin")
    lse_fp64.cpu().numpy().tofile(DATA_DIR / "gqa_split_lse.bin")
    print(f"Saved split-seqlen GQA reference (B={B}, Hq={Hq}, Hkv={Hkv}, "
          f"Sq={Sq}, Skv={Skv}, D={D}) to {DATA_DIR}/\n")


if __name__ == "__main__":
    save_reference()
