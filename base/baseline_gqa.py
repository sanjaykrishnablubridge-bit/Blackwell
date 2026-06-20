"""
Reference generator + precision/latency comparison for the SM120 GQA kernel.

Grouped-Query Attention: Hq query heads share Hkv key/value heads, with
G = Hq/Hkv query heads per kv head (query head hq reads kv head hq // G).
Inputs/outputs are bf16 with fp32 accumulation, matching gqa_v1 in
src/SM120/attention/GQA_sm120.cu.

Saves (all float32 .bin so the CUDA side can reuse loadBin; the stored values are
the bf16-ROUNDED inputs widened back to float32, so the kernel and the reference
start from identical input bits):
  - gqa_q.bin                 [B, Hq,  S, D]
  - gqa_k.bin / gqa_v.bin     [B, Hkv, S, D]
  - gqa_o.bin                 bf16 SDPA output (apples-to-apples target), [B, Hq, S, D]
  - gqa_o_fp64.bin            fp64 ground-truth output,                   [B, Hq, S, D]
  - gqa_lse.bin               fp64 ground-truth log-sum-exp of the logits,[B, Hq, S]

Dimensions match GQA_sm120.cu: B=16, Hq=12, Hkv=4, S=4096, D=64.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import math
import triton
import triton.language as tl
import triton.testing

B, Hq, Hkv, S, D = 16, 12, 4, 4096, 64
G = Hq // Hkv

DATA_DIR = Path(__file__).parent.parent / "data"


def fp64_reference(q, k, v, scale):
    """Double-precision ground truth + log-sum-exp, computed one head at a time.

    The full [B, Hq, S, S] score tensor would be ~25 GB in fp64, so we loop over
    (b, hq) and keep only one [S, S] block live at a time.  Inputs are
    [B, Hq, S, D] (q) and [B, Hkv, S, D] (k, v); GQA routing is hkv = hq // G.
    """
    out = torch.empty(B, Hq, S, D, dtype=torch.float32)
    lse = torch.empty(B, Hq, S,    dtype=torch.float32)
    for b in range(B):
        for hq in range(Hq):
            hkv = hq // G
            q64 = q[b, hq ].double()                       # [S, D]
            k64 = k[b, hkv].double()                       # [S, D]
            v64 = v[b, hkv].double()                       # [S, D]
            scores = (q64 @ k64.transpose(-2, -1)) * scale  # [S, S]
            lse[b, hq] = torch.logsumexp(scores, dim=-1).float().cpu()
            w = torch.softmax(scores, dim=-1)
            out[b, hq] = (w @ v64).float().cpu()
    return out, lse


def sdpa_gqa(q, k, v, scale):
    """PyTorch GQA via scaled_dot_product_attention (enable_gqa expands KV heads)."""
    return F.scaled_dot_product_attention(q, k, v, scale=scale, enable_gqa=True)


# ── Triton flash-attention forward (GQA, non-causal) ──────────────────────────
# One program computes a BLOCK_M tile of query rows for one (batch, query-head),
# streaming the keys in BLOCK_N tiles with an ONLINE softmax (running max/sum).
# GQA routing is the same as the CUDA kernel: kv head = query head // G.

@triton.jit
def _gqa_fwd_kernel(
    Q, K, V, O,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    Hq, S, G, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
):
    start_m = tl.program_id(0)        # which BLOCK_M query tile
    off_bh  = tl.program_id(1)        # flattened (batch, query-head)
    off_b   = off_bh // Hq
    off_hq  = off_bh %  Hq
    off_hkv = off_hq // G             # GQA: shared kv head

    q_base = Q + off_b * stride_qb + off_hq  * stride_qh
    k_base = K + off_b * stride_kb + off_hkv * stride_kh
    v_base = V + off_b * stride_vb + off_hkv * stride_vh
    o_base = O + off_b * stride_ob + off_hq  * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)

    # load this block of queries once: [BLOCK_M, D]
    q = tl.load(q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                mask=offs_m[:, None] < S, other=0.0)

    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)  # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                # running denom
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)             # running Σ p·V

    for start_n in range(0, S, BLOCK_N):
        offs_n_curr = start_n + offs_n
        # K tile loaded transposed -> [D, BLOCK_N] so tl.dot(q, k) gives Q·Kᵀ
        k = tl.load(k_base + offs_n_curr[None, :] * stride_ks + offs_d[:, None] * stride_kd,
                    mask=offs_n_curr[None, :] < S, other=0.0)
        qk = tl.dot(q, k) * scale                              # [BLOCK_M, BLOCK_N]
        qk = tl.where(offs_n_curr[None, :] < S, qk, -float('inf'))

        m_ij  = tl.maximum(m_i, tl.max(qk, axis=1))            # new running max
        p     = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)                             # rescale old stats
        l_i   = l_i * alpha + tl.sum(p, axis=1)
        acc   = acc * alpha[:, None]

        v = tl.load(v_base + offs_n_curr[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                    mask=offs_n_curr[:, None] < S, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    acc = acc / l_i[:, None]
    tl.store(o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od,
             acc.to(O.dtype.element_ty), mask=offs_m[:, None] < S)


def triton_gqa(q, k, v, scale, BLOCK_M=128, BLOCK_N=64):
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    o = torch.empty_like(q)
    grid = (triton.cdiv(S, BLOCK_M), B * Hq)
    _gqa_fwd_kernel[grid](
        q, k, v, o,
        *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        Hq, S, G, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
        num_warps=4, num_stages=2,
    )
    return o


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

    out_bf16 = sdpa_gqa(q, k, v, scale)                  # bf16 reference (matching precision)
    out_fp64, lse_fp64 = fp64_reference(q, k, v, scale)  # fp64 ground truth

    print(f"=== Precision vs FP64 ground truth  ({B}x{Hq}x{S}x{D}, GQA G={G}) ===")
    report("SDPA bf16 (enable_gqa)", out_fp64, out_bf16.float().cpu())

    # Save as float32 (bf16 widened) so the CUDA side reuses loadBin().
    q.float().cpu().numpy().tofile(DATA_DIR / "gqa_q.bin")
    k.float().cpu().numpy().tofile(DATA_DIR / "gqa_k.bin")
    v.float().cpu().numpy().tofile(DATA_DIR / "gqa_v.bin")
    out_bf16.float().cpu().numpy().tofile(DATA_DIR / "gqa_o.bin")
    out_fp64.cpu().numpy().tofile(DATA_DIR / "gqa_o_fp64.bin")
    lse_fp64.cpu().numpy().tofile(DATA_DIR / "gqa_lse.bin")
    print(f"Saved GQA reference data (B={B}, Hq={Hq}, Hkv={Hkv}, S={S}, D={D}) to {DATA_DIR}/\n")


def main():
    torch.manual_seed(42)
    q = torch.rand(B, Hq,  S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.rand(B, Hkv, S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.rand(B, Hkv, S, D, device="cuda", dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(D)

    # compile so Inductor can fuse/pick the best backend
    compiled = torch.compile(lambda q, k, v: sdpa_gqa(q, k, v, scale))
    _ = compiled(q, k, v)
    torch.cuda.synchronize()

    # ── precision: Triton vs PyTorch bf16 SDPA (both bf16, fp32 accumulation) ──
    out_sdpa = sdpa_gqa(q, k, v, scale)
    out_tri  = triton_gqa(q, k, v, scale)
    diff = (out_sdpa.float() - out_tri.float()).abs()
    print(f"Triton vs SDPA bf16   max_abs={diff.max().item():.3e}  "
          f"mean_abs={diff.mean().item():.3e}\n")

    print(f"=== Latency  ({B}x{Hq}x{S}x{D}, GQA G={G}, bf16) ===")
    fns = {
        "SDPA bf16 (enable_gqa)": lambda: sdpa_gqa(q, k, v, scale),
        "SDPA bf16 (compiled)  ": lambda: compiled(q, k, v),
        "Triton flash (GQA)    ": lambda: triton_gqa(q, k, v, scale),
    }
    # attention FLOPs (algorithmic): 4 * B * Hq * S * S * D  (QKᵀ + P·V, factor 2 for MAC)
    flops = 4 * B * Hq * S * S * D
    header = f"{'Kernel':<26} {'median ms':>10}  {'p5 ms':>8}  {'p95 ms':>8}  {'TFLOP/s':>10}"
    print(header)
    print("-" * len(header))
    for name, fn in fns.items():
        med, p5, p95 = triton.testing.do_bench(fn, warmup=25, rep=100,
                                                quantiles=[0.5, 0.05, 0.95])
        tflops = flops / (med * 1e-3) / 1e12
        print(f"{name:<26} {med:>10.4f}  {p5:>8.4f}  {p95:>8.4f}  {tflops:>10.2f}")


if __name__ == "__main__":
    save_reference()
    main()
