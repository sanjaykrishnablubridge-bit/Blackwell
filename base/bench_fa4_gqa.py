#!/usr/bin/env python3
"""Benchmark FlashAttention-4's Grouped-Query Attention (GQA) kernel on Blackwell (B200/B300).

FA-4 is the optimized attention kernel for datacenter Blackwell (sm_100/sm_103).
FlashAttention-3 is Hopper-only and is intentionally NOT benchmarked here -- it has no
Blackwell kernel. Baseline for comparison is PyTorch SDPA with native GQA (enable_gqa).

Usage:
    python3 bench_fa4_gqa.py                      # default GQA sweep, bf16, causal
    python3 bench_fa4_gqa.py --dtype fp16 --no-causal
    python3 bench_fa4_gqa.py --seqlens 4096,8192,16384 --heads 64 --kv-heads 8 --hdim 128
    # profile a single point with ncu/nsys:
    python3 bench_fa4_gqa.py --seqlens 8192 --heads 32 --kv-heads 8 --iters 1 --no-sdpa
"""
import argparse, importlib, time
import torch
import torch.nn.functional as F


def find_fa4():
    """Locate the FA-4 forward entrypoint across known module layouts. Returns (fn, name)."""
    for m in ("flash_attn.cute.interface", "flash_attn.cute",
              "flash_attn_interface", "flash_attn"):
        try:
            mod = importlib.import_module(m)
        except Exception:
            continue
        for fn in ("flash_attn_func",):
            if hasattr(mod, fn):
                return getattr(mod, fn), f"{m}.{fn}"
    raise ImportError(
        "Could not find flash_attn_func. Run setup_fa4_b300.sh and check its smoke-test "
        "output, then set the import here to the module it reported.")


def bench(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms/iter


def flops(b, hq, s, d, causal):
    # QK^T + PV: 2 matmuls of 2*b*hq*s*s*d each. Causal ~halves work.
    f = 4 * b * hq * s * s * d
    return f * 0.5 if causal else f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seqlens", default="2048,4096,8192,16384")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--heads", type=int, default=32, help="query heads")
    p.add_argument("--kv-heads", type=int, default=8, help="KV heads (GQA group = heads/kv-heads)")
    p.add_argument("--hdim", type=int, default=128)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--no-causal", dest="causal", action="store_false")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--no-sdpa", dest="sdpa", action="store_false", help="skip SDPA baseline")
    args = p.parse_args()

    assert args.heads % args.kv_heads == 0, "heads must be divisible by kv-heads"
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    dev = "cuda"
    fa4, fa4_name = find_fa4()
    g = args.heads // args.kv_heads
    print(f"FA-4 entrypoint: {fa4_name}")
    print(f"GQA: {args.heads} q-heads / {args.kv_heads} kv-heads (group={g}), "
          f"hdim={args.hdim}, {args.dtype}, causal={args.causal}, batch={args.batch}\n")
    hdr = f"{'seqlen':>8} {'FA4 ms':>9} {'FA4 TFLOP/s':>12} {'SDPA ms':>9} {'SDPA TFLOP/s':>13} {'speedup':>8}"
    print(hdr); print("-" * len(hdr))

    for s in [int(x) for x in args.seqlens.split(",")]:
        b, hq, hkv, d = args.batch, args.heads, args.kv_heads, args.hdim
        # FA-4 / flash layout: (batch, seqlen, nheads, headdim)
        q = torch.randn(b, s, hq, d, device=dev, dtype=dt)
        k = torch.randn(b, s, hkv, d, device=dev, dtype=dt)
        v = torch.randn(b, s, hkv, d, device=dev, dtype=dt)
        fl = flops(b, hq, s, d, args.causal)

        try:
            fa_ms = bench(lambda: fa4(q, k, v, causal=args.causal), args.iters, args.warmup)
            fa_tf = fl / (fa_ms * 1e-3) / 1e12
        except Exception as e:
            print(f"{s:>8}  FA-4 failed: {type(e).__name__}: {e}")
            continue

        if args.sdpa:
            # SDPA layout: (batch, nheads, seqlen, headdim); enable_gqa broadcasts kv heads.
            qs = q.transpose(1, 2); ks = k.transpose(1, 2); vs = v.transpose(1, 2)
            sd_ms = bench(lambda: F.scaled_dot_product_attention(
                qs, ks, vs, is_causal=args.causal, enable_gqa=True), args.iters, args.warmup)
            sd_tf = fl / (sd_ms * 1e-3) / 1e12
            print(f"{s:>8} {fa_ms:>9.3f} {fa_tf:>12.1f} {sd_ms:>9.3f} {sd_tf:>13.1f} {sd_ms/fa_ms:>7.2f}x")
        else:
            print(f"{s:>8} {fa_ms:>9.3f} {fa_tf:>12.1f} {'-':>9} {'-':>13} {'-':>8}")


if __name__ == "__main__":
    main()
