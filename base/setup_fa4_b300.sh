#!/usr/bin/env bash
# Setup FlashAttention-4 on a fresh datacenter-Blackwell (B200 sm_100 / B300 sm_103) box.
#
# NOTE: This installs FlashAttention-4 (flash_attn/cute, CuTeDSL). FlashAttention-3 is
# Hopper-only (sm_90a) and CANNOT target Blackwell -- FA-4 is the optimized GQA kernel
# for B200/B300. Run this on the Vast instance, not locally.
set -euo pipefail

echo "==> GPU / driver / CUDA"
nvidia-smi
python3 - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| cap", torch.cuda.get_device_capability(), "|", torch.cuda.get_device_name())
cap = torch.cuda.get_device_capability()
sm = cap[0]*10 + cap[1]
assert sm in (100, 103), f"Expected datacenter Blackwell (sm_100/sm_103), got sm_{sm}. FA-4's optimized path needs B200/B300."
print(f"OK: sm_{sm} -> FA-4 has an optimized GQA path for this arch.")
PY

# ninja makes the build 3-5 min instead of ~2h
echo "==> ninja"
pip uninstall -y ninja >/dev/null 2>&1 || true
pip install -U ninja

echo "==> clone flash-attention"
cd "${WORK:-$HOME}"
[ -d flash-attention ] || git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention

# Pick the CUDA extra matching the box's toolkit (cu13 for CUDA 13.x, otherwise the cu12 default).
CUDA_MAJOR="$(python3 -c 'import torch;print(torch.version.cuda.split(".")[0])')"
echo "==> install FA-4 (flash_attn/cute), CUDA major = ${CUDA_MAJOR}"
if [ "$CUDA_MAJOR" = "13" ]; then
  pip install -e "flash_attn/cute[dev,cu13]"
else
  pip install -e "flash_attn/cute[dev]"
fi

echo "==> smoke test: locate FA-4 GQA entrypoint"
python3 - <<'PY'
import importlib, inspect
cands = ["flash_attn.cute.interface", "flash_attn.cute", "flash_attn_interface", "flash_attn"]
for m in cands:
    try:
        mod = importlib.import_module(m)
    except Exception as e:
        print(f"  {m}: import failed ({type(e).__name__})"); continue
    fns = [n for n in dir(mod) if "flash_attn" in n and n.endswith("func")]
    print(f"  {m}: {fns}")
PY
echo "==> done. Run: python3 bench_fa4_gqa.py"
