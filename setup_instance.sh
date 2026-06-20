#!/bin/bash
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

echo "========================================="
echo "  GPU Tools Setup Script (B300 / Ubuntu 24.04)"
echo "========================================="

# ── 1. CUDA repo ─────────────────────────────────────────────────────────────
echo ""
echo ">>> [1/5] Adding CUDA repository..."

wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
  || err "Failed to download cuda-keyring"

dpkg -i cuda-keyring_1.1-1_all.deb || warn "dpkg install had warnings (conflict likely — fixing)"

# Remove conflicting repo list entries left by previous keyring installs
rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list

# Re-add the repo source pointing to the correct keyring
echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" \
  > /etc/apt/sources.list.d/cuda.list

apt-get update -qq || err "apt-get update failed"
ok "CUDA repo added"

# ── 2. nsys ───────────────────────────────────────────────────────────────────
echo ""
echo ">>> [2/5] Installing Nsight Systems (nsys)..."

apt-get install -y nsight-systems-2026.1.3 || err "Failed to install nsight-systems-2026.1.3"

# nsys is auto-linked to /usr/local/bin/nsys by the package — verify
if command -v nsys &>/dev/null; then
  ok "nsys installed: $(nsys --version 2>&1 | head -1)"
else
  err "nsys not found in PATH after install"
fi

# ── 3. ncu ────────────────────────────────────────────────────────────────────
echo ""
echo ">>> [3/5] Installing Nsight Compute (ncu)..."

apt-get install -y nsight-compute-2026.2.0 || err "Failed to install nsight-compute-2026.2.0"

# ncu binary is not auto-linked — symlink it
NCU_BIN="/opt/nvidia/nsight-compute/2026.2.0/ncu"
if [ -f "$NCU_BIN" ]; then
  ln -sf "$NCU_BIN" /usr/local/bin/ncu
  ok "ncu installed: $(ncu --version 2>&1 | head -1)"
else
  err "ncu binary not found at $NCU_BIN"
fi

# ── 4. ncu permission check ───────────────────────────────────────────────────
echo ""
echo ">>> [4/5] Checking ncu profiling permissions..."

PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "unreadable")
echo "    perf_event_paranoid = $PARANOID"

if [ "$PARANOID" = "unreadable" ]; then
  warn "/proc/sys/kernel/perf_event_paranoid is unreadable — likely a restricted container"
elif [ "$PARANOID" -le 0 ]; then
  ok "perf_event_paranoid=$PARANOID — GPU counters should be accessible"
else
  warn "perf_event_paranoid=$PARANOID — attempting to lower it..."
  echo 0 > /proc/sys/kernel/perf_event_paranoid 2>/dev/null && \
    ok "Set perf_event_paranoid to 0" || \
    warn "Could not set perf_event_paranoid (read-only fs — container lacks SYS_ADMIN cap)"
  echo ""
  echo "    To fix ncu counter permissions, re-provision your Vast.ai instance with:"
  echo "      --cap-add SYS_ADMIN  or  privileged mode enabled"
  echo "    Until then, use: ncu --set basic  (some metrics will be unavailable)"
fi

# Also fix libnvidia-ml symlink needed for building CUDA code against NVML
if [ ! -f /usr/lib/x86_64-linux-gnu/libnvidia-ml.so ]; then
  if [ -f /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 ]; then
    ln -s /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-ml.so
    ok "Created libnvidia-ml.so symlink"
  else
    warn "libnvidia-ml.so.1 not found — skipping symlink"
  fi
else
  ok "libnvidia-ml.so symlink already exists"
fi

# ── 5. Python packages ────────────────────────────────────────────────────────
echo ""
echo ">>> [5/5] Installing numpy and torch..."

# Detect venv
if [ -f /venv/main/bin/pip ]; then
  PIP=/venv/main/bin/pip
  ok "Using venv pip: $PIP"
else
  PIP="pip"
  warn "No venv found at /venv/main — using system pip"
fi

$PIP install -q numpy || err "Failed to install numpy"
ok "numpy installed"

$PIP install -q torch --index-url https://download.pytorch.org/whl/cu128 || err "Failed to install torch"
ok "torch installed"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================="
echo "  Setup Complete — Summary"
echo "========================================="
echo "  nsys:  $(nsys --version 2>&1 | head -1)"
echo "  ncu:   $(ncu --version 2>&1 | head -1)"
echo "  numpy: $($PIP show numpy 2>/dev/null | grep Version)"
echo "  torch: $($PIP show torch 2>/dev/null | grep Version)"
echo ""
echo "  ncu permission status: perf_event_paranoid=$PARANOID"
if [ "$PARANOID" -gt 0 ] 2>/dev/null; then
  echo "  ⚠  Run 'ncu --set basic' until container privileges are fixed"
fi
echo "========================================="
