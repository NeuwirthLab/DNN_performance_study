#!/bin/bash -l
#SBATCH --job-name=rn50_amd_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2                 # adjust: e.g. gpu:mi300x:2 on some sites
#SBATCH --time=00:30:00
#SBATCH --output=smoke_amd_%j.out
#SBATCH --error=smoke_amd_%j.err
#SBATCH --export=ALL

# ============================================================================
# AMD/ROCm smoke test — mirror of smoke_test_helma.slurm for the HIP port.
#
# Steps:
#   1. Filesystem check
#   2. Module load (rocm, miopen, rccl, cmake)
#   3. RCCL detection
#   4. GPU detection (rocm-smi, infer gfx arch)
#   5. hipify any un-ported .cu files (idempotent)
#   6. cmake + build of train_multigpu_tiny via hip/CMakeLists.txt
#   7. Run 2 epochs on Tiny ImageNet with 2 GPUs
# ============================================================================

echo "========================================================"
echo " ResNet-50 AMD/HIP Smoke Test"
echo " Job   : $SLURM_JOB_ID"
echo " Node  : $SLURM_NODELIST"
echo " Start : $(date)"
echo "========================================================"

# ── EDIT THESE TWO PATHS ────────────────────────────────────────────────────
WORKSPACE_BASE="${WORKSPACE_BASE:-$HOME/workspace}"
PROJECT_DIR="${PROJECT_DIR:-$WORKSPACE_BASE/resnet50_hip_runtime}"
# ────────────────────────────────────────────────────────────────────────────

# ── 1. Filesystem ───────────────────────────────────────────────────────────
echo "[1/7] Filesystem check"
if [ ! -d "$PROJECT_DIR" ]; then
    echo "  FAIL: $PROJECT_DIR not found"; exit 1
fi
echo "  ✓ project dir: $PROJECT_DIR"
echo "  ✓ TMPDIR: $TMPDIR  (free: $(df -h $TMPDIR | tail -1 | awk '{print $4}'))"

# ── 2. Modules ──────────────────────────────────────────────────────────────
echo ""
echo "[2/7] Module loading"
module purge
# EDIT module names to match your AMD cluster; these are common defaults.
module load rocm          || { echo "  FAIL: could not load rocm module"; exit 1; }
module load miopen        2>/dev/null || true   # often bundled inside rocm
module load rccl          2>/dev/null || true   # often bundled inside rocm
module load cmake

echo "  ✓ hipcc : $(hipcc --version | head -1)"
echo "  ✓ cmake : $(cmake --version | head -1 | awk '{print $3}')"
echo "  ✓ ROCM_PATH = ${ROCM_PATH:-/opt/rocm}"

# ── 3. RCCL detection ───────────────────────────────────────────────────────
echo ""
echo "[3/7] RCCL detection"
RCCL_ROOT=""
for candidate in "$ROCM_PATH" "/opt/rocm" "$EBROOTRCCL"; do
    [ -n "$candidate" ] && [ -f "$candidate/include/rccl/rccl.h" ] && \
        RCCL_ROOT="$candidate" && break
done
if [ -z "$RCCL_ROOT" ]; then
    echo "  FAIL: rccl.h not found"; exit 1
fi
echo "  ✓ RCCL : $RCCL_ROOT"

# ── 4. GPU detection ────────────────────────────────────────────────────────
echo ""
echo "[4/7] GPU detection"
GPU_COUNT=$(rocm-smi --showproductname | grep -c "Card series" || echo "?")
GPU_INFO=$(rocm-smi --showproductname | head -20)
echo "$GPU_INFO"

# Derive gfx arch. Use rocminfo to get exact gfx target.
GFX_ARCH=$(rocminfo 2>/dev/null | awk '/gfx[0-9a-f]+/{for(i=1;i<=NF;i++) if($i ~ /gfx/){print $i; exit}}')
[ -z "$GFX_ARCH" ] && GFX_ARCH="gfx942"
echo "  ✓ GPU arch : $GFX_ARCH"

# ── 5. Hipify any lingering pure-CUDA tokens ────────────────────────────────
echo ""
echo "[5/7] hipify sweep (idempotent)"
cd "$PROJECT_DIR"
if command -v hipify-perl >/dev/null 2>&1; then
    bash scripts/port_to_hip.sh || { echo "  FAIL: hipify script"; exit 1; }
else
    echo "  skip: hipify-perl not on PATH (our shims may cover it; continuing)"
fi

# ── 6. Build ────────────────────────────────────────────────────────────────
echo ""
echo "[6/7] Build (train_multigpu_tiny via hip/)"
BUILD_DIR="$TMPDIR/smoke_build_hip"
rm -rf "$BUILD_DIR" && mkdir -p "$BUILD_DIR"

cmake -S "$PROJECT_DIR/hip" -B "$BUILD_DIR" \
      -DCMAKE_HIP_ARCHITECTURES="$GFX_ARCH" \
      -DCMAKE_BUILD_TYPE=Release \
      -DUSE_NCCL=ON \
      > "$BUILD_DIR/cmake.log" 2>&1
if [ $? -ne 0 ]; then
    echo "  FAIL: cmake — last 30 lines:"; tail -30 "$BUILD_DIR/cmake.log"; exit 1
fi

cmake --build "$BUILD_DIR" -j8 --target train_multigpu_tiny \
      > "$BUILD_DIR/build.log" 2>&1
if [ $? -ne 0 ]; then
    echo "  FAIL: build — last 40 lines:"; tail -40 "$BUILD_DIR/build.log"; exit 1
fi
echo "  ✓ built: $BUILD_DIR/train_multigpu_tiny"

# ── 7. Training ─────────────────────────────────────────────────────────────
echo ""
echo "[7/7] Training (2 epochs, Tiny ImageNet, 2 GPUs)"
TINY_DIR="$PROJECT_DIR/tiny-imagenet-200"
if [ ! -d "$TINY_DIR" ]; then
    echo "  FAIL: dataset missing at $TINY_DIR"
    echo "  wget http://cs231n.stanford.edu/tiny-imagenet-200.zip && unzip ..."
    exit 1
fi

echo "  Starting at $(date)"
"$BUILD_DIR/train_multigpu_tiny" "$TINY_DIR" 128 2 0.1
STATUS=$?
echo "  Finished at $(date) — exit $STATUS"
[ $STATUS -eq 0 ] && echo "  ✓ AMD/HIP SMOKE TEST PASSED" \
                  || { echo "  ✗ FAILED"; exit $STATUS; }
