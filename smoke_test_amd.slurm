#!/bin/bash -l
#SBATCH --job-name=smoke_test_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --account=l0003099
#SBATCH --partition=gpu:mi300x:2
#SBATCH --gres=gpu:mi300x:2
#SBATCH --time=02:00:00
#SBATCH --output=smoke_test_results/smoke_amd_%j.out
#SBATCH --error=smoke_test_results/smoke_amd_%j.err
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
WORKSPACE_BASE="${WORKSPACE_BASE:-$HPC_SCRATCH}"
PROJECT_DIR="${PROJECT_DIR:-$WORKSPACE_BASE/resent50/performance-analysis-dl-models-resnet50}"
# ────────────────────────────────────────────────────────────────────────────

# ── 1. Filesystem ───────────────────────────────────────────────────────────
echo "[1/7] Filesystem check"
if [ ! -d "$PROJECT_DIR" ]; then
    echo "  FAIL: $PROJECT_DIR not found"; exit 1
fi
echo "  ✓ project dir: $PROJECT_DIR"
echo "  ✓ TMPDIR: $TMPDIR  (free: $(df -h $TMPDIR | tail -1 | awk '{print $4}'))"

# ── 2. Modules and Environment ────────────────────────────────────────────────
echo ""
echo "[2/7] Module loading"
#module purge
#module load rocm          2>/dev/null || true
#module load rccl          2>/dev/null || true
module load cmake/3.30.5-dy3s

export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"

#echo "  ✓ hipcc : $(hipcc --version | head -1)"
echo "  ✓ cmake : $(cmake --version | head -1 | awk '{print $3}')"
echo "  ✓ ROCM_PATH = ${ROCM_PATH}"

# ── 3. RCCL detection ───────────────────────────────────────────────────────
echo ""
echo "[3/7] RCCL detection"
RCCL_ROOT=""
for candidate in "$ROCM_PATH" "/opt/rocm"; do
    if [ -n "$candidate" ] && { [ -f "$candidate/include/rccl/rccl.h" ] || [ -f "$candidate/include/rccl.h" ]; }; then
        RCCL_ROOT="$candidate"
        break
    fi
done

if [ -z "$RCCL_ROOT" ]; then
    echo "  FAIL: rccl.h not found under \$ROCM_PATH/include/rccl or /opt/rocm/include/rccl"
    exit 1
fi

if [ ! -d "$RCCL_ROOT/lib" ] && [ ! -d "$RCCL_ROOT/lib64" ]; then
    echo "  WARN: RCCL headers found, but no lib/lib64 directory detected under $RCCL_ROOT"
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

# ── 5. Hipify sweep — DISABLED ───────────────────────────────────────────────
# Our approach keeps source files in CUDA naming and uses gpu_compat.h /
# dnn_compat.h shims to remap at compile time via -DUSE_HIP. Running
# hipify-perl again would undo this and break the build.
echo ""
echo "[5/7] hipify sweep — SKIPPED (using compat-shim approach)"

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
TINY_DIR="$PROJECT_DIR/tiny-imagenet-200"
if [ ! -d "$TINY_DIR" ]; then
    echo "  FAIL: dataset missing at $TINY_DIR"
    echo "  wget http://cs231n.stanford.edu/tiny-imagenet-200.zip && unzip ..."
    exit 1
fi

GPU_COUNT=$(rocm-smi --showproductname 2>/dev/null | grep -ci "Card Series" || true)
if [ -z "$GPU_COUNT" ] || [ "$GPU_COUNT" -eq 0 ]; then
    GPU_COUNT=${SLURM_GPUS_PER_NODE:-2}
fi

GLOBAL_BATCH="${TINY_GLOBAL_BATCH:-256}"
EPOCHS="${TINY_EPOCHS:-20}"
LR="${TINY_LR:-0.1}"
LOCAL_BATCH=$((GLOBAL_BATCH / GPU_COUNT))

echo ""
echo "=== Training Configuration ==="
echo "  - GPUs            : $GPU_COUNT"
echo "  - Global batch    : $GLOBAL_BATCH"
echo "  - Local batch     : $LOCAL_BATCH"
echo "  - Epochs          : $EPOCHS"
echo "  - Learning rate   : $LR"
echo "  - Data location   : $TINY_DIR"
echo "  - Checkpoint every: 15 epochs"
echo ""
echo "[7/7] Training (${EPOCHS} epochs, Tiny ImageNet, ${GPU_COUNT} GPUs)"
echo "  Starting at $(date)"

# Ensure Img/sec column is never wrapped in the SLURM log file
export COLUMNS=200

"$BUILD_DIR/train_multigpu_tiny" "$TINY_DIR" "$GLOBAL_BATCH" "$EPOCHS" "$LR"
STATUS=$?

echo ""
echo "=== Job Summary ==="
echo "  Finished at $(date)"
if [ $STATUS -eq 0 ]; then
    echo "  ✓ AMD/HIP TRAINING COMPLETE"
    echo "  Checkpoints saved to: $PROJECT_DIR/checkpoints/"
    echo "  Job ID: $SLURM_JOB_ID"
else
    echo "  ✗ Training FAILED (exit $STATUS)"
    exit $STATUS
fi
echo "=== Job Complete ==="
