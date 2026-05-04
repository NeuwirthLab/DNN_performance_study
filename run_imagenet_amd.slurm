#!/bin/bash -l
#SBATCH --job-name=imagenet_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --account=l0003099
#SBATCH --partition=gpu:mi300x:2
#SBATCH --gres=gpu:mi300x:2
#SBATCH --time=24:00:00
#SBATCH --output=imagenet_results/imagenet_amd_%j.out
#SBATCH --error=imagenet_results/imagenet_amd_%j.err
# ============================================================================
# AMD/ROCm ImageNet-1K training — full DDP run on 2× MI300X.
#
# Uses the CPU multi-threaded loader (mt_image_loader.h + prefetch_loader.h) —
# no rocAL required. Once rocAL becomes available we will add a USE_ROCAL
# branch alongside the existing USE_DALI branch in train_multigpu.cu.
#
# Mirrors smoke_test_amd.slurm structure; differs only in: build target,
# data dir, hyperparams, and time limit.
# ============================================================================

mkdir -p imagenet_results

echo "========================================================"
echo " ResNet-50 AMD/HIP ImageNet-1K Training"
echo " Job   : $SLURM_JOB_ID"
echo " Node  : $SLURM_NODELIST"
echo " Start : $(date)"
echo "========================================================"

# ── Paths ───────────────────────────────────────────────────────────────────
WORKSPACE_BASE="${WORKSPACE_BASE:-$HPC_SCRATCH}"
PROJECT_DIR="${PROJECT_DIR:-$WORKSPACE_BASE/resent50/performance-analysis-dl-models-resnet50}"
IMAGENET_DIR="${IMAGENET_DIR:-$PROJECT_DIR/ImageNet-1K/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC}"

# ── 1. Filesystem ───────────────────────────────────────────────────────────
echo "[1/7] Filesystem check"
if [ ! -d "$PROJECT_DIR" ]; then
    echo "  FAIL: $PROJECT_DIR not found"; exit 1
fi
if [ ! -d "$IMAGENET_DIR/train" ]; then
    echo "  FAIL: ImageNet train dir not found at $IMAGENET_DIR/train"; exit 1
fi
if [ ! -d "$IMAGENET_DIR/val" ]; then
    echo "  FAIL: ImageNet val dir not found at $IMAGENET_DIR/val"; exit 1
fi
echo "  ✓ project dir : $PROJECT_DIR"
echo "  ✓ imagenet    : $IMAGENET_DIR"
echo "  ✓ TMPDIR      : $TMPDIR  (free: $(df -h $TMPDIR | tail -1 | awk '{print $4}'))"

# ── 2. Module loading ────────────────────────────────────────────────────────
echo ""
echo "[2/7] Module loading"
module load cmake/3.30.5-dy3s

export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:/opt/amdgpu/lib64:$LD_LIBRARY_PATH"

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
    echo "  FAIL: rccl.h not found"; exit 1
fi
echo "  ✓ RCCL : $RCCL_ROOT"

# ── 4. GPU detection ────────────────────────────────────────────────────────
echo ""
echo "[4/7] GPU detection"
GPU_INFO=$(rocm-smi --showproductname 2>/dev/null | head -20)
echo "$GPU_INFO"

GFX_ARCH=$(rocminfo 2>/dev/null | awk '/gfx[0-9a-f]+/{for(i=1;i<=NF;i++) if($i ~ /gfx/){print $i; exit}}')
[ -z "$GFX_ARCH" ] && GFX_ARCH="gfx942"
echo "  ✓ GPU arch : $GFX_ARCH"

GPU_COUNT=$(rocm-smi --showproductname 2>/dev/null | grep -ci "Card Series" || true)
if [ -z "$GPU_COUNT" ] || [ "$GPU_COUNT" -eq 0 ]; then
    GPU_COUNT=${SLURM_GPUS_PER_NODE:-2}
fi
echo "  ✓ GPU count: $GPU_COUNT"

# ── 5. Hipify sweep — DISABLED ───────────────────────────────────────────────
echo ""
echo "[5/7] hipify sweep — SKIPPED (compat-shim approach via gpu_compat.h)"

# ── 6. Build ────────────────────────────────────────────────────────────────
echo ""
echo "[6/7] Build (train_multigpu via hip/)"
BUILD_DIR="$TMPDIR/imagenet_build_hip"
rm -rf "$BUILD_DIR" && mkdir -p "$BUILD_DIR"

cmake -S "$PROJECT_DIR/hip" -B "$BUILD_DIR" \
      -DCMAKE_HIP_ARCHITECTURES="$GFX_ARCH" \
      -DCMAKE_BUILD_TYPE=Release \
      -DUSE_NCCL=ON \
      > "$BUILD_DIR/cmake.log" 2>&1
if [ $? -ne 0 ]; then
    echo "  FAIL: cmake — last 30 lines:"; tail -30 "$BUILD_DIR/cmake.log"; exit 1
fi

cmake --build "$BUILD_DIR" -j16 --target train_multigpu \
      > "$BUILD_DIR/build.log" 2>&1
if [ $? -ne 0 ]; then
    echo "  FAIL: build — last 60 lines:"; tail -60 "$BUILD_DIR/build.log"; exit 1
fi
echo "  ✓ built: $BUILD_DIR/train_multigpu"

# ── 7. Training ─────────────────────────────────────────────────────────────
GLOBAL_BATCH="${IMAGENET_GLOBAL_BATCH:-512}"
EPOCHS="${IMAGENET_EPOCHS:-90}"
LR="${IMAGENET_LR:-0.4}"
LOCAL_BATCH=$((GLOBAL_BATCH / GPU_COUNT))

echo ""
echo "=== Training Configuration ==="
echo "  - GPUs            : $GPU_COUNT"
echo "  - Global batch    : $GLOBAL_BATCH"
echo "  - Local batch     : $LOCAL_BATCH"
echo "  - Epochs          : $EPOCHS"
echo "  - Learning rate   : $LR  (linear-scaled for global batch $GLOBAL_BATCH)"
echo "  - Data location   : $IMAGENET_DIR"
echo "  - Decode threads  : 8 per GPU (CPU mt_image_loader)"
echo ""
echo "[7/7] Training (${EPOCHS} epochs, ImageNet-1K, ${GPU_COUNT} GPUs)"
echo "  Starting at $(date)"

# Ensure progress lines never wrap in the SLURM log file
export COLUMNS=200

"$BUILD_DIR/train_multigpu" "$IMAGENET_DIR" "$GLOBAL_BATCH" "$EPOCHS" "$LR"
STATUS=$?

echo ""
echo "=== Job Summary ==="
echo "  Finished at $(date)"
if [ $STATUS -eq 0 ]; then
    echo "  ✓ AMD/HIP IMAGENET-1K TRAINING COMPLETE"
    echo "  Checkpoints saved to: $PROJECT_DIR/checkpoints/"
    echo "  Job ID: $SLURM_JOB_ID"
else
    echo "  ✗ Training FAILED (exit $STATUS)"
    exit $STATUS
fi
echo "=== Job Complete ==="
