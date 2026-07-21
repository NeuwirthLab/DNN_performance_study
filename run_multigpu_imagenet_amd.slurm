#!/bin/bash -l
#SBATCH --job-name=rn50_multigpu_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --account=l0003099
#SBATCH --partition=gpu:mi300x
#SBATCH --gres=gpu:mi300x:2
#SBATCH --time=02:00:00
#SBATCH --output=logs/rn50_multigpu_amd_%j.out
#SBATCH --error=logs/rn50_multigpu_amd_%j.err
#SBATCH --export=NONE
# ============================================================================
# AMD/ROCm port of run_multigpu_tmpdir_dali.slurm
#
# What this does (mirrors the NVIDIA script's logic):
#   1. Filesystem + module load (rocm/rccl/cmake)
#   2. Locate RCCL
#   3. Detect MI300X gfx arch via rocminfo
#   4. Stage ImageNet-1K zip to TMPDIR + arrange val/ into class subfolders
#   5. Build train_multigpu via hip/CMakeLists.txt (USE_HIP, USE_NCCL=ON)
#   6. Run training — uses mt_image_loader + prefetch_loader (NO DALI)
#
# DALI replacement note:
#   DALI is NVIDIA-only. The C++ binary's #ifdef USE_DALI block is omitted so
#   the build automatically falls back to mt_image_loader.h + prefetch_loader.h
#   (multi-threaded stb_image decode + double-buffered prefetch). Throughput
#   is typically 70-85% of DALI-mixed on MI300X with 16 CPU workers.
#
# Required env (set before sbatch, or edit defaults below):
#   PROJECT_DIR    — repo root  (default: $HPC_SCRATCH/resent50/...)
# Optional env:
#   IMAGENET_DIR   — path to pre-extracted ImageNet-1K root (with train/ and val/)
#   IMAGENET_ZIP   — full path to imagenet-object-localization-challenge.zip
#   GLOBAL_BATCH (default 256), EPOCHS (default 5), LEARNING_RATE (default 0.1)
# ============================================================================

set -e

echo "========================================================"
echo " ResNet-50 Multi-GPU ImageNet Training (AMD/ROCm)"
echo " Job   : $SLURM_JOB_ID"
echo " Node  : $SLURM_NODELIST"
echo " TMPDIR: $TMPDIR  (free $(df -h "$TMPDIR" | tail -1 | awk '{print $4}'))"
echo " Start : $(date)"
echo "========================================================"

# ── 1. Project paths ────────────────────────────────────────────────────────
WORKSPACE_BASE="${WORKSPACE_BASE:-$HPC_SCRATCH}"
PROJECT_DIR="${PROJECT_DIR:-$WORKSPACE_BASE/resent50/performance-analysis-dl-models-resnet50}"

# Data is pre-extracted in-repo under ImageNet-1K/. The Kaggle zip layout
# nests data at competitions/<comp>/ILSVRC/Data/CLS-LOC/{train,val}.
IMAGENET_DIR="${IMAGENET_DIR:-$PROJECT_DIR/ImageNet-1K/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC}"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "FAIL: PROJECT_DIR not found: $PROJECT_DIR"; exit 1
fi
if [ ! -d "$IMAGENET_DIR/train" ] || [ ! -d "$IMAGENET_DIR/val" ]; then
    echo "FAIL: IMAGENET_DIR must contain train/ and val/: $IMAGENET_DIR"
    echo "      Override with: IMAGENET_DIR=/path/to/imagenet1k bash $0"
    exit 1
fi
echo "  ✓ project dir : $PROJECT_DIR"
echo "  ✓ imagenet dir: $IMAGENET_DIR"

# ── 2. Modules + ROCm env (mirror smoke_test_amd.slurm) ─────────────────────
echo ""
echo "[2/6] Module loading"
module load cmake

export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"

echo "  ✓ cmake : $(cmake --version | head -1 | awk '{print $3}')"
echo "  ✓ ROCM_PATH = ${ROCM_PATH}"

# ── 3. Locate RCCL ──────────────────────────────────────────────────────────
echo ""
echo "[3/6] RCCL detection"
RCCL_ROOT=""
for candidate in "$ROCM_PATH" "/opt/rocm"; do
    if [ -n "$candidate" ] && \
       { [ -f "$candidate/include/rccl/rccl.h" ] || [ -f "$candidate/include/rccl.h" ]; }; then
        RCCL_ROOT="$candidate"
        break
    fi
done
if [ -z "$RCCL_ROOT" ]; then
    echo "  FAIL: rccl.h not found under \$ROCM_PATH/include"
    exit 1
fi
echo "  ✓ RCCL : $RCCL_ROOT"

# ── 4. GPU detection ────────────────────────────────────────────────────────
echo ""
echo "[4/6] GPU detection"
GPU_INFO=$(rocm-smi --showproductname 2>/dev/null | head -20 || true)
echo "$GPU_INFO"

GFX_ARCH=$(rocminfo 2>/dev/null | awk '/gfx[0-9a-f]+/{for(i=1;i<=NF;i++) if($i ~ /^gfx/){print $i; exit}}')
[ -z "$GFX_ARCH" ] && GFX_ARCH="gfx942"

GPU_COUNT="${SLURM_GPUS_ON_NODE:-2}"
echo "  ✓ GPU arch  : $GFX_ARCH"
echo "  ✓ GPU count : $GPU_COUNT"

# ── 5. Verify dataset ───────────────────────────────────────────────────────
echo ""
echo "=== Dataset ==="
IMAGENET_DATA="$IMAGENET_DIR"
echo "Using pre-extracted ImageNet-1K at: $IMAGENET_DATA"
echo "  - Train images: $(find "$IMAGENET_DATA/train" -name '*.JPEG' | wc -l)"
echo "  - Val images:   $(find -L "$IMAGENET_DATA/val" -name '*.JPEG' | wc -l)"
echo ""

# ── 6. Training configuration ───────────────────────────────────────────────
GLOBAL_BATCH="${GLOBAL_BATCH:-256}"
EPOCHS="${EPOCHS:-5}"
LEARNING_RATE="${LEARNING_RATE:-0.1}"
MAKE_JOBS="${MAKE_JOBS:-16}"
RN50_PROGRESS_INTERVAL="${RN50_PROGRESS_INTERVAL:-50}"
export RN50_PROGRESS_INTERVAL

if [ "$GPU_COUNT" -le 0 ]; then
    echo "ERROR: invalid GPU count: $GPU_COUNT"; exit 1
fi
if [ $((GLOBAL_BATCH % GPU_COUNT)) -ne 0 ]; then
    echo "ERROR: GLOBAL_BATCH=$GLOBAL_BATCH must be divisible by GPU_COUNT=$GPU_COUNT"; exit 1
fi
LOCAL_BATCH=$((GLOBAL_BATCH / GPU_COUNT))

echo "=== Training Configuration ==="
echo "  GPU arch     : $GFX_ARCH"
echo "  GPUs         : $GPU_COUNT"
echo "  Global batch : $GLOBAL_BATCH"
echo "  Local batch  : $LOCAL_BATCH"
echo "  Epochs       : $EPOCHS"
echo "  LR           : $LEARNING_RATE"
echo "  Loader       : mt_image_loader + prefetch_loader (DALI is NVIDIA-only)"
echo "  Progress log : every $RN50_PROGRESS_INTERVAL batches"
echo ""

# ── 7. RCCL tuning (single-node MI300X, XGMI) ───────────────────────────────
export NCCL_DEBUG=INFO
export NCCL_ALGO=Tree
export NCCL_P2P_LEVEL=SYS         # XGMI on MI300X (NVL is NVLink-specific)
export NCCL_IB_DISABLE=1          # single-node — no IB
export HIP_VISIBLE_DEVICES="0,1"
export HSA_FORCE_FINE_GRAIN_PCIE=1

# Persistent MIOpen kernel cache so the autotune cost is paid only on the first run.
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$HOME/.cache/miopen}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

# ── 8. Build train_multigpu via hip/CMakeLists.txt ──────────────────────────
echo "[5/6] Build (train_multigpu via hip/)"
BUILD_DIR="$TMPDIR/rn50_amd_imagenet_build"
rm -rf "$BUILD_DIR" && mkdir -p "$BUILD_DIR"

cmake -S "$PROJECT_DIR/hip" -B "$BUILD_DIR" \
      -DCMAKE_HIP_ARCHITECTURES="$GFX_ARCH" \
      -DCMAKE_BUILD_TYPE=Release \
      -DUSE_NCCL=ON \
      > "$BUILD_DIR/cmake.log" 2>&1
if [ $? -ne 0 ]; then
    echo "  FAIL: cmake — last 40 lines:"; tail -40 "$BUILD_DIR/cmake.log"; exit 1
fi

cmake --build "$BUILD_DIR" -j"$MAKE_JOBS" --target train_multigpu \
      > "$BUILD_DIR/build.log" 2>&1
if [ $? -ne 0 ]; then
    echo "  FAIL: build — last 60 lines:"; tail -60 "$BUILD_DIR/build.log"; exit 1
fi
echo "  ✓ built: $BUILD_DIR/train_multigpu"
echo ""

# ── 9. Train ────────────────────────────────────────────────────────────────
echo "[6/6] Training ($EPOCHS epochs, ImageNet-1K, $GPU_COUNT GPUs)"
echo "  Starting at $(date)"
export COLUMNS=200

cd "$PROJECT_DIR"   # so checkpoints/ writes land in the repo, like the NVIDIA run
stdbuf -oL "$BUILD_DIR/train_multigpu" "$IMAGENET_DATA" "$GLOBAL_BATCH" "$EPOCHS" "$LEARNING_RATE"
STATUS=$?

echo ""
echo "=== Job Summary ==="
echo "  Finished at $(date)"
if [ $STATUS -eq 0 ]; then
    echo "  ✓ AMD/HIP IMAGENET TRAINING COMPLETE"
    echo "  Checkpoints saved to: $PROJECT_DIR/checkpoints/"
    echo "  Job ID: $SLURM_JOB_ID"
else
    echo "  ✗ Training FAILED (exit $STATUS)"
    exit $STATUS
fi
echo "=== Job Complete ==="
