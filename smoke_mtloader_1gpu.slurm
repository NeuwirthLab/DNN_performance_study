#!/bin/bash -l
#SBATCH --job-name=rn50_smoke_mt1gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --account=l0003099
#SBATCH --partition=acc
#SBATCH --gres=gpu:mi300x:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/rn50_smoke_mt1gpu_%j.out
#SBATCH --error=logs/rn50_smoke_mt1gpu_%j.err
#SBATCH --export=NONE
# ============================================================================
# SMOKE TEST — ResNet-50 / ImageNet-1K, 1 GPU, MTImageLoader + turbojpeg, 16 threads.
# Runs the REAL 1000-class training pipeline (build -> scan -> load -> fwd/bwd/opt
# -> val) but caps the training loop at RN50_MAX_STEPS so it finishes in minutes.
# PASS = builds with turbojpeg, loader reports 16 decode threads, loss trends down,
#        a plausible img/s is printed, exit 0.
#   Knobs: GLOBAL_BATCH(256) EPOCHS(1) RN50_MAX_STEPS(150) RN50_DECODE_THREADS(16)
# ============================================================================
set -e
echo "==== RN50 MTLoader 1-GPU SMOKE | job=$SLURM_JOB_ID node=$SLURM_NODELIST $(date) ===="

PROJECT_DIR="${PROJECT_DIR:-$HPC_SCRATCH/resent50/performance-analysis-dl-models-resnet50}"
IMAGENET_DIR="${IMAGENET_DIR:-$PROJECT_DIR/ImageNet-1K/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC}"
[ -d "$IMAGENET_DIR/train" ] && [ -d "$IMAGENET_DIR/val" ] || { echo "FAIL: no train/ + val/ under $IMAGENET_DIR"; exit 1; }

# ── modules + ROCm ──
module load cmake >/dev/null 2>&1 || true
export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
# turbojpeg 3.x (tj3* API required by hip/turbojpeg_compat.h)
export TURBOJPEG_ROOT="${TURBOJPEG_ROOT:-/shared/spapps_2026_01_avx2/gcc-11.5.0/libjpeg-turbo-3.1.3-gx4a65sxms56s5xcm5s5gebgyl74xf3h}"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${TURBOJPEG_ROOT}/lib64:${LD_LIBRARY_PATH:-}"

GFX_ARCH=$(rocminfo 2>/dev/null | awk '/gfx[0-9a-f]+/{for(i=1;i<=NF;i++) if($i ~ /^gfx/){print $i; exit}}'); [ -z "$GFX_ARCH" ] && GFX_ARCH=gfx942

# ── 1 GPU, 16 decode threads ──
export HIP_VISIBLE_DEVICES=0                       # single GPU
GPU_COUNT=1
export RN50_DECODE_THREADS="${RN50_DECODE_THREADS:-16}"   # MTImageLoader worker threads
GLOBAL_BATCH="${GLOBAL_BATCH:-256}"; LOCAL_BATCH=$((GLOBAL_BATCH / GPU_COUNT))
EPOCHS="${EPOCHS:-1}"; LR="${LEARNING_RATE:-0.1}"
export RN50_MAX_STEPS="${RN50_MAX_STEPS:-0}"       # 0 = FULL epoch (no step cap)
export RN50_PROGRESS_INTERVAL="${RN50_PROGRESS_INTERVAL:-25}"
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$HOME/.cache/miopen}"; export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"; mkdir -p "$MIOPEN_USER_DB_PATH"
export NCCL_DEBUG=WARN NCCL_P2P_LEVEL=SYS NCCL_IB_DISABLE=1 HSA_FORCE_FINE_GRAIN_PCIE=1
# Keep the trainer's proven default precision (BF16 through-mode trains cleanly).
# BF16 has no MIOpen solver for BatchNorm *inference*, so skip validation in this
# loader/throughput smoke (RN50_SKIP_VAL).
export RN50_SKIP_VAL=1
# EXPERIMENT: FP32 with a FRESHLY-WIPED MIOpen cache. FP32 is the historical fast path
# (~1246 img/s, job 52393128) but was aborting ("Empty code object path",
# resnet50.cu:1044) because the old ~/.cache/miopen mixed MIOpen 3.4.0/3.5.1 kernels.
# Cache moved aside; DEFAULT find-mode (unset) compiles kernels fresh under the current
# MIOpen. If this survives + hits ~1200 -> FP32 recovered; if it still aborts, the FP32
# BatchNorm-fwd kernel genuinely will not build here and BF16 (~700) is the only path.
export RN50_ENABLE_BF16=0

echo "  arch=$GFX_ARCH gpus=$GPU_COUNT global_batch=$GLOBAL_BATCH local=$LOCAL_BATCH epochs=$EPOCHS max_steps=$RN50_MAX_STEPS decode_threads=$RN50_DECODE_THREADS"
echo "  turbojpeg_root=$TURBOJPEG_ROOT"
echo "  train imgs=$(find "$IMAGENET_DIR/train" -name '*.JPEG' | wc -l)  val imgs=$(find -L "$IMAGENET_DIR/val" -name '*.JPEG' | wc -l)"

# ── build train_multigpu WITH turbojpeg (USE_TURBOJPEG=ON → MTImageLoader uses tj3*) ──
BUILD_DIR="$TMPDIR/rn50_smoke_mt1gpu_build"; rm -rf "$BUILD_DIR"; mkdir -p "$BUILD_DIR"
echo "[build] cmake (USE_NCCL=ON USE_TURBOJPEG=ON USE_ROCAL=OFF) ..."
USE_TJ="${USE_TJ:-ON}"    # ON = MTImageLoader+turbojpeg; OFF = STB (matches working FP32 scaling runs)
cmake -S "$PROJECT_DIR/hip" -B "$BUILD_DIR" \
      -DCMAKE_HIP_ARCHITECTURES="$GFX_ARCH" -DCMAKE_BUILD_TYPE=Release \
      -DUSE_NCCL=ON -DUSE_TURBOJPEG="$USE_TJ" -DUSE_ROCAL=OFF \
      > "$BUILD_DIR/cmake.log" 2>&1 || { echo "FAIL cmake:"; tail -30 "$BUILD_DIR/cmake.log"; exit 1; }
echo "  build: USE_TURBOJPEG=$USE_TJ $(grep 'turbojpeg :' "$BUILD_DIR/cmake.log" | head -1)"
cmake --build "$BUILD_DIR" -j"${MAKE_JOBS:-16}" --target train_multigpu > "$BUILD_DIR/build.log" 2>&1 || { echo "FAIL build:"; tail -60 "$BUILD_DIR/build.log"; exit 1; }
BIN="$BUILD_DIR/train_multigpu"; [ -x "$BIN" ] || { echo "FAIL: $BIN missing"; exit 1; }
echo "  ✓ built $BIN"

# ── run the smoke ──
echo "[run] starting real 1000-class training (capped at $RN50_MAX_STEPS steps) $(date)"
cd "$PROJECT_DIR"
stdbuf -oL "$BIN" "$IMAGENET_DIR" "$GLOBAL_BATCH" "$EPOCHS" "$LR"; STATUS=$?
echo "==== SMOKE done rc=$STATUS $(date) ===="
[ $STATUS -eq 0 ] && echo "  ✓ PASS (see loss trend + img/s above)" || { echo "  ✗ FAIL rc=$STATUS"; exit $STATUS; }
