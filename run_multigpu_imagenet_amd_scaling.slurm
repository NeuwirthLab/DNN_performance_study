#!/bin/bash -l
#SBATCH --job-name=rn50_scaling_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=4G
#SBATCH --account=l0003099
#SBATCH --partition=gpu:mi300x
#SBATCH --gres=gpu:mi300x:8
#SBATCH --time=06:00:00
#SBATCH --output=logs/rn50_scaling_amd_%j.out
#SBATCH --error=logs/rn50_scaling_amd_%j.err
#SBATCH --export=NONE
# ============================================================================
# AMD/ROCm scaling sweep on a single node: trains ResNet-50 on ImageNet-1K
# with 1, 2, 4, 8 GPUs in succession (whatever subset is actually available)
# and writes a CSV of {ngpus, batch, epoch_time, img/sec} for direct plotting.
#
# Single-node ONLY (the train_multigpu binary uses ncclCommInitAll, which
# bootstraps from the local process). Multi-node DDP would need a different
# entrypoint — out of scope here.
#
# Modes (runs both back-to-back by default):
#   weak    per-GPU batch fixed at $LOCAL_BATCH; global = LOCAL_BATCH * ngpus.
#           Ideal: img/sec scales linearly with ngpus.
#   strong  global batch fixed at $STRONG_GLOBAL_BATCH; local = global / ngpus.
#           Ideal: epoch_time halves each doubling.
#
# Common overrides (all optional):
#   SCALING_MODES="weak strong"  which modes to run (default: both)
#   GPU_COUNTS="1 2 3 4 5 6 7 8" which sweep points to attempt
#   LOCAL_BATCH=256              per-GPU batch (weak mode)
#   STRONG_GLOBAL_BATCH=256      fixed global batch (strong mode)
#   EPOCHS=1                     epochs per run (1 is plenty for steady-state)
#   LEARNING_RATE=0.1
#
# Submit:
#   sbatch run_multigpu_imagenet_amd_scaling.slurm
#   SCALING_MODES=strong sbatch --export=ALL,SCALING_MODES=strong run_multigpu_imagenet_amd_scaling.slurm
# ============================================================================

set -uo pipefail   # -e off so a single failed run doesn't abort the sweep

echo "========================================================"
echo " ResNet-50 AMD/ROCm Scaling Sweep"
echo " Job   : $SLURM_JOB_ID"
echo " Node  : $SLURM_NODELIST"
echo " TMPDIR: $TMPDIR  (free $(df -h "$TMPDIR" | tail -1 | awk '{print $4}'))"
echo " Start : $(date)"
echo "========================================================"

# ── 1. Project paths ────────────────────────────────────────────────────────
WORKSPACE_BASE="${WORKSPACE_BASE:-$HPC_SCRATCH}"
PROJECT_DIR="${PROJECT_DIR:-$WORKSPACE_BASE/resent50/performance-analysis-dl-models-resnet50}"
IMAGENET_DIR="${IMAGENET_DIR:-$PROJECT_DIR/ImageNet-1K/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC}"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "FAIL: PROJECT_DIR not found: $PROJECT_DIR"; exit 1
fi
if [ ! -d "$IMAGENET_DIR/train" ] || [ ! -d "$IMAGENET_DIR/val" ]; then
    echo "FAIL: IMAGENET_DIR must contain train/ and val/: $IMAGENET_DIR"; exit 1
fi
echo "  ✓ project dir : $PROJECT_DIR"
echo "  ✓ imagenet dir: $IMAGENET_DIR"

# ── 2. Modules + ROCm env ───────────────────────────────────────────────────
echo ""
echo "[2/6] Module loading"
module load cmake
export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:/opt/amdgpu/lib64:$LD_LIBRARY_PATH"
echo "  ✓ cmake : $(cmake --version | head -1 | awk '{print $3}')"
echo "  ✓ ROCM_PATH = ${ROCM_PATH}"

# libjpeg-turbo (TurboJPEG v3) — fast JPEG decode in mt_image_loader.
# If TURBOJPEG_ROOT is set, cmake will pick it up and the build defines
# USE_TURBOJPEG; otherwise the build silently falls back to stb_image.
# Default to the AVX2 Spack install (matches MI300x Genoa/Zen 4 hosts).
export TURBOJPEG_ROOT="${TURBOJPEG_ROOT:-/shared/spapps_2026_01_avx2/gcc-11.5.0/libjpeg-turbo-3.1.3-gx4a65sxms56s5xcm5s5gebgyl74xf3h}"
if [ -f "$TURBOJPEG_ROOT/lib64/libturbojpeg.so" ]; then
    export LD_LIBRARY_PATH="$TURBOJPEG_ROOT/lib64:$LD_LIBRARY_PATH"
    echo "  ✓ turbojpeg : $TURBOJPEG_ROOT"
else
    echo "  ! turbojpeg not at \$TURBOJPEG_ROOT — build will fall back to stb_image"
    unset TURBOJPEG_ROOT
fi

# MIOpen: persist the kernel cache and skip per-batch-size autotune ("Find").
# Without these, every change in local_batch triggers a fresh search that does
# its own miopenFind + temporary cudaMalloc/cudaFree round — wasted seconds
# per sweep point. FIND_MODE=3 (FAST) uses the cached heuristic only.
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$HOME/.cache/miopen}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-3}"
export MIOPEN_FIND_ENFORCE="${MIOPEN_FIND_ENFORCE:-3}"
export RN50_ENABLE_BF16=0   # FP32 baseline — unset now defaults to BF16 through-mode
mkdir -p "$MIOPEN_USER_DB_PATH"
echo "  ✓ MIOpen cache : $MIOPEN_USER_DB_PATH (FIND_MODE=$MIOPEN_FIND_MODE)"

# ── 3. RCCL detection ───────────────────────────────────────────────────────
echo ""
echo "[3/6] RCCL detection"
RCCL_ROOT=""
for c in "$ROCM_PATH" "/opt/rocm"; do
    if [ -n "$c" ] && { [ -f "$c/include/rccl/rccl.h" ] || [ -f "$c/include/rccl.h" ]; }; then
        RCCL_ROOT="$c"; break
    fi
done
[ -z "$RCCL_ROOT" ] && { echo "  FAIL: rccl.h not found"; exit 1; }
echo "  ✓ RCCL : $RCCL_ROOT"

# ── 4. GPU detection ────────────────────────────────────────────────────────
echo ""
echo "[4/6] GPU detection"
rocm-smi --showproductname 2>/dev/null | head -20 || true
GFX_ARCH=$(rocminfo 2>/dev/null | awk '/gfx[0-9a-f]+/{for(i=1;i<=NF;i++) if($i ~ /^gfx/){print $i; exit}}')
[ -z "$GFX_ARCH" ] && GFX_ARCH="gfx942"

# Authoritative GPU count — slurm tells us, fall back to rocminfo agent count.
AVAIL_GPUS="${SLURM_GPUS_ON_NODE:-}"
if [ -z "$AVAIL_GPUS" ]; then
    AVAIL_GPUS=$(rocminfo 2>/dev/null | grep -c 'Device Type:.*GPU' || echo 0)
fi
echo "  ✓ GPU arch     : $GFX_ARCH"
echo "  ✓ GPUs on node : $AVAIL_GPUS"

# ── 5. Build once (shared across the whole sweep) ───────────────────────────
echo ""
echo "[5/6] Build (train_multigpu via hip/, single build, used for all sweep points)"
BUILD_DIR="$TMPDIR/rn50_scaling_build"
rm -rf "$BUILD_DIR" && mkdir -p "$BUILD_DIR"

cmake -S "$PROJECT_DIR/hip" -B "$BUILD_DIR" \
      -DCMAKE_HIP_ARCHITECTURES="$GFX_ARCH" \
      -DCMAKE_BUILD_TYPE=Release \
      -DUSE_NCCL=ON \
      -DUSE_TURBOJPEG=OFF \
      > "$BUILD_DIR/cmake.log" 2>&1
if [ $? -ne 0 ]; then
    echo "  FAIL: cmake — last 40 lines:"; tail -40 "$BUILD_DIR/cmake.log"; exit 1
fi
cmake --build "$BUILD_DIR" -j16 --target train_multigpu \
      > "$BUILD_DIR/build.log" 2>&1
if [ $? -ne 0 ]; then
    echo "  FAIL: build — last 60 lines:"; tail -60 "$BUILD_DIR/build.log"; exit 1
fi
echo "  ✓ built: $BUILD_DIR/train_multigpu"
echo ""

# ── 6. Sweep configuration ──────────────────────────────────────────────────
GPU_COUNTS="${GPU_COUNTS:-1 2 3 4 5 6 7 8}"
LOCAL_BATCH="${LOCAL_BATCH:-256}"
STRONG_GLOBAL_BATCH="${STRONG_GLOBAL_BATCH:-256}"
EPOCHS="${EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-0.1}"
SCALING_MODES="${SCALING_MODES:-weak strong}"
RN50_PROGRESS_INTERVAL="${RN50_PROGRESS_INTERVAL:-100}"
export RN50_PROGRESS_INTERVAL

# Common RCCL tuning — same as run_multigpu_imagenet_amd.slurm.
export NCCL_DEBUG=WARN              # quieter than INFO; we don't need topo prints in every sweep point
export NCCL_ALGO=Tree
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_DISABLE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$HOME/.cache/miopen}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

RESULTS_CSV="$PROJECT_DIR/scaling_results_${SLURM_JOB_ID}.csv"
LOG_DIR="$PROJECT_DIR/scaling_logs_${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"
echo "ngpus,scaling_mode,global_batch,local_batch,epoch_time_s,img_per_sec,exit_code" > "$RESULTS_CSV"

echo "=== Sweep plan ==="
echo "  Modes        : $SCALING_MODES"
echo "  GPU counts   : $GPU_COUNTS"
echo "  Available    : $AVAIL_GPUS"
echo "  Weak local   : $LOCAL_BATCH (per-GPU; global = local × ngpus)"
echo "  Strong global: $STRONG_GLOBAL_BATCH (fixed; local = global / ngpus)"
echo "  Epochs/run   : $EPOCHS"
echo "  LR           : $LEARNING_RATE"
echo "  Results CSV  : $RESULTS_CSV"
echo "  Per-run logs : $LOG_DIR/"
echo ""

# ── 7. Loop over sweep points ───────────────────────────────────────────────
SWEEP_START=$(date +%s)
for SCALING_MODE in $SCALING_MODES; do
echo ""
echo "########################################################"
echo "# SCALING_MODE = $SCALING_MODE"
echo "########################################################"
for N in $GPU_COUNTS; do
    if [ "$N" -gt "$AVAIL_GPUS" ]; then
        echo "[skip] ngpus=$N — only $AVAIL_GPUS GPUs allocated on this node"
        echo "$N,$SCALING_MODE,,,,SKIPPED,need_more_gpus" >> "$RESULTS_CSV"
        continue
    fi

    # Compute batch sizes per scaling mode.
    if [ "$SCALING_MODE" = "weak" ]; then
        GLOBAL_BATCH=$((LOCAL_BATCH * N))
        LB=$LOCAL_BATCH
    else
        GLOBAL_BATCH=$STRONG_GLOBAL_BATCH
        if [ $((STRONG_GLOBAL_BATCH % N)) -ne 0 ]; then
            echo "[skip] strong mode: STRONG_GLOBAL_BATCH=$STRONG_GLOBAL_BATCH not divisible by ngpus=$N"
            echo "$N,$SCALING_MODE,,,,SKIPPED,batch_not_divisible" >> "$RESULTS_CSV"
            continue
        fi
        LB=$((STRONG_GLOBAL_BATCH / N))
    fi

    # Restrict the binary's view of the world to the first N GPUs.
    # Both env vars matter on AMD: HIP_VISIBLE_DEVICES gates HIP, ROCR_VISIBLE_DEVICES
    # gates the lower-level ROCr runtime that NCCL also reads.
    DEV_LIST=$(seq -s, 0 $((N - 1)))
    export HIP_VISIBLE_DEVICES="$DEV_LIST"
    export ROCR_VISIBLE_DEVICES="$DEV_LIST"

    LOG="$LOG_DIR/run_n${N}.log"
    echo ""
    echo "================================================================"
    echo " Sweep point: ngpus=$N  global_batch=$GLOBAL_BATCH  local_batch=$LB"
    echo " HIP_VISIBLE_DEVICES=$DEV_LIST"
    echo " Started at $(date)"
    echo "================================================================"

    RUN_START=$(date +%s)
    stdbuf -oL "$BUILD_DIR/train_multigpu" \
        "$IMAGENET_DIR" "$GLOBAL_BATCH" "$EPOCHS" "$LEARNING_RATE" \
        > "$LOG" 2>&1
    RC=$?
    RUN_SEC=$(( $(date +%s) - RUN_START ))

    # Extract the last completed epoch's time and img/sec from the table row:
    #   Epoch  LR       Train Loss   Train Acc    Val Loss     Val Acc      Time(s)    Img/sec
    #   1      0.02000  6.9012       0.50         6.7345       0.50         683.4      1875
    EPOCH_SEC=$(awk '/^ *[0-9]+ +[0-9.]+/{t=$(NF-1); s=$NF} END{print t}' "$LOG")
    IMG_PER_SEC=$(awk '/^ *[0-9]+ +[0-9.]+/{t=$(NF-1); s=$NF} END{print s}' "$LOG")
    : "${EPOCH_SEC:=NA}"
    : "${IMG_PER_SEC:=NA}"

    if [ "$RC" -eq 0 ]; then
        echo "  ✓ ngpus=$N  epoch_time=${EPOCH_SEC}s  img/sec=${IMG_PER_SEC}  (wall ${RUN_SEC}s)"
    else
        echo "  ✗ ngpus=$N  exit=$RC  (wall ${RUN_SEC}s)  — last 30 lines:"
        tail -30 "$LOG"
    fi
    echo "$N,$SCALING_MODE,$GLOBAL_BATCH,$LB,$EPOCH_SEC,$IMG_PER_SEC,$RC" >> "$RESULTS_CSV"
done
done

# ── 8. Summary ──────────────────────────────────────────────────────────────
SWEEP_SEC=$(( $(date +%s) - SWEEP_START ))
echo ""
echo "================================================================"
echo " Sweep complete in ${SWEEP_SEC}s"
echo " Results: $RESULTS_CSV"
echo "================================================================"
echo ""
column -s, -t "$RESULTS_CSV" 2>/dev/null || cat "$RESULTS_CSV"
echo ""
echo "Per-run training logs: $LOG_DIR/"
