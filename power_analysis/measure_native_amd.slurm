#!/bin/bash -l
# ============================================================================
# measure_native_amd.slurm  --  power measurement for the NATIVE HIP trainer
# ----------------------------------------------------------------------------
# Wraps the same native binary used by run_multigpu_imagenet_amd_scaling.slurm
# (train_multigpu / train_multigpu_tiny) in measure_power.sh, so you get socket
# power + energy + J/image for the C++/HIP path exactly like the PyTorch path.
#
# Defaults run the small tiny-imagenet binary on ONE GPU for a quick, bounded
# measurement. Override to point at the full ImageNet binary / more GPUs:
#
#   sbatch power_analysis/measure_native_amd.slurm                      # tiny, 1 GPU
#   BINARY=$PWD/build_profile/train_multigpu DATA=$PWD/ImageNet-1K/.../CLS-LOC \
#     GPUS=8 BATCH=2048 sbatch power_analysis/measure_native_amd.slurm  # full sweep point
#
# To power-profile EVERY scaling point, call measure_power.sh once per ngpus
# inside run_multigpu_imagenet_amd_scaling.slurm's loop (see the chat notes).
# ============================================================================
#SBATCH --job-name=rn50_pow_native
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --account=l0003099
#SBATCH --partition=acc_short
#SBATCH --gres=gpu:mi300x:1
#SBATCH --time=00:25:00
#SBATCH --output=power_analysis/results/native/native_power_%j.out
#SBATCH --error=power_analysis/results/native/native_power_%j.err
#SBATCH --export=NONE

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "=== native-trainer power measurement ==="
echo "Job $SLURM_JOB_ID on ${SLURM_NODELIST:-?}  start $(date)"

# ── ROCm env (mirrors the scaling script) ───────────────────────────────────
module --force purge >/dev/null 2>&1 || true
module load cmake    >/dev/null 2>&1 || true
export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:/opt/amdgpu/lib64:${LD_LIBRARY_PATH:-}"
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$HOME/.cache/miopen}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-3}"
mkdir -p "$MIOPEN_USER_DB_PATH"
export NCCL_DEBUG=WARN NCCL_ALGO=Tree NCCL_P2P_LEVEL=SYS NCCL_IB_DISABLE=1 \
       HSA_FORCE_FINE_GRAIN_PCIE=1

# ── workload selection ──────────────────────────────────────────────────────
BINARY="${BINARY:-$PWD/build_profile/train_multigpu_tiny}"
DATA="${DATA:-$PWD/tiny-imagenet-200}"
GPUS="${GPUS:-1}"
BATCH="${BATCH:-256}"      # GLOBAL batch (native binary splits across visible GPUs)
EPOCHS="${EPOCHS:-1}"
LR="${LR:-0.1}"
WARMUP="${WARMUP:-30}"     # seconds trimmed from front of train window

# ── build the native binary if missing (mirrors the scaling script's cmake) ──
if [ ! -x "$BINARY" ]; then
    echo "Binary not present; building $(basename "$BINARY") from hip/ ..."
    GFX_ARCH=$(rocminfo 2>/dev/null | awk '/gfx[0-9a-f]+/{for(i=1;i<=NF;i++) if($i ~ /^gfx/){print $i; exit}}')
    [ -z "$GFX_ARCH" ] && GFX_ARCH="gfx942"
    BUILD_DIR="$(dirname "$BINARY")"; TARGET="$(basename "$BINARY")"
    mkdir -p "$BUILD_DIR"
    echo "  arch=$GFX_ARCH  build_dir=$BUILD_DIR  target=$TARGET"
    cmake -S "$PWD/hip" -B "$BUILD_DIR" \
          -DCMAKE_HIP_ARCHITECTURES="$GFX_ARCH" \
          -DCMAKE_BUILD_TYPE=Release -DUSE_NCCL=ON -DUSE_TURBOJPEG=OFF \
          > "$BUILD_DIR/cmake_${TARGET}.log" 2>&1 \
      && cmake --build "$BUILD_DIR" -j16 --target "$TARGET" \
          > "$BUILD_DIR/build_${TARGET}.log" 2>&1
    if [ ! -x "$BINARY" ]; then
        echo "FAIL: build of $TARGET failed; tail of build log:"
        tail -40 "$BUILD_DIR/build_${TARGET}.log" 2>/dev/null; exit 1
    fi
    echo "  built: $BINARY"
fi
[ -d "$DATA/train" ]  || { echo "FAIL: data has no train/: $DATA"; exit 1; }

# restrict the binary to the first $GPUS allocated devices (matches scaling script)
DEV_LIST="$(seq -s, 0 $((GPUS - 1)))"
export HIP_VISIBLE_DEVICES="$DEV_LIST"
export ROCR_VISIBLE_DEVICES="$DEV_LIST"

OUT="$PWD/power_analysis/results/native/native_${PREC_TAG:-fp32}_gpu${GPUS}_${SLURM_JOB_ID:-$(date +%s)}"
echo "Binary : $BINARY"
echo "Data   : $DATA   global_batch=$BATCH epochs=$EPOCHS gpus=$GPUS"
echo "Out    : $OUT"

# Sample ALL physical GPUs. On a whole-node (gres=...:8) allocation we own every
# GPU so all are active; on a partial/shared allocation this also dodges the
# SLURM-index != amd-smi-index mismatch -- analyze_power.py picks the active GPUs.
bash "$PWD/power_analysis/measure_power.sh" \
    -o "$OUT" -i 0.5 -b 20 -c 10 -w "$WARMUP" -g all \
    -- stdbuf -oL "$BINARY" "$DATA" "$BATCH" "$EPOCHS" "$LR"

echo "=== done $(date) ==="
echo "Outputs: $OUT  (power_summary.csv, power_samples.csv, power_timeline.png)"
