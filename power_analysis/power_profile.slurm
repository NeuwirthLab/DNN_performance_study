#!/bin/bash -l
# ============================================================================
# power_profile.slurm  --  measure ResNet-50 GPU power on MI300X
# ----------------------------------------------------------------------------
# Mirrors the env/container setup of ngc_resnet_benchmark_amd.slurm, but wraps
# ONE training run in a power-measurement harness:
#
#   [idle baseline] -> [training (warmup + steady)] -> [cooldown]
#        ^ sampler runs across the whole thing, markers bound each phase ^
#
# Pipeline:
#   1. read the hardware ENERGY accumulator (exact ground-truth ΔE)   -> energy_before.csv
#   2. start gpu_power_sampler.sh in the background                   -> power_samples.csv
#   3. IDLE_SECONDS of nothing                                        -> baseline window
#   4. run containerized DDP ResNet-50, timestamp start/end           -> markers.csv
#   5. COOLDOWN_SECONDS of nothing
#   6. stop sampler, read energy accumulator again                    -> energy_after.csv
#   7. analyze_power.py -> summary CSV + timeline PNG + J/image, img/s/W
#
# Submit:
#   sbatch power_analysis/power_profile.slurm                  # 8 GPU, FP32
#   GPUS=4 AMP=1 sbatch power_analysis/power_profile.slurm     # 4 GPU, AMP
# For a scaling-vs-power study, submit once per GPU count (GPUS=1,2,4,8) and
# point analyze_power.py at all the run dirs.
# ============================================================================
#SBATCH --job-name=rn50_power_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=4G
#SBATCH --account=l0003099
#SBATCH --partition=gpu:mi300x
#SBATCH --gres=gpu:mi300x:8
#SBATCH --time=02:00:00
#SBATCH --output=rn50_power_amd_%j.out
#SBATCH --error=rn50_power_amd_%j.err
#SBATCH --export=NONE

set -euo pipefail

echo "=== ResNet-50 Power Profile (MI300X) ==="
echo "Job   : ${SLURM_JOB_ID:-local}   Node: ${SLURM_NODELIST:-?}   Start: $(date)"

# ---------------------------------------------------------------------------
# Modules + ROCm (same as the benchmark scripts)
# ---------------------------------------------------------------------------
module --force purge >/dev/null 2>&1 || true
module load cmake    >/dev/null 2>&1 || true
export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:/opt/amdgpu/lib64:${LD_LIBRARY_PATH:-}"

PROJECT_PWD="$(pwd)"
SAMPLER="$PROJECT_PWD/power_analysis/gpu_power_sampler.sh"
ANALYZER="$PROJECT_PWD/power_analysis/analyze_power.py"
chmod +x "$SAMPLER" 2>/dev/null || true

# pick a power backend for the energy-accumulator snapshots
SMI=""
command -v amd-smi  >/dev/null 2>&1 && SMI="amd-smi"
[ -z "$SMI" ] && command -v rocm-smi >/dev/null 2>&1 && SMI="rocm-smi"
echo "Power backend : ${SMI:-NONE FOUND}"

# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------
GPUS="${GPUS:-8}"
LOCAL_BATCH="${LOCAL_BATCH:-256}"
EPOCHS="${EPOCHS:-1}"
WORKERS="${WORKERS:-8}"
IDLE_SECONDS="${IDLE_SECONDS:-30}"        # baseline window (scenario #1)
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-15}"
WARMUP_SECONDS="${WARMUP_SECONDS:-45}"    # trimmed from front of train window (scenario #2)
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-1.0}" # seconds between samples (~1 Hz)

AMP_FLAG=""; AMP_TAG="fp32"
if [ "${AMP:-0}" = "1" ]; then AMP_FLAG="--amp"; AMP_TAG="amp"; fi

# ---------------------------------------------------------------------------
# Data + container (identical resolution logic to ngc_resnet_benchmark_amd)
# ---------------------------------------------------------------------------
IMAGENET_DATA="${IMAGENET_DATA:-$PROJECT_PWD/ImageNet-1K/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC}"
[ -d "$IMAGENET_DATA/train" ] || { echo "ERROR: train not found under $IMAGENET_DATA"; exit 1; }

CONTAINER_IMAGE="${CONTAINER_IMAGE:-$PROJECT_PWD/pytorch_rocm.sif}"
CONTAINER_IMAGE="$(readlink -f "$CONTAINER_IMAGE" 2>/dev/null || echo "$CONTAINER_IMAGE")"
[ -f "$CONTAINER_IMAGE" ] || { echo "ERROR: container not found: $CONTAINER_IMAGE"; exit 1; }
ENTRYPOINT="${ENTRYPOINT:-$PROJECT_PWD/pytorch_resnet50_ddp.py}"
[ -f "$ENTRYPOINT" ] || { echo "ERROR: entrypoint not found: $ENTRYPOINT"; exit 1; }

export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$HOME/.cache/miopen}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
HOST_TMP="${HOST_TMP:-$PROJECT_PWD/.container_tmp}"; mkdir -p "$HOST_TMP"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}" NCCL_ALGO="${NCCL_ALGO:-Tree}" \
       NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}" NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" \
       HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE:-1}"

# ---------------------------------------------------------------------------
# Output dir + phase markers
# ---------------------------------------------------------------------------
RUN_TAG="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="$PROJECT_PWD/power_analysis/results/power_${AMP_TAG}_gpu${GPUS}_${RUN_TAG}"
mkdir -p "$OUT_DIR"
POWER_CSV="$OUT_DIR/power_samples.csv"
MARKERS="$OUT_DIR/markers.csv"
TRAIN_LOG="$OUT_DIR/train.log"
echo "event,timestamp_s" > "$MARKERS"
mark() { echo "$1,$(date +%s.%N)" >> "$MARKERS"; echo "[mark] $1"; }

echo "Run dir       : $OUT_DIR"
echo "Config        : GPUS=$GPUS local_batch=$LOCAL_BATCH epochs=$EPOCHS amp=$AMP_TAG"
echo "Windows (s)   : idle=$IDLE_SECONDS warmup_trim=$WARMUP_SECONDS cooldown=$COOLDOWN_SECONDS"

# energy accumulator BEFORE (exact ground-truth integration; best-effort) -----
[ -n "$SMI" ] && { $SMI metric -g all --energy --csv > "$OUT_DIR/energy_before.csv" 2>/dev/null \
    || $SMI --showenergycounter --csv > "$OUT_DIR/energy_before.csv" 2>/dev/null || true; }

# ---------------------------------------------------------------------------
# Start the background sampler
# ---------------------------------------------------------------------------
bash "$SAMPLER" "$POWER_CSV" "$SAMPLE_INTERVAL" &
SAMPLER_PID=$!
# always stop the sampler, even on error/timeout
trap '[ -n "${SAMPLER_PID:-}" ] && kill -TERM "$SAMPLER_PID" 2>/dev/null || true' EXIT
sleep 2  # let it write a header + first rows

# ---------------------------------------------------------------------------
# Phase 1: idle baseline
# ---------------------------------------------------------------------------
mark idle_start
sleep "$IDLE_SECONDS"
mark idle_end

# ---------------------------------------------------------------------------
# Phase 2: training (warmup + steady-state) inside the container
# ---------------------------------------------------------------------------
VISIBLE="$(seq -s, 0 $((GPUS - 1)))"
mark train_start
set +e
apptainer exec --rocm --cleanenv \
    --env HIP_VISIBLE_DEVICES="$VISIBLE" \
    --env ROCR_VISIBLE_DEVICES="$VISIBLE" \
    --env MIOPEN_USER_DB_PATH="$MIOPEN_USER_DB_PATH" \
    --env MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_CUSTOM_CACHE_DIR" \
    --env NCCL_DEBUG="$NCCL_DEBUG" --env NCCL_ALGO="$NCCL_ALGO" \
    --env NCCL_P2P_LEVEL="$NCCL_P2P_LEVEL" --env NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
    --env HSA_FORCE_FINE_GRAIN_PCIE="$HSA_FORCE_FINE_GRAIN_PCIE" \
    --env TMPDIR=/container_tmp --env TMP=/container_tmp --env TEMP=/container_tmp \
    --bind "${IMAGENET_DATA}:/data" \
    --bind "${ENTRYPOINT}:/workspace/train.py" \
    --bind "${OUT_DIR}:/results" \
    --bind "${MIOPEN_USER_DB_PATH}:${MIOPEN_USER_DB_PATH}" \
    --bind "${HOST_TMP}:/container_tmp" \
    "$CONTAINER_IMAGE" \
    bash -c "set -euo pipefail
        python -u -m torch.distributed.run \
            --nproc_per_node=${GPUS} --master_port=29500 \
            /workspace/train.py \
            -b ${LOCAL_BATCH} --workers ${WORKERS} --epochs ${EPOCHS} \
            --workspace /results ${AMP_FLAG} /data" 2>&1 | tee "$TRAIN_LOG"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e
mark train_end
echo "Training exit : $TRAIN_STATUS"

# ---------------------------------------------------------------------------
# Phase 3: cooldown, then stop sampler + energy AFTER
# ---------------------------------------------------------------------------
sleep "$COOLDOWN_SECONDS"
mark cooldown_end
kill -TERM "$SAMPLER_PID" 2>/dev/null || true
wait "$SAMPLER_PID" 2>/dev/null || true
SAMPLER_PID=""
[ -n "$SMI" ] && { $SMI metric -g all --energy --csv > "$OUT_DIR/energy_after.csv" 2>/dev/null \
    || $SMI --showenergycounter --csv > "$OUT_DIR/energy_after.csv" 2>/dev/null || true; }

# throughput, if the trainer emitted it (train.total_ips : <val>)
THROUGHPUT="$(awk '/train.total_ips/{for(i=1;i<=NF;++i) if($i=="train.total_ips") v=$(i+2)} END{if(v!="")printf "%.2f",v}' "$TRAIN_LOG" 2>/dev/null)"
echo "Throughput    : ${THROUGHPUT:-unknown} img/s"

# ---------------------------------------------------------------------------
# Analyze (pure-stdlib; plotting optional). Prefer container python (has
# matplotlib); fall back to node python3 for the numeric summary.
# ---------------------------------------------------------------------------
THRU_ARG=(); [ -n "${THROUGHPUT:-}" ] && THRU_ARG=(--throughput "$THROUGHPUT")
echo ""; echo "=== Power analysis ==="
apptainer exec --rocm --cleanenv \
    --bind "${OUT_DIR}:${OUT_DIR}" --bind "${ANALYZER}:${ANALYZER}" \
    "$CONTAINER_IMAGE" \
    python3 "$ANALYZER" --run-dir "$OUT_DIR" --warmup "$WARMUP_SECONDS" "${THRU_ARG[@]}" \
  || python3 "$ANALYZER" --run-dir "$OUT_DIR" --warmup "$WARMUP_SECONDS" "${THRU_ARG[@]}" \
  || echo "  (analyzer failed; raw CSVs are in $OUT_DIR)"

echo ""; echo "Outputs       : $OUT_DIR"
echo "Finished      : $(date)"
exit "$TRAIN_STATUS"
