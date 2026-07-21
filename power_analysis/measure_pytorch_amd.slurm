#!/bin/bash -l
# ============================================================================
# measure_pytorch_amd.slurm  --  power measurement for the PyTorch DDP trainer
# ----------------------------------------------------------------------------
# The PyTorch counterpart to measure_native_amd.slurm. It runs the SAME
# ResNet-50 / ImageNet-1K workload, but via the stock-torch DDP entrypoint
# (pytorch_resnet50_ddp.py) inside the rocm/pytorch container, wrapped in the
# exact same measure_power.sh harness. Result: socket power + HW-accumulator
# energy + J/image + img/s/W for the PyTorch path, directly comparable to the
# native C++/HIP numbers from measure_native_amd.slurm.
#
# The measured command is the containerized `torch.distributed.run` launch
# (mirroring ngc_resnet_benchmark_amd.slurm's run_one()). measure_power.sh
# brackets it with [idle baseline] -> [training] -> [cooldown], samples ALL
# physical GPUs at 2 Hz, trims a generous warmup window (MIOpen autotune +
# torch warmup are slow on the first run), and reads the HW energy counter
# before/after. The entrypoint prints `train.total_ips : <val>` at the end,
# which measure_power.sh auto-captures as throughput for the efficiency
# metrics (J/image, img/s/W, EDP).
#
# ── Usage ───────────────────────────────────────────────────────────────────
#   # default: 1 GPU, local batch 256, 1 epoch, FP32
#   sbatch power_analysis/measure_pytorch_amd.slurm
#
#   # 8-GPU AMP run, bigger per-GPU batch
#   GPUS=8 AMP=1 LOCAL_BATCH=256 sbatch power_analysis/measure_pytorch_amd.slurm
#
#   # short smoke test on acc_short with a time-bounded training window
#   # (override the partition/time at submit time; TRAIN_TIMEOUT caps training)
#   TRAIN_TIMEOUT=900 sbatch --partition=acc_short --time=00:25:00 \
#       power_analysis/measure_pytorch_amd.slurm
#
# ── Parameters (env vars, with defaults) ─────────────────────────────────────
#   GPUS          number of GPUs the trainer sees (default 1; up to alloc)
#   LOCAL_BATCH   per-GPU batch size                              (default 256)
#   EPOCHS        epochs to train                                 (default 1)
#   WORKERS       dataloader workers per rank                     (default 8)
#   AMP           1 = torch.amp autocast, 0 = FP32                (default 0)
#   TRAIN_TIMEOUT optional wall-clock cap (s) on the training cmd (default: none)
#   IMAGENET_DATA override dataset CLS-LOC root
#   CONTAINER_IMAGE / ENTRYPOINT  override container / train.py
#
# ── AMD-vs-native comparison intent ──────────────────────────────────────────
# Run this with the same GPUS / global-batch as measure_native_amd.slurm and
# compare the per-GPU steady power, J/image and img/s/W in the two
# power_summary.csv files. Both paths sample with `-g all` and let
# analyze_power.py pick the active GPU(s), so the only difference is the
# software stack (native HIP vs PyTorch-ROCm), which is exactly what we want
# to isolate.
# ============================================================================
#SBATCH --job-name=rn50_pow_pt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=4G
#SBATCH --account=l0003099
#SBATCH --partition=acc
#SBATCH --gres=gpu:mi300x:8
#SBATCH --time=04:00:00
#SBATCH --output=power_analysis/results/pytorch/pytorch_power_%j.out
#SBATCH --error=power_analysis/results/pytorch/pytorch_power_%j.err
#SBATCH --export=NONE

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
PROJECT_PWD="$(pwd)"

echo "=== PyTorch DDP ResNet-50 power measurement (AMD MI300X) ==="
echo "Job ${SLURM_JOB_ID:-local} on ${SLURM_NODELIST:-?}  start $(date)"

# ── ROCm env (mirrors measure_native_amd.slurm / ngc_resnet_benchmark_amd) ───
module --force purge >/dev/null 2>&1 || true
module load cmake    >/dev/null 2>&1 || true
export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:/opt/amdgpu/lib64:${LD_LIBRARY_PATH:-}"

# ── workload parameters ──────────────────────────────────────────────────────
GPUS="${GPUS:-1}"
LOCAL_BATCH="${LOCAL_BATCH:-256}"
EPOCHS="${EPOCHS:-1}"
WORKERS="${WORKERS:-8}"
AMP="${AMP:-0}"
TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-}"   # optional wall-clock cap on training cmd
GLOBAL_BATCH=$((LOCAL_BATCH * GPUS))

AVAIL_GPUS="${SLURM_GPUS_ON_NODE:-8}"
if [ "$GPUS" -gt "$AVAIL_GPUS" ]; then
    echo "FAIL: requested GPUS=$GPUS exceeds allocation ($AVAIL_GPUS)"; exit 1
fi

# ── data + container + entrypoint ────────────────────────────────────────────
IMAGENET_DATA="${IMAGENET_DATA:-$PROJECT_PWD/ImageNet-1K/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC}"
if [ ! -d "$IMAGENET_DATA/train" ]; then
    echo "FAIL: ImageNet train/ not found under: $IMAGENET_DATA"
    echo "      Override with IMAGENET_DATA=/path/to/CLS-LOC sbatch $0"; exit 1
fi

CONTAINER_IMAGE="${CONTAINER_IMAGE:-$PROJECT_PWD/pytorch_rocm.sif}"
CONTAINER_IMAGE="$(readlink -f "$CONTAINER_IMAGE" 2>/dev/null || echo "$CONTAINER_IMAGE")"
[ -f "$CONTAINER_IMAGE" ] || { echo "FAIL: container not found: $CONTAINER_IMAGE"; exit 1; }

ENTRYPOINT="${ENTRYPOINT:-$PROJECT_PWD/pytorch_resnet50_ddp.py}"
[ -f "$ENTRYPOINT" ] || { echo "FAIL: entrypoint not found: $ENTRYPOINT"; exit 1; }

# AMP toggle (passed through to the entrypoint)
AMP_FLAG=""
PREC_TAG="fp32"
if [ "$AMP" = "1" ]; then AMP_FLAG="--amp --amp-dtype ${AMP_DTYPE:-bfloat16}"; PREC_TAG="amp_${AMP_DTYPE:-bfloat16}"; fi

# ── output dir (matches the requested naming scheme) ─────────────────────────
RUN_TAG="${SLURM_JOB_ID:-$(date +%s)}"
OUT="$PROJECT_PWD/power_analysis/results/pytorch/pytorch_${PREC_TAG}_gpu${GPUS}_${RUN_TAG}"
mkdir -p "$OUT"

echo "Container   : $CONTAINER_IMAGE"
echo "Entrypoint  : $ENTRYPOINT"
echo "Data        : $IMAGENET_DATA"
echo "GPUs        : $GPUS (alloc $AVAIL_GPUS)   precision: $PREC_TAG"
echo "Batch       : local=$LOCAL_BATCH global=$GLOBAL_BATCH  epochs=$EPOCHS workers=$WORKERS"
[ -n "$TRAIN_TIMEOUT" ] && echo "Train cap   : ${TRAIN_TIMEOUT}s (timeout wrapper)"
echo "Out         : $OUT"

# ── MIOpen kernel cache + disk-backed container TMPDIR ───────────────────────
# (same reasoning as ngc_resnet_benchmark_amd.slurm: MIOpen's LLVM JIT needs a
# real-disk scratch dir, the apptainer tmpfs overlay is too small).
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$HOME/.cache/miopen}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
HOST_TMP="${HOST_TMP:-$PROJECT_PWD/.container_tmp}"
mkdir -p "$HOST_TMP"

# ── RCCL tuning (single-node MI300X XGMI) ────────────────────────────────────
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ALGO="${NCCL_ALGO:-Tree}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE:-1}"

# Restrict the trainer to the first $GPUS devices (its own 0..N-1 view). NOTE:
# the power SAMPLER still uses `-g all` below — on this SHARED node SLURM's
# job-local GPU index does NOT match amd-smi's physical index, so we sample
# everything and let analyze_power.py pick the GPU(s) that actually draw power.
VISIBLE_LIST="$(seq -s, 0 $((GPUS - 1)))"

BIND_ARGS=(
    --bind "${IMAGENET_DATA}:/data"
    --bind "${ENTRYPOINT}:/workspace/train.py"
    --bind "${OUT}:/results"
    --bind "${MIOPEN_USER_DB_PATH}:${MIOPEN_USER_DB_PATH}"
    --bind "${HOST_TMP}:/container_tmp"
)

# Optional wall-clock cap so a smoke test on acc_short cannot overrun: prefix
# the in-container python with `timeout`. timeout exits 124 on the cap, which
# is fine here — we only want enough steady-state training to measure power.
TIMEOUT_PREFIX=""
[ -n "$TRAIN_TIMEOUT" ] && TIMEOUT_PREFIX="timeout --signal=INT ${TRAIN_TIMEOUT}"

# ── the measured command: containerized torch.distributed.run ────────────────
# Built as an argv array and handed verbatim to measure_power.sh, which does
# its own `2>&1 | tee command.log` and then auto-detects `train.total_ips`.
TRAIN_CMD=(
    apptainer exec --rocm --cleanenv
        --env HIP_VISIBLE_DEVICES="$VISIBLE_LIST"
        --env ROCR_VISIBLE_DEVICES="$VISIBLE_LIST"
        --env MIOPEN_USER_DB_PATH="$MIOPEN_USER_DB_PATH"
        --env MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_CUSTOM_CACHE_DIR"
        --env RN_ARCH="${RN_ARCH:-resnet50}"
        --env RN50_NO_GRADSCALER="${RN50_NO_GRADSCALER:-0}"
        --env NCCL_DEBUG="$NCCL_DEBUG"
        --env NCCL_ALGO="$NCCL_ALGO"
        --env NCCL_P2P_LEVEL="$NCCL_P2P_LEVEL"
        --env NCCL_IB_DISABLE="$NCCL_IB_DISABLE"
        --env HSA_FORCE_FINE_GRAIN_PCIE="$HSA_FORCE_FINE_GRAIN_PCIE"
        --env TMPDIR="/container_tmp"
        --env TMP="/container_tmp"
        --env TEMP="/container_tmp"
        "${BIND_ARGS[@]}"
        "$CONTAINER_IMAGE"
        bash -c "
            set -uo pipefail
            ${TIMEOUT_PREFIX} python -u -m torch.distributed.run \
                --nproc_per_node=${GPUS} \
                --master_port=29500 \
                /workspace/train.py \
                -b ${LOCAL_BATCH} \
                --workers ${WORKERS} \
                --epochs ${EPOCHS} \
                --workspace /results \
                ${AMP_FLAG} \
                /data
        "
)

# ── run under the power harness ──────────────────────────────────────────────
# idle 20s, cooldown 10s, warmup-trim 60s (PyTorch/MIOpen warmup is long),
# interval 0.5s, sample ALL physical GPUs (-g all).
bash "$PROJECT_PWD/power_analysis/measure_power.sh" \
    -o "$OUT" -i 0.5 -b 20 -c 10 -w 60 -g all \
    -- "${TRAIN_CMD[@]}"
STATUS=$?

echo "=== done $(date)  (measure_power.sh exit: $STATUS) ==="
echo "Outputs: $OUT  (power_summary.csv, power_samples.csv, power_timeline.png, command.log)"
exit "$STATUS"
