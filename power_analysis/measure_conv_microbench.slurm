#!/bin/bash -l
# ============================================================================
# measure_conv_microbench.slurm -- power measurement for conv_power_microbench.py
# (matched-duty RN-50 fwd+bwd, FP16 vs BF16), 1 GPU, under measure_power.sh.
# Mirrors measure_pytorch_energy_8gpu.slurm's container/harness setup but runs the
# microbench directly (no torch.distributed). Precision via DT={float16,bfloat16}.
#   DT=float16  sbatch power_analysis/measure_conv_microbench.slurm
# ============================================================================
#SBATCH --job-name=conv_ubench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --account=l0003099
#SBATCH --partition=acc
#SBATCH --gres=gpu:mi300x:1
#SBATCH --time=00:40:00
#SBATCH --reservation=gaod0001
#SBATCH --nodelist=gaod0001
#SBATCH --output=power_analysis/results/pytorch/conv_ubench_%j.out
#SBATCH --error=power_analysis/results/pytorch/conv_ubench_%j.err
#SBATCH --export=NONE

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
PROJECT_PWD="$(pwd)"

echo "=== conv power microbench (matched duty) ==="
echo "Job ${SLURM_JOB_ID:-local} on ${SLURM_NODELIST:-?}  start $(date)"

module --force purge >/dev/null 2>&1 || true
export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:/opt/amdgpu/lib64:${LD_LIBRARY_PATH:-}"

DT="${DT:-float16}"
BATCH="${BATCH:-256}"
ITERS_PER_BLOCK="${ITERS_PER_BLOCK:-400}"
BLOCKS="${BLOCKS:-3}"

CONTAINER_IMAGE="$(readlink -f "${CONTAINER_IMAGE:-$PROJECT_PWD/pytorch_rocm.sif}" 2>/dev/null || echo "$PROJECT_PWD/pytorch_rocm.sif")"
[ -f "$CONTAINER_IMAGE" ] || { echo "FAIL: container not found: $CONTAINER_IMAGE"; exit 1; }
ENTRYPOINT="$PROJECT_PWD/power_analysis/conv_power_microbench.py"
[ -f "$ENTRYPOINT" ] || { echo "FAIL: entrypoint not found: $ENTRYPOINT"; exit 1; }

export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$HOME/.cache/miopen}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
HOST_TMP="${HOST_TMP:-$PROJECT_PWD/.container_tmp}"; mkdir -p "$HOST_TMP"

RUN_TAG="${SLURM_JOB_ID:-$(date +%s)}"
OUT="$PROJECT_PWD/power_analysis/results/pytorch/conv_ubench_${DT}_${RUN_TAG}"
mkdir -p "$OUT"
echo "DT=$DT batch=$BATCH iters/block=$ITERS_PER_BLOCK blocks=$BLOCKS  out=$OUT"

BIND_ARGS=(
    --bind "${ENTRYPOINT}:/workspace/bench.py"
    --bind "${OUT}:/results"
    --bind "${MIOPEN_USER_DB_PATH}:${MIOPEN_USER_DB_PATH}"
    --bind "${HOST_TMP}:/container_tmp"
)

TRAIN_CMD=(
    apptainer exec --rocm --cleanenv
        --env HIP_VISIBLE_DEVICES=0
        --env ROCR_VISIBLE_DEVICES=0
        --env MIOPEN_USER_DB_PATH="$MIOPEN_USER_DB_PATH"
        --env MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_CUSTOM_CACHE_DIR"
        --env DT="$DT" --env BATCH="$BATCH"
        --env ITERS_PER_BLOCK="$ITERS_PER_BLOCK" --env BLOCKS="$BLOCKS"
        --env TMPDIR="/container_tmp" --env TMP="/container_tmp" --env TEMP="/container_tmp"
        "${BIND_ARGS[@]}"
        "$CONTAINER_IMAGE"
        python -u /workspace/bench.py
)

# same harness knobs as the training measurement: idle 20s, cooldown 10s,
# warmup-trim 60s, 0.5s interval, sample all GPUs, skip block 0 as warmup.
bash "$PROJECT_PWD/power_analysis/measure_power.sh" \
    -o "$OUT" -i 0.5 -b 20 -c 10 -w 60 -g all -E 1 \
    -- "${TRAIN_CMD[@]}"
STATUS=$?
echo "=== done $(date)  (measure_power.sh exit: $STATUS) ==="
exit "$STATUS"
