#!/bin/bash -l
# ============================================================================
# probe_fp16_energy.slurm -- 1-GPU PyTorch per-FLOP energy probe: is FP16 MFMA
# genuinely cheaper than the shared 3.426 pJ/FLOP coefficient (coefficient bias),
# and does the missing loss scaler change power (gradient-underflow test)?
# Runs fp16_conv_energy_probe.py under measure_power.sh, then analyze_fp16_probe.py.
# 1 GPU on purpose: e_fp16 must be measured at the SAME geometry as e_bf16.
#   sbatch --export=ALL power_analysis/probe_fp16_energy.slurm
# ============================================================================
#SBATCH --job-name=fp16_eprobe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --account=l0003099
#SBATCH --partition=acc
#SBATCH --gres=gpu:mi300x:1
#SBATCH --time=00:30:00
#SBATCH --reservation=gaod0001
#SBATCH --nodelist=gaod0001
#SBATCH --output=power_analysis/results/pytorch/fp16_eprobe_%j.out
#SBATCH --error=power_analysis/results/pytorch/fp16_eprobe_%j.err
#SBATCH --export=NONE

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
PROJECT_PWD="$(pwd)"
echo "=== FP16 per-FLOP energy probe (1 GPU) job ${SLURM_JOB_ID:-local} on ${SLURM_NODELIST:-?} $(date) ==="

module --force purge >/dev/null 2>&1 || true
module load cmake    >/dev/null 2>&1 || true
export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:/opt/amdgpu/lib64:${LD_LIBRARY_PATH:-}"

CONTAINER_IMAGE="${CONTAINER_IMAGE:-$PROJECT_PWD/pytorch_rocm.sif}"
CONTAINER_IMAGE="$(readlink -f "$CONTAINER_IMAGE" 2>/dev/null || echo "$CONTAINER_IMAGE")"
[ -f "$CONTAINER_IMAGE" ] || { echo "FAIL: container not found: $CONTAINER_IMAGE"; exit 1; }
PROBE="$PROJECT_PWD/power_analysis/fp16_conv_energy_probe.py"
[ -f "$PROBE" ] || { echo "FAIL: probe not found: $PROBE"; exit 1; }

export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$HOME/.cache/miopen}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
HOST_TMP="${HOST_TMP:-$PROJECT_PWD/.container_tmp}"; mkdir -p "$HOST_TMP"

RUN_TAG="${SLURM_JOB_ID:-$(date +%s)}"
OUT="$PROJECT_PWD/power_analysis/results/pytorch/fp16_eprobe_${RUN_TAG}"
mkdir -p "$OUT"
echo "Out: $OUT   container: $CONTAINER_IMAGE"

PROBE_ITERS="${PROBE_ITERS:-400}"; PROBE_ROUNDS="${PROBE_ROUNDS:-2}"; PROBE_BATCH="${PROBE_BATCH:-256}"

TRAIN_CMD=(
    apptainer exec --rocm --cleanenv
        --env HIP_VISIBLE_DEVICES=0
        --env ROCR_VISIBLE_DEVICES=0
        --env MIOPEN_USER_DB_PATH="$MIOPEN_USER_DB_PATH"
        --env MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_CUSTOM_CACHE_DIR"
        --env PROBE_ITERS="$PROBE_ITERS"
        --env PROBE_ROUNDS="$PROBE_ROUNDS"
        --env PROBE_BATCH="$PROBE_BATCH"
        --env RN_ARCH="${RN_ARCH:-resnet50}"
        --env TMPDIR="/container_tmp" --env TMP="/container_tmp" --env TEMP="/container_tmp"
        --bind "${PROBE}:/workspace/probe.py"
        --bind "${OUT}:/results"
        --bind "${MIOPEN_USER_DB_PATH}:${MIOPEN_USER_DB_PATH}"
        --bind "${HOST_TMP}:/container_tmp"
        "$CONTAINER_IMAGE"
        python -u /workspace/probe.py
)

# sample ALL physical GPUs at 2 Hz; short idle baseline; no epoch windowing.
bash "$PROJECT_PWD/power_analysis/measure_power.sh" \
    -o "$OUT" -i 0.5 -b 10 -c 5 -w 0 -E 0 -g all \
    -- "${TRAIN_CMD[@]}"
STATUS=$?
echo "=== measure done (exit $STATUS); analyzing ==="

/usr/bin/python3.11 "$PROJECT_PWD/power_analysis/analyze_fp16_probe.py" "$OUT" 2>&1 | tee "$OUT/fp16_probe_report.txt"
echo "=== report: $OUT/fp16_probe_report.txt ==="
exit "$STATUS"
