#!/bin/bash -l
# ============================================================================
# sweep_energy_1node_valsubset.slurm
#   1-node PyTorch energy SCALING sweep (1,2,4,8 GPU) on the VAL-SUBSET data.
# ----------------------------------------------------------------------------
# This is the 1-node like-for-like ANCHOR for the completed 2-node sweep
# (job 52899806, sweep_energy_2node.slurm). The pre-existing 1-node PyTorch
# energy runs used the FULL train/ set (deleted ~Jun 28), so they are NOT
# directly comparable to the 2-node runs which used val/ as a surrogate.
# This script reproduces the 2-node data config exactly on ONE node so:
#   * 1-node 8-GPU  vs  2-node 8-GPU (4+4)  -> isolates inter-node comm energy
#   * 1-node 1->8 GPU scaling curve         -> single-node energy baseline
#
# Identical to sweep_energy.slurm EXCEPT:
#   - PyTorch stack only (native uses its own loader; out of scope here)
#   - binds val/ as both train and val (train/ is empty after the deletion)
#   - binds val_flat/ at its absolute path so the val/ symlinks resolve
#   - EPOCHS=10, STRONG_GLOBAL_BATCH=4096, LOCAL_BATCH=256 (match 2-node sweep)
#   - runs inside the reservation on a single node (gaod0001)
#
# Submit:
#   sbatch power_analysis/sweep_energy_1node_valsubset.slurm
# ============================================================================
#SBATCH --job-name=rn50_esweep_1n_val
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=4G
#SBATCH --account=l0003099
#SBATCH --partition=acc
#SBATCH --gres=gpu:mi300x:8
#SBATCH --time=04:00:00
#SBATCH --reservation=gaod_powcap
#SBATCH --nodelist=gaod0001
#SBATCH --output=power_analysis/results/sweep1nval_%j.out
#SBATCH --error=power_analysis/results/sweep1nval_%j.err
#SBATCH --export=NONE

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
PROJECT_PWD="$(pwd)"

STACK="pytorch"
SCALING_MODES="${SCALING_MODES:-strong weak}"
GPU_COUNTS="${GPU_COUNTS:-8 4 2 1}"
LOCAL_BATCH="${LOCAL_BATCH:-256}"
STRONG_GLOBAL_BATCH="${STRONG_GLOBAL_BATCH:-4096}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-0.1}"
WORKERS="${WORKERS:-8}"
AMP="${AMP:-0}"
WARMUP="${WARMUP:-45}"
REPEATS="${REPEATS:-1}"
PREC_TAG="fp32"; [ "$AMP" = "1" ] && PREC_TAG="amp"

echo "=== 1-node val-subset energy scaling sweep ($STACK) ==="
echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST:-?}  start $(date)"
echo "modes=[$SCALING_MODES] gpus=[$GPU_COUNTS] local=$LOCAL_BATCH strong_global=$STRONG_GLOBAL_BATCH epochs=$EPOCHS prec=$PREC_TAG"

# ── ROCm env ────────────────────────────────────────────────────────────────
module --force purge >/dev/null 2>&1 || true
module load cmake    >/dev/null 2>&1 || true
export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:/opt/amdgpu/lib64:${LD_LIBRARY_PATH:-}"
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$PROJECT_PWD/.miopen_${STACK}_valsubset_cache}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"; export MIOPEN_FIND_MODE=3
mkdir -p "$MIOPEN_USER_DB_PATH"
# 1-node: intra-node XGMI only, IB disabled (matches the original 1-node sweep).
export NCCL_DEBUG=WARN NCCL_ALGO=Tree NCCL_P2P_LEVEL=SYS NCCL_IB_DISABLE=1 \
       HSA_FORCE_FINE_GRAIN_PCIE=1

IMAGENET_DATA="${IMAGENET_DATA:-$PROJECT_PWD/ImageNet-1K/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC}"
# val-subset surrogate: val/ has 1000 classes x 50 imgs (symlinks -> val_flat/).
[ -d "$IMAGENET_DATA/val" ] || { echo "FAIL: no ImageNet val/ at $IMAGENET_DATA"; exit 1; }

SWEEP_DIR="$PROJECT_PWD/power_analysis/results/${STACK}/sweep_1node_valsubset_${PREC_TAG}_${SLURM_JOB_ID}"
mkdir -p "$SWEEP_DIR"
echo "Sweep dir: $SWEEP_DIR"

# ── pytorch container setup ─────────────────────────────────────────────────
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$PROJECT_PWD/pytorch_rocm.sif}"
ENTRYPOINT="${ENTRYPOINT:-$PROJECT_PWD/pytorch_resnet50_ddp.py}"
[ -f "$CONTAINER_IMAGE" ] && [ -f "$ENTRYPOINT" ] || { echo "FAIL: container/entrypoint"; exit 1; }
HOST_TMP="$PROJECT_PWD/.container_tmp"; mkdir -p "$HOST_TMP"
AMP_FLAG=""; [ "$AMP" = "1" ] && AMP_FLAG="--amp"
echo "Container: $CONTAINER_IMAGE"

# ── one sweep point ─────────────────────────────────────────────────────────
run_point() {
    local mode="$1" N="$2" local_b="$3" global_b="$4" rep="${5:-1}"
    local out="$SWEEP_DIR/${mode}_gpu${N}_r${rep}"
    echo ""; echo "######## $STACK $mode N=$N local=$local_b global=$global_b rep=$rep ########"
    mkdir -p "$out"
    export HIP_VISIBLE_DEVICES="$(seq -s, 0 $((N-1)))"
    export ROCR_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"

    bash "$PROJECT_PWD/power_analysis/measure_power.sh" \
        -o "$out" -i 0.5 -b 15 -c 8 -w "$WARMUP" -g all \
        -- apptainer exec --rocm --cleanenv \
            --env HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
            --env ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
            --env MIOPEN_USER_DB_PATH="$MIOPEN_USER_DB_PATH" \
            --env MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_CUSTOM_CACHE_DIR" \
            --env NCCL_DEBUG="$NCCL_DEBUG" --env NCCL_ALGO="$NCCL_ALGO" \
            --env NCCL_P2P_LEVEL="$NCCL_P2P_LEVEL" --env NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
            --env HSA_FORCE_FINE_GRAIN_PCIE="$HSA_FORCE_FINE_GRAIN_PCIE" \
            --env TMPDIR=/container_tmp --env TMP=/container_tmp --env TEMP=/container_tmp \
            --bind "${IMAGENET_DATA}/val:/data/train" \
            --bind "${IMAGENET_DATA}/val:/data/val" \
            --bind "${IMAGENET_DATA}/val_flat:${IMAGENET_DATA}/val_flat" \
            --bind "${ENTRYPOINT}:/workspace/train.py" \
            --bind "${out}:/results" \
            --bind "${MIOPEN_USER_DB_PATH}:${MIOPEN_USER_DB_PATH}" \
            --bind "${HOST_TMP}:/container_tmp" \
            "$CONTAINER_IMAGE" \
            bash -c "set -uo pipefail; python -u -m torch.distributed.run \
                --nproc_per_node=${N} --master_port=29500 /workspace/train.py \
                -b ${local_b} --workers ${WORKERS} --epochs ${EPOCHS} \
                --workspace /results ${AMP_FLAG} /data"
}

# ── sweep loop ──────────────────────────────────────────────────────────────
AVAIL_GPUS="${SLURM_GPUS_ON_NODE:-8}"
for REP in $(seq 1 "$REPEATS"); do
echo ""; echo "============ REPEAT $REP / $REPEATS ============"
for MODE in $SCALING_MODES; do
  for N in $GPU_COUNTS; do
    [ "$N" -gt "$AVAIL_GPUS" ] && { echo "skip N=$N (>$AVAIL_GPUS)"; continue; }
    if [ "$MODE" = "weak" ]; then
        LB=$LOCAL_BATCH; GB=$((LOCAL_BATCH * N))
    else
        GB=$STRONG_GLOBAL_BATCH
        [ $((GB % N)) -ne 0 ] && { echo "skip strong N=$N (global $GB not divisible)"; continue; }
        LB=$((GB / N))
        # strong small-N has a very large per-GPU batch; cap to avoid OOM
        # (MI300X 192GB handles up to ~1024 local for RN50 fp32 comfortably).
        if [ "$LB" -gt 1024 ]; then echo "skip strong N=$N (local $LB > 1024 OOM-guard)"; continue; fi
    fi
    run_point "$MODE" "$N" "$LB" "$GB" "$REP"
  done
done
done

# ── aggregate + plot ────────────────────────────────────────────────────────
echo ""; echo "=== aggregating sweep ==="
PLOT="$PROJECT_PWD/power_analysis/plot_energy_stats.py"
apptainer exec "$PROJECT_PWD/pytorch_rocm.sif" python3 "$PLOT" --sweep-dir "$SWEEP_DIR" \
  || python3 "$PLOT" --sweep-dir "$SWEEP_DIR" \
  || echo "(plotting failed; run_metrics.csv files are in $SWEEP_DIR/*/ )"

echo ""; echo "=== 1-node val-subset sweep done $(date) ==="
echo "Outputs: $SWEEP_DIR"
