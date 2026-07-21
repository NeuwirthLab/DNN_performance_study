#!/bin/bash -l
# ============================================================================
# sweep_energy.slurm  --  power/energy SCALING sweep (1,2,4,8 GPU) for one stack
# ----------------------------------------------------------------------------
# Runs the power harness at every (mode, ngpus) point inside ONE whole-node
# allocation, so for N<8 the other GPUs are idle-but-OURS (no shared-node
# mapping ambiguity) and analyze_power.py auto-selects the active N.
#
#   weak   : per-GPU batch fixed (LOCAL_BATCH); global = LOCAL_BATCH * N
#            -> headline metric J/image (constant per-GPU load)
#   strong : global batch fixed (STRONG_GLOBAL_BATCH); local = global / N
#            -> headline metric EDP / time-to-solution (fixed work)
#
# Each point yields results/<stack>/sweep_<jobid>/<mode>_gpu<N>/ with
# power_summary.csv + run_metrics.csv (throughput, energy, J/image, img/s/W,
# EDP). plot_energy_sweep.py turns the run_metrics.csv files into curves.
#
# Submit one job per stack (they run on separate nodes, in parallel):
#   STACK=native  sbatch --export=ALL,STACK=native  power_analysis/sweep_energy.slurm
#   STACK=pytorch sbatch --export=ALL,STACK=pytorch power_analysis/sweep_energy.slurm
# ============================================================================
#SBATCH --job-name=rn50_esweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=4G
#SBATCH --account=l0003099
#SBATCH --partition=acc
#SBATCH --gres=gpu:mi300x:8
#SBATCH --time=06:00:00
#SBATCH --output=power_analysis/results/sweep_%j.out
#SBATCH --error=power_analysis/results/sweep_%j.err
#SBATCH --export=NONE

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
PROJECT_PWD="$(pwd)"

STACK="${STACK:?set STACK=native or STACK=pytorch}"
# strong first (EDP-relevant), and big GPU counts first so the priority 8/4-GPU
# points complete early even if a 1-GPU point later runs long / times out.
SCALING_MODES="${SCALING_MODES:-strong weak}"
GPU_COUNTS="${GPU_COUNTS:-8 4 2 1}"
LOCAL_BATCH="${LOCAL_BATCH:-256}"
STRONG_GLOBAL_BATCH="${STRONG_GLOBAL_BATCH:-2048}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-0.1}"
WORKERS="${WORKERS:-8}"
AMP="${AMP:-0}"
WARMUP="${WARMUP:-45}"
REPEATS="${REPEATS:-1}"   # repeats per point for mean+/-std (stats runs use 5)
PREC_TAG="fp32"; [ "$AMP" = "1" ] && PREC_TAG="amp"

echo "=== energy scaling sweep ($STACK) ==="
echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST:-?}  start $(date)"
echo "modes=[$SCALING_MODES] gpus=[$GPU_COUNTS] local=$LOCAL_BATCH strong_global=$STRONG_GLOBAL_BATCH prec=$PREC_TAG"

# ── ROCm env ────────────────────────────────────────────────────────────────
module --force purge >/dev/null 2>&1 || true
module load cmake    >/dev/null 2>&1 || true
export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:/opt/amdgpu/lib64:${LD_LIBRARY_PATH:-}"
# Per-stack MIOpen cache, NOT the shared ~/.cache/miopen which earlier native
# crashes corrupted ("Duplicate ID" -> nondeterministic SIGABRT). Keeps the
# native and pytorch caches isolated and clean.
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$PROJECT_PWD/.miopen_${STACK}_cache}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"; export MIOPEN_FIND_MODE=3
mkdir -p "$MIOPEN_USER_DB_PATH"
export NCCL_DEBUG=WARN NCCL_ALGO=Tree NCCL_P2P_LEVEL=SYS NCCL_IB_DISABLE=1 \
       HSA_FORCE_FINE_GRAIN_PCIE=1

IMAGENET_DATA="${IMAGENET_DATA:-$PROJECT_PWD/ImageNet-1K/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC}"
[ -d "$IMAGENET_DATA/train" ] || { echo "FAIL: no ImageNet train/ at $IMAGENET_DATA"; exit 1; }

SWEEP_DIR="$PROJECT_PWD/power_analysis/results/${STACK}/sweep_${PREC_TAG}_${SLURM_JOB_ID}"
mkdir -p "$SWEEP_DIR"
echo "Sweep dir: $SWEEP_DIR"

# ── stack-specific setup ────────────────────────────────────────────────────
if [ "$STACK" = "native" ]; then
    BINARY="${BINARY:-$PROJECT_PWD/build_profile/train_multigpu}"
    if [ ! -x "$BINARY" ]; then
        echo "Building $(basename "$BINARY") ..."
        GFX_ARCH=$(rocminfo 2>/dev/null | awk '/gfx[0-9a-f]+/{for(i=1;i<=NF;i++) if($i ~ /^gfx/){print $i; exit}}')
        [ -z "$GFX_ARCH" ] && GFX_ARCH="gfx942"
        BD="$(dirname "$BINARY")"; TG="$(basename "$BINARY")"; mkdir -p "$BD"
        cmake -S "$PROJECT_PWD/hip" -B "$BD" -DCMAKE_HIP_ARCHITECTURES="$GFX_ARCH" \
              -DCMAKE_BUILD_TYPE=Release -DUSE_NCCL=ON -DUSE_TURBOJPEG=OFF \
              > "$BD/cmake_${TG}.log" 2>&1 \
          && cmake --build "$BD" -j16 --target "$TG" > "$BD/build_${TG}.log" 2>&1
        [ -x "$BINARY" ] || { echo "FAIL: build"; tail -40 "$BD/build_${TG}.log"; exit 1; }
    fi
    echo "Native binary: $BINARY"
elif [ "$STACK" = "pytorch" ]; then
    CONTAINER_IMAGE="${CONTAINER_IMAGE:-$PROJECT_PWD/pytorch_rocm.sif}"
    ENTRYPOINT="${ENTRYPOINT:-$PROJECT_PWD/pytorch_resnet50_ddp.py}"
    [ -f "$CONTAINER_IMAGE" ] && [ -f "$ENTRYPOINT" ] || { echo "FAIL: container/entrypoint"; exit 1; }
    HOST_TMP="$PROJECT_PWD/.container_tmp"; mkdir -p "$HOST_TMP"
    AMP_FLAG=""; [ "$AMP" = "1" ] && AMP_FLAG="--amp"
    echo "Container: $CONTAINER_IMAGE"
else
    echo "FAIL: STACK must be native or pytorch (got '$STACK')"; exit 1
fi

# ── one sweep point ─────────────────────────────────────────────────────────
run_point() {
    local mode="$1" N="$2" local_b="$3" global_b="$4" rep="${5:-1}"
    local out="$SWEEP_DIR/${mode}_gpu${N}_r${rep}"
    echo ""; echo "######## $STACK $mode N=$N local=$local_b global=$global_b rep=$rep ########"
    export HIP_VISIBLE_DEVICES="$(seq -s, 0 $((N-1)))"
    export ROCR_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"

    if [ "$STACK" = "native" ]; then
        bash "$PROJECT_PWD/power_analysis/measure_power.sh" \
            -o "$out" -i 0.5 -b 15 -c 8 -w "$WARMUP" -g all \
            -- stdbuf -oL "$BINARY" "$IMAGENET_DATA" "$global_b" "$EPOCHS" "$LR"
    else
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
                --bind "${IMAGENET_DATA}:/data" \
                --bind "${ENTRYPOINT}:/workspace/train.py" \
                --bind "${out}:/results" \
                --bind "${MIOPEN_USER_DB_PATH}:${MIOPEN_USER_DB_PATH}" \
                --bind "${HOST_TMP}:/container_tmp" \
                "$CONTAINER_IMAGE" \
                bash -c "set -uo pipefail; python -u -m torch.distributed.run \
                    --nproc_per_node=${N} --master_port=29500 /workspace/train.py \
                    -b ${local_b} --workers ${WORKERS} --epochs ${EPOCHS} \
                    --workspace /results ${AMP_FLAG} /data"
    fi
}

# ── sweep loop ──────────────────────────────────────────────────────────────
AVAIL_GPUS="${SLURM_GPUS_ON_NODE:-8}"
# repeats are the OUTER loop so the 5 samples of each point are spread across
# time (different thermal/clock states) rather than thermally-correlated bursts.
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
    fi
    run_point "$MODE" "$N" "$LB" "$GB" "$REP"
  done
done
done

# ── aggregate + plot (matplotlib lives in the torch container) ──────────────
echo ""; echo "=== aggregating sweep (mean +/- std across repeats) ==="
PLOT="$PROJECT_PWD/power_analysis/plot_energy_stats.py"
apptainer exec "$PROJECT_PWD/pytorch_rocm.sif" python3 "$PLOT" --sweep-dir "$SWEEP_DIR" \
  || python3 "$PLOT" --sweep-dir "$SWEEP_DIR" \
  || echo "(plotting failed; run_metrics.csv files are in $SWEEP_DIR/*/ )"

echo ""; echo "=== sweep done $(date) ==="
echo "Outputs: $SWEEP_DIR  (sweep_summary.csv + *.png + per-point run_metrics.csv)"
