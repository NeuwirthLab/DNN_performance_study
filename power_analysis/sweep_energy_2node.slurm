#!/bin/bash -l
# ============================================================================
# sweep_energy_2node.slurm  --  2-node power/energy sweep for PyTorch-ROCm DDP
# ----------------------------------------------------------------------------
# Extends sweep_energy.slurm from 1 node (max 8 GPU) to 2 nodes (max 16 GPU).
# Allocates 2 whole nodes so the idle GPUs on each node belong to THIS job
# (no shared-node GPU mapping ambiguity in amd-smi).
#
# Stacks supported:
#   pytorch  -- DDP via torch.distributed.run (torchrun) inside pytorch_rocm.sif.
#              Each sweep point: srun launches one torchrun per node, both
#              rendezvous at MASTER_ADDR:29500.
#
# Native (HIP) is NOT supported at 2-node: train_multigpu.cu uses
# ncclCommInitAll(), a single-process-only API.  Multi-node native would need
# a full refactor to ncclCommInitRank + file/socket UID exchange (future work).
#
# GPU counts swept (NGPU_COUNTS, total across both nodes):
#   default: 16 8
#   16 = 8+8 (full 2-node)    -- new territory vs 1-node
#    8 = 4+4 (half load/node) -- direct compare vs 1-node N=8 (topology diff)
#
# Scaling modes:
#   weak   : per-GPU batch fixed (LOCAL_BATCH); global = LOCAL_BATCH * N
#   strong : global batch fixed (STRONG_GLOBAL_BATCH); local = global / N
#
# Outputs: power_analysis/results/pytorch/sweep_2node_<jobid>/<mode>_gpu<N>_r<R>/
#   power_summary.csv, run_metrics.csv, command.log, per-node/ sub-dirs
#
# Submit:
#   sbatch power_analysis/sweep_energy_2node.slurm
#   NGPU_COUNTS="16" sbatch --export=ALL,NGPU_COUNTS=16 power_analysis/sweep_energy_2node.slurm
# ============================================================================
#SBATCH --job-name=rn50_esweep_2n
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=4G
#SBATCH --account=l0003099
#SBATCH --partition=acc
#SBATCH --gres=gpu:mi300x:8
#SBATCH --time=04:00:00
#SBATCH --reservation=gaod_powcap
#SBATCH --nodelist=gaod[0001-0002]
#SBATCH --output=power_analysis/results/sweep2n_%j.out
#SBATCH --error=power_analysis/results/sweep2n_%j.err
#SBATCH --export=NONE

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
PROJECT_PWD="$(pwd)"

# ── sweep parameters ─────────────────────────────────────────────────────────
SCALING_MODES="${SCALING_MODES:-strong weak}"
NGPU_COUNTS="${NGPU_COUNTS:-16 8}"    # total GPUs across BOTH nodes
GPUS_PER_NODE=8                        # MI300X node always has 8
LOCAL_BATCH="${LOCAL_BATCH:-256}"
STRONG_GLOBAL_BATCH="${STRONG_GLOBAL_BATCH:-4096}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-0.1}"
WORKERS="${WORKERS:-8}"
AMP="${AMP:-0}"
WARMUP="${WARMUP:-45}"
REPEATS="${REPEATS:-1}"
AMP_FLAG=""; [ "$AMP" = "1" ] && AMP_FLAG="--amp"
PREC_TAG="fp32"; [ "$AMP" = "1" ] && PREC_TAG="amp"
# Model arch (torchvision name) selectable via RN_ARCH; default keeps RN50.
# Threaded into the container --env below so OOD models (resnet101, mobilenet_v2,
# vgg16, ...) get a true 2-node energy sweep, not just RN50.
RN_ARCH="${RN_ARCH:-resnet50}"
ARCH_TAG="$RN_ARCH"

echo "=== energy scaling sweep (pytorch, 2-node) ==="
echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST:-?}  start $(date)"
echo "arch=$RN_ARCH modes=[$SCALING_MODES] ngpus=[$NGPU_COUNTS] local=$LOCAL_BATCH strong_global=$STRONG_GLOBAL_BATCH prec=$PREC_TAG"

# ── ROCm env ─────────────────────────────────────────────────────────────────
module --force purge >/dev/null 2>&1 || true
module load cmake    >/dev/null 2>&1 || true
export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:/opt/amdgpu/lib64:${LD_LIBRARY_PATH:-}"
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$PROJECT_PWD/.miopen_pytorch_2n_cache}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
export MIOPEN_FIND_MODE=3
mkdir -p "$MIOPEN_USER_DB_PATH"

# ── RCCL / NCCL env for 2-node ───────────────────────────────────────────────
# Key difference from the 1-node sweep: IB must be ENABLED (not disabled) so
# RCCL can use the inter-node interconnect (InfiniBand / RoCE).
# NCCL_SOCKET_IFNAME excludes loopback and Docker bridges; adjust if your
# cluster's IB interface is named differently (ib0, mlx5_0, hsn0, etc.).
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=SYS          # intra-node XGMI (unchanged)
export NCCL_IB_DISABLE=0           # ENABLE IB for inter-node traffic
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"  # exclude loopback
export HSA_FORCE_FINE_GRAIN_PCIE=1
# Fail FAST on a RCCL fault instead of hanging the whole walltime: a bad IB
# fabric otherwise stalls collective init silently. Async error handling makes
# RCCL raise (and the rank exit) so the point fails quickly and the sweep moves on.
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"  # legacy name (older torch)

# Derive MASTER_ADDR from the first node in SLURM's allocation.
MASTER_ADDR=$(scontrol show hostnames "${SLURM_NODELIST}" | head -1)
export MASTER_ADDR
echo "MASTER_ADDR=$MASTER_ADDR"

# ── container + entrypoint checks ────────────────────────────────────────────
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$PROJECT_PWD/pytorch_rocm.sif}"
ENTRYPOINT="${ENTRYPOINT:-$PROJECT_PWD/pytorch_resnet50_ddp.py}"
[ -f "$CONTAINER_IMAGE" ] || { echo "FAIL: container not found: $CONTAINER_IMAGE"; exit 1; }
[ -f "$ENTRYPOINT"      ] || { echo "FAIL: entrypoint not found: $ENTRYPOINT"; exit 1; }
HOST_TMP="$PROJECT_PWD/.container_tmp"; mkdir -p "$HOST_TMP"

IMAGENET_DATA="${IMAGENET_DATA:-$PROJECT_PWD/ImageNet-1K/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC}"
[ -d "$IMAGENET_DATA/train" ] || { echo "FAIL: no ImageNet train/ at $IMAGENET_DATA"; exit 1; }
# Which CLS-LOC subdir feeds /data/train. Default `val` (50k subset) = legacy
# behaviour; set TRAIN_SUBDIR=train for the full 1.28M-image train set (extracted
# 2026-07-01). /data/val stays on val/ regardless.
TRAIN_SUBDIR="${TRAIN_SUBDIR:-val}"
[ -d "$IMAGENET_DATA/$TRAIN_SUBDIR" ] || { echo "FAIL: no $TRAIN_SUBDIR/ at $IMAGENET_DATA"; exit 1; }
echo "TRAIN_SUBDIR=$TRAIN_SUBDIR  (train images from $IMAGENET_DATA/$TRAIN_SUBDIR)"

SWEEP_DIR="$PROJECT_PWD/power_analysis/results/pytorch/sweep_2node_${ARCH_TAG}_${PREC_TAG}_${SLURM_JOB_ID}"
mkdir -p "$SWEEP_DIR"
echo "Sweep dir: $SWEEP_DIR"

# ── one sweep point ──────────────────────────────────────────────────────────
run_point() {
    local mode="$1" N="$2" local_b="$3" global_b="$4" rep="${5:-1}"
    local out="$SWEEP_DIR/${mode}_gpu${N}_r${rep}"
    local n_per_node=$(( N / 2 ))   # always 2-node; each node gets N/2 GPUs
    # Unique rendezvous port per point: a lingering rank from the previous point
    # can hold 29500 in TIME_WAIT -> "Address already in use". Monotonic counter.
    : "${PORT_IDX:=0}"; PORT_IDX=$(( PORT_IDX + 1 )); local port=$(( 29500 + PORT_IDX ))

    echo ""; echo "######## pytorch 2-node $mode N=$N (${n_per_node}/node) local=$local_b global=$global_b rep=$rep ########"
    mkdir -p "$out"

    # Write a per-node launch script to the shared filesystem.
    # Variables from the sweep context are expanded NOW (at write time);
    # \${SLURM_PROCID} is escaped so it expands at srun execution time on each
    # node (0 on node-0, 1 on node-1) — this is the node_rank for torchrun.
    local LAUNCH="$out/_launch_node.sh"
    cat > "$LAUNCH" << LAUNCH_EOF
#!/bin/bash
set -uo pipefail
VISIBLE_LIST=\$(seq -s, 0 $(( n_per_node - 1 )))
# Per-NODE MIOpen cache: both nodes must NOT share one sqlite DB on NFS or
# concurrent FindEx writes corrupt it ("Duplicate ID"). \$(hostname -s) makes
# each node own its cache dir; evaluated at run time on each node.
MN_DB="${MIOPEN_USER_DB_PATH}_\$(hostname -s)"
mkdir -p "\$MN_DB"
apptainer exec --rocm --cleanenv \
    --env HIP_VISIBLE_DEVICES="\$VISIBLE_LIST" \
    --env ROCR_VISIBLE_DEVICES="\$VISIBLE_LIST" \
    --env RN_ARCH="${RN_ARCH}" \
    --env MIOPEN_USER_DB_PATH="\$MN_DB" \
    --env MIOPEN_CUSTOM_CACHE_DIR="\$MN_DB" \
    --env NCCL_DEBUG="${NCCL_DEBUG}" \
    --env NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL}" \
    --env NCCL_IB_DISABLE="${NCCL_IB_DISABLE}" \
    --env NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
    --env HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE}" \
    --env TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING}" \
    --env NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING}" \
    --env MASTER_ADDR="${MASTER_ADDR}" \
    --env TMPDIR=/container_tmp --env TMP=/container_tmp \
    --bind "${IMAGENET_DATA}/${TRAIN_SUBDIR}:/data/train" \
    --bind "${IMAGENET_DATA}/val:/data/val" \
    --bind "${IMAGENET_DATA}/val_flat:${IMAGENET_DATA}/val_flat" \
    --bind "${ENTRYPOINT}:/workspace/train.py" \
    --bind "${out}:/results" \
    --bind "\$MN_DB:\$MN_DB" \
    --bind "${HOST_TMP}:/container_tmp" \
    "${CONTAINER_IMAGE}" \
    python -u -m torch.distributed.run \
        --nnodes=2 \
        --nproc_per_node=${n_per_node} \
        --master_addr="${MASTER_ADDR}" \
        --master_port=${port} \
        --node_rank=\${SLURM_PROCID} \
        /workspace/train.py \
        -b ${local_b} --workers ${WORKERS} --epochs ${EPOCHS} \
        --workspace /results ${AMP_FLAG} /data
LAUNCH_EOF
    chmod +x "$LAUNCH"

    # measure_power_multinode.sh starts srun sidecar samplers on both nodes,
    # then runs the srun training step, then aggregates energy across nodes.
    bash "$PROJECT_PWD/power_analysis/measure_power_multinode.sh" \
        -o "$out" -i 0.5 -b 15 -c 8 -w "$WARMUP" \
        -- \
        srun --ntasks=2 --ntasks-per-node=1 /usr/bin/bash "$LAUNCH"
}

# ── sweep loop ───────────────────────────────────────────────────────────────
TOTAL_GPUS=$(( 2 * GPUS_PER_NODE ))

for REP in $(seq 1 "$REPEATS"); do
echo ""; echo "============ REPEAT $REP / $REPEATS ============"
for MODE in $SCALING_MODES; do
  for N in $NGPU_COUNTS; do
    if [ "$N" -gt "$TOTAL_GPUS" ]; then
        echo "skip N=$N (>$TOTAL_GPUS available on 2 nodes)"; continue
    fi
    if [ $(( N % 2 )) -ne 0 ]; then
        echo "skip N=$N (must be even across 2 nodes)"; continue
    fi
    if [ "$MODE" = "weak" ]; then
        LB=$LOCAL_BATCH; GB=$(( LOCAL_BATCH * N ))
    else
        GB=$STRONG_GLOBAL_BATCH
        [ $(( GB % N )) -ne 0 ] && { echo "skip strong N=$N ($GB not divisible)"; continue; }
        LB=$(( GB / N ))
    fi
    run_point "$MODE" "$N" "$LB" "$GB" "$REP"
  done
done
done

# ── aggregate sweep summary via plot_energy_stats.py ─────────────────────────
echo ""; echo "=== aggregating sweep ==="
PLOT="$PROJECT_PWD/power_analysis/plot_energy_stats.py"
apptainer exec "$CONTAINER_IMAGE" python3 "$PLOT" --sweep-dir "$SWEEP_DIR" \
  || python3 "$PLOT" --sweep-dir "$SWEEP_DIR" \
  || echo "(plotting skipped; run_metrics.csv files in $SWEEP_DIR/*/)"

echo ""; echo "=== 2-node sweep done $(date) ==="
echo "Outputs: $SWEEP_DIR"
