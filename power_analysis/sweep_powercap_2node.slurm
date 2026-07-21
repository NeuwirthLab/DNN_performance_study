#!/bin/bash -l
# ============================================================================
# sweep_powercap_2node.slurm  --  2-node (16-GPU) power-CAP energy sweep
# ----------------------------------------------------------------------------
# Merges two existing scripts:
#   * sweep_energy_2node.slurm  -> the 2-node PyTorch-DDP launch machinery
#                                  (run_point + measure_power_multinode.sh)
#   * powercap_sweep.slurm      -> the per-cap loop + summary/plot adapters
# ...into the missing intersection: vary the GPU power cap across MULTIPLE
# torchvision models at full 2-node (16-GPU) strong scaling.
#
# Matrix (default):  ARCHS x CAPS  =  4 models x 6 caps = 24 measured points,
#   strong scaling, N=16, one repeat. PyTorch DDP only -- the native HIP
#   trainer is single-process (ncclCommInitAll) and cannot run 2-node.
#
# *** POWER-CAP PRIVILEGE (multinode) ***
# `rocm-smi --setpoweroverdrive` is NOT in sudoers on the gaod nodes. The
# working path (verified 2026-06-30, jobs 52916515/546/565) is:
#     sudo -n amd-smi set -g all --power-cap <WATTS>
# applied to BOTH nodes via a 2-task srun. set_cap_2node() below does this.
#
# Submit (full sweep):
#   sbatch power_analysis/sweep_powercap_2node.slurm
# Subset / overrides:
#   ARCHS="resnet50" CAPS="300 600 750" \
#     sbatch --export=ALL,ARCHS,CAPS power_analysis/sweep_powercap_2node.slurm
# ============================================================================
#SBATCH --job-name=pcap_2node
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
#SBATCH --output=power_analysis/results/pcap2n_%j.out
#SBATCH --error=power_analysis/results/pcap2n_%j.err
#SBATCH --export=NONE

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
PROJECT_PWD="$(pwd)"

# ── sweep parameters ─────────────────────────────────────────────────────────
ARCHS="${ARCHS:-resnet50 resnet101 mobilenet_v2 vgg16}"   # torchvision names
CAPS="${CAPS:-750 700 600 500 400 300}"                   # W per GPU, high->low
# Fixed scaling point: strong, full 2-node (16 GPU). global=4096 -> local 256.
N=16
GPUS_PER_NODE=8
STRONG_GLOBAL_BATCH="${STRONG_GLOBAL_BATCH:-4096}"
LOCAL_BATCH=$(( STRONG_GLOBAL_BATCH / N ))
# Keep EPOCHS/WARMUP identical to the default-power 2-node sweep so the in-sweep
# cap_750 point reproduces the existing strong-16 anchor (~0.345 J/img, 52902858).
EPOCHS="${EPOCHS:-10}"
LR="${LR:-0.1}"
WORKERS="${WORKERS:-8}"
WARMUP="${WARMUP:-45}"
AMP="${AMP:-0}"
AMP_FLAG=""; [ "$AMP" = "1" ] && AMP_FLAG="--amp"
PREC_TAG="fp32"; [ "$AMP" = "1" ] && PREC_TAG="amp"

echo "=== 2-node power-CAP sweep (pytorch, strong N=$N) ==="
echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST:-?}  start $(date)"
echo "archs=[$ARCHS] caps=[$CAPS] local=$LOCAL_BATCH global=$STRONG_GLOBAL_BATCH epochs=$EPOCHS prec=$PREC_TAG"

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

# ── RCCL / NCCL env for 2-node (IB ENABLED for inter-node, as in the energy sweep)
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=SYS          # intra-node XGMI
export NCCL_IB_DISABLE=0           # ENABLE IB for inter-node traffic
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"
export HSA_FORCE_FINE_GRAIN_PCIE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

MASTER_ADDR=$(scontrol show hostnames "${SLURM_NODELIST}" | head -1)
export MASTER_ADDR
echo "MASTER_ADDR=$MASTER_ADDR"

# ── container + data checks ──────────────────────────────────────────────────
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

SWEEP_DIR="$PROJECT_PWD/power_analysis/results/pytorch/powercap_2node_${PREC_TAG}_${SLURM_JOB_ID}"
mkdir -p "$SWEEP_DIR"
echo "Sweep dir: $SWEEP_DIR"

# ── power-cap helpers (BOTH nodes, verified sudo amd-smi path) ────────────────
# srun must execve a real binary under --export=NONE, so wrap in /usr/bin/bash
# and use ABSOLUTE paths (bare `sudo`/`amd-smi` are not on srun's exec PATH).
# Invocation form proven by probe_powercap_confirm.slurm (job 52916546).
set_cap_2node() {   # $1 = watts; sets every GPU on both nodes
    srun --ntasks=2 --ntasks-per-node=1 /usr/bin/bash -c \
         "sudo -n /usr/bin/amd-smi set -g all --power-cap $1"
}
# ALWAYS restore the default cap on both nodes on any exit (reset = the proven
# restore path) so the next reservation user does NOT inherit a throttled GPU.
trap 'echo "[trap] restoring default cap on both nodes"; \
      srun --ntasks=2 --ntasks-per-node=1 /usr/bin/bash -c \
      "sudo -n /usr/bin/amd-smi reset -g all --power-cap" >/dev/null 2>&1 || true' \
      EXIT INT TERM

# ── one sweep point (strong, N=16) ───────────────────────────────────────────
# Mirrors run_point() from sweep_energy_2node.slurm; out dir is per arch+cap.
run_point() {
    local arch="$1" cap="$2" out="$3"
    local local_b="$LOCAL_BATCH" global_b="$STRONG_GLOBAL_BATCH"
    local n_per_node=$(( N / 2 ))
    : "${PORT_IDX:=0}"; PORT_IDX=$(( PORT_IDX + 1 )); local port=$(( 29500 + PORT_IDX ))

    echo ""; echo "######## $arch  cap=${cap}W  strong N=$N (${n_per_node}/node) local=$local_b global=$global_b ########"
    mkdir -p "$out"

    local LAUNCH="$out/_launch_node.sh"
    cat > "$LAUNCH" << LAUNCH_EOF
#!/bin/bash
set -uo pipefail
VISIBLE_LIST=\$(seq -s, 0 $(( n_per_node - 1 )))
MN_DB="${MIOPEN_USER_DB_PATH}_\$(hostname -s)"
mkdir -p "\$MN_DB"
apptainer exec --rocm --cleanenv \
    --env HIP_VISIBLE_DEVICES="\$VISIBLE_LIST" \
    --env ROCR_VISIBLE_DEVICES="\$VISIBLE_LIST" \
    --env RN_ARCH="${arch}" \
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

    # cap is already set before this call -> idle/train/cooldown all under cap.
    bash "$PROJECT_PWD/power_analysis/measure_power_multinode.sh" \
        -o "$out" -i 0.5 -b 15 -c 8 -w "$WARMUP" \
        -- \
        srun --ntasks=2 --ntasks-per-node=1 /usr/bin/bash "$LAUNCH"
}

# ── preflight: confirm the cap-set path works on BOTH nodes (don't burn window) ─
echo ""; echo "=== preflight: confirm sudo amd-smi cap-set on both nodes ==="
if set_cap_2node 600; then
    echo "  cap-set OK (set 600 W on both nodes)"
    srun --ntasks=2 --ntasks-per-node=1 /usr/bin/bash -c \
        'echo "[$(hostname -s)] cap readback:"; sudo -n /usr/bin/amd-smi metric -g all --power 2>/dev/null | grep -iE "GPU|cap" | head -12' \
        2>/dev/null || true
else
    echo "FAIL: sudo -n amd-smi set --power-cap denied on the allocation."
    echo "      The multinode cap path is not available; aborting before wasting the window."
    exit 1
fi

# ── sweep loop: model outer (MIOpen warms once), cap inner ────────────────────
for ARCH in $ARCHS; do
    ARCH_DIR="$SWEEP_DIR/$ARCH"; mkdir -p "$ARCH_DIR"
    echo ""; echo "============ model: $ARCH ============"
    for CAP in $CAPS; do
        if set_cap_2node "$CAP"; then
            echo "  cap set to ${CAP} W on both nodes"
        else
            echo "  !! could not set cap ${CAP} W -- running at whatever cap is active"
        fi
        run_point "$ARCH" "$CAP" "$ARCH_DIR/cap_${CAP}"
    done

    echo ""; echo "=== aggregate cap sweep for $ARCH ==="
    apptainer exec "$CONTAINER_IMAGE" python3 \
        "$PROJECT_PWD/power_analysis/build_powercap_summary.py" \
        --sweep-dir "$ARCH_DIR" --label "${ARCH} MI300X 2node" \
      || python3 "$PROJECT_PWD/power_analysis/build_powercap_summary.py" \
        --sweep-dir "$ARCH_DIR" --label "${ARCH} MI300X 2node" \
      || echo "  (summary skipped for $ARCH; run_metrics.csv files in $ARCH_DIR/cap_*/)"
    apptainer exec "$CONTAINER_IMAGE" python3 \
        "$PROJECT_PWD/power_analysis/plot_powercap.py" \
        --run "${ARCH} MI300X 2node=$ARCH_DIR/power_sweep_summary.csv" \
        --out-dir "$ARCH_DIR" --tag "${ARCH}_powercap_2node" \
      || echo "  (plot skipped for $ARCH)"
done

# restore default cap (trap also does this, but be explicit)
srun --ntasks=2 --ntasks-per-node=1 /usr/bin/bash -c \
    "sudo -n /usr/bin/amd-smi reset -g all --power-cap" >/dev/null 2>&1 || true

echo ""; echo "=== 2-node power-cap sweep done $(date) ==="
echo "Outputs: $SWEEP_DIR"
