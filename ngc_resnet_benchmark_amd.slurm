#!/bin/bash -l
#SBATCH --job-name=ngc_resnet_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=4G
#SBATCH --account=l0003099
#SBATCH --partition=gpu:mi300x
#SBATCH --gres=gpu:mi300x:8
#SBATCH --time=04:00:00
#SBATCH --output=logs/ngc_resnet_amd_%j.out
#SBATCH --error=logs/ngc_resnet_amd_%j.err
#SBATCH --export=NONE

set -euo pipefail

echo "=== PyTorch NGC ResNet-50 Scaling Benchmark (AMD MI300x, Strong + Weak, 1..8 GPUs) ==="
echo "Job ID         : ${SLURM_JOB_ID:-local-run}"
echo "Node           : ${SLURM_NODELIST:-?}"
echo "Allocated GPUs : ${SLURM_GPUS_ON_NODE:-?} x MI300x"
echo "Start          : $(date)"

# ============================================================================
# 1. MODULES + ROCm ENV
# ----------------------------------------------------------------------------
# Mirrors run_multigpu_imagenet_amd_scaling.slurm. No CUDA modules — we run
# everything inside the rocm/pytorch container (pytorch_rocm.sif) and bind
# data straight from the project tree.
# ============================================================================
echo "Loading modules..."
module --force purge >/dev/null 2>&1 || true
module load cmake >/dev/null 2>&1 || true

export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:/opt/amdgpu/lib64:${LD_LIBRARY_PATH:-}"

PROJECT_PWD="$(pwd)"

# ============================================================================
# 2. DATA — read directly from the repo, no TMPDIR staging
# ----------------------------------------------------------------------------
# ImageNet-1K is pre-extracted under the repo. The Kaggle layout nests data at
# competitions/<comp>/ILSVRC/Data/CLS-LOC/{train,val}. The user-provided
# "imagenet" directory at the repo root is empty; ImageNet-1K/ holds the
# actual unpacked Kaggle archive used by the rest of the AMD scripts.
# ============================================================================
IMAGENET_DATA="${IMAGENET_DATA:-$PROJECT_PWD/ImageNet-1K/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC}"

if [ ! -d "$IMAGENET_DATA/train" ] || [ ! -d "$IMAGENET_DATA/val" ]; then
    echo "ERROR: ImageNet train/val not found under: $IMAGENET_DATA"
    echo "       Override with IMAGENET_DATA=/path/to/CLS-LOC sbatch $0"
    exit 1
fi

echo "Verified dataset structure:"
echo "  - Path        : $IMAGENET_DATA"
echo "  - Train images: $(find "$IMAGENET_DATA/train" -name '*.JPEG' | wc -l)"
echo "  - Val images  : $(find -L "$IMAGENET_DATA/val"   -name '*.JPEG' | wc -l)"
echo ""

# ============================================================================
# 3. CONTAINER + ENTRYPOINT SCRIPT
# ----------------------------------------------------------------------------
# We use a self-contained PyTorch DDP entrypoint (pytorch_resnet50_ddp.py)
# instead of the NGC DeepLearningExamples ConvNets harness. That avoids the
# whole NVIDIA-only dep graph (dllogger, DALI, pynvml/gpu_affinity, apex.contrib
# variants) and keeps us on stock torch + torchvision, which the rocm/pytorch
# container provides out of the box.
# ============================================================================
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$PROJECT_PWD/pytorch_rocm.sif}"
CONTAINER_IMAGE="$(readlink -f "$CONTAINER_IMAGE" 2>/dev/null || echo "$CONTAINER_IMAGE")"
if [ -z "$CONTAINER_IMAGE" ] || [ ! -f "$CONTAINER_IMAGE" ]; then
    echo "ERROR: ROCm PyTorch container not found: '${CONTAINER_IMAGE:-<empty>}'"
    echo "       Build it via: apptainer build pytorch_rocm.sif docker://rocm/pytorch:latest"
    echo "       Or override:  CONTAINER_IMAGE=/abs/path/to/pytorch_rocm.sif sbatch $0"
    exit 1
fi
echo "Container : $CONTAINER_IMAGE"

ENTRYPOINT="${ENTRYPOINT:-$PROJECT_PWD/pytorch_resnet50_ddp.py}"
if [ ! -f "$ENTRYPOINT" ]; then
    echo "ERROR: entrypoint script not found: $ENTRYPOINT"
    exit 1
fi
echo "Entrypoint: $ENTRYPOINT"

# Optional mixed-precision toggle. Set AMP=1 to enable torch.amp autocast.
AMP_FLAG=""
if [ "${AMP:-0}" = "1" ]; then
    AMP_FLAG="--amp"
    echo "AMP       : enabled (torch.amp autocast)"
else
    echo "AMP       : disabled (FP32)"
fi

# ============================================================================
# 4. SWEEP PARAMETERS  (same defaults / semantics as the NVIDIA script)
# ============================================================================
EPOCHS="${EPOCHS:-1}"
WORKERS="${WORKERS:-8}"
WEAK_LOCAL_BATCH="${WEAK_LOCAL_BATCH:-256}"
STRONG_GLOBAL_BATCH="${STRONG_GLOBAL_BATCH:-512}"
GPU_COUNTS="${GPU_COUNTS:-1 2 4 8}"

AVAIL_GPUS="${SLURM_GPUS_ON_NODE:-8}"
echo "Available GPUs  : $AVAIL_GPUS"
echo "Weak local batch: $WEAK_LOCAL_BATCH (global = local * gpus)"
echo "Strong global   : $STRONG_GLOBAL_BATCH (local = global / gpus)"
echo "Epochs / run    : $EPOCHS"

RUN_TAG="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="$PROJECT_PWD/benchmark_results/ngc_amd_${RUN_TAG}"
mkdir -p "$RESULTS_DIR"

WEAK_CSV="$RESULTS_DIR/weak_scaling_summary.csv"
STRONG_CSV="$RESULTS_DIR/strong_scaling_summary.csv"
printf "gpus,local_batch,global_batch,epochs,throughput_img_per_sec,status\n" > "$WEAK_CSV"
printf "gpus,local_batch,global_batch,epochs,throughput_img_per_sec,status\n" > "$STRONG_CSV"
echo "Results dir     : $RESULTS_DIR"

# Persistent MIOpen kernel cache — skip autotune after the first sweep point.
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$HOME/.cache/miopen}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

# Container TMPDIR — the apptainer default tmpfs overlay is too small for
# MIOpen's LLVM JIT (hundreds of MB of intermediate .o files on first kernel
# compile). Bind a disk-backed host dir as the container's TMPDIR.
HOST_TMP="${HOST_TMP:-$PROJECT_PWD/.container_tmp}"
mkdir -p "$HOST_TMP"
echo "Container TMP   : $HOST_TMP"

# RCCL tuning, single-node MI300x XGMI (mirrors run_multigpu_imagenet_amd_scaling.slurm).
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ALGO="${NCCL_ALGO:-Tree}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE:-1}"

BIND_ARGS=(
    --bind "${IMAGENET_DATA}:/data"
    --bind "${ENTRYPOINT}:/workspace/train.py"
    --bind "${RESULTS_DIR}:/results"
    --bind "${MIOPEN_USER_DB_PATH}:${MIOPEN_USER_DB_PATH}"
    --bind "${HOST_TMP}:/container_tmp"
)

# ============================================================================
# 5. RUNNER  (single-node only — torchrun --nproc_per_node=N inside container)
# ----------------------------------------------------------------------------
# Mirrors the single-node path of the NVIDIA script. AMD-specific changes:
#   * apptainer --rocm instead of --nv
#   * HIP_VISIBLE_DEVICES + ROCR_VISIBLE_DEVICES instead of CUDA_VISIBLE_DEVICES
#   * MIOpen cache forwarded so kernels persist across sweep points
#   * --no-checkpoints to avoid disk churn from per-epoch ckpt writes
# Args: <tag> <gpus> <local_batch> <summary_csv>
# ============================================================================
run_one() {
    local tag="$1"
    local gpus="$2"
    local local_batch="$3"
    local summary_csv="$4"

    local global_batch=$((local_batch * gpus))
    local run_dir="$RESULTS_DIR/${tag}_gpu_${gpus}"
    local run_log="$run_dir/train.log"
    mkdir -p "$run_dir"

    if [ "$gpus" -gt "$AVAIL_GPUS" ]; then
        echo "SKIP [${tag}] gpus=$gpus exceeds allocation ($AVAIL_GPUS)"
        printf "%s,%s,%s,%s,%s,%s\n" \
            "$gpus" "$local_batch" "$global_batch" "$EPOCHS" "NA" "skipped" \
            >> "$summary_csv"
        return 0
    fi

    echo ""
    echo "=== [${tag}] Run: gpus=${gpus} local_batch=${local_batch} global_batch=${global_batch} ==="

    local visible_list
    visible_list=$(seq -s, 0 $((gpus - 1)))
    echo "HIP_VISIBLE_DEVICES=$visible_list"

    local status=0
    set +e
    apptainer exec --rocm --cleanenv \
        --env HIP_VISIBLE_DEVICES="$visible_list" \
        --env ROCR_VISIBLE_DEVICES="$visible_list" \
        --env MIOPEN_USER_DB_PATH="$MIOPEN_USER_DB_PATH" \
        --env MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_CUSTOM_CACHE_DIR" \
        --env NCCL_DEBUG="$NCCL_DEBUG" \
        --env NCCL_ALGO="$NCCL_ALGO" \
        --env NCCL_P2P_LEVEL="$NCCL_P2P_LEVEL" \
        --env NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
        --env HSA_FORCE_FINE_GRAIN_PCIE="$HSA_FORCE_FINE_GRAIN_PCIE" \
        --env TMPDIR="/container_tmp" \
        --env TMP="/container_tmp" \
        --env TEMP="/container_tmp" \
        "${BIND_ARGS[@]}" \
        "$CONTAINER_IMAGE" \
        bash -c "
        set -euo pipefail
        python -u -m torch.distributed.run \
            --nproc_per_node=${gpus} \
            --master_port=29500 \
            /workspace/train.py \
            -b ${local_batch} \
            --workers ${WORKERS} \
            --epochs ${EPOCHS} \
            --workspace /results/${tag}_gpu_${gpus} \
            ${AMP_FLAG} \
            /data
        " 2>&1 | tee "$run_log"
    status=${PIPESTATUS[0]}
    set -e

    # pytorch_resnet50_ddp.py emits one line at end:  train.total_ips : <value>
    local throughput=""
    if [ -f "$run_log" ]; then
        throughput=$(awk '/train.total_ips/ {for (i=1; i<=NF; ++i) if ($i == "train.total_ips") v=$(i+2)} END{if(v!="") printf "%.2f", v}' "$run_log")
    fi
    [ -z "$throughput" ] && throughput="NA"

    local stat_str
    if [ "$status" -eq 0 ]; then stat_str="ok"; else stat_str="failed($status)"; fi

    printf "%s,%s,%s,%s,%s,%s\n" \
        "$gpus" "$local_batch" "$global_batch" "$EPOCHS" "$throughput" "$stat_str" \
        >> "$summary_csv"
    echo "Result [${tag}]: gpus=$gpus throughput=${throughput} img/s status=$stat_str"
}

# ============================================================================
# 6. WEAK SCALING SWEEP  (per-GPU batch fixed, global grows)
# ============================================================================
echo ""
echo "########## WEAK SCALING (local_batch=${WEAK_LOCAL_BATCH}) ##########"
for gpus in $GPU_COUNTS; do
    run_one "weak" "$gpus" "$WEAK_LOCAL_BATCH" "$WEAK_CSV"
done

# ============================================================================
# 7. STRONG SCALING SWEEP  (global batch fixed, per-GPU shrinks)
# ============================================================================
echo ""
echo "########## STRONG SCALING (global_batch=${STRONG_GLOBAL_BATCH}) ##########"
for gpus in $GPU_COUNTS; do
    if [ $((STRONG_GLOBAL_BATCH % gpus)) -ne 0 ]; then
        echo "SKIP [strong] gpus=$gpus: STRONG_GLOBAL_BATCH=$STRONG_GLOBAL_BATCH not divisible by $gpus"
        printf "%s,%s,%s,%s,%s,%s\n" \
            "$gpus" "NA" "$STRONG_GLOBAL_BATCH" "$EPOCHS" "NA" "skipped_div" \
            >> "$STRONG_CSV"
        continue
    fi
    local_batch=$((STRONG_GLOBAL_BATCH / gpus))
    run_one "strong" "$gpus" "$local_batch" "$STRONG_CSV"
done

# ============================================================================
# 8. SUMMARY
# ============================================================================
print_table() {
    awk -F, 'NR==1 {printf "%-6s %-12s %-13s %-8s %-20s %-12s\n", "GPUs","LocalBatch","GlobalBatch","Epochs","Throughput(img/s)","Status"; next} {printf "%-6s %-12s %-13s %-8s %-20s %-12s\n",$1,$2,$3,$4,$5,$6}' "$1"
}

echo ""
echo "=== Weak Scaling Summary ==="
print_table "$WEAK_CSV"
echo ""
echo "=== Strong Scaling Summary ==="
print_table "$STRONG_CSV"

echo ""
echo "Weak CSV   : $WEAK_CSV"
echo "Strong CSV : $STRONG_CSV"
echo "Run logs   : $RESULTS_DIR/{weak,strong}_gpu_*/train.log"
echo "Finished   : $(date)"
echo "=== Benchmark Complete ==="
