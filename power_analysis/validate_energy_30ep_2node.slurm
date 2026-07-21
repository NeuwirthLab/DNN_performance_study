#!/bin/bash -l
# ============================================================================
# validate_energy_30ep_2node.slurm  --  30-epoch PyTorch energy validation (2-node)
# ----------------------------------------------------------------------------
# Runs ResNet-50/ImageNet-1K for EPOCHS total epochs (default 35 = 5 warmup +
# 30 measured) across 2 nodes × 8 GPUs = 16 GPUs under the multi-node power
# harness. Produces run_metrics.csv that compose_energy.py validate --nnodes 2
# can use to verify predicted vs measured energy at 16 GPUs.
#
# PyTorch ONLY — native 2-node is not supported (ncclCommInitAll is
# single-process; multi-node native would need a full NCCL rank refactor).
#
# Submit:
#   sbatch --export=NONE power_analysis/validate_energy_30ep_2node.slurm
#
# After the job finishes, validate with:
#   /usr/bin/python3.11 power_analysis/compose_energy.py validate \
#     --coefficients power_analysis/coefficients_final.json \
#     --pytorch-energy power_analysis/results/pytorch/validate30_2node_fp32_<JOBID>/weak_gpu16 \
#     --pytorch-gpus 16 --nnodes 2 \
#     --validation-out power_analysis/validation_2node_v1.json
# ============================================================================
#SBATCH --job-name=rn50_val30_2n
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=4G
#SBATCH --account=l0003099
#SBATCH --partition=acc
#SBATCH --gres=gpu:mi300x:8
#SBATCH --time=01:30:00
#SBATCH --reservation=gaod_powcap
#SBATCH --nodelist=gaod[0001-0002]
#SBATCH --output=power_analysis/results/validate30_2n_%j.out
#SBATCH --error=power_analysis/results/validate30_2n_%j.err
#SBATCH --export=NONE

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
PROJECT_PWD="$(pwd)"

LOCAL_BATCH="${LOCAL_BATCH:-256}"
EPOCHS="${EPOCHS:-35}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
LR="${LR:-0.1}"
WORKERS="${WORKERS:-8}"
AMP="${AMP:-0}"
AMP_FLAG=""; [ "$AMP" = "1" ] && AMP_FLAG="--amp"
PREC_TAG="fp32"; [ "$AMP" = "1" ] && PREC_TAG="amp"

GPUS_PER_NODE=8
TOTAL_GPUS=$(( 2 * GPUS_PER_NODE ))
GLOBAL_BATCH=$(( LOCAL_BATCH * TOTAL_GPUS ))

echo "=== 30-epoch energy validation (pytorch, 2-node × 8 GPU = ${TOTAL_GPUS} GPU) ==="
echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST:-?}  $(date)"
echo "EPOCHS=$EPOCHS  WARMUP_EPOCHS=$WARMUP_EPOCHS  LOCAL_BATCH=$LOCAL_BATCH  GLOBAL_BATCH=$GLOBAL_BATCH  prec=$PREC_TAG"

# ── ROCm env ─────────────────────────────────────────────────────────────────
module --force purge >/dev/null 2>&1 || true
module load cmake    >/dev/null 2>&1 || true
export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:/opt/amdgpu/lib64:${LD_LIBRARY_PATH:-}"

# Reuse the pre-warmed shared 2-node MIOpen cache (gfx942130.ukdb already
# populated from previous sweep runs).  FIND_MODE=3 (Hybrid) tries immediate
# API first, compiles on miss; avoids 15-20 min cold-compile overhead.
# Both nodes share this NFS path; gfx942 ISA is identical across gaod0001/0002
# so compiled binaries are portable between them.
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$PROJECT_PWD/.miopen_pytorch_2n_cache}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
export MIOPEN_FIND_MODE=3
mkdir -p "$MIOPEN_USER_DB_PATH"

# ── RCCL / NCCL env for 2-node ───────────────────────────────────────────────
# IB must be ENABLED for inter-node RCCL traffic (InfiniBand / RoCE).
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"
export HSA_FORCE_FINE_GRAIN_PCIE=1

MASTER_ADDR=$(scontrol show hostnames "${SLURM_NODELIST}" | head -1)
export MASTER_ADDR
echo "MASTER_ADDR=$MASTER_ADDR"

# ── container + entrypoint checks ─────────────────────────────────────────────
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$PROJECT_PWD/pytorch_rocm.sif}"
ENTRYPOINT="${ENTRYPOINT:-$PROJECT_PWD/pytorch_resnet50_ddp.py}"
[ -f "$CONTAINER_IMAGE" ] || { echo "FAIL: container not found: $CONTAINER_IMAGE"; exit 1; }
[ -f "$ENTRYPOINT"      ] || { echo "FAIL: entrypoint not found: $ENTRYPOINT"; exit 1; }
HOST_TMP="$PROJECT_PWD/.container_tmp"; mkdir -p "$HOST_TMP"

IMAGENET_DATA="${IMAGENET_DATA:-$PROJECT_PWD/ImageNet-1K/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC}"
[ -d "$IMAGENET_DATA/val" ] || { echo "FAIL: no val/ at $IMAGENET_DATA"; exit 1; }

# ── output dir ───────────────────────────────────────────────────────────────
OUT_BASE="$PROJECT_PWD/power_analysis/results/pytorch/validate30_2node_${PREC_TAG}_${SLURM_JOB_ID}"
mkdir -p "$OUT_BASE"
OUT="$OUT_BASE/weak_gpu${TOTAL_GPUS}"
mkdir -p "$OUT"
echo "Output: $OUT"

# ── warmup trim ──────────────────────────────────────────────────────────────
# At 16 GPU with full 1.28M ImageNet: 1,281,167 img ÷ ~15,000 img/s ≈ 85s/epoch.
# (The earlier 2-node sweep used val/ 50k subset → 76s was val-subset throughput;
#  full dataset adds DataLoader IO overhead, so use 90s conservative.)
# MIOpen cache is warm so no cold-compile overhead on top of this.
EPOCH_S_EST=90   # conservative; full ImageNet at 16 GPU
WARMUP_S=$(( WARMUP_EPOCHS * EPOCH_S_EST ))
echo "Warmup trim: ${WARMUP_EPOCHS} epochs × ${EPOCH_S_EST}s = ${WARMUP_S}s"

# ── per-node launch script (written to shared NFS for srun to pick up) ───────
LAUNCH="$OUT/_launch_node.sh"
cat > "$LAUNCH" << LAUNCH_EOF
#!/bin/bash
set -uo pipefail
VISIBLE_LIST=\$(seq -s, 0 $(( GPUS_PER_NODE - 1 )))
apptainer exec --rocm --cleanenv \\
    --env HIP_VISIBLE_DEVICES="\$VISIBLE_LIST" \\
    --env ROCR_VISIBLE_DEVICES="\$VISIBLE_LIST" \\
    --env MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH}" \\
    --env MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}" \\
    --env MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE}" \\
    --env NCCL_DEBUG="${NCCL_DEBUG}" \\
    --env NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL}" \\
    --env NCCL_IB_DISABLE="${NCCL_IB_DISABLE}" \\
    --env NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \\
    --env HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE}" \\
    --env MASTER_ADDR="${MASTER_ADDR}" \\
    --env TMPDIR=/container_tmp --env TMP=/container_tmp \\
    --bind "${IMAGENET_DATA}:/data" \\
    --bind "${ENTRYPOINT}:/workspace/train.py" \\
    --bind "${OUT}:/results" \\
    --bind "${MIOPEN_USER_DB_PATH}:${MIOPEN_USER_DB_PATH}" \\
    --bind "${HOST_TMP}:/container_tmp" \\
    "${CONTAINER_IMAGE}" \\
    python -u -m torch.distributed.run \\
        --nnodes=2 \\
        --nproc_per_node=${GPUS_PER_NODE} \\
        --master_addr="${MASTER_ADDR}" \\
        --master_port=29500 \\
        --node_rank=\${SLURM_PROCID} \\
        /workspace/train.py \\
        -b ${LOCAL_BATCH} --workers ${WORKERS} --epochs ${EPOCHS} \\
        --workspace /results ${AMP_FLAG} /data
LAUNCH_EOF
chmod +x "$LAUNCH"

echo "Launch script: $LAUNCH"
echo "Training: ${EPOCHS} epochs × ~${EPOCH_S_EST}s = ~$(( EPOCHS * EPOCH_S_EST ))s"

# ── run under multi-node power harness ───────────────────────────────────────
bash "$PROJECT_PWD/power_analysis/measure_power_multinode.sh" \
    -o "$OUT" -i 0.5 -b 20 -c 10 -w "$WARMUP_S" -E "$WARMUP_EPOCHS" \
    -- \
    srun --ntasks=2 --ntasks-per-node=1 /usr/bin/bash "$LAUNCH"

STATUS=$?
echo ""
echo "=== done $(date)  exit=$STATUS ==="
echo "Output dir: $OUT"
echo ""
echo "To validate energy model at 2-node, run:"
echo "  /usr/bin/python3.11 power_analysis/compose_energy.py validate \\"
echo "    --coefficients power_analysis/coefficients_final.json \\"
echo "    --pytorch-energy $OUT \\"
echo "    --pytorch-gpus ${TOTAL_GPUS} --nnodes 2 \\"
echo "    --validation-out power_analysis/validation_2node_v1.json"
exit "$STATUS"
