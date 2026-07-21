#!/bin/bash -l
# ============================================================================
# validate_energy_30ep.slurm  --  30-epoch validation run for energy model
# ----------------------------------------------------------------------------
# Runs ResNet-50/ImageNet-1K for EPOCHS total epochs (default 35 = 5 warmup +
# 30 measured) under the power harness, producing a clean run_metrics.csv that
# compose_energy.py validate can use to verify predicted vs measured energy.
#
# Runs ONE (stack, GPUS) point per job. Submit 4 jobs for full validation:
#
#   # native 1-GPU  (~11 h)
#   STACK=native GPUS=1 \
#     sbatch --time=12:00:00 power_analysis/validate_energy_30ep.slurm
#
#   # native 8-GPU  (~1.5 h)
#   STACK=native GPUS=8 \
#     sbatch --time=02:30:00 power_analysis/validate_energy_30ep.slurm
#
#   # pytorch 1-GPU  (~10 h)
#   STACK=pytorch GPUS=1 \
#     sbatch --time=12:00:00 power_analysis/validate_energy_30ep.slurm
#
#   # pytorch 8-GPU  (~1.5 h)
#   STACK=pytorch GPUS=8 \
#     sbatch --time=02:30:00 power_analysis/validate_energy_30ep.slurm
#
# After all 4 jobs finish, run compose_energy.py validate (see end of this
# file) using the output directories printed by each job.
# ============================================================================
#SBATCH --job-name=rn50_val30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=4G
#SBATCH --account=l0003099
#SBATCH --partition=acc
#SBATCH --gres=gpu:mi300x:8
#SBATCH --time=12:00:00
#SBATCH --output=power_analysis/results/validate30_%j.out
#SBATCH --error=power_analysis/results/validate30_%j.err
#SBATCH --export=NONE

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
PROJECT_PWD="$(pwd)"

STACK="${STACK:?STACK=native or STACK=pytorch}"
GPUS="${GPUS:-1}"
LOCAL_BATCH="${LOCAL_BATCH:-256}"
EPOCHS="${EPOCHS:-35}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
LR="${LR:-0.1}"
WORKERS="${WORKERS:-8}"
AMP="${AMP:-0}"
PREC_TAG="fp32"; [ "$AMP" = "1" ] && PREC_TAG="amp"

echo "=== 30-epoch energy validation ($STACK, $GPUS GPU) ==="
echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST:-?}  $(date)"
echo "EPOCHS=$EPOCHS  WARMUP_EPOCHS=$WARMUP_EPOCHS  LOCAL_BATCH=$LOCAL_BATCH  AMP=$PREC_TAG"

# ── ROCm env ────────────────────────────────────────────────────────────────
module --force purge >/dev/null 2>&1 || true
module load cmake    >/dev/null 2>&1 || true
export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:/opt/amdgpu/lib64:${LD_LIBRARY_PATH:-}"
# Per-job MIOpen DB: avoids cross-node cache conflicts (node-specific compiled binaries
# are not portable; a stale gfx942 .udb.txt from node A causes "Empty code object path"
# on node B). FIND_MODE=1 (Normal) compiles kernels on demand; overhead is absorbed by
# the WARMUP_EPOCHS discard window.
export MIOPEN_USER_DB_PATH="/tmp/miopen_val30_${SLURM_JOB_ID:-$$}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
export MIOPEN_FIND_MODE=1
mkdir -p "$MIOPEN_USER_DB_PATH"
export NCCL_DEBUG=WARN NCCL_ALGO=Tree NCCL_P2P_LEVEL=SYS \
       NCCL_IB_DISABLE=1 HSA_FORCE_FINE_GRAIN_PCIE=1

AVAIL_GPUS="${SLURM_GPUS_ON_NODE:-8}"
if [ "$GPUS" -gt "$AVAIL_GPUS" ]; then
    echo "FAIL: GPUS=$GPUS > allocated $AVAIL_GPUS"; exit 1
fi
export HIP_VISIBLE_DEVICES="$(seq -s, 0 $((GPUS-1)))"
export ROCR_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"

IMAGENET_DATA="${IMAGENET_DATA:-$PROJECT_PWD/ImageNet-1K/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC}"
[ -d "$IMAGENET_DATA/train" ] || { echo "FAIL: no train/ at $IMAGENET_DATA"; exit 1; }

# ── output dir ──────────────────────────────────────────────────────────────
OUT_BASE="$PROJECT_PWD/power_analysis/results/${STACK}/validate30_${PREC_TAG}_${SLURM_JOB_ID}"
mkdir -p "$OUT_BASE"
OUT="$OUT_BASE/weak_gpu${GPUS}"
echo "Output: $OUT"

# ── warmup trim: 5 epochs worth of seconds (conservative estimate) ───────────
# measure_power.sh -w discards the first W seconds when computing steady_W.
# E_dyn_J (hardware counter) still covers the full window; P_dyn=E_dyn/train_s
# is stable regardless because power is steady after ~60s of GPU load.
# Rule: estimate 1 epoch = 1,281,167 images / expected_throughput
if [ "$GPUS" -eq 1 ]; then
    EPOCH_S_EST=1200    # conservative: ~1100s measured, use 1200s
else
    EPOCH_S_EST=160     # conservative: ~142s measured, use 160s
fi
WARMUP_S=$(( WARMUP_EPOCHS * EPOCH_S_EST ))
echo "Warmup trim: ${WARMUP_EPOCHS} epochs × ${EPOCH_S_EST}s = ${WARMUP_S}s"

GLOBAL_BATCH=$(( LOCAL_BATCH * GPUS ))

# ── stack-specific launch ────────────────────────────────────────────────────
if [ "$STACK" = "native" ]; then
    BINARY="${BINARY:-$PROJECT_PWD/build_profile/train_multigpu}"
    if [ ! -x "$BINARY" ]; then
        echo "Building train_multigpu ..."
        GFX_ARCH=$(rocminfo 2>/dev/null | awk '/gfx[0-9a-f]+/{for(i=1;i<=NF;i++) if($i~/^gfx/){print $i;exit}}')
        [ -z "$GFX_ARCH" ] && GFX_ARCH="gfx942"
        BD="$(dirname "$BINARY")"; mkdir -p "$BD"
        cmake -S "$PROJECT_PWD/hip" -B "$BD" -DCMAKE_HIP_ARCHITECTURES="$GFX_ARCH" \
              -DCMAKE_BUILD_TYPE=Release -DUSE_NCCL=ON -DUSE_TURBOJPEG=OFF \
              >"$BD/cmake.log" 2>&1 \
          && cmake --build "$BD" -j16 --target train_multigpu >"$BD/build.log" 2>&1
        [ -x "$BINARY" ] || { echo "FAIL: build"; tail -40 "$BD/build.log"; exit 1; }
    fi
    # Force FP32 + NCHW mode. The current binary defaults to:
    #   RN50_ENABLE_BF16 unset → BF16 through-mode (crashes: MIOpen has no BF16 BN inference)
    #   RN50_NHWC        unset → NHWC layout (crashes: out_desc_ NHWC but bn_desc_ hardcoded
    #                            NCHW → mixed-layout BN variant missing compiled binary in DB)
    # Both must be pinned to match the old binary that produced the baseline results.
    export RN50_ENABLE_BF16=0
    export RN50_NHWC=0
    echo "Binary: $BINARY  global_batch=$GLOBAL_BATCH  RN50_ENABLE_BF16=$RN50_ENABLE_BF16  RN50_NHWC=$RN50_NHWC"
    bash "$PROJECT_PWD/power_analysis/measure_power.sh" \
        -o "$OUT" -i 0.5 -b 20 -c 10 -w "$WARMUP_S" -g all \
        -- stdbuf -oL "$BINARY" "$IMAGENET_DATA" "$GLOBAL_BATCH" "$EPOCHS" "$LR"

elif [ "$STACK" = "pytorch" ]; then
    CONTAINER_IMAGE="${CONTAINER_IMAGE:-$PROJECT_PWD/pytorch_rocm.sif}"
    ENTRYPOINT="${ENTRYPOINT:-$PROJECT_PWD/pytorch_resnet50_ddp.py}"
    [ -f "$CONTAINER_IMAGE" ] || { echo "FAIL: container not found: $CONTAINER_IMAGE"; exit 1; }
    [ -f "$ENTRYPOINT" ]     || { echo "FAIL: entrypoint not found: $ENTRYPOINT"; exit 1; }
    HOST_TMP="$PROJECT_PWD/.container_tmp"; mkdir -p "$HOST_TMP"
    AMP_FLAG=""; [ "$AMP" = "1" ] && AMP_FLAG="--amp"
    mkdir -p "$OUT"
    echo "Container: $CONTAINER_IMAGE  local_batch=$LOCAL_BATCH"
    bash "$PROJECT_PWD/power_analysis/measure_power.sh" \
        -o "$OUT" -i 0.5 -b 20 -c 10 -w "$WARMUP_S" -g all \
        -- apptainer exec --rocm --cleanenv \
            --env HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
            --env ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
            --env MIOPEN_USER_DB_PATH="$MIOPEN_USER_DB_PATH" \
            --env MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_CUSTOM_CACHE_DIR" \
            --env NCCL_DEBUG="$NCCL_DEBUG" --env NCCL_ALGO="$NCCL_ALGO" \
            --env NCCL_P2P_LEVEL="$NCCL_P2P_LEVEL" \
            --env NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
            --env HSA_FORCE_FINE_GRAIN_PCIE="$HSA_FORCE_FINE_GRAIN_PCIE" \
            --env TMPDIR=/container_tmp --env TMP=/container_tmp --env TEMP=/container_tmp \
            --bind "${IMAGENET_DATA}:/data" \
            --bind "${ENTRYPOINT}:/workspace/train.py" \
            --bind "${OUT}:/results" \
            --bind "${MIOPEN_USER_DB_PATH}:${MIOPEN_USER_DB_PATH}" \
            --bind "${HOST_TMP}:/container_tmp" \
            "$CONTAINER_IMAGE" \
            bash -c "set -uo pipefail
                     python -u -m torch.distributed.run \
                         --nproc_per_node=${GPUS} --master_port=29500 \
                         /workspace/train.py \
                         -b ${LOCAL_BATCH} --workers ${WORKERS} \
                         --epochs ${EPOCHS} --workspace /results \
                         ${AMP_FLAG} /data"
else
    echo "FAIL: STACK must be native or pytorch"; exit 1
fi

STATUS=$?
echo ""
echo "=== done $(date)  exit=$STATUS ==="
echo "Output dir: $OUT"
echo ""
echo "To validate (after all 4 jobs finish), run:"
echo "  /usr/bin/python3.11 power_analysis/compose_energy.py validate \\"
echo "    --coefficients power_analysis/coefficients_final.json \\"
echo "    --native-energy  power_analysis/results/native/validate30_fp32_NATIVE1GPU_JOBID/weak_gpu1 \\"
echo "    --pytorch-energy power_analysis/results/pytorch/validate30_fp32_PYTORCH1GPU_JOBID/weak_gpu1 \\"
echo "    --native-gpus 1 --pytorch-gpus 1 \\"
echo "    --validation-out power_analysis/validation_1gpu_final.json"
exit "$STATUS"
