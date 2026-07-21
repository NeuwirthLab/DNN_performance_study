#!/bin/bash -l
# ============================================================================
# powercap_sweep.slurm  --  GPU power-cap (DVFS) energy sweep for ResNet-50
# ----------------------------------------------------------------------------
# For each cap in CAPS: set the GPU power cap -> run the workload under
# measure_power.sh -> reset. Then build_powercap_summary.py emits the
# power_sweep_summary.csv that plot_powercap.py consumes.
#
# *** POWER-CAP PRIVILEGE: via the av30nuci reservation on gaod0002 ***
# `amd-smi set --power-cap` returns AMDSMI_STATUS_NO_PERM for a normal user, but
# `rocm-smi --setpoweroverdrive` (set_cap below) DOES work, and running on the
# DEDICATED reserved node gaod0002 (reservation=gaod_powcap) means the cap affects
# only this job's GPUs -- no neighbour workload perturbs the measurement. Earlier
# this sweep could only capture the default-cap point (job 52410392, shared node);
# the reservation unblocks the full cap sweep.
#
# Submit (single node, gaod0002, PyTorch FP32 cap sweep):
#   STACK=pytorch GPUS=8 sbatch power_analysis/powercap_sweep.slurm
# Override node / reservation on the CLI if needed:
#   sbatch --nodelist=gaod0001 --reservation=gaod_powcap power_analysis/powercap_sweep.slurm
# ============================================================================
#SBATCH --job-name=rn50_powercap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=4G
#SBATCH --account=l0003099
#SBATCH --partition=acc
#SBATCH --gres=gpu:mi300x:8
#SBATCH --time=04:00:00
#SBATCH --reservation=gaod_powcap
#SBATCH --nodelist=gaod0002
#SBATCH --output=power_analysis/results/powercap_%j.out
#SBATCH --error=power_analysis/results/powercap_%j.err
#SBATCH --export=NONE
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"; PROJECT_PWD="$(pwd)"
export PATH=/opt/rocm/bin:$PATH

STACK="${STACK:?STACK=native|pytorch}"
GPUS="${GPUS:-8}"
LOCAL_BATCH="${LOCAL_BATCH:-256}"
# epoch-based measurement: run EPOCHS total, discard the first WARMUP_EPOCHS as
# warmup, measure (and average) the rest. Default 5 epochs = 1 warmup + 4 measured.
EPOCHS="${EPOCHS:-5}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
WORKERS="${WORKERS:-8}"
# precision: fp32 (default) or bf16. bf16 => native RN50_ENABLE_BF16=1 (boundary:
# convs in BF16, BN/loss FP32) and pytorch --amp --amp-dtype bfloat16 -- matched
# autocast semantics across both stacks.
PRECISION="${PRECISION:-fp32}"
PT_AMP=""
export RN50_ENABLE_BF16="${RN50_ENABLE_BF16:-0}"
if [ "$PRECISION" = bf16 ]; then
    export RN50_ENABLE_BF16="${BF16_MODE:-1}"   # 1=boundary, 2=through
    PT_AMP="--amp --amp-dtype bfloat16"
fi
export RN50_NHWC="${RN50_NHWC:-0}"              # 1 = native channels-last (NHWC)
echo "### PRECISION=$PRECISION  RN50_ENABLE_BF16=$RN50_ENABLE_BF16  PT_AMP='$PT_AMP'"
CAPS="${CAPS:-750 700 650 600 550 500}"   # W per GPU, high->low
SWEEP_DIR="$PROJECT_PWD/power_analysis/results/${STACK}/powercap_${SLURM_JOB_ID}"
mkdir -p "$SWEEP_DIR"

# ---- env / workload selection (mirrors sweep_energy.slurm) ------------------
module load cmake >/dev/null 2>&1 || true
export ROCM_PATH=/opt/rocm LD_LIBRARY_PATH="/opt/rocm/lib:/opt/amdgpu/lib64:${LD_LIBRARY_PATH:-}"
# per-JOB MIOpen cache so parallel jobs (e.g. two nodes at once) never share one
# sqlite DB -> avoids the "Duplicate ID" corruption from concurrent writes.
export MIOPEN_USER_DB_PATH="$PROJECT_PWD/.miopen_${STACK}_${SLURM_JOB_ID}_cache"
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-3}"   # 3=FAST (default); 1=exhaustive (NHWC needs full solver set)
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"; mkdir -p "$MIOPEN_USER_DB_PATH"
export NCCL_DEBUG=WARN NCCL_ALGO=Tree NCCL_P2P_LEVEL=SYS NCCL_IB_DISABLE=1 HSA_FORCE_FINE_GRAIN_PCIE=1
IMAGENET_DATA="${IMAGENET_DATA:-$PROJECT_PWD/ImageNet-1K/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC}"
# disk-backed container TMPDIR for MIOpen's LLVM JIT (the apptainer tmpfs overlay
# is tiny -> "No space left on device" when MIOpen compiles kernels).
WORKERS="${WORKERS:-8}"
HOST_TMP="$PROJECT_PWD/.container_tmp"; mkdir -p "$HOST_TMP"

# build the native binary if missing (train_multigpu.cu / resnet50.cu)
NATIVE_BIN="$PROJECT_PWD/build_profile/train_multigpu"
if [ "$STACK" = native ] && [ ! -x "$NATIVE_BIN" ]; then
    echo "Building train_multigpu ..."
    GFX=$(rocminfo 2>/dev/null | awk '/gfx[0-9a-f]+/{for(i=1;i<=NF;i++) if($i ~ /^gfx/){print $i; exit}}'); [ -z "$GFX" ] && GFX=gfx942
    cmake -S "$PROJECT_PWD/hip" -B "$PROJECT_PWD/build_profile" -DCMAKE_HIP_ARCHITECTURES="$GFX" \
          -DCMAKE_BUILD_TYPE=Release -DUSE_NCCL=ON -DUSE_TURBOJPEG=OFF > "$PROJECT_PWD/build_profile/cmake_pc.log" 2>&1 \
      && cmake --build "$PROJECT_PWD/build_profile" -j16 --target train_multigpu > "$PROJECT_PWD/build_profile/build_pc.log" 2>&1
    [ -x "$NATIVE_BIN" ] || { echo "FAIL: native build"; tail -40 "$PROJECT_PWD/build_profile/build_pc.log"; exit 1; }
    echo "  built train_multigpu"
fi
GLOBAL_BATCH=$((LOCAL_BATCH * GPUS))
GPU_IDS="$(seq -s, 0 $((GPUS-1)))"
export HIP_VISIBLE_DEVICES="$GPU_IDS"
export ROCR_VISIBLE_DEVICES="$GPU_IDS"

set_cap() {   # $1 = watts ; rocm-smi overdrive works for a NORMAL user (amd-smi
              # --power-cap returns NO_PERM and silently floors low caps). Sets all
              # GPUs on the node at once; returns non-zero if not permitted.
    /usr/bin/rocm-smi --setpoweroverdrive "$1" --autorespond yes >/dev/null 2>&1
}
# ALWAYS restore the default 750 W cap on any exit (timeout / scancel / error) so
# the next reservation user does NOT inherit a throttled GPU on this dedicated node.
trap 'echo "[trap] restoring 750 W cap"; /usr/bin/rocm-smi --setpoweroverdrive 750 --autorespond yes >/dev/null 2>&1 || true' EXIT INT TERM

run_workload() {  # $1 = out dir
    if [ "$STACK" = native ]; then
        bash "$PROJECT_PWD/power_analysis/measure_power.sh" -o "$1" -i 0.5 -b 15 -c 8 -w 45 -E "$WARMUP_EPOCHS" -g all \
          -- env RN50_ENABLE_BF16="$RN50_ENABLE_BF16" RN50_NHWC="$RN50_NHWC" stdbuf -oL "$PROJECT_PWD/build_profile/train_multigpu" "$IMAGENET_DATA" "$GLOBAL_BATCH" "$EPOCHS" 0.1
    else
        bash "$PROJECT_PWD/power_analysis/measure_power.sh" -o "$1" -i 0.5 -b 15 -c 8 -w 60 -E "$WARMUP_EPOCHS" -g all \
          -- apptainer exec --rocm --cleanenv \
              --env HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" --env ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
              --env MIOPEN_USER_DB_PATH="$MIOPEN_USER_DB_PATH" --env MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_CUSTOM_CACHE_DIR" \
              --env TMPDIR=/container_tmp --env TMP=/container_tmp --env TEMP=/container_tmp \
              --bind "${IMAGENET_DATA}/val:/data/train" --bind "${IMAGENET_DATA}/val:/data/val" \
              --bind "$PROJECT_PWD/pytorch_resnet50_ddp.py:/workspace/train.py" \
              --bind "$1:/results" --bind "${MIOPEN_USER_DB_PATH}:${MIOPEN_USER_DB_PATH}" \
              --bind "${HOST_TMP}:/container_tmp" \
              "$PROJECT_PWD/pytorch_rocm.sif" \
              bash -c "python -u -m torch.distributed.run --nproc_per_node=${GPUS} --master_port=29500 \
                       /workspace/train.py -b ${LOCAL_BATCH} --workers ${WORKERS} --epochs ${EPOCHS} ${PT_AMP} --workspace /results /data"
    fi
}

echo "=== power-cap sweep ($STACK, ${GPUS} GPU) caps=[$CAPS] ==="
for CAP in $CAPS; do
    echo ""; echo "######## cap = ${CAP} W ########"
    if set_cap "$CAP"; then
        echo "  cap set to ${CAP} W"
    else
        echo "  !! could not set cap to ${CAP} W (permission?) -- running at whatever cap is active"
    fi
    run_workload "$SWEEP_DIR/cap_${CAP}"
done
set_cap 750 || true   # restore default

echo ""; echo "=== build summary + plots ==="
apptainer exec "$PROJECT_PWD/pytorch_rocm.sif" python3 "$PROJECT_PWD/power_analysis/build_powercap_summary.py" \
    --sweep-dir "$SWEEP_DIR" --label "${STACK} MI300X"
apptainer exec "$PROJECT_PWD/pytorch_rocm.sif" python3 "$PROJECT_PWD/power_analysis/plot_powercap.py" \
    --run "${STACK} MI300X=$SWEEP_DIR/power_sweep_summary.csv" \
    --out-dir "$SWEEP_DIR" --tag "${STACK}_powercap" || true
echo "=== done; outputs in $SWEEP_DIR ==="
