#!/bin/bash -l
#SBATCH --job-name=rn50_bf16_prof
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --account=l0003099
#SBATCH --partition=gpu:mi300x:2
#SBATCH --gres=gpu:mi300x:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/rn50_bf16_prof_%j.out
#SBATCH --error=logs/rn50_bf16_prof_%j.err
#SBATCH --export=NONE
#
# =============================================================================
# profile_bf16_compare.slurm — kernel-trace (timing) profile of the native
# trainer at 224², FP32 vs BF16, 1 GPU. Goal: quantify per-category GPU time
# (Conv / GEMM / BatchNorm / CASTS / layout / elementwise) so we can see how
# the 66% matmul share changed under BF16 and what the fp32↔bf16 casts cost.
#
# Timing-only --kernel-trace (no PMC counters) — avoids the TCC-fault/retry
# circus of the PMC harness; category time shares don't need FLOP counters.
# Protocol per arm: fresh MIOpen db → warmup epoch (untraced, absorbs FindEx)
# → traced epoch on the 4-class 224² subset (conv shapes = full ImageNet).
# =============================================================================
set -uo pipefail
echo "==== native 224² kernel-trace FP32 vs BF16 | job ${SLURM_JOB_ID:-local} | $(date) ===="

WORKSPACE_BASE="${WORKSPACE_BASE:-$HPC_SCRATCH}"
PROJECT_DIR="${PROJECT_DIR:-$WORKSPACE_BASE/resent50/performance-analysis-dl-models-resnet50}"
CLS="$PROJECT_DIR/ImageNet-1K/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC"
cd "$PROJECT_DIR"
module load cmake 2>/dev/null || true
export ROCM_PATH=/opt/rocm; export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$LD_LIBRARY_PATH"
export TMPDIR="${TMPDIR_PROFILE:-$WORKSPACE_BASE/rocprof_tmp}"; mkdir -p "$TMPDIR"
command -v rocprofv3 >/dev/null 2>&1 || { echo "FAIL: no rocprofv3"; exit 1; }
ulimit -c 0 2>/dev/null || true
export HSA_ENABLE_COREDUMP=0 ROCM_DISABLE_CRASH_DUMP=1 AMD_LOG_LEVEL=0

GFX_ARCH=$(rocminfo 2>/dev/null | awk '/gfx[0-9a-f]+/{for(i=1;i<=NF;i++) if($i ~ /^gfx/){print $i; exit}}'); [ -z "$GFX_ARCH" ] && GFX_ARCH=gfx942

# build (incremental, same dir as the in1k profile harness)
BUILD_DIR="${BUILD_DIR:-$PROJECT_DIR/build_profile_in1k}"; mkdir -p "$BUILD_DIR"
cmake -S "$PROJECT_DIR/hip" -B "$BUILD_DIR" -DCMAKE_HIP_ARCHITECTURES="$GFX_ARCH" -DCMAKE_BUILD_TYPE=Release -DUSE_NCCL=ON > "$BUILD_DIR/cmake.log" 2>&1 || { echo FAIL cmake; tail -20 "$BUILD_DIR/cmake.log"; exit 1; }
cmake --build "$BUILD_DIR" -j"${SLURM_CPUS_PER_TASK:-16}" --target train_multigpu > "$BUILD_DIR/build.log" 2>&1 || { echo FAIL build; tail -20 "$BUILD_DIR/build.log"; exit 1; }
BIN="$BUILD_DIR/train_multigpu"; [ -x "$BIN" ] || { echo "FAIL: no binary"; exit 1; }

# data: 4-class 224² subset
N_CLASSES="${N_CLASSES:-4}"; DATA_DIR="$PROJECT_DIR/imagenet_subset_${N_CLASSES}"
if [ ! -d "$DATA_DIR/train" ] || [ "$(ls "$DATA_DIR/train" 2>/dev/null | wc -l)" -ne "$N_CLASSES" ]; then
    rm -rf "$DATA_DIR"; mkdir -p "$DATA_DIR/train" "$DATA_DIR/val"
    for c in $(ls "$CLS/train" | head -"$N_CLASSES"); do ln -sfn "$CLS/train/$c" "$DATA_DIR/train/$c"; [ -d "$CLS/val/$c" ] && ln -sfn "$CLS/val/$c" "$DATA_DIR/val/$c"; done
fi
LOCAL_BATCH="${LOCAL_BATCH:-256}"; LR="${LR:-0.1}"
ROOT="$PROJECT_DIR/profiles/bf16_prof_${SLURM_JOB_ID:-$$}"; mkdir -p "$ROOT"
echo "  ✓ binary=$BIN  data=$DATA_DIR ($N_CLASSES cls, 224²)  out=$ROOT"

profile_arm() {  # $1=tag  $2=RN50_ENABLE_BF16 value (0=off, 1=boundary, 2=through)
    local tag="$1" bf="$2"
    local db="$ROOT/db_$tag"; rm -rf "$db"; mkdir -p "$db"
    export MIOPEN_USER_DB_PATH="$db" MIOPEN_CUSTOM_CACHE_DIR="$db" MIOPEN_FIND_MODE=3
    if [ "$bf" != "0" ]; then export RN50_ENABLE_BF16="$bf"; else export RN50_ENABLE_BF16=0; fi  # =0 = explicit FP32 (unset now defaults to through)
    local adir="$ROOT/$tag"; mkdir -p "$adir"
    echo ""; echo "## $tag (RN50_ENABLE_BF16=${RN50_ENABLE_BF16:-unset})"
    echo "  [warmup untraced]"
    HIP_VISIBLE_DEVICES=0 timeout -k 30 600 "$BIN" "$DATA_DIR" "$LOCAL_BATCH" 1 "$LR" > "$adir/warmup.log" 2>&1 || echo "  (warmup rc=$?)"
    # The trainer intermittently GPU-faults/hangs in a WrW solver (key
    # 2x512x28x28…) — seen in fp32 AND bf16, worse under rocprofv3. Retry the
    # traced epoch; -k + setsid force SIGKILL on a process group stuck on a
    # hung GPU (the bare `timeout` left an unkillable child → job TIMEOUT).
    local attempt tcsv=""
    for attempt in $(seq 1 "${MAX_RETRY:-4}"); do
        echo "  [traced epoch, attempt $attempt]"
        rm -rf "$adir/trace"
        HIP_VISIBLE_DEVICES=0 timeout -k 30 420 setsid rocprofv3 --kernel-trace --output-format csv -d "$adir/trace" \
            -- "$BIN" "$DATA_DIR" "$LOCAL_BATCH" 1 "$LR" > "$adir/trace.log" 2>&1 || echo "  (trace rc=$?)"
        tcsv=$(find "$adir/trace" -name '*kernel_trace*.csv' -size +100k 2>/dev/null | head -1)
        [ -n "$tcsv" ] && break
        sleep 10
    done
    echo "  trace csv: ${tcsv:-MISSING after ${MAX_RETRY:-4} attempts}"
}

for arm in ${ARMS:-fp32 bf16}; do
    case "$arm" in
        fp32) profile_arm fp32 0 ;;
        bf16) profile_arm bf16 1 ;;
        thru) profile_arm thru 2 ;;
    esac
done

echo ""
echo "Traces in $ROOT — analyze from the login node (aggregate End-Start by kernel name)."
echo "==== done $(date) ===="
