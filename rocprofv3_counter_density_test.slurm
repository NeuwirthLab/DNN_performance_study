#!/bin/bash -l
#SBATCH --job-name=rn50_pmcdensity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --account=l0003099
#SBATCH --partition=gpu:mi300x
#SBATCH --gres=gpu:mi300x:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/pmc_density_%j.out
#SBATCH --error=logs/pmc_density_%j.err
#SBATCH --export=NONE
#
# =============================================================================
# rocprofv3_counter_density_test.slurm — falsifier for the rocprof-compute
# failure (jobs 52547742/52548465): hypothesis = the GPU fault is caused by
# COUNTER DENSITY per pass (rocprof-compute packs 28 counters into one pmc
# line; our working raw-rocprofv3 runs used ≤2), not by the tool itself.
#
# Runs raw rocprofv3 on the native trainer with the EXACT pmc_perf_0 line from
# rocprof-compute, then progressively thinner prefixes: 28, 14, 8, 4, 2
# counters. Each gets MAX_TRY=2 attempts with a hard timeout. Output: a
# verdict table "n_counters → ok/fault" mapping the density threshold.
# =============================================================================
set -uo pipefail
echo "==== rocprofv3 counter-density falsifier | job ${SLURM_JOB_ID:-local} | $(date) ===="

WORKSPACE_BASE="${WORKSPACE_BASE:-$HPC_SCRATCH}"
PROJECT_DIR="${PROJECT_DIR:-$WORKSPACE_BASE/resent50/performance-analysis-dl-models-resnet50}"
CLS="$PROJECT_DIR/ImageNet-1K/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC"
cd "$PROJECT_DIR"
module load cmake 2>/dev/null || true
export ROCM_PATH=/opt/rocm; export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:${LD_LIBRARY_PATH:-}"
export TMPDIR="${TMPDIR_PROFILE:-$WORKSPACE_BASE/rocprof_tmp}"; mkdir -p "$TMPDIR"
ulimit -c 0 2>/dev/null || true
export HSA_ENABLE_COREDUMP=0 ROCM_DISABLE_CRASH_DUMP=1 AMD_LOG_LEVEL=0
command -v rocprofv3 >/dev/null 2>&1 || { echo "FAIL: no rocprofv3"; exit 1; }

GFX_ARCH=$(rocminfo 2>/dev/null | awk '/gfx[0-9a-f]+/{for(i=1;i<=NF;i++) if($i ~ /^gfx/){print $i; exit}}'); [ -z "$GFX_ARCH" ] && GFX_ARCH=gfx942

MIOPEN_DB="$WORKSPACE_BASE/miopen_pmcdensity_${SLURM_JOB_ID:-$$}"; rm -rf "$MIOPEN_DB"; mkdir -p "$MIOPEN_DB"
export MIOPEN_USER_DB_PATH="$MIOPEN_DB" MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_DB" MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-2}"

BUILD_DIR="${BUILD_DIR:-$PROJECT_DIR/build_profile_in1k}"; mkdir -p "$BUILD_DIR"
cmake -S "$PROJECT_DIR/hip" -B "$BUILD_DIR" -DCMAKE_HIP_ARCHITECTURES="$GFX_ARCH" -DCMAKE_BUILD_TYPE=Release -DUSE_NCCL=ON > "$BUILD_DIR/cmake.log" 2>&1 || { echo FAIL cmake; exit 1; }
cmake --build "$BUILD_DIR" -j"${SLURM_CPUS_PER_TASK:-16}" --target train_multigpu > "$BUILD_DIR/build.log" 2>&1 || { echo FAIL build; exit 1; }
BIN="$BUILD_DIR/train_multigpu"

N_CLASSES=4; DATA_DIR="$PROJECT_DIR/imagenet_subset_${N_CLASSES}"
if [ ! -d "$DATA_DIR/train" ] || [ "$(ls "$DATA_DIR/train" 2>/dev/null | wc -l)" -ne "$N_CLASSES" ]; then
    rm -rf "$DATA_DIR"; mkdir -p "$DATA_DIR/train" "$DATA_DIR/val"
    for c in $(ls "$CLS/train" | head -"$N_CLASSES"); do ln -sfn "$CLS/train/$c" "$DATA_DIR/train/$c"; [ -d "$CLS/val/$c" ] && ln -sfn "$CLS/val/$c" "$DATA_DIR/val/$c"; done
fi
LOCAL_BATCH=256; LR=0.1
export RN50_PROGRESS_INTERVAL=10
ROOT="$PROJECT_DIR/profiles/pmc_density_${SLURM_JOB_ID:-$$}"; mkdir -p "$ROOT"

echo "  [warmup] …"
HIP_VISIBLE_DEVICES=0 "$BIN" "$DATA_DIR" "$LOCAL_BATCH" 1 "$LR" > "$ROOT/warmup.log" 2>&1 || echo "  (warmup rc=$?)"

# the EXACT 28-counter line rocprof-compute generated as pmc_perf_0 (job 52548465)
ALL28="SQ_CYCLES SQ_BUSY_CYCLES SQ_BUSY_CU_CYCLES SQ_WAVES SQ_WAVE_CYCLES SQC_TC_INST_REQ SQC_TC_DATA_READ_REQ SQC_TC_DATA_WRITE_REQ TA_TA_BUSY_sum TA_BUFFER_WAVEFRONTS_sum TD_TD_BUSY_sum TD_TC_STALL_sum TCP_GATE_EN1_sum TCP_GATE_EN2_sum TCP_TD_TCP_STALL_CYCLES_sum TCP_TCR_TCP_STALL_CYCLES_sum TCC_CYCLE_sum TCC_BUSY_sum TCC_PROBE_sum TCC_PROBE_ALL_sum CPC_CPC_STAT_BUSY CPC_CPC_STAT_IDLE CPF_CPF_STAT_BUSY CPF_CPF_STAT_STALL SPI_CSN_WINDOW_VALID SPI_CSN_BUSY GRBM_COUNT GRBM_GUI_ACTIVE"

PERLINE_TIMEOUT="${PERLINE_TIMEOUT:-150}"; MAX_TRY="${MAX_TRY:-2}"; KIR="1-50"
declare -A VERDICT
for n in 28 14 8 4 2; do
    line=$(echo "$ALL28" | tr ' ' '\n' | head -"$n" | tr '\n' ' ')
    VERDICT[$n]="fault"
    for try in $(seq 1 "$MAX_TRY"); do
        t="$ROOT/n${n}_try${try}"; rm -rf "$t"; mkdir -p "$t"
        echo "pmc: $line" > "$t/pmc.txt"
        echo "--- n=$n try=$try ---"
        HIP_VISIBLE_DEVICES=0 timeout -k 15 "$PERLINE_TIMEOUT" \
            rocprofv3 -i "$t/pmc.txt" --kernel-iteration-range "$KIR" \
            --output-format csv -d "$t/out" \
            -- stdbuf -oL "$BIN" "$DATA_DIR" "$LOCAL_BATCH" 1 "$LR" > "$t/log" 2>&1
        rc=$?
        csv=$(find "$t/out" -name '*_counter_collection.csv' 2>/dev/null | head -1)
        if [ -n "$csv" ] && [ "$(wc -l < "$csv")" -gt 1 ]; then
            echo "    ✓ n=$n OK (try $try, $(wc -l < "$csv") rows)"; VERDICT[$n]="ok"
            break
        fi
        reason=$(grep -aoE "HW Exception.*|Memory access fault.*|Unable to configure.*|malformed.*" "$t/log" | head -1)
        [ "$rc" = 124 ] || [ "$rc" = 137 ] && reason="${reason:-hang (timeout)}"
        echo "    ✗ n=$n try=$try rc=$rc ${reason:+— $reason}"
        pkill -9 -f "$BIN" 2>/dev/null || true; pkill -9 -f rocprofv3 2>/dev/null || true; sleep 5
    done
done

echo ""
echo "=== VERDICT: counters-per-pass → result ==="
for n in 28 14 8 4 2; do printf "  %2d counters : %s\n" "$n" "${VERDICT[$n]}"; done
echo "=== done $(date) | logs in $ROOT ==="
