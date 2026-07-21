#!/bin/bash -l
#SBATCH --job-name=rn50_profile_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --account=l0003099
#SBATCH --gres=gpu:mi300x:1
#SBATCH --time=04:00:00   # ~10 min/pass x 15 pmc: lines => ~2.5h; 2h gets killed
#SBATCH --output=logs/rn50_profile_amd_%j.out
#SBATCH --error=logs/rn50_profile_amd_%j.err
#SBATCH --export=NONE

set -uo pipefail

echo "========================================================"
echo " ResNet-50 MI300X Profiling"
echo " Job   : $SLURM_JOB_ID"
echo " Node  : $SLURM_NODELIST"
echo " Start : $(date)"
echo "========================================================"

# ── Paths ───────────────────────────────────────────────────────────────────
WORKSPACE_BASE="${WORKSPACE_BASE:-$HPC_SCRATCH}"
PROJECT_DIR="${PROJECT_DIR:-$WORKSPACE_BASE/resent50/performance-analysis-dl-models-resnet50}"
TINY_DATA_DIR="${PROJECT_DIR}/tiny-imagenet-200"

if [ ! -d "$TINY_DATA_DIR/train" ] || [ ! -d "$TINY_DATA_DIR/val" ]; then
    echo "FAIL: tiny-imagenet-200 not found at $TINY_DATA_DIR"; exit 1
fi
echo "  ✓ tiny data : $TINY_DATA_DIR"

# ── Modules + ROCm ──────────────────────────────────────────────────────────
module load cmake
export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"

# ── rocprofv3 (system tool, no venv needed) ─────────────────────────────────
# We use rocprofv3 directly (not rocprof-compute) because rocprof-compute v3.1
# still drives rocprofv1 internally, which hangs/segfaults on MI300X + MIOpen
# (deprecated upstream). rocprofv3 uses callback-based instrumentation and
# runs the same workload to completion.
if ! command -v rocprofv3 >/dev/null 2>&1; then
    echo "FAIL: rocprofv3 not on PATH"; exit 1
fi
echo "  ✓ rocprofv3 : $(command -v rocprofv3)"

# ── Node diagnostic: partition mode + visible-devices mask ──────────────────
# These are the prime suspects when the same rocprofv3 command works in an
# interactive shell on one node but fails under sbatch on another. Print at
# top of every job so the .out file always tells us what HW was used.
echo "=== Node diagnostic ==="
echo "  ROCR_VISIBLE_DEVICES = ${ROCR_VISIBLE_DEVICES:-<unset>}"
echo "  HIP_VISIBLE_DEVICES  = ${HIP_VISIBLE_DEVICES:-<unset>}"
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-<unset>}"
rocm-smi --showcomputepartition 2>/dev/null | sed 's/^/  /' || \
    echo "  (rocm-smi --showcomputepartition not supported on this ROCm)"
rocminfo 2>/dev/null | awk '/^Agent/{ag=$0} /Name:.*gfx/{print "  agent:", ag; print "  arch :", $0} /Compute Unit:/{print "  CUs  :", $0}' | head -20
echo "======================="

# Redirect LLVM/HIP/MIOpen JIT scratch off the small node /tmp onto scratch.
export TMPDIR="${TMPDIR_PROFILE:-$WORKSPACE_BASE/rocprof_tmp}"
mkdir -p "$TMPDIR"
echo "  ✓ TMPDIR    : $TMPDIR"

# ── MIOpen / RCCL env ───────────────────────────────────────────────────────
# Keep this set MINIMAL — it matches what the verified manual rocprofv3 run
# used. Earlier we set MIOPEN_FIND_MODE=NORMAL and a 2 GB workspace override
# plus several NCCL knobs; those weren't in the working manual command and
# coincided with a "Memory access fault by GPU" under rocprofv3, so we drop
# them. The MIOpen cache lives in scratch so we never reuse a cache that
# was corrupted by earlier rocprofv1 crashes (which left "Duplicate ID"
# entries in ~/.cache/miopen*).
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$WORKSPACE_BASE/miopen_profile_db}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
echo "  ✓ MIOpen db : $MIOPEN_USER_DB_PATH"

# ── Build ───────────────────────────────────────────────────────────────────
GFX_ARCH=$(rocminfo 2>/dev/null | awk '/gfx[0-9a-f]+/{for(i=1;i<=NF;i++) if($i ~ /^gfx/){print $i; exit}}')
[ -z "$GFX_ARCH" ] && GFX_ARCH="gfx942"
echo "  ✓ GPU arch  : $GFX_ARCH"

# Build into a PERSISTENT path (was $TMPDIR/... — node-local /tmp is wiped
# between jobs, which broke any rocprof-compute call that ran outside this
# script and pointed at the previous job's binary).
BUILD_DIR="${BUILD_DIR:-$PROJECT_DIR/build_profile}"
mkdir -p "$BUILD_DIR"

cmake -S "$PROJECT_DIR/hip" -B "$BUILD_DIR" \
      -DCMAKE_HIP_ARCHITECTURES="$GFX_ARCH" \
      -DCMAKE_BUILD_TYPE=Release \
      -DUSE_NCCL=ON \
      > "$BUILD_DIR/cmake.log" 2>&1
if [ $? -ne 0 ]; then
    echo "  FAIL: cmake"; tail -40 "$BUILD_DIR/cmake.log"; exit 1
fi

# Incremental build: only recompiles changed sources. Use REBUILD=1 to force.
if [ "${REBUILD:-0}" = "1" ]; then
    cmake --build "$BUILD_DIR" --target clean > /dev/null 2>&1 || true
fi
cmake --build "$BUILD_DIR" -j16 --target train_multigpu_tiny \
      > "$BUILD_DIR/build.log" 2>&1
if [ $? -ne 0 ]; then
    echo "  FAIL: build"; tail -60 "$BUILD_DIR/build.log"; exit 1
fi
if [ ! -x "$BUILD_DIR/train_multigpu_tiny" ]; then
    echo "  FAIL: binary missing after build at $BUILD_DIR/train_multigpu_tiny"
    tail -40 "$BUILD_DIR/build.log"; exit 1
fi
echo "  ✓ built: $BUILD_DIR/train_multigpu_tiny ($(du -h "$BUILD_DIR/train_multigpu_tiny" | awk '{print $1}'))"

# ── Profile (rocprofv3 multi-pass via -i pmc.txt) ───────────────────────────
PROFILE_ROOT="$PROJECT_DIR/profiles/amd_${SLURM_JOB_ID}"
mkdir -p "$PROFILE_ROOT"

GLOBAL_BATCH="${GLOBAL_BATCH:-64}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-0.1}"

# --kernel-iteration-range bounds per-pass work; collect counters on the
# first N iterations of each unique kernel name (set ROCPROF_KIR="" for all).
ROCPROF_KIR="${ROCPROF_KIR:-1-50}"

# 2 counters per pass, all from the same hardware block — guaranteed to
# fit in one MI300X HW pass. rocprofv3 re-runs the workload once per
# `pmc:` line, producing one *_counter_collection.csv per pass.
# Override the whole set via ROCPROF_PMC_FILE=/path/to/your.txt.
PMC_FILE="${ROCPROF_PMC_FILE:-$PROFILE_ROOT/pmc.txt}"
if [ ! -f "$PMC_FILE" ] || [ "$PMC_FILE" = "$PROFILE_ROOT/pmc.txt" ]; then
    cat > "$PMC_FILE" <<'EOF'
# All names verified against `rocprofv3 --list-avail` on gfx942 (MI300X).
# An invalid name silently poisons (drops) its whole pmc: line, taking any
# valid counter on the same line with it. Removed vs the old set:
#   SQC_HITS / SQC_MISSES        -> no scalar-L1 hit counters exist on gfx942
#   SQ_BUSY_CYCLES               -> real name is SQ_BUSY_CU_CYCLES
#   SQ_WAIT_ANY, SQ_WAIT_INST_LDS-> do not exist on this build
#   SQ_INSTS_VALU_MFMA_MOPS_I8   -> only F16/BF16/F32/F64 MFMA MOPS exist
# Keep counts modest: rocprofv3 repacks all counters into the fewest passes
# and silently truncates the tail if it runs out of budget.
# ── VALU compute throughput ────────────────────────────────────────────
pmc: SQ_WAVES SQ_INSTS_VALU_FMA_F32
pmc: SQ_INSTS_VALU_ADD_F32 SQ_INSTS_VALU_MUL_F32
pmc: SQ_INSTS_VALU_TRANS_F32
pmc: SQ_INSTS_VALU_FMA_F16 SQ_INSTS_VALU_ADD_F16
pmc: SQ_INSTS_VALU_MUL_F16 SQ_INSTS_VALU_TRANS_F16
# ── MFMA / matrix-core FLOPs: MfmaFlops = (F16+BF16+F32+F64)*512.
# FP32 nonzero, FP16/BF16 = 0 in the FP32 trainer = the "no tensor cores"
# evidence, and gives real matrix-core FLOPs for the roofline.
pmc: SQ_INSTS_VALU_MFMA_MOPS_F32 SQ_INSTS_VALU_MFMA_MOPS_F16
pmc: SQ_INSTS_VALU_MFMA_MOPS_BF16 SQ_INSTS_VALU_MFMA_MOPS_F64
# ── HBM traffic ────────────────────────────────────────────────────────
pmc: TCC_EA0_RDREQ_sum TCC_EA0_WRREQ_sum
pmc: TCC_EA0_RDREQ_32B_sum TCC_EA0_WRREQ_64B_sum
# ── L1 vector cache hit rate (1 - TCP->TCC traffic / total accesses) ────
pmc: TCP_TOTAL_CACHE_ACCESSES_sum TCP_TCC_READ_REQ_sum
pmc: TCP_TCC_WRITE_REQ_sum
# ── L2 cache hit rate = TCC_HIT_sum / (TCC_HIT_sum + TCC_MISS_sum) ──────
pmc: TCC_HIT_sum TCC_MISS_sum
# ── GPU activity (GUI busy = GRBM_GUI_ACTIVE / GRBM_COUNT) + CU busy ────
pmc: GRBM_GUI_ACTIVE GRBM_COUNT
pmc: SQ_BUSY_CU_CYCLES
# ── LDS (shared memory) instruction traffic ────────────────────────────
pmc: SQ_INSTS_LDS
EOF
fi
N_PMC_LINES=$(grep -cE '^pmc:' "$PMC_FILE")

echo ""
echo "=== rocprofv3 profile (multi-pass) ==="
echo "  workload : train_multigpu_tiny  batch=$GLOBAL_BATCH  epochs=$EPOCHS"
echo "  out      : $PROFILE_ROOT"
echo "  pmc file : $PMC_FILE  ($N_PMC_LINES pmc: lines → $N_PMC_LINES workload re-runs)"
echo "  iter rng : ${ROCPROF_KIR:-<all>}"
echo ""
echo "----- pmc.txt -----"
cat "$PMC_FILE"
echo "-------------------"
echo ""

KIR_FLAG=()
[ -n "$ROCPROF_KIR" ] && KIR_FLAG=(--kernel-iteration-range "$ROCPROF_KIR")

rocprofv3 -i "$PMC_FILE" \
          "${KIR_FLAG[@]}" \
          --output-format csv \
          -d "$PROFILE_ROOT" \
  -- stdbuf -oL "$BUILD_DIR/train_multigpu_tiny" \
            "$TINY_DATA_DIR" "$GLOBAL_BATCH" "$EPOCHS" "$LR"
STATUS=$?
echo ""
echo "  profile exit: $STATUS"

# ── Verify per-pass output ──────────────────────────────────────────────────
# rocprofv3 writes one *_counter_collection.csv per pass (per pmc: line).
# Compare the number we got vs the number we asked for; that's how we detect
# a single pass silently failing.
mapfile -t CSV_FILES < <(find "$PROFILE_ROOT" -name '*_counter_collection.csv' | sort)
N_CSV=${#CSV_FILES[@]}
echo "  CSVs found  : $N_CSV / $N_PMC_LINES expected"

if [ "$N_CSV" -eq 0 ]; then
    echo "  ✗ No counter CSVs were written. rocprofv3 failed before any pass completed."
    echo "  Full $PROFILE_ROOT tree:"
    find "$PROFILE_ROOT" -maxdepth 3 -type f | sed 's/^/      /'
    exit "${STATUS:-1}"
fi

if [ "$N_CSV" -lt "$N_PMC_LINES" ]; then
    echo "  ! WARNING: $((N_PMC_LINES - N_CSV)) pass(es) did not produce a CSV."
    echo "    Likely cause: counters on that line exceed the HW per-pass limit."
    echo "    Successful CSVs are listed below; missing passes are the pmc: lines"
    echo "    whose counters do NOT appear in any CSV's Counter_Name column."
fi

# ── Per-pass summary ────────────────────────────────────────────────────────
echo ""
echo "=== Per-pass summary ==="
for csv in "${CSV_FILES[@]}"; do
    rows=$(wc -l < "$csv")
    counters=$(awk -F',' 'NR>1{print $16}' "$csv" | sort -u | tr '\n' ',' | sed 's/,$//')
    echo ""
    echo "  $csv"
    echo "    rows     : $rows"
    echo "    counters : $counters"
done

# ── Top kernels (aggregated across passes) ──────────────────────────────────
echo ""
echo "=== Top-10 kernels by dispatch count (across all passes) ==="
# Column 11 is Kernel_Name; demangle/truncate the longest C++ names for readability.
for csv in "${CSV_FILES[@]}"; do
    awk -F',' 'NR>1{print $11}' "$csv"
done | sort | uniq -c | sort -nr | head -10 \
     | awk '{name=$2; for(i=3;i<=NF;i++) name=name" "$i;
             if(length(name)>80) name=substr(name,1,77)"...";
             printf "  %8d  %s\n", $1, name}'

echo ""
echo "=== Done at $(date) ==="
exit $STATUS