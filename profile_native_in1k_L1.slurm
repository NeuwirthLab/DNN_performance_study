#!/bin/bash -l
#SBATCH --job-name=rn50_in1k_L1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --account=l0003099
#SBATCH --partition=gpu:mi300x
#SBATCH --gres=gpu:mi300x:1
#SBATCH --time=00:45:00
#SBATCH --output=logs/rn50_in1k_L1_%j.out
#SBATCH --error=logs/rn50_in1k_L1_%j.err
#SBATCH --export=NONE
#
# =============================================================================
# profile_native_in1k_L1.slurm — capture the ONE native 224² counter that the
# full retry run (job 52488168) couldn't: TCP_TCC_WRITE_REQ_sum (needed for L1
# hit-rate). In 52488168 lines 1-5 paired TWO counters each and all succeeded
# on try 1; line 6 was TCP_TCC_WRITE_REQ ALONE (a 1-counter AQL packet) and
# GPU-faulted on all 6 retries. Hypothesis: the solo-counter packet is the
# trigger. So here we PAIR it with a benign counter (SQ_WAVES) and collect it
# FIRST on a clean GPU. On success we drop it into the 52488168 profile as
# pmc_6 and re-run analyze_roofline so the summary gains L1.
# =============================================================================
set -uo pipefail
echo "==== native in1k L1 (paired TCP write-req) | job ${SLURM_JOB_ID:-local} | $(date) ===="

WORKSPACE_BASE="${WORKSPACE_BASE:-$HPC_SCRATCH}"
PROJECT_DIR="${PROJECT_DIR:-$WORKSPACE_BASE/resent50/performance-analysis-dl-models-resnet50}"
cd "$PROJECT_DIR"
export ROCM_PATH=/opt/rocm; export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$LD_LIBRARY_PATH"
export TMPDIR="${TMPDIR_PROFILE:-$WORKSPACE_BASE/rocprof_tmp}"; mkdir -p "$TMPDIR"
ulimit -c 0 2>/dev/null || true
export HSA_ENABLE_COREDUMP=0 ROCM_DISABLE_CRASH_DUMP=1 AMD_LOG_LEVEL=0
command -v rocprofv3 >/dev/null 2>&1 || { echo "FAIL: no rocprofv3"; exit 1; }

MIOPEN_DB="$WORKSPACE_BASE/miopen_in1k_L1_${SLURM_JOB_ID:-$$}"; rm -rf "$MIOPEN_DB"; mkdir -p "$MIOPEN_DB"
export MIOPEN_USER_DB_PATH="$MIOPEN_DB" MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_DB" MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-2}"
export RN50_ENABLE_BF16=0   # FP32 baseline — unset now defaults to BF16 through-mode

BIN="${BIN:-$PROJECT_DIR/build_profile_in1k/train_multigpu}"; [ -x "$BIN" ] || { echo "FAIL: no binary $BIN"; exit 1; }
N_CLASSES="${N_CLASSES:-4}"; DATA_DIR="$PROJECT_DIR/imagenet_subset_${N_CLASSES}"
[ -d "$DATA_DIR/train" ] || { echo "FAIL: no data $DATA_DIR"; exit 1; }
LOCAL_BATCH="${LOCAL_BATCH:-256}"; LR=0.1; KIR="${ROCPROF_KIR:-1-50}"
MAX_RETRY="${MAX_RETRY:-12}"; PERLINE_TIMEOUT="${PERLINE_TIMEOUT:-150}"
TARGET="${TARGET:-$PROJECT_DIR/profiles/native_in1k_retry_52488168}"   # merge here as pmc_6
[ -d "$TARGET" ] || { echo "FAIL: target profile $TARGET missing"; exit 1; }
echo "  bin=$BIN  data=$DATA_DIR ($N_CLASSES cls, 224²)  target=$TARGET"

echo "  [warmup] populate db (untraced) …"
HIP_VISIBLE_DEVICES=0 "$BIN" "$DATA_DIR" "$LOCAL_BATCH" 1 "$LR" > "$TMPDIR/L1_warmup_${SLURM_JOB_ID:-$$}.log" 2>&1 || echo "  (warmup rc=$?)"

# Pair the fault-prone solo counter with a benign one so the AQL packet is
# well-formed; collected first on a clean GPU.
LINE="TCP_TCC_WRITE_REQ_sum SQ_WAVES"
got=0
for try in $(seq 1 "$MAX_RETRY"); do
  t="$TMPDIR/_L1try_${SLURM_JOB_ID:-$$}_${try}"; rm -rf "$t"; mkdir -p "$t"
  echo "pmc: $LINE" > "$t/pmc.txt"
  HIP_VISIBLE_DEVICES=0 timeout -k 15 "$PERLINE_TIMEOUT" \
      rocprofv3 -i "$t/pmc.txt" --kernel-iteration-range "$KIR" \
      --output-format csv -d "$t/out" \
      -- stdbuf -oL "$BIN" "$DATA_DIR" "$LOCAL_BATCH" 1 "$LR" > "$t/log" 2>&1
  rc=$?
  csv=$(find "$t/out" -name '*_counter_collection.csv' 2>/dev/null | head -1)
  if [ -n "$csv" ] && [ "$(wc -l < "$csv")" -gt 1 ] && grep -q TCP_TCC_WRITE_REQ "$csv"; then
    rm -rf "$TARGET/pmc_6"; mkdir -p "$TARGET/pmc_6"; cp -r "$(dirname "$csv")" "$TARGET/pmc_6/"
    echo "  ✓ TCP_TCC_WRITE_REQ captured (try $try) → $TARGET/pmc_6"; got=1; rm -rf "$t"; break
  fi
  if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then
    echo "  … try $try TIMED OUT (${PERLINE_TIMEOUT}s) — killed, retrying"
  else
    echo "  … try $try faulted (rc=$rc), retrying"
  fi
  pkill -9 -f "$BIN" 2>/dev/null || true; pkill -9 -f rocprofv3 2>/dev/null || true
  sleep 5; rm -rf "$t"
done
[ "$got" = 1 ] || { echo "✗ TCP_TCC_WRITE_REQ still unobtainable after $MAX_RETRY tries (paired)"; exit 2; }

# re-merge the now-complete 52488168 profile → summary gains L1 hit-rate
PY=$(command -v python3.11 || command -v python3)
echo "  re-running analyze_roofline on $TARGET …"
"$PY" "$PROJECT_DIR/analyze_roofline.py" "$TARGET" 2>&1 | tail -30 || true
B="$PROJECT_DIR/imagenet1k_profile_bundle/data/roofline_cache"
[ -f "$TARGET/roofline_summary.csv" ] && cp -f "$TARGET/roofline_summary.csv" \
    "$B/native_imagenet1k_224_roofline_summary.csv" && echo "  ✓ bundle updated"
echo "=== done $(date) | target=$TARGET ==="
