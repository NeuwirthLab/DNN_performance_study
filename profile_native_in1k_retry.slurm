#!/bin/bash -l
#SBATCH --job-name=rn50_native_in1k_retry
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --account=l0003099
#SBATCH --partition=gpu:mi300x
#SBATCH --gres=gpu:mi300x:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/rn50_native_in1k_retry_%j.out
#SBATCH --error=logs/rn50_native_in1k_retry_%j.err
#SBATCH --export=NONE
#
# =============================================================================
# profile_native_in1k_retry.slurm — robust native 224² PMC profile via
# PER-COUNTER-LINE rocprofv3 calls with retry.
#
# WHY: train_multigpu intermittently GPU-faults under rocprofv3 while collecting
# the TCC (L2/HBM) counters at 224² (malformed AQL packet). When all 15 pmc
# lines run in ONE rocprofv3 process, the first such fault aborts the whole run.
# Here each counter line is a SEPARATE rocprofv3 invocation (1 binary re-run) so
# a fault kills only that attempt; we RETRY each line until it yields a CSV.
# All lines use one fixed config (bs256, 224², 4-class subset) so the per-kernel
# counters merge cleanly into one roofline_summary. FLOP (SQ) lines basically
# never fault; the TCC/TCP lines may take a couple of attempts.
#
# Knobs: LOCAL_BATCH=256  N_CLASSES=4  MAX_RETRY=5  ROCPROF_KIR=1-50
# =============================================================================
set -uo pipefail
echo "==== native ImageNet-1K 224² PMC (per-line retry) | job ${SLURM_JOB_ID:-local} | $(date) ===="

WORKSPACE_BASE="${WORKSPACE_BASE:-$HPC_SCRATCH}"
PROJECT_DIR="${PROJECT_DIR:-$WORKSPACE_BASE/resent50/performance-analysis-dl-models-resnet50}"
CLS="$PROJECT_DIR/ImageNet-1K/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC"
cd "$PROJECT_DIR"
module load cmake 2>/dev/null || true
export ROCM_PATH=/opt/rocm; export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$LD_LIBRARY_PATH"
export TMPDIR="${TMPDIR_PROFILE:-$WORKSPACE_BASE/rocprof_tmp}"; mkdir -p "$TMPDIR"
command -v rocprofv3 >/dev/null 2>&1 || { echo "FAIL: no rocprofv3"; exit 1; }
# The TCC counter passes intermittently GPU-fault; the runtime then tries to
# write a multi-GB GPU core dump and HANGS, blocking the whole run. Disable core
# dumps (best-effort env + ulimit) so a fault aborts fast; the per-attempt
# `timeout` below is the real guarantee against hangs.
ulimit -c 0 2>/dev/null || true
export HSA_ENABLE_COREDUMP=0 ROCM_DISABLE_CRASH_DUMP=1 AMD_LOG_LEVEL=0

GFX_ARCH=$(rocminfo 2>/dev/null | awk '/gfx[0-9a-f]+/{for(i=1;i<=NF;i++) if($i ~ /^gfx/){print $i; exit}}'); [ -z "$GFX_ARCH" ] && GFX_ARCH=gfx942

# fresh clean MIOpen db + FAST find-mode (few writes → avoids the corruption that
# helped trigger the AQL fault; per-line isolation handles the rest).
MIOPEN_DB="$WORKSPACE_BASE/miopen_native_in1k_retry_${SLURM_JOB_ID:-$$}"; rm -rf "$MIOPEN_DB"; mkdir -p "$MIOPEN_DB"
export MIOPEN_USER_DB_PATH="$MIOPEN_DB" MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_DB" MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-2}"
export RN50_ENABLE_BF16=0   # FP32 baseline — unset now defaults to BF16 through-mode

# build (incremental; cached from prior runs)
BUILD_DIR="${BUILD_DIR:-$PROJECT_DIR/build_profile_in1k}"; mkdir -p "$BUILD_DIR"
cmake -S "$PROJECT_DIR/hip" -B "$BUILD_DIR" -DCMAKE_HIP_ARCHITECTURES="$GFX_ARCH" -DCMAKE_BUILD_TYPE=Release -DUSE_NCCL=ON > "$BUILD_DIR/cmake.log" 2>&1 || { echo FAIL cmake; tail -20 "$BUILD_DIR/cmake.log"; exit 1; }
cmake --build "$BUILD_DIR" -j"${SLURM_CPUS_PER_TASK:-16}" --target train_multigpu > "$BUILD_DIR/build.log" 2>&1 || { echo FAIL build; tail -20 "$BUILD_DIR/build.log"; exit 1; }
BIN="$BUILD_DIR/train_multigpu"; [ -x "$BIN" ] || { echo "FAIL: no binary"; exit 1; }

# data: 4-class 224² subset (conv shapes identical to full ImageNet-1K)
N_CLASSES="${N_CLASSES:-4}"; DATA_DIR="$PROJECT_DIR/imagenet_subset_${N_CLASSES}"
if [ ! -d "$DATA_DIR/train" ] || [ "$(ls "$DATA_DIR/train" 2>/dev/null | wc -l)" -ne "$N_CLASSES" ]; then
    rm -rf "$DATA_DIR"; mkdir -p "$DATA_DIR/train" "$DATA_DIR/val"
    for c in $(ls "$CLS/train" | head -"$N_CLASSES"); do ln -sfn "$CLS/train/$c" "$DATA_DIR/train/$c"; [ -d "$CLS/val/$c" ] && ln -sfn "$CLS/val/$c" "$DATA_DIR/val/$c"; done
fi
echo "  ✓ binary=$BIN  data=$DATA_DIR ($N_CLASSES cls, 224²)  db=$MIOPEN_DB (FIND_MODE=$MIOPEN_FIND_MODE)"

LOCAL_BATCH="${LOCAL_BATCH:-256}"; LR="${LR:-0.1}"; KIR="${ROCPROF_KIR:-1-50}"; MAX_RETRY="${MAX_RETRY:-6}"
# Hard cap per attempt: working passes finish in <60s, so a run still alive at
# PERLINE_TIMEOUT is a fault-hang → kill it and retry (the missing piece that
# let the old run block forever on a single faulted pass).
PERLINE_TIMEOUT="${PERLINE_TIMEOUT:-150}"
ROOT="$PROJECT_DIR/profiles/native_in1k_retry_${SLURM_JOB_ID:-$$}"; mkdir -p "$ROOT"

# warmup (untraced — no counters, cannot hit the TCC fault) to populate db.
echo "  [warmup] …"; HIP_VISIBLE_DEVICES=0 "$BIN" "$DATA_DIR" "$LOCAL_BATCH" 1 "$LR" > "$ROOT/warmup.log" 2>&1 || echo "  (warmup rc=$?)"

# Essential counters for roofline (FLOPs+bytes) + cache (L1/L2), one line each.
LINES=(
  "SQ_INSTS_VALU_FMA_F32 SQ_INSTS_VALU_MFMA_MOPS_F32"   # FLOPs: VALU FMA + matrix
  "SQ_INSTS_VALU_ADD_F32 SQ_INSTS_VALU_MUL_F32"         # FLOPs: add/mul
  "TCC_EA0_RDREQ_sum TCC_EA0_WRREQ_sum"                 # HBM bytes  (fault-prone)
  "TCC_HIT_sum TCC_MISS_sum"                            # L2 cache   (fault-prone)
  "TCP_TOTAL_CACHE_ACCESSES_sum TCP_TCC_READ_REQ_sum"   # L1 cache   (fault-prone)
  "TCP_TCC_WRITE_REQ_sum"                               # L1 cache
)
idx=0; OK=0
for line in "${LINES[@]}"; do
  idx=$((idx+1)); got=0
  for try in $(seq 1 "$MAX_RETRY"); do
    t="$ROOT/_try_${idx}_${try}"; rm -rf "$t"; mkdir -p "$t"
    echo "pmc: $line" > "$t/pmc.txt"
    HIP_VISIBLE_DEVICES=0 timeout -k 15 "$PERLINE_TIMEOUT" \
        rocprofv3 -i "$t/pmc.txt" --kernel-iteration-range "$KIR" \
        --output-format csv -d "$t/out" \
        -- stdbuf -oL "$BIN" "$DATA_DIR" "$LOCAL_BATCH" 1 "$LR" > "$t/log" 2>&1
    rc=$?
    csv=$(find "$t/out" -name '*_counter_collection.csv' 2>/dev/null | head -1)
    if [ -n "$csv" ] && [ "$(wc -l < "$csv")" -gt 1 ]; then
      mkdir -p "$ROOT/pmc_$idx"; cp -r "$(dirname "$csv")" "$ROOT/pmc_$idx/"
      echo "  ✓ line $idx ok (try $try): $line"; got=1; rm -rf "$t"; break
    fi
    if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then
      echo "  … line $idx try $try TIMED OUT (${PERLINE_TIMEOUT}s GPU-fault hang) — killed, retrying: $line"
    else
      echo "  … line $idx try $try faulted (rc=$rc), retrying: $line"
    fi
    # a timeout-killed rocprofv3 can orphan the binary still holding the GPU;
    # reap any strays before the next attempt so it gets a clean device.
    pkill -9 -f "$BIN" 2>/dev/null || true; pkill -9 -f rocprofv3 2>/dev/null || true
    sleep 5; rm -rf "$t"
  done
  [ "$got" = 1 ] && OK=$((OK+1)) || echo "  ✗ line $idx FAILED after $MAX_RETRY tries: $line"
done
echo "=== collected $OK / ${#LINES[@]} counter lines ==="

# ── merge + roofline + cache (analyze_roofline globs pmc_*/<host>/*.csv) ──────
PY=$(command -v python3.11 || command -v python3)
"$PY" "$PROJECT_DIR/analyze_roofline.py" "$ROOT" 2>&1 | tail -40 || true
"$PY" "$PROJECT_DIR/plot_roofline.py" --root "$ROOT" \
    --out "$PROJECT_DIR/figures/roofline_native_imagenet1k_${SLURM_JOB_ID:-local}.pdf" \
    --csv-out "$ROOT/by_kernel_native_imagenet1k.csv" 2>&1 | tail -20 || true

# drop matched native data into the bundle
B="$PROJECT_DIR/imagenet1k_profile_bundle/data/roofline_cache"
[ -f "$ROOT/roofline_summary.csv" ] && cp -f "$ROOT/roofline_summary.csv" "$B/native_imagenet1k_224_roofline_summary.csv" && echo "  ✓ bundle: native_imagenet1k_224_roofline_summary.csv"
[ -f "$ROOT/by_kernel_native_imagenet1k.csv" ] && cp -f "$ROOT/by_kernel_native_imagenet1k.csv" "$B/native_imagenet1k_224_by_kernel.csv"
echo "=== done $(date) | out=$ROOT ==="