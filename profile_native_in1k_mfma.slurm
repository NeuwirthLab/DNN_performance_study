#!/bin/bash -l
#SBATCH --job-name=rn50_in1k_mfma
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --account=l0003099
#SBATCH --partition=gpu:mi300x
#SBATCH --gres=gpu:mi300x:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/rn50_in1k_mfma_%j.out
#SBATCH --error=logs/rn50_in1k_mfma_%j.err
#SBATCH --export=NONE
#
# =============================================================================
# profile_native_in1k_mfma.slurm — complete the native 224² profile
# (52488168) with the THREE counter lines its reduced set skipped, so the
# roofline summary gains the two metrics still NaN:
#   MFMA TFLOPs   needs all four MOPS counters; 52488168 line 1 captured F32
#                 only → collect F16/BF16/F64 (analyze_roofline.py:292-294)
#   GPU active %  needs GRBM_GUI_ACTIVE + GRBM_COUNT (analyze_roofline.py:291)
# Same proven harness as profile_native_in1k_L1.slurm (per-line rocprofv3 with
# timeout+retry, fault-prone solo counters paired with a benign one).
#
# MERGE-VALIDITY GUARD: resnet50.cu gained ~550 lines of BF16 work since the
# Jun 6 binary, and build_profile_in1k/train_multigpu was relinked today by the
# bf16 profile job. BF16 is runtime-gated (RN50_ENABLE_BF16 unset/0 = FP32
# path) so the default kernel set SHOULD be unchanged — but we verify: the new
# lines only merge into 52488168 as pmc_7..9 if the dispatched kernel-name set
# matches the Jun 6 pmc_1 set. On mismatch we fall back to re-collecting all
# nine lines into a fresh, self-consistent profile dir instead (and say so
# loudly — the bundle then switches to the new profile wholesale).
# =============================================================================
set -uo pipefail
echo "==== native in1k 224² MFMA+GRBM completion | job ${SLURM_JOB_ID:-local} | $(date) ===="

WORKSPACE_BASE="${WORKSPACE_BASE:-$HPC_SCRATCH}"
PROJECT_DIR="${PROJECT_DIR:-$WORKSPACE_BASE/resent50/performance-analysis-dl-models-resnet50}"
cd "$PROJECT_DIR"
export ROCM_PATH=/opt/rocm; export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$LD_LIBRARY_PATH"
export TMPDIR="${TMPDIR_PROFILE:-$WORKSPACE_BASE/rocprof_tmp}"; mkdir -p "$TMPDIR"
ulimit -c 0 2>/dev/null || true
export HSA_ENABLE_COREDUMP=0 ROCM_DISABLE_CRASH_DUMP=1 AMD_LOG_LEVEL=0
command -v rocprofv3 >/dev/null 2>&1 || { echo "FAIL: no rocprofv3"; exit 1; }

# force the FP32 default path — the Jun 6 profile predates the BF16 work
export RN50_ENABLE_BF16=0; unset RN50_ENABLE_BF16_STAGE1 2>/dev/null || true

MIOPEN_DB="$WORKSPACE_BASE/miopen_in1k_mfma_${SLURM_JOB_ID:-$$}"; rm -rf "$MIOPEN_DB"; mkdir -p "$MIOPEN_DB"
export MIOPEN_USER_DB_PATH="$MIOPEN_DB" MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_DB" MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-2}"

BIN="${BIN:-$PROJECT_DIR/build_profile_in1k/train_multigpu}"; [ -x "$BIN" ] || { echo "FAIL: no binary $BIN"; exit 1; }
N_CLASSES="${N_CLASSES:-4}"; DATA_DIR="$PROJECT_DIR/imagenet_subset_${N_CLASSES}"
[ -d "$DATA_DIR/train" ] || { echo "FAIL: no data $DATA_DIR"; exit 1; }
LOCAL_BATCH="${LOCAL_BATCH:-256}"; LR=0.1; KIR="${ROCPROF_KIR:-1-50}"
MAX_RETRY="${MAX_RETRY:-8}"; PERLINE_TIMEOUT="${PERLINE_TIMEOUT:-150}"
TARGET="${TARGET:-$PROJECT_DIR/profiles/native_in1k_retry_52488168}"
[ -d "$TARGET/pmc_1" ] || { echo "FAIL: target profile $TARGET missing pmc_1"; exit 1; }
ROOT="$PROJECT_DIR/profiles/native_in1k_mfma_${SLURM_JOB_ID:-$$}"; mkdir -p "$ROOT"
echo "  bin=$BIN  data=$DATA_DIR ($N_CLASSES cls, 224²)  target=$TARGET  staging=$ROOT"

echo "  [warmup] populate MIOpen db (untraced) …"
HIP_VISIBLE_DEVICES=0 "$BIN" "$DATA_DIR" "$LOCAL_BATCH" 1 "$LR" > "$ROOT/warmup.log" 2>&1 || echo "  (warmup rc=$?)"

# one collection attempt loop, shared by every line
collect_line () {  # $1=pmc line  $2=dest dir (created on success)  → rc 0/1
  local line="$1" dest="$2" try t rc csv
  for try in $(seq 1 "$MAX_RETRY"); do
    t="$TMPDIR/_mfma_${SLURM_JOB_ID:-$$}_$(echo "$line" | tr ' ' '_' | cut -c1-40)_${try}"
    rm -rf "$t"; mkdir -p "$t"; echo "pmc: $line" > "$t/pmc.txt"
    HIP_VISIBLE_DEVICES=0 timeout -k 15 "$PERLINE_TIMEOUT" \
        rocprofv3 -i "$t/pmc.txt" --kernel-iteration-range "$KIR" \
        --output-format csv -d "$t/out" \
        -- stdbuf -oL "$BIN" "$DATA_DIR" "$LOCAL_BATCH" 1 "$LR" > "$t/log" 2>&1
    rc=$?
    csv=$(find "$t/out" -name '*_counter_collection.csv' 2>/dev/null | head -1)
    if [ -n "$csv" ] && [ "$(wc -l < "$csv")" -gt 1 ]; then
      rm -rf "$dest"; mkdir -p "$dest"; cp -r "$(dirname "$csv")" "$dest/"
      echo "  ✓ ok (try $try): $line"; rm -rf "$t"; return 0
    fi
    if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then
      echo "  … try $try TIMED OUT (${PERLINE_TIMEOUT}s fault-hang) — killed, retrying: $line"
    else
      echo "  … try $try faulted (rc=$rc), retrying: $line"
    fi
    pkill -9 -f "$BIN" 2>/dev/null || true; pkill -9 -f rocprofv3 2>/dev/null || true
    sleep 5; rm -rf "$t"
  done
  echo "  ✗ FAILED after $MAX_RETRY tries: $line"; return 1
}

# the three missing lines (SQ/GRBM lines have never been seen to fault; the
# lone F64 counter is paired with benign SQ_WAVES like the L1 fix)
NEW_LINES=(
  "SQ_INSTS_VALU_MFMA_MOPS_F16 SQ_INSTS_VALU_MFMA_MOPS_BF16"
  "SQ_INSTS_VALU_MFMA_MOPS_F64 SQ_WAVES"
  "GRBM_GUI_ACTIVE GRBM_COUNT"
)
idx=6; FAILED=0
for line in "${NEW_LINES[@]}"; do
  idx=$((idx+1))
  collect_line "$line" "$ROOT/pmc_$idx" || FAILED=1
done
[ "$FAILED" = 0 ] || { echo "FAIL: could not collect all new lines"; exit 2; }

# ── merge-validity check: kernel-name set of new GRBM pass vs Jun 6 pmc_1 ────
PY=$(command -v python3.11 || command -v python3)
MERGE_OK=$("$PY" - "$TARGET/pmc_1" "$ROOT/pmc_9" <<'EOF'
import csv, glob, sys
def kset(d):
    f = glob.glob(f"{d}/*/*_counter_collection.csv")[0]
    with open(f) as fh:
        return {r["Kernel_Name"] for r in csv.DictReader(fh)}
old, new = kset(sys.argv[1]), kset(sys.argv[2])
only_old, only_new = old - new, new - old
inter = len(old & new); union = len(old | new)
print(f"kernels: old={len(old)} new={len(new)} jaccard={inter/union:.3f}", file=sys.stderr)
for k in sorted(only_old)[:5]: print(f"  only-Jun6: {k[:90]}", file=sys.stderr)
for k in sorted(only_new)[:5]: print(f"  only-new : {k[:90]}", file=sys.stderr)
print("yes" if old == new else "no")
EOF
)
echo "  merge-validity: kernel sets identical = $MERGE_OK"

if [ "$MERGE_OK" = "yes" ]; then
  for i in 7 8 9; do rm -rf "$TARGET/pmc_$i"; cp -r "$ROOT/pmc_$i" "$TARGET/pmc_$i"; done
  PROFILE="$TARGET"
  echo "  ✓ merged pmc_7..9 into $TARGET"
else
  echo "  !! KERNEL SET CHANGED since Jun 6 (BF16 work touched the default path)."
  echo "  !! Falling back: re-collecting the six original lines into $ROOT for a"
  echo "  !! self-consistent fresh profile (bundle will switch to it wholesale)."
  OLD_LINES=(
    "SQ_INSTS_VALU_FMA_F32 SQ_INSTS_VALU_MFMA_MOPS_F32"
    "SQ_INSTS_VALU_ADD_F32 SQ_INSTS_VALU_MUL_F32"
    "TCC_EA0_RDREQ_sum TCC_EA0_WRREQ_sum"
    "TCC_HIT_sum TCC_MISS_sum"
    "TCP_TOTAL_CACHE_ACCESSES_sum TCP_TCC_READ_REQ_sum"
    "TCP_TCC_WRITE_REQ_sum SQ_WAVES"
  )
  idx=0; FAILED=0
  for line in "${OLD_LINES[@]}"; do
    idx=$((idx+1))
    collect_line "$line" "$ROOT/pmc_$idx" || FAILED=1
  done
  [ "$FAILED" = 0 ] || { echo "FAIL: fallback re-collection incomplete"; exit 3; }
  PROFILE="$ROOT"
fi

# ── re-analyze; hard-fail if the two target metrics are still NaN ────────────
echo "  analyze_roofline on $PROFILE …"
"$PY" "$PROJECT_DIR/analyze_roofline.py" "$PROFILE" 2>&1 | tail -25 || true
SUMMARY="$PROFILE/roofline_summary.csv"
[ -f "$SUMMARY" ] || { echo "FAIL: no roofline_summary.csv"; exit 4; }
"$PY" - "$SUMMARY" <<'EOF' || exit 5
import csv, sys, math
rows = list(csv.DictReader(open(sys.argv[1])))
bad = {c: sum(1 for r in rows if r.get(c,"").lower() in ("","nan","none"))
       for c in ("mfma_tflops", "gpu_active_pct")}
print(f"  validation: {len(rows)} kernels; NaN counts {bad}")
top = sorted(rows, key=lambda r: -float(r["total_ms"]))[:8]
print(f"  {'kernel':32s} {'tot_ms':>8s} {'mfma_TF':>8s} {'act%':>6s}")
for r in top:
    print(f"  {r['label'][:32]:32s} {float(r['total_ms']):8.1f} "
          f"{float(r['mfma_tflops']):8.2f} {float(r['gpu_active_pct']):6.1f}")
sys.exit(1 if any(v == len(rows) for v in bad.values()) else 0)
EOF

# ── refresh bundle + regenerate the matched-224² compare figures ─────────────
B="$PROJECT_DIR/imagenet1k_profile_bundle/data/roofline_cache"
cp -f "$SUMMARY" "$B/native_imagenet1k_224_roofline_summary.csv" && echo "  ✓ bundle updated (source: $PROFILE)"
"$PY" "$PROJECT_DIR/tools/compare_roofline_cache.py" \
    --native "$SUMMARY" \
    --pytorch "$PROJECT_DIR/profiles/pt_amd_52438059/roofline_summary.csv" \
    --device MI300X \
    --out-dir "$PROJECT_DIR/figures/compare_in1k_224" 2>&1 | tail -10 || \
    echo "  (compare figures failed — run tools/compare_roofline_cache.py manually)"
echo "=== done $(date) | profile=$PROFILE ==="
