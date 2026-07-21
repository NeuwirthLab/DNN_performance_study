#!/bin/bash -l
#SBATCH --job-name=rn50_profile_native_in1k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --account=l0003099
#SBATCH --partition=gpu:mi300x
#SBATCH --gres=gpu:mi300x:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/rn50_profile_native_in1k_%j.out
#SBATCH --error=logs/rn50_profile_native_in1k_%j.err
#SBATCH --export=NONE
#
# =============================================================================
# profile_native_imagenet_amd.slurm — PMC COUNTER profile of the NATIVE HIP
# trainer on ImageNet-1K 224x224 (the MATCHED counterpart of
# profile_pytorch_amd.slurm).
#
# WHY this exists: every existing native PMC profile (profiles/amd_*) used
# train_multigpu_TINY at 64x64, batch 64 — NOT comparable to the PyTorch PMC
# profile (224x224, batch 256). This profiles the ORIGINAL/production trainer
# `train_multigpu` (root train_multigpu.cu, model.Forward(...,224,224)) on the
# SAME imagenet_subset_200 data, batch 256, 1 GPU that PyTorch used — so the
# roofline + L1/L2 cache numbers line up apples-to-apples.
#
# train_multigpu vs the hyp_in1k native arm's train_multigpu_exp differ by only
# the RN50_SYNTHETIC_DATA knob (a no-op when unset), so the kernel stream here
# is identical to the throughput head-to-head too.
#
# Knobs (env): LOCAL_BATCH=256  N_CLASSES=200  USE_FULL=0  ROCPROF_KIR=1-50
#              EPOCHS=1  ROCPROF_PMC_FILE=<path>
# =============================================================================

set -uo pipefail
echo "========================================================"
echo " NATIVE ResNet-50 ImageNet-1K 224² PMC-counter profiling"
echo " Job   : ${SLURM_JOB_ID:-local}   Node: ${SLURM_NODELIST:-$(hostname)}"
echo " Start : $(date)"
echo "========================================================"

# ── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE_BASE="${WORKSPACE_BASE:-$HPC_SCRATCH}"
PROJECT_DIR="${PROJECT_DIR:-$WORKSPACE_BASE/resent50/performance-analysis-dl-models-resnet50}"
CLS="$PROJECT_DIR/ImageNet-1K/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC"
FULL_DATA="$PROJECT_DIR/imagenet"
cd "$PROJECT_DIR"

module load cmake 2>/dev/null || true
export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$LD_LIBRARY_PATH"
export TMPDIR="${TMPDIR_PROFILE:-$WORKSPACE_BASE/rocprof_tmp}"; mkdir -p "$TMPDIR"

command -v rocprofv3 >/dev/null 2>&1 || { echo "FAIL: rocprofv3 not on PATH"; exit 1; }
echo "  ✓ rocprofv3 : $(command -v rocprofv3)"

GFX_ARCH=$(rocminfo 2>/dev/null | awk '/gfx[0-9a-f]+/{for(i=1;i<=NF;i++) if($i ~ /^gfx/){print $i; exit}}'); [ -z "$GFX_ARCH" ] && GFX_ARCH="gfx942"
echo "  ✓ GPU arch  : $GFX_ARCH"

# ── MIOpen cache in scratch — FRESH per job ──────────────────────────────────
# Counter mode re-runs the binary once per pmc: line (15x). All re-runs share
# one MIOpen db; each appends, accruing "Duplicate ID" entries. By the ~8th
# re-run a corrupt entry produced a malformed AQL packet → GPU memory-access
# fault (HSA_STATUS_ERROR_INVALID_PACKET_FORMAT) at 224²/bs256. A FRESH per-job
# db (never reused across jobs) avoids carrying corruption between runs.
MIOPEN_DB="$WORKSPACE_BASE/miopen_profile_db_in1k_${SLURM_JOB_ID:-$$}"
rm -rf "$MIOPEN_DB"; mkdir -p "$MIOPEN_DB"
export MIOPEN_USER_DB_PATH="$MIOPEN_DB" MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_DB"
# FAST find-mode: heuristic (not exhaustive) → quick warmup + far fewer perf-db
# writes, which is what accrued the "Duplicate ID" corruption that produced the
# malformed-AQL GPU fault. Override with MIOPEN_FIND_MODE=<n> if needed.
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-2}"
export RN50_ENABLE_BF16=0   # FP32 baseline — unset now defaults to BF16 through-mode
echo "  ✓ fresh MIOpen db: $MIOPEN_DB  (FIND_MODE=$MIOPEN_FIND_MODE)"

# ── Build the ORIGINAL native trainer (target: train_multigpu) ───────────────
BUILD_DIR="${BUILD_DIR:-$PROJECT_DIR/build_profile_in1k}"; mkdir -p "$BUILD_DIR"
cmake -S "$PROJECT_DIR/hip" -B "$BUILD_DIR" -DCMAKE_HIP_ARCHITECTURES="$GFX_ARCH" \
      -DCMAKE_BUILD_TYPE=Release -DUSE_NCCL=ON > "$BUILD_DIR/cmake.log" 2>&1 \
    || { echo "FAIL cmake"; tail -40 "$BUILD_DIR/cmake.log"; exit 1; }
cmake --build "$BUILD_DIR" -j"${SLURM_CPUS_PER_TASK:-16}" --target train_multigpu \
      > "$BUILD_DIR/build.log" 2>&1 \
    || { echo "FAIL build"; grep -nE "error:|No such file" "$BUILD_DIR/build.log" | head; tail -40 "$BUILD_DIR/build.log"; exit 1; }
BIN="$BUILD_DIR/train_multigpu"
[ -x "$BIN" ] || { echo "FAIL: $BIN missing"; tail -20 "$BUILD_DIR/build.log"; exit 1; }
echo "  ✓ native binary (original train_multigpu.cu): $BIN"

# ── Data: same 224² subset PyTorch used ──────────────────────────────────────
USE_FULL="${USE_FULL:-0}"; N_CLASSES="${N_CLASSES:-200}"
if [ "$USE_FULL" = "1" ]; then
    DATA_DIR="$FULL_DATA"; [ -d "$DATA_DIR/train" ] || { echo "FAIL: $DATA_DIR/train missing"; exit 1; }
else
    DATA_DIR="$PROJECT_DIR/imagenet_subset_${N_CLASSES}"
    if [ ! -d "$DATA_DIR/train" ] || [ "$(ls "$DATA_DIR/train" 2>/dev/null | wc -l)" -ne "$N_CLASSES" ]; then
        echo "  building subset: $N_CLASSES classes -> $DATA_DIR"
        [ -d "$CLS/train" ] || { echo "FAIL: $CLS/train missing"; exit 1; }
        rm -rf "$DATA_DIR"; mkdir -p "$DATA_DIR/train" "$DATA_DIR/val"
        for c in $(ls "$CLS/train" | head -"$N_CLASSES"); do
            ln -sfn "$CLS/train/$c" "$DATA_DIR/train/$c"
            [ -d "$CLS/val/$c" ] && ln -sfn "$CLS/val/$c" "$DATA_DIR/val/$c"
        done
    fi
fi
echo "  ✓ data (224²): $DATA_DIR"

# ── Knobs ────────────────────────────────────────────────────────────────────
LOCAL_BATCH="${LOCAL_BATCH:-256}"; EPOCHS="${EPOCHS:-1}"; LR="${LR:-0.1}"
ROCPROF_KIR="${ROCPROF_KIR:-1-50}"
PROFILE_ROOT="$PROJECT_DIR/profiles/native_in1k_${SLURM_JOB_ID:-$$}"; mkdir -p "$PROFILE_ROOT"

# ── PMC counter set — IDENTICAL to profile_rn50_amd.slurm / profile_pytorch ──
PMC_FILE="${ROCPROF_PMC_FILE:-$PROFILE_ROOT/pmc.txt}"
if [ ! -f "$PMC_FILE" ] || [ "$PMC_FILE" = "$PROFILE_ROOT/pmc.txt" ]; then
    cat > "$PMC_FILE" <<'EOF'
pmc: SQ_WAVES SQ_INSTS_VALU_FMA_F32
pmc: SQ_INSTS_VALU_ADD_F32 SQ_INSTS_VALU_MUL_F32
pmc: SQ_INSTS_VALU_TRANS_F32
pmc: SQ_INSTS_VALU_FMA_F16 SQ_INSTS_VALU_ADD_F16
pmc: SQ_INSTS_VALU_MUL_F16 SQ_INSTS_VALU_TRANS_F16
pmc: SQ_INSTS_VALU_MFMA_MOPS_F32 SQ_INSTS_VALU_MFMA_MOPS_F16
pmc: SQ_INSTS_VALU_MFMA_MOPS_BF16 SQ_INSTS_VALU_MFMA_MOPS_F64
pmc: TCC_EA0_RDREQ_sum TCC_EA0_WRREQ_sum
pmc: TCC_EA0_RDREQ_32B_sum TCC_EA0_WRREQ_64B_sum
pmc: TCP_TOTAL_CACHE_ACCESSES_sum TCP_TCC_READ_REQ_sum
pmc: TCP_TCC_WRITE_REQ_sum
pmc: TCC_HIT_sum TCC_MISS_sum
pmc: GRBM_GUI_ACTIVE GRBM_COUNT
pmc: SQ_BUSY_CU_CYCLES
pmc: SQ_INSTS_LDS
EOF
fi
N_PMC_LINES=$(grep -cE '^pmc:' "$PMC_FILE")
GLOBAL_BATCH=$LOCAL_BATCH   # 1 GPU → global == local, matches PyTorch per-GPU 256

echo ""
echo "=== rocprofv3 counter profile (multi-pass) ==="
echo "  workload : train_multigpu  $DATA_DIR  global_batch=$GLOBAL_BATCH  epochs=$EPOCHS  1 GPU"
echo "  pmc lines: $N_PMC_LINES  → $N_PMC_LINES workload re-runs"
echo "  iter rng : ${ROCPROF_KIR:-<all>}   out: $PROFILE_ROOT"

# Warm MIOpen cache once (untraced) so the N counter re-runs reuse tuned convs.
echo "  [warmup] populating MIOpen cache…"
HIP_VISIBLE_DEVICES=0 "$BIN" "$DATA_DIR" "$GLOBAL_BATCH" 1 "$LR" > "$PROFILE_ROOT/warmup.log" 2>&1 \
    || echo "  (warmup rc=$? — continuing; check warmup.log)"

KIR_FLAG=(); [ -n "$ROCPROF_KIR" ] && KIR_FLAG=(--kernel-iteration-range "$ROCPROF_KIR")
HIP_VISIBLE_DEVICES=0 rocprofv3 -i "$PMC_FILE" "${KIR_FLAG[@]}" \
    --output-format csv -d "$PROFILE_ROOT" \
    -- stdbuf -oL "$BIN" "$DATA_DIR" "$GLOBAL_BATCH" "$EPOCHS" "$LR" \
    > "$PROFILE_ROOT/profile.log" 2>&1
STATUS=$?
echo "  profile exit: $STATUS"

mapfile -t CSV_FILES < <(find "$PROFILE_ROOT" -name '*_counter_collection.csv' | sort)
N_CSV=${#CSV_FILES[@]}
echo "  CSVs found  : $N_CSV / $N_PMC_LINES expected"
if [ "$N_CSV" -eq 0 ]; then
    echo "  ✗ No counter CSVs. Tail of profile.log:"; tail -40 "$PROFILE_ROOT/profile.log"; exit "${STATUS:-1}"
fi
[ "$N_CSV" -lt "$N_PMC_LINES" ] && echo "  ! WARNING: $((N_PMC_LINES-N_CSV)) pass(es) produced no CSV."

# ── Post-process: roofline + cache, with the shared friendly labels ──────────
echo ""; echo "=== Post-processing (roofline + cache table) ==="
PY=$(command -v python3.11 || command -v python3)
"$PY" "$PROJECT_DIR/analyze_roofline.py" "$PROFILE_ROOT" 2>&1 | tail -40 || true
"$PY" "$PROJECT_DIR/plot_roofline.py" --root "$PROFILE_ROOT" \
    --out "$PROJECT_DIR/figures/roofline_native_imagenet1k_${SLURM_JOB_ID:-local}.pdf" \
    --csv-out "$PROFILE_ROOT/by_kernel_native_imagenet1k.csv" 2>&1 | tail -25 || true

# ── Drop the matched native data straight into the bundle ────────────────────
BUNDLE="$PROJECT_DIR/imagenet1k_profile_bundle/data/roofline_cache"
if [ -d "$BUNDLE" ]; then
    cp -f "$PROFILE_ROOT/roofline_summary.csv" "$BUNDLE/native_imagenet1k_224_roofline_summary.csv" 2>/dev/null
    cp -f "$PROFILE_ROOT/by_kernel_native_imagenet1k.csv" "$BUNDLE/native_imagenet1k_224_by_kernel.csv" 2>/dev/null
    echo "  ✓ copied matched native 224² roofline/cache into the bundle"
fi

echo ""
echo "  roofline : figures/roofline_native_imagenet1k_${SLURM_JOB_ID:-local}.pdf"
echo "  cache tbl: $PROFILE_ROOT/roofline_summary.csv  (l1_hit_pct / l2_hit_pct)"
echo "  NEXT: re-run imagenet1k_profile_bundle/scripts/plot_bundle_all.py for the matched"
echo "        native-vs-PyTorch roofline + cache panels."
echo "=== Done at $(date) ==="
exit "$STATUS"
