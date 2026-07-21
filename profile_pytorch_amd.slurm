#!/bin/bash -l
#SBATCH --job-name=rn50_profile_pytorch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --account=l0003099
#SBATCH --partition=gpu:mi300x
#SBATCH --gres=gpu:mi300x:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/rn50_profile_pytorch_%j.out
#SBATCH --error=logs/rn50_profile_pytorch_%j.err
#SBATCH --export=NONE
#
# =============================================================================
# profile_pytorch_amd.slurm — PMC COUNTER profile of PyTorch ResNet-50 on MI300X
#
# Companion to profile_rn50_amd.slurm (which profiles the NATIVE HIP trainer).
# This runs rocprofv3 in COUNTER mode (-i pmc.txt, multi-pass) over the PyTorch
# workload INSIDE the rocm/pytorch apptainer container, producing the same
# *_counter_collection.csv layout the native profiler emits — so
# analyze_roofline.py and plot_roofline.py work on it UNCHANGED, and the
# resulting roofline / L1-L2 cache-hit (thrashing) numbers line up 1:1 with the
# already-profiled native tests.
#
# WHY a separate run: the existing PyTorch profiles under profiles/hyp_in1k_*
# are --kernel-trace only (timing). They have NO FLOP/byte/TCC/TCP counters, so
# arithmetic intensity and cache-hit rates cannot be computed from them.
#
# Mirrors the proven single-process / workers=0 PyTorch trace invocation from
# hyp_imagenet_exp_amd.slurm (rocprofv3 collects in-process; the DataLoader's
# fork()ed workers inherit its intercept lib and segfault the HSA layer, so we
# trace 1 GPU, 0 workers — the GPU kernel stream is identical).
#
# Knobs (env):
#   CONTAINER_IMAGE=./pytorch_rocm.sif   LOCAL_BATCH=256   PROFILE_STEPS=30
#   ROCPROF_KIR=1-50   N_CLASSES=200   USE_FULL=0   ROCPROF_PMC_FILE=<path>
# =============================================================================

set -uo pipefail
echo "========================================================"
echo " PyTorch ResNet-50 MI300X PMC-counter profiling"
echo " Job   : ${SLURM_JOB_ID:-local}"
echo " Node  : ${SLURM_NODELIST:-$(hostname)}"
echo " Start : $(date)"
echo "========================================================"

# ── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE_BASE="${WORKSPACE_BASE:-$HPC_SCRATCH}"
PROJECT_DIR="${PROJECT_DIR:-$WORKSPACE_BASE/resent50/performance-analysis-dl-models-resnet50}"
CLS="$PROJECT_DIR/ImageNet-1K/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC"
FULL_DATA="$PROJECT_DIR/imagenet"
cd "$PROJECT_DIR"

export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$LD_LIBRARY_PATH"
export TMPDIR="${TMPDIR_PROFILE:-$WORKSPACE_BASE/rocprof_tmp}"; mkdir -p "$TMPDIR"
export APPTAINER_TMPDIR="$TMPDIR"

# ── Container ────────────────────────────────────────────────────────────────
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$PROJECT_DIR/pytorch_rocm.sif}"
command -v apptainer >/dev/null 2>&1 || { echo "FAIL: apptainer not on PATH"; exit 1; }
# Accept BOTH a .sif file and an extracted sandbox DIRECTORY. Running from a
# sandbox dir avoids squashfuse_ll, which stalled on the /work parallel FS under
# the 15-pass counter re-reads and SIGKILLed jobs 53044744/45 (see git memory).
[ -e "$CONTAINER_IMAGE" ] || { echo "FAIL: container image not found: $CONTAINER_IMAGE"; exit 1; }
if [ -d "$CONTAINER_IMAGE" ]; then echo "  ✓ container : $CONTAINER_IMAGE (sandbox dir, no squashfuse)";
else echo "  ✓ container : $CONTAINER_IMAGE (.sif via squashfuse)"; fi

# ── Data: full or fast subset (224² conv shapes identical either way) ────────
USE_FULL="${USE_FULL:-0}"; N_CLASSES="${N_CLASSES:-200}"
if [ "$USE_FULL" = "1" ]; then
    DATA_DIR="$FULL_DATA"
    [ -d "$DATA_DIR/train" ] || { echo "FAIL: $DATA_DIR/train missing"; exit 1; }
    echo "  ✓ data: FULL ImageNet-1K ($DATA_DIR)"
else
    DATA_DIR="$PROJECT_DIR/imagenet_subset_${N_CLASSES}"
    if [ ! -d "$DATA_DIR/train" ] || [ "$(ls "$DATA_DIR/train" 2>/dev/null | wc -l)" -ne "$N_CLASSES" ]; then
        echo "  building subset: $N_CLASSES classes -> $DATA_DIR"
        [ -d "$CLS/train" ] || { echo "FAIL: $CLS/train missing (ImageNet-1K not staged)"; exit 1; }
        rm -rf "$DATA_DIR"; mkdir -p "$DATA_DIR/train" "$DATA_DIR/val"
        for c in $(ls "$CLS/train" | head -"$N_CLASSES"); do
            ln -sfn "$CLS/train/$c" "$DATA_DIR/train/$c"
            [ -d "$CLS/val/$c" ] && ln -sfn "$CLS/val/$c" "$DATA_DIR/val/$c"
        done
    fi
    echo "  ✓ data: SUBSET $N_CLASSES classes ($DATA_DIR)"
fi

# ── Knobs ────────────────────────────────────────────────────────────────────
LOCAL_BATCH="${LOCAL_BATCH:-256}"
PROFILE_STEPS="${PROFILE_STEPS:-30}"     # profiled steps after warmup (per pass)
ROCPROF_KIR="${ROCPROF_KIR:-1-50}"       # counters on first N iters of each kernel
PROFILE_ROOT="$PROJECT_DIR/profiles/pt_amd_${SLURM_JOB_ID:-$$}"; mkdir -p "$PROFILE_ROOT"
PT_DB="$PROFILE_ROOT/db"; mkdir -p "$PT_DB"
HOST_TMP="$PROJECT_DIR/.container_tmp"; mkdir -p "$HOST_TMP"

# ── PMC counter set — IDENTICAL to profile_rn50_amd.slurm ────────────────────
# Same names → analyze_roofline.py / plot_roofline.py treat native & PyTorch
# profiles identically. All verified on gfx942; an invalid name silently drops
# its whole pmc: line, so do not edit without checking `rocprofv3 --list-avail`.
PMC_FILE="${ROCPROF_PMC_FILE:-$PROFILE_ROOT/pmc.txt}"
if [ ! -f "$PMC_FILE" ] || [ "$PMC_FILE" = "$PROFILE_ROOT/pmc.txt" ]; then
    cat > "$PMC_FILE" <<'EOF'
# ── VALU compute throughput ────────────────────────────────────────────
pmc: SQ_WAVES SQ_INSTS_VALU_FMA_F32
pmc: SQ_INSTS_VALU_ADD_F32 SQ_INSTS_VALU_MUL_F32
pmc: SQ_INSTS_VALU_TRANS_F32
pmc: SQ_INSTS_VALU_FMA_F16 SQ_INSTS_VALU_ADD_F16
pmc: SQ_INSTS_VALU_MUL_F16 SQ_INSTS_VALU_TRANS_F16
# ── MFMA / matrix-core FLOPs ───────────────────────────────────────────
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
# ── GPU activity + CU busy ─────────────────────────────────────────────
pmc: GRBM_GUI_ACTIVE GRBM_COUNT
pmc: SQ_BUSY_CU_CYCLES
# ── LDS (shared memory) instruction traffic ────────────────────────────
pmc: SQ_INSTS_LDS
EOF
fi
N_PMC_LINES=$(grep -cE '^pmc:' "$PMC_FILE")
cp -f "$PMC_FILE" "$PROFILE_ROOT/pmc.txt" 2>/dev/null || true

echo ""
echo "=== Plan ==="
echo "  workload    : PyTorch resnet50  local_batch=$LOCAL_BATCH  workers=0  1 GPU"
echo "  profiled    : $PROFILE_STEPS steps/pass  (after warmup_batches)"
echo "  pmc lines   : $N_PMC_LINES  → $N_PMC_LINES workload re-runs (one HW pass each)"
echo "  iter range  : ${ROCPROF_KIR:-<all>}"
echo "  out         : $PROFILE_ROOT"
echo "----- pmc.txt -----"; cat "$PMC_FILE"; echo "-------------------"

# Shared container env (mirrors the working trace block in hyp_imagenet_exp).
COMMON_ENV=(
    --env HIP_VISIBLE_DEVICES=0 --env ROCR_VISIBLE_DEVICES=0
    --env MIOPEN_USER_DB_PATH="$PT_DB" --env MIOPEN_CUSTOM_CACHE_DIR="$PT_DB"
    --env HIP_FORCE_DEV_KERNARG=1
    --env RN_ARCH="${RN_ARCH:-resnet50}"
    --env RN50_NO_GRADSCALER="${RN50_NO_GRADSCALER:-0}"
    --env RANK=0 --env LOCAL_RANK=0 --env WORLD_SIZE=1
    --env MASTER_ADDR=127.0.0.1 --env MASTER_PORT="$((20000 + ${SLURM_JOB_ID:-29551} % 20000))"
    --env TMPDIR=/container_tmp --env TMP=/container_tmp --env TEMP=/container_tmp
)
COMMON_BIND=(
    --bind "$DATA_DIR:/data"
    --bind "$PROJECT_DIR/pytorch_resnet50_ddp.py:/workspace/train.py"
    --bind "$PROFILE_ROOT:/results"
    --bind "$PT_DB:$PT_DB"
    --bind "$HOST_TMP:/container_tmp"
)

# ── Precision (FP32 default; AMP=1 -> bf16 autocast, matches the power run) ──
AMP_ARGS=""
[ "${AMP:-0}" = "1" ] && AMP_ARGS="--amp --amp-dtype ${AMP_DTYPE:-bfloat16}"
echo "  precision   : ${AMP_ARGS:-FP32}"

# ── Warmup pass (NO rocprof) ─────────────────────────────────────────────────
# Counter collection re-runs the workload once per pmc: line. Without a warm
# MIOpen cache, EVERY pass would re-autotune the convolutions (minutes each).
# This single untraced run populates $PT_DB so the N counter passes reuse the
# tuned kernels — turning ~15 autotunes into one.
echo ""; echo "=== Warmup (populate MIOpen cache; not profiled) ==="
apptainer exec --rocm --cleanenv "${COMMON_ENV[@]}" "${COMMON_BIND[@]}" \
    "$CONTAINER_IMAGE" bash -c "
        command -v rocprofv3 >/dev/null 2>&1 || { echo 'FAIL: rocprofv3 not in container'; exit 3; }
        python -u /workspace/train.py -b ${LOCAL_BATCH} --workers 0 --epochs 1 \
            --memory-format nchw --profile-steps ${PROFILE_STEPS} ${AMP_ARGS} --workspace /results /data
    " > "$PROFILE_ROOT/warmup.log" 2>&1
WRC=$?
echo "  warmup exit: $WRC"
if [ "$WRC" = "3" ]; then
    echo "  ✗ rocprofv3 is not installed inside $CONTAINER_IMAGE — cannot counter-profile."
    echo "    Use a rocm/pytorch image that bundles rocprofv3, or install it in the .sif."
    tail -20 "$PROFILE_ROOT/warmup.log"; exit 3
fi

# ── Counter profile (rocprofv3 multi-pass -i pmc.txt) ────────────────────────
echo ""; echo "=== rocprofv3 counter profile (multi-pass) ==="
KIR_ARG=""; [ -n "$ROCPROF_KIR" ] && KIR_ARG="--kernel-iteration-range ${ROCPROF_KIR}"
apptainer exec --rocm --cleanenv "${COMMON_ENV[@]}" "${COMMON_BIND[@]}" \
    "$CONTAINER_IMAGE" bash -c "
        rocprofv3 -i /results/pmc.txt ${KIR_ARG} --output-format csv -d /results \
          -- python -u /workspace/train.py -b ${LOCAL_BATCH} --workers 0 --epochs 1 \
             --memory-format nchw --profile-steps ${PROFILE_STEPS} ${AMP_ARGS} --workspace /results /data
    " > "$PROFILE_ROOT/profile.log" 2>&1
STATUS=$?
echo "  profile exit: $STATUS"

# ── Verify per-pass output ───────────────────────────────────────────────────
mapfile -t CSV_FILES < <(find "$PROFILE_ROOT" -name '*_counter_collection.csv' | sort)
N_CSV=${#CSV_FILES[@]}
echo "  CSVs found  : $N_CSV / $N_PMC_LINES expected"
if [ "$N_CSV" -eq 0 ]; then
    echo "  ✗ No counter CSVs written. Tail of profile.log:"
    tail -40 "$PROFILE_ROOT/profile.log"
    exit "${STATUS:-1}"
fi
[ "$N_CSV" -lt "$N_PMC_LINES" ] && \
    echo "  ! WARNING: $((N_PMC_LINES - N_CSV)) pass(es) produced no CSV (HW per-pass limit / dropped counter)."

echo ""; echo "=== Top-10 kernels by dispatch count (across passes) ==="
for csv in "${CSV_FILES[@]}"; do awk -F',' 'NR>1{print $11}' "$csv"; done \
    | sort | uniq -c | sort -nr | head -10 \
    | awk '{name=$2; for(i=3;i<=NF;i++) name=name" "$i; if(length(name)>80) name=substr(name,1,77)"..."; printf "  %8d  %s\n",$1,name}'

# ── Roofline + cache analysis (host python) ──────────────────────────────────
echo ""; echo "=== Post-processing (roofline + cache table) ==="
PY=$(command -v python3.11 || command -v python3)
"$PY" "$PROJECT_DIR/analyze_roofline.py" "$PROFILE_ROOT" 2>&1 | tail -40 || true
"$PY" "$PROJECT_DIR/plot_roofline.py" --root "$PROFILE_ROOT" \
    --out "$PROJECT_DIR/figures/roofline_pytorch_${SLURM_JOB_ID:-local}.pdf" \
    --csv-out "$PROFILE_ROOT/by_kernel_pytorch.csv" 2>&1 | tail -25 || true

echo ""
echo "  roofline  : figures/roofline_pytorch_${SLURM_JOB_ID:-local}.pdf"
echo "  cache tbl : $PROFILE_ROOT/roofline_summary.csv  (l1_hit_pct / l2_hit_pct columns)"
echo "=== Done at $(date) ==="
exit "$STATUS"
