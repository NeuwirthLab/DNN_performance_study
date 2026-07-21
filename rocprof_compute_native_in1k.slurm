#!/bin/bash -l
#SBATCH --job-name=rn50_rcc_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --account=l0003099
#SBATCH --partition=gpu:mi300x
#SBATCH --gres=gpu:mi300x:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/rocprof_compute_native_%j.out
#SBATCH --error=logs/rocprof_compute_native_%j.err
#SBATCH --export=NONE
#
# =============================================================================
# rocprof_compute_native_in1k.slurm v2 — TEST rocprof-compute (rocprofiler-
# compute 3.1.0, ex-Omniperf) on the native HIP ResNet-50, single MI300X.
#
# v1 findings (job 52547742): ROCPROF=rocprofv3 backend engages fine, but the
# FIRST counter pass (SQ_IFETCH_LEVEL, an accumulating "level" counter) GPU-
# faulted the trainer both tries (HW Exception GPU Hang / memory access fault)
# while profiling EVERY dispatch of the epoch. Also learned: the tool WIPES its
# --path dir at startup (logs must live outside it).
#
# v2 strategy — profile only a steady-state dispatch window (raw rocprofv3
# never faulted on SQ counters when iteration-limited), with fallbacks that
# split the counter space if the SQ "level" passes still fault:
#   attempt 1: full blocks + roofline, --dispatch 15000:20000  (~3 steps)
#   on failure, BOTH of (complementary halves):
#   attempt 2: all blocks EXCEPT SQ (caches/memory side), --no-roof
#   attempt 3: SQ block only (FLOPs/instr side), narrow window, --no-roof
# Success = any workload with a populated pmc_perf.csv; each found workload is
# analyzed. Backend is always rocprofv3 (v1 default backend = known May hang).
#
# Knobs: LOCAL_BATCH=256 N_CLASSES=4 DISPATCH_RANGE=15000:20000
#        ATTEMPT1_TIMEOUT=2400 FALLBACK_TIMEOUT=1800
# =============================================================================
set -uo pipefail
echo "==== rocprof-compute native test v2 | job ${SLURM_JOB_ID:-local} | $(date) ===="

WORKSPACE_BASE="${WORKSPACE_BASE:-$HPC_SCRATCH}"
PROJECT_DIR="${PROJECT_DIR:-$WORKSPACE_BASE/resent50/performance-analysis-dl-models-resnet50}"
CLS="$PROJECT_DIR/ImageNet-1K/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC"
cd "$PROJECT_DIR"
module load cmake 2>/dev/null || true
export ROCM_PATH=/opt/rocm; export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:${LD_LIBRARY_PATH:-}"
export TMPDIR="${TMPDIR_PROFILE:-$WORKSPACE_BASE/rocprof_tmp}"; mkdir -p "$TMPDIR"

# a faulted pass must abort fast, not hang writing a multi-GB core dump
ulimit -c 0 2>/dev/null || true
export HSA_ENABLE_COREDUMP=0 ROCM_DISABLE_CRASH_DUMP=1 AMD_LOG_LEVEL=0

command -v rocprof-compute >/dev/null 2>&1 || { echo "FAIL: no rocprof-compute in $ROCM_PATH/bin"; exit 1; }

# ── helper venv (rocprof-compute python deps; cached from May runs) ──────────
ROCPROF_VENV="${ROCPROF_VENV:-$HOME/amd_tools/venv_rocprof}"
if [ ! -f "$ROCPROF_VENV/bin/activate" ]; then
    mkdir -p "$(dirname "$ROCPROF_VENV")"; python3 -m venv "$ROCPROF_VENV"
fi
source "$ROCPROF_VENV/bin/activate"
REQ=$(find "$ROCM_PATH/libexec" -maxdepth 2 -name requirements.txt -path '*rocprof*' 2>/dev/null | head -1)
if [ -n "$REQ" ]; then
    timeout 300 python3 -m pip install -q -r "$REQ" 2>&1 | tail -2 || echo "  (pip install skipped/failed — venv may already be complete)"
fi
echo "  ✓ rocprof-compute: $(command -v rocprof-compute)"
rocprof-compute --version 2>&1 | sed 's/^/    /' | head -6 || true
echo "  ✓ venv: $ROCPROF_VENV"
export ROCPROF=rocprofv3

GFX_ARCH=$(rocminfo 2>/dev/null | awk '/gfx[0-9a-f]+/{for(i=1;i<=NF;i++) if($i ~ /^gfx/){print $i; exit}}'); [ -z "$GFX_ARCH" ] && GFX_ARCH=gfx942

# fresh MIOpen db + FAST find-mode (same recipe as the rocprofv3 retry run)
MIOPEN_DB="$WORKSPACE_BASE/miopen_rcc_test_${SLURM_JOB_ID:-$$}"; rm -rf "$MIOPEN_DB"; mkdir -p "$MIOPEN_DB"
export MIOPEN_USER_DB_PATH="$MIOPEN_DB" MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_DB" MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-2}"
export RN50_ENABLE_BF16=0   # FP32 baseline — unset now defaults to BF16 through-mode

# ── build (incremental, cached) ──────────────────────────────────────────────
BUILD_DIR="${BUILD_DIR:-$PROJECT_DIR/build_profile_in1k}"; mkdir -p "$BUILD_DIR"
cmake -S "$PROJECT_DIR/hip" -B "$BUILD_DIR" -DCMAKE_HIP_ARCHITECTURES="$GFX_ARCH" -DCMAKE_BUILD_TYPE=Release -DUSE_NCCL=ON > "$BUILD_DIR/cmake.log" 2>&1 || { echo FAIL cmake; tail -20 "$BUILD_DIR/cmake.log"; exit 1; }
cmake --build "$BUILD_DIR" -j"${SLURM_CPUS_PER_TASK:-16}" --target train_multigpu > "$BUILD_DIR/build.log" 2>&1 || { echo FAIL build; tail -20 "$BUILD_DIR/build.log"; exit 1; }
BIN="$BUILD_DIR/train_multigpu"; [ -x "$BIN" ] || { echo "FAIL: no binary"; exit 1; }

# ── data: 4-class 224² subset (conv shapes identical to full ImageNet-1K) ───
N_CLASSES="${N_CLASSES:-4}"; DATA_DIR="$PROJECT_DIR/imagenet_subset_${N_CLASSES}"
if [ ! -d "$DATA_DIR/train" ] || [ "$(ls "$DATA_DIR/train" 2>/dev/null | wc -l)" -ne "$N_CLASSES" ]; then
    rm -rf "$DATA_DIR"; mkdir -p "$DATA_DIR/train" "$DATA_DIR/val"
    for c in $(ls "$CLS/train" | head -"$N_CLASSES"); do ln -sfn "$CLS/train/$c" "$DATA_DIR/train/$c"; [ -d "$CLS/val/$c" ] && ln -sfn "$CLS/val/$c" "$DATA_DIR/val/$c"; done
fi

LOCAL_BATCH="${LOCAL_BATCH:-256}"; LR="${LR:-0.1}"
# epoch on subset_4 ≈ 31k dispatches (~1500/step, from 52488168's counter csv);
# 15000:20000 = mid-epoch steady state, well past MIOpen warm-up + init
DISPATCH_RANGE="${DISPATCH_RANGE:-15000:20000}"
DISPATCH_RANGE_NARROW="${DISPATCH_RANGE_NARROW:-16000:17500}"
export RN50_PROGRESS_INTERVAL=10
ROOT="$PROJECT_DIR/profiles/rcc_native_${SLURM_JOB_ID:-$$}"; mkdir -p "$ROOT"
echo "  ✓ binary=$BIN  data=$DATA_DIR ($N_CLASSES cls, 224², bs$LOCAL_BATCH)  out=$ROOT"
echo "  ✓ dispatch window=$DISPATCH_RANGE (narrow=$DISPATCH_RANGE_NARROW)"

# ground truth on supported flags for future iterations (cheap, node-only)
rocprof-compute profile --help > "$ROOT/profile_help.txt" 2>&1 || true

# ── warmup (untraced) to populate MIOpen db so profiled passes are uniform ──
echo "  [warmup] …"
HIP_VISIBLE_DEVICES=0 "$BIN" "$DATA_DIR" "$LOCAL_BATCH" 1 "$LR" > "$ROOT/warmup.log" 2>&1 || echo "  (warmup rc=$? — check $ROOT/warmup.log)"

# ── attempts ─────────────────────────────────────────────────────────────────
ATTEMPT1_TIMEOUT="${ATTEMPT1_TIMEOUT:-2400}"
FALLBACK_TIMEOUT="${FALLBACK_TIMEOUT:-1800}"
run_attempt() { # $1=attempt-id $2=timeout $3...=extra rocprof-compute flags
    local id="$1" tmo="$2"; shift 2
    local dir="$ROOT/attempt_$id" log="$ROOT/attempt_${id}.log"   # log OUTSIDE --path: the tool wipes --path at startup
    rm -rf "$dir"; mkdir -p "$dir"
    echo ""
    echo "=== attempt $id: flags='$*' (timeout ${tmo}s) ==="
    HIP_VISIBLE_DEVICES=0 timeout -k 30 "$tmo" \
        rocprof-compute profile --name native_in1k_rcc --path "$dir" "$@" \
        -- stdbuf -oL "$BIN" "$DATA_DIR" "$LOCAL_BATCH" 1 "$LR" \
        > "$log" 2>&1
    local rc=$?
    { [ "$rc" = 124 ] || [ "$rc" = 137 ]; } && echo "  ✗ attempt $id TIMED OUT (hang) — killed"
    # reap strays: a killed rocprof-compute can orphan the binary on the GPU
    pkill -9 -f "$BIN" 2>/dev/null || true; pkill -9 -f rocprofv3 2>/dev/null || true; sleep 5
    local csv; csv=$(find "$dir" -name 'pmc_perf.csv' 2>/dev/null | head -1)
    if [ -n "$csv" ] && [ "$(wc -l < "$csv")" -gt 1 ]; then
        echo "  ✓ attempt $id OK: $(wc -l < "$csv") rows in $csv"; return 0
    fi
    echo "  ✗ attempt $id failed (rc=$rc) — last 15 lines of $log:"
    tail -15 "$log" | sed 's/^/    | /'
    return 1
}

if run_attempt 1 "$ATTEMPT1_TIMEOUT" --dispatch "$DISPATCH_RANGE"; then
    echo "  → full-blocks profile succeeded; no fallbacks needed"
else
    echo "  → falling back to split counter spaces (complementary halves)"
    run_attempt 2 "$FALLBACK_TIMEOUT" --dispatch "$DISPATCH_RANGE" --no-roof \
        -b TCC TCP TA TD SPI CPC CPF SQC || true
    run_attempt 3 "$FALLBACK_TIMEOUT" --dispatch "$DISPATCH_RANGE_NARROW" --no-roof \
        -b SQ || true
fi

# ── analyze every workload that materialized ─────────────────────────────────
mapfile -t CSVS < <(find "$ROOT" -name 'pmc_perf.csv' 2>/dev/null)
if [ "${#CSVS[@]}" -eq 0 ]; then
    echo ""
    echo "=== ALL ATTEMPTS FAILED — rocprof-compute still unusable on this stack; logs: $ROOT/attempt_*.log ==="
    exit 2
fi
AN="$ROOT/analysis"; mkdir -p "$AN"
for csv in "${CSVS[@]}"; do
    WL=$(dirname "$csv"); tag=$(echo "$WL" | grep -o 'attempt_[0-9]*')
    echo ""
    echo "=== rocprof-compute ANALYZE ($WL) ==="
    timeout 600 rocprof-compute analyze -p "$WL" > "$AN/analysis_${tag}.txt" 2>&1 \
        && echo "  ✓ report: $AN/analysis_${tag}.txt ($(wc -l < "$AN/analysis_${tag}.txt") lines)" \
        || { echo "  ! analyze failed — tail:"; tail -10 "$AN/analysis_${tag}.txt" | sed 's/^/    | /'; }
done

echo ""
echo "=== Top-of-report preview ==="
head -80 "$AN"/analysis_*.txt 2>/dev/null | head -100 || true
echo ""
echo "=== done $(date) | analysis=$AN ==="
