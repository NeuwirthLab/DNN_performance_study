#!/bin/bash -l
#SBATCH --job-name=rn50_tiny_roofline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --account=l0003099
#SBATCH --gres=gpu:mi300x:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/ncu_min_amd-%j.out
#SBATCH --error=logs/ncu_min_amd-%j.err
#SBATCH --export=NONE

# ============================================================================
# MINIMAL rocprof-compute profiling for ResNet-50 on MI300X.
#
# Strategy (smallest possible profiled run):
#   1. Use train_multigpu_tiny (Tiny ImageNet 200, 64x64 imgs, ~390 steps).
#   2. Use --roof-only: collects ONLY the roofline counter group, no replay.
#   3. No -k / -d filters in the first pass (docs warn against combining).
#
# Follow-up passes (after this works) — uncomment one block at a time:
#   - kernel filter   : -k MIOpen   (only MIOpen-named conv kernels)
#   - dispatch filter : -d 0:500    (first 500 dispatches across all kernels)
#   - block filter    : -b SQ TCC   (only SQ + L2 counters)
# ============================================================================

set -uo pipefail

echo "========================================================"
echo " ResNet-50 TINY + rocprof-compute (--roof-only)"
echo " Job   : $SLURM_JOB_ID"
echo " Node  : $SLURM_NODELIST"
echo " Start : $(date)"
echo "========================================================"

# ── Paths ───────────────────────────────────────────────────────────────────
WORKSPACE_BASE="${WORKSPACE_BASE:-$HPC_SCRATCH}"
PROJECT_DIR="${PROJECT_DIR:-$WORKSPACE_BASE/resent50/performance-analysis-dl-models-resnet50}"
TINY_DATA_DIR="${TINY_DATA_DIR:-$PROJECT_DIR/tiny-imagenet-200}"

if [ ! -d "$TINY_DATA_DIR/train" ] || [ ! -d "$TINY_DATA_DIR/val" ]; then
    echo "FAIL: tiny-imagenet-200 not found at $TINY_DATA_DIR"; exit 1
fi
echo "  ✓ tiny data : $TINY_DATA_DIR"

# ── Modules + ROCm ──────────────────────────────────────────────────────────
module load cmake
export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"

# ── rocprof-compute venv (reuse the one set up by ncu_amd.slurm) ────────────
ROCPROF_VENV="${ROCPROF_VENV:-$HOME/amd_tools/venv_rocprof}"
if [ ! -f "$ROCPROF_VENV/bin/activate" ]; then
    echo "  → creating venv at $ROCPROF_VENV"
    mkdir -p "$(dirname "$ROCPROF_VENV")"
    python3 -m venv "$ROCPROF_VENV"
fi
source "$ROCPROF_VENV/bin/activate"
python3 -m pip install --upgrade pip > /dev/null
python3 -m pip install -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt > /dev/null
echo "  ✓ rocprof-compute: $(rocprof-compute --version 2>&1 | head -2 | tail -1)"

# ── MIOpen hygiene (prevent GPU-hang from a corrupt user-db) ────────────────
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$HOME/.cache/miopen_min}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

# Bigger workspace cap so MIOpen can pick the fast solvers
export MIOPEN_DEBUG_CONV_WORKSPACE_LIMIT_BYTES=2147483648
export MIOPEN_FIND_MODE=NORMAL

# RCCL is no-op at 1 GPU but match the env from the working full script
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_DISABLE=1
export HIP_VISIBLE_DEVICES="0"

# ── Profiler backend selection ──────────────────────────────────────────────
# rocprof-compute 3.1.0 (rocm-6.4.0 branch) reads $ROCPROF and does:
#   rocprof_cmd = $ROCPROF
#   if not shutil.which(rocprof_cmd): rocprof_cmd = "rocprof"   # falls back to v1
# So $ROCPROF MUST resolve to a real binary on PATH. "rocprofiler-sdk" is a
# library name, NOT a binary — using it silently downgrades to rocprofv1 with:
#   WARNING: Unable to resolve path to rocprofiler-sdk binary. Reverting to default.
# (Seen in job 52205850.) The correct value is "rocprofv3".
if command -v rocprofv3 >/dev/null 2>&1; then
    export ROCPROF="rocprofv3"
    echo "  ✓ ROCPROF=rocprofv3  ($(command -v rocprofv3))"
else
    echo "  ! rocprofv3 not found on PATH — will fall back to rocprofv1 (expect crash)"
fi

# ── Build train_multigpu_tiny ───────────────────────────────────────────────
GFX_ARCH=$(timeout 30 rocminfo 2>/dev/null \
            | awk '/gfx[0-9a-f]+/{for(i=1;i<=NF;i++) if($i ~ /^gfx/){print $i; exit}}' || true)
[ -z "$GFX_ARCH" ] && GFX_ARCH="gfx942"
echo "  ✓ GPU arch  : $GFX_ARCH"

BUILD_DIR="$TMPDIR/rn50_tiny_build"
rm -rf "$BUILD_DIR" && mkdir -p "$BUILD_DIR"

cmake -S "$PROJECT_DIR/hip" -B "$BUILD_DIR" \
      -DCMAKE_HIP_ARCHITECTURES="$GFX_ARCH" \
      -DCMAKE_BUILD_TYPE=Release \
      -DUSE_NCCL=ON \
      > "$BUILD_DIR/cmake.log" 2>&1
if [ $? -ne 0 ]; then
    echo "  FAIL: cmake — last 40 lines:"; tail -40 "$BUILD_DIR/cmake.log"; exit 1
fi

cmake --build "$BUILD_DIR" -j16 --target train_multigpu_tiny \
      > "$BUILD_DIR/build.log" 2>&1
if [ $? -ne 0 ]; then
    echo "  FAIL: build — last 60 lines:"; tail -60 "$BUILD_DIR/build.log"; exit 1
fi
echo "  ✓ built: $BUILD_DIR/train_multigpu_tiny"

# ── Profile ─────────────────────────────────────────────────────────────────
PROFILE_ROOT="$PROJECT_DIR/profiles/min_${SLURM_JOB_ID}"
mkdir -p "$PROFILE_ROOT"

GLOBAL_BATCH="${GLOBAL_BATCH:-256}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-0.1}"

echo ""
echo "=== rocprof-compute profile (--roof-only) ==="
echo "  workload : train_multigpu_tiny   batch=$GLOBAL_BATCH  epochs=$EPOCHS"
echo "  out      : $PROFILE_ROOT"
echo ""

rocprof-compute profile \
    --name rn50_tiny_min \
    --path "$PROFILE_ROOT" \
    --roof-only \
    -- stdbuf -oL "$BUILD_DIR/train_multigpu_tiny" "$TINY_DATA_DIR" "$GLOBAL_BATCH" "$EPOCHS" "$LR"
STATUS=$?
echo "  profile exit: $STATUS"

# ── Analyze (roofline only) ─────────────────────────────────────────────────
WORKLOAD_DIR=$(find "$PROFILE_ROOT" -maxdepth 3 -type d -name "$GFX_ARCH" | head -1)
if [ -n "$WORKLOAD_DIR" ]; then
    ANALYSIS_DIR="$PROFILE_ROOT/analysis"
    mkdir -p "$ANALYSIS_DIR"

    rocprof-compute analyze \
        --path "$WORKLOAD_DIR" \
        --report-roofline \
        > "$ANALYSIS_DIR/roofline.txt" 2>&1 || true

    echo "  ✓ roofline report : $ANALYSIS_DIR/roofline.txt"
    echo ""
    echo "=== Top of roofline report ==="
    head -40 "$ANALYSIS_DIR/roofline.txt" || true
else
    echo "  ! no $GFX_ARCH workload dir found under $PROFILE_ROOT"
    find "$PROFILE_ROOT" -maxdepth 3 -type d | sed 's/^/      /' || true
fi

echo ""
echo "=== Done at $(date) ==="
exit $STATUS
