#!/bin/bash -l
#SBATCH --job-name=rn50_multigpu_amd_profiled
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --account=l0003099
#SBATCH --gres=gpu:mi300x:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/ncu_imagenet_amd-%j.out
#SBATCH --error=logs/ncu_imagenet_amd-%j.err
#SBATCH --export=NONE

# ============================================================================
# AMD/ROCm ResNet-50 Multi-GPU Training + rocprof-compute Roofline Profiling
#
# What this does:
#   1. Filesystem + module load (rocm/rccl/cmake)
#   2. Locate RCCL
#   3. Activate rocprof-compute profiler (optional, toggle with ENABLE_ROCPROF)
#   4. Detect MI300X gfx arch
#   5. Verify ImageNet-1K dataset
#   6. Build train_multigpu via hip/CMakeLists.txt
#   7. Run training under rocprof-compute profile (safe with RCCL, no replay)
#   8. Analyze: generate roofline, kernel tables, memory hierarchy reports
#   9. Print instructions to launch the Dash web viewer via SSH tunnel
#
# Toggle profiling without editing:
#   sbatch --export=ENABLE_ROCPROF=0 run_multigpu_amd_profiled.slurm
#   sbatch --export=ROCPROF_VENV=/path/to/admin/venv run_multigpu_amd_profiled.slurm
#
# DALI replacement note:
#   DALI is NVIDIA-only. The C++ binary falls back to mt_image_loader +
#   prefetch_loader (multi-threaded stb_image decode + double-buffered prefetch).
# ============================================================================

set -uo pipefail   # -e OFF — matches the working scaling/rocAL scripts.
                    # With -e on, a silent non-zero in rocminfo / find /
                    # pipefail pipelines kills the job with no error in .out.

echo "========================================================"
echo " ResNet-50 Multi-GPU ImageNet Training (AMD/ROCm)"
echo " + rocprof-compute Roofline & Kernel Profiling"
echo " Job   : $SLURM_JOB_ID"
echo " Node  : $SLURM_NODELIST"
echo " TMPDIR: $TMPDIR  (free $(df -h "$TMPDIR" | tail -1 | awk '{print $4}'))"
echo " Start : $(date)"
echo "========================================================"

# ── 1. Project paths ────────────────────────────────────────────────────────
WORKSPACE_BASE="${WORKSPACE_BASE:-$HPC_SCRATCH}"
PROJECT_DIR="${PROJECT_DIR:-$WORKSPACE_BASE/resent50/performance-analysis-dl-models-resnet50}"

# Data is pre-extracted in-repo under ImageNet-1K/.
IMAGENET_DIR="${IMAGENET_DIR:-$PROJECT_DIR/ImageNet-1K/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC}"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "FAIL: PROJECT_DIR not found: $PROJECT_DIR"; exit 1
fi
if [ ! -d "$IMAGENET_DIR/train" ] || [ ! -d "$IMAGENET_DIR/val" ]; then
    echo "FAIL: IMAGENET_DIR must contain train/ and val/: $IMAGENET_DIR"
    echo "      Override with: IMAGENET_DIR=/path/to/imagenet1k bash $0"
    exit 1
fi
echo "  ✓ project dir : $PROJECT_DIR"
echo "  ✓ imagenet dir: $IMAGENET_DIR"

# ── 2. Modules + ROCm env ───────────────────────────────────────────────────
echo ""
echo "[2/7] Module loading"
module load cmake

export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"

echo "  ✓ cmake : $(cmake --version | head -1 | awk '{print $3}')"
echo "  ✓ ROCM_PATH = ${ROCM_PATH}"

# ── 3. Locate RCCL ──────────────────────────────────────────────────────────
echo ""
echo "[3/7] RCCL detection"
RCCL_ROOT=""
for candidate in "$ROCM_PATH" "/opt/rocm"; do
    if [ -n "$candidate" ] && \
       { [ -f "$candidate/include/rccl/rccl.h" ] || [ -f "$candidate/include/rccl.h" ]; }; then
        RCCL_ROOT="$candidate"
        break
    fi
done
if [ -z "$RCCL_ROOT" ]; then
    echo "  FAIL: rccl.h not found under \$ROCM_PATH/include"
    exit 1
fi
echo "  ✓ RCCL : $RCCL_ROOT"

# ── 4. rocprof-compute profiler setup ───────────────────────────────────────
echo ""
echo "[4/7] rocprof-compute profiler setup"

ENABLE_ROCPROF="${ENABLE_ROCPROF:-1}"
ROCPROF_VENV="${ROCPROF_VENV:-$HOME/amd_tools/venv_rocprof}"
ROCPROF_KERNEL_REGEX="${ROCPROF_KERNEL_REGEX:-conv|gemm|bn|batch_norm|sgd|reduce|allreduce|rcclKernel|hip|miopen|winograd|dgrad|wgrad|implicit_gemm|xmma|mfma|valu|lds|barrier}"

if [ "$ENABLE_ROCPROF" = "1" ]; then
    # Create venv on first run (rocprof-compute + its requirements.txt only
    # exist on the GPU compute node, so the install must happen here).
    if [ ! -f "$ROCPROF_VENV/bin/activate" ]; then
        echo "  → creating venv at $ROCPROF_VENV"
        mkdir -p "$(dirname "$ROCPROF_VENV")"
        python3 -m venv "$ROCPROF_VENV"
    fi
    source "$ROCPROF_VENV/bin/activate"

    # Install rocprof-compute deps + GUI deps (idempotent; cached after first run).
    python3 -m pip install --upgrade pip
    python3 -m pip install -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt
    python3 -m pip install dash plotly dash-bootstrap-components pandas

    echo "  ✓ rocprof-compute: $(rocprof-compute --version 2>&1 | head -2 | tail -1)"
    echo "  ✓ Kernel regex    : $ROCPROF_KERNEL_REGEX"
else
    echo "  ! rocprof-compute disabled (ENABLE_ROCPROF=$ENABLE_ROCPROF)"
fi

# ── 5. GPU detection ────────────────────────────────────────────────────────
echo ""
echo "[5/7] GPU detection"
rocm-smi --showproductname 2>/dev/null | head -20 || true
echo "  ... rocminfo (timeout 30s) ..."
GFX_ARCH=$(timeout 30 rocminfo 2>/dev/null \
            | awk '/gfx[0-9a-f]+/{for(i=1;i<=NF;i++) if($i ~ /^gfx/){print $i; exit}}' \
            || true)
[ -z "$GFX_ARCH" ] && GFX_ARCH="gfx942"

GPU_COUNT="${SLURM_GPUS_ON_NODE:-1}"
echo "  ✓ GPU arch  : $GFX_ARCH"
echo "  ✓ GPU count : $GPU_COUNT"

# ── 6. Verify dataset ───────────────────────────────────────────────────────
echo ""
echo "=== Dataset ==="
IMAGENET_DATA="$IMAGENET_DIR"
echo "Using pre-extracted ImageNet-1K at: $IMAGENET_DATA"
echo "  - Train images: $(find "$IMAGENET_DATA/train" -name '*.JPEG' | wc -l)"
echo "  - Val images:   $(find -L "$IMAGENET_DATA/val" -name '*.JPEG' | wc -l)"
echo ""

# ── 7. Training configuration ───────────────────────────────────────────────
GLOBAL_BATCH="${GLOBAL_BATCH:-256}"
EPOCHS="${EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-0.1}"
MAKE_JOBS="${MAKE_JOBS:-16}"
RN50_PROGRESS_INTERVAL="${RN50_PROGRESS_INTERVAL:-50}"
export RN50_PROGRESS_INTERVAL

if [ "$GPU_COUNT" -le 0 ]; then
    echo "ERROR: invalid GPU count: $GPU_COUNT"; exit 1
fi
if [ $((GLOBAL_BATCH % GPU_COUNT)) -ne 0 ]; then
    echo "ERROR: GLOBAL_BATCH=$GLOBAL_BATCH must be divisible by GPU_COUNT=$GPU_COUNT"; exit 1
fi
LOCAL_BATCH=$((GLOBAL_BATCH / GPU_COUNT))

echo "=== Training Configuration ==="
echo "  GPU arch     : $GFX_ARCH"
echo "  GPUs         : $GPU_COUNT"
echo "  Global batch : $GLOBAL_BATCH"
echo "  Local batch  : $LOCAL_BATCH"
echo "  Epochs       : $EPOCHS"
echo "  LR           : $LEARNING_RATE"
echo "  Loader       : mt_image_loader + prefetch_loader (DALI is NVIDIA-only)"
echo "  Progress log : every $RN50_PROGRESS_INTERVAL batches"
echo ""

# ── 8. RCCL tuning (single-node MI300X, XGMI) ───────────────────────────────
export NCCL_DEBUG=INFO
export NCCL_ALGO=Tree
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_DISABLE=1
export HIP_VISIBLE_DEVICES="0"
export HSA_FORCE_FINE_GRAIN_PCIE=1

# Persistent MIOpen kernel cache
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$HOME/.cache/miopen}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

# ── 9. Build train_multigpu ────────────────────────────────────────────────
echo "[6/7] Build (train_multigpu via hip/)"
BUILD_DIR="$TMPDIR/rn50_amd_imagenet_build"
rm -rf "$BUILD_DIR" && mkdir -p "$BUILD_DIR"

cmake -S "$PROJECT_DIR/hip" -B "$BUILD_DIR" \
      -DCMAKE_HIP_ARCHITECTURES="$GFX_ARCH" \
      -DCMAKE_BUILD_TYPE=Release \
      -DUSE_NCCL=ON \
      > "$BUILD_DIR/cmake.log" 2>&1
if [ $? -ne 0 ]; then
    echo "  FAIL: cmake — last 40 lines:"; tail -40 "$BUILD_DIR/cmake.log"; exit 1
fi

cmake --build "$BUILD_DIR" -j"$MAKE_JOBS" --target train_multigpu \
      > "$BUILD_DIR/build.log" 2>&1
if [ $? -ne 0 ]; then
    echo "  FAIL: build — last 60 lines:"; tail -60 "$BUILD_DIR/build.log"; exit 1
fi-ä
echo "  ✓ built: $BUILD_DIR/train_multigpu"
echo ""

# Create profile output dir on persistent storage (not TMPDIR)
PROFILE_ROOT="$PROJECT_DIR/profiles/mi300x_${SLURM_JOB_ID}"
mkdir -p "$PROFILE_ROOT"

# ── 10. Train / Profile / Analyze ───────────────────────────────────────────
echo "[7/7] Training ($EPOCHS epochs, ImageNet-1K, $GPU_COUNT GPUs)"
echo "  Starting at $(date)"
export COLUMNS=200

cd "$PROJECT_DIR"

if [ "$ENABLE_ROCPROF" = "1" ]; then
    WORKLOAD_NAME="rn50_mi300x_${SLURM_JOB_ID}"

    echo ""
    echo "=== rocprof-compute PROFILE ==="
    echo "  Workload name : $WORKLOAD_NAME"
    echo "  Profile root  : $PROFILE_ROOT"
    echo "  GPUs          : all visible (safe with RCCL, no kernel replay)"

    # rocprof-compute profile: runs the app normally while rocprofv2 collects
    # HW counters into a SQLite workload database.
    # rocprof-compute uses --path (alias -p) for the workload directory,
    # not --output-dir. Also drop --kernel: it expects substring matches (one
    # per -k), not a pipe-separated regex; passed verbatim it filters out
    # everything. Profile everything for now; filter at analyze time.
    rocprof-compute profile \
        --name "$WORKLOAD_NAME" \
        --path "$PROFILE_ROOT" \
        -- stdbuf -oL "$BUILD_DIR/train_multigpu" "$IMAGENET_DATA" "$GLOBAL_BATCH" "$EPOCHS" "$LEARNING_RATE"
    STATUS=$?

    # ------------------------------------------------------------------------
    # 10b. ANALYZE: roofline, kernel tables, memory hierarchy
    # ------------------------------------------------------------------------
    echo ""
    echo "=== rocprof-compute ANALYZE ==="

    # Auto-discover the gfx942 workload directory
    WORKLOAD_DIR=$(find "$PROFILE_ROOT" -maxdepth 3 -type d -name "gfx942" | head -1)
    if [ -z "$WORKLOAD_DIR" ]; then
        WORKLOAD_DIR=$(find "$PROFILE_ROOT" -type d | grep -E "gfx942|MI300" | head -1)
    fi

    if [ -n "$WORKLOAD_DIR" ]; then
        ANALYSIS_DIR="$PROFILE_ROOT/analysis"
        mkdir -p "$ANALYSIS_DIR"

        # Text report (console-friendly)
        rocprof-compute analyze \
            --path "$WORKLOAD_DIR" \
            --report-summary \
            --report-kernels \
            --report-roofline \
            > "$ANALYSIS_DIR/analysis.txt" 2>&1 || true

        echo "  ✓ Text report  : $ANALYSIS_DIR/analysis.txt"

        # CSV export (for custom plotting)
        rocprof-compute analyze \
            --path "$WORKLOAD_DIR" \
            --report-summary \
            --report-kernels \
            --report-roofline \
            --format csv \
            --output-dir "$ANALYSIS_DIR/csv" \
            > /dev/null 2>&1 || true
            
        [ -d "$ANALYSIS_DIR/csv" ] && echo "  ✓ CSV reports  : $ANALYSIS_DIR/csv"

        # HTML export (static, no server needed)
        rocprof-compute analyze \
            --path "$WORKLOAD_DIR" \
            --report-summary \
            --report-kernels \
            --report-roofline \
            --format html \
            --output-dir "$ANALYSIS_DIR/html" \
            > /dev/null 2>&1 || true
        [ -d "$ANALYSIS_DIR/html" ] && echo "  ✓ HTML reports : $ANALYSIS_DIR/html"

        # --------------------------------------------------------------------
        # 10c. Quick metric extraction (top MI300X metrics)
        # --------------------------------------------------------------------
        echo ""
        echo "=== Quick MI300X Metrics ==="
        grep -E "(Kernel|GPU_ID|Gfx|Occupancy|VALU|MFMA|Matrix|L2|L3|HBM|xGMI|DRAM|Bandwidth|AI|Roofline|Bound|Cache)" \
            "$ANALYSIS_DIR/analysis.txt" 2>/dev/null | head -60 || true

        echo ""
        echo "=== Profile Artifacts ==="
        echo "  Workload DB  : $WORKLOAD_DIR"
        echo "  Analysis dir : $ANALYSIS_DIR"
    else
        echo "  ! WARNING: Could not find gfx942 workload dir under $PROFILE_ROOT"
        echo "    Directory tree:"
        find "$PROFILE_ROOT" -maxdepth 3 -type d | sed 's/^/      /' || true
    fi

else
    # Non-profiled run (original behavior)
    stdbuf -oL "$BUILD_DIR/train_multigpu" "$IMAGENET_DATA" "$GLOBAL_BATCH" "$EPOCHS" "$LEARNING_RATE"
    STATUS=$?
fi

# ── 11. Job Summary ─────────────────────────────────────────────────────────
echo ""
echo "=== Job Summary ==="
echo "  Finished at $(date)"
if [ $STATUS -eq 0 ]; then
    echo "  ✓ AMD/HIP IMAGENET TRAINING COMPLETE"
    echo "  Checkpoints saved to: $PROJECT_DIR/checkpoints/"
    [ "$ENABLE_ROCPROF" = "1" ] && echo "  Profile saved to    : $PROFILE_ROOT"
    echo "  Job ID: $SLURM_JOB_ID"
else
    echo "  ✗ Training FAILED (exit $STATUS)"
    exit $STATUS
fi

# ── 12. Web Viewer Instructions ─────────────────────────────────────────────
if [ "$ENABLE_ROCPROF" = "1" ] && [ -n "${WORKLOAD_DIR:-}" ]; then
    echo ""
    echo "========================================================"
    echo "  HOW TO VIEW THE ROOFLINE / KERNEL REPORTS"
    echo "========================================================"
    echo ""
    echo "  [A] Dash Web Viewer (interactive, recommended)"
    echo "      1. On your laptop, open a new terminal and run:"
    echo "         ssh -N -L 8050:localhost:8050 $(whoami)@$(hostname -f)"
    echo ""
    echo "      2. On the login node (in another ssh session), run:"
    echo "         source $ROCPROF_VENV/bin/activate"
    echo "         rocprof-compute analyze --path $WORKLOAD_DIR --gui"
    echo ""
    echo "      3. On your laptop browser, open:"
    echo "         http://localhost:8050"
    echo ""
    echo "      Tabs to click:  Roofline  |  Memory Hierarchy  |  Kernel Comparison"
    echo ""
    echo "  [B] Static HTML (no tunnel needed)"
    echo "      If generated above, scp this folder to your laptop:"
    echo "      scp -r $(whoami)@$(hostname -f):$ANALYSIS_DIR/html ./mi300x_report_html"
    echo "      Then open ./mi300x_report_html/index.html in a browser."
    echo ""
    echo "  [C] Text / CSV (command-line only)"
    echo "      cat $ANALYSIS_DIR/analysis.txt | less"
    echo "      # CSV files are under $ANALYSIS_DIR/csv/"
    echo "========================================================"
fi

echo "=== Job Complete ==="