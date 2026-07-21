#!/bin/bash -l
#SBATCH --job-name=rn50_tiny_noprof
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --account=l0003099
#SBATCH --gres=gpu:mi300x:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/ncu_min_noprof-%j.out
#SBATCH --error=logs/ncu_min_noprof-%j.err
#SBATCH --export=NONE

# Sanity-run: train_multigpu_tiny WITHOUT rocprof-compute.
# If this crashes with the same "Write access to a read-only page" → it's
# the tiny code path. If it runs fine → the page fault is rocprofv1's fault.

set -uo pipefail

echo "[noprof] start $(date)"

WORKSPACE_BASE="${WORKSPACE_BASE:-$HPC_SCRATCH}"
PROJECT_DIR="${PROJECT_DIR:-$WORKSPACE_BASE/resent50/performance-analysis-dl-models-resnet50}"
TINY_DATA_DIR="${TINY_DATA_DIR:-$PROJECT_DIR/tiny-imagenet-200}"

module load cmake
export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"

# Same MIOpen hygiene as the working full-ImageNet run
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$HOME/.cache/miopen_min}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
export MIOPEN_DEBUG_CONV_WORKSPACE_LIMIT_BYTES=2147483648
export MIOPEN_FIND_MODE=NORMAL

export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_DISABLE=1
export HIP_VISIBLE_DEVICES="0"

# Synchronous HIP so a crash points to the actual kernel
export HIP_LAUNCH_BLOCKING=1

GFX_ARCH=$(timeout 30 rocminfo 2>/dev/null \
            | awk '/gfx[0-9a-f]+/{for(i=1;i<=NF;i++) if($i ~ /^gfx/){print $i; exit}}' || true)
[ -z "$GFX_ARCH" ] && GFX_ARCH="gfx942"

BUILD_DIR="$TMPDIR/rn50_tiny_build_noprof"
rm -rf "$BUILD_DIR" && mkdir -p "$BUILD_DIR"

cmake -S "$PROJECT_DIR/hip" -B "$BUILD_DIR" \
      -DCMAKE_HIP_ARCHITECTURES="$GFX_ARCH" \
      -DCMAKE_BUILD_TYPE=Release \
      -DUSE_NCCL=ON \
      > "$BUILD_DIR/cmake.log" 2>&1 \
   || { tail -40 "$BUILD_DIR/cmake.log"; exit 1; }

cmake --build "$BUILD_DIR" -j16 --target train_multigpu_tiny \
      > "$BUILD_DIR/build.log" 2>&1 \
   || { tail -60 "$BUILD_DIR/build.log"; exit 1; }

# Dump rocprof-compute help so we can see the actual backend-selection flag
# (the public docs don't list it, but the binary exposes it).
ROCPROF_VENV="${ROCPROF_VENV:-$HOME/amd_tools/venv_rocprof}"
if [ -f "$ROCPROF_VENV/bin/activate" ]; then
    source "$ROCPROF_VENV/bin/activate"
    echo ""
    echo "========================================================"
    echo "rocprof-compute profile --help  (backend-related lines)"
    echo "========================================================"
    rocprof-compute profile --help 2>&1 | grep -iE 'profiler|sdk|backend|rocprofv|--name|--path|--roof|kernel|dispatch|block' || true
    echo "--------------------------------------------------------"
    echo ""
fi

echo "[noprof] launching binary"
stdbuf -oL "$BUILD_DIR/train_multigpu_tiny" "$TINY_DATA_DIR" 256 1 0.1
STATUS=$?
echo "[noprof] exit=$STATUS  finish=$(date)"
exit $STATUS
