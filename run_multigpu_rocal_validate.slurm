#!/bin/bash -l
#SBATCH --job-name=rn50_rocal_validate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --account=l0003099
#SBATCH --partition=acc
#SBATCH --reservation=gaod_powcap
#SBATCH --nodelist=gaod0001
#SBATCH --gres=gpu:mi300x:2
#SBATCH --time=00:20:00
#SBATCH --output=logs/rocal/rn50_rocal_validate_%j.out
#SBATCH --error=logs/rocal/rn50_rocal_validate_%j.out
#SBATCH --export=ALL
# ============================================================================
# AMD/ROCm + rocAL ImageNet-1K training on 2× MI300X.
#
# Differs from run_multigpu_imagenet_amd.slurm only in:
#   • cmake flag -DUSE_ROCAL=ON, plus -DROCAL_ROOT pointing at the user-local
#     extraction (rocAL is not installed system-wide on this cluster).
#   • LD_LIBRARY_PATH includes ~/local/opt/rocm-6.4.0/lib so librocal.so is
#     resolvable at runtime.
#   • Comments call out the rocAL data path instead of the CPU loader.
#
# Loader path (build-time selection):
#   USE_ROCAL=ON  → train_multigpu links against librocal, uses
#                   rocal_imagenet_loader.h (GPU JPEG → resize → CMN → NCHW fp32).
#   USE_ROCAL=OFF → falls back to mt_image_loader.h + prefetch_loader.h
#                   (CPU stb_image, what run_multigpu_imagenet_amd.slurm uses).
# ============================================================================

set -uo pipefail   # NB: -e off — cmake/build commands check $? explicitly so we
                    # can preserve build logs to the repo before exiting.

echo "========================================================"
echo " ResNet-50 Multi-GPU ImageNet (AMD/ROCm + rocAL)"
echo " Job   : $SLURM_JOB_ID"
echo " Node  : $SLURM_NODELIST"
echo " TMPDIR: $TMPDIR  (free $(df -h "$TMPDIR" | tail -1 | awk '{print $4}'))"
echo " Start : $(date)"
echo "========================================================"

# ── 1. Project paths ────────────────────────────────────────────────────────
WORKSPACE_BASE="${WORKSPACE_BASE:-$HPC_SCRATCH}"
PROJECT_DIR="${PROJECT_DIR:-/work/scratch/av30nuci/resent50/performance-analysis-dl-models-resnet50}"

IMAGENET_DIR="${IMAGENET_DIR:-$PROJECT_DIR/rocal_research/in1k_mini}"

# Where rocAL was unpacked from rpm2cpio. Override with ROCAL_ROOT=... to
# point at a system install once one is available.
ROCAL_ROOT="${ROCAL_ROOT:-$HOME/local/opt/rocm-6.4.0}"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "FAIL: PROJECT_DIR not found: $PROJECT_DIR"; exit 1
fi
if [ ! -d "$IMAGENET_DIR/train" ] || [ ! -d "$IMAGENET_DIR/val" ]; then
    echo "FAIL: IMAGENET_DIR must contain train/ and val/: $IMAGENET_DIR"; exit 1
fi
if [ ! -f "$ROCAL_ROOT/include/rocal/rocal_api.h" ]; then
    echo "FAIL: rocAL header not found at $ROCAL_ROOT/include/rocal/rocal_api.h"
    echo "      Override with: ROCAL_ROOT=/path bash $0"; exit 1
fi
if [ ! -f "$ROCAL_ROOT/lib/librocal.so" ]; then
    echo "FAIL: librocal.so not found at $ROCAL_ROOT/lib/librocal.so"; exit 1
fi
echo "  ✓ project dir : $PROJECT_DIR"
echo "  ✓ imagenet dir: $IMAGENET_DIR"
echo "  ✓ rocAL root  : $ROCAL_ROOT"

# ── 2. Modules + ROCm env ───────────────────────────────────────────────────
echo ""
echo "[2/6] Module loading"
module load cmake

export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"

# rocAL transitively depends on several libraries that are NOT in /usr/lib64.
# Strategy: instead of adding broad library directories to LD_LIBRARY_PATH
# (which shadows unrelated libs like libstdc++ — bit us once on Ovito's bundle),
# we *symlink only the specific sonames we need* into $ROCAL_ROOT/lib, which is
# already on LD_LIBRARY_PATH. Dynamic linker then sees only what we want.
echo "Wiring runtime deps for rocAL via symlinks into $ROCAL_ROOT/lib:"
mkdir -p "$ROCAL_ROOT/lib"

# libturbojpeg.so.0 — from spack's libjpeg-turbo build.
TURBOJPEG_SRC=$(find /shared/spapps_2026_01_avx2/gcc-11.5.0 \
                     -maxdepth 4 -name 'libturbojpeg.so.0' 2>/dev/null | head -1)
if [ -n "$TURBOJPEG_SRC" ]; then
    ln -sf "$TURBOJPEG_SRC" "$ROCAL_ROOT/lib/libturbojpeg.so.0"
    echo "  ✓ libturbojpeg.so.0 → $TURBOJPEG_SRC"
else
    echo "  ✗ libturbojpeg.so.0 not found in /shared/spapps_*"
fi

# FFmpeg 5.x — rocAL 2.2.0's DT_NEEDED is .so.59/.57/.6. Cluster spack only has
# FFmpeg 7.1 (.so.61), but Ovito 3.8.2 bundles a matching FFmpeg 5.x.
OVITO_FFMPEG="/shared/apps/ovito/3.8.2/lib/ovito"
for soname in libavcodec.so.59 libavformat.so.59 libavutil.so.57 libswscale.so.6; do
    if [ -f "$OVITO_FFMPEG/$soname" ]; then
        ln -sf "$OVITO_FFMPEG/$soname" "$ROCAL_ROOT/lib/$soname"
        echo "  ✓ $soname → $OVITO_FFMPEG/$soname"
    else
        echo "  ✗ $soname not found at $OVITO_FFMPEG"
    fi
done

# rocAL's runtime deps live in the user-local prefix AND at /opt/rocm/lib.
# Order matters — user-local first so our librocal.so + the FFmpeg/turbojpeg
# symlinks above are preferred over anything else with the same name.
export LD_LIBRARY_PATH="$ROCAL_ROOT/lib:$ROCM_PATH/lib:/opt/amdgpu/lib64:$LD_LIBRARY_PATH"

echo "  ✓ cmake : $(cmake --version | head -1 | awk '{print $3}')"
echo "  ✓ ROCM_PATH = ${ROCM_PATH}"

# ── 3. Locate RCCL ──────────────────────────────────────────────────────────
echo ""
echo "[3/6] RCCL detection"
RCCL_ROOT=""
for candidate in "$ROCM_PATH" "/opt/rocm"; do
    if [ -n "$candidate" ] && \
       { [ -f "$candidate/include/rccl/rccl.h" ] || [ -f "$candidate/include/rccl.h" ]; }; then
        RCCL_ROOT="$candidate"; break
    fi
done
if [ -z "$RCCL_ROOT" ]; then
    echo "  FAIL: rccl.h not found under \$ROCM_PATH/include"; exit 1
fi
echo "  ✓ RCCL : $RCCL_ROOT"

# ── 4. GPU detection ────────────────────────────────────────────────────────
echo ""
echo "[4/6] GPU detection"
GPU_INFO=$(rocm-smi --showproductname 2>/dev/null | head -20 || true)
echo "$GPU_INFO"

GFX_ARCH=$(rocminfo 2>/dev/null | awk '/gfx[0-9a-f]+/{for(i=1;i<=NF;i++) if($i ~ /^gfx/){print $i; exit}}')
[ -z "$GFX_ARCH" ] && GFX_ARCH="gfx942"
GPU_COUNT="${SLURM_GPUS_ON_NODE:-2}"
echo "  ✓ GPU arch  : $GFX_ARCH"
echo "  ✓ GPU count : $GPU_COUNT"

# ── 5. Dataset verify ───────────────────────────────────────────────────────
echo ""
echo "=== Dataset ==="
echo "Using pre-extracted ImageNet-1K at: $IMAGENET_DIR"
echo "  - Train images: $(find "$IMAGENET_DIR/train" -name '*.JPEG' | wc -l)"
echo "  - Val images:   $(find -L "$IMAGENET_DIR/val" -name '*.JPEG' | wc -l)"
echo ""

# ── 6. Training configuration ───────────────────────────────────────────────
GLOBAL_BATCH="${GLOBAL_BATCH:-256}"
EPOCHS="${EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-0.1}"
MAKE_JOBS="${MAKE_JOBS:-16}"
RN50_PROGRESS_INTERVAL="${RN50_PROGRESS_INTERVAL:-50}"
export RN50_PROGRESS_INTERVAL

if [ "$GPU_COUNT" -le 0 ]; then echo "ERROR: invalid GPU count: $GPU_COUNT"; exit 1; fi
if [ $((GLOBAL_BATCH % GPU_COUNT)) -ne 0 ]; then
    echo "ERROR: GLOBAL_BATCH=$GLOBAL_BATCH must be divisible by GPU_COUNT=$GPU_COUNT"; exit 1
fi
LOCAL_BATCH=$((GLOBAL_BATCH / GPU_COUNT))

# rocAL backend: CPU until the host's libvx_rpp.so.1 ships gfx942 RPP kernels
# (currently fails vxPublishKernels with VX_ERROR_NOT_SUPPORTED). Override
# with: ROCAL_BACKEND=GPU sbatch run_multigpu_imagenet_amd_rocal.slurm
export ROCAL_BACKEND="${ROCAL_BACKEND:-GPU}"

echo "=== Training Configuration ==="
echo "  GPU arch     : $GFX_ARCH"
echo "  GPUs         : $GPU_COUNT"
echo "  Global batch : $GLOBAL_BATCH"
echo "  Local batch  : $LOCAL_BATCH"
echo "  Epochs       : $EPOCHS"
echo "  LR           : $LEARNING_RATE"
echo "  Loader       : rocAL (USE_ROCAL=ON, ROCAL_BACKEND=$ROCAL_BACKEND)"
echo "  Pipeline     : JPEG → Resize(256) → CMN(224) → NCHW fp32 → device"
echo "  Progress log : every $RN50_PROGRESS_INTERVAL batches"
echo ""

# ── 7. RCCL tuning (single-node MI300X, XGMI) ───────────────────────────────
export NCCL_DEBUG=INFO
export NCCL_ALGO=Tree
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_DISABLE=1
export HIP_VISIBLE_DEVICES="0,1"
export HSA_FORCE_FINE_GRAIN_PCIE=1

# Persistent MIOpen kernel cache — autotune cost paid only on first run.
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$HOME/.cache/miopen}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

# ── 8. Build train_multigpu via hip/CMakeLists.txt with USE_ROCAL=ON ────────
echo "[5/6] Build (train_multigpu via hip/, USE_ROCAL=ON)"
BUILD_DIR="$TMPDIR/rn50_amd_imagenet_rocal_build"
rm -rf "$BUILD_DIR" && mkdir -p "$BUILD_DIR"

LOG_OUT="$PROJECT_DIR/rocal_build_logs_${SLURM_JOB_ID:-$$}"
mkdir -p "$LOG_OUT"

cmake -S "$PROJECT_DIR/hip" -B "$BUILD_DIR" \
      -DCMAKE_HIP_ARCHITECTURES="$GFX_ARCH" \
      -DCMAKE_BUILD_TYPE=Release \
      -DUSE_NCCL=ON \
      -DUSE_ROCAL=ON \
      -DROCAL_ROOT="$ROCAL_ROOT" \
      > "$BUILD_DIR/cmake.log" 2>&1
CMAKE_RC=$?
cp "$BUILD_DIR/cmake.log" "$LOG_OUT/cmake.log" 2>/dev/null || true
if [ "$CMAKE_RC" -ne 0 ]; then
    echo "  FAIL: cmake (rc=$CMAKE_RC) — last 60 lines:"
    tail -60 "$BUILD_DIR/cmake.log"
    echo "  → full log: $LOG_OUT/cmake.log"
    exit 1
fi

cmake --build "$BUILD_DIR" -j"$MAKE_JOBS" --target train_multigpu \
      > "$BUILD_DIR/build.log" 2>&1
BUILD_RC=$?
cp "$BUILD_DIR/build.log" "$LOG_OUT/build.log" 2>/dev/null || true
if [ "$BUILD_RC" -ne 0 ]; then
    echo "  FAIL: build (rc=$BUILD_RC) — last 80 lines:"
    tail -80 "$BUILD_DIR/build.log"
    echo "  → full log: $LOG_OUT/build.log"
    exit 1
fi
echo "  ✓ built: $BUILD_DIR/train_multigpu"

# Confirm the binary actually pulled in librocal.
echo "  Linked rocAL?:"
ldd "$BUILD_DIR/train_multigpu" 2>&1 | grep -E 'rocal|rpp|openvx' || echo "    (none — USE_ROCAL did not link)"
echo ""

# ── 9. Train ────────────────────────────────────────────────────────────────
echo "[6/6] Training ($EPOCHS epochs, ImageNet-1K, $GPU_COUNT GPUs, rocAL)"
echo "  Starting at $(date)"
export COLUMNS=200

cd "$PROJECT_DIR"
stdbuf -oL "$BUILD_DIR/train_multigpu" "$IMAGENET_DIR" "$GLOBAL_BATCH" "$EPOCHS" "$LEARNING_RATE"
STATUS=$?

echo ""
echo "=== Job Summary ==="
echo "  Finished at $(date)"
if [ $STATUS -eq 0 ]; then
    echo "  ✓ AMD/HIP + rocAL IMAGENET TRAINING COMPLETE"
    echo "  Checkpoints saved to: $PROJECT_DIR/checkpoints/"
    echo "  Job ID: $SLURM_JOB_ID"
else
    echo "  ✗ Training FAILED (exit $STATUS)"
    exit $STATUS
fi
echo "=== Job Complete ==="
