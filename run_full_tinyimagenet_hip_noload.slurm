#!/bin/bash -l
#SBATCH --job-name=ResNet50_TinyImagenet_Full_HIP_NoModules
#SBATCH --nodes=2                     # adjust as needed
#SBATCH --ntasks-per-node=1           # one task per node (one GPU per task)
#SBATCH --gpus-per-node=4             # GPUs per node
#SBATCH --time=02:00:00               # walltime
#SBATCH --partition=compute
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=NONE

# ------------------------------------------------------------------
# AMD/ROCm full‑scale Tiny‑Imagenet training (no explicit module loads)
# ------------------------------------------------------------------

echo "========================================================"
 echo " ResNet-50 AMD/HIP Full Test (no module loads)"
 echo " Job   : $SLURM_JOB_ID"
 echo " Node  : $SLURM_NODELIST"
 echo " Start : $(date)"
 echo "========================================================"

# ------------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------------
WORKSPACE_BASE="${WORKSPACE_BASE:-$HPC_SCRATCH}"
PROJECT_DIR="${PROJECT_DIR:-$WORKSPACE_BASE/resent50/performance-analysis-dl-models-resnet50}"

# ------------------------------------------------------------------
# 2. Build environment (use smake if available, otherwise direct CMake)
# ------------------------------------------------------------------
if command -v smake >/dev/null 2>&1; then
  echo "  ✓ using smake to prepare build environment"
  smake clean && smake
else
  echo "  ✓ using direct CMake (module loads omitted)"
  # Assume ROCm environment variables are already exported in the session.
  export USE_HIP=1
  export HCC_AMDGPU_TARGET=gfx942
  export HIP_VISIBLE_DEVICES=0,1,2,3
  export OMP_NUM_THREADS=4
fi

# Verify required tools are present
if ! command -v hipcc >/dev/null 2>&1; then
  echo "  FAIL: hipcc not found in PATH"
  exit 1
fi
if ! command -v cmake >/dev/null 2>&1; then
  echo "  FAIL: cmake not found in PATH"
  exit 1
fi

echo "  ✓ hipcc : $(hipcc --version | head -1)"
 echo "  ✓ cmake : $(cmake --version | head -1 | awk '{print $3}')"
 echo "  ✓ ROCM_PATH = $ROCM_PATH"

# ------------------------------------------------------------------
# 3. GPU detection
# ------------------------------------------------------------------
GPU_COUNT=$(rocm-smi --showproductname | grep -c "Card series" || echo "?")
GPU_INFO=$(rocm-smi --showproductname | head -20)
echo "$GPU_INFO"
GFX_ARCH=$(rocminfo 2>/dev/null | awk '/gfx[0-9a-f]+/{for(i=1;i<=NF;i++) if($i ~ /gfx/){print $i; exit}}')
[ -z "$GFX_ARCH" ] && GFX_ARCH="gfx942"
echo "  ✓ GPU arch : $GFX_ARCH"

# ------------------------------------------------------------------
# 4. Build (reuse binary if already present)
# ------------------------------------------------------------------
BUILD_DIR="$TMPDIR/full_build_hip"
rm -rf "$BUILD_DIR" && mkdir -p "$BUILD_DIR"
cmake -S "$PROJECT_DIR/hip" -B "$BUILD_DIR" \
      -DCMAKE_HIP_ARCHITECTURES="$GFX_ARCH" \
      -DCMAKE_BUILD_TYPE=Release \
      -DUSE_NCCL=ON \
      > "$BUILD_DIR/cmake.log" 2>&1
if [ $? -ne 0 ]; then
  echo "  FAIL: cmake — last 30 lines:"; tail -30 "$BUILD_DIR/cmake.log"; exit 1
fi
cmake --build "$BUILD_DIR" -j8 --target train_multigpu_tiny \
      > "$BUILD_DIR/build.log" 2>&1
if [ $? -ne 0 ]; then
  echo "  FAIL: build — last 40 lines:"; tail -40 "$BUILD_DIR/build.log"; exit 1
fi
echo "  ✓ built: $BUILD_DIR/train_multigpu_tiny"

# ------------------------------------------------------------------
# 5. Run full training (e.g., 10 epochs) on Tiny‑Imagenet
# ------------------------------------------------------------------
TINY_DIR="$PROJECT_DIR/tiny-imagenet-200"
if [ ! -d "$TINY_DIR" ]; then
  echo "  FAIL: Tiny‑Imagenet dataset not found at $TINY_DIR"
  exit 1
fi

echo "\nMulti‑GPU Training: $SLURM_NNODES nodes × $SLURM_GPUS_PER_NODE GPUs | Global Batch: 128 | Local per GPU: 64"
"$BUILD_DIR/train_multigpu_tiny" "$TINY_DIR" 128 $SLURM_GPUS_PER_NODE 0.1
STATUS=$?
if [ $STATUS -eq 0 ]; then
  echo "\nTraining completed successfully."
else
  echo "\nTraining failed with status $STATUS."
fi

# ------------------------------------------------------------------
# 6. Final summary (the binary already prints an epoch table)
# ------------------------------------------------------------------
exit $STATUS
