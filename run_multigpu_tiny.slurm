#!/bin/bash -l
#SBATCH --job-name=rn50_tmpdir
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:2
#SBATCH --time=3:00:00
#SBATCH --output=rn50_multigpu_%j.out
#SBATCH --error=rn50_multigpu_%j.err
#SBATCH --export=NONE

set -e

echo "=== ResNet-50 Multi-GPU Training with TMPDIR Data Staging ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_PER_NODE"
echo "TMPDIR: $TMPDIR"
echo "TMPDIR Space: $(df -h $TMPDIR | tail -1)"
echo ""

# ============================================================================
# 1. LOAD MODULES
# ============================================================================
echo "Loading modules..."
module purge
module load cuda/12.6.2
module load cudnn/9.8.0.87-12
module load cmake
module load nvhpc
module load user-spack

echo "CUDA: $(nvcc --version | head -1)"
echo ""

# ============================================================================
# 2. LOCATE & VERIFY NCCL
# ============================================================================
echo "Locating NCCL..."

NCCL_ROOT=""
# Dynamically search standard paths
for candidate in "$EBROOTNCCL" "$NCCL_HOME" "$CUDA_HOME" "/usr/local/cuda" "/opt/cuda"; do
    if [ -n "$candidate" ] && [ -f "$candidate/include/nccl.h" ]; then
        NCCL_ROOT="$candidate"
        break
    fi
done

# If not found but NVHPC is loaded, dynamically find the latest NCCL version inside it
if [ -z "$NCCL_ROOT" ] && [ -n "$NVHPC_ROOT" ]; then
    echo "Searching NVHPC explicitly for NCCL (this takes a moment)..."
    # Find nccl.h, sort reverse to grab the newest version, and extract the root directory
    NCCL_H_PATH=$(find "$NVHPC_ROOT" -type f -name "nccl.h" 2>/dev/null | sort -r | head -n 1)
    if [ -n "$NCCL_H_PATH" ]; then
        NCCL_ROOT=$(dirname $(dirname "$NCCL_H_PATH"))
        echo "Auto-detected NVHPC bundled NCCL at: $NCCL_ROOT"
    fi
fi

if [ -z "$NCCL_ROOT" ]; then
    echo "ERROR: NCCL not found on this cluster!"
    echo "Please ensure NCCL is loaded (e.g., 'module load nccl') or set \$NCCL_HOME manually."
    exit 1
fi

export LD_LIBRARY_PATH="$NCCL_ROOT/lib:$LD_LIBRARY_PATH"

echo "NCCL root    : $NCCL_ROOT"
echo "NCCL include : $NCCL_ROOT/include"
echo "NCCL lib     : $NCCL_ROOT/lib"
echo ""

# ============================================================================
# 3. NAVIGATE TO PROJECT
# ============================================================================
PROJECT_DIR="$SLURM_SUBMIT_DIR"
echo "Project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

# ============================================================================
# 4. SET TINY IMAGENET DIR
# ============================================================================
echo "=== Data Definition Phase ==="

TINY_DATA_DIR="$PROJECT_DIR/tiny-imagenet-200"

if [ ! -d "$TINY_DATA_DIR" ]; then
    echo "ERROR: Tiny ImageNet directory not found at: $TINY_DATA_DIR"
    echo "Please ensure the dataset is extracted there."
    exit 1
fi

echo "Tiny ImageNet directory: $TINY_DATA_DIR"
echo "  - Train images: $(find $TINY_DATA_DIR/train -name '*.JPEG' 2>/dev/null | wc -l)"
echo "  - Val images:   $(find $TINY_DATA_DIR/val   -name '*.JPEG' 2>/dev/null | wc -l)"
echo ""

# ============================================================================
# 5. BUILD RESNET WITH NCCL
# ============================================================================
echo "=== Building Phase ==="

# Auto-detect GPU architecture (H100 vs A100 vs V100)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "Detected GPU: $GPU_NAME"
if   echo "$GPU_NAME" | grep -qi "H100"; then ARCH=90
elif echo "$GPU_NAME" | grep -qi "A100"; then ARCH=80
elif echo "$GPU_NAME" | grep -qi "V100"; then ARCH=70
else ARCH=80; fi
echo "Using CUDA architecture: sm_${ARCH}"

# Auto-detect cuDNN
CUDNN_PATH=""
for search_dir in "$EBROOTCUDNN" "$CUDNN_ROOT" "$CUDA_HOME" /usr/local/cuda /usr/local /usr; do
    if [ -n "$search_dir" ] && [ -f "$search_dir/include/cudnn.h" ]; then
        CUDNN_PATH="$search_dir"
        break
    fi
done

rm -rf "$PROJECT_DIR/build"
mkdir -p "$PROJECT_DIR/build"
cd "$PROJECT_DIR/build"

echo "Running CMake..."
cmake .. \
    -DUSE_NCCL=ON \
    -DNCCL_ROOT="$NCCL_ROOT" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${ARCH}" \
    ${CUDNN_PATH:+-DCUDNN_PATH="${CUDNN_PATH}"}

if [ $? -ne 0 ]; then
    echo "ERROR: CMake configuration failed"
    exit 1
fi

echo "Building with parallel jobs..."
make -j16 train_multigpu_tiny

if [ $? -ne 0 ]; then
    echo "ERROR: Build failed"
    exit 1
fi

echo "Build completed successfully"
BUILD_DIR="$(pwd)"
cd "$PROJECT_DIR"
echo ""

# ============================================================================
# 6. RUN TRAINING
# ============================================================================
echo "=== Training Phase ==="
echo "Training configuration:"
echo "  - GPUs: 4"
echo "  - Global batch size: 256 (64 per GPU)"
echo "  - Epochs: 90"
echo "  - Learning rate: 0.1"
echo "  - Data location: $TINY_DATA_DIR"
echo ""

# KEY: Pass the EXTRACTED directory path
echo "Starting training at $(date)..."
"$BUILD_DIR/train_multigpu_tiny" "$TINY_DATA_DIR" 256 90 0.1

TRAINING_STATUS=$?

echo ""
echo "Training completed at $(date)"
echo "Training exit status: $TRAINING_STATUS"
echo ""

# ============================================================================
# 7. CLEANUP & SUMMARY
# ============================================================================
echo "=== Job Summary ==="

if [ $TRAINING_STATUS -eq 0 ]; then
    echo "✓ Training completed successfully"
else
    echo "✗ Training failed with exit code: $TRAINING_STATUS"
    exit $TRAINING_STATUS
fi

echo "Checkpoints saved to: $PROJECT_DIR/checkpoints/"
echo "Job ID: $SLURM_JOB_ID"
echo "Total runtime: $(($(date +%s) - $(stat -c %Y "$PROJECT_DIR/build" 2>/dev/null || echo 0))) seconds"
echo ""
echo "=== Job Complete ==="
