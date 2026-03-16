#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_hpc.sh  –  SLURM job script for ResNet-50 cuDNN on HPC
#
# Usage: sbatch run_hpc.sh
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=resnet50_cudnn
#SBATCH --output=resnet50_%j.out        # stdout log  (%j = job ID)
#SBATCH --error=resnet50_%j.err         # stderr log
#SBATCH --partition=h100               # Helma H100 partition
#SBATCH --gres=gpu:h100:1              # 1 × H100 GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00                 # 2 hours max

set -e   # exit immediately on any error (MUST be after all #SBATCH lines)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load required modules
#    Run  `module avail cudnn`  interactively on your HPC to get the exact names
# ─────────────────────────────────────────────────────────────────────────────
module purge
module load CUDA/12.6.0             # verify with: module spider cuda
module load cuDNN                   # verify with: module spider cudnn
# module load GCC/13.3.0            # uncomment if cmake needs a newer gcc

echo "=== Environment ==="
nvcc --version
nvidia-smi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 2. Navigate to project directory
# ─────────────────────────────────────────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Auto-detect cuDNN path (in case the module doesn't set include paths)
# ─────────────────────────────────────────────────────────────────────────────
CUDNN_PATH=""
for search_dir in \
    /usr/local/cuda \
    /usr/local \
    /usr \
    "$CUDA_HOME" \
    "$EBROOTCUDNN" \
    "$CUDNN_ROOT"
do
    if [ -n "$search_dir" ] && [ -f "$search_dir/include/cudnn.h" ]; then
        CUDNN_PATH="$search_dir"
        break
    fi
done

if [ -z "$CUDNN_PATH" ]; then
    # Last resort: search the filesystem
    CUDNN_H=$(find /usr /opt /apps 2>/dev/null -name "cudnn.h" | head -1)
    if [ -n "$CUDNN_H" ]; then
        CUDNN_PATH=$(dirname "$(dirname "$CUDNN_H")")
    fi
fi

if [ -n "$CUDNN_PATH" ]; then
    echo "Found cuDNN at: $CUDNN_PATH"
    export CUDNN_PATH
else
    echo "WARNING: cudnn.h not found automatically – relying on module paths"
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 4. Auto-detect GPU architecture (numeric code for CMake)
# ─────────────────────────────────────────────────────────────────────────────
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "Detected GPU: $GPU_NAME"

if   echo "$GPU_NAME" | grep -qi "H100";         then ARCH=90
elif echo "$GPU_NAME" | grep -qi "A100";         then ARCH=80
elif echo "$GPU_NAME" | grep -qi "A40\|A30";     then ARCH=86
elif echo "$GPU_NAME" | grep -qi "V100";         then ARCH=70
elif echo "$GPU_NAME" | grep -qi "T4";           then ARCH=75
elif echo "$GPU_NAME" | grep -qi "3090\|3080\|3070"; then ARCH=86
elif echo "$GPU_NAME" | grep -qi "2080\|2070";   then ARCH=75
else
    echo "Unknown GPU – defaulting to sm_80"
    ARCH=80
fi
echo "Using CUDA architecture: $ARCH"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 5. Compile with CMake
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Compiling ==="
rm -rf build && mkdir -p build
cd build
cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
    ${CUDNN_PATH:+-DCUDNN_PATH="$CUDNN_PATH"}
make -j"$(nproc)"
cd ..
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 6. Run – Tiny ImageNet training sweep over batch sizes 32 → 1024 (step 32)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Running Tiny ImageNet batch-size sweep ==="

DATASET="../tiny-imagenet-200"
EPOCHS=30
LR=0.01

for BATCH in $(seq 32 32 1024); do
    echo "--- Batch size: $BATCH ---"
    ./build/train_tinyimagenet "$DATASET" "$BATCH" "$EPOCHS" "$LR"
done

echo ""
echo "=== Job Complete ==="
