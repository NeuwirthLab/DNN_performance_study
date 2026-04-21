#!/bin/bash
#SBATCH --job-name=resnet50_tiny
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1         # Request 1 GPU (e.g. H100)
#SBATCH --time=06:00:00        # Request 6 hours of runtime
#SBATCH --output=resnet_%j.out # Standard output and error log

# ─────────────────────────────────────────────────────────────────────────────
# Usage: sbatch run_resnet.slurm
# ─────────────────────────────────────────────────────────────────────────────

# 1. Load modules (Adjust to your cluster's names)
# Check availability with 'module spider cuda' or 'module avail'

module load cuda/12.6.2            # verify with: module spider cuda
module load cudnn/9.8.0.87-12  
module load python/3.12-conda
module load cmake 

# 2. Compile
echo "Compiling on $(hostname)..."
rm -rf build && mkdir -p build && cd build

# Detect GPU architecture automatically if possible, otherwise default to 80 (A100)
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed 's/\.//g' | head -n 1)
ARCH=${ARCH:-80}
echo "Detected GPU Architecture: ${ARCH}"

cmake -DCMAKE_CUDA_ARCHITECTURES=${ARCH} ..
make -j$(nproc)

# 3. Run
echo "Starting training..."
# Ensure the dataset path is correct relative to the build directory
./train_tinyimagenet ../tiny-imagenet-200 32 10 0.01

echo "Job finished at $(date)"
