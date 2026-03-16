#!/bin/bash
#SBATCH --job-name=resnet50_tiny
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1           # Request 1 GPU (e.g. H100)
#SBATCH --time=04:00:00        # Request 4 hours of runtime
#SBATCH --output=resnet_%j.out # Standard output and error log

# ─────────────────────────────────────────────────────────────────────────────
# Usage: sbatch run_resnet.slurm
# ─────────────────────────────────────────────────────────────────────────────

# 1. Load modules (Adjust to your cluster's names)
# Check availability with 'module spider cuda' or 'module avail'
module purge
module load gcc/11.2.0   # Adjust version
module load cuda/12.1.1  # Adjust version
module load cmake/3.26   # Adjust version

# 2. Compile
echo "Compiling on $(hostname)..."
rm -rf build && mkdir -p build && cd build

# Detect GPU architecture automatically if possible, otherwise default to 80 (A100)
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed 's/\.//g' | head -n 1)
ARCH=${ARCH:-80}
echo "Detected GPU Architecture: ${ARCH}"

cmake -DCMAKE_CUDA_ARCHITECTURES=${ARCH} ..
make -j${SLURM_CPUS_PER_TASK}

# 3. Run
echo "Starting training..."
# Ensure the dataset path is correct relative to the build directory
./train_tinyimagenet ../tiny-imagenet-200 32 10 0.01

echo "Job finished at $(date)"
