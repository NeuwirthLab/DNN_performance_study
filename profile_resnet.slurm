#!/bin/bash
#SBATCH --job-name=resnet50_profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1           # Request 1 GPU (e.g. H100)
#SBATCH --time=00:15:00        # Request 15 minutes of runtime
#SBATCH --output=profile_%j.out # Standard output and error log

# ─────────────────────────────────────────────────────────────────────────────
# Usage: sbatch profile_resnet.slurm
# This runs a short training session under Nsight Systems (`nsys profile`).
# Download `resnet_profile.nsys-rep` to your local machine to view it.
# ─────────────────────────────────────────────────────────────────────────────

# 1. Load modules (Adjust to your cluster's names)
module purge
module load gcc/11.2.0   # Adjust version
module load cuda/12.1.1  # Adjust version
module load cmake/3.26   # Adjust version

# 2. Compile
echo "Compiling on $(hostname)..."
rm -rf build && mkdir -p build && cd build

ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed 's/\.//g' | head -n 1)
ARCH=${ARCH:-80}
echo "Detected GPU Architecture: ${ARCH}"

cmake -DCMAKE_CUDA_ARCHITECTURES=${ARCH} ..
make -j${SLURM_CPUS_PER_TASK}

# 3. Profile
echo "Starting Nsight Systems Profiling..."
# Profile for 1 epoch, small batch size to just capture the steady-state
nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --output=../resnet_profile \
    --force-overwrite=true \
    ./train_tinyimagenet ../tiny-imagenet-200 32 1 0.01

echo "Profiling finished at $(date). Download resnet_profile.nsys-rep."
