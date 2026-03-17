#!/bin/bash
#SBATCH --job-name=resnet50_imagenet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      # Need more CPUs for image pre-processing
#SBATCH --gres=gpu:1           # Request 1 GPU (e.g. H100)
#SBATCH --time=24:00:00        # ImageNet takes a long time
#SBATCH --output=rn50_imgnet_%j.out # Standard output and error log

# ─────────────────────────────────────────────────────────────────────────────
# Usage: sbatch run_imagenet.slurm
# ─────────────────────────────────────────────────────────────────────────────

# 1. Load modules
module purge
module load gcc/11.2.0        # Keep your existing GCC version
module load cuda/12.6.2       # High-performance CUDA 12
module load cudnn/9.8.0.87-12 # Latest cuDNN built for CUDA 12
module load cmake/3.26

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
echo "Starting ImageNet-1k training..."
# Ensure the dataset path is correct relative to the build directory
# Format: ./train_imagenet [dataset_path] [batch_size] [epochs] [lr]
./train_imagenet ../imagenet 128 90 0.1

echo "Job finished at $(date)"
