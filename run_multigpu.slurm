#!/bin/bash
#SBATCH --job-name=rn50_multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32            # ~8 CPUs per GPU for data loading
#SBATCH --gres=gpu:4                  # Request 4 GPUs on a single node
#SBATCH --time=24:00:00              # Multi-GPU ~4x faster than single-GPU
#SBATCH --output=rn50_multigpu_%j.out
#SBATCH --error=rn50_multigpu_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# Usage: sbatch run_multigpu.slurm
# Adjust --gres=gpu:N for the number of GPUs you want to use.
# ─────────────────────────────────────────────────────────────────────────────

# 1. Load modules
module purge
module load gcc/11.2.0
module load cuda/12.6.2
module load cudnn/9.8.0.87-12
module load cmake/3.26
# NCCL is typically bundled with the CUDA module on modern HPC clusters.
# If not, add: module load nccl

echo "=== Environment ==="
nvcc --version
nvidia-smi
echo ""

# 2. Navigate to project
cd "$SLURM_SUBMIT_DIR"

# 3. Auto-detect cuDNN path
CUDNN_PATH=""
for search_dir in /usr/local/cuda /usr/local /usr "$CUDA_HOME" "$EBROOTCUDNN" "$CUDNN_ROOT"; do
    if [ -n "$search_dir" ] && [ -f "$search_dir/include/cudnn.h" ]; then
        CUDNN_PATH="$search_dir"
        break
    fi
done

# 4. Detect GPU architecture
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "Detected GPU: $GPU_NAME"
if   echo "$GPU_NAME" | grep -qi "H100"; then ARCH=90
elif echo "$GPU_NAME" | grep -qi "A100"; then ARCH=80
elif echo "$GPU_NAME" | grep -qi "V100"; then ARCH=70
else ARCH=80; fi
echo "Using CUDA architecture: sm_${ARCH}"

# 5. Compile with NCCL enabled
echo "=== Compiling with NCCL... ==="
rm -rf build && mkdir -p build && cd build
cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES="${ARCH}" \
    -DUSE_NCCL=ON \
    ${CUDNN_PATH:+-DCUDNN_PATH="${CUDNN_PATH}"}
make -j"$(nproc)"
cd ..

# 6. Run multi-GPU training
echo "=== Starting Multi-GPU DDP Training... ==="
# Format: ./train_multigpu [dataset] [global_batch] [epochs] [lr]
# Global batch 256 is split evenly across 4 GPUs → 64 per GPU
./build/train_multigpu ./imagenet 256 90 0.1

echo ""
echo "=== Job Complete at $(date) ==="
