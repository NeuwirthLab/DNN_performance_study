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

module load cuda/12.6.2
module load cudnn/9.8.0.87-12
module load cmake
# Load nvhpc — it bundles NCCL under its comm_libs directory
module load nvhpc

echo "=== Environment ==="
nvcc --version
nvidia-smi
echo ""

# 2. Navigate to project
cd "$SLURM_SUBMIT_DIR"

# 3. Auto-detect cuDNN path
CUDNN_PATH=""
for search_dir in "$EBROOTCUDNN" "$CUDNN_ROOT" "$CUDA_HOME" /usr/local/cuda /usr/local /usr; do
    if [ -n "$search_dir" ] && [ -f "$search_dir/include/cudnn.h" ]; then
        CUDNN_PATH="$search_dir"
        break
    fi
done
echo "cuDNN path : ${CUDNN_PATH:-not found}"

# 4. Detect GPU architecture
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "Detected GPU: $GPU_NAME"
if   echo "$GPU_NAME" | grep -qi "H100"; then ARCH=90
elif echo "$GPU_NAME" | grep -qi "A100"; then ARCH=80
elif echo "$GPU_NAME" | grep -qi "V100"; then ARCH=70
else ARCH=80; fi
echo "Using CUDA architecture: sm_${ARCH}"

# 5. Locate NCCL — hardcoded Helma SPACK path + fallbacks
echo ""
echo "=== Locating NCCL ==="

# Standalone SPACK NCCL 2.22.3 on NHR FAU Helma (CUDA 12 compatible)
NCCL_SPACK_ROOT="/apps/SPACK/0.23.0/opt/linux-almalinux9-x86_64_v4/gcc-11.4.1/nccl-2.22.3-1-eu6kb6f7p3xfp7py2fgc7q4vodhmt2yw"
# nvhpc bundled NCCL for CUDA 12.6 (fallback)
NCCL_NVHPC_ROOT="/apps/SPACK/0.23.0/opt/linux-almalinux9-x86_64_v4/gcc-11.4.1/nvhpc-24.11-obaxxxcjmxuc4sx2lt2ljfn56ci3f7zt/Linux_x86_64/24.11/comm_libs/12.6/nccl"

NCCL_ROOT_DIR=""
for candidate in \
    "$NCCL_SPACK_ROOT" \
    "$NCCL_NVHPC_ROOT" \
    "$EBROOTNCCL" \
    "$NCCL_HOME" \
    "$NCCL_ROOT"
do
    if [ -n "$candidate" ] && [ -f "$candidate/include/nccl.h" ]; then
        NCCL_ROOT_DIR="$candidate"
        break
    fi
done

if [ -z "$NCCL_ROOT_DIR" ]; then
    echo "ERROR: NCCL not found. Expected at: $NCCL_SPACK_ROOT"
    exit 1
fi

NCCL_INC_PATH="${NCCL_ROOT_DIR}/include"
NCCL_LIB_PATH=$(ls "${NCCL_ROOT_DIR}/lib/libnccl.so" \
                    "${NCCL_ROOT_DIR}/lib64/libnccl.so" 2>/dev/null | head -1)

echo "NCCL root    : ${NCCL_ROOT_DIR}"
echo "NCCL include : ${NCCL_INC_PATH}"
echo "NCCL library : ${NCCL_LIB_PATH}"

if [ -z "$NCCL_LIB_PATH" ]; then
    echo "ERROR: libnccl.so not found under ${NCCL_ROOT_DIR}/lib[64]"
    exit 1
fi

# 6. Compile with NCCL
echo ""
echo "=== Compiling with NCCL... ==="
rm -rf build && mkdir -p build && cd build

cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES="${ARCH}" \
    -DUSE_NCCL=ON \
    -DNCCL_LIB="${NCCL_LIB_PATH}" \
    -DNCCL_INCLUDE="${NCCL_INC_PATH}" \
    ${CUDNN_PATH:+-DCUDNN_PATH="${CUDNN_PATH}"}

make -j"$(nproc)"
cd ..

# 6. Run multi-GPU training
echo "=== Starting Multi-GPU DDP Training... ==="
# Format: ./train_multigpu [dataset] [global_batch] [epochs] [lr]
# Global batch 256 is split evenly across 4 GPUs → 64 per GPU
# TODO: set IMAGENET_DIR to where your ImageNet data lives on Helma scratch
IMAGENET_DIR="${WORK}/imagenet"   # update this path!
if [ ! -d "${IMAGENET_DIR}/train" ]; then
    echo "ERROR: ImageNet not found at ${IMAGENET_DIR}/train"
    echo "Set IMAGENET_DIR in this script to the correct path."
    exit 1
fi
./build/train_multigpu "${IMAGENET_DIR}" 256 90 0.1

echo ""
echo "=== Job Complete at $(date) ==="
