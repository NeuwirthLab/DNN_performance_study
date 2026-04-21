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

# 7. Extract ImageNet to Node-Local TMPDIR
echo ""
echo "=== Extracting ImageNet to Node-Local TMPDIR ($TMPDIR) ==="
IMAGENET_ZIP="/home/atuin/m102cd/m102cd10/resnet_multi_gpu/imagenet/imagenet-object-localization-challenge.zip"
if [ ! -f "${IMAGENET_ZIP}" ]; then
    echo "ERROR: ImageNet zip not found at ${IMAGENET_ZIP}"
    exit 1
fi

# We use $TMPDIR provided by SLURM
NODE_DATA_DIR="$TMPDIR/imagenet_data"

if [ ! -d "$NODE_DATA_DIR/ILSVRC/Data/CLS-LOC" ]; then
    mkdir -p "$NODE_DATA_DIR"
    # Unzip quietly, suppress output as there are 1.2M files
    echo "Unzipping (this will take a few minutes)..."
    unzip -q -n "$IMAGENET_ZIP" -d "$NODE_DATA_DIR"
    
    echo "Unzip complete. Running validation set prep..."
    # The Kaggle zip puts validation images flat in LOC/val. We need to arrange them.
    # We will run a quick inline python script to arrange them into 1000 WNID folders based on LOC_val_solution.csv
    python3 -c "
import os, shutil, csv
val_dir = '$NODE_DATA_DIR/ILSVRC/Data/CLS-LOC/val'
csv_path = '$NODE_DATA_DIR/LOC_val_solution.csv'

if os.path.exists(val_dir) and os.path.exists(csv_path):
    print('Arranging validation set...')
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            img_id = row[0]
            # Prediction string format: 'n01440764 1 2 3 4' -> we just want the WNID
            wnid = row[1].split()[0]
            
            # create target dir
            target_dir = os.path.join(val_dir, wnid)
            os.makedirs(target_dir, exist_ok=True)
            
            src_file = os.path.join(val_dir, img_id + '.JPEG')
            dst_file = os.path.join(target_dir, img_id + '.JPEG')
            
            if os.path.exists(src_file):
                shutil.move(src_file, dst_file)
    print('Validation set arranged successfully.')
else:
    print('Warning: LOC_val_solution.csv or val_dir not found. Skipping val prep.')
"
else
    echo "Dataset already found in $TMPDIR, skipping unzip."
fi

# 8. Run multi-GPU training
echo ""
echo "=== Starting Multi-GPU DDP Training... ==="
# Format: ./train_multigpu [extracted_data_dir] [global_batch] [epochs] [lr]
# Global batch 256 is split evenly across 4 GPUs → 64 per GPU
./build/train_multigpu "$NODE_DATA_DIR/ILSVRC/Data/CLS-LOC" 256 90 0.1

echo ""
echo "=== Job Complete at $(date) ==="
