#!/bin/bash -l
#SBATCH --job-name=rn50_multigpu_tmpdir
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:4
#SBATCH --time=3:00:00
#SBATCH --output=alex_rn50_multigpu_%j.out
#SBATCH --error=alex_rn50_multigpu_%j.err
#SBATCH --export=ALL

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
module load cuda/12.6.1
module load cudnn/9.2.0.82-12.x  
module load cmake
module load nvhpc
module load user-spack/0.19.1 
echo "CUDA: $(nvcc --version | head -1)"
echo ""

# ============================================================================
# 2. LOCATE & VERIFY NCCL
# ============================================================================
echo "Locating NCCL..."
NCCL_ROOT=""
for candidate in "$EBROOTNCCL" "$NCCL_HOME" "$CUDA_HOME" "/usr/local/cuda" "/opt/cuda"; do
    if [ -n "$candidate" ] && [ -f "$candidate/include/nccl.h" ]; then
        NCCL_ROOT="$candidate"
        break
    fi
done
if [ -z "$NCCL_ROOT" ] && [ -n "$NVHPC_ROOT" ]; then
    echo "Searching NVHPC explicitly for NCCL (this takes a moment)..."
    NCCL_H_PATH=$(find "$NVHPC_ROOT" -type f -name "nccl.h" 2>/dev/null | sort -r | head -n 1)
    if [ -n "$NCCL_H_PATH" ]; then
        NCCL_ROOT=$(dirname $(dirname "$NCCL_H_PATH"))
        echo "Auto-detected NVHPC bundled NCCL at: $NCCL_ROOT"
    fi
fi
if [ -z "$NCCL_ROOT" ]; then
    echo "ERROR: NCCL not found on this cluster!"
    exit 1
fi
export LD_LIBRARY_PATH="$NCCL_ROOT/lib:$LD_LIBRARY_PATH"
echo "NCCL root    : $NCCL_ROOT"
echo ""

# ============================================================================
# 3. NAVIGATE TO PROJECT
# ============================================================================
PROJECT_DIR="$SLURM_SUBMIT_DIR"
echo "Project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

# ============================================================================
# 4. EXTRACT IMAGENET TO TMPDIR
# ============================================================================
echo "=== Data Staging Phase ==="
IMAGENET_ZIP="/home/atuin/m102cd/m102cd10/resnet_multi_gpu/imagenet/imagenet-object-localization-challenge.zip"
if [ ! -f "$IMAGENET_ZIP" ]; then
    echo "ERROR: ImageNet ZIP not found at: $IMAGENET_ZIP"
    exit 1
fi
ZIP_SIZE=$(du -h "$IMAGENET_ZIP" | awk '{print $1}')
echo "ImageNet ZIP size: $ZIP_SIZE"

TMPDIR_DATA="$TMPDIR/imagenet_data"
mkdir -p "$TMPDIR_DATA"

if [ -d "$TMPDIR_DATA/ILSVRC" ] && [ -d "$TMPDIR_DATA/ILSVRC/Data/CLS-LOC/train" ]; then
    echo "ImageNet already extracted in TMPDIR, skipping extraction."
else
    echo "Extracting ImageNet ZIP to TMPDIR (10-15 minutes)..."
    START_TIME=$(date +%s)
    unzip -q "$IMAGENET_ZIP" -d "$TMPDIR_DATA"
    EXTRACTION_STATUS=$?
    END_TIME=$(date +%s)
    EXTRACTION_TIME=$((END_TIME - START_TIME))
    if [ $EXTRACTION_STATUS -eq 0 ]; then
        echo "Extraction completed in $((EXTRACTION_TIME / 60)) minutes $((EXTRACTION_TIME % 60)) seconds"
        echo "TMPDIR data size: $(du -sh "$TMPDIR_DATA" | awk '{print $1}')"
        echo "Total files: $(find $TMPDIR_DATA -type f | wc -l)"
    else
        echo "ERROR: Extraction failed with status $EXTRACTION_STATUS"
        exit 1
    fi

    echo "Arranging validation set into class subfolders..."
    export IMAGENET_VAL="$TMPDIR_DATA/ILSVRC/Data/CLS-LOC/val"
    export IMAGENET_CSV="$TMPDIR_DATA/LOC_val_solution.csv"
    python3 - << 'PYEOF'
import os, shutil, csv, sys
val_dir  = os.environ.get('IMAGENET_VAL',  '')
csv_path = os.environ.get('IMAGENET_CSV',  '')
if not os.path.isdir(val_dir):
    print(f"ERROR: val_dir not found: {val_dir}", file=sys.stderr); sys.exit(1)
if not os.path.isfile(csv_path):
    print(f"WARNING: LOC_val_solution.csv not found at {csv_path}. Skipping val prep.")
    sys.exit(0)
moved = 0
with open(csv_path, newline='') as f:
    for row in csv.DictReader(f):
        img_id = row['ImageId']
        wnid   = row['PredictionString'].split()[0]
        src = os.path.join(val_dir, img_id + '.JPEG')
        if not os.path.isfile(src):
            continue
        dst_dir = os.path.join(val_dir, wnid)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.move(src, os.path.join(dst_dir, img_id + '.JPEG'))
        moved += 1
print(f"Moved {moved} validation images into class subfolders.")
PYEOF
fi

IMAGENET_DATA="$TMPDIR_DATA/ILSVRC/Data/CLS-LOC"
if [ ! -d "$IMAGENET_DATA/train" ]; then
    echo "ERROR: train/ directory not found at $IMAGENET_DATA/train/"
    ls -la "$IMAGENET_DATA/" 2>/dev/null || true
    exit 1
fi
echo "Verified dataset structure:"
echo "  - Train images: $(find $IMAGENET_DATA/train -name '*.JPEG' | wc -l)"
echo "  - Val images:   $(find $IMAGENET_DATA/val   -name '*.JPEG' | wc -l)"
echo ""

# ============================================================================
# 5. BUILD RESNET WITH NCCL
# ============================================================================
echo "=== Building Phase ==="
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "Detected GPU: $GPU_NAME"
if   echo "$GPU_NAME" | grep -qi "H100"; then ARCH=90
elif echo "$GPU_NAME" | grep -qi "A100"; then ARCH=80
elif echo "$GPU_NAME" | grep -qi "V100"; then ARCH=70
else ARCH=80; fi
echo "Using CUDA architecture: sm_${ARCH}"

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
cmake --build . -j16 --target train_multigpu
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
echo "  - GPUs: 4 x $GPU_NAME"
echo "  - Global batch size: 256 (64 per GPU)"
echo "  - Epochs: 90"
echo "  - Learning rate: 0.1"
echo "  - Data location: $IMAGENET_DATA"
echo ""

echo "Starting training at $(date)..."
"$BUILD_DIR/train_multigpu" "$IMAGENET_DATA" 256 90 0.1
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
    echo "Training completed successfully"
else
    echo "Training failed with exit code: $TRAINING_STATUS"
    exit $TRAINING_STATUS
fi
echo "Checkpoints saved to: $PROJECT_DIR/checkpoints/"
echo "Job ID: $SLURM_JOB_ID"
echo ""
echo "=== Job Complete ==="
