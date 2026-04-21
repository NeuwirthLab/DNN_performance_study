#!/bin/bash -l
#SBATCH --job-name=rn50_multigpu_tmpdir
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:2
#SBATCH --time=3:00:00
#SBATCH --output=rn50_multigpu_%j.out
#SBATCH --error=rn50_multigpu_%j.err
#SBATCH --export=ALL

echo "=== ResNet-50 Multi-GPU Training/ inference with TMPDIR Data Staging ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 2 x H100"
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
module load python
# NOTE: do NOT load user-spack — it tries to mkdir /home/woody which fails on compute nodes
# NOTE: do NOT load gcc — nvhpc bundles its own compiler compatible with CUDA 12.6
#       loading system gcc (14.x) causes: "unsupported GNU version" compile error

echo "CUDA: $(nvcc --version | head -1)"
echo ""

# ============================================================================
# 2. LOCATE & VERIFY NCCL
# ============================================================================
echo "Locating NCCL..."


export LD_LIBRARY_PATH="$NCCL_ROOT/lib:$LD_LIBRARY_PATH"
echo "NCCL root: $NCCL_ROOT"

# NCCL performance tuning for 2-GPU NVLink topology
export NCCL_ALGO=Tree
export NCCL_P2P_LEVEL=NVL
export NCCL_DEBUG=INFO

echo ""

# ============================================================================
# 3. PROJECT DIRECTORY — hnvme workspace (accessible from compute nodes)
# ============================================================================
# Source: /home/atuin/m102cd/m102cd10/resnet_multi_gpu/resnet50  ← NFS, NOT accessible
# Target: /hnvme/workspace/m102cd10-amrit/resnet50               ← hnvme, accessible ✓
#
# One-time setup on login node:
#   rsync -av --exclude='build/' --exclude='*.exe' \
#     /home/atuin/m102cd/m102cd10/resnet_multi_gpu/resnet50/ \
#     /hnvme/workspace/m102cd10-amrit/resnet50/

PROJECT_DIR="/hnvme/workspace/m102cd10-amrit/resnet50"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory not found: $PROJECT_DIR"
    echo "Run this on the LOGIN node first:"
    echo "  rsync -av /home/atuin/m102cd/m102cd10/resnet_multi_gpu/resnet50/ \\"
    echo "            /hnvme/workspace/m102cd10-amrit/resnet50/"
    exit 1
fi

echo "Project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

# ============================================================================
# 4. EXTRACT IMAGENET TO TMPDIR (node-local NVMe)
# ============================================================================
echo "=== Data Staging Phase ==="

# ZIP lives in the hnvme workspace (NOT in /home/atuin — NFS, not accessible)
# One-time setup on login node:
#   mkdir -p /hnvme/workspace/m102cd10-amrit/imagenet
#   rsync -av --progress \
#     /home/atuin/m102cd/m102cd10/resnet_multi_gpu/imagenet/imagenet-object-localization-challenge.zip \
#     /hnvme/workspace/m102cd10-amrit/imagenet/
IMAGENET_ZIP="/hnvme/workspace/m102cd10-amrit/imagenet/imagenet-object-localization-challenge.zip"

if [ ! -f "$IMAGENET_ZIP" ]; then
    echo "ERROR: ImageNet ZIP not found at: $IMAGENET_ZIP"
    echo ""
    echo "Run on the LOGIN node:"
    echo "  mkdir -p /hnvme/workspace/m102cd10-amrit/imagenet"
    echo "  rsync -av --progress \\"
    echo "    /home/atuin/m102cd/m102cd10/resnet_multi_gpu/imagenet/imagenet-object-localization-challenge.zip \\"
    echo "    /hnvme/workspace/m102cd10-amrit/imagenet/"
    exit 1
fi

ZIP_SIZE=$(du -h "$IMAGENET_ZIP" | awk '{print $1}')
echo "ImageNet ZIP: $IMAGENET_ZIP ($ZIP_SIZE)"

# TMPDIR is the node-local NVMe — fastest possible I/O for training
TMPDIR_DATA="$TMPDIR/imagenet_data"
mkdir -p "$TMPDIR_DATA"

if [ -d "$TMPDIR_DATA/ILSVRC" ] && [ -d "$TMPDIR_DATA/ILSVRC/Data/CLS-LOC/train" ]; then
    echo "Already extracted in TMPDIR, skipping."
else
    echo "Extracting to TMPDIR (node-local NVMe): $TMPDIR_DATA"
    START_TIME=$(date +%s)
    unzip -q "$IMAGENET_ZIP" -d "$TMPDIR_DATA"
    EXTRACTION_STATUS=$?
    END_TIME=$(date +%s)
    EXTRACTION_TIME=$((END_TIME - START_TIME))

    if [ $EXTRACTION_STATUS -ne 0 ]; then
        echo "ERROR: Extraction failed with status $EXTRACTION_STATUS"
        exit 1
    fi
    echo "Extracted in $((EXTRACTION_TIME / 60))m $((EXTRACTION_TIME % 60))s"

    # Arrange flat Kaggle val/ into val/<wnid>/*.JPEG
    echo "Arranging validation set..."
    export IMAGENET_VAL="$TMPDIR_DATA/ILSVRC/Data/CLS-LOC/val"
    export IMAGENET_CSV="$TMPDIR_DATA/LOC_val_solution.csv"
    python3 - << 'PYEOF'
import os, shutil, csv, sys
val_dir  = os.environ.get('IMAGENET_VAL', '')
csv_path = os.environ.get('IMAGENET_CSV', '')
if not os.path.isdir(val_dir):
    print(f"ERROR: val_dir not found: {val_dir}", file=sys.stderr); sys.exit(1)
if not os.path.isfile(csv_path):
    print(f"WARNING: LOC_val_solution.csv not found. Skipping."); sys.exit(0)
moved = 0
with open(csv_path, newline='') as f:
    for row in csv.DictReader(f):
        img_id = row['ImageId']
        wnid   = row['PredictionString'].split()[0]
        src = os.path.join(val_dir, img_id + '.JPEG')
        if not os.path.isfile(src): continue
        dst_dir = os.path.join(val_dir, wnid)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.move(src, os.path.join(dst_dir, img_id + '.JPEG'))
        moved += 1
print(f"Moved {moved} validation images into class subfolders.")
PYEOF
fi

IMAGENET_DATA="$TMPDIR_DATA/ILSVRC/Data/CLS-LOC"
if [ ! -d "$IMAGENET_DATA/train" ]; then
    echo "ERROR: $IMAGENET_DATA/train not found after extraction"
    exit 1
fi
echo "Dataset ready at: $IMAGENET_DATA"
echo ""

# ============================================================================
# 5. BUILD — write build artefacts to TMPDIR, not NFS
# ============================================================================
echo "=== Building Phase ==="

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "Detected GPU: $GPU_NAME"
if   echo "$GPU_NAME" | grep -qi "H100"; then ARCH=90
elif echo "$GPU_NAME" | grep -qi "A100"; then ARCH=80
elif echo "$GPU_NAME" | grep -qi "V100"; then ARCH=70
else ARCH=80; fi
echo "CUDA architecture: sm_${ARCH}"

CUDNN_PATH=""
for search_dir in "$EBROOTCUDNN" "$CUDNN_ROOT" "$CUDA_HOME" /usr/local/cuda /usr/local /usr; do
    if [ -n "$search_dir" ] && [ -f "$search_dir/include/cudnn.h" ]; then
        CUDNN_PATH="$search_dir"; break
    fi
done

# Build in TMPDIR to avoid writing compiled objects to NFS
BUILD_DIR="$TMPDIR/resnet50_build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Running CMake (build dir: $BUILD_DIR)..."
cmake "$PROJECT_DIR" \
    -DUSE_NCCL=ON \
    -DNCCL_ROOT="$NCCL_ROOT" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${ARCH}" \
    ${CUDNN_PATH:+-DCUDNN_PATH="${CUDNN_PATH}"}

if [ $? -ne 0 ]; then echo "ERROR: CMake failed"; exit 1; fi

make -j16 train_multigpu
if [ $? -ne 0 ]; then echo "ERROR: Build failed"; exit 1; fi

echo "Build successful: $BUILD_DIR/train_multigpu"
cd "$PROJECT_DIR"
echo ""

# ============================================================================
# 6. RUN TRAINING
# ============================================================================
echo "=== Training Phase ==="
echo "  GPUs            : 2 x H100"
echo "  Global batch    : 256 (128 per GPU)"
echo "  Epochs          : 5"
echo "  Learning rate   : 0.1"
echo "  Data (TMPDIR)   : $IMAGENET_DATA"
echo "  Checkpoints     : $PROJECT_DIR/checkpoints/"
echo ""

# Checkpoints go to Lustre ($PROJECT_DIR on $WORK) so they persist after job ends
# TMPDIR is wiped when the job finishes
mkdir -p "$PROJECT_DIR/checkpoints"

echo "Starting training at $(date)..."
# stdbuf -oL forces line-buffered stdout so output appears in .out file in real time
# without this, C++ buffers output and nothing shows until buffer fills (~8KB)
stdbuf -oL "$BUILD_DIR/train_multigpu" "$IMAGENET_DATA" 256 5 0.1

TRAINING_STATUS=$?
echo ""
echo "Training finished at $(date) — exit status: $TRAINING_STATUS"

# ============================================================================
# 7. SUMMARY
# ============================================================================
echo ""
echo "=== Job Summary ==="
if [ $TRAINING_STATUS -eq 0 ]; then
    echo "Training completed successfully"
    echo "Checkpoints: $PROJECT_DIR/checkpoints/"
else
    echo "Training FAILED with exit code: $TRAINING_STATUS"
    exit $TRAINING_STATUS
fi
echo "Job ID: $SLURM_JOB_ID"
echo "=== Job Complete ==="
