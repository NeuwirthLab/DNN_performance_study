#!/bin/bash -l
#SBATCH --job-name=ngc_resnet_benchmark
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=ngc_resnet_benchmark_%j.out
#SBATCH --error=ngc_resnet_benchmark_%j.err
#SBATCH --export=NONE

set -euo pipefail

echo "=== PyTorch NGC ResNet-50 Benchmark ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "TMPDIR: $TMPDIR"

# ============================================================================
# 1. LOAD MODULES
# ============================================================================
echo "Loading modules..."
module purge
module load nvhpc
module load cuda/12.6.2
module load python


# ============================================================================
# 2. EXTRACT IMAGENET TO TMPDIR
# ============================================================================
IMAGENET_ZIP="/hnvme/workspace/m102cd10-amrit/imagenet/imagenet-object-localization-challenge.zip"

if [ ! -f "$IMAGENET_ZIP" ]; then
    echo "ERROR: ImageNet ZIP not found: $IMAGENET_ZIP"
    exit 1
fi

echo "=== Data Staging Phase ==="

TMPDIR_DATA="$TMPDIR/imagenet_data"
mkdir -p "$TMPDIR_DATA"

if [ -d "$TMPDIR_DATA/ILSVRC" ] && [ -d "$TMPDIR_DATA/ILSVRC/Data/CLS-LOC/train" ]; then
    echo "ImageNet already extracted in TMPDIR, skipping extraction."
else
    echo "Extracting ImageNet ZIP to TMPDIR: $TMPDIR_DATA"
    unzip -q "$IMAGENET_ZIP" -d "$TMPDIR_DATA"

    echo "Arranging validation set into class subfolders..."
    export IMAGENET_VAL="$TMPDIR_DATA/ILSVRC/Data/CLS-LOC/val"
    export IMAGENET_CSV="$TMPDIR_DATA/LOC_val_solution.csv"
    python3 - <<'PYEOF'
import csv
import os
import shutil
import sys

val_dir = os.environ.get("IMAGENET_VAL", "")
csv_path = os.environ.get("IMAGENET_CSV", "")

if not os.path.isdir(val_dir):
    print(f"ERROR: val_dir not found: {val_dir}", file=sys.stderr)
    sys.exit(1)
if not os.path.isfile(csv_path):
    print(f"WARNING: LOC_val_solution.csv not found at {csv_path}. Skipping val prep.")
    sys.exit(0)

moved = 0
with open(csv_path, newline="") as handle:
    for row in csv.DictReader(handle):
        image_id = row["ImageId"]
        wnid = row["PredictionString"].split()[0]

        source = os.path.join(val_dir, image_id + ".JPEG")
        if not os.path.isfile(source):
            continue

        destination_dir = os.path.join(val_dir, wnid)
        os.makedirs(destination_dir, exist_ok=True)
        shutil.move(source, os.path.join(destination_dir, image_id + ".JPEG"))
        moved += 1

print(f"Moved {moved} validation images into class subfolders.")
PYEOF
fi

IMAGENET_DATA="$TMPDIR_DATA/ILSVRC/Data/CLS-LOC"
if [ ! -d "$IMAGENET_DATA/train" ]; then
    echo "ERROR: train/ directory not found at $IMAGENET_DATA/train"
    exit 1
fi

echo "Verified dataset structure:"
echo "  - Train images: $(find "$IMAGENET_DATA/train" -name '*.JPEG' | wc -l)"
echo "  - Val images:   $(find "$IMAGENET_DATA/val" -name '*.JPEG' | wc -l)"
echo ""

# ============================================================================
# 3. RUN NVIDIA NGC BENCHMARK
# ============================================================================

# Ensure DeepLearningExamples exists
if [ ! -d "DeepLearningExamples" ]; then
    echo "Cloning DeepLearningExamples repo..."
    git clone https://github.com/NVIDIA/DeepLearningExamples.git
fi

PROJECT_PWD=$(pwd)

BIND_ARGS=(
    --bind "${IMAGENET_DATA}:/data"
    --bind "${PROJECT_PWD}/DeepLearningExamples/PyTorch/Classification/ConvNets/resnet50v1.5:/workspace/resnet50v1.5"
)

echo "Running Benchmark in NGC Container..."
apptainer exec --nv --cleanenv \
    "${BIND_ARGS[@]}" \
    docker://nvcr.io/nvidia/pytorch:23.12-py3 \
    bash -c 'cd /workspace/resnet50v1.5 && python -m torch.distributed.run \
    --nproc_per_node=2 \
    main.py \
    -a resnet50 \
    -b 128 \
    --workers 8 \
    --amp \
    --workspace /data \
    --training-only \
    --epochs 2 \
    /data'

echo ""
echo "=== Benchmark Complete ==="
