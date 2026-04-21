#!/bin/bash -l
#SBATCH --job-name=dali_bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --output=dali_bench_%j.out
#SBATCH --error=dali_bench_%j.err
#SBATCH --export=ALL

echo "=== DALI Data Loading Benchmark ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo ""

# ── 1. Load modules ──────────────────────────────────────────────────────────
module purge
module load cuda/12.6.2
module load cudnn/9.8.0.87-12
module load cmake
module load nvhpc

echo "CUDA: $(nvcc --version | grep release)"
echo ""

# ── 2. Install DALI if not present ───────────────────────────────────────────
# DALI must match the CUDA major version (12.x -> cuda120)
echo "Checking DALI installation..."
if python3 -c "import nvidia.dali" 2>/dev/null; then
    echo "  DALI already installed"
    python3 -c "import nvidia.dali; print(f'  Version: {nvidia.dali.__version__}')"
else
    echo "  Installing DALI (nvidia-dali-cuda120)..."
    pip install --user nvidia-dali-cuda120 2>&1 | tail -3
    echo "  Done."
fi
echo ""

# ── 3. Project and data paths ────────────────────────────────────────────────
PROJECT_DIR="/hnvme/workspace/m102cd10-amrit/resnet50"
IMAGENET_ZIP="/hnvme/workspace/m102cd10-amrit/imagenet/imagenet-object-localization-challenge.zip"

# Extract to TMPDIR (node-local NVMe)
TMPDIR_DATA="$TMPDIR/imagenet_data"
IMAGENET_DATA="$TMPDIR_DATA/ILSVRC/Data/CLS-LOC"

if [ -d "$IMAGENET_DATA/train" ]; then
    echo "ImageNet already extracted in TMPDIR."
else
    echo "Extracting ImageNet ZIP to TMPDIR..."
    mkdir -p "$TMPDIR_DATA"
    START=$(date +%s)
    unzip -q "$IMAGENET_ZIP" -d "$TMPDIR_DATA"
    END=$(date +%s)
    echo "  Extracted in $((($END-$START)/60))m $(((END-START)%60))s"

    # Rearrange validation set
    echo "  Arranging validation set..."
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
print(f"  Moved {moved} validation images into class subfolders.")
PYEOF
fi

echo "Dataset: $IMAGENET_DATA"
NTRAIN=$(find "$IMAGENET_DATA/train" -name '*.JPEG' | wc -l)
NVAL=$(find "$IMAGENET_DATA/val" -name '*.JPEG' | wc -l)
echo "  Train: $NTRAIN images"
echo "  Val  : $NVAL images"
echo ""

# ── 4. Generate DALI file lists ──────────────────────────────────────────────
cd "$TMPDIR"
echo "Generating DALI file lists..."
python3 "$PROJECT_DIR/dali_benchmark.py" --generate-lists --data-dir "$IMAGENET_DATA"
echo ""

# ── 5. Run DALI benchmark ────────────────────────────────────────────────────
echo "=== DALI Train Pipeline Benchmark (ImageNet 224x224) ==="
python3 "$PROJECT_DIR/dali_benchmark.py" \
    --file-list "$TMPDIR/file_list_train.txt" \
    --batch-size 128 \
    --num-threads 8 \
    --device-id 0 \
    --image-size 224 \
    --warmup 20 \
    --iters 200

echo ""
echo "=== DALI Val Pipeline Benchmark (ImageNet 224x224) ==="
python3 "$PROJECT_DIR/dali_benchmark.py" \
    --file-list "$TMPDIR/file_list_val.txt" \
    --batch-size 128 \
    --num-threads 8 \
    --device-id 0 \
    --image-size 224 \
    --val \
    --warmup 10 \
    --iters 100

echo ""
echo "=== Benchmark Complete ==="
echo "Compare DALI images/sec with your C++ stb_image training output."
echo "If DALI is 3-5x faster, integrate via DALI C API for full GPU utilization."
