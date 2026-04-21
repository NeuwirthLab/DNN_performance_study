#!/bin/bash -l
#SBATCH --job-name=rn50_multigpu_dali
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:2
#SBATCH --time=2:00:00
#SBATCH --output=rn50_multigpu_dali_%j.out
#SBATCH --error=rn50_multigpu_dali_%j.err
#SBATCH --export=NONE

set -e

echo "=== ResNet-50 Multi-GPU Training with TMPDIR Staging + Native DALI ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "TMPDIR: $TMPDIR"
echo "TMPDIR Space: $(df -h "$TMPDIR" | tail -1)"
echo ""

# ============================================================================
# 1. LOAD MODULES
# ============================================================================
echo "Loading modules..."
# This cluster keeps gpu-env/2025 as a sticky module; plain `module purge` exits with code 1 under `set -e`, which aborts the job before it starts.
# module --force purge

# Prefer the newer toolchain, but fall back to the versions already used in
# the repo's known-good Slurm scripts for this cluster.
module load nvhpc
module load cmake
module load nvhpc
module load python
module load cuda/12.6.2
module load cudnn
echo "CUDA: $(nvcc --version | head -1)"
echo ""

# ============================================================================
# 2. LOCATE NCCL / cuDNN / CONTAINER
# ============================================================================
echo "Locating NCCL..."

NCCL_ROOT=""
for candidate in "${EBROOTNCCL:-}" "${NCCL_HOME:-}" "${CUDA_HOME:-}" "/usr/local/cuda" "/opt/cuda"; do
    if [ -n "${candidate:-}" ] && [ -f "$candidate/include/nccl.h" ]; then
        NCCL_ROOT="$candidate"
        break
    fi
done

if [ -z "$NCCL_ROOT" ] && [ -n "${NVHPC_ROOT:-}" ]; then
    NCCL_H_PATH=$(find "$NVHPC_ROOT" -type f -name "nccl.h" 2>/dev/null | sort -r | head -n 1)
    if [ -n "$NCCL_H_PATH" ]; then
        NCCL_ROOT=$(dirname "$(dirname "$NCCL_H_PATH")")
    fi
fi

if [ -z "$NCCL_ROOT" ]; then
    echo "ERROR: NCCL not found"
    exit 1
fi

echo "NCCL root: $NCCL_ROOT"

CUDNN_PATH=""
for search_dir in "${EBROOTCUDNN:-}" "${CUDNN_ROOT:-}" "${CUDA_HOME:-}" /usr/local/cuda /usr/local /usr; do
    if [ -n "${search_dir:-}" ] && [ -f "$search_dir/include/cudnn.h" ]; then
        CUDNN_PATH="$search_dir"
        break
    fi
done

PROJECT_DIR="/hnvme/workspace/m102cd10-amrit/resnet50"
CONTAINER_IMAGE="$PROJECT_DIR/containers/nvidia-pytorch.sif"
IMAGENET_ZIP="/hnvme/workspace/m102cd10-amrit/imagenet/imagenet-object-localization-challenge.zip"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory not found: $PROJECT_DIR"
    exit 1
fi

if [ ! -f "$CONTAINER_IMAGE" ]; then
    echo "ERROR: Container image not found: $CONTAINER_IMAGE"
    exit 1
fi

if [ ! -f "$IMAGENET_ZIP" ]; then
    echo "ERROR: ImageNet ZIP not found: $IMAGENET_ZIP"
    exit 1
fi

echo "Project directory: $PROJECT_DIR"
echo "Container image: $CONTAINER_IMAGE"
echo "ImageNet ZIP: $IMAGENET_ZIP"
echo ""

# ============================================================================
# 3. EXTRACT IMAGENET TO TMPDIR
# ============================================================================
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
# 4. TRAINING CONFIGURATION
# ============================================================================
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if echo "$GPU_NAME" | grep -qi "H100"; then
    ARCH=90
elif echo "$GPU_NAME" | grep -qi "A100"; then
    ARCH=80
elif echo "$GPU_NAME" | grep -qi "V100"; then
    ARCH=70
else
    ARCH=80
fi

NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}}
GLOBAL_BATCH=${GLOBAL_BATCH:-256}
EPOCHS=${EPOCHS:-5}
LEARNING_RATE=${LEARNING_RATE:-0.1}
DALI_DECODER=${DALI_DECODER:-mixed}
DALI_THREADS=${DALI_THREADS:-8}
MAKE_JOBS=${MAKE_JOBS:-16}

if [ "$NUM_GPUS" -le 0 ]; then
    echo "ERROR: invalid GPU count: $NUM_GPUS"
    exit 1
fi
if [ $((GLOBAL_BATCH % NUM_GPUS)) -ne 0 ]; then
    echo "ERROR: GLOBAL_BATCH=$GLOBAL_BATCH must be divisible by NUM_GPUS=$NUM_GPUS"
    exit 1
fi

LOCAL_BATCH=$((GLOBAL_BATCH / NUM_GPUS))

echo "=== Training Configuration ==="
echo "GPU arch        : sm_${ARCH}"
echo "GPUs            : $NUM_GPUS"
echo "Global batch    : $GLOBAL_BATCH"
echo "Local batch     : $LOCAL_BATCH"
echo "Epochs          : $EPOCHS"
echo "Learning rate   : $LEARNING_RATE"
echo "DALI decoder    : $DALI_DECODER"
echo "DALI threads    : $DALI_THREADS"
echo ""

# ============================================================================
# 5. PREP TMPDIR PATHS
# ============================================================================
ASSET_ROOT="$TMPDIR/dali_assets"
FILELIST_DIR="$ASSET_ROOT/filelists"
PIPELINE_DIR="$ASSET_ROOT/pipelines"
BUILD_DIR="$TMPDIR/resnet50_build_dali"
DALI_PREFIX="$TMPDIR/dali_native"
CONTAINER_SCRIPT="$TMPDIR/run_resnet50_dali_inside_container.sh"

rm -rf "$ASSET_ROOT" "$BUILD_DIR" "$DALI_PREFIX"
mkdir -p "$FILELIST_DIR" "$PIPELINE_DIR" "$BUILD_DIR"

# ============================================================================
# 6. RUN DALI ASSET GENERATION + BUILD + TRAINING INSIDE CONTAINER
# ============================================================================
echo "=== Container Build + Training Phase ==="

BIND_ARGS=(
    --bind "$PROJECT_DIR:$PROJECT_DIR"
    --bind "$TMPDIR:$TMPDIR"
    --bind "$NCCL_ROOT:$NCCL_ROOT"
)

if [ -n "$CUDNN_PATH" ] && [ -d "$CUDNN_PATH" ]; then
    BIND_ARGS+=(--bind "$CUDNN_PATH:$CUDNN_PATH")
fi

{
    printf '#!/bin/bash\n'
    printf 'set -euo pipefail\n\n'
    printf 'PROJECT_DIR=%q\n' "$PROJECT_DIR"
    printf 'IMAGENET_DATA=%q\n' "$IMAGENET_DATA"
    printf 'FILELIST_DIR=%q\n' "$FILELIST_DIR"
    printf 'PIPELINE_DIR=%q\n' "$PIPELINE_DIR"
    printf 'BUILD_DIR=%q\n' "$BUILD_DIR"
    printf 'DALI_PREFIX=%q\n' "$DALI_PREFIX"
    printf 'NCCL_ROOT=%q\n' "$NCCL_ROOT"
    printf 'CUDNN_PATH=%q\n' "$CUDNN_PATH"
    printf 'NUM_GPUS=%q\n' "$NUM_GPUS"
    printf 'GLOBAL_BATCH=%q\n' "$GLOBAL_BATCH"
    printf 'LOCAL_BATCH=%q\n' "$LOCAL_BATCH"
    printf 'EPOCHS=%q\n' "$EPOCHS"
    printf 'LEARNING_RATE=%q\n' "$LEARNING_RATE"
    printf 'DALI_DECODER=%q\n' "$DALI_DECODER"
    printf 'DALI_THREADS=%q\n' "$DALI_THREADS"
    printf 'MAKE_JOBS=%q\n' "$MAKE_JOBS"
    printf 'ARCH=%q\n\n' "$ARCH"
    cat <<'EOF'
unset PYTHONPATH
unset PYTHONHOME
unset CONDA_PREFIX
unset VIRTUAL_ENV

echo "[container] Python: $(python3 --version)"
echo "[container] Probing DALI installation..."

DALI_PY_ROOT=$(python3 - <<'PYEOF'
import pathlib
import nvidia.dali
print(pathlib.Path(nvidia.dali.__file__).resolve().parent)
PYEOF
)

if [ ! -f "$DALI_PY_ROOT/include/dali/c_api.h" ]; then
    echo "ERROR: DALI c_api.h not found under $DALI_PY_ROOT"
    exit 1
fi
if [ ! -f "$DALI_PY_ROOT/libdali.so" ]; then
    echo "ERROR: libdali.so not found under $DALI_PY_ROOT"
    exit 1
fi

mkdir -p "$DALI_PREFIX/include" "$DALI_PREFIX/lib"
ln -sfn "$DALI_PY_ROOT/include/dali" "$DALI_PREFIX/include/dali"
for library in libdali.so libdali_core.so libdali_imgcodec.so libdali_kernels.so libdali_operators.so; do
    if [ -f "$DALI_PY_ROOT/$library" ]; then
        ln -sfn "$DALI_PY_ROOT/$library" "$DALI_PREFIX/lib/$library"
    fi
done

if [ -d "$DALI_PY_ROOT/.libs" ]; then
    mkdir -p "$DALI_PREFIX/lib/.libs"
    for library in "$DALI_PY_ROOT/.libs"/*.so*; do
        if [ -e "$library" ]; then
            ln -sfn "$library" "$DALI_PREFIX/lib/.libs/$(basename "$library")"
        fi
    done
fi

export LD_LIBRARY_PATH="$DALI_PREFIX/lib:$DALI_PREFIX/lib/.libs:$NCCL_ROOT/lib:${LD_LIBRARY_PATH:-}"
export RN50_DALI_ASSET_DIR="$PIPELINE_DIR"
export RN50_DALI_OPERATOR_LIB="$DALI_PREFIX/lib/libdali_operators.so"

# NCCL performance tuning for 2-GPU NVLink topology
export NCCL_ALGO=Tree
export NCCL_P2P_LEVEL=NVL
export NCCL_DEBUG=INFO

echo "[container] Generating per-rank ImageNet file lists..."
python3 "$PROJECT_DIR/tools/generate_imagenet_dali_filelists.py" \
    --data-dir "$IMAGENET_DATA" \
    --output-dir "$FILELIST_DIR" \
    --num-shards "$NUM_GPUS"

echo "[container] Serializing DALI pipelines..."
python3 "$PROJECT_DIR/tools/serialize_imagenet_dali.py" \
    --filelist-dir "$FILELIST_DIR" \
    --output-dir "$PIPELINE_DIR" \
    --batch-size "$LOCAL_BATCH" \
    --num-threads "$DALI_THREADS" \
    --decoder "$DALI_DECODER"

echo "[container] Building train_multigpu with USE_DALI=ON..."
cd "$BUILD_DIR"
cmake "$PROJECT_DIR" \
    -DUSE_NCCL=ON \
    -DUSE_DALI=ON \
    -DNCCL_ROOT="$NCCL_ROOT" \
    -DDALI_ROOT="$DALI_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,$DALI_PREFIX/lib -Wl,-rpath,$DALI_PREFIX/lib/.libs -Wl,-rpath-link,$DALI_PREFIX/lib -Wl,-rpath-link,$DALI_PREFIX/lib/.libs" \
    ${CUDNN_PATH:+-DCUDNN_PATH="$CUDNN_PATH"}

make -j"$MAKE_JOBS" train_multigpu

echo "[container] Starting training at $(date)..."
stdbuf -oL "$BUILD_DIR/train_multigpu" "$IMAGENET_DATA" "$GLOBAL_BATCH" "$EPOCHS" "$LEARNING_RATE"
EOF
} > "$CONTAINER_SCRIPT"

chmod +x "$CONTAINER_SCRIPT"

apptainer exec --cleanenv --nv "${BIND_ARGS[@]}" "$CONTAINER_IMAGE" "$CONTAINER_SCRIPT"
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
    echo "DALI assets: $PIPELINE_DIR"
else
    echo "Training FAILED with exit code: $TRAINING_STATUS"
    exit $TRAINING_STATUS
fi
echo "Job ID: $SLURM_JOB_ID"
echo "=== Job Complete ==="