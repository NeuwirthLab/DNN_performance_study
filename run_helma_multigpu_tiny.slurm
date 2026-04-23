#!/bin/bash -l
#SBATCH --job-name=rn50_multigpu_tmpdir
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:2
#SBATCH --time=0:20:00
#SBATCH --output=tiny_helma_%j.out
#SBATCH --error=tiny_helma_%j.err
#SBATCH --export=NONE


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

#module load cuda/12.6.1
module load cuda/12.6.2
module load cudnn/9.8.0.87-12
#module load cudnn/9.2.0.82-12.x  
module load cmake
module load nvhpc
module load user-spack/0.23.0 2>/dev/null || true
#module load gcc/12.1.0
echo "CUDA: $(nvcc --version | head -1)"
echo ""

# ============================================================================
# 2. LOCATE & VERIFY NCCL
# ============================================================================
echo "Locating NCCL..."

NCCL_ROOT=""
# Dynamically search standard paths
for candidate in "$EBROOTNCCL" "$NCCL_HOME" "$CUDA_HOME" "/usr/local/cuda" "/opt/cuda"; do
    if [ -n "$candidate" ] && [ -f "$candidate/include/nccl.h" ]; then
        NCCL_ROOT="$candidate"
        break
    fi
done

# If not found but NVHPC is loaded, dynamically find the latest NCCL version inside it
if [ -z "$NCCL_ROOT" ] && [ -n "$NVHPC_ROOT" ]; then
    echo "Searching NVHPC explicitly for NCCL (this takes a moment)..."
    # Find nccl.h, sort reverse to grab the newest version, and extract the root directory
    NCCL_H_PATH=$(find "$NVHPC_ROOT" -type f -name "nccl.h" 2>/dev/null | sort -r | head -n 1)
    if [ -n "$NCCL_H_PATH" ]; then
        NCCL_ROOT=$(dirname $(dirname "$NCCL_H_PATH"))
        echo "Auto-detected NVHPC bundled NCCL at: $NCCL_ROOT"
    fi
fi

if [ -z "$NCCL_ROOT" ]; then
    echo "ERROR: NCCL not found on this cluster!"
    echo "Please ensure NCCL is loaded (e.g., 'module load nccl') or set \$NCCL_HOME manually."
    exit 1
fi

export LD_LIBRARY_PATH="$NCCL_ROOT/lib:$LD_LIBRARY_PATH"

echo "NCCL root    : $NCCL_ROOT"
echo "NCCL include : $NCCL_ROOT/include"
echo "NCCL lib     : $NCCL_ROOT/lib"
echo ""

# ============================================================================
# 3. NAVIGATE TO PROJECT
# ============================================================================
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
echo "Project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

# ============================================================================
# 4. STAGE TINY IMAGENET IN TMPDIR
# ============================================================================
echo "=== Data Staging Phase ==="

TINY_ZIP_PATH="$PROJECT_DIR/tiny-imagenet-200.zip"
if [ ! -f "$TINY_ZIP_PATH" ] && [ -f "$PROJECT_DIR/tiny-image200.zip" ]; then
    TINY_ZIP_PATH="$PROJECT_DIR/tiny-image200.zip"
fi
TMPDIR_DATA_ROOT="$TMPDIR/tiny-imagenet"
TINY_DATA_DIR="$TMPDIR_DATA_ROOT/tiny-imagenet-200"

if [ ! -f "$TINY_ZIP_PATH" ]; then
    echo "ERROR: Tiny ImageNet ZIP not found at: $TINY_ZIP_PATH"
    echo "Expected file: tiny-imagenet-200.zip"
    exit 1
fi

mkdir -p "$TMPDIR_DATA_ROOT"

if [ -d "$TINY_DATA_DIR/train" ] && [ -d "$TINY_DATA_DIR/val" ]; then
    echo "Tiny ImageNet already extracted in TMPDIR, skipping extraction."
else
    echo "Extracting Tiny ImageNet ZIP into TMPDIR..."
    unzip -q "$TINY_ZIP_PATH" -d "$TMPDIR_DATA_ROOT"
fi

if [ ! -d "$TINY_DATA_DIR/train" ] || [ ! -d "$TINY_DATA_DIR/val" ]; then
    echo "ERROR: extracted dataset is incomplete under: $TINY_DATA_DIR"
    exit 1
fi

TRAIN_COUNT=$(find "$TINY_DATA_DIR/train" -name '*.JPEG' 2>/dev/null | wc -l)
VAL_COUNT=$(find "$TINY_DATA_DIR/val" -name '*.JPEG' 2>/dev/null | wc -l)

if [ "$TRAIN_COUNT" -le 0 ] || [ "$VAL_COUNT" -le 0 ]; then
    echo "ERROR: dataset validation failed (train=$TRAIN_COUNT, val=$VAL_COUNT)"
    exit 1
fi

echo "Tiny ImageNet directory: $TINY_DATA_DIR"
echo "  - Train images: $TRAIN_COUNT"
echo "  - Val images:   $VAL_COUNT"
echo ""

# ============================================================================
# 5. BUILD RESNET WITH NCCL
# ============================================================================
echo "=== Building Phase ==="

# Auto-detect GPU architecture (H100 vs A100 vs V100)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "Detected GPU: $GPU_NAME"
if   echo "$GPU_NAME" | grep -qi "H100"; then ARCH=90
elif echo "$GPU_NAME" | grep -qi "A100"; then ARCH=80
elif echo "$GPU_NAME" | grep -qi "V100"; then ARCH=70
else ARCH=80; fi
echo "Using CUDA architecture: sm_${ARCH}"

if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    NUM_GPUS=2
fi
if [ "$NUM_GPUS" -le 0 ]; then
    echo "ERROR: invalid GPU count: $NUM_GPUS"
    exit 1
fi

GLOBAL_BATCH="${TINY_GLOBAL_BATCH:-64}"
EPOCHS="${TINY_EPOCHS:-2}"
LEARNING_RATE="${TINY_LR:-0.05}"
USE_DALI_TINY="${TINY_USE_DALI:-1}"
MAKE_JOBS="${MAKE_JOBS:-16}"
MAX_LOCAL_BATCH="${TINY_MAX_LOCAL_BATCH:-64}"

if [ $((GLOBAL_BATCH % NUM_GPUS)) -ne 0 ]; then
    echo "ERROR: TINY_GLOBAL_BATCH=$GLOBAL_BATCH must be divisible by NUM_GPUS=$NUM_GPUS"
    exit 1
fi

LOCAL_BATCH=$((GLOBAL_BATCH / NUM_GPUS))

# Keep local batch in a safe range by default to avoid launch failures.
if [ "$LOCAL_BATCH" -gt "$MAX_LOCAL_BATCH" ]; then
    echo "WARNING: local batch $LOCAL_BATCH exceeds TINY_MAX_LOCAL_BATCH=$MAX_LOCAL_BATCH"
    GLOBAL_BATCH=$((MAX_LOCAL_BATCH * NUM_GPUS))
    LOCAL_BATCH=$((GLOBAL_BATCH / NUM_GPUS))
    echo "Adjusted global batch to $GLOBAL_BATCH (local batch $LOCAL_BATCH)"
fi

RN50_PROGRESS_INTERVAL="${RN50_PROGRESS_INTERVAL:-10}"
export RN50_PROGRESS_INTERVAL

# Auto-detect cuDNN
CUDNN_PATH=""
for search_dir in "$EBROOTCUDNN" "$CUDNN_ROOT" "$CUDA_HOME" /usr/local/cuda /usr/local /usr; do
    if [ -n "$search_dir" ] && [ -f "$search_dir/include/cudnn.h" ]; then
        CUDNN_PATH="$search_dir"
        break
    fi
done

# ============================================================================
# 6. BUILD + RUN (OPTIONAL DALI VIA CONTAINER)
# ============================================================================
echo "=== Training Configuration ==="
echo "  - GPUs: $NUM_GPUS"
echo "  - Global batch size: $GLOBAL_BATCH"
echo "  - Local batch size:  $LOCAL_BATCH"
echo "  - Epochs: $EPOCHS"
echo "  - Learning rate: $LEARNING_RATE"
echo "  - Data location: $TINY_DATA_DIR"
echo "  - DALI mode: $USE_DALI_TINY"
echo ""

TMP_BASE="${TMPDIR:-/tmp}/tiny_helma_${SLURM_JOB_ID:-manual}"
BUILD_DIR="$TMP_BASE/build"
CONTAINER_IMAGE="$PROJECT_DIR/containers/nvidia-pytorch.sif"

if [ "$USE_DALI_TINY" = "1" ] && [ -f "$CONTAINER_IMAGE" ]; then
    echo "=== DALI Container Phase ==="
    ASSET_ROOT="$TMP_BASE/dali_assets"
    FILELIST_DIR="$ASSET_ROOT/filelists"
    PIPELINE_DIR="$ASSET_ROOT/pipelines"
    DALI_PREFIX="$TMP_BASE/dali_native"
    CONTAINER_SCRIPT="$TMP_BASE/run_tiny_inside_container.sh"

    rm -rf "$ASSET_ROOT" "$BUILD_DIR" "$DALI_PREFIX"
    mkdir -p "$FILELIST_DIR" "$PIPELINE_DIR" "$BUILD_DIR"

    {
        printf '#!/bin/bash\n'
        printf 'set -euo pipefail\n\n'
        printf 'PROJECT_DIR=%q\n'    "$PROJECT_DIR"
        printf 'TINY_DATA_DIR=%q\n'  "$TINY_DATA_DIR"
        printf 'FILELIST_DIR=%q\n'   "$FILELIST_DIR"
        printf 'PIPELINE_DIR=%q\n'   "$PIPELINE_DIR"
        printf 'BUILD_DIR=%q\n'      "$BUILD_DIR"
        printf 'DALI_PREFIX=%q\n'    "$DALI_PREFIX"
        printf 'NCCL_ROOT=%q\n'      "$NCCL_ROOT"
        printf 'CUDNN_PATH=%q\n'     "$CUDNN_PATH"
        printf 'NUM_GPUS=%q\n'       "$NUM_GPUS"
        printf 'GLOBAL_BATCH=%q\n'   "$GLOBAL_BATCH"
        printf 'LOCAL_BATCH=%q\n'    "$LOCAL_BATCH"
        printf 'EPOCHS=%q\n'         "$EPOCHS"
        printf 'LEARNING_RATE=%q\n'  "$LEARNING_RATE"
        printf 'ARCH=%q\n'           "$ARCH"
        printf 'MAKE_JOBS=%q\n\n'   "$MAKE_JOBS"
        cat <<'EOF'
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV

echo "[container] Python: $(python3 --version)"
echo "[container] Probing DALI installation..."

DALI_PY_ROOT=$(python3 - <<'PYEOF'
import pathlib, nvidia.dali
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
for lib in libdali.so libdali_core.so libdali_imgcodec.so libdali_kernels.so libdali_operators.so; do
    [ -f "$DALI_PY_ROOT/$lib" ] && ln -sfn "$DALI_PY_ROOT/$lib" "$DALI_PREFIX/lib/$lib"
done
if [ -d "$DALI_PY_ROOT/.libs" ]; then
    mkdir -p "$DALI_PREFIX/lib/.libs"
    for lib in "$DALI_PY_ROOT/.libs"/*.so*; do
        [ -e "$lib" ] && ln -sfn "$lib" "$DALI_PREFIX/lib/.libs/$(basename "$lib")"
    done
fi

export LD_LIBRARY_PATH="$DALI_PREFIX/lib:$DALI_PREFIX/lib/.libs:$NCCL_ROOT/lib:${LD_LIBRARY_PATH:-}"
export RN50_DALI_ASSET_DIR="$PIPELINE_DIR"

echo "[container] Generating per-rank tiny file lists..."
python3 "$PROJECT_DIR/tools/generate_tiny_imagenet_dali_filelists.py" \
    --data-dir "$TINY_DATA_DIR" \
    --output-dir "$FILELIST_DIR" \
    --num-shards "$NUM_GPUS"

echo "[container] Serializing tiny DALI pipelines..."
python3 "$PROJECT_DIR/tools/serialize_tiny_imagenet_dali.py" \
    --filelist-dir "$FILELIST_DIR" \
    --output-dir "$PIPELINE_DIR" \
    --batch-size "$LOCAL_BATCH" \
    --num-threads 8 \
    --decoder mixed

echo "[container] Building train_multigpu_tiny with USE_DALI=ON..."
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

make -j"$MAKE_JOBS" train_multigpu_tiny

echo "[container] Starting training at $(date)..."
"$BUILD_DIR/train_multigpu_tiny" "$TINY_DATA_DIR" "$GLOBAL_BATCH" "$EPOCHS" "$LEARNING_RATE"
EOF
    } > "$CONTAINER_SCRIPT"

    chmod +x "$CONTAINER_SCRIPT"

    BIND_ARGS=(
        --bind "$PROJECT_DIR:$PROJECT_DIR"
        --bind "$TMP_BASE:$TMP_BASE"
        --bind "$NCCL_ROOT:$NCCL_ROOT"
    )
    if [ -n "$CUDNN_PATH" ] && [ -d "$CUDNN_PATH" ]; then
        BIND_ARGS+=(--bind "$CUDNN_PATH:$CUDNN_PATH")
    fi

    apptainer exec --cleanenv --nv "${BIND_ARGS[@]}" "$CONTAINER_IMAGE" "$CONTAINER_SCRIPT"
    TRAINING_STATUS=$?
else
    if [ "$USE_DALI_TINY" = "1" ]; then
        echo "WARNING: TINY_USE_DALI=1 but container not found at $CONTAINER_IMAGE"
        echo "         Falling back to bare-metal non-DALI build"
    fi

    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    echo "Running CMake (bare metal)..."
    cmake "$PROJECT_DIR" \
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
    make -j"$MAKE_JOBS" train_multigpu_tiny

    if [ $? -ne 0 ]; then
        echo "ERROR: Build failed"
        exit 1
    fi

    echo "Starting training at $(date)..."
    "$BUILD_DIR/train_multigpu_tiny" "$TINY_DATA_DIR" "$GLOBAL_BATCH" "$EPOCHS" "$LEARNING_RATE"
    TRAINING_STATUS=$?
fi

echo ""
echo "Training completed at $(date)"
echo "Training exit status: $TRAINING_STATUS"
echo ""

# ============================================================================
# 7. CLEANUP & SUMMARY
# ============================================================================
echo "=== Job Summary ==="

if [ $TRAINING_STATUS -eq 0 ]; then
    echo " Training completed successfully"
else
    echo " Training failed with exit code: $TRAINING_STATUS"
    exit $TRAINING_STATUS
fi

echo "Checkpoints saved to: $PROJECT_DIR/checkpoints/"
echo "Job ID: $SLURM_JOB_ID"
echo "Total runtime: $(($(date +%s) - $(stat -c %Y "$PROJECT_DIR/build" 2>/dev/null || echo 0))) seconds"
echo ""
echo "=== Job Complete ==="
