#!/bin/bash -l
#SBATCH --job-name=rn50_dali_nsys
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:2
#SBATCH --time=01:00:00
#SBATCH --output=rn50_dali_nsys_%j.out
#SBATCH --error=rn50_dali_nsys_%j.err
#SBATCH --export=NONE

set -euo pipefail

# -----------------------------------------------------------------------------
# ResNet-50 2xH100 DALI profiling run using Nsight Systems.
# This script mirrors the production DALI path and profiles only train_multigpu.
# -----------------------------------------------------------------------------

echo "=== ResNet-50 Multi-GPU DALI Profiling (Nsight Systems) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "TMPDIR: ${TMPDIR:-<unset>}"

# 1) Modules
# This cluster keeps gpu-env/2025 as a sticky module; plain module purge
# exits with code 1 under strict shell settings, so force it.
module --force purge

# Prefer the newer toolchain, but fall back to the versions already used in
# the repo's known-good Slurm scripts for this cluster.
module load nvhpc || true
module load cmake
module load python || true
module load user-spack/0.19.1 || true

if ! module load cuda/12.6.2; then
    module load cuda/12.6.1
fi

if ! module load cudnn/9.8.0.87-12; then
    module load cudnn/9.2.0.82-12.x
fi

echo "CUDA: $(nvcc --version | head -1)"
echo "NSYS: $(nsys --version | head -1)"

# 2) Paths and tunables
PROJECT_DIR="/hnvme/workspace/m102cd10-amrit/resnet50"
CONTAINER_IMAGE="$PROJECT_DIR/containers/nvidia-pytorch.sif"
IMAGENET_ZIP="/hnvme/workspace/m102cd10-amrit/imagenet/imagenet-object-localization-challenge.zip"

NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-2}}
GLOBAL_BATCH=${GLOBAL_BATCH:-256}
EPOCHS=${EPOCHS:-1}
LEARNING_RATE=${LEARNING_RATE:-0.1}
DALI_DECODER=${DALI_DECODER:-mixed}
DALI_THREADS=${DALI_THREADS:-8}
MAKE_JOBS=${MAKE_JOBS:-16}
RN50_PROGRESS_INTERVAL=${RN50_PROGRESS_INTERVAL:-50}

# Nsight options
NSYS_TRACE=${NSYS_TRACE:-cuda,nvtx,cublas,cudnn,osrt}
NSYS_SAMPLE=${NSYS_SAMPLE:-none}
NSYS_CAPTURE_RANGE=${NSYS_CAPTURE_RANGE:-none}

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
if [ "$NUM_GPUS" -le 0 ]; then
    echo "ERROR: invalid NUM_GPUS=$NUM_GPUS"
    exit 1
fi
if [ $((GLOBAL_BATCH % NUM_GPUS)) -ne 0 ]; then
    echo "ERROR: GLOBAL_BATCH=$GLOBAL_BATCH must be divisible by NUM_GPUS=$NUM_GPUS"
    exit 1
fi

LOCAL_BATCH=$((GLOBAL_BATCH / NUM_GPUS))

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

# 3) Locate NCCL/CUDNN roots for CMake and runtime bindings
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

CUDNN_PATH=""
for search_dir in "${EBROOTCUDNN:-}" "${CUDNN_ROOT:-}" "${CUDA_HOME:-}" /usr/local/cuda /usr/local /usr; do
    if [ -n "${search_dir:-}" ] && [ -f "$search_dir/include/cudnn.h" ]; then
        CUDNN_PATH="$search_dir"
        break
    fi
done

# 4) Stage dataset to local NVMe
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
        src = os.path.join(val_dir, image_id + ".JPEG")
        if not os.path.isfile(src):
            continue
        dst_dir = os.path.join(val_dir, wnid)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.move(src, os.path.join(dst_dir, image_id + ".JPEG"))
        moved += 1
print(f"Moved {moved} validation images into class subfolders.")
PYEOF
fi

IMAGENET_DATA="$TMPDIR_DATA/ILSVRC/Data/CLS-LOC"
if [ ! -d "$IMAGENET_DATA/train" ]; then
    echo "ERROR: train/ directory not found at $IMAGENET_DATA/train"
    exit 1
fi

echo "Dataset ready: $IMAGENET_DATA"
echo "Train images: $(find "$IMAGENET_DATA/train" -name '*.JPEG' | wc -l)"
echo "Val images:   $(find "$IMAGENET_DATA/val" -name '*.JPEG' | wc -l)"

# 5) Working directories and output artifacts
ASSET_ROOT="$TMPDIR/dali_assets"
FILELIST_DIR="$ASSET_ROOT/filelists"
PIPELINE_DIR="$ASSET_ROOT/pipelines"
BUILD_DIR="$TMPDIR/resnet50_build_dali"
DALI_PREFIX="$TMPDIR/dali_native"
INNER_PREP_SCRIPT="$TMPDIR/rn50_profile_prepare.sh"
INNER_TRAIN_SCRIPT="$TMPDIR/rn50_profile_train.sh"

PROFILE_DIR="$PROJECT_DIR/profiles/nsys_${SLURM_JOB_ID}"
mkdir -p "$PROFILE_DIR"
NSYS_BASENAME="$PROFILE_DIR/rn50_dali_multigpu"
NSYS_REP="${NSYS_BASENAME}.nsys-rep"
NSYS_SUMMARY="$PROFILE_DIR/nsys_summary.txt"

rm -rf "$ASSET_ROOT" "$BUILD_DIR" "$DALI_PREFIX"
mkdir -p "$FILELIST_DIR" "$PIPELINE_DIR" "$BUILD_DIR"

# 6) Container bindings
BIND_ARGS=(
    --bind "$PROJECT_DIR:$PROJECT_DIR"
    --bind "$TMPDIR:$TMPDIR"
    --bind "$NCCL_ROOT:$NCCL_ROOT"
)
if [ -n "$CUDNN_PATH" ] && [ -d "$CUDNN_PATH" ]; then
    BIND_ARGS+=(--bind "$CUDNN_PATH:$CUDNN_PATH")
fi

# 7) Container prep/build script
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
    printf 'LOCAL_BATCH=%q\n' "$LOCAL_BATCH"
    printf 'DALI_THREADS=%q\n' "$DALI_THREADS"
    printf 'DALI_DECODER=%q\n' "$DALI_DECODER"
    printf 'MAKE_JOBS=%q\n' "$MAKE_JOBS"
    printf 'ARCH=%q\n\n' "$ARCH"
    cat <<'EOF'
unset PYTHONPATH
unset PYTHONHOME
unset CONDA_PREFIX
unset VIRTUAL_ENV

echo "[container/prepare] Python: $(python3 --version)"

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
for lib in libdali.so libdali_core.so libdali_imgcodec.so libdali_kernels.so libdali_operators.so; do
    if [ -f "$DALI_PY_ROOT/$lib" ]; then
        ln -sfn "$DALI_PY_ROOT/$lib" "$DALI_PREFIX/lib/$lib"
    fi
done

if [ -d "$DALI_PY_ROOT/.libs" ]; then
    mkdir -p "$DALI_PREFIX/lib/.libs"
    for lib in "$DALI_PY_ROOT/.libs"/*.so*; do
        if [ -e "$lib" ]; then
            ln -sfn "$lib" "$DALI_PREFIX/lib/.libs/$(basename "$lib")"
        fi
    done
fi

export LD_LIBRARY_PATH="$DALI_PREFIX/lib:$DALI_PREFIX/lib/.libs:$NCCL_ROOT/lib:${LD_LIBRARY_PATH:-}"
export RN50_DALI_ASSET_DIR="$PIPELINE_DIR"
export RN50_DALI_OPERATOR_LIB="$DALI_PREFIX/lib/libdali_operators.so"

# NCCL tuning
export NCCL_ALGO=Tree
export NCCL_P2P_LEVEL=NVL
export NCCL_DEBUG=INFO

echo "[container/prepare] Generating file lists..."
python3 "$PROJECT_DIR/tools/generate_imagenet_dali_filelists.py" \
    --data-dir "$IMAGENET_DATA" \
    --output-dir "$FILELIST_DIR" \
    --num-shards "$NUM_GPUS"

echo "[container/prepare] Serializing pipelines..."
python3 "$PROJECT_DIR/tools/serialize_imagenet_dali.py" \
    --filelist-dir "$FILELIST_DIR" \
    --output-dir "$PIPELINE_DIR" \
    --batch-size "$LOCAL_BATCH" \
    --num-threads "$DALI_THREADS" \
    --decoder "$DALI_DECODER"

echo "[container/prepare] Building train_multigpu..."
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

echo "[container/prepare] Build complete."
EOF
} > "$INNER_PREP_SCRIPT"
chmod +x "$INNER_PREP_SCRIPT"

# 8) Container train script (profile target)
{
    printf '#!/bin/bash\n'
    printf 'set -euo pipefail\n\n'
    printf 'BUILD_DIR=%q\n' "$BUILD_DIR"
    printf 'IMAGENET_DATA=%q\n' "$IMAGENET_DATA"
    printf 'GLOBAL_BATCH=%q\n' "$GLOBAL_BATCH"
    printf 'EPOCHS=%q\n' "$EPOCHS"
    printf 'LEARNING_RATE=%q\n' "$LEARNING_RATE"
    printf 'PIPELINE_DIR=%q\n' "$PIPELINE_DIR"
    printf 'DALI_PREFIX=%q\n' "$DALI_PREFIX"
    printf 'NCCL_ROOT=%q\n' "$NCCL_ROOT"
    printf 'RN50_PROGRESS_INTERVAL=%q\n\n' "$RN50_PROGRESS_INTERVAL"
    cat <<'EOF'
unset PYTHONPATH
unset PYTHONHOME
unset CONDA_PREFIX
unset VIRTUAL_ENV

export LD_LIBRARY_PATH="$DALI_PREFIX/lib:$DALI_PREFIX/lib/.libs:$NCCL_ROOT/lib:${LD_LIBRARY_PATH:-}"
export RN50_DALI_ASSET_DIR="$PIPELINE_DIR"
export RN50_DALI_OPERATOR_LIB="$DALI_PREFIX/lib/libdali_operators.so"
export RN50_PROGRESS_INTERVAL="$RN50_PROGRESS_INTERVAL"

# Keep the same NCCL path/settings as production run
export NCCL_ALGO=Tree
export NCCL_P2P_LEVEL=NVL
export NCCL_DEBUG=INFO

echo "[container/train] Starting profiled training at $(date)"
stdbuf -oL "$BUILD_DIR/train_multigpu" "$IMAGENET_DATA" "$GLOBAL_BATCH" "$EPOCHS" "$LEARNING_RATE"
echo "[container/train] Training completed at $(date)"
EOF
} > "$INNER_TRAIN_SCRIPT"
chmod +x "$INNER_TRAIN_SCRIPT"

# 9) Build once in container
apptainer exec --cleanenv --nv "${BIND_ARGS[@]}" "$CONTAINER_IMAGE" "$INNER_PREP_SCRIPT"

# 10) Nsight Systems profile: profile only training command via apptainer
# --trace-fork-before-exec helps keep tracing through launcher boundaries.
nsys profile \
    --trace="$NSYS_TRACE" \
    --sample="$NSYS_SAMPLE" \
    --cpuctxsw=none \
    --capture-range="$NSYS_CAPTURE_RANGE" \
    --trace-fork-before-exec=true \
    --force-overwrite=true \
    --output="$NSYS_BASENAME" \
    apptainer exec --cleanenv --nv "${BIND_ARGS[@]}" "$CONTAINER_IMAGE" "$INNER_TRAIN_SCRIPT"
PROFILE_STATUS=$?

# 11) Post-process report into a compact text summary
if [ -f "$NSYS_REP" ]; then
    {
        echo "=== NVTX Summary ==="
        nsys stats --report nvtx_sum --format table --timeunit ms "$NSYS_REP"
        echo
        echo "=== CUDA GPU Kernel Summary ==="
        nsys stats --report cuda_gpu_kern_sum --format table --timeunit ms "$NSYS_REP"
        echo
        echo "=== CUDA API Summary ==="
        nsys stats --report cuda_api_sum --format table --timeunit ms "$NSYS_REP"
        echo
        echo "=== OS Runtime Summary ==="
        nsys stats --report osrt_sum --format table --timeunit ms "$NSYS_REP"
    } > "$NSYS_SUMMARY"
fi

echo
echo "=== Profiling Artifacts ==="
echo "Nsight report : $NSYS_REP"
echo "Text summary  : $NSYS_SUMMARY"
echo "Profile status: $PROFILE_STATUS"

if [ $PROFILE_STATUS -ne 0 ]; then
    echo "ERROR: Profiled run failed"
    exit $PROFILE_STATUS
fi

echo "Done."
