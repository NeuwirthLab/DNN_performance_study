#!/bin/bash -l
#SBATCH --job-name=dali_imagenet_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=1:00:00
#SBATCH --output=dali_imagenet_smoke_%j.out
#SBATCH --error=dali_imagenet_smoke_%j.err
#SBATCH --export=ALL

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
set -euo pipefail

echo "=== Standalone Native DALI C API Smoke Test ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "TMPDIR: $TMPDIR"
echo "TMPDIR Space: $(df -h "$TMPDIR" | tail -1)"
echo ""

module purge
module load cuda/12.6.2
module load cudnn/9.8.0.87-12
module load cmake
module load nvhpc
module load python

PROJECT_DIR="/hnvme/workspace/m102cd10-amrit/resnet50"
CONTAINER_IMAGE="$PROJECT_DIR/containers/nvidia-pytorch.sif"
IMAGENET_ZIP="/hnvme/workspace/m102cd10-amrit/imagenet/imagenet-object-localization-challenge.zip"
TMPDIR_DATA="$TMPDIR/imagenet_data"
IMAGENET_DATA_OVERRIDE="${DALI_SMOKE_DATA_DIR:-}"
IMAGENET_DATA="$TMPDIR_DATA/ILSVRC/Data/CLS-LOC"
ASSET_ROOT="$TMPDIR/dali_smoke_assets"
FILELIST_DIR="$ASSET_ROOT/filelists"
PIPELINE_DIR="$ASSET_ROOT/pipelines"
BUILD_DIR="$TMPDIR/dali_smoke_build"
DALI_PREFIX="$TMPDIR/dali_native"

LOCAL_BATCH=${DALI_SMOKE_BATCH:-128}
DALI_THREADS=${DALI_SMOKE_THREADS:-8}
MAX_BATCHES=${DALI_SMOKE_MAX_BATCHES:-32}
DALI_DECODER=${DALI_SMOKE_DECODER:-mixed}
MAX_IMAGES=${DALI_SMOKE_MAX_IMAGES:-0}
MAX_CLASSES=${DALI_SMOKE_MAX_CLASSES:-0}

if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory not found: $PROJECT_DIR"
    exit 1
fi
if [ ! -f "$CONTAINER_IMAGE" ]; then
    echo "ERROR: Container image not found: $CONTAINER_IMAGE"
    exit 1
fi
if [ -z "$IMAGENET_DATA_OVERRIDE" ] && [ ! -f "$IMAGENET_ZIP" ]; then
    echo "ERROR: ImageNet ZIP not found: $IMAGENET_ZIP"
    exit 1
fi

if [ "$MAX_IMAGES" -gt 0 ] && [ "$MAX_IMAGES" -lt "$LOCAL_BATCH" ]; then
    echo "ERROR: DALI_SMOKE_MAX_IMAGES ($MAX_IMAGES) must be >= DALI_SMOKE_BATCH ($LOCAL_BATCH)"
    exit 1
fi

mkdir -p "$TMPDIR_DATA" "$FILELIST_DIR" "$PIPELINE_DIR"

if [ -n "$IMAGENET_DATA_OVERRIDE" ]; then
    IMAGENET_DATA="$IMAGENET_DATA_OVERRIDE"
    echo "Using pre-extracted dataset override: $IMAGENET_DATA"
else
    if [ ! -d "$IMAGENET_DATA/train" ]; then
        if [ "$MAX_IMAGES" -gt 0 ] || [ "$MAX_CLASSES" -gt 0 ]; then
            echo "Extracting ImageNet subset to TMPDIR..."
            python3 - <<PY_SUBSET
from pathlib import Path
from zipfile import ZipFile

zip_path = Path(r"$IMAGENET_ZIP")
output_root = Path(r"$TMPDIR_DATA")
max_images = int(r"$MAX_IMAGES")
max_classes = int(r"$MAX_CLASSES")

with ZipFile(zip_path) as archive:
    members = sorted(
        name for name in archive.namelist()
        if name.startswith("ILSVRC/Data/CLS-LOC/train/") and name.lower().endswith(".jpeg")
    )

    selected = []
    chosen_classes = []
    chosen_set = set()

    for name in members:
        parts = name.split("/")
        class_name = parts[4]
        if class_name not in chosen_set:
            if max_classes > 0 and len(chosen_classes) >= max_classes:
                continue
            chosen_classes.append(class_name)
            chosen_set.add(class_name)

        selected.append(name)
        if max_images > 0 and len(selected) >= max_images:
            break

    for member in selected:
        archive.extract(member, path=output_root)

print(f"subset_classes={len(chosen_classes)}")
print(f"subset_images={len(selected)}")
PY_SUBSET
        else
            echo "Extracting full ImageNet ZIP to TMPDIR..."
            unzip -q "$IMAGENET_ZIP" -d "$TMPDIR_DATA"
        fi
    fi
fi

if [ ! -d "$IMAGENET_DATA/train" ]; then
    echo "ERROR: Train directory missing after extraction: $IMAGENET_DATA/train"
    exit 1
fi

echo "=== Container DALI Import Test ==="
apptainer exec --cleanenv --nv \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --bind "$TMPDIR:$TMPDIR" \
    "$CONTAINER_IMAGE" python3 - <<PY_IMPORT
import pathlib
import sys

import nvidia.dali
import nvidia.dali.backend_impl

print(f"sys.executable = {sys.executable}")
print(f"nvidia.dali = {pathlib.Path(nvidia.dali.__file__).resolve()}")
print("backend ok")
PY_IMPORT

echo "=== Generate Train File List ==="
apptainer exec --cleanenv --nv \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --bind "$TMPDIR:$TMPDIR" \
    "$CONTAINER_IMAGE" python3 - <<PY_FILELIST
from pathlib import Path

tmpdir = Path(r"$TMPDIR")
train_root = Path(r"$IMAGENET_DATA") / "train"
output_path = tmpdir / "dali_smoke_assets" / "filelists" / "train_rank0.txt"
max_images = int(r"$MAX_IMAGES")
max_classes = int(r"$MAX_CLASSES")

class_names = sorted(entry.name for entry in train_root.iterdir() if entry.is_dir())
if max_classes > 0:
    class_names = class_names[:max_classes]
records = []
for label, class_name in enumerate(class_names):
    class_dir = train_root / class_name
    for image in sorted(class_dir.iterdir()):
        if image.is_file() and image.suffix.lower() in {".jpeg", ".jpg"}:
            records.append((str(image.resolve()), label))
            if max_images > 0 and len(records) >= max_images:
                break
    if max_images > 0 and len(records) >= max_images:
        break

records.sort(key=lambda item: item[0])
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text("\n".join(f"{path} {label}" for path, label in records) + "\n")

print(f"classes={len(class_names)}")
print(f"train_records={len(records)}")
print(f"max_classes={max_classes}")
print(f"max_images={max_images}")
print(f"file_list={output_path}")
PY_FILELIST

echo "=== Serialize Train Pipeline ==="
apptainer exec --cleanenv --nv \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --bind "$TMPDIR:$TMPDIR" \
    "$CONTAINER_IMAGE" python3 - <<PY_SERIALIZE
from pathlib import Path

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

tmpdir = Path(r"$TMPDIR")
file_list = tmpdir / "dali_smoke_assets" / "filelists" / "train_rank0.txt"
output_path = tmpdir / "dali_smoke_assets" / "pipelines" / "train_rank0.dali"
batch_size = int(r"$LOCAL_BATCH")
num_threads = int(r"$DALI_THREADS")
decoder_device = r"$DALI_DECODER"
resize_device = "gpu" if decoder_device == "mixed" else "cpu"

@pipeline_def
def train_pipeline(file_list: str, decoder_device: str, resize_device: str):
    images, labels = fn.readers.file(
        file_list=file_list,
        random_shuffle=True,
        stick_to_shard=True,
        pad_last_batch=False,
        name="Reader",
    )
    images = fn.decoders.image_random_crop(
        images,
        device=decoder_device,
        output_type=types.RGB,
        random_aspect_ratio=[0.75, 1.333333],
        random_area=[0.08, 1.0],
        num_attempts=10,
    )
    images = fn.resize(images, device=resize_device, resize_x=224, resize_y=224)
    if resize_device == "cpu":
        images = images.gpu()
    mirror = fn.random.coin_flip(probability=0.5)
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=(224, 224),
        mean=[0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0],
        std=[0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0],
        mirror=mirror,
    )
    labels = fn.cast(labels, dtype=types.INT32).gpu()
    return images, labels

pipe = train_pipeline(
    batch_size=batch_size,
    num_threads=num_threads,
    device_id=0,
    seed=42,
    exec_pipelined=True,
    exec_async=True,
    prefetch_queue_depth=2,
    file_list=str(file_list),
    decoder_device=decoder_device,
    resize_device=resize_device,
)
pipe.build()
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_bytes(pipe.serialize())
print(f"pipeline={output_path}")
print(f"size_bytes={output_path.stat().st_size}")
PY_SERIALIZE

echo "=== Build Native DALI Prefix And C++ Smoke Test ==="
rm -rf "$BUILD_DIR" "$DALI_PREFIX"
mkdir -p "$BUILD_DIR" "$DALI_PREFIX/include" "$DALI_PREFIX/lib/.libs"

apptainer exec --cleanenv --nv \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --bind "$TMPDIR:$TMPDIR" \
    "$CONTAINER_IMAGE" bash -lc '
set -euo pipefail

DALI_PY_ROOT=$(python3 - <<"PY"
import pathlib
import nvidia.dali
print(pathlib.Path(nvidia.dali.__file__).resolve().parent)
PY
)

mkdir -p "'$DALI_PREFIX'/include" "'$DALI_PREFIX'/lib"
ln -sfn "$DALI_PY_ROOT/include/dali" "'$DALI_PREFIX'/include/dali"

for lib in "$DALI_PY_ROOT"/*.so*; do
    if [ -e "$lib" ]; then
        ln -sfn "$lib" "'$DALI_PREFIX'/lib/$(basename "$lib")"
    fi
done

if [ -d "$DALI_PY_ROOT/.libs" ]; then
    mkdir -p "'$DALI_PREFIX'/lib/.libs"
    for lib in "$DALI_PY_ROOT/.libs"/*.so*; do
        if [ -e "$lib" ]; then
            ln -sfn "$lib" "'$DALI_PREFIX'/lib/.libs/$(basename "$lib")"
        fi
    done
fi

export LD_LIBRARY_PATH="'$DALI_PREFIX'/lib:'$DALI_PREFIX'/lib/.libs:${LD_LIBRARY_PATH:-}"
export RN50_DALI_OPERATOR_LIB="'$DALI_PREFIX'/lib/libdali_operators.so"

cd "'$BUILD_DIR'"
cmake "'$PROJECT_DIR'" \
    -DUSE_DALI=ON \
    -DDALI_ROOT="'$DALI_PREFIX'" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,'$DALI_PREFIX'/lib -Wl,-rpath,'$DALI_PREFIX'/lib/.libs -Wl,-rpath-link,'$DALI_PREFIX'/lib -Wl,-rpath-link,'$DALI_PREFIX'/lib/.libs"

make -j8 dali_imagenet_smoke

stdbuf -oL "'$BUILD_DIR'/dali_imagenet_smoke" \
    "'$PIPELINE_DIR'/train_rank0.dali" \
    "'$FILELIST_DIR'/train_rank0.txt" \
    "'$LOCAL_BATCH'" \
    0 \
    "'$DALI_THREADS'" \
    "'$MAX_BATCHES'"
'