#!/usr/bin/env bash
# port_to_hip.sh — run hipify-perl on the CUDA sources in-place.
#
# Idempotent: safe to rerun. Skips files that already contain <hip/hip_runtime.h>.
# Run from the repo root on an AMD/ROCm machine where hipify-perl is installed.
set -euo pipefail

if ! command -v hipify-perl >/dev/null 2>&1; then
    echo "error: hipify-perl not on PATH. Load the rocm module (or add /opt/rocm/bin)." >&2
    exit 1
fi

FILES=(
    resnet50.cu resnet50.cuh
    main.cu train.cu train_tinyimagenet.cu train_imagenet.cu
    train_multigpu.cu train_multigpu_tiny.cu
    benchmark.cu train_bench.cu
    tiny_dataloader.h imagenet_loader.h prefetch_loader.h mt_image_loader.h
)

converted=0
skipped=0
for f in "${FILES[@]}"; do
    [ -f "$f" ] || { echo "  skip (missing): $f"; continue; }
    if grep -q "hip/hip_runtime.h\|USE_HIP" "$f"; then
        echo "  skip (already ported): $f"
        skipped=$((skipped+1))
        continue
    fi
    echo "  hipify: $f"
    cp "$f" "$f.cuda.bak"
    hipify-perl -inplace "$f" >/dev/null
    converted=$((converted+1))
done

echo ""
echo "Summary: converted=$converted  skipped=$skipped"
echo "Originals saved as *.cuda.bak (tracked in .gitignore — remove once confident)."
