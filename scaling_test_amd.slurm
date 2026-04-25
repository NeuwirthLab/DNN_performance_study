#!/bin/bash -l
#SBATCH --job-name=scaling_test_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --account=l0003099
#SBATCH --partition=gpu:mi300x:2
#SBATCH --gres=gpu:mi300x:2
#SBATCH --time=02:00:00
#SBATCH --output=scaling_results_%j.out

echo "========================================================"
echo " ResNet-50 AMD/HIP Scaling Test"
echo " Job   : $SLURM_JOB_ID"
echo " Node  : $SLURM_NODELIST"
echo "========================================================"

WORKSPACE_BASE="${WORKSPACE_BASE:-$HPC_SCRATCH}"
PROJECT_DIR="${PROJECT_DIR:-$WORKSPACE_BASE/resent50/performance-analysis-dl-models-resnet50}"
TINY_DIR="$PROJECT_DIR/tiny-imagenet-200"
CSV_FILE="$PROJECT_DIR/scaling_metrics.csv"

# ── Environment ─────────────────────────────────────────────────────────────
module purge 2>/dev/null || true
module load cmake/3.30.5-dy3s 2>/dev/null || true

export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$LD_LIBRARY_PATH"
export COLUMNS=200   # prevent table truncation in logs

# ── Build ───────────────────────────────────────────────────────────────────
BUILD_DIR="$TMPDIR/scaling_build_hip"
rm -rf "$BUILD_DIR" && mkdir -p "$BUILD_DIR"
echo "Building train_multigpu_tiny ..."
cmake -S "$PROJECT_DIR/hip" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DUSE_NCCL=ON > /dev/null 2>&1
cmake --build "$BUILD_DIR" -j8 --target train_multigpu_tiny > /dev/null 2>&1

if [ ! -f "$BUILD_DIR/train_multigpu_tiny" ]; then
    echo "FAIL: train_multigpu_tiny binary not found after build"
    exit 1
fi
echo "Build OK: $BUILD_DIR/train_multigpu_tiny"

EPOCHS=5  # Short run for measuring throughput
LR=0.1

# ── GPU Detection ───────────────────────────────────────────────────────────
# rocm-smi outputs "Card Series:" with capital S
ALLOCATED_GPUS=$(rocm-smi --showproductname 2>/dev/null | grep -ci "Card Series" || true)
if [ -z "$ALLOCATED_GPUS" ] || [ "$ALLOCATED_GPUS" -eq 0 ]; then
    # Fallback: parse --gres count from SLURM
    ALLOCATED_GPUS=$(echo "$SLURM_JOB_GPUS" | tr ',' '\n' | wc -l 2>/dev/null || true)
    if [ -z "$ALLOCATED_GPUS" ] || [ "$ALLOCATED_GPUS" -eq 0 ]; then
        ALLOCATED_GPUS=2
    fi
fi

echo "Detected GPUs: $ALLOCATED_GPUS"
echo ""

echo "GPUs,TestType,GlobalBatch,LocalBatch,ImgPerSec" > "$CSV_FILE"

# ── STRONG SCALING ─────────────────────────────────────────────────────────
# Fixed Global Batch = 256. As GPUs increase, Local Batch decreases.
STRONG_GLOBAL_BATCH=256
echo "=== STRONG SCALING (Fixed Global Batch = $STRONG_GLOBAL_BATCH) ==="

for (( i=1; i<=ALLOCATED_GPUS; i*=2 )); do
    DEVS=$(seq -s, 0 $((i-1)))
    LOCAL_BATCH=$((STRONG_GLOBAL_BATCH / i))
    
    echo "Running $i GPU(s)... (Local Batch = $LOCAL_BATCH)"
    
    OUTPUT=$(HIP_VISIBLE_DEVICES=$DEVS "$BUILD_DIR/train_multigpu_tiny" "$TINY_DIR" "$STRONG_GLOBAL_BATCH" "$EPOCHS" "$LR" 2>&1)
    # Grab the last epoch's Img/sec (column 8 in the table, last data line before "complete")
    IMG_SEC=$(echo "$OUTPUT" | grep -E "^\s*[0-9]+" | tail -1 | awk '{print $8}')
    
    echo "$i,Strong,$STRONG_GLOBAL_BATCH,$LOCAL_BATCH,$IMG_SEC" >> "$CSV_FILE"
    echo "  -> Throughput: $IMG_SEC Img/sec"
    echo ""
done

# ── WEAK SCALING ───────────────────────────────────────────────────────────
# Fixed Local Batch = 64. As GPUs increase, Global Batch increases.
WEAK_LOCAL_BATCH=64
echo "=== WEAK SCALING (Fixed Local Batch = $WEAK_LOCAL_BATCH) ==="

for (( i=1; i<=ALLOCATED_GPUS; i*=2 )); do
    DEVS=$(seq -s, 0 $((i-1)))
    WEAK_GLOBAL_BATCH=$((WEAK_LOCAL_BATCH * i))
    
    echo "Running $i GPU(s)... (Global Batch = $WEAK_GLOBAL_BATCH)"
    
    OUTPUT=$(HIP_VISIBLE_DEVICES=$DEVS "$BUILD_DIR/train_multigpu_tiny" "$TINY_DIR" "$WEAK_GLOBAL_BATCH" "$EPOCHS" "$LR" 2>&1)
    IMG_SEC=$(echo "$OUTPUT" | grep -E "^\s*[0-9]+" | tail -1 | awk '{print $8}')
    
    echo "$i,Weak,$WEAK_GLOBAL_BATCH,$WEAK_LOCAL_BATCH,$IMG_SEC" >> "$CSV_FILE"
    echo "  -> Throughput: $IMG_SEC Img/sec"
    echo ""
done

echo "========================================================"
echo " Scaling tests complete!"
echo " CSV: $CSV_FILE"
echo "========================================================"
cat "$CSV_FILE"
