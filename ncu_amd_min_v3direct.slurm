#!/bin/bash -l
#SBATCH --job-name=rn50_v3direct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --account=l0003099
#SBATCH --gres=gpu:mi300x:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/ncu_v3direct-%j.out
#SBATCH --error=logs/ncu_v3direct-%j.err
#SBATCH --export=NONE

# Direct rocprofv3 (bypassing rocprof-compute) to localize the crash.
# TEST A: --kernel-trace (no counters)        -> isolates attachment vs counters
# TEST B: --pmc 2 counters                     -> narrowest counter collection

set -uo pipefail

echo "[v3direct] start $(date) node=$SLURM_NODELIST job=$SLURM_JOB_ID"

WORKSPACE_BASE="${WORKSPACE_BASE:-$HPC_SCRATCH}"
PROJECT_DIR="${PROJECT_DIR:-$WORKSPACE_BASE/resent50/performance-analysis-dl-models-resnet50}"
TINY_DATA_DIR="${TINY_DATA_DIR:-$PROJECT_DIR/tiny-imagenet-200}"

module load cmake
export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"

export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$HOME/.cache/miopen_min}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
export MIOPEN_DEBUG_CONV_WORKSPACE_LIMIT_BYTES=2147483648
export MIOPEN_FIND_MODE=NORMAL

export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_DISABLE=1
export HIP_VISIBLE_DEVICES="0"

GFX_ARCH=$(timeout 30 rocminfo 2>/dev/null \
            | awk '/gfx[0-9a-f]+/{for(i=1;i<=NF;i++) if($i ~ /^gfx/){print $i; exit}}' || true)
[ -z "$GFX_ARCH" ] && GFX_ARCH="gfx942"

echo "[v3direct] rocprofv3: $(rocprofv3 --version 2>&1 | head -1)"
echo "[v3direct] GPU arch : $GFX_ARCH"

BUILD_DIR="$TMPDIR/rn50_tiny_v3"
rm -rf "$BUILD_DIR" && mkdir -p "$BUILD_DIR"

cmake -S "$PROJECT_DIR/hip" -B "$BUILD_DIR" \
      -DCMAKE_HIP_ARCHITECTURES="$GFX_ARCH" \
      -DCMAKE_BUILD_TYPE=Release \
      -DUSE_NCCL=ON \
      > "$BUILD_DIR/cmake.log" 2>&1 \
   || { tail -40 "$BUILD_DIR/cmake.log"; exit 1; }

cmake --build "$BUILD_DIR" -j16 --target train_multigpu_tiny \
      > "$BUILD_DIR/build.log" 2>&1 \
   || { tail -60 "$BUILD_DIR/build.log"; exit 1; }
echo "[v3direct] built train_multigpu_tiny"

OUT_ROOT="$PROJECT_DIR/profiles/v3direct_${SLURM_JOB_ID}"
mkdir -p "$OUT_ROOT/A_trace_only" "$OUT_ROOT/B_pmc_small"

echo ""
echo "================== TEST A: --kernel-trace (no counters) =================="
rocprofv3 \
    --kernel-trace \
    --output-directory "$OUT_ROOT/A_trace_only" \
    --output-format csv \
    --summary \
    -- stdbuf -oL "$BUILD_DIR/train_multigpu_tiny" "$TINY_DATA_DIR" 256 1 0.1
STATUS_A=$?
echo "[v3direct] TEST A exit=$STATUS_A"

PMC_FILE="$OUT_ROOT/pmc_tiny.txt"
cat > "$PMC_FILE" <<'EOF'
pmc: SQ_INSTS_VALU_ADD_F32 TCC_EA0_RDREQ_sum
EOF

echo ""
echo "================== TEST B: --pmc (2 counters) =================="
echo "  PMC set: $(cat "$PMC_FILE")"
rocprofv3 \
    -i "$PMC_FILE" \
    --output-directory "$OUT_ROOT/B_pmc_small" \
    --output-format csv \
    -- stdbuf -oL "$BUILD_DIR/train_multigpu_tiny" "$TINY_DATA_DIR" 256 1 0.1
STATUS_B=$?
echo "[v3direct] TEST B exit=$STATUS_B"

echo ""
echo "================== Verdict =================="
echo "  TEST A (trace only) : exit=$STATUS_A"
echo "  TEST B (2-counter)  : exit=$STATUS_B"
if [ "$STATUS_A" -eq 0 ] && [ "$STATUS_B" -ne 0 ]; then
    echo "  -> Counter collection is the bug; tracing works."
elif [ "$STATUS_A" -eq 0 ] && [ "$STATUS_B" -eq 0 ]; then
    echo "  -> Both pass; rocprof-compute is the wrapper bug. Use rocprofv3 directly."
elif [ "$STATUS_A" -ne 0 ]; then
    echo "  -> Even pure tracing crashes; bug is in the binary under rocprofv3."
fi
echo "  out: $OUT_ROOT"
echo "[v3direct] done $(date)"
