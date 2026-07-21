#!/bin/bash -l
# ============================================================================
# run_multinode_mt16_1node.slurm
#   SINGLE-NODE (8-GPU) variant of run_multinode_mt16.slurm. Same native HIP
#   ResNet-50 DDP trainer (train_multinode) and same measure_power harness, run
#   on ONE node so it schedules immediately on the idle gaod0001 instead of
#   waiting on the 2-node gaod[0001-0002] reservation (gaod0002 draining).
#   Purpose: fresh 1-node native 8-GPU energy datapoint (256/GPU) as the control
#   that separates a per-GPU coefficient gap from a 2-node comm/idle artifact.
#   Deltas vs the 2-node script: nodes=1, ntasks=8, GLOBAL_BATCH=2048 (256/GPU),
#   NCCL_IB_DISABLE=1 (intra-node XGMI only). Everything else identical; the
#   srun launch already supports "8 tasks / 1 node".
#
#   One process per GPU (srun --ntasks=8 --ntasks-per-node=8). Data loader =
#   turbojpeg CPU path (PrefetchLoader<MTImageLoader>) by default; rocAL opt-in
#   via USE_ROCAL=ON. Energy measured via measure_power_multinode.sh.
#
#   Build target: train_multinode  (USE_NCCL=ON USE_TURBOJPEG=ON, USE_ROCAL=OFF
#   default; CPU prefetch path uses prefetch_loader.h, present in repo).
#
#   Knobs (env):
#     GLOBAL_BATCH=4096  EPOCHS=1  LR=0.1
#     SUBSET_GLOB="n0144*"   -> stage only matching train classes (fast smoke)
#     USE_POWER=1            -> wrap in measure_power_multinode.sh (0 = bare srun)
#     NCCL_DEBUG=WARN        -> set INFO for the first comm/IB smoke
#   Submit:  sbatch run_multinode_imagenet_amd.slurm
#            SUBSET_GLOB='n0144*' EPOCHS=1 USE_POWER=0 NCCL_DEBUG=INFO \
#              sbatch --export=ALL,SUBSET_GLOB='n0144*',EPOCHS=1,USE_POWER=0,NCCL_DEBUG=INFO \
#              run_multinode_imagenet_amd.slurm
# ============================================================================
#SBATCH --job-name=rn50_mt16_1node
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --account=l0003099
#SBATCH --partition=acc
#SBATCH --gres=gpu:mi300x:8
#SBATCH --gpu-bind=none
#SBATCH --time=02:00:00
#SBATCH --nodelist=gaod0001
#SBATCH --output=logs/run_multinode_amd_1node_%j.out
#SBATCH --error=logs/run_multinode_amd_1node_%j.err
#SBATCH --export=NONE

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
PROJECT_DIR="$(pwd)"

echo "=== rn50 1-node native 8-GPU (RCCL/XGMI) — job ${SLURM_JOB_ID} on ${SLURM_NODELIST} ==="
echo "start $(date)"

# ── config ───────────────────────────────────────────────────────────────────
GLOBAL_BATCH="${GLOBAL_BATCH:-2048}"   # 1 node x 8 GPU x 256/GPU (matches coeff/validation batch)
EPOCHS="${EPOCHS:-1}"
LR="${LR:-0.1}"
MAKE_JOBS="${MAKE_JOBS:-16}"
USE_POWER="${USE_POWER:-1}"
SUBSET_GLOB="${SUBSET_GLOB:-}"            # empty -> full train set
ROCAL_ROOT="${ROCAL_ROOT:-$HOME/local/opt/rocm-6.4.0}"
IMAGENET_ZIP="${IMAGENET_ZIP:-$PROJECT_DIR/ImageNet-1K/imagenet-object-localization-challenge.zip}"
IN_REPO_VAL="${IN_REPO_VAL:-$PROJECT_DIR/ImageNet-1K/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val}"
export RN50_PROGRESS_INTERVAL="${RN50_PROGRESS_INTERVAL:-50}"
# Decode threads per rank (empty -> binary falls back to SLURM_CPUS_PER_TASK).
export RN50_DECODE_THREADS="${RN50_DECODE_THREADS:-}"
# Ring-buffered staging + per-step host-sync removal (set BOTH together).
export RN50_RINGBUF="${RN50_RINGBUF:-}"
export RN50_NO_STEP_SYNC="${RN50_NO_STEP_SYNC:-}"
export RN50_RING_DEPTH="${RN50_RING_DEPTH:-}"

[ -f "$ROCAL_ROOT/include/rocal/rocal_api.h" ] || { echo "FAIL: rocAL header missing: $ROCAL_ROOT/include/rocal/rocal_api.h"; exit 1; }
[ -f "$ROCAL_ROOT/lib/librocal.so" ]           || { echo "FAIL: librocal.so missing: $ROCAL_ROOT/lib"; exit 1; }
[ -f "$IMAGENET_ZIP" ]                          || { echo "FAIL: imagenet zip missing: $IMAGENET_ZIP"; exit 1; }
[ -d "$IN_REPO_VAL" ]                           || { echo "FAIL: in-repo val dir missing: $IN_REPO_VAL"; exit 1; }

# ── ROCm + rocAL + MPI runtime wiring (symlinks land in $HOME -> visible both nodes) ──
module load cmake >/dev/null 2>&1 || true
# MPI is the PROVEN 2-node RCCL bootstrap on this cluster (see rccl_bench_2node.slurm):
# the file-based ncclUniqueId handshake does NOT propagate across nodes on /work.
module load openmpi >/dev/null 2>&1 || module load openmpi/4.1.8-6xzv >/dev/null 2>&1 || true
export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
mkdir -p "$ROCAL_ROOT/lib"
TJ=$(find /shared/spapps_2026_01_avx2/gcc-11.5.0 -maxdepth 4 -name 'libturbojpeg.so.0' 2>/dev/null | head -1)
[ -n "$TJ" ] && ln -sf "$TJ" "$ROCAL_ROOT/lib/libturbojpeg.so.0"
OVITO_FFMPEG="/shared/apps/ovito/3.8.2/lib/ovito"
for so in libavcodec.so.59 libavformat.so.59 libavutil.so.57 libswscale.so.6; do
    [ -f "$OVITO_FFMPEG/$so" ] && ln -sf "$OVITO_FFMPEG/$so" "$ROCAL_ROOT/lib/$so"
done
# MPI runtime libs: local copy of libmpi.so.40 (on /work, reachable from both nodes,
# so compute nodes don't need /shared mounted) + the module's MPI_HOME/lib.
MPI_HOME="$(dirname "$(dirname "$(command -v mpirun 2>/dev/null || echo /usr/bin/x)")")"
LOCAL_MPI_LIB="$PROJECT_DIR/rccl-tests/mpi_lib"
# libjpeg-turbo (fast JPEG decode) — cmake finds it via TURBOJPEG_ROOT when USE_TURBOJPEG=ON.
export TURBOJPEG_ROOT="${TURBOJPEG_ROOT:-/shared/spapps_2026_01_avx2/gcc-11.5.0/libjpeg-turbo-3.1.3-gx4a65sxms56s5xcm5s5gebgyl74xf3h}"
export LD_LIBRARY_PATH="$ROCAL_ROOT/lib:$ROCM_PATH/lib:/opt/amdgpu/lib64:${LOCAL_MPI_LIB}:${MPI_HOME}/lib:${TURBOJPEG_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
export ROCAL_BACKEND="${ROCAL_BACKEND:-CPU}"   # GPU fails vxPublishKernels on gfx942
echo "  mpirun: $(command -v mpirun || echo MISSING)  MPI_HOME=$MPI_HOME"

# ── RCCL / IB env for INTER-node (differs from single-node which disables IB) ──
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_P2P_LEVEL=SYS          # intra-node XGMI
export NCCL_IB_DISABLE=1           # single-node: XGMI only, no inter-node IB
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"
export HSA_FORCE_FINE_GRAIN_PCIE=1
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-3}"   # 1=NORMAL (compiles on miss); 3=immediate (needs cached kernel)
MASTER_ADDR=$(scontrol show hostnames "${SLURM_NODELIST}" | head -1); export MASTER_ADDR
export NCCL_ID_FILE="$PROJECT_DIR/.nccl_id_${SLURM_JOB_ID}"
rm -f "$NCCL_ID_FILE"

GFX_ARCH=$(rocminfo 2>/dev/null | awk '/gfx[0-9a-f]+/{for(i=1;i<=NF;i++) if($i ~ /^gfx/){print $i; exit}}'); [ -z "$GFX_ARCH" ] && GFX_ARCH=gfx942
echo "config: GLOBAL_BATCH=$GLOBAL_BATCH EPOCHS=$EPOCHS arch=$GFX_ARCH rocal_backend=$ROCAL_BACKEND use_power=$USE_POWER subset='${SUBSET_GLOB:-<full>}' decode_threads='${RN50_DECODE_THREADS:-<cpus-per-task>}' ringbuf='${RN50_RINGBUF:-0}' no_step_sync='${RN50_NO_STEP_SYNC:-0}'"

# ── build train_multinode to SHARED-FS dir (both nodes must see the binary) ──
BUILD_DIR="$PROJECT_DIR/.build_multinode"
mkdir -p "$BUILD_DIR"          # must exist before the cmake.log redirect below
echo "[build] cmake configure -> $BUILD_DIR"
cmake -S "$PROJECT_DIR/hip" -B "$BUILD_DIR" \
      -DCMAKE_HIP_ARCHITECTURES="$GFX_ARCH" -DCMAKE_BUILD_TYPE=Release \
      -DUSE_NCCL=ON -DUSE_ROCAL="${USE_ROCAL:-OFF}" -DUSE_MPI=ON -DUSE_TURBOJPEG="${USE_TURBOJPEG:-ON}" \
      > "$BUILD_DIR/cmake.log" 2>&1 || { echo "FAIL cmake — tail:"; tail -40 "$BUILD_DIR/cmake.log"; exit 1; }
cmake --build "$BUILD_DIR" -j"$MAKE_JOBS" --target train_multinode \
      > "$BUILD_DIR/build.log" 2>&1 || { echo "FAIL build — tail:"; tail -80 "$BUILD_DIR/build.log"; exit 1; }
BIN="$BUILD_DIR/train_multinode"
[ -x "$BIN" ] || { echo "FAIL: $BIN not built"; exit 1; }
echo "  ✓ built $BIN"
echo "  linked: $(ldd "$BIN" 2>&1 | grep -oE 'librccl[^ ]*|librocal[^ ]*' | sort -u | tr '\n' ' ')"

# ── Stage ImageNet-1K — the ngc_resnet_benchmark.slurm method (full unzip +
# arrange val via LOC_val_solution.csv), but to a PERSISTENT /work path instead of
# $TMPDIR: on gaod $TMPDIR is /tmp (~512 MB), too small for 147 GB. /work is 40 TB
# and SHARED by both nodes, so one extraction serves single- AND multi-node (all
# ranks read it; no per-node race). Idempotent via a completion marker.
TMPDIR_DATA="${IMAGENET_EXTRACT:-$PROJECT_DIR/ImageNet-1K/competitions/imagenet-object-localization-challenge}"
export DATA_DIR="$TMPDIR_DATA/ILSVRC/Data/CLS-LOC"
mkdir -p "$TMPDIR_DATA"
echo "[stage] ImageNet-1K -> $TMPDIR_DATA (/work, persistent; free: $(df -h "$TMPDIR_DATA" 2>/dev/null | tail -1 | awk '{print $4}'))"
if [ -f "$TMPDIR_DATA/.staged.done" ] && [ -n "$(ls -A "$DATA_DIR/train" 2>/dev/null)" ]; then
    echo "  already extracted, skipping unzip"
else
    echo "  unzip $IMAGENET_ZIP (full ~147 GB, 1.28M files on Lustre — ~30-60 min) ..."
    unzip -q -n "$IMAGENET_ZIP" -d "$TMPDIR_DATA" || { echo "FAIL: unzip"; exit 1; }
    echo "  arranging val into class subfolders (LOC_val_solution.csv) ..."
    IMAGENET_VAL="$DATA_DIR/val" IMAGENET_CSV="$TMPDIR_DATA/LOC_val_solution.csv" python3 - <<'PYEOF'
import csv, os, shutil, sys
val_dir = os.environ["IMAGENET_VAL"]; csv_path = os.environ["IMAGENET_CSV"]
if not os.path.isfile(csv_path):
    print("  WARNING: LOC_val_solution.csv not found; skipping val prep"); sys.exit(0)
moved = 0
with open(csv_path, newline="") as h:
    for row in csv.DictReader(h):
        iid = row["ImageId"]; wnid = row["PredictionString"].split()[0]
        src = os.path.join(val_dir, iid + ".JPEG")
        if not os.path.isfile(src):
            continue
        dst = os.path.join(val_dir, wnid); os.makedirs(dst, exist_ok=True)
        shutil.move(src, os.path.join(dst, iid + ".JPEG")); moved += 1
print(f"  moved {moved} val images into class subfolders")
PYEOF
    touch "$TMPDIR_DATA/.staged.done"
fi
echo "  staged: DATA_DIR=$DATA_DIR train=$(ls "$DATA_DIR/train" 2>/dev/null | wc -l)cls/$(find "$DATA_DIR/train" -name '*.JPEG' 2>/dev/null | wc -l)imgs val=$(ls "$DATA_DIR/val" 2>/dev/null | wc -l)cls"

# ── per-node launch wrapper (hostname-keyed MIOpen cache; then exec the binary) ──
# Job-unique filename: GLOBAL_BATCH/EPOCHS are baked in, so concurrent jobs
# sharing BUILD_DIR must not overwrite each other's wrapper.
WRAP="$BUILD_DIR/_launch_rank_${SLURM_JOB_ID}.sh"
cat > "$WRAP" <<WRAP_EOF
#!/bin/bash
export MIOPEN_USER_DB_PATH="$PROJECT_DIR/.miopen_mn_\$(hostname -s)"
export MIOPEN_CUSTOM_CACHE_DIR="\$MIOPEN_USER_DB_PATH"
mkdir -p "\$MIOPEN_USER_DB_PATH"
exec "$BIN" "$DATA_DIR" "$GLOBAL_BATCH" "$EPOCHS" "$LR"
WRAP_EOF
chmod +x "$WRAP"

OUT_DIR="$PROJECT_DIR/power_analysis/results/native_multinode/mn_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"
SRUN_CMD=(srun --mpi=pmi2 --gpu-bind=none --export=ALL /usr/bin/bash "$WRAP")   # uses the job allocation (8 tasks/1 node, or 16/2 nodes)

echo "[run] $( [ "$USE_POWER" = 1 ] && echo 'measured' || echo 'bare' ) launch at $(date)"
if [ "$USE_POWER" = 1 ]; then
    bash "$PROJECT_DIR/power_analysis/measure_power_multinode.sh" \
        -o "$OUT_DIR" -i 0.5 -b 15 -c 8 -w 30 -- "${SRUN_CMD[@]}"
    RC=$?
else
    "${SRUN_CMD[@]}"; RC=$?
fi

rm -f "$NCCL_ID_FILE" "$WRAP"
echo "=== done $(date)  rc=$RC  outputs: $OUT_DIR ==="
exit $RC
