#!/bin/bash -l
# ============================================================================
# rccl_bench_2node.slurm  --  2-node RCCL AllReduce energy bench
# ----------------------------------------------------------------------------
# Calibrates e_comm_2node: the inter-node allreduce energy coefficient
# (J/byte) needed by compose_energy.py for the 2-node energy model.
#
# This is the 2-node analogue of the intra-node allreduce calibration run
# used for the existing e_comm coefficient.
#
# What it does:
#   1. Builds rccl-tests (all_reduce_perf) if needed.
#   2. Runs allreduce_perf with mpirun across 2 nodes (16 ranks, 1 GPU/rank).
#   3. Wraps the benchmark in measure_power_multinode.sh to capture total
#      energy (both nodes) for a range of message sizes.
#   4. Computes e_comm_2node = E_dyn / V_allreduce for the ResNet-50
#      gradient bucket size (~97 MB), and writes a components/allreduce_2node/
#      run_metrics.csv that compose_energy.py extract reads.
#
# Outputs:
#   power_analysis/results/components/allreduce_2node/
#     run_metrics.csv    (for compose_energy.py extract --run e_comm_2node=...)
#     power_summary.csv  (per-node breakdown)
#     command.log        (rccl-tests stdout)
#
# Submit:
#   sbatch power_analysis/rccl_bench_2node.slurm
# ============================================================================
#SBATCH --job-name=rn50_rccl_2n
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --account=l0003099
#SBATCH --partition=acc
#SBATCH --gres=gpu:mi300x:8
#SBATCH --time=01:30:00
#SBATCH --reservation=gaod_powcap
#SBATCH --nodelist=gaod[0001-0002]
#SBATCH --output=power_analysis/results/rccl2n_%j.out
#SBATCH --error=power_analysis/results/rccl2n_%j.err
#SBATCH --export=NONE

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
PROJECT_PWD="$(pwd)"

echo "==== RCCL AllReduce 2-node energy bench | job ${SLURM_JOB_ID:-local} | $(date) ===="
echo "Nodes: ${SLURM_NODELIST:-?}"

# ── ROCm + MPI env ───────────────────────────────────────────────────────────
module --force purge >/dev/null 2>&1 || true
module load cmake    >/dev/null 2>&1 || true
module load openmpi  >/dev/null 2>&1 || module load openmpi/4.1.8-6xzv >/dev/null 2>&1 || true

export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
MPI_HOME="$(dirname "$(dirname "$(command -v mpirun 2>/dev/null || echo /usr/bin/x)")")"
# Use local copy of libmpi.so.40 (on /work/scratch, accessible from all nodes)
# so compute nodes don't need /shared mounted.
LOCAL_MPI_LIB="$PROJECT_PWD/rccl-tests/mpi_lib"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:${LOCAL_MPI_LIB}:${MPI_HOME}/lib:${LD_LIBRARY_PATH:-}"
echo "  mpirun : $(command -v mpirun || echo MISSING)  MPI_HOME=$MPI_HOME"

# ── RCCL env for 2-node (IB enabled) ────────────────────────────────────────
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"
export HIP_FORCE_DEV_KERNARG=1
export RCCL_ENABLE_CONTEXT_TRACKING=1

MASTER_ADDR=$(scontrol show hostnames "${SLURM_NODELIST}" | head -1)
export MASTER_ADDR
echo "  MASTER_ADDR=$MASTER_ADDR"

# ── Build rccl-tests (clone if needed) ──────────────────────────────────────
RT_DIR="$PROJECT_PWD/rccl-tests"
RT_BIN="$RT_DIR/build/all_reduce_perf"
# Build with MPI=1 (rank IDs set via MPI); clean to avoid stale .o from any prior MPI=0 build
rm -rf "$RT_DIR/build"
[ -d "$RT_DIR/.git" ] || git clone https://github.com/ROCm/rccl-tests.git "$RT_DIR" 2>&1 | tail -3
echo "  building rccl-tests (MPI=1) ..."
make -C "$RT_DIR" MPI=1 MPI_HOME="$MPI_HOME" NCCL_HOME="$ROCM_PATH" \
     -j"${SLURM_CPUS_PER_TASK:-8}" > "$RT_DIR/build.log" 2>&1 \
    || { echo "FAIL build rccl-tests"; tail -40 "$RT_DIR/build.log"; exit 1; }
[ -x "$RT_BIN" ] || { echo "FAIL: $RT_BIN not built"; exit 1; }
echo "  ✓ rccl-tests: $RT_BIN"

# ── Output directory (under components/ for compose_energy.py discover) ──────
OUT_BASE="$PROJECT_PWD/power_analysis/results/components/allreduce_2node"
mkdir -p "$OUT_BASE"
echo "  out: $OUT_BASE"

# ── Benchmark command: 16 ranks across 2 nodes, 1 GPU/rank ──────────────────
# Message size sweep: 8 B to 1 GB by factor 2.
# -w 20 warmup, -n 50 measured iterations per size.
# --bind-to none: let SLURM/ROCr handle GPU affinity.
BENCH_CMD=(
    srun --mpi=pmi2 --export=ALL --ntasks=16 --ntasks-per-node=8
    "$RT_BIN" -b 8 -e 1G -f 2 -g 1 -w 20 -n 50
)

echo "  bench: ${BENCH_CMD[*]}"
echo ""

# ── Wrap the benchmark in measure_power_multinode.sh ─────────────────────────
# The harness captures total power on BOTH nodes throughout the benchmark, then
# analyze_multinode_power.py sums the energy across nodes. The resulting
# run_metrics.csv carries E_dyn_J = total dynamic energy for the benchmark run.
#
# e_comm_2node derivation (offline, in compose_energy.py):
#   V_ar = 97.49e6 B  (ResNet-50 gradient bucket, same as the 1-node calibration)
#   iters_measured  = 50 * (number_of_size_points_at_97MB_target)
#   e_comm_2node = (E_dyn_J / train_s) / (V_ar / t_step_real)
#                = P_dyn_W  * t_step_real / V_ar       [J/byte]
#
# For the component replay the benchmark itself IS the "training command" --
# compose_energy.py will read command.log for the [RATE]/[WORK] lines if
# present, or fall back to the driver-rate method.
#
# We add -t (throughput hint) = gradient bytes per second measured by rccl-tests
# at the ~97 MB row so the analyzer can cross-check.

bash "$PROJECT_PWD/power_analysis/measure_power_multinode.sh" \
    -o "$OUT_BASE" \
    -i 0.5 \
    -b 20 \
    -c 10 \
    -w 30 \
    -- "${BENCH_CMD[@]}"

# ── Extract bandwidth at the gradient-bucket size from command.log ─────────
echo ""
echo "=== bandwidth at ~97 MB (gradient bucket) row ==="
LOG="$OUT_BASE/command.log"
if [ -f "$LOG" ]; then
    awk 'NR>1 && $1 ~ /^[0-9]+$/ {
        mb = $1/1048576
        if (mb >= 64 && mb <= 160) {
            busbw = $(NF-1)
            printf "  size=%8.1f MB  busbw=%s GB/s\n", mb, busbw
        }
    }' "$LOG"
    echo ""
    echo "  e_comm_2node calibration:"
    echo "  (offline) Run: python3 power_analysis/compose_energy.py extract \\"
    echo "      --run e_comm_2node=$OUT_BASE"
    echo "  Then validate with: python3 power_analysis/compose_energy.py validate \\"
    echo "      --nnodes 2 --pytorch-energy <2node-training-run-dir>"
fi

echo ""
echo "==== rccl_bench_2node done $(date) ===="
echo "Outputs: $OUT_BASE"
echo "  run_metrics.csv  -> feed to compose_energy.py extract as e_comm_2node"
