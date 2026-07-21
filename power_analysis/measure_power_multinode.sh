#!/usr/bin/env bash
# ============================================================================
# measure_power_multinode.sh  --  multi-node power harness for MI300X
# ----------------------------------------------------------------------------
# Same interface as measure_power.sh, but samples GPU power on ALL allocated
# SLURM nodes simultaneously. Each node writes to its own sub-directory:
#
#     $OUT_DIR/node_<hostname>/power_samples.csv
#
# A shared STOP sentinel file ($OUT_DIR/STOP) tells all samplers to exit.
# Shared markers.csv is written by this (primary) node and read by
# analyze_multinode_power.py to window idle / train phases.
#
# Requires: SLURM allocation with N>=2 nodes (reads SLURM_NODELIST /
#           SLURM_JOB_NUM_NODES). Samplers launch via `srun --overlap`
#           so the main training step can run concurrently.
#
# Usage (same as measure_power.sh):
#   measure_power_multinode.sh [opts] -- srun ... torchrun ... train.py
#
# Options:
#   -o DIR     output directory (default: power_analysis/results/mn_<jobid>)
#   -i SEC     sample interval seconds        (default 1.0)
#   -b SEC     idle baseline window seconds   (default 20)
#   -c SEC     cooldown window seconds        (default 10)
#   -w SEC     steady-state warmup trim secs  (default 30)
#   -t IMG/S   throughput hint for J/image metrics
#   -E N       warmup epochs to skip in throughput averaging
#   -n N       number of nodes (default: $SLURM_JOB_NUM_NODES or 1)
#   -h         this help
#   --         everything after is the command to measure
# ============================================================================
set -uo pipefail

# ---- defaults --------------------------------------------------------------
INTERVAL=1.0
IDLE_SECONDS=20
COOLDOWN_SECONDS=10
WARMUP_SECONDS=30
THROUGHPUT=""
OUT_DIR=""
WARMUP_EPOCHS=""
NNODES="${SLURM_JOB_NUM_NODES:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYZER="$SCRIPT_DIR/analyze_multinode_power.py"

usage() {
    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        -o) OUT_DIR="$2"; shift 2 ;;
        -i) INTERVAL="$2"; shift 2 ;;
        -b) IDLE_SECONDS="$2"; shift 2 ;;
        -c) COOLDOWN_SECONDS="$2"; shift 2 ;;
        -w) WARMUP_SECONDS="$2"; shift 2 ;;
        -t) THROUGHPUT="$2"; shift 2 ;;
        -E) WARMUP_EPOCHS="$2"; shift 2 ;;
        -n) NNODES="$2"; shift 2 ;;
        -h|--help) usage ;;
        --) shift; break ;;
        *) echo "unknown option: $1" >&2; usage ;;
    esac
done
CMD=("$@")

# ---- output dir + markers --------------------------------------------------
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/results/mn_${SLURM_JOB_ID:-$TS}}"
mkdir -p "$OUT_DIR"
MARKERS="$OUT_DIR/markers.csv"
CMD_LOG="$OUT_DIR/command.log"
echo "event,timestamp_s" > "$MARKERS"
mark() { echo "$1,$(date +%s.%N)" >> "$MARKERS"; echo "[mark] $1"; }

echo "=== measure_power_multinode.sh ==="
echo "  nodes       : $NNODES  (SLURM_NODELIST=${SLURM_NODELIST:-?})"
echo "  out dir     : $OUT_DIR"
echo "  interval    : ${INTERVAL}s  idle: ${IDLE_SECONDS}s  cooldown: ${COOLDOWN_SECONDS}s"
[ ${#CMD[@]} -gt 0 ] && echo "  command     : ${CMD[*]}" || echo "  command     : (idle only)"

# ---- write the per-node sampler script to the shared filesystem ------------
# Each srun task picks it up and samples its LOCAL node's GPUs.
SAMPLER_SCRIPT="$OUT_DIR/_node_sampler.sh"
cat > "$SAMPLER_SCRIPT" << 'SAMPLER_EOF'
#!/bin/bash
# Per-node GPU power sampler launched by srun --overlap.
# Args: $1=OUT_DIR  $2=INTERVAL
set -uo pipefail
_OUT="$1"
_INTERVAL="${2:-1.0}"
_NODE=$(hostname -s)
_MYDIR="$_OUT/node_$_NODE"
mkdir -p "$_MYDIR"

if command -v amd-smi >/dev/null 2>&1; then
    _BACKEND="amd-smi"
elif command -v rocm-smi >/dev/null 2>&1; then
    _BACKEND="rocm-smi"
else
    echo "[sampler-$_NODE] ERROR: no AMD SMI tool on PATH" >&2; exit 1
fi

_clean() { grep -v -E '^WARNING:|missing the following required groups|^$'; }
_snap() {
    if [ "$_BACKEND" = "amd-smi" ]; then
        amd-smi metric -g all -p -c -t -u --csv 2>/dev/null | _clean
    else
        rocm-smi --showpower --showclocks --showtemp --showuse --csv 2>/dev/null | _clean
    fi
}
# Per-node HW energy-accumulator snapshot. The single-node harness uses ΔE
# (after-before) as the <5% ground-truth cross-check against the trapezoid
# integral; for multinode this MUST run on EVERY node (one node's amd-smi
# --energy only sees its OWN GPUs), so each sampler writes its own snapshots.
_energy_snap() {
    if [ "$_BACKEND" = "amd-smi" ]; then
        amd-smi metric -g all --energy --csv 2>/dev/null | _clean
    else
        rocm-smi --showenergycounter --csv 2>/dev/null | _clean
    fi
}

HEADER="$(_snap | head -1)"
[ -z "$HEADER" ] && { echo "[sampler-$_NODE] ERROR: no data from backend" >&2; exit 1; }
echo "timestamp_s,$HEADER" > "$_MYDIR/power_samples.csv"
echo "[sampler-$_NODE] started (backend=$_BACKEND interval=${_INTERVAL}s)" >&2

# energy accumulator BEFORE (this node's GPUs) -> analyze_power.py reads it for ΔE
_energy_snap > "$_MYDIR/energy_before.csv" 2>/dev/null || true

while [ ! -f "$_OUT/STOP" ]; do
    _ts=$(date +%s.%N)
    _snap | tail -n +2 | sed "s/^/$_ts,/" >> "$_MYDIR/power_samples.csv"
    sleep "$_INTERVAL"
done

# energy accumulator AFTER (this node's GPUs)
_energy_snap > "$_MYDIR/energy_after.csv" 2>/dev/null || true

echo "[sampler-$_NODE] STOP; $(( $(wc -l < "$_MYDIR/power_samples.csv") - 1 )) rows" >&2
SAMPLER_EOF
chmod +x "$SAMPLER_SCRIPT"

# ---- energy snapshot BEFORE ------------------------------------------------
# We do a best-effort local-node energy capture via amd-smi --energy; the
# multi-node aggregation (via per-node power_samples.csv) is the primary path.
if command -v amd-smi >/dev/null 2>&1; then
    amd-smi metric -g all --energy --csv 2>/dev/null > "$OUT_DIR/energy_before.csv" || true
fi

# ---- launch sidecar samplers on ALL nodes via srun --overlap ---------------
# --overlap allows this step to share the allocation with the main training step.
rm -f "$OUT_DIR/STOP"
srun --overlap \
     --ntasks="$NNODES" --ntasks-per-node=1 \
     /usr/bin/bash "$SAMPLER_SCRIPT" "$OUT_DIR" "$INTERVAL" &
SRUN_PID=$!

cleanup() {
    touch "$OUT_DIR/STOP" 2>/dev/null || true
    wait "$SRUN_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 3   # let every node's sampler write its header

# ---- phase 1: idle baseline ------------------------------------------------
mark idle_start
sleep "$IDLE_SECONDS"
mark idle_end

# ---- phase 2: the measured command -----------------------------------------
STATUS=0
if [ ${#CMD[@]} -gt 0 ]; then
    mark train_start
    "${CMD[@]}" 2>&1 | tee "$CMD_LOG"
    STATUS=${PIPESTATUS[0]}
    mark train_end
    echo "[measure] command exit status: $STATUS"
else
    mark train_start; mark train_end
fi

# ---- phase 3: cooldown + stop samplers -------------------------------------
sleep "$COOLDOWN_SECONDS"
mark cooldown_end
touch "$OUT_DIR/STOP"
wait "$SRUN_PID" 2>/dev/null || true
SRUN_PID=""

if command -v amd-smi >/dev/null 2>&1; then
    amd-smi metric -g all --energy --csv 2>/dev/null > "$OUT_DIR/energy_after.csv" || true
fi

# ---- auto-detect throughput (rank-0 log lines) ----------------------------
_W="${WARMUP_EPOCHS:-0}"
if [ -z "$THROUGHPUT" ] && [ -f "$CMD_LOG" ]; then
    THROUGHPUT="$(awk -v w="$_W" \
        '$1=="epoch" && $3=="train.compute_ips"{e=$2+0; if(e>=w){s+=$5;n++}} \
         END{if(n)printf "%.2f",s/n}' "$CMD_LOG")"
fi

# ---- capture epoch markers -------------------------------------------------
if [ -f "$CMD_LOG" ]; then
    grep -aE '\[PWRMARK\] epoch ' "$CMD_LOG" 2>/dev/null \
        | awk '{print "epoch"$3","$4}' >> "$MARKERS"
fi

# ---- aggregate analysis across all nodes -----------------------------------
echo ""; echo "=== multi-node analysis ==="
THRU_ARG=(); [ -n "$THROUGHPUT" ] && THRU_ARG=(--throughput "$THROUGHPUT")
EPOCH_ARG=(); [ -n "$WARMUP_EPOCHS" ] && EPOCH_ARG=(--warmup-epochs "$WARMUP_EPOCHS")

if [ -f "$ANALYZER" ] && command -v python3 >/dev/null 2>&1; then
    python3 "$ANALYZER" \
        --run-dir "$OUT_DIR" \
        --warmup "$WARMUP_SECONDS" \
        "${THRU_ARG[@]}" "${EPOCH_ARG[@]}" \
        || echo "  (analyzer error; per-node CSVs in $OUT_DIR/node_*/)"
else
    echo "  analyze_multinode_power.py unavailable — raw CSVs in $OUT_DIR/node_*/"
fi

echo ""; echo "  outputs: $OUT_DIR"
exit "$STATUS"
