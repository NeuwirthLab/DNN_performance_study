#!/usr/bin/env bash
# ============================================================================
# measure_power.sh  --  run a power measurement around ANY command (MI300X)
# ----------------------------------------------------------------------------
# Self-contained: the sampler loop is embedded, so this one file is all you
# need. It wraps whatever command you give it in:
#
#     [idle baseline] -> [your command] -> [cooldown]
#        ^ socket power sampled across the whole run, phases marked ^
#
# and then hands the CSVs to analyze_power.py (same dir) for the numbers.
#
# Run it INSIDE a GPU allocation (amd-smi/rocm-smi only exist on the compute
# node):
#     salloc -p gpu:mi300x --gres=gpu:mi300x:8 -A l0003099 -t 1:00:00
#     ./power_analysis/measure_power.sh -- python train.py ...
#
# ...or call it from inside an sbatch script (it inherits the allocation).
#
# Examples:
#   # measure a containerized training run
#   ./measure_power.sh -t 12000 -- \
#       apptainer exec --rocm pytorch_rocm.sif python -m torch.distributed.run ...
#
#   # measure just idle power (no command) for the baseline floor
#   ./measure_power.sh -b 60 -o results/idle_floor
#
#   # custom windows + 2 Hz sampling
#   ./measure_power.sh -i 0.5 -b 30 -c 15 -w 45 -- ./my_benchmark
# ============================================================================
set -uo pipefail

# ---- defaults --------------------------------------------------------------
INTERVAL=1.0          # seconds between samples (~1 Hz; CLI fork-limited)
IDLE_SECONDS=20       # idle baseline window before the command
COOLDOWN_SECONDS=10   # quiet window after the command
WARMUP_SECONDS=30     # trimmed from front of command window for steady-state
THROUGHPUT=""         # optional img/s hint -> enables J/image, img/s/W
OUT_DIR=""
GPU_TARGET=""         # which physical GPUs to sample; default = your allocation
WARMUP_EPOCHS=""      # if set, measure from epoch <N> onward (epoch-based warmup)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYZER="$SCRIPT_DIR/analyze_power.py"

usage() {
    sed -n '2,/^# ===.*$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  -o DIR     output directory (default: power_analysis/results/measure_<ts>)
  -i SEC     sample interval seconds        (default 1.0)
  -b SEC     idle baseline window seconds    (default 20)
  -c SEC     cooldown window seconds         (default 10)
  -w SEC     steady-state warmup trim seconds(default 30)
  -t IMG/S   throughput hint for efficiency metrics
  -g LIST    GPUs to sample, e.g. "0" or "0,1,2,3" (default: your allocation's
             visible devices; "all" samples every GPU on the node)
  -h         this help
  --         everything after is the command to measure (optional)
EOF
}

# ---- parse options ---------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        -o) OUT_DIR="$2"; shift 2;;
        -i) INTERVAL="$2"; shift 2;;
        -b) IDLE_SECONDS="$2"; shift 2;;
        -c) COOLDOWN_SECONDS="$2"; shift 2;;
        -w) WARMUP_SECONDS="$2"; shift 2;;
        -t) THROUGHPUT="$2"; shift 2;;
        -g) GPU_TARGET="$2"; shift 2;;
        -E) WARMUP_EPOCHS="$2"; shift 2;;
        -h|--help) usage; exit 0;;
        --) shift; break;;
        *)  echo "unknown option: $1" >&2; usage; exit 2;;
    esac
done
CMD=("$@")   # the workload (may be empty -> idle-only measurement)

# ---- pick a telemetry backend (AMD or NVIDIA, so the same harness measures
#      the native-HIP, PyTorch-ROCm, and CUDA builds) ------------------------
if command -v amd-smi >/dev/null 2>&1; then
    BACKEND="amd-smi"
elif command -v rocm-smi >/dev/null 2>&1; then
    BACKEND="rocm-smi"
elif command -v nvidia-smi >/dev/null 2>&1; then
    BACKEND="nvidia-smi"
else
    echo "ERROR: no GPU telemetry tool (amd-smi / rocm-smi / nvidia-smi) on PATH." >&2
    echo "       Are you inside a GPU allocation? (salloc -p gpu:... )" >&2
    exit 1
fi

# ---- which GPUs to sample --------------------------------------------------
# On a SHARED node, sampling "all" captures neighbours' GPUs, so we default to
# the allocation's visible-device mask. -g overrides.
if [ -z "$GPU_TARGET" ]; then
    case "$BACKEND" in
        amd-smi|rocm-smi) GPU_TARGET="${ROCR_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-all}}";;
        nvidia-smi)       GPU_TARGET="${CUDA_VISIBLE_DEVICES:-all}";;
    esac
fi
# only a numeric / comma-numeric mask is usable as device indices
if [ "$GPU_TARGET" != "all" ] && ! [[ "$GPU_TARGET" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "[measure] WARN: visible-device mask '$GPU_TARGET' is not numeric; sampling ALL GPUs" >&2
    GPU_TARGET="all"
fi
_ids_space() { echo "${GPU_TARGET//,/ }"; }   # "0,1" -> "0 1" for amd-smi/rocm-smi

# ---- output dir + markers --------------------------------------------------
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/results/measure_${SLURM_JOB_ID:-$TS}}"
mkdir -p "$OUT_DIR"
POWER_CSV="$OUT_DIR/power_samples.csv"
MARKERS="$OUT_DIR/markers.csv"
CMD_LOG="$OUT_DIR/command.log"
echo "event,timestamp_s" > "$MARKERS"
mark() { echo "$1,$(date +%s.%N)" >> "$MARKERS"; echo "[mark] $1"; }

echo "=== measure_power.sh ==="
echo "  backend     : $BACKEND"
echo "  out dir     : $OUT_DIR"
echo "  interval    : ${INTERVAL}s   idle: ${IDLE_SECONDS}s  cooldown: ${COOLDOWN_SECONDS}s  warmup-trim: ${WARMUP_SECONDS}s"
echo "  sampling GPU: $GPU_TARGET"
[ ${#CMD[@]} -gt 0 ] && echo "  command     : ${CMD[*]}" || echo "  command     : (none -- idle measurement only)"

# ---- embedded sampler ------------------------------------------------------
# amd-smi prints a "missing render/video groups" WARNING to STDOUT on some
# setups, which would corrupt the CSV -- strip any non-CSV noise lines.
_clean() { grep -v -E '^WARNING:|missing the following required groups|^$'; }
_snapshot() { _snapshot_raw | _clean; }
_energy_snapshot() { _energy_raw | _clean; }
_snapshot_raw() {
    case "$BACKEND" in
        amd-smi)
            if [ "$GPU_TARGET" = all ]; then amd-smi metric -g all -p -c -t -u --csv 2>/dev/null
            else amd-smi metric -g $(_ids_space) -p -c -t -u --csv 2>/dev/null; fi ;;
        rocm-smi)
            if [ "$GPU_TARGET" = all ]; then rocm-smi --showpower --showclocks --showtemp --showuse --csv 2>/dev/null
            else rocm-smi -d $(_ids_space) --showpower --showclocks --showtemp --showuse --csv 2>/dev/null; fi ;;
        nvidia-smi)
            local q="index,power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu"
            if [ "$GPU_TARGET" = all ]; then nvidia-smi --query-gpu="$q" --format=csv 2>/dev/null
            else nvidia-smi -i "$GPU_TARGET" --query-gpu="$q" --format=csv 2>/dev/null; fi ;;
    esac
}
_energy_raw() {
    case "$BACKEND" in
        amd-smi)
            if [ "$GPU_TARGET" = all ]; then amd-smi metric -g all --energy --csv 2>/dev/null
            else amd-smi metric -g $(_ids_space) --energy --csv 2>/dev/null; fi ;;
        rocm-smi) rocm-smi --showenergycounter --csv 2>/dev/null ;;
        nvidia-smi)
            if [ "$GPU_TARGET" = all ]; then nvidia-smi --query-gpu=index,total_energy_consumption --format=csv 2>/dev/null
            else nvidia-smi -i "$GPU_TARGET" --query-gpu=index,total_energy_consumption --format=csv 2>/dev/null; fi ;;
    esac
}
_sampler() {
    local header; header="$(_snapshot | head -1)"
    [ -z "$header" ] && { echo "ERROR: backend returned no data" >&2; exit 1; }
    echo "timestamp_s,${header}" > "$POWER_CSV"
    while :; do
        local ts; ts="$(date +%s.%N)"
        _snapshot | tail -n +2 | sed "s/^/${ts},/" >> "$POWER_CSV"
        sleep "$INTERVAL"
    done
}

# ---- energy accumulator BEFORE (exact ΔE ground-truth; best effort) --------
_energy_snapshot > "$OUT_DIR/energy_before.csv" 2>/dev/null || true

# ---- start sampler; guarantee it dies on any exit --------------------------
_sampler &
SPID=$!
cleanup() { [ -n "${SPID:-}" ] && kill -TERM "$SPID" 2>/dev/null; }
trap cleanup EXIT INT TERM
sleep 2   # let the header + first rows land

# ---- phase 1: idle baseline ------------------------------------------------
mark idle_start
sleep "$IDLE_SECONDS"
mark idle_end

# ---- phase 2: the command (train_* names so analyze_power.py picks them up) -
STATUS=0
if [ ${#CMD[@]} -gt 0 ]; then
    mark train_start
    "${CMD[@]}" 2>&1 | tee "$CMD_LOG"
    STATUS=${PIPESTATUS[0]}
    mark train_end
    echo "[measure] command exit status: $STATUS"
else
    # idle-only run: give the analyzer a (degenerate) train window = idle window
    mark train_start; mark train_end
fi

# ---- phase 3: cooldown, stop sampler, energy AFTER -------------------------
sleep "$COOLDOWN_SECONDS"
mark cooldown_end
kill -TERM "$SPID" 2>/dev/null || true
wait "$SPID" 2>/dev/null || true
SPID=""
_energy_snapshot > "$OUT_DIR/energy_after.csv" 2>/dev/null || true

# auto-detect throughput = AVERAGE per-epoch throughput over the MEASURED epochs
# (skip the first WARMUP_EPOCHS). NOTE: do NOT use the last epoch or NGC's
# "train.total_ips" summary -- that line is just a copy of the FINAL epoch, so a
# single outlier final epoch skews J/image and img/s/W (seen as a spurious 5-10%
# shift between 5- and 10-epoch runs at the same cap).
#  (a) PyTorch:  "epoch <n> train.compute_ips : <val>"   (n 0-indexed)
#  (b) native :  "<ep> <lr(0..1)> ... <Time(s)> <Img/sec>"  (ep 1-indexed)
_W="${WARMUP_EPOCHS:-0}"
if [ -z "$THROUGHPUT" ] && [ -f "$CMD_LOG" ]; then
    THROUGHPUT="$(awk -v w="$_W" '$1=="epoch" && $3=="train.compute_ips"{e=$2+0; if(e>=w){s+=$5;n++}} END{if(n)printf "%.2f",s/n}' "$CMD_LOG")"
fi
if [ -z "$THROUGHPUT" ] && [ -f "$CMD_LOG" ]; then
    THROUGHPUT="$(awk -v w="$_W" 'NF>=8 && $1 ~ /^[0-9]+$/ && $2 ~ /^0\.[0-9]+$/ {e=$1+0; if(e>w){s+=$NF;n++}} END{if(n)printf "%.2f",s/n}' "$CMD_LOG")"
fi

# ---- capture per-epoch wall-clock markers the trainer printed --------------
# Trainers emit "[PWRMARK] epoch <n> <unix_seconds>" at each epoch start; fold
# them into markers.csv so the analyzer can do epoch-based warmup/averaging.
if [ -f "$CMD_LOG" ]; then
    grep -aE '\[PWRMARK\] epoch ' "$CMD_LOG" 2>/dev/null \
        | awk '{print "epoch"$3","$4}' >> "$MARKERS"
fi

# ---- analyze ---------------------------------------------------------------
echo ""; echo "=== analysis ==="
THRU_ARG=(); [ -n "$THROUGHPUT" ] && THRU_ARG=(--throughput "$THROUGHPUT")
EPOCH_ARG=(); [ -n "$WARMUP_EPOCHS" ] && EPOCH_ARG=(--warmup-epochs "$WARMUP_EPOCHS")
if [ -f "$ANALYZER" ] && command -v python3 >/dev/null 2>&1; then
    python3 "$ANALYZER" --run-dir "$OUT_DIR" --warmup "$WARMUP_SECONDS" "${THRU_ARG[@]}" "${EPOCH_ARG[@]}" \
        || echo "  (analyzer error; raw CSVs in $OUT_DIR)"
else
    # minimal fallback: mean of the 3rd column (the prefixed power CSV) over all samples
    echo "  analyze_power.py / python3 unavailable -- quick fallback summary:"
    awk -F',' 'NR>1{n++; s+=$3; if($3>mx)mx=$3} END{if(n)printf "    samples=%d  mean(col3)=%.1f  max(col3)=%.1f\n", n, s/n, mx}' "$POWER_CSV"
    echo "    full data: $POWER_CSV  (run analyze_power.py later for proper numbers)"
fi

echo ""; echo "  outputs: $OUT_DIR"
exit "$STATUS"
