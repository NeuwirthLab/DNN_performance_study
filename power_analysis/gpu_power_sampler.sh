#!/usr/bin/env bash
# ============================================================================
# gpu_power_sampler.sh  --  background socket-power sampler for AMD Instinct
#                           (MI300X / gfx942), one tidy CSV row per (sample,GPU)
# ----------------------------------------------------------------------------
# WHY this exists:
#   amd-smi / rocm-smi only live on the compute node (and inside the rocm
#   container) -- never on the submit node. So power has to be measured *next
#   to the workload*, inside the SLURM allocation. This script is meant to be
#   launched in the background by a job wrapper, then SIGTERM'd when the
#   workload finishes (power_profile.slurm does exactly that).
#
# WHAT it logs (best-effort, whatever the backend exposes):
#   wall-clock timestamp, GPU index, socket power (W), gfx/mem clocks,
#   temperature, utilization. Column NAMES come straight from the backend's
#   own CSV header, so the analyzer matches them by regex rather than position.
#
# Backend choice:
#   amd-smi  -> supported tool on ROCm 6.x (preferred)
#   rocm-smi -> deprecated but still present; automatic fallback
#
# Sampling rate:
#   Each sample forks the SMI CLI once, so ~1-2 Hz is the practical ceiling.
#   That captures idle / steady-state / peak / scaling / energy fine. For
#   intra-iteration phase power (fwd vs bwd vs allreduce) you need the amdsmi
#   Python binding at 50-100 Hz instead -- see README note in analyze_power.py.
#
# Usage:
#   ./gpu_power_sampler.sh <out_csv> [interval_s]
#   kill -TERM <pid>     # stop cleanly (flushes + reports row count)
# ============================================================================
set -uo pipefail

OUT="${1:?usage: gpu_power_sampler.sh <out_csv> [interval_s]}"
INTERVAL="${2:-1.0}"
mkdir -p "$(dirname "$OUT")"

# --- pick a telemetry backend -----------------------------------------------
if command -v amd-smi >/dev/null 2>&1; then
    BACKEND="amd-smi"
elif command -v rocm-smi >/dev/null 2>&1; then
    BACKEND="rocm-smi"
else
    echo "[sampler] ERROR: neither amd-smi nor rocm-smi found on PATH" >&2
    exit 1
fi
echo "[sampler] backend=$BACKEND interval=${INTERVAL}s out=$OUT" >&2

# --- one CSV line per call, machine-readable --------------------------------
# -g all : every visible GPU. -p power, -c clocks, -t temperature, -u util.
sample_amdsmi() { amd-smi metric -g all -p -c -t -u --csv 2>/dev/null; }
sample_rocmsmi() { rocm-smi --showpower --showclocks --showtemp --showuse --csv 2>/dev/null; }

# amd-smi may print a render/video group WARNING to STDOUT -> strip non-CSV noise
_clean() { grep -v -E '^WARNING:|missing the following required groups|^$'; }
snapshot() { { if [ "$BACKEND" = "amd-smi" ]; then sample_amdsmi; else sample_rocmsmi; fi; } | _clean; }

# --- write header once, prefixed with our wall-clock column -----------------
HEADER="$(snapshot | head -1)"
if [ -z "$HEADER" ]; then
    echo "[sampler] ERROR: backend returned no data (check ROCR/HIP_VISIBLE_DEVICES)" >&2
    exit 1
fi
echo "timestamp_s,${HEADER}" > "$OUT"

# --- sample loop; SIGTERM/SIGINT breaks out cleanly -------------------------
STOP=0
trap 'STOP=1' TERM INT
while [ "$STOP" -eq 0 ]; do
    ts="$(date +%s.%N)"
    snapshot | tail -n +2 | sed "s/^/${ts},/" >> "$OUT"
    sleep "$INTERVAL"
done
echo "[sampler] stopped; wrote $(($(wc -l < "$OUT") - 1)) sample rows to $OUT" >&2
