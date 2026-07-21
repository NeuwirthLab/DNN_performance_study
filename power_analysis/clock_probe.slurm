#!/bin/bash -l
# ============================================================================
# clock_probe.slurm  --  can a NORMAL user set the gfx clock on this cluster?
# ----------------------------------------------------------------------------
# Gap plan A2 (HIPC2026_gap_plan.md). The cap sweep cannot reach low
# frequencies (caps <400 W are floored), so discriminating the AccelWattch
# cubic from a quadratic needs DIRECT clock control. Precedent: rocm-smi
# --setpoweroverdrive works for a normal user while amd-smi set --power-cap
# returns NO_PERM (probe 52410392) -- so rocm-smi sysfs paths may be writable.
#
# Probes, in order (read-only first, then set + readback + restore):
#   1. rocm-smi --showclkfrq          : list gfx DPM levels (read-only)
#   2. rocm-smi --setperflevel manual : required before --setsclk
#   3. rocm-smi --setsclk <low level> : the actual permission test
#   4. readback under a short GPU burn (idle clock gates to 122 MHz, so only
#      a loaded readback proves the ceiling took effect)
#   5. restore --setperflevel auto
#
# Submit: sbatch power_analysis/clock_probe.slurm
# ============================================================================
#SBATCH --job-name=clk_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --account=l0003099
#SBATCH --partition=acc
#SBATCH --gres=gpu:mi300x:1
#SBATCH --time=00:15:00
#SBATCH --output=power_analysis/results/_diag/clock_probe_%j.out
#SBATCH --error=power_analysis/results/_diag/clock_probe_%j.err
#SBATCH --export=NONE
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"; PROJECT_PWD="$(pwd)"
export PATH=/opt/rocm/bin:$PATH

SMI=/usr/bin/rocm-smi; [ -x "$SMI" ] || SMI=rocm-smi
echo "=== clock probe, node $(hostname), $(date) ==="
echo "--- 1. DPM gfx clock levels (read-only) ---"
"$SMI" --showclkfrq || true
echo "--- current perf level ---"
"$SMI" --showperflevel || true

echo "--- 2. setperflevel manual ---"
if "$SMI" --setperflevel manual --autorespond yes; then PERM_PERF=yes; else PERM_PERF=no; fi
echo "PERM setperflevel: $PERM_PERF"

echo "--- 3. setsclk to level 1 (low) ---"
if "$SMI" --setsclk 1 --autorespond yes; then PERM_SCLK=yes; else PERM_SCLK=no; fi
echo "PERM setsclk: $PERM_SCLK"
"$SMI" --showclkfrq || true

echo "--- 4. loaded readback (30 s busy loop, sample sclk at 1 Hz) ---"
# tiny HIP burn: repeated large GEMM via rocblas-bench if present, else a
# python torch loop inside nothing -- fall back to hipcc one-liner kernel.
BURN_OK=no
if command -v rocblas-bench >/dev/null 2>&1; then
    timeout 30 rocblas-bench -f gemm -m 8192 -n 8192 -k 8192 --iters 1000 >/dev/null 2>&1 &
    BURN_PID=$!; BURN_OK=yes
else
    # fallback: hashcat-free generic burn via dd on GPU is not a thing; use
    # rocm_bandwidth_test loop if available
    if command -v rocm_bandwidth_test >/dev/null 2>&1; then
        ( for i in $(seq 1 30); do rocm_bandwidth_test -s 0 -d 0 >/dev/null 2>&1; done ) &
        BURN_PID=$!; BURN_OK=yes
    fi
fi
if [ "$BURN_OK" = yes ]; then
    for i in $(seq 1 10); do
        "$SMI" --showgpuclocks 2>/dev/null | grep -i "sclk\|gfx" | head -2
        sleep 2
    done
    kill "$BURN_PID" 2>/dev/null; wait "$BURN_PID" 2>/dev/null
else
    echo "no burn tool found -- readback is UNLOADED (idle clock gating applies):"
    "$SMI" --showgpuclocks || true
fi

echo "--- 5. restore auto ---"
"$SMI" --setperflevel auto --autorespond yes || true
"$SMI" --showperflevel || true

echo "=== VERDICT: setperflevel=$PERM_PERF setsclk=$PERM_SCLK ==="
echo "yes/yes -> frequency sweep is GO (gap plan A2): extend fit range below 558 MHz"
echo "no/*    -> reword form claim as assumption + sensitivity (already done in §5b)"
