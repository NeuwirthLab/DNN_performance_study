#!/bin/bash -l
# ============================================================================
# probe_powercap.slurm  --  EXHAUSTIVE power-cap permission probe (MI300X)
# ----------------------------------------------------------------------------
# Tests EVERY plausible mechanism for setting the GPU power cap so we know
# definitively which (if any) path is permitted under the av30nuci reservation.
# Read-only telemetry is known to work; this focuses on the WRITE paths.
#
#   sbatch power_analysis/probe_powercap.slurm
# ============================================================================
#SBATCH --job-name=probe_pcap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --account=l0003099
#SBATCH --partition=acc
#SBATCH --gres=gpu:mi300x:1
#SBATCH --time=00:08:00
#SBATCH --reservation=gaod_powcap
#SBATCH --nodelist=gaod0002
#SBATCH --output=power_analysis/results/probe_pcap_%j.out
#SBATCH --export=NONE

export PATH="/opt/rocm/bin:$PATH"
SECTION() { echo ""; echo "==================== $* ===================="; }

echo "##### power-cap probe on $(hostname)  $(date) #####"

SECTION "0. Identity & groups"
echo "whoami : $(whoami)"
echo "id     : $(id)"
echo "groups : $(groups)"
echo "  (need 'render' and/or 'video' for direct device control)"

# NOTE: NO sudo anywhere — we are normal HPC users, not admins. Only non-sudo
# user-level paths are tested below; cap-set requires an admin group/ACL grant.

SECTION "2. Is /usr/bin/rocm-smi a wrapper that calls sudo?"
file /usr/bin/rocm-smi 2>&1
echo "--- first 30 lines ---"
head -30 /usr/bin/rocm-smi 2>&1
echo "--- does it reference sudo? ---"
grep -n "sudo" /usr/bin/rocm-smi 2>&1 | head -10 || echo "  (no sudo references / binary)"

SECTION "3. Tool versions"
echo "--- amd-smi version ---"; amd-smi version 2>&1 | head -5
echo "--- rocm-smi version ---"; rocm-smi --version 2>&1 | head -5

SECTION "4. READ current cap (amd-smi metric --power, GPU0)"
amd-smi metric --power -g 0 2>&1 | head -20

SECTION "5. sysfs power-cap nodes (the ground truth)"
for card in /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap; do
    [ -e "$card" ] || continue
    dir=$(dirname "$card")
    echo "node: $card"
    echo "   perms : $(ls -l "$card" 2>&1)"
    echo "   cur   : $(cat "$card" 2>&1) uW"
    echo "   max   : $(cat "$dir/power1_cap_max" 2>&1) uW"
    echo "   min   : $(cat "$dir/power1_cap_min" 2>&1) uW"
    break   # just the first card is enough to learn perms
done

SECTION "6. WRITE attempt A: amd-smi set --power-cap 700 (direct)"
amd-smi set --power-cap 700 -g 0 2>&1; echo "exit=$?"

SECTION "8. WRITE attempt C: rocm-smi --setpoweroverdrive 700 (direct)"
rocm-smi --setpoweroverdrive 700 --autorespond yes 2>&1; echo "exit=$?"

SECTION "10. WRITE attempt E: direct sysfs write (700 W = 700000000 uW)"
FIRST_CAP=$(ls /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap 2>/dev/null | head -1)
if [ -n "$FIRST_CAP" ]; then
    echo "target: $FIRST_CAP"
    echo 700000000 > "$FIRST_CAP" 2>&1 && echo "  DIRECT WRITE OK" || echo "  direct write FAILED (exit=$?)"
    echo "  cur now: $(cat "$FIRST_CAP" 2>&1) uW"
else
    echo "  no power1_cap sysfs node found"
fi

SECTION "11. Verify cap after all attempts"
amd-smi metric --power -g 0 2>&1 | head -8

SECTION "12. Restore default (best-effort)"
DEF=$(cat /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap_max 2>/dev/null | head -1)
[ -n "$DEF" ] && { rocm-smi --resetpoweroverdrive --autorespond yes 2>&1 | head -3; amd-smi reset --power-cap -g 0 2>&1 | head -3; }
echo ""
echo "##### probe done $(date) #####"
