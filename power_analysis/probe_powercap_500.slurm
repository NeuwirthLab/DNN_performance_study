#!/bin/bash -l
# ============================================================================
# probe_powercap_500.slurm -- does the NEW gaod_powcap reservation grant cap-set?
# Sets the MI300X power cap to 500 W via every mechanism, reads it back to see
# if it actually took, then restores the default. Quick yes/no answer.
#   sbatch power_analysis/probe_powercap_500.slurm
# ============================================================================
#SBATCH --job-name=probe_pcap500
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
#SBATCH --output=power_analysis/results/probe_pcap500_%j.out
#SBATCH --export=NONE

export PATH="/opt/rocm/bin:$PATH"
SECTION() { echo ""; echo "==================== $* ===================="; }
CAP_W=500
CAP_UW=500000000
read_cap() { amd-smi metric --power -g 0 2>&1 | grep -iE 'cap|socket|current' | head -8; }

echo "##### 500W cap probe on $(hostname)  $(date)  (reservation gaod_powcap) #####"
echo "whoami=$(whoami)  groups=$(groups)"

SECTION "1. READ cap BEFORE"
read_cap
FIRST_CAP=$(ls /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap 2>/dev/null | head -1)
echo "sysfs node: $FIRST_CAP"
echo "  perms: $(ls -l "$FIRST_CAP" 2>&1)"
echo "  cur  : $(cat "$FIRST_CAP" 2>&1) uW   max: $(cat "$(dirname "$FIRST_CAP")/power1_cap_max" 2>&1) uW"

SECTION "2. SET 500 W -- attempt A: amd-smi set --power-cap"
amd-smi set --power-cap $CAP_W -g 0 2>&1; echo "exit=$?"

SECTION "3. SET 500 W -- attempt B: rocm-smi --setpoweroverdrive"
rocm-smi --setpoweroverdrive $CAP_W --autorespond yes 2>&1 | head -5; echo "exit=${PIPESTATUS[0]}"

SECTION "4. SET 500 W -- attempt C: direct sysfs write ($CAP_UW uW)"
# NO sudo: we are normal users on an HPC system, not admins.
if [ -n "$FIRST_CAP" ]; then
    echo $CAP_UW > "$FIRST_CAP" 2>&1 && echo "  DIRECT WRITE OK" || echo "  direct write FAILED (exit=$?)"
fi

SECTION "5. VERIFY -- did the cap actually become 500 W?"
echo "sysfs cur now: $(cat "$FIRST_CAP" 2>&1) uW   (expect 500000000 if it took)"
read_cap

SECTION "6. RESTORE default cap"
rocm-smi --resetpoweroverdrive --autorespond yes 2>&1 | head -3
amd-smi reset --power-cap -g 0 2>&1 | head -3
echo "sysfs cur after restore: $(cat "$FIRST_CAP" 2>&1) uW"
echo ""
echo "##### probe done $(date) #####"
