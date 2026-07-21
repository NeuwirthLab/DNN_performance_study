#!/bin/bash -l
# ============================================================================
# probe_powercap_nosudo.slurm  --  power-cap WRITE test, NO sudo
# ----------------------------------------------------------------------------
# The reservation (av30nuci) is supposed to grant power-cap privilege directly.
# This tries every cap-set path WITHOUT sudo, on the reserved node.
#   sbatch power_analysis/probe_powercap_nosudo.slurm
# ============================================================================
#SBATCH --job-name=pcap_nosudo
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
#SBATCH --output=power_analysis/results/pcap_nosudo_%j.out
#SBATCH --export=NONE

export PATH="/opt/rocm/bin:$PATH"
S() { echo ""; echo "==================== $* ===================="; }
echo "##### NO-SUDO power-cap probe on $(hostname)  $(date)  reservation=gaod_powcap #####"
echo "id: $(id)"

S "1. READ current cap (amd-smi metric --power, GPU0)"
amd-smi metric --power -g 0 2>&1 | head -15

S "2. sysfs cap node (ground-truth perms + bounds)"
CAP=$(ls /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap 2>/dev/null | head -1)
if [ -n "$CAP" ]; then
    d=$(dirname "$CAP")
    echo "node : $CAP"
    echo "perms: $(ls -l "$CAP")"
    echo "cur  : $(cat "$CAP") uW"
    echo "max  : $(cat "$d/power1_cap_max" 2>&1) uW"
    echo "min  : $(cat "$d/power1_cap_min" 2>&1) uW"
else
    echo "no power1_cap node found"
fi

S "3. WRITE A: amd-smi set --power-cap 700 (W)"
amd-smi set --power-cap 700 -g 0 2>&1; echo "exit=$?"

S "4. verify after A"
amd-smi metric --power -g 0 2>&1 | grep -iE "cap|socket" | head -6

S "5. WRITE B: /opt/rocm/bin/rocm-smi --setpoweroverdrive 700"
/opt/rocm/bin/rocm-smi --setpoweroverdrive 700 --autorespond yes 2>&1 | head -20; echo "exit=${PIPESTATUS[0]}"

S "6. WRITE C: direct sysfs write 700000000 uW (no sudo)"
if [ -n "$CAP" ]; then
    echo 700000000 > "$CAP" 2>&1 && echo "DIRECT WRITE OK" || echo "direct write FAILED exit=$?"
    echo "cur now: $(cat "$CAP" 2>&1) uW"
fi

S "7. final cap state"
amd-smi metric --power -g 0 2>&1 | grep -iE "cap|socket" | head -6

S "8. restore default cap"
amd-smi reset --power-cap -g 0 2>&1 | head -3
/opt/rocm/bin/rocm-smi --resetpoweroverdrive --autorespond yes 2>&1 | head -3
echo ""
echo "##### done $(date) #####"
