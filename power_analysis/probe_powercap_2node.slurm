#!/bin/bash -l
# ============================================================================
# probe_powercap_2node.slurm  --  MULTI-NODE power-cap WRITE test
# ----------------------------------------------------------------------------
# Verifies whether the new `gaod_powcap` reservation grants cap-set privilege
# on BOTH reserved nodes (gaod0001 + gaod0002) at once. Earlier today the old
# `av30nuci` reservation failed this (NO_PERM; probe 52900932). Each node runs
# the same probe via `srun --ntasks-per-node=1`: read cap -> try set 700 W via
# every mechanism -> read back -> restore default.
#
#   sbatch power_analysis/probe_powercap_2node.slurm
# ============================================================================
#SBATCH --job-name=pcap_2node
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=acc_short
#SBATCH --gres=gpu:mi300x:1
#SBATCH --time=00:10:00
#SBATCH --reservation=gaod_powcap
#SBATCH --nodelist=gaod[0001-0002]
#SBATCH --output=power_analysis/results/pcap_2node_%j.out
#SBATCH --export=NONE

echo "##### 2-NODE power-cap probe  job=$SLURM_JOB_ID  $(date)  reservation=gaod_powcap #####"
echo "nodelist=$SLURM_NODELIST"

# The per-node probe body. Runs once on each allocated node.
srun --ntasks=2 --ntasks-per-node=1 /usr/bin/bash -c '
export PATH="/opt/rocm/bin:$PATH"
H=$(hostname -s)
S() { echo ""; echo "==[$H]== $* =="; }
TARGET_W=700

echo "########## node $H  id=$(id -nG) ##########"

S "1. sysfs cap node (perms + bounds)"
CAP=$(ls /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap 2>/dev/null | head -1)
if [ -n "$CAP" ]; then
    d=$(dirname "$CAP")
    echo "node : $CAP"
    echo "perms: $(ls -l "$CAP")"
    echo "cur  : $(cat "$CAP" 2>&1) uW"
    echo "max  : $(cat "$d/power1_cap_max" 2>&1) uW"
    echo "min  : $(cat "$d/power1_cap_min" 2>&1) uW"
else
    echo "no power1_cap node found"
fi

S "2. READ current cap (amd-smi)"
amd-smi metric --power -g 0 2>&1 | grep -iE "cap|socket" | head -6

S "3. WRITE A: rocm-smi --setpoweroverdrive $TARGET_W"
/opt/rocm/bin/rocm-smi --setpoweroverdrive $TARGET_W --autorespond yes 2>&1 | head -10; echo "exit=${PIPESTATUS[0]}"

S "4. WRITE B: amd-smi set --power-cap $TARGET_W -g 0"
amd-smi set --power-cap $TARGET_W -g 0 2>&1; echo "exit=$?"

S "5. WRITE C: direct sysfs write ${TARGET_W}000000 uW"
if [ -n "$CAP" ]; then
    echo ${TARGET_W}000000 > "$CAP" 2>&1 && echo "DIRECT WRITE OK" || echo "direct write FAILED exit=$?"
    echo "cur now: $(cat "$CAP" 2>&1) uW"
fi

S "6. VERIFY cap after writes"
amd-smi metric --power -g 0 2>&1 | grep -iE "cap|socket" | head -6

S "7. restore default cap"
amd-smi reset --power-cap -g 0 2>&1 | head -3
/opt/rocm/bin/rocm-smi --resetpoweroverdrive --autorespond yes 2>&1 | head -3
echo "##### node $H done #####"
'
echo ""
echo "##### 2-node probe complete  $(date) #####"
