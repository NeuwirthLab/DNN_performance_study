#!/bin/bash -l
#SBATCH --job-name=pcap_sudo
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=acc_short
#SBATCH --gres=gpu:mi300x:1
#SBATCH --time=00:06:00
#SBATCH --reservation=gaod_powcap
#SBATCH --nodelist=gaod[0001-0002]
#SBATCH --output=power_analysis/results/pcap_sudo_%j.out
#SBATCH --export=NONE

echo "##### sudo cap-set probe  job=$SLURM_JOB_ID  $(date) #####"
srun --ntasks=2 --ntasks-per-node=1 /usr/bin/bash -c '
export PATH="/opt/rocm/bin:$PATH"
H=$(hostname -s)
echo ""; echo "==[$H]== groups: $(id -nG) =="
echo "==[$H]== sudo -n -l (non-interactive allowed commands) =="
sudo -n -l 2>&1 | head -30
echo "==[$H]== try: sudo -n rocm-smi --setpoweroverdrive 700 =="
sudo -n /opt/rocm/bin/rocm-smi --setpoweroverdrive 700 --autorespond yes 2>&1 | head -8; echo "exit=${PIPESTATUS[0]}"
echo "==[$H]== readback =="
amd-smi metric --power -g 0 2>&1 | grep -iE "cap|socket" | head -4
echo "==[$H]== restore: sudo -n rocm-smi --resetpoweroverdrive =="
sudo -n /opt/rocm/bin/rocm-smi --resetpoweroverdrive --autorespond yes 2>&1 | head -4; echo "exit=${PIPESTATUS[0]}"
'
echo "##### done $(date) #####"
