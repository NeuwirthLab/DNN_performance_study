#!/bin/bash -l
#SBATCH --job-name=pcap_confirm
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=acc_short
#SBATCH --gres=gpu:mi300x:8
#SBATCH --time=00:08:00
#SBATCH --reservation=gaod_powcap
#SBATCH --nodelist=gaod[0001-0002]
#SBATCH --output=power_analysis/results/pcap_confirm_%j.out
#SBATCH --export=NONE

echo "##### sudo amd-smi cap-set CONFIRM  job=$SLURM_JOB_ID  $(date) #####"
srun --ntasks=2 --ntasks-per-node=1 /usr/bin/bash -c '
H=$(hostname -s)
CAPW=600
echo ""; echo "==[$H]== BEFORE: cap on all GPUs =="
sudo -n /usr/bin/amd-smi metric -g all --power 2>/dev/null | grep -iE "GPU|cap" | head -20
echo "==[$H]== SET cap=${CAPW}W on ALL GPUs: sudo amd-smi set -g all --power-cap ${CAPW} =="
sudo -n /usr/bin/amd-smi set -g all --power-cap ${CAPW} 2>&1 | head -20; echo "exit=$?"
echo "==[$H]== AFTER set: cap readback =="
sudo -n /usr/bin/amd-smi metric -g all --power 2>/dev/null | grep -iE "GPU|cap" | head -20
echo "==[$H]== RESTORE: sudo amd-smi reset -g all --power-cap =="
sudo -n /usr/bin/amd-smi reset -g all --power-cap 2>&1 | head -8; echo "exit=$?"
echo "==[$H]== FINAL cap readback =="
sudo -n /usr/bin/amd-smi metric -g all --power 2>/dev/null | grep -iE "GPU|cap" | head -20
echo "##### $H done #####"
'
echo "##### done $(date) #####"
