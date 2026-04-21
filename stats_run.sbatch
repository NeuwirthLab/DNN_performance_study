#!/bin/bash
# Submit ResNet-50 DALI training jobs sweeping GPU count 1–4 (H100),
# then submit a dependent stats job that extracts images/sec from each run.
#
# Usage: bash stats_run.sbatch

SUBMIT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_TEMPLATE="$SUBMIT_DIR/run_multigpu_tmpdir_dali.slurm"
STATS_CSV="$SUBMIT_DIR/gpu_scaling_imgps.csv"
JOB_IDS=()
JOB_OUTFILES=()

for NUM_GPUS in 1 2 3 4; do
    GLOBAL_BATCH=$((64 * NUM_GPUS))   # 64 samples per GPU
    CPUS=$((8 * NUM_GPUS))            # 8 CPU threads per GPU
    OUTFILE="rn50_dali_gpu${NUM_GPUS}_%j.out"

    JID=$(sbatch \
        --job-name="rn50_dali_gpu${NUM_GPUS}" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="${CPUS}" \
        --gres="gpu:h100:${NUM_GPUS}" \
        --time=2:00:00 \
        --output="${OUTFILE}" \
        --error="rn50_dali_gpu${NUM_GPUS}_%j.err" \
        --export="NONE,NUM_GPUS=${NUM_GPUS},GLOBAL_BATCH=${GLOBAL_BATCH}" \
        "$SLURM_TEMPLATE" \
        | awk '{print $NF}')

    echo "Submitted ${NUM_GPUS} GPU job -> Job ID: $JID  (batch=${GLOBAL_BATCH}, cpus=${CPUS})"
    JOB_IDS+=("$JID")
    # The actual filename replaces %j with the job ID
    JOB_OUTFILES+=("$SUBMIT_DIR/rn50_dali_gpu${NUM_GPUS}_${JID}.out")
done

# Build a colon-separated dependency list
DEPENDENCY=$(IFS=:; echo "${JOB_IDS[*]}")

# Inline the output file paths so the stats job can find them
OUTFILE_LIST=$(IFS=' '; echo "${JOB_OUTFILES[*]}")

echo ""
echo "All training jobs submitted: ${JOB_IDS[*]}"
echo "Submitting stats job dependent on all four completing..."

sbatch --job-name=rn50_dali_stats \
       --nodes=1 --ntasks=1 --cpus-per-task=1 \
       --time=0:05:00 \
       --dependency="afterany:${DEPENDENCY}" \
       --output="$SUBMIT_DIR/rn50_dali_stats_%j.out" \
       --error="$SUBMIT_DIR/rn50_dali_stats_%j.err" \
       --export=NONE \
       --wrap="$(cat <<WRAP
#!/bin/bash
set -euo pipefail

STATS_CSV="$STATS_CSV"
OUTFILES=($OUTFILE_LIST)
GPU_COUNTS=(1 2 3 4)

echo "GPUs,Avg_ImgPerSec,Peak_ImgPerSec" > "\$STATS_CSV"
printf '%-6s  %-18s  %-16s\n' 'GPUs' 'Avg img/s' 'Peak img/s'
printf '%s\n' '----------------------------------------------'

for i in "\${!OUTFILES[@]}"; do
    OUTF="\${OUTFILES[\$i]}"
    GPUS="\${GPU_COUNTS[\$i]}"

    if [ ! -f "\$OUTF" ]; then
        echo "WARNING: output file not found: \$OUTF"
        echo "\$GPUS,N/A,N/A" >> "\$STATS_CSV"
        continue
    fi

    # Epoch summary rows: lines that start with a digit and contain 8 space-separated
    # numeric columns (epoch lr train_loss train_acc val_loss val_acc time img/s).
    # We pick the img/s column (field 8).
    VALS=\$(grep -E '^\s*[0-9]+\s+[0-9]+\.' "\$OUTF" | awk '{print \$8}' | grep -E '^[0-9]+$')

    if [ -z "\$VALS" ]; then
        echo "WARNING: no epoch summary rows found in \$OUTF"
        echo "\$GPUS,N/A,N/A" >> "\$STATS_CSV"
        continue
    fi

    AVG=\$(echo "\$VALS" | awk '{s+=\$1; n++} END {printf "%.0f", s/n}')
    PEAK=\$(echo "\$VALS" | sort -n | tail -1)

    printf '%-6s  %-18s  %-16s\n' "\$GPUS" "\$AVG" "\$PEAK"
    echo "\$GPUS,\$AVG,\$PEAK" >> "\$STATS_CSV"
done

echo ""
echo "Results saved to: \$STATS_CSV"
WRAP
)"

echo "Stats job submitted (runs after all training jobs finish)."
echo "Results will be written to: $STATS_CSV"
