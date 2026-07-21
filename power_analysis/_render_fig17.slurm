#!/bin/bash -l
#SBATCH --job-name=render_fig17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --account=l0003099
#SBATCH --partition=deflt_short
#SBATCH --time=00:15:00
#SBATCH --output=/work/scratch/av30nuci/resent50/performance-analysis-dl-models-resnet50/power_analysis/_render_fig17_%j.out
#SBATCH --error=/work/scratch/av30nuci/resent50/performance-analysis-dl-models-resnet50/power_analysis/_render_fig17_%j.out
#SBATCH --export=ALL

set -uo pipefail
ROOT=/work/scratch/av30nuci/resent50/performance-analysis-dl-models-resnet50
PY=/usr/bin/python3.11

echo "=== render fig17 on $(hostname) $(date) ==="
$PY -c "import matplotlib, sys; print('python', sys.version.split()[0], 'matplotlib', matplotlib.__version__)" \
    || { echo "FAIL: matplotlib not available for $PY on this node"; exit 3; }

echo "--- [1/3] energy-model prediction ---"
cd "$ROOT/power_analysis" && $PY predict_4gpu_energy.py
echo "  predict exit=$?"

echo "--- [2/3] regenerate master figures (incl rn50_17) ---"
cd "$ROOT/plot_bundle/mi300x_master" && MPLBACKEND=Agg $PY resnet50_mi300x_plots.py 2>&1 | tail -4
echo "  plot exit=${PIPESTATUS[0]}"

SRC="$ROOT/plot_bundle/mi300x_master/plots/amd_mi300x/rn50_17_energy_4gpu_amd_mi300x"
DST="$ROOT/power_analysis/paper_submission/fig_validation_4gpu"
echo "--- [3/3] copy to paper ---"
if [ -f "$SRC.pdf" ]; then
    cp -f "$SRC.pdf" "$DST.pdf" && cp -f "$SRC.png" "$DST.png"
    echo "  copied -> $DST.pdf (+ .png)"
    ls -la "$DST.pdf" "$DST.png"
else
    echo "FAIL: $SRC.pdf not produced"; exit 4
fi
echo "=== done $(date) ==="
