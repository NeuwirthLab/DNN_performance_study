#!/bin/bash -l
#SBATCH --job-name=pytorch_resnet50_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu:mi300x:2
#SBATCH --gres=gpu:mi300x:2
#SBATCH --time=01:00:00
#SBATCH --output=pytorch_resnet50_%j.out

echo "========================================================"
echo " Running PyTorch ResNet50 in AMD Container"
echo " Node: $SLURM_NODELIST"
echo "========================================================"

# 1. Set Temporary Directories (Critical for Apptainer)
export TMPDIR=${HPC_SCRATCH}
export APPTAINER_TMPDIR=${HPC_SCRATCH}

IMAGE_FILE="pytorch_rocm.sif"

# 2. Pull the official ROCm PyTorch image if it doesn't exist
if [ ! -f "$IMAGE_FILE" ]; then
    echo "Building Apptainer image from docker://rocm/pytorch:latest..."
    apptainer build $IMAGE_FILE docker://rocm/pytorch:latest
fi

echo "Image ready. Launching Python benchmark..."

# 3. Execute the python script INSIDE the container
# NOTE: We use --rocm here instead of --nv (which is used for NVIDIA)
apptainer exec --rocm $IMAGE_FILE python pytorch_resnet50_benchmark.py --batch-size 256
