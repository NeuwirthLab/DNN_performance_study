#!/bin/bash -l
#SBATCH --job-name=build_rn50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --account=l0003099
#SBATCH --partition=acc
#SBATCH --gres=gpu:mi300x:1
#SBATCH --time=00:20:00
#SBATCH --output=power_analysis/results/build_%j.out
#SBATCH --error=power_analysis/results/build_%j.err

cd /work/scratch/av30nuci/resent50/performance-analysis-dl-models-resnet50 || exit 1

module --force purge 2>/dev/null; module load cmake 2>/dev/null; true

export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:/opt/amdgpu/lib64:${LD_LIBRARY_PATH:-}"

echo "=== Build train_multigpu (patience=num_epochs fix) ==="
echo "Date : $(date)"
echo "Node : ${SLURM_NODELIST:-?}"
hipcc --version > /tmp/hipcc_ver.txt 2>&1; cat /tmp/hipcc_ver.txt

GFX_ARCH=$(rocminfo 2>/dev/null | awk '/gfx[0-9a-f]+/{for(i=1;i<=NF;i++) if($i~/^gfx/){print $i;exit}}')
[ -z "$GFX_ARCH" ] && GFX_ARCH="gfx942"
echo "GFX target: $GFX_ARCH"

cmake -S hip -B build_profile \
  -DCMAKE_HIP_ARCHITECTURES="$GFX_ARCH" \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_NCCL=ON -DUSE_TURBOJPEG=OFF
RC=$?; echo "cmake RC=$RC"; [ $RC -eq 0 ] || exit $RC

cmake --build build_profile -j16 --target train_multigpu
RC=$?; echo "build RC=$RC"; [ $RC -eq 0 ] || exit $RC

ls -lh build_profile/train_multigpu
echo "=== BUILD OK ==="
