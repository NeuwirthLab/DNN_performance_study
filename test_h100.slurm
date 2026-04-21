#!/bin/bash -l
#SBATCH --job-name=h100_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:2
#SBATCH --time=00:05:00
#SBATCH --output=h100_test_%j.out
#SBATCH --error=h100_test_%j.err
#SBATCH --export=ALL

echo "=== H100 GPU + NCCL Quick Test ==="
echo "Job: $SLURM_JOB_ID | Node: $SLURM_NODELIST | $(date)"
echo ""

module purge
module load cuda/12.6.2
module load nvhpc

nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Find NCCL
NCCL_H=$(find "$NVHPC_ROOT" -name "nccl.h" 2>/dev/null | sort -r | head -1)
NCCL_ROOT=$(dirname $(dirname "$NCCL_H"))
export LD_LIBRARY_PATH="$NCCL_ROOT/lib:$LD_LIBRARY_PATH"
echo "NCCL: $NCCL_ROOT"

# Compile
cat > "$TMPDIR/test.cu" << 'CU'
#include <stdio.h>
#include <cuda_runtime.h>
#include <nccl.h>

int main() {
    int n;
    cudaGetDeviceCount(&n);
    printf("GPUs: %d\n", n);
    for (int i = 0; i < n; i++) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, i);
        printf("  [%d] %s %.1fGB\n", i, p.name, p.totalGlobalMem/1e9);
    }

    // NCCL init
    ncclComm_t comms[8];
    int devs[8];
    for (int i = 0; i < n; i++) devs[i] = i;
    ncclCommInitAll(comms, n, devs);
    printf("\nNCCL init: OK (%d GPUs)\n", n);

    // AllReduce
    int N = 1<<20;
    float *send[8], *recv[8];
    cudaStream_t s[8];
    for (int i = 0; i < n; i++) {
        cudaSetDevice(i);
        cudaMalloc(&send[i], N*4);
        cudaMalloc(&recv[i], N*4);
        cudaStreamCreate(&s[i]);
    }

    // Warmup
    for (int w = 0; w < 10; w++) {
        ncclGroupStart();
        for (int i = 0; i < n; i++)
            ncclAllReduce(send[i], recv[i], N, ncclFloat, ncclSum, comms[i], s[i]);
        ncclGroupEnd();
    }
    for (int i = 0; i < n; i++) { cudaSetDevice(i); cudaStreamSynchronize(s[i]); }

    // Benchmark
    cudaEvent_t t0, t1;
    cudaSetDevice(0);
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0, s[0]);
    for (int it = 0; it < 200; it++) {
        ncclGroupStart();
        for (int i = 0; i < n; i++)
            ncclAllReduce(send[i], recv[i], N, ncclFloat, ncclSum, comms[i], s[i]);
        ncclGroupEnd();
    }
    cudaEventRecord(t1, s[0]);
    for (int i = 0; i < n; i++) { cudaSetDevice(i); cudaStreamSynchronize(s[i]); }

    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("AllReduce: %.2f ms avg | %.2f GB/s\n", ms/200, (N*4.0/1e9)/(ms/200/1000));
    printf("\nRESULT: PASS\n");

    for (int i = 0; i < n; i++) {
        cudaSetDevice(i);
        cudaFree(send[i]); cudaFree(recv[i]);
        cudaStreamDestroy(s[i]); ncclCommDestroy(comms[i]);
    }
    return 0;
}
CU

nvcc "$TMPDIR/test.cu" -o "$TMPDIR/test" -I"$NCCL_ROOT/include" -L"$NCCL_ROOT/lib" -lnccl
if [ $? -ne 0 ]; then echo "COMPILE FAILED"; exit 1; fi

echo ""
"$TMPDIR/test"
echo ""
echo "Done: $(date)"