#!/bin/bash -l
#SBATCH --job-name=gpu_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:2
#SBATCH --time=00:05:00
#SBATCH --output=gpu_test_%j.out
#SBATCH --error=gpu_test_%j.err
#SBATCH --export=ALL

echo "=== Simple GPU Test ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo ""

module purge
module load cuda/12.6.2

nvidia-smi
echo ""

cat > "$TMPDIR/test.cu" << 'CUEOF'
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vector_add(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int nGPUs;
    cudaGetDeviceCount(&nGPUs);
    printf("GPUs found: %d\n\n", nGPUs);

    for (int g = 0; g < nGPUs; g++) {
        cudaSetDevice(g);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, g);
        printf("GPU %d: %s | %.1f GB | SM count: %d\n",
               g, prop.name, prop.totalGlobalMem/1e9, prop.multiProcessorCount);

        int N = 1 << 24;  // 16M elements
        size_t bytes = N * sizeof(float);
        float *h_a = (float*)malloc(bytes);
        float *h_b = (float*)malloc(bytes);
        float *h_c = (float*)malloc(bytes);
        for (int i = 0; i < N; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

        float *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
        cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

        // Warmup
        vector_add<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();

        // Benchmark
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int i = 0; i < 1000; i++)
            vector_add<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("  1000 kernel launches: %.1f ms (%.3f ms avg)\n", ms, ms/1000);

        // Verify
        cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
        printf("  Result check: c[0]=%.1f (expected 3.0) -> %s\n\n",
               h_c[0], h_c[0]==3.0f ? "PASS" : "FAIL");

        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        free(h_a); free(h_b); free(h_c);
    }
    printf("Done!\n");
    return 0;
}
CUEOF

nvcc "$TMPDIR/test.cu" -o "$TMPDIR/test"
"$TMPDIR/test"

echo ""
echo "Finished at $(date)"
EOF
