#!/bin/bash -l
#SBATCH --job-name=test_nccl
#SBATCH --output=test_nccl_%j.out
#SBATCH --error=test_nccl_%j.err

#SBATCH --gres=gpu:h100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

#SBATCH --time=00:10:00

echo "=== NCCL Test Script ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo ""

# ---- Load modules ----
echo "Loading modules..."

#module load cuda/12.6.1
module load cuda/12.6.2
module load cudnn/9.8.0.87-12
#module load cudnn/9.2.0.82-12.x  
module load cmake
module load nvhpc
module load user-spack/0.19.1 
#module load gcc/12.1.0
echo "CUDA: $(nvcc --version | head -1)"
echo ""
# Explicitly set the NCCL_ROOT to the standalone SPACK path
NCCL_ROOT="/apps/SPACK/0.23.0/opt/linux-almalinux9-x86_64_v4/gcc-11.4.1/nccl-2.22.3-1-eu6kb6f7p3xfp7py2fgc7q4vodhmt2yw"

echo "=== Module Check ==="
module list
echo ""

# ---- Verify NCCL paths ----
echo "=== NCCL Path Verification ==="
echo "NCCL_ROOT=$NCCL_ROOT"
echo ""

echo "Checking nccl.h:"
ls -la "$NCCL_ROOT/include/nccl.h" 2>&1
echo ""

echo "Checking libnccl.so:"
ls -la "$NCCL_ROOT/lib/libnccl.so" 2>&1
echo ""

# ---- Check GPUs ----
echo "=== GPU Check ==="
nvidia-smi
echo ""

# ---- Compile NCCL test program ----
echo "=== Compiling NCCL Test ==="
NCCL_TEST_DIR=$(mktemp -d)

cat > "$NCCL_TEST_DIR/nccl_test.cu" << 'CUDA_EOF'
#include <stdio.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define CUDA_CHECK(cmd) do { \
    cudaError_t e = cmd; \
    if (e != cudaSuccess) { \
        printf("CUDA error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        return 1; \
    } \
} while(0)

#define NCCL_CHECK(cmd) do { \
    ncclResult_t r = cmd; \
    if (r != ncclSuccess) { \
        printf("NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
        return 1; \
    } \
} while(0)

int main() {
    int nDevices;
    CUDA_CHECK(cudaGetDeviceCount(&nDevices));
    printf("Found %d CUDA devices\n\n", nDevices);

    if (nDevices < 2) {
        printf("WARNING: Need at least 2 GPUs for NCCL test\n");
    }

    // Print NCCL version
    int version;
    NCCL_CHECK(ncclGetVersion(&version));
    printf("NCCL Version: %d.%d.%d\n\n", version / 10000, (version % 10000) / 100, version % 100);

    // Create communicator
    ncclComm_t* comms = (ncclComm_t*)malloc(nDevices * sizeof(ncclComm_t));
    int* devList = (int*)malloc(nDevices * sizeof(int));
    for (int i = 0; i < nDevices; i++) devList[i] = i;

    printf("Initializing NCCL communicator for %d GPUs...\n", nDevices);
    NCCL_CHECK(ncclCommInitAll(comms, nDevices, devList));
    printf("NCCL communicator initialized successfully!\n\n");

    // Allocate buffers and do AllReduce test
    int dataSize = 1024 * 1024;  // 1M elements
    float** sendBuffs = (float**)malloc(nDevices * sizeof(float*));
    float** recvBuffs = (float**)malloc(nDevices * sizeof(float*));
    cudaStream_t* streams = (cudaStream_t*)malloc(nDevices * sizeof(cudaStream_t));

    for (int i = 0; i < nDevices; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMalloc(&sendBuffs[i], dataSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&recvBuffs[i], dataSize * sizeof(float)));
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        // Fill with value (i+1)
        float val = (float)(i + 1);
        CUDA_CHECK(cudaMemset(sendBuffs[i], 0, dataSize * sizeof(float)));
    }

    printf("Running NCCL AllReduce test (%d floats across %d GPUs)...\n", dataSize, nDevices);

    // Group AllReduce
    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < nDevices; i++) {
        NCCL_CHECK(ncclAllReduce(sendBuffs[i], recvBuffs[i], dataSize, ncclFloat, ncclSum, comms[i], streams[i]));
    }
    NCCL_CHECK(ncclGroupEnd());

    // Sync
    for (int i = 0; i < nDevices; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    printf("NCCL AllReduce completed successfully!\n\n");

    // Cleanup
    for (int i = 0; i < nDevices; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaFree(sendBuffs[i]));
        CUDA_CHECK(cudaFree(recvBuffs[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        ncclCommDestroy(comms[i]);
    }

    free(comms);
    free(devList);
    free(sendBuffs);
    free(recvBuffs);
    free(streams);

    printf("=== ALL NCCL TESTS PASSED ===\n");
    return 0;
}
CUDA_EOF

nvcc "$NCCL_TEST_DIR/nccl_test.cu" \
    -o "$NCCL_TEST_DIR/nccl_test" \
    -I"$NCCL_ROOT/include" \
    -L"$NCCL_ROOT/lib" \
    -lnccl \
    -lcudart \
    -std=c++11

if [ $? -ne 0 ]; then
    echo "FAILED: Could not compile NCCL test"
    exit 1
fi
echo "Compilation successful!"
echo ""

# ---- Run test ----
echo "=== Running NCCL Test ==="
export LD_LIBRARY_PATH="$NCCL_ROOT/lib:$LD_LIBRARY_PATH"
"$NCCL_TEST_DIR/nccl_test"
EXIT_CODE=$?

# Cleanup
rm -rf "$NCCL_TEST_DIR"

echo ""
echo "=== Test finished with exit code: $EXIT_CODE ==="
echo "Date: $(date)"
exit $EXIT_CODE
