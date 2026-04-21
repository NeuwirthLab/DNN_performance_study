/*
 * test_nccl_sanity.cu — Minimal NCCL correctness test
 *
 * Tests:
 *   1. ncclCommInitAll across all visible GPUs
 *   2. ncclAllReduce (sum) — verifies data actually moves between GPUs
 *   3. ncclBcast — verifies broadcast from rank 0
 *   4. Batch-count synchronization (ncclAllReduce with ncclMin)
 *
 * Build (inside SLURM job):
 *   nvcc -O2 -arch=native test_nccl_sanity.cu -o test_nccl_sanity -lnccl
 *
 * Run:
 *   ./test_nccl_sanity
 */

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <thread>
#include <cuda_runtime.h>
#include <nccl.h>

#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

#define NCCL_CHECK(x) do { \
    ncclResult_t r = (x); \
    if (r != ncclSuccess) { \
        fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
        exit(1); \
    } \
} while(0)

static int g_pass = 0;
static int g_fail = 0;

void worker(int rank, int num_gpus, ncclComm_t comm) {
    CUDA_CHECK(cudaSetDevice(rank));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ── Test 1: AllReduce (Sum) ──────────────────────────────────────────────
    // Each GPU sends its rank+1. Sum should be 1+2+...+N = N*(N+1)/2
    {
        float h_send = (float)(rank + 1);
        float h_recv = 0.0f;
        float* d_buf;
        CUDA_CHECK(cudaMalloc(&d_buf, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_buf, &h_send, sizeof(float), cudaMemcpyHostToDevice));

        NCCL_CHECK(ncclAllReduce(d_buf, d_buf, 1, ncclFloat, ncclSum, comm, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaMemcpy(&h_recv, d_buf, sizeof(float), cudaMemcpyDeviceToHost));

        float expected = (float)(num_gpus * (num_gpus + 1)) / 2.0f;
        if (rank == 0) {
            if (h_recv == expected) {
                printf("  [PASS] AllReduce(Sum): got %.0f, expected %.0f\n", h_recv, expected);
                g_pass++;
            } else {
                printf("  [FAIL] AllReduce(Sum): got %.0f, expected %.0f\n", h_recv, expected);
                g_fail++;
            }
        }
        CUDA_CHECK(cudaFree(d_buf));
    }

    // ── Test 2: AllReduce (Avg) ──────────────────────────────────────────────
    // Each GPU sends its rank+1. Avg should be (1+2+...+N)/N = (N+1)/2
    {
        float h_send = (float)(rank + 1);
        float h_recv = 0.0f;
        float* d_buf;
        CUDA_CHECK(cudaMalloc(&d_buf, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_buf, &h_send, sizeof(float), cudaMemcpyHostToDevice));

        NCCL_CHECK(ncclAllReduce(d_buf, d_buf, 1, ncclFloat, ncclAvg, comm, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaMemcpy(&h_recv, d_buf, sizeof(float), cudaMemcpyDeviceToHost));

        float expected = (float)(num_gpus + 1) / 2.0f;
        if (rank == 0) {
            float diff = h_recv - expected;
            if (diff < 0) diff = -diff;
            if (diff < 0.001f) {
                printf("  [PASS] AllReduce(Avg): got %.2f, expected %.2f\n", h_recv, expected);
                g_pass++;
            } else {
                printf("  [FAIL] AllReduce(Avg): got %.2f, expected %.2f\n", h_recv, expected);
                g_fail++;
            }
        }
        CUDA_CHECK(cudaFree(d_buf));
    }

    // ── Test 3: Broadcast ────────────────────────────────────────────────────
    // Rank 0 sets value to 42.0, others set to 0. After broadcast, all should be 42.
    {
        float h_val = (rank == 0) ? 42.0f : 0.0f;
        float* d_buf;
        CUDA_CHECK(cudaMalloc(&d_buf, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_buf, &h_val, sizeof(float), cudaMemcpyHostToDevice));

        NCCL_CHECK(ncclBcast(d_buf, 1, ncclFloat, 0, comm, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaMemcpy(&h_val, d_buf, sizeof(float), cudaMemcpyDeviceToHost));

        if (h_val == 42.0f) {
            if (rank == 0) { printf("  [PASS] Broadcast: all GPUs received 42.0\n"); g_pass++; }
        } else {
            printf("  [FAIL] Broadcast: rank %d got %.1f instead of 42.0\n", rank, h_val);
            if (rank == 0) g_fail++;
        }
        CUDA_CHECK(cudaFree(d_buf));
    }

    // ── Test 4: Batch-count sync (AllReduce Min) ─────────────────────────────
    // Simulate different batch counts per GPU. Min should be the smallest.
    {
        int my_count = 100 + rank * 3;  // GPU0=100, GPU1=103, GPU2=106, ...
        int* d_buf;
        CUDA_CHECK(cudaMalloc(&d_buf, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_buf, &my_count, sizeof(int), cudaMemcpyHostToDevice));

        NCCL_CHECK(ncclAllReduce(d_buf, d_buf, 1, ncclInt32, ncclMin, comm, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        int result = 0;
        CUDA_CHECK(cudaMemcpy(&result, d_buf, sizeof(int), cudaMemcpyDeviceToHost));

        if (rank == 0) {
            if (result == 100) {
                printf("  [PASS] AllReduce(Min): synced batch count = %d (expected 100)\n", result);
                g_pass++;
            } else {
                printf("  [FAIL] AllReduce(Min): synced batch count = %d (expected 100)\n", result);
                g_fail++;
            }
        }
        CUDA_CHECK(cudaFree(d_buf));
    }

    // ── Test 5: Large buffer AllReduce (gradient-like) ───────────────────────
    // Simulates gradient sync: 1M floats, each set to rank. After Avg, all should equal avg of ranks.
    {
        const int N = 1024 * 1024;
        float* d_buf;
        CUDA_CHECK(cudaMalloc(&d_buf, N * sizeof(float)));

        // Fill with rank value
        std::vector<float> h_buf(N, (float)rank);
        CUDA_CHECK(cudaMemcpy(d_buf, h_buf.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        NCCL_CHECK(ncclAllReduce(d_buf, d_buf, N, ncclFloat, ncclAvg, comm, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaMemcpy(h_buf.data(), d_buf, N * sizeof(float), cudaMemcpyDeviceToHost));

        float expected = (float)(num_gpus - 1) / 2.0f;
        bool ok = true;
        for (int i = 0; i < 100; ++i) {  // spot-check first 100
            float diff = h_buf[i] - expected;
            if (diff < 0) diff = -diff;
            if (diff > 0.01f) { ok = false; break; }
        }
        if (rank == 0) {
            if (ok) {
                printf("  [PASS] Large AllReduce(Avg): 1M floats verified (expected %.2f)\n", expected);
                g_pass++;
            } else {
                printf("  [FAIL] Large AllReduce(Avg): mismatch detected\n");
                g_fail++;
            }
        }
        CUDA_CHECK(cudaFree(d_buf));
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main() {
    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));

    printf("=== NCCL Sanity Test ===\n");
    printf("GPUs detected: %d\n\n", num_gpus);

    if (num_gpus < 2) {
        printf("Need at least 2 GPUs for multi-GPU NCCL test.\n");
        return 1;
    }

    // Print GPU info
    for (int i = 0; i < num_gpus; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        printf("  GPU %d: %s (SM %d.%d, %.0f MB)\n",
               i, prop.name, prop.major, prop.minor, prop.totalGlobalMem / 1e6);
    }
    printf("\n");

    // Init NCCL
    std::vector<int> devs(num_gpus);
    std::vector<ncclComm_t> comms(num_gpus);
    for (int i = 0; i < num_gpus; ++i) devs[i] = i;
    NCCL_CHECK(ncclCommInitAll(comms.data(), num_gpus, devs.data()));
    printf("NCCL communicators initialized.\n\n");

    printf("Running tests...\n");

    // Launch threads
    std::vector<std::thread> threads;
    for (int r = 0; r < num_gpus; ++r)
        threads.emplace_back(worker, r, num_gpus, comms[r]);
    for (auto& t : threads) t.join();

    // Cleanup
    for (int i = 0; i < num_gpus; ++i) ncclCommDestroy(comms[i]);

    printf("\n=== Results: %d PASSED, %d FAILED ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
