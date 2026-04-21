/*
 * main.cu  –  ResNet-50 demo: random-weight forward pass + timing
 *
 * Usage:
 *   ./resnet50 [batch_size]
 *
 * The program:
 *   1. Builds ResNet-50 and initialises all weights randomly.
 *   2. Allocates a random [N, 3, 224, 224] input on the GPU.
 *   3. Runs one warm-up forward pass (to trigger cuDNN autotuning).
 *   4. Runs `num_runs` timed forward passes and reports throughput.
 *   5. Copies back the softmax output and prints the top-5 class
 *      indices for the first image in the batch.
 */

#include "resnet50.cuh"
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cstdlib>
#include <ctime>

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static void print_device_info()
{
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::cout << "GPU: " << prop.name
              << "  (SM " << prop.major << "." << prop.minor << ")"
              << "  VRAM: " << prop.totalGlobalMem / (1 << 20) << " MB\n";
}

static std::vector<int> top_k(const float* scores, int K, int num_classes)
{
    // Returns indices of the K highest-scoring classes (unsorted).
    std::vector<int> idx(num_classes);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                      [&](int a, int b){ return scores[a] > scores[b]; });
    idx.resize(K);
    return idx;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    srand(42);

    int batch_size  = (argc > 1) ? std::atoi(argv[1]) : 1;
    int num_classes = 1000;
    int num_runs    = 20;

    std::cout << "=== ResNet-50 CUDA/cuDNN demo ===\n";
    print_device_info();
    std::cout << "Batch size : " << batch_size  << "\n";
    std::cout << "Classes    : " << num_classes << "\n\n";

    // ── Build model ───────────────────────────────────────────────────────
    std::cout << "Building ResNet-50 …\n";
    ResNet50 model(num_classes);
    model.InitWeights();
    model.SetTraining(false);   // inference mode for benchmarking
    std::cout << "Done.\n\n";

    // ── Allocate random input [N, 3, 224, 224] on GPU ─────────────────────
    const int input_elems = batch_size * 3 * 224 * 224;
    float* d_input = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, input_elems * sizeof(float)));
    gpu_fill_random(d_input, input_elems, 1.0f);   // random pixel values

    // ── Warm-up (triggers cuDNN algorithm selection) ───────────────────────
    std::cout << "Warm-up pass …\n";
    const float* d_out = model.Forward(d_input, batch_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Done.\n\n";

    // ── Timed runs ────────────────────────────────────────────────────────
    std::cout << "Running " << num_runs << " timed forward passes (epochs) …\n";
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    float total_ms = 0.f;
    for (int r = 0; r < num_runs; ++r) {
        CUDA_CHECK(cudaEventRecord(ev_start));
        d_out = model.Forward(d_input, batch_size);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        total_ms += ms;
    }
    float avg_ms = total_ms / num_runs;
    float fps    = 1000.f * batch_size / avg_ms;

    std::cout << "Average forward time : " << avg_ms  << " ms\n";
    std::cout << "Throughput           : " << fps     << " images/s\n\n";

    // ── Copy back and show top-5 for image 0 ──────────────────────────────
    std::vector<float> h_out(batch_size * num_classes);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                          batch_size * num_classes * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Scores for image 0
    const float* scores0 = h_out.data();
    auto top5 = top_k(scores0, 5, num_classes);

    std::cout << "Top-5 predictions for image 0 (random weights):\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << "  class " << top5[i]
                  << "  prob = " << scores0[top5[i]] << "\n";
    }

    // ── Clean up ──────────────────────────────────────────────────────────
    cudaFree(d_input);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    return 0;
}
