/*
 * benchmark.cu  –  Comprehensive ResNet-50 ML benchmark
 *
 * Usage:
 *   ./benchmark [max_batch_size]
 *
 * Reports:
 *   - GPU hardware info
 *   - Model statistics (parameters, theoretical FLOPS)
 *   - Per-layer profiling with timing breakdown
 *   - Kernel-category breakdown (Conv/GEMM, BN, ReLU, Pool, FC)
 *   - Batch-size scaling (throughput, latency, achieved TFLOPS)
 *   - Latency percentiles (p50, p90, p99, min, max)
 *   - GPU memory usage report
 *
 * Compile:
 *   nvcc -O2 -arch=sm_75 -std=c++17 benchmark.cu resnet50.cu -o benchmark -lcublas
 */

#include "resnet50.cuh"
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <cstdio>
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static void print_separator(int width = 70)
{
    for (int i = 0; i < width; ++i) putchar('-');
    putchar('\n');
}

static void get_mem(size_t& free_mb, size_t& total_mb)
{
    size_t f, t;
    CUDA_CHECK(cudaMemGetInfo(&f, &t));
    free_mb  = f / (1 << 20);
    total_mb = t / (1 << 20);
}

static std::string flops_str(double flops)
{
    if (flops >= 1e12) { char buf[32]; snprintf(buf, 32, "%.2f TFLOPS", flops / 1e12); return buf; }
    if (flops >= 1e9)  { char buf[32]; snprintf(buf, 32, "%.2f GFLOPS", flops / 1e9);  return buf; }
    if (flops >= 1e6)  { char buf[32]; snprintf(buf, 32, "%.2f MFLOPS", flops / 1e6);  return buf; }
    char buf[32]; snprintf(buf, 32, "%.0f FLOPS", flops); return buf;
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. GPU Info
// ─────────────────────────────────────────────────────────────────────────────

static void print_gpu_info()
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    double mem_bw = (double)prop.memoryBusWidth / 8.0 * prop.memoryClockRate * 2.0 * 1e-6;
    double peak_tflops = (double)prop.multiProcessorCount *
                         // 128 FP32 cores per SM is typical; varies by arch
                         128.0 * prop.clockRate * 2.0 * 1e-9;

    printf("\n=== GPU Hardware ===\n");
    printf("Device          : %s\n", prop.name);
    printf("Compute Cap.    : %d.%d\n", prop.major, prop.minor);
    printf("SMs             : %d\n", prop.multiProcessorCount);
    printf("GPU Clock       : %d MHz\n", prop.clockRate / 1000);
    printf("Memory Clock    : %d MHz\n", prop.memoryClockRate / 1000);
    printf("Memory Bus      : %d-bit\n", prop.memoryBusWidth);
    printf("Memory BW (est) : %.0f GB/s\n", mem_bw);
    printf("VRAM            : %zu MB\n", prop.totalGlobalMem / (1 << 20));
    printf("L2 Cache        : %d KB\n", prop.l2CacheSize / 1024);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Model Stats
// ─────────────────────────────────────────────────────────────────────────────

static void print_model_stats(const ResNet50& model, int num_classes)
{
    size_t params = model.ParamCount();
    double flops1 = ResNet50::ResNet50FLOPS(1, num_classes);

    printf("\n=== Model Statistics ===\n");
    printf("Architecture    : ResNet-50 (cuBLAS, no cuDNN)\n");
    printf("Parameters      : %zu (%.1f M)\n", params, params / 1e6);
    printf("FLOPs (batch=1) : %s\n", flops_str(flops1).c_str());
    printf("Input shape     : [N, 3, 224, 224]\n");
    printf("Output          : [N, %d] (softmax)\n", num_classes);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Per-layer profiling
// ─────────────────────────────────────────────────────────────────────────────

static void print_layer_profile(ResNet50& model, float* d_input, int batch)
{
    printf("\n=== Per-Layer Profiling (batch=%d) ===\n", batch);
    print_separator();
    printf("%-20s %10s %8s %14s\n", "Stage", "Time (ms)", "   %  ", "GFLOPS");
    print_separator();

    std::vector<LayerProfile> profile;

    // Warm-up
    model.ForwardProfile(d_input, batch, profile);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    model.ForwardProfile(d_input, batch, profile);
    CUDA_CHECK(cudaDeviceSynchronize());

    float total_ms = 0.f;
    for (auto& p : profile) total_ms += p.time_ms;

    for (auto& p : profile) {
        float pct = total_ms > 0 ? 100.f * p.time_ms / total_ms : 0.f;
        double gflops = p.time_ms > 0 ? p.flops / (p.time_ms * 1e6) : 0.0;
        printf("%-20s %10.3f %7.1f%% %14.1f\n",
               p.name.c_str(), p.time_ms, pct, gflops);
    }

    print_separator();
    double total_flops = ResNet50::ResNet50FLOPS(batch, 1000);
    double achieved_tflops = total_ms > 0 ? total_flops / (total_ms * 1e9) : 0.0;
    printf("%-20s %10.3f %7s  %14.1f\n",
           "TOTAL", total_ms, "100%", achieved_tflops * 1000.0);

    // Aggregate by category
    printf("\n=== Kernel Category Breakdown ===\n");
    print_separator();
    printf("%-20s %10s %8s\n", "Category", "Time (ms)", "   %  ");
    print_separator();

    float t_conv = 0, t_pool = 0, t_head = 0, t_other = 0;
    for (auto& p : profile) {
        if (p.name.find("Conv1") != std::string::npos ||
            p.name.find("Layer") != std::string::npos)
            t_conv += p.time_ms;
        else if (p.name.find("Pool") != std::string::npos)
            t_pool += p.time_ms;
        else if (p.name.find("FC") != std::string::npos ||
                 p.name.find("Softmax") != std::string::npos)
            t_head += p.time_ms;
        else
            t_other += p.time_ms;
    }
    auto print_cat = [&](const char* name, float t) {
        printf("%-20s %10.3f %7.1f%%\n", name, t,
               total_ms > 0 ? 100.f * t / total_ms : 0.f);
    };
    print_cat("Conv+BN+ReLU", t_conv);
    print_cat("Pooling", t_pool);
    print_cat("FC+Softmax", t_head);
    if (t_other > 0) print_cat("Other", t_other);
    print_separator();
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Batch-size scaling
// ─────────────────────────────────────────────────────────────────────────────

static void print_batch_scaling(int max_batch, int num_classes)
{
    printf("\n=== Batch-Size Scaling ===\n");
    print_separator();
    printf("%6s %10s %12s %12s %10s\n",
           "Batch", "Lat (ms)", "Thput (img/s)", "TFLOPS", "GPU Mem");
    print_separator();

    int batches[] = {1, 2, 4, 8, 16, 32, 64};
    const int num_runs = 20;

    for (int bi = 0; bi < 7; ++bi) {
        int bs = batches[bi];
        if (bs > max_batch) break;

        // Check available memory first
        size_t free_mem, total_mem;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        size_t input_bytes = (size_t)bs * 3 * 224 * 224 * sizeof(float);
        if (input_bytes > free_mem / 2) {
            printf("%6d %10s\n", bs, "OOM");
            break;
        }

        try {
            ResNet50 model(num_classes);
            model.InitWeights();
            model.SetTraining(false);

            float* d_input = nullptr;
            CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
            gpu_fill_random(d_input, bs * 3 * 224 * 224, 1.0f);

            // Warm-up
            model.Forward(d_input, bs);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Timed runs
            cudaEvent_t ev_start, ev_stop;
            CUDA_CHECK(cudaEventCreate(&ev_start));
            CUDA_CHECK(cudaEventCreate(&ev_stop));

            float total_ms = 0.f;
            for (int r = 0; r < num_runs; ++r) {
                CUDA_CHECK(cudaEventRecord(ev_start));
                model.Forward(d_input, bs);
                CUDA_CHECK(cudaEventRecord(ev_stop));
                CUDA_CHECK(cudaEventSynchronize(ev_stop));
                float ms = 0.f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
                total_ms += ms;
            }
            float avg_ms = total_ms / num_runs;
            float throughput = 1000.f * bs / avg_ms;
            double total_flops = ResNet50::ResNet50FLOPS(bs, num_classes);
            double tflops = total_flops / (avg_ms * 1e9);

            size_t used_mb;
            {
                size_t f2, t2;
                CUDA_CHECK(cudaMemGetInfo(&f2, &t2));
                used_mb = (t2 - f2) / (1 << 20);
            }

            printf("%6d %10.2f %12.1f %12.3f %8zu MB\n",
                   bs, avg_ms, throughput, tflops, used_mb);

            CUDA_CHECK(cudaEventDestroy(ev_start));
            CUDA_CHECK(cudaEventDestroy(ev_stop));
            cudaFree(d_input);
        } catch (const std::exception& e) {
            printf("%6d %10s  (%s)\n", bs, "FAILED", e.what());
            // Reset device after OOM
            cudaDeviceReset();
            break;
        }
    }
    print_separator();
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Latency percentiles
// ─────────────────────────────────────────────────────────────────────────────

static void print_latency_percentiles(ResNet50& model, float* d_input, int batch)
{
    const int num_runs = 100;

    printf("\n=== Latency Distribution (batch=%d, %d runs) ===\n", batch, num_runs);

    // Warm-up
    for (int i = 0; i < 5; ++i) {
        model.Forward(d_input, batch);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    std::vector<float> latencies(num_runs);
    for (int r = 0; r < num_runs; ++r) {
        CUDA_CHECK(cudaEventRecord(ev_start));
        model.Forward(d_input, batch);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        CUDA_CHECK(cudaEventElapsedTime(&latencies[r], ev_start, ev_stop));
    }

    std::sort(latencies.begin(), latencies.end());
    float mean = std::accumulate(latencies.begin(), latencies.end(), 0.f) / num_runs;
    float stddev = 0.f;
    for (float l : latencies) stddev += (l - mean) * (l - mean);
    stddev = sqrtf(stddev / num_runs);

    print_separator();
    printf("Min      : %8.3f ms\n", latencies[0]);
    printf("p50      : %8.3f ms\n", latencies[num_runs / 2]);
    printf("p90      : %8.3f ms\n", latencies[(int)(num_runs * 0.9)]);
    printf("p95      : %8.3f ms\n", latencies[(int)(num_runs * 0.95)]);
    printf("p99      : %8.3f ms\n", latencies[(int)(num_runs * 0.99)]);
    printf("Max      : %8.3f ms\n", latencies[num_runs - 1]);
    printf("Mean     : %8.3f ms\n", mean);
    printf("Std Dev  : %8.3f ms\n", stddev);
    printf("Thput    : %8.1f images/s (at mean)\n", 1000.f * batch / mean);
    print_separator();

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Memory report
// ─────────────────────────────────────────────────────────────────────────────

static void print_memory_report(ResNet50& model, float* d_input, int batch)
{
    printf("\n=== GPU Memory Report ===\n");
    print_separator();

    size_t free_before, total;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total));
    size_t used = (total - free_before) / (1 << 20);

    printf("Total VRAM      : %zu MB\n", total / (1 << 20));
    printf("Used (model+inp): %zu MB\n", used);
    printf("Free            : %zu MB\n", free_before / (1 << 20));

    // Run forward to see peak usage
    model.Forward(d_input, batch);
    CUDA_CHECK(cudaDeviceSynchronize());

    size_t free_after;
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total));
    size_t peak_used = (total - free_after) / (1 << 20);

    printf("Peak (after fwd): %zu MB\n", peak_used);
    printf("Fwd workspace   : %zu MB\n", peak_used - used);
    print_separator();
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    srand(42);

    int max_batch   = (argc > 1) ? std::atoi(argv[1]) : 32;
    int num_classes = 1000;
    int profile_batch = 1;

    printf("============================================================\n");
    printf("       ResNet-50 CUDA Benchmark Suite\n");
    printf("============================================================\n");

    // 1. GPU info
    print_gpu_info();

    // 2. Build model for profiling
    ResNet50 model(num_classes);
    model.InitWeights();
    model.SetTraining(false);

    // 3. Model stats
    print_model_stats(model, num_classes);

    // 4. Allocate input
    const int input_elems = profile_batch * 3 * 224 * 224;
    float* d_input = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, input_elems * sizeof(float)));
    gpu_fill_random(d_input, input_elems, 1.0f);

    // 5. Per-layer profiling
    print_layer_profile(model, d_input, profile_batch);

    // 6. Latency percentiles
    print_latency_percentiles(model, d_input, profile_batch);

    // 7. Memory report
    print_memory_report(model, d_input, profile_batch);

    // Cleanup single-batch model
    cudaFree(d_input);

    // 8. Batch-size scaling (creates fresh models per batch)
    print_batch_scaling(max_batch, num_classes);

    printf("\nBenchmark complete.\n");
    return 0;
}
