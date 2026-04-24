#include "hip/hip_runtime.h"
/*
 * train_bench.cu  –  Training convergence benchmark for ResNet-50
 *
 * Usage:
 *   ./train_bench [batch_size] [num_epochs] [learning_rate] [schedule]
 *
 *   schedule: constant | step | cosine   (default: cosine)
 *
 * Reports per epoch:
 *   - Loss, Top-1 accuracy
 *   - Gradient L2 norm
 *   - Current learning rate
 *   - Forward / backward / total step time
 *
 * Also reports:
 *   - Convergence summary
 *   - Peak GPU memory usage
 *   - Average training throughput
 *
 * Compile:
 *   nvcc -O2 -arch=sm_75 -std=c++17 train_bench.cu resnet50.cu -o train_bench -lcublas
 */

#include "resnet50.cuh"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>

// ─────────────────────────────────────────────────────────────────────────────
// LR Schedulers
// ─────────────────────────────────────────────────────────────────────────────

enum LRSchedule { CONSTANT, STEP, COSINE };

static float lr_constant(float base_lr, int /*epoch*/, int /*total*/)
{
    return base_lr;
}

static float lr_step(float base_lr, int epoch, int total)
{
    // Halve every 1/3 of total epochs
    int step = total / 3;
    if (step < 1) step = 1;
    int drops = epoch / step;
    return base_lr * powf(0.5f, (float)drops);
}

static float lr_cosine(float base_lr, int epoch, int total)
{
    // Cosine annealing with 5-epoch warmup
    int warmup = (total > 10) ? 5 : 0;
    if (epoch < warmup) {
        return base_lr * (float)(epoch + 1) / (float)warmup;
    }
    float progress = (float)(epoch - warmup) / (float)(total - warmup);
    return base_lr * 0.5f * (1.f + cosf(progress * 3.14159265f));
}

static float get_lr(LRSchedule sched, float base_lr, int epoch, int total)
{
    switch (sched) {
        case CONSTANT: return lr_constant(base_lr, epoch, total);
        case STEP:     return lr_step(base_lr, epoch, total);
        case COSINE:   return lr_cosine(base_lr, epoch, total);
    }
    return base_lr;
}

static const char* schedule_name(LRSchedule s)
{
    switch (s) {
        case CONSTANT: return "constant";
        case STEP:     return "step_decay";
        case COSINE:   return "cosine_warmup";
    }
    return "unknown";
}

static LRSchedule parse_schedule(const char* s)
{
    if (!s) return COSINE;
    if (strcmp(s, "constant") == 0 || strcmp(s, "const") == 0) return CONSTANT;
    if (strcmp(s, "step") == 0)    return STEP;
    if (strcmp(s, "cosine") == 0)  return COSINE;
    fprintf(stderr, "Unknown schedule '%s', using cosine\n", s);
    return COSINE;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static void print_separator(int width = 100)
{
    for (int i = 0; i < width; ++i) putchar('-');
    putchar('\n');
}

static size_t get_free_memory()
{
    size_t free_mem, total_mem;
    CUDA_CHECK(hipMemGetInfo(&free_mem, &total_mem));
    return free_mem;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    srand(42);

    // Parse arguments
    int   batch_size   = (argc > 1) ? std::atoi(argv[1]) : 4;
    int   num_epochs   = (argc > 2) ? std::atoi(argv[2]) : 50;
    float base_lr      = (argc > 3) ? std::atof(argv[3]) : 0.01f;
    LRSchedule schedule = parse_schedule((argc > 4) ? argv[4] : "cosine");
    int   num_classes  = 10;
    float momentum     = 0.9f;
    float weight_decay = 1e-4f;

    // GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("============================================================\n");
    printf("       ResNet-50 Training Convergence Benchmark\n");
    printf("============================================================\n\n");

    printf("GPU             : %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("VRAM            : %zu MB\n", prop.totalGlobalMem / (1 << 20));
    printf("Batch size      : %d\n", batch_size);
    printf("Epochs          : %d\n", num_epochs);
    printf("Base LR         : %.4f\n", base_lr);
    printf("LR Schedule     : %s\n", schedule_name(schedule));
    printf("Momentum        : %.2f\n", momentum);
    printf("Weight decay    : %.1e\n", weight_decay);
    printf("Classes         : %d\n", num_classes);
    printf("Free GPU memory : %zu MB\n\n", get_free_memory() / (1 << 20));

    // Build model
    printf("Building ResNet-50 ...\n");
    ResNet50 model(num_classes);
    model.InitWeights();
    model.SetTraining(true);

    size_t param_count = model.ParamCount();
    printf("Parameters      : %zu (%.1f M)\n", param_count, param_count / 1e6);
    printf("Post-init memory: %zu MB free\n\n", get_free_memory() / (1 << 20));

    // Allocate synthetic data
    const int input_elems = batch_size * 3 * 224 * 224;
    float* d_images = nullptr;
    int*   d_labels = nullptr;
    CUDA_CHECK(cudaMalloc(&d_images, (size_t)input_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, (size_t)batch_size  * sizeof(int)));
    gpu_fill_random(d_images, input_elems, 1.0f);

    std::vector<int> h_labels(batch_size);
    for (int i = 0; i < batch_size; ++i)
        h_labels[i] = rand() % num_classes;
    CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(),
                          batch_size * sizeof(int), cudaMemcpyHostToDevice));

    // Warm-up
    printf("Warm-up forward pass ...\n");
    model.Forward(d_images, batch_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    size_t mem_after_warmup = get_free_memory();
    printf("Post-warmup     : %zu MB free\n\n", mem_after_warmup / (1 << 20));

    // CUDA events for timing
    cudaEvent_t ev_fwd_start, ev_fwd_stop;
    cudaEvent_t ev_bwd_start, ev_bwd_stop;
    cudaEvent_t ev_step_start, ev_step_stop;
    CUDA_CHECK(cudaEventCreate(&ev_fwd_start));
    CUDA_CHECK(cudaEventCreate(&ev_fwd_stop));
    CUDA_CHECK(cudaEventCreate(&ev_bwd_start));
    CUDA_CHECK(cudaEventCreate(&ev_bwd_stop));
    CUDA_CHECK(cudaEventCreate(&ev_step_start));
    CUDA_CHECK(cudaEventCreate(&ev_step_stop));

    // Storage for epoch metrics
    std::vector<float> losses(num_epochs), accs(num_epochs), grad_norms(num_epochs);
    std::vector<float> lrs(num_epochs);
    std::vector<float> fwd_times(num_epochs), bwd_times(num_epochs), total_times(num_epochs);

    // Header
    printf("Starting training ...\n");
    print_separator();
    printf("%5s  %8s  %7s  %10s  %8s  %8s  %8s  %8s\n",
           "Epoch", "Loss", "Acc %", "Grad Norm", "LR", "Fwd(ms)", "Bwd(ms)", "Tot(ms)");
    print_separator();

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float lr = get_lr(schedule, base_lr, epoch, num_epochs);
        lrs[epoch] = lr;

        CUDA_CHECK(cudaEventRecord(ev_step_start));

        // Zero gradients
        model.ZeroGrad();

        // Forward
        CUDA_CHECK(cudaEventRecord(ev_fwd_start));
        model.Forward(d_images, batch_size);
        CUDA_CHECK(cudaEventRecord(ev_fwd_stop));

        // Compute loss/accuracy
        float loss = model.Loss(d_labels, batch_size);
        float acc  = model.Top1Accuracy(d_labels, batch_size);

        // Backward
        CUDA_CHECK(cudaEventRecord(ev_bwd_start));
        model.Backward(d_labels, batch_size);
        CUDA_CHECK(cudaEventRecord(ev_bwd_stop));

        // Gradient norm (before update)
        float gnorm = model.GradNorm();

        // SGD step
        model.UpdateSGD(lr, momentum, weight_decay);

        CUDA_CHECK(cudaEventRecord(ev_step_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_step_stop));

        // Collect times
        float fwd_ms = 0, bwd_ms = 0, step_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&fwd_ms, ev_fwd_start, ev_fwd_stop));
        CUDA_CHECK(cudaEventElapsedTime(&bwd_ms, ev_bwd_start, ev_bwd_stop));
        CUDA_CHECK(cudaEventElapsedTime(&step_ms, ev_step_start, ev_step_stop));

        losses[epoch]      = loss;
        accs[epoch]        = acc;
        grad_norms[epoch]  = gnorm;
        fwd_times[epoch]   = fwd_ms;
        bwd_times[epoch]   = bwd_ms;
        total_times[epoch] = step_ms;

        printf("%5d  %8.4f  %6.1f%%  %10.4f  %.1e  %8.1f  %8.1f  %8.1f\n",
               epoch + 1, loss, acc * 100.f, gnorm, lr,
               fwd_ms, bwd_ms, step_ms);
    }
    print_separator();

    // ── Convergence Summary ──────────────────────────────────────────────────

    printf("\n=== Convergence Summary ===\n");
    print_separator();

    // Final eval mode pass
    model.SetTraining(false);
    model.Forward(d_images, batch_size);
    float final_loss = model.Loss(d_labels, batch_size);
    float final_acc  = model.Top1Accuracy(d_labels, batch_size);

    printf("Final loss (eval)   : %.4f\n", final_loss);
    printf("Final acc  (eval)   : %.1f %%\n", final_acc * 100.f);

    // Find best epoch
    int best_epoch = 0;
    for (int i = 1; i < num_epochs; ++i)
        if (losses[i] < losses[best_epoch]) best_epoch = i;
    printf("Best training loss  : %.4f (epoch %d)\n", losses[best_epoch], best_epoch + 1);

    // Convergence speed: epoch where loss first drops below 0.1
    int converge_epoch = -1;
    for (int i = 0; i < num_epochs; ++i)
        if (losses[i] < 0.1f) { converge_epoch = i + 1; break; }
    if (converge_epoch > 0)
        printf("Loss < 0.1 at epoch : %d\n", converge_epoch);
    else
        printf("Loss < 0.1 at epoch : not reached\n");

    // Gradient norm statistics
    float gn_min = *std::min_element(grad_norms.begin(), grad_norms.end());
    float gn_max = *std::max_element(grad_norms.begin(), grad_norms.end());
    float gn_final = grad_norms.back();
    printf("Grad norm (min/max) : %.4f / %.4f\n", gn_min, gn_max);
    printf("Grad norm (final)   : %.4f\n", gn_final);

    // Timing summary
    float avg_fwd   = std::accumulate(fwd_times.begin(), fwd_times.end(), 0.f) / num_epochs;
    float avg_bwd   = std::accumulate(bwd_times.begin(), bwd_times.end(), 0.f) / num_epochs;
    float avg_total  = std::accumulate(total_times.begin(), total_times.end(), 0.f) / num_epochs;
    float total_train = std::accumulate(total_times.begin(), total_times.end(), 0.f);

    printf("\n=== Timing Summary ===\n");
    print_separator();
    printf("Avg forward         : %.1f ms\n", avg_fwd);
    printf("Avg backward        : %.1f ms\n", avg_bwd);
    printf("Avg step (total)    : %.1f ms\n", avg_total);
    printf("Bwd/Fwd ratio       : %.2fx\n", avg_fwd > 0 ? avg_bwd / avg_fwd : 0.f);
    printf("Training throughput : %.1f images/s\n",
           avg_total > 0 ? 1000.f * batch_size / avg_total : 0.f);
    printf("Total training time : %.1f s\n", total_train / 1000.f);

    // Memory
    size_t free_final, total_mem;
    CUDA_CHECK(hipMemGetInfo(&free_final, &total_mem));
    printf("\n=== Memory ===\n");
    print_separator();
    printf("Total VRAM          : %zu MB\n", total_mem / (1 << 20));
    printf("Peak usage (est)    : %zu MB\n", (total_mem - free_final) / (1 << 20));
    printf("Free at end         : %zu MB\n", free_final / (1 << 20));

    print_separator();
    printf("\nTraining benchmark complete.\n");

    // Cleanup
    cudaFree(d_images);
    cudaFree(d_labels);
    cudaEventDestroy(ev_fwd_start);
    cudaEventDestroy(ev_fwd_stop);
    cudaEventDestroy(ev_bwd_start);
    cudaEventDestroy(ev_bwd_stop);
    cudaEventDestroy(ev_step_start);
    cudaEventDestroy(ev_step_stop);

    return 0;
}
