/*
 * train.cu  –  ResNet-50 training loop (synthetic data)
 *
 * Usage:
 *   ./train [batch_size] [num_epochs] [learning_rate]
 *
 * Demonstrates:
 *   - Forward pass with training-mode batch normalization
 *   - Cross-entropy loss computation
 *   - Backward pass (backpropagation through all layers)
 *   - SGD optimizer with momentum and weight decay
 *   - Top-1 accuracy tracking
 *
 * The model trains on a fixed random batch and should overfit,
 * showing loss → 0 and accuracy → 100%.
 *
 * Compile:
 *   nvcc -O2 -arch=sm_75 -std=c++17 train.cu resnet50.cu -o train -lcublas
 */

#include "resnet50.cuh"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static void print_device_info()
{
    int dev = 0;
    hipDeviceProp_t prop;
    CUDA_CHECK(hipGetDeviceProperties(&prop, dev));
    std::cout << "GPU: " << prop.name
              << "  (SM " << prop.major << "." << prop.minor << ")"
              << "  VRAM: " << prop.totalGlobalMem / (1 << 20) << " MB\n";
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

    // ── Parse arguments ──────────────────────────────────────────────────
    int   batch_size  = (argc > 1) ? std::atoi(argv[1]) : 4;
    int   num_epochs  = (argc > 2) ? std::atoi(argv[2]) : 50;
    float lr          = (argc > 3) ? std::atof(argv[3]) : 0.01f;
    int   num_classes = 10;
    float momentum    = 0.9f;
    float weight_decay = 1e-4f;

    std::cout << "=== ResNet-50 Training (CUDA/cuBLAS) ===\n";
    print_device_info();
    std::cout << "Batch size      : " << batch_size   << "\n";
    std::cout << "Epochs          : " << num_epochs   << "\n";
    std::cout << "Learning rate   : " << lr           << "\n";
    std::cout << "Classes         : " << num_classes  << "\n";
    std::cout << "Momentum        : " << momentum     << "\n";
    std::cout << "Weight decay    : " << weight_decay << "\n";
    std::cout << "Free GPU memory : " << get_free_memory() / (1 << 20) << " MB\n\n";

    // ── Build model ──────────────────────────────────────────────────────
    std::cout << "Building ResNet-50 …\n";
    ResNet50 model(num_classes);
    model.InitWeights();
    model.SetTraining(true);
    std::cout << "Model built. Free GPU memory: "
              << get_free_memory() / (1 << 20) << " MB\n\n";

    // ── Allocate synthetic training data ─────────────────────────────────
    // Fixed random batch: [batch_size, 3, 224, 224] images, random labels
    const int input_elems = batch_size * 3 * 224 * 224;
    float* d_images = nullptr;
    int*   d_labels = nullptr;

    CUDA_CHECK(hipMalloc(&d_images, (size_t)input_elems * sizeof(float)));
    CUDA_CHECK(hipMalloc(&d_labels, (size_t)batch_size  * sizeof(int)));

    // Generate random images (normalized to [-1, 1])
    gpu_fill_random(d_images, input_elems, 1.0f);

    // Generate random labels [0, num_classes)
    std::vector<int> h_labels(batch_size);
    for (int i = 0; i < batch_size; ++i)
        h_labels[i] = rand() % num_classes;
    CUDA_CHECK(hipMemcpy(d_labels, h_labels.data(),
                          batch_size * sizeof(int), hipMemcpyHostToDevice));

    std::cout << "Labels: [";
    for (int i = 0; i < batch_size; ++i)
        std::cout << h_labels[i] << (i + 1 < batch_size ? ", " : "");
    std::cout << "]\n\n";

    // ── Warm-up forward pass ─────────────────────────────────────────────
    std::cout << "Warm-up forward pass …\n";
    model.Forward(d_images, batch_size);
    CUDA_CHECK(hipDeviceSynchronize());
    std::cout << "Done. Free GPU memory: "
              << get_free_memory() / (1 << 20) << " MB\n\n";

    // ── Training loop ────────────────────────────────────────────────────
    std::cout << "Starting training …\n";
    std::cout << "─────────────────────────────────────────────\n";
    std::cout << "Epoch    Loss        Top-1 Acc    Time (ms)\n";
    std::cout << "─────────────────────────────────────────────\n";

    hipEvent_t ev_start, ev_stop;
    CUDA_CHECK(hipEventCreate(&ev_start));
    CUDA_CHECK(hipEventCreate(&ev_stop));

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        CUDA_CHECK(hipEventRecord(ev_start));

        // 1. Zero gradients
        model.ZeroGrad();

        // 2. Forward pass
        model.Forward(d_images, batch_size);

        // 3. Compute loss and accuracy (before backward modifies anything)
        float loss = model.Loss(d_labels, batch_size);
        float acc  = model.Top1Accuracy(d_labels, batch_size);

        // 4. Backward pass (backpropagation)
        model.Backward(d_labels, batch_size);

        // 5. SGD optimizer step
        model.UpdateSGD(lr, momentum, weight_decay);

        CUDA_CHECK(hipEventRecord(ev_stop));
        CUDA_CHECK(hipEventSynchronize(ev_stop));
        float ms = 0.f;
        CUDA_CHECK(hipEventElapsedTime(&ms, ev_start, ev_stop));

        printf("%5d    %8.4f    %6.1f %%      %7.1f\n",
               epoch + 1, loss, acc * 100.f, ms);
    }

    std::cout << "─────────────────────────────────────────────\n";
    std::cout << "Training complete.\n\n";

    // ── Final evaluation ─────────────────────────────────────────────────
    model.SetTraining(false);
    model.Forward(d_images, batch_size);
    float final_loss = model.Loss(d_labels, batch_size);
    float final_acc  = model.Top1Accuracy(d_labels, batch_size);
    std::cout << "Final (eval mode): loss=" << final_loss
              << "  acc=" << final_acc * 100.f << " %\n";

    // ── Cleanup ──────────────────────────────────────────────────────────
    hipFree(d_images);
    hipFree(d_labels);
    hipEventDestroy(ev_start);
    hipEventDestroy(ev_stop);

    return 0;
}
