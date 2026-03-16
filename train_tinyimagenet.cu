/*
 * train_tinyimagenet.cu  –  ResNet-50 training loop on Tiny ImageNet
 *
 * Features:
 *   - Validation evaluation after each epoch
 *   - Cosine annealing LR scheduler with warmup
 *   - Early stopping based on validation loss
 *   - Best model checkpoint (lowest val loss)
 *   - Data augmentation (random horizontal flip)
 *   - TF32 tensor-core acceleration (Ampere+)
 *   - Throughput & performance logging to CSV
 *
 * Usage:
 *   ./train_tinyimagenet [data_dir] [batch_size] [num_epochs] [learning_rate]
 *
 * Compile:
 *   nvcc -O2 -arch=sm_75 -std=c++17 train_tinyimagenet.cu resnet50.cu -o train_tinyimagenet -lcublas -lcudnn
 */

#define STB_IMAGE_IMPLEMENTATION
#include "tiny_dataloader.h"
#include "resnet50.cuh"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <string>
#include <filesystem>
#include <chrono>
#include <nvtx3/nvToolsExt.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fs = std::filesystem;

// ── Cosine annealing with linear warmup ──────────────────────────────────────
static float cosine_lr(float base_lr, int epoch, int num_epochs, int warmup_epochs)
{
    if (epoch < warmup_epochs) {
        return base_lr * ((float)(epoch + 1) / (float)warmup_epochs);
    }
    int t = epoch - warmup_epochs;
    int T = num_epochs - warmup_epochs;
    if (T <= 0) return base_lr;
    return base_lr * 0.5f * (1.0f + cosf((float)M_PI * t / T));
}

// ── Validation loop ──────────────────────────────────────────────────────────
static void evaluate(ResNet50& model, TinyImageNetLoader& val_loader,
                     float* d_images, int* d_labels, int batch_size,
                     float& out_loss, float& out_acc)
{
    model.SetTraining(false);
    val_loader.Reset();

    float total_loss = 0.0f;
    float total_acc  = 0.0f;
    int   batches    = 0;

    std::vector<float> h_images;
    std::vector<int>   h_labels;

    while (val_loader.HasNextBatch()) {
        val_loader.GetNextBatch(h_images, h_labels);
        CUDA_CHECK(cudaMemcpy(d_images, h_images.data(), h_images.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(), h_labels.size() * sizeof(int),   cudaMemcpyHostToDevice));

        model.Forward(d_images, batch_size, 64, 64);
        total_loss += model.Loss(d_labels, batch_size);
        total_acc  += model.Top1Accuracy(d_labels, batch_size);
        batches++;
    }

    out_loss = (batches > 0) ? total_loss / batches : 0.0f;
    out_acc  = (batches > 0) ? total_acc  / batches : 0.0f;

    model.SetTraining(true);
}

int main(int argc, char** argv)
{
    std::string data_dir  = (argc > 1) ? argv[1] : "./tiny-imagenet-200";
    int batch_size        = (argc > 2) ? std::atoi(argv[2]) : 32;
    int num_epochs        = (argc > 3) ? std::atoi(argv[3]) : 50;
    float base_lr         = (argc > 4) ? std::atof(argv[4]) : 0.01f;
    int num_classes       = 200;
    float momentum        = 0.9f;
    float weight_decay    = 1e-4f;
    int warmup_epochs     = 5;
    int patience          = 10;  // early stopping patience

    std::cout << "=== ResNet-50 Tiny ImageNet Training ===\n";
    std::cout << "Data dir        : " << data_dir << "\n";
    std::cout << "Batch size      : " << batch_size << "\n";
    std::cout << "Epochs          : " << num_epochs << "\n";
    std::cout << "Base LR         : " << base_lr << "\n";
    std::cout << "Warmup epochs   : " << warmup_epochs << "\n";
    std::cout << "Early stop pat. : " << patience << "\n";

    // ── Data loaders ─────────────────────────────────────────────────────────
    TinyImageNetLoader train_loader(data_dir, batch_size, "train", true);  // augmentation on
    TinyImageNetLoader val_loader(data_dir, batch_size, "val", false);     // no augmentation

    if (train_loader.NumSamples() == 0) {
        std::cerr << "Failed to load training data from " << data_dir << ".\n";
        return 1;
    }
    if (val_loader.NumSamples() == 0) {
        std::cerr << "Warning: No validation data found. Validation will be skipped.\n";
    }

    // ── Build Model ──────────────────────────────────────────────────────────
    std::cout << "Building ResNet-50 for " << num_classes << " classes...\n";
    ResNet50 model(num_classes);
    model.InitWeights();
    model.SetTraining(true);

    // ── Enable TF32 for Ampere+ GPUs ─────────────────────────────────────────
    cublasSetMathMode(model.Cublas(), CUBLAS_TF32_TENSOR_OP_MATH);
    std::cout << "TF32 tensor math enabled (Ampere+ GPUs).\n";

    std::cout << "Model parameters: " << model.ParamCount() << "\n";

    // ── GPU buffers ──────────────────────────────────────────────────────────
    float* d_images = nullptr;
    int*   d_labels = nullptr;
    CUDA_CHECK(cudaMalloc(&d_images, batch_size * 3 * 64 * 64 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, batch_size * sizeof(int)));

    std::vector<float> h_images;
    std::vector<int>   h_labels;

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // ── CSV log ──────────────────────────────────────────────────────────────
    fs::create_directories("checkpoints");
    std::ofstream csv_log("checkpoints/training_log.csv");
    csv_log << "epoch,lr,train_loss,train_acc,val_loss,val_acc,epoch_time_s,images_per_sec\n";

    // ── Resume checkpoint ────────────────────────────────────────────────────
    int start_epoch = 0;
    for (int e = num_epochs - 1; e >= 0; --e) {
        std::string pt_path = "checkpoints/resnet50_epoch" + std::to_string(e) + ".bin";
        if (fs::exists(pt_path)) {
            std::cout << "Found checkpoint: " << pt_path << ". Resuming...\n";
            if (model.LoadWeights(pt_path)) {
                start_epoch = e + 1;
                break;
            }
        }
    }

    if (start_epoch >= num_epochs) {
        std::cout << "Training already completed up to epoch " << num_epochs << ".\n";
        return 0;
    }

    // ── Early stopping state ─────────────────────────────────────────────────
    float best_val_loss = FLT_MAX;
    int   patience_counter = 0;

    // If resuming, run a quick val to set best_val_loss baseline
    if (start_epoch > 0 && val_loader.NumSamples() > 0) {
        float vl, va;
        evaluate(model, val_loader, d_images, d_labels, batch_size, vl, va);
        best_val_loss = vl;
        printf("Resumed | Baseline val loss: %.4f | val acc: %.2f%%\n", vl, va * 100.0f);
    }

    // ── Training loop ────────────────────────────────────────────────────────
    std::cout << "\nStarting Training...\n";
    printf("%-6s %-8s %-12s %-12s %-12s %-12s %-10s %-12s\n",
           "Epoch", "LR", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "Time(s)", "Img/sec");
    printf("--------------------------------------------------------------------------------------------\n");

    for (int epoch = start_epoch; epoch < num_epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();

        // Cosine LR with warmup
        float lr = cosine_lr(base_lr, epoch, num_epochs, warmup_epochs);

        nvtxRangePushA("Epoch");
        train_loader.Reset();

        float epoch_loss = 0.0f;
        float epoch_acc  = 0.0f;
        int   batches    = 0;

        while (train_loader.HasNextBatch()) {
            CUDA_CHECK(cudaEventRecord(ev_start));

            nvtxRangePushA("DataLoad_H2D");
            train_loader.GetNextBatch(h_images, h_labels);
            CUDA_CHECK(cudaMemcpy(d_images, h_images.data(), h_images.size() * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(), h_labels.size() * sizeof(int),   cudaMemcpyHostToDevice));
            nvtxRangePop();

            nvtxRangePushA("TrainStep");
            model.ZeroGrad();
            model.Forward(d_images, batch_size, 64, 64);

            float loss = model.Loss(d_labels, batch_size);
            float acc  = model.Top1Accuracy(d_labels, batch_size);

            model.Backward(d_labels, batch_size);
            model.UpdateSGD(lr, momentum, weight_decay);
            nvtxRangePop();

            epoch_loss += loss;
            epoch_acc  += acc;
            batches++;

            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            float ms = 0.f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));

            if (batches % 100 == 0) {
                printf("  [Batch %d/%d] Loss: %.4f | Acc: %.2f%% | %.1f ms/batch\n",
                    batches, train_loader.NumSamples() / batch_size, loss, acc * 100.0f, ms);
            }
        }
        nvtxRangePop();

        float avg_train_loss = epoch_loss / batches;
        float avg_train_acc  = epoch_acc  / batches;

        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_sec = std::chrono::duration<double>(epoch_end - epoch_start).count();
        double img_per_sec = (double)(batches * batch_size) / epoch_sec;

        // ── Validation ───────────────────────────────────────────────────────
        float val_loss = 0.0f, val_acc = 0.0f;
        if (val_loader.NumSamples() > 0) {
            evaluate(model, val_loader, d_images, d_labels, batch_size, val_loss, val_acc);
        }

        // ── Print epoch summary ──────────────────────────────────────────────
        printf("%-6d %-8.5f %-12.4f %-12.2f %-12.4f %-12.2f %-10.1f %-12.0f\n",
               epoch + 1, lr, avg_train_loss, avg_train_acc * 100.0f,
               val_loss, val_acc * 100.0f, epoch_sec, img_per_sec);

        // ── CSV log ──────────────────────────────────────────────────────────
        csv_log << epoch + 1 << "," << lr << ","
                << avg_train_loss << "," << avg_train_acc << ","
                << val_loss << "," << val_acc << ","
                << epoch_sec << "," << img_per_sec << "\n";
        csv_log.flush();

        // ── Save epoch checkpoint ────────────────────────────────────────────
        std::string pt_path = "checkpoints/resnet50_epoch" + std::to_string(epoch) + ".bin";
        model.SaveWeights(pt_path);

        // ── Best model + early stopping ──────────────────────────────────────
        if (val_loader.NumSamples() > 0) {
            if (val_loss < best_val_loss) {
                best_val_loss = val_loss;
                patience_counter = 0;
                model.SaveWeights("checkpoints/resnet50_best.bin");
                printf("  >> New best model saved (val_loss=%.4f)\n", val_loss);
            } else {
                patience_counter++;
                if (patience_counter >= patience) {
                    printf("\n=== Early stopping at epoch %d (no improvement for %d epochs) ===\n",
                           epoch + 1, patience);
                    break;
                }
            }
        }
    }

    // ── Cleanup ──────────────────────────────────────────────────────────────
    csv_log.close();
    cudaFree(d_images);
    cudaFree(d_labels);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    printf("\nTraining complete. Best val loss: %.4f\n", best_val_loss);
    printf("Checkpoints in: checkpoints/\n");
    printf("Training log:   checkpoints/training_log.csv\n");

    return 0;
}
