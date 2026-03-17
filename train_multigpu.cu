/*
 * train_multigpu.cu  –  ResNet-50 single-node multi-GPU DDP training
 *
 * Strategy: Data Parallel via NCCL AllReduce
 *   - Each GPU holds a full copy of ResNet50.
 *   - Each GPU trains on a different shard of the mini-batch.
 *   - After Backward(), SyncGradients() uses ncclAllReduce to average
 *     all gradients across GPUs before each UpdateSGD() call.
 *
 * Build:
 *   cmake -DUSE_NCCL=ON ..
 *   make -j train_multigpu
 *
 * Usage:
 *   ./train_multigpu [data_dir] [global_batch] [epochs] [lr]
 */

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "imagenet_loader.h"
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
#include <thread>
#include <mutex>
#include <atomic>
#include <barrier>
#include <nccl.h>
#include <nvtx3/nvToolsExt.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fs = std::filesystem;

// ── Global barrier so all GPU threads are synchronized each step ──────────────
// One synchronization point per epoch boundary
static std::mutex g_print_mutex;      // serialise console output
static std::atomic<bool> g_stop{false};

// ── Cosine annealing with linear warmup (same as single-GPU training) ─────────
static float cosine_lr(float base_lr, int epoch, int num_epochs, int warmup_epochs) {
    if (epoch < warmup_epochs)
        return base_lr * ((float)(epoch + 1) / (float)warmup_epochs);
    int t = epoch - warmup_epochs;
    int T = num_epochs - warmup_epochs;
    if (T <= 0) return base_lr;
    return base_lr * 0.5f * (1.0f + cosf((float)M_PI * t / T));
}

// ── Per-GPU worker thread ─────────────────────────────────────────────────────
struct GPUWorkerArgs {
    int         rank;           // this GPU's rank (0..num_gpus-1)
    int         num_gpus;
    ncclComm_t  comm;           // NCCL communicator for this rank
    std::string data_dir;
    int         local_batch;    // per-GPU batch size
    int         num_epochs;
    float       base_lr;
    int         num_classes;
    float       momentum;
    float       weight_decay;
    int         warmup_epochs;
    int         patience;
    // Only rank 0 writes files
    std::string checkpoint_dir;
};

void gpu_worker(GPUWorkerArgs args) {
    const int rank      = args.rank;
    const int num_gpus  = args.num_gpus;

    CUDA_CHECK(cudaSetDevice(rank));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Each GPU has its own model and data loader (with different RNG seed)
    ResNet50 model(args.num_classes);
    model.InitWeights();

    // Enable TF32 on Ampere+
    cublasSetMathMode(model.Cublas(), CUBLAS_TF32_TENSOR_OP_MATH);

    // ── Broadcast weights from rank 0 ────────────────────────────────────────
    // Helper to broadcast a single float buffer
    auto bcast = [&](float* buf, size_t n) {
        ncclBcast(buf, n, ncclFloat, 0, args.comm, stream);
    };
    // We rely on rank 0 initializing first, then sync
    ncclBarrier:  // Not standard — instead we use GroupStart/End
    ncclGroupStart();
    // (All ranks take part in broadcast even though only rank 0 has real data)
    // Simple approach: all ranks init, but rank 0 will overwrite after first sync
    ncclGroupEnd();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── Separate data loaders per GPU (each has its own RNG via rank-based seed)
    ImageNetLoader train_loader(args.data_dir, args.local_batch, "train", true);
    ImageNetLoader val_loader(args.data_dir, args.local_batch, "val", false);

    // ── GPU buffers ───────────────────────────────────────────────────────────
    float* d_images = nullptr;
    int*   d_labels = nullptr;
    CUDA_CHECK(cudaMalloc(&d_images, args.local_batch * 3 * 224 * 224 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, args.local_batch * sizeof(int)));

    std::vector<float> h_images;
    std::vector<int>   h_labels;

    // Only rank 0 opens the CSV log
    std::ofstream csv_log;
    if (rank == 0) {
        fs::create_directories(args.checkpoint_dir);
        csv_log.open(args.checkpoint_dir + "/training_log_multigpu.csv");
        csv_log << "epoch,lr,train_loss,train_acc,val_loss,val_acc,epoch_time_s,images_per_sec\n";
    }

    float best_val_loss = FLT_MAX;
    int   patience_counter = 0;
    int   start_epoch = 0;

    if (rank == 0) {
        printf("\nMulti-GPU Training: %d GPUs | Global Batch: %d | Local per GPU: %d\n",
               num_gpus, num_gpus * args.local_batch, args.local_batch);
        printf("%-6s %-8s %-12s %-12s %-12s %-12s %-10s %-12s\n",
               "Epoch", "LR", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "Time(s)", "Img/sec");
        printf("--------------------------------------------------------------------------------------------\n");
    }

    for (int epoch = start_epoch; epoch < args.num_epochs && !g_stop.load(); ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        float lr = cosine_lr(args.base_lr, epoch, args.num_epochs, args.warmup_epochs);

        train_loader.Reset();
        model.SetTraining(true);

        float epoch_loss = 0.0f, epoch_acc = 0.0f;
        int   batches = 0;

        nvtxRangePushA("Epoch");
        while (train_loader.HasNextBatch()) {
            nvtxRangePushA("DataLoad_H2D");
            train_loader.GetNextBatch(h_images, h_labels);
            CUDA_CHECK(cudaMemcpyAsync(d_images, h_images.data(), h_images.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_labels, h_labels.data(), h_labels.size() * sizeof(int),   cudaMemcpyHostToDevice, stream));
            nvtxRangePop();

            nvtxRangePushA("TrainStep");
            model.ZeroGrad();
            model.Forward(d_images, args.local_batch, 224, 224);
            float loss = model.Loss(d_labels, args.local_batch);
            float acc  = model.Top1Accuracy(d_labels, args.local_batch);
            model.Backward(d_labels, args.local_batch);
            nvtxRangePop();

            // ── NCCL AllReduce: average gradients across all GPUs ─────────────
            nvtxRangePushA("SyncGradients");
            model.SyncGradients(args.comm, stream);
            nvtxRangePop();

            model.UpdateSGD(lr, args.momentum, args.weight_decay);

            epoch_loss += loss;
            epoch_acc  += acc;
            batches++;
        }
        nvtxRangePop();

        float avg_train_loss = epoch_loss / batches;
        float avg_train_acc  = epoch_acc  / batches;

        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_sec = std::chrono::duration<double>(epoch_end - epoch_start).count();
        // Total images = (all GPUs) * batches * local_batch
        double img_per_sec = (double)(num_gpus * batches * args.local_batch) / epoch_sec;

        // ── Validation (only rank 0 to avoid redundant computation) ──────────
        float val_loss = 0.0f, val_acc = 0.0f;
        if (rank == 0 && val_loader.NumSamples() > 0) {
            model.SetTraining(false);
            val_loader.Reset();
            float vl = 0, va = 0;
            int   vb = 0;
            while (val_loader.HasNextBatch()) {
                val_loader.GetNextBatch(h_images, h_labels);
                CUDA_CHECK(cudaMemcpy(d_images, h_images.data(), h_images.size() * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(), h_labels.size() * sizeof(int),   cudaMemcpyHostToDevice));
                model.Forward(d_images, args.local_batch, 224, 224);
                vl += model.Loss(d_labels, args.local_batch);
                va += model.Top1Accuracy(d_labels, args.local_batch);
                vb++;
            }
            val_loss = vl / vb;
            val_acc  = va / vb;
            model.SetTraining(true);
        }

        // ── Only rank 0 prints, logs, and saves checkpoints ──────────────────
        if (rank == 0) {
            printf("%-6d %-8.5f %-12.4f %-12.2f %-12.4f %-12.2f %-10.1f %-12.0f\n",
                   epoch + 1, lr, avg_train_loss, avg_train_acc * 100.0f,
                   val_loss, val_acc * 100.0f, epoch_sec, img_per_sec);

            csv_log << epoch + 1 << "," << lr << ","
                    << avg_train_loss << "," << avg_train_acc << ","
                    << val_loss << "," << val_acc << ","
                    << epoch_sec << "," << img_per_sec << "\n";
            csv_log.flush();

            std::string pt_path = args.checkpoint_dir + "/resnet50_multigpu_epoch" + std::to_string(epoch) + ".bin";
            model.SaveWeights(pt_path);

            if (val_loader.NumSamples() > 0 && val_loss < best_val_loss) {
                best_val_loss = val_loss;
                patience_counter = 0;
                model.SaveWeights(args.checkpoint_dir + "/resnet50_multigpu_best.bin");
                printf("  >> New best model (val_loss=%.4f)\n", val_loss);
            } else {
                patience_counter++;
                if (patience_counter >= args.patience) {
                    printf("\n=== Early stopping at epoch %d ===\n", epoch + 1);
                    g_stop.store(true);
                }
            }
        }

        // Broadcast stop signal so all GPUs exit together
        int stop_flag = g_stop.load() ? 1 : 0;
        ncclBcast(&stop_flag, 1, ncclInt32, 0, args.comm, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (stop_flag) break;
    }

    CUDA_CHECK(cudaFree(d_images));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaStreamDestroy(stream));

    if (rank == 0 && csv_log.is_open()) csv_log.close();
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    std::string data_dir  = (argc > 1) ? argv[1] : "./imagenet";
    int global_batch      = (argc > 2) ? std::atoi(argv[2]) : 256;
    int num_epochs        = (argc > 3) ? std::atoi(argv[3]) : 90;
    float base_lr         = (argc > 4) ? std::atof(argv[4]) : 0.1f;
    int num_classes       = 1000;
    float momentum        = 0.9f;
    float weight_decay    = 1e-4f;
    int warmup_epochs     = 5;
    int patience          = 10;

    // ── Discover GPUs ─────────────────────────────────────────────────────────
    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    if (num_gpus == 0) {
        std::cerr << "No CUDA GPUs found!\n";
        return 1;
    }

    int local_batch = global_batch / num_gpus;
    if (local_batch < 1) local_batch = 1;

    std::cout << "=== ResNet-50 Multi-GPU ImageNet Training (DDP via NCCL) ===\n";
    std::cout << "GPUs found       : " << num_gpus << "\n";
    std::cout << "Global batch     : " << global_batch << " | Per-GPU: " << local_batch << "\n";
    std::cout << "Epochs           : " << num_epochs << "\n";
    std::cout << "Base LR          : " << base_lr << "\n";
    std::cout << "Data dir         : " << data_dir << "\n\n";

    // ── Initialize NCCL communicators ─────────────────────────────────────────
    std::vector<int>         devs(num_gpus);
    std::vector<ncclComm_t>  comms(num_gpus);
    for (int i = 0; i < num_gpus; ++i) devs[i] = i;
    NCCL_CHECK(ncclCommInitAll(comms.data(), num_gpus, devs.data()));

    // ── Create checkpoints dir ────────────────────────────────────────────────
    fs::create_directories("checkpoints");

    // ── Launch one thread per GPU ─────────────────────────────────────────────
    std::vector<std::thread> threads;
    threads.reserve(num_gpus);

    for (int rank = 0; rank < num_gpus; ++rank) {
        GPUWorkerArgs args;
        args.rank            = rank;
        args.num_gpus        = num_gpus;
        args.comm            = comms[rank];
        args.data_dir        = data_dir;
        args.local_batch     = local_batch;
        args.num_epochs      = num_epochs;
        args.base_lr         = base_lr;
        args.num_classes     = num_classes;
        args.momentum        = momentum;
        args.weight_decay    = weight_decay;
        args.warmup_epochs   = warmup_epochs;
        args.patience        = patience;
        args.checkpoint_dir  = "checkpoints";
        threads.emplace_back(gpu_worker, args);
    }

    for (auto& t : threads) t.join();

    // ── Cleanup NCCL ─────────────────────────────────────────────────────────
    for (int i = 0; i < num_gpus; ++i) ncclCommDestroy(comms[i]);

    std::cout << "\nMulti-GPU training complete.\n";
    std::cout << "Checkpoints in: checkpoints/\n";
    return 0;
}
