#include "hip/hip_runtime.h"
/*
 * train_multigpu.cu  -  ResNet-50 single-node multi-GPU DDP training
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
#include <thread>
#include <mutex>
#include <atomic>
#ifdef USE_HIP
#include <rccl/rccl.h>
#else
#include <nccl.h>
#include <nvtx3/nvToolsExt.h>
#endif
#ifndef nvtxRangePushA
#define nvtxRangePushA(x) ((void)0)
#define nvtxRangePop()    ((void)0)
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── NCCL error-checking macro ─────────────────────────────────────────────────
#define NCCL_CHECK(cmd) do {                                        \
    ncclResult_t _r = (cmd);                                        \
    if (_r != ncclSuccess) {                                        \
        fprintf(stderr, "NCCL error at %s:%d — %s\n",              \
                __FILE__, __LINE__, ncclGetErrorString(_r));         \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)

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
    int         save_every_n;   // save checkpoint every N epochs (0 = only at end)
    // Only rank 0 writes files
    std::string checkpoint_dir;
};

void gpu_worker(GPUWorkerArgs args) {
    const int rank      = args.rank;
    const int num_gpus  = args.num_gpus;

    try {
    CUDA_CHECK(cudaSetDevice(rank));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Each GPU has its own model and data loader (with different RNG seed)
    /**
     * @brief Instantiates a ResNet50 model for multi-GPU training on tiny datasets.
     * 
     * @param args.num_classes The number of output classes for the classification task.
     *                          This parameter configures the final fully connected layer
     *                          of the ResNet50 network to match the target number of classes.
     * 
     * @return ResNet50 A ResNet50 model object initialized and ready for training.
     *                  The model contains 50 convolutional layers with residual connections,
     *                  optimized for distributed training across multiple GPUs.
     * 
     * @note This instantiation is typically used in a multi-GPU training pipeline where
     *       the model will be replicated or distributed across multiple CUDA devices
     *       for improved training performance on smaller datasets.
     */
    ResNet50 model(args.num_classes);
    model.SetStream(stream);
    model.InitWeights();

    // Enable TF32 on Ampere+
    CUBLAS_CHECK(cublasSetMathMode(model.Cublas(), CUBLAS_TF32_TENSOR_OP_MATH));

    // ── Broadcast weights from rank 0 to all GPUs ──────────────────────────
    model.BroadcastWeights(args.comm, stream);
    if (rank == 0) printf("[Rank 0] Weights broadcast to all GPUs\n");

    // ── Data loaders (each GPU gets a unique shard) ─────────────────────────
    printf("[Rank %d] Loading dataset from: %s\n", rank, args.data_dir.c_str());
    TinyImageNetLoader train_loader(args.data_dir, args.local_batch, "train", true, rank, num_gpus);
    printf("[Rank %d] Train loader ready: %d samples\n", rank, (int)train_loader.NumSamples());
    TinyImageNetLoader val_loader(args.data_dir, args.local_batch, "val", false, rank, num_gpus);
    printf("[Rank %d] Val loader ready: %d samples\n", rank, (int)val_loader.NumSamples());

    // ── GPU buffers ───────────────────────────────────────────────────────────
    float* d_images = nullptr;
    int*   d_labels = nullptr;
    CUDA_CHECK(cudaMalloc(&d_images, args.local_batch * 3 * 64 * 64 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, args.local_batch * sizeof(int)));

    // Device-side scratch for NCCL reductions (stop flag + batch count sync)
    int* d_stop_flag = nullptr;
    int* d_batch_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_stop_flag, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_batch_count, sizeof(int)));

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

        // ── Synchronize batch count across GPUs ──────────────────────────────
        // Each GPU may have a slightly different number of samples due to
        // sharding. We must ensure ALL GPUs run the SAME number of batches,
        // otherwise NCCL collective calls (AllReduce) will deadlock.
        int my_batches = (int)train_loader.NumSamples() / args.local_batch;
        CUDA_CHECK(cudaMemcpy(d_batch_count, &my_batches, sizeof(int), cudaMemcpyHostToDevice));
        NCCL_CHECK(ncclAllReduce(d_batch_count, d_batch_count, 1, ncclInt32, ncclMin, args.comm, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int synced_batches = 0;
        CUDA_CHECK(cudaMemcpy(&synced_batches, d_batch_count, sizeof(int), cudaMemcpyDeviceToHost));
        if (synced_batches < 1) synced_batches = 1;

        float epoch_loss = 0.0f, epoch_acc = 0.0f;
        int   batches = 0;

        nvtxRangePushA("Epoch");
        for (int step = 0; step < synced_batches && train_loader.HasNextBatch(); ++step) {
            nvtxRangePushA("DataLoad_H2D");
            train_loader.GetNextBatch(h_images, h_labels);
            CUDA_CHECK(cudaMemcpyAsync(d_images, h_images.data(), h_images.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_labels, h_labels.data(), h_labels.size() * sizeof(int),   cudaMemcpyHostToDevice, stream));
            nvtxRangePop();

            nvtxRangePushA("TrainStep");
            model.ZeroGrad();
            model.Forward(d_images, args.local_batch, 64, 64);
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

        // ── Validation (ALL GPUs participate to stay synchronized) ────────────
        float val_loss = 0.0f, val_acc = 0.0f;
        if (val_loader.NumSamples() > 0) {
            model.SetTraining(false);
            val_loader.Reset();
            int my_val_batches = (int)val_loader.NumSamples() / args.local_batch;
            float vl = 0, va = 0;
            int   vb = 0;
            for (int vs = 0; vs < my_val_batches && val_loader.HasNextBatch(); ++vs) {
                val_loader.GetNextBatch(h_images, h_labels);
                CUDA_CHECK(cudaMemcpy(d_images, h_images.data(), h_images.size() * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(), h_labels.size() * sizeof(int),   cudaMemcpyHostToDevice));
                model.Forward(d_images, args.local_batch, 64, 64);
                vl += model.Loss(d_labels, args.local_batch);
                va += model.Top1Accuracy(d_labels, args.local_batch);
                vb++;
            }
            if (vb > 0) {
                val_loss = vl / vb;
                val_acc  = va / vb;
            }
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

            // Save checkpoint only every save_every_n epochs (or at end)
            if (args.save_every_n > 0 && ((epoch + 1) % args.save_every_n == 0)) {
                std::string pt_path = args.checkpoint_dir + "/resnet50_multigpu_epoch" + std::to_string(epoch + 1) + ".bin";
                model.SaveWeights(pt_path);
                printf("  >> Checkpoint saved: %s\n", pt_path.c_str());
            }

            if (val_loader.NumSamples() > 0 && val_loss < best_val_loss) {
                best_val_loss = val_loss;
                patience_counter = 0;
                // We only save every save_every_n epochs now to prevent disk IO
                // model.SaveWeights(args.checkpoint_dir + "/resnet50_multigpu_best.bin");
            } else {
                patience_counter++;
                if (patience_counter >= args.patience) {
                    printf("\n=== Early stopping at epoch %d ===\n", epoch + 1);
                    g_stop.store(true);
                }
            }
        }

        // Broadcast stop signal using DEVICE memory (NCCL requires it)
        int h_stop = g_stop.load() ? 1 : 0;
        CUDA_CHECK(cudaMemcpy(d_stop_flag, &h_stop, sizeof(int), cudaMemcpyHostToDevice));
        NCCL_CHECK(ncclBcast(d_stop_flag, 1, ncclInt32, 0, args.comm, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(&h_stop, d_stop_flag, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_stop) break;
    }

    CUDA_CHECK(cudaFree(d_images));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_stop_flag));
    CUDA_CHECK(cudaFree(d_batch_count));
    CUDA_CHECK(cudaStreamDestroy(stream));

    if (rank == 0 && csv_log.is_open()) csv_log.close();
    } catch (const std::exception& e) {
        {
            std::lock_guard<std::mutex> lock(g_print_mutex);
            std::cerr << "[Rank " << rank << "] Fatal error: " << e.what() << "\n";
        }
        g_stop.store(true);
        ncclCommAbort(args.comm);
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    std::string data_dir  = (argc > 1) ? argv[1] : "./imagenet";
    int global_batch      = (argc > 2) ? std::atoi(argv[2]) : 256;
    int num_epochs        = (argc > 3) ? std::atoi(argv[3]) : 20;
    float base_lr         = (argc > 4) ? std::atof(argv[4]) : 0.1f;
    int num_classes       = 200;
    float momentum        = 0.9f;
    float weight_decay    = 1e-4f;
    int warmup_epochs     = 2;
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
        args.save_every_n    = 15;          // save checkpoint every 15 epochs
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
