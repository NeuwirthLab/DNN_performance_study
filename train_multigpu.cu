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
#include "imagenet_loader.h"
#ifdef USE_DALI
#include "dali_imagenet_loader.h"
#else
#include "mt_image_loader.h"
#include "prefetch_loader.h"
#endif
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
#include <cstring>
#include <limits>

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
static constexpr int kTrainMetricSampleBatches = 4;

static int ProgressLogInterval() {
    const char* env_value = std::getenv("RN50_PROGRESS_INTERVAL");
    if (!env_value || env_value[0] == '\0') return 100;
    int interval = std::atoi(env_value);
    return interval > 0 ? interval : 100;
}

static size_t ShardedSampleCount(size_t total_samples, int rank, int world_size) {
    if (world_size <= 1) return total_samples;
    if (rank < 0 || rank >= world_size || static_cast<size_t>(rank) >= total_samples) return 0;
    return ((total_samples - 1 - static_cast<size_t>(rank)) / static_cast<size_t>(world_size)) + 1;
}

#ifdef USE_DALI
static std::string ResolveDaliAssetDir() {
    const char* env_dir = std::getenv("RN50_DALI_ASSET_DIR");
    if (env_dir && env_dir[0] != '\0') {
        return env_dir;
    }
    return (fs::current_path() / "dali_assets").string();
}

static std::string DaliPipelinePath(const std::string& asset_dir,
                                    const std::string& split,
                                    int rank) {
    return (fs::path(asset_dir) / (split + "_rank" + std::to_string(rank) + ".dali")).string();
}
#endif

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
    // Pre-scanned datasets (built once in main, shared across all workers)
    const std::vector<ImageRecord>* train_dataset;
    const std::vector<ImageRecord>* val_dataset;
    const std::vector<std::string>* wnids;
};

void gpu_worker(GPUWorkerArgs args) {
    const int rank      = args.rank;
    const int num_gpus  = args.num_gpus;

    CUDA_CHECK(cudaSetDevice(rank));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Each GPU has its own model and data loader (with different RNG seed)
    ResNet50 model(args.num_classes);
    model.SetStream(stream);
    model.InitWeights();

    // Enable TF32 on Ampere+ (NVIDIA-only; AMD MI300X picks precision automatically)
#ifndef USE_HIP
    cublasSetMathMode(model.Cublas(), CUBLAS_TF32_TENSOR_OP_MATH);
#endif

    // ── Broadcast weights from rank 0 to all GPUs ──────────────────────────
    model.BroadcastWeights(args.comm, stream);
    if (rank == 0) printf("[Rank 0] Weights broadcast to all GPUs\n");

    // ── Data loaders (using pre-scanned paths — no filesystem scanning here) ─
    printf("[Rank %d] Creating data loaders from pre-scanned paths...\n", rank);
    int decode_threads = 8;  // match cpus-per-task / num_gpus
#ifdef USE_DALI
    const std::string dali_asset_dir = ResolveDaliAssetDir();
    const size_t train_samples = ShardedSampleCount(args.train_dataset->size(), rank, num_gpus);
    const size_t val_samples = ShardedSampleCount(args.val_dataset->size(), rank, num_gpus);
    DaliImageNetLoader train_loader(
        DaliPipelinePath(dali_asset_dir, "train", rank),
        train_samples,
        args.local_batch,
        decode_threads,
        rank,
        2);
    DaliImageNetLoader val_loader(
        DaliPipelinePath(dali_asset_dir, "val", rank),
        val_samples,
        args.local_batch,
        decode_threads,
        rank,
        2);
    printf("[Rank %d] DALI train loader ready: %zu samples (asset dir: %s)\n",
           rank, train_loader.NumSamples(), dali_asset_dir.c_str());
    printf("[Rank %d] DALI val loader ready: %zu samples\n",
           rank, val_loader.NumSamples());
#else
    // Use multi-threaded loader: 8 threads decode images in parallel per batch.
    // This is the key fix for GPU utilization — single-threaded stb_image was the
    // bottleneck causing 28-42% GPU util on H100.
    MTImageLoader train_loader_base(*args.train_dataset, *args.wnids,
                                    args.local_batch, "train", true, rank, num_gpus,
                                    decode_threads);
    printf("[Rank %d] Train loader ready: %d samples (%d decode threads)\n",
           rank, (int)train_loader_base.NumSamples(), decode_threads);
    MTImageLoader val_loader(*args.val_dataset, *args.wnids,
                             args.local_batch, "val", false, rank, num_gpus,
                             decode_threads);
    printf("[Rank %d] Val loader ready: %d samples\n", rank, (int)val_loader.NumSamples());

    // Wrap train loader with prefetch — background thread fills queue with
    // multi-threaded decoded batches while GPU trains.
    PrefetchLoader<MTImageLoader> train_loader(train_loader_base, /*queue_depth=*/3);
    printf("[Rank %d] Prefetch loader active (queue_depth=3, %d decode threads)\n",
           rank, decode_threads);
#endif

    // ── GPU buffers ───────────────────────────────────────────────────────────
    float* d_images = nullptr;
    int*   d_labels = nullptr;
    CUDA_CHECK(cudaMalloc(&d_images, args.local_batch * 3 * 224 * 224 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, args.local_batch * sizeof(int)));

    // Device-side scratch for NCCL reductions (stop flag + batch count sync)
    int* d_stop_flag = nullptr;
    int* d_batch_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_stop_flag, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_batch_count, sizeof(int)));

    std::vector<float> h_images;
    std::vector<int>   h_labels;
#ifndef USE_DALI
    float* h_images_pinned = nullptr;
    int* h_labels_pinned = nullptr;
    const size_t image_elements = static_cast<size_t>(args.local_batch) * 3 * 224 * 224;
    const size_t label_elements = static_cast<size_t>(args.local_batch);
    CUDA_CHECK(hipHostMalloc(&h_images_pinned, image_elements * sizeof(float)));
    CUDA_CHECK(hipHostMalloc(&h_labels_pinned, label_elements * sizeof(int)));

    auto stage_batch_to_device = [&](const std::vector<float>& images, const std::vector<int>& labels) {
        if (images.size() > image_elements || labels.size() > label_elements) {
            throw std::runtime_error("Host staging buffer too small for decoded batch");
        }
        if (!images.empty()) {
            std::memcpy(h_images_pinned, images.data(), images.size() * sizeof(float));
            CUDA_CHECK(cudaMemcpyAsync(d_images, h_images_pinned,
                                       images.size() * sizeof(float),
                                       cudaMemcpyHostToDevice, stream));
        }
        if (!labels.empty()) {
            std::memcpy(h_labels_pinned, labels.data(), labels.size() * sizeof(int));
            CUDA_CHECK(cudaMemcpyAsync(d_labels, h_labels_pinned,
                                       labels.size() * sizeof(int),
                                       cudaMemcpyHostToDevice, stream));
        }
    };
#endif

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

    const int progress_interval = ProgressLogInterval();

    for (int epoch = start_epoch; epoch < args.num_epochs && !g_stop.load(); ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        float lr = cosine_lr(args.base_lr, epoch, args.num_epochs, args.warmup_epochs);

        train_loader.Reset();
        model.SetTraining(true);

        // Start prefetch thread for this epoch — begins loading batches immediately
    #ifndef USE_DALI
        train_loader.StartPrefetch();
    #endif

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

        int   batches = 0;

        nvtxRangePushA("Epoch");
        for (int step = 0; step < synced_batches && train_loader.HasNextBatch(); ++step) {
            nvtxRangePushA("DataLoad_H2D");
#ifdef USE_DALI
            train_loader.GetNextBatch(d_images, d_labels, stream);
#else
            train_loader.GetNextBatch(h_images, h_labels);
            stage_batch_to_device(h_images, h_labels);
#endif
            nvtxRangePop();

            nvtxRangePushA("TrainStep");
            model.ZeroGrad();
            model.Forward(d_images, args.local_batch, 224, 224);
            model.Backward(d_labels, args.local_batch);
            nvtxRangePop();

            nvtxRangePushA("SyncGradients");
            model.SyncGradients(args.comm, stream);
            nvtxRangePop();

            model.UpdateSGD(lr, args.momentum, args.weight_decay);
            batches++;

            if (rank == 0 && (batches % progress_interval == 0 || batches == synced_batches)) {
                auto progress_now = std::chrono::high_resolution_clock::now();
                double elapsed_sec = std::chrono::duration<double>(progress_now - epoch_start).count();
                double epoch_progress = synced_batches > 0
                    ? (100.0 * static_cast<double>(batches) / static_cast<double>(synced_batches))
                    : 0.0;
                double images_done = static_cast<double>(batches) * static_cast<double>(num_gpus * args.local_batch);
                double running_img_per_sec = elapsed_sec > 0.0 ? (images_done / elapsed_sec) : 0.0;
                double eta_sec = (batches > 0)
                    ? elapsed_sec * static_cast<double>(synced_batches - batches) / static_cast<double>(batches)
                    : 0.0;

                printf("[Epoch %d] Step %d/%d (%.1f%%) | elapsed %.1fs | %.0f img/s | ETA %.1fs\n",
                       epoch + 1, batches, synced_batches, epoch_progress,
                       elapsed_sec, running_img_per_sec, eta_sec);
            }
        }
        nvtxRangePop();

        // Stop prefetch thread before validation
    #ifndef USE_DALI
        train_loader.StopPrefetch();
    #endif

        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_sec = std::chrono::duration<double>(epoch_end - epoch_start).count();
        // Total images = (all GPUs) * batches * local_batch
        double img_per_sec = (double)(num_gpus * batches * args.local_batch) / epoch_sec;

        float avg_train_loss = std::numeric_limits<float>::quiet_NaN();
        float avg_train_acc  = std::numeric_limits<float>::quiet_NaN();
        if (batches > 0) {
            nvtxRangePushA("TrainMetricsSample");
            model.SetTraining(false);
            train_loader.Reset();
#ifndef USE_DALI
            train_loader.StartPrefetch();
#endif
            const int metric_batches = std::min(synced_batches, kTrainMetricSampleBatches);
            float sampled_loss = 0.0f;
            float sampled_acc = 0.0f;
            int sampled_batches = 0;
            for (int step = 0; step < metric_batches && train_loader.HasNextBatch(); ++step) {
#ifdef USE_DALI
                train_loader.GetNextBatch(d_images, d_labels, stream);
#else
                train_loader.GetNextBatch(h_images, h_labels);
                stage_batch_to_device(h_images, h_labels);
#endif
                model.Forward(d_images, args.local_batch, 224, 224);
                sampled_loss += model.Loss(d_labels, args.local_batch);
                sampled_acc  += model.Top1Accuracy(d_labels, args.local_batch);
                sampled_batches++;
            }
#ifndef USE_DALI
            train_loader.StopPrefetch();
#endif
            if (sampled_batches > 0) {
                avg_train_loss = sampled_loss / sampled_batches;
                avg_train_acc  = sampled_acc / sampled_batches;
            }
            train_loader.Reset();
            model.SetTraining(true);
            nvtxRangePop();
        }

        // ── Validation (ALL GPUs participate to stay synchronized) ────────────
        // Even though only rank 0 reports results, all GPUs must run the same
        // number of iterations to avoid NCCL deadlocks if any collective
        // calls happen during forward passes.
        float val_loss = 0.0f, val_acc = 0.0f;
        if (val_loader.NumSamples() > 0) {
            model.SetTraining(false);
            val_loader.Reset();
            int my_val_batches = (int)val_loader.NumSamples() / args.local_batch;
            float vl = 0, va = 0;
            int   vb = 0;
            for (int vs = 0; vs < my_val_batches && val_loader.HasNextBatch(); ++vs) {
#ifdef USE_DALI
                val_loader.GetNextBatch(d_images, d_labels, stream);
#else
                val_loader.GetNextBatch(h_images, h_labels);
                stage_batch_to_device(h_images, h_labels);
#endif
                model.Forward(d_images, args.local_batch, 224, 224);
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
#ifndef USE_DALI
    if (h_images_pinned) hipHostFree(h_images_pinned);
    if (h_labels_pinned) hipHostFree(h_labels_pinned);
#endif
    CUDA_CHECK(cudaStreamDestroy(stream));

    if (rank == 0 && csv_log.is_open()) csv_log.close();
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    // Disable stdout/stderr buffering so output appears in SLURM .out files
    // in real time instead of being held in an 8KB buffer
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    std::string data_dir  = (argc > 1) ? argv[1] : "./imagenet";
    int global_batch      = (argc > 2) ? std::atoi(argv[2]) : 256;
    int num_epochs        = (argc > 3) ? std::atoi(argv[3]) : 5;
    float base_lr         = (argc > 4) ? std::atof(argv[4]) : 0.1f;
    int num_classes       = 1000;
    float momentum        = 0.9f;
    float weight_decay    = 1e-4f;
    int warmup_epochs     = 1;
    int patience          = 2;

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

    // ── Pre-scan dataset paths ONCE in main thread ────────────────────────────
    // This avoids 2-4 GPU threads simultaneously scanning 1.28M files,
    // which causes filesystem thrashing and NCCL deadlocks on HPC.
    std::cout << "Pre-scanning dataset paths (single-threaded)...\n";
    auto [train_dataset, train_wnids] = ImageNetLoader::PreScanPaths(data_dir, "train");
    auto [val_dataset,   val_wnids]   = ImageNetLoader::PreScanPaths(data_dir, "val");
    // Use train wnids as the canonical class list (val reuses same mapping)
    const auto& wnids = train_wnids;
    std::cout << "Pre-scan complete. Launching GPU workers...\n\n";

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
        args.train_dataset   = &train_dataset;
        args.val_dataset     = &val_dataset;
        args.wnids           = &wnids;
        threads.emplace_back(gpu_worker, args);
    }

    for (auto& t : threads) t.join();

    // ── Cleanup NCCL ─────────────────────────────────────────────────────────
    for (int i = 0; i < num_gpus; ++i) ncclCommDestroy(comms[i]);

    std::cout << "\nMulti-GPU training complete.\n";
    std::cout << "Checkpoints in: checkpoints/\n";
    return 0;
}
