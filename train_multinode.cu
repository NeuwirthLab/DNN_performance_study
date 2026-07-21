#include "hip/hip_runtime.h"
/*
 * train_multinode.cu  -  ResNet-50 MULTI-NODE multi-GPU DDP training (RCCL)
 *
 * Multi-process sibling of train_multigpu.cu. One process per GPU (launched by
 * `srun --ntasks=N --ntasks-per-node=8`); the proven single-node trainer
 * (train_multigpu.cu, one process + N threads + ncclCommInitAll) is left
 * untouched.
 *
 * Differences vs train_multigpu.cu:
 *   - NCCL/RCCL bootstrap is ncclGetUniqueId (global rank 0) + a shared-FILE id
 *     handshake (no MPI dependency) + ncclCommInitRank(world_size, id, rank)
 *     on every rank, instead of single-process ncclCommInitAll.
 *   - rank / world_size come from SLURM env (SLURM_PROCID / SLURM_NTASKS); the
 *     GPU is bound by SLURM_LOCALID (rank 8+k on node 1 -> local GPU k).
 *   - Each process runs exactly ONE rank's training body (no std::thread).
 * Everything else (model, collectives, loader sharding) is identical and reused
 * unchanged: ResNet50::{SyncGradients,BroadcastWeights}, the batch-count
 * min-allreduce, the stop-flag bcast, and ImageNetLoader's (rank,world_size)
 * stride-sharding all operate on the global 16-GPU communicator transparently.
 *
 * NCCL->RCCL is NOT a macro shim: hip/gpu_compat.h does #include <rccl/rccl.h>,
 * so the nccl* names ARE the RCCL functions (RCCL 2.22.3 API on ROCm 6.4.0,
 * verified to expose ncclGetUniqueId/ncclCommInitRank/ncclUniqueId).
 *
 * Data loader: rocAL (build -DUSE_ROCAL=ON -DROCAL_ROOT=...). The CPU prefetch
 * path needs prefetch_loader.h, which is absent from the repo; rocAL is the
 * working loader and its 1-process-per-GPU use here avoids the multi-GPU race.
 *
 * Build:  cmake -DUSE_NCCL=ON -DUSE_ROCAL=ON -DROCAL_ROOT=... ..  (no MPI needed)
 *         make -j train_multinode
 * Usage:  srun ... ./train_multinode [data_dir] [global_batch] [epochs] [lr]
 *         env NCCL_ID_FILE=/work/.../nccl_id.$SLURM_JOB_ID   (shared-FS path)
 */

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "imagenet_loader.h"
#if defined(USE_ROCAL)
#include "rocal_imagenet_loader.h"
#else
#include "mt_image_loader.h"
#include "prefetch_loader.h"   // NOTE: absent from repo — build with -DUSE_ROCAL=ON
#endif

#if defined(USE_ROCAL)
#define RN50_GPU_LOADER 1
#endif
#include "resnet50.cuh"

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>
#include <string>
#include <filesystem>
#include <chrono>
#include <thread>
#include <atomic>
#include <cstring>
#include <limits>
#include <unistd.h>
#ifdef USE_MPI
#include <mpi.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── NCCL/RCCL error-checking macro ────────────────────────────────────────────
#define NCCL_CHECK(cmd) do {                                        \
    ncclResult_t _r = (cmd);                                        \
    if (_r != ncclSuccess) {                                        \
        fprintf(stderr, "NCCL error at %s:%d — %s\n",              \
                __FILE__, __LINE__, ncclGetErrorString(_r));         \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)

namespace fs = std::filesystem;

static std::atomic<bool> g_stop{false};
static constexpr int kTrainMetricSampleBatches = 4;

static int ProgressLogInterval() {
    const char* env_value = std::getenv("RN50_PROGRESS_INTERVAL");
    if (!env_value || env_value[0] == '\0') return 100;
    int interval = std::atoi(env_value);
    return interval > 0 ? interval : 100;
}

static int EnvInt(const char* name, int fallback) {
    const char* v = std::getenv(name);
    if (!v || v[0] == '\0') return fallback;
    int x = std::atoi(v);
    return x;
}

static size_t ShardedSampleCount(size_t total_samples, int rank, int world_size) {
    if (world_size <= 1) return total_samples;
    if (rank < 0 || rank >= world_size || static_cast<size_t>(rank) >= total_samples) return 0;
    return ((total_samples - 1 - static_cast<size_t>(rank)) / static_cast<size_t>(world_size)) + 1;
}

// ── Cosine annealing with linear warmup (same as single-GPU training) ─────────
static float cosine_lr(float base_lr, int epoch, int num_epochs, int warmup_epochs) {
    if (epoch < warmup_epochs)
        return base_lr * ((float)(epoch + 1) / (float)warmup_epochs);
    int t = epoch - warmup_epochs;
    int T = num_epochs - warmup_epochs;
    if (T <= 0) return base_lr;
    return base_lr * 0.5f * (1.0f + cosf((float)M_PI * t / T));
}

// ── Shared-file ncclUniqueId handshake (no MPI) ───────────────────────────────
// global rank 0 mints the id and writes it atomically (tmp + rename) to a path
// on the SHARED /work filesystem (visible to both nodes); every other rank polls
// until the file is the full 128 bytes, then all call ncclCommInitRank. The id
// embeds rank 0's bootstrap address, so RCCL connects over NCCL_SOCKET_IFNAME.
static void exchange_nccl_id(ncclUniqueId* id, int global_rank, const std::string& id_file) {
#ifdef USE_MPI
    if (global_rank == 0) NCCL_CHECK(ncclGetUniqueId(id));
    MPI_Bcast(id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    return;
#else
    if (global_rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(id));
        const std::string tmp = id_file + ".tmp." + std::to_string((long)getpid());
        std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
        if (!ofs) { fprintf(stderr, "[rank0] cannot write NCCL id tmp %s\n", tmp.c_str()); exit(1); }
        ofs.write(reinterpret_cast<const char*>(id), sizeof(ncclUniqueId));
        ofs.close();
        std::remove(id_file.c_str());                 // drop any stale id (defensive)
        if (std::rename(tmp.c_str(), id_file.c_str()) != 0) {
            fprintf(stderr, "[rank0] rename %s -> %s failed\n", tmp.c_str(), id_file.c_str());
            exit(1);
        }
    } else {
        const double timeout_s = 180.0;
        auto t0 = std::chrono::steady_clock::now();
        for (;;) {
            std::ifstream ifs(id_file, std::ios::binary | std::ios::ate);
            if (ifs) {
                std::streamoff sz = ifs.tellg();
                if (sz == static_cast<std::streamoff>(sizeof(ncclUniqueId))) {
                    ifs.seekg(0);
                    ifs.read(reinterpret_cast<char*>(id), sizeof(ncclUniqueId));
                    if (ifs.gcount() == static_cast<std::streamsize>(sizeof(ncclUniqueId))) break;
                }
            }
            double waited = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            if (waited > timeout_s) {
                fprintf(stderr, "[rank %d] timeout (%.0fs) waiting for NCCL id file %s\n",
                        global_rank, timeout_s, id_file.c_str());
                exit(1);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
#endif
}

// ── One rank's training body (was gpu_worker; now runs in its own process) ────
struct RankArgs {
    int         global_rank;    // 0..world_size-1  (sharding, seeds, rank-0 gates)
    int         local_rank;     // 0..gpus_per_node-1  (cudaSetDevice on this node)
    int         world_size;     // total GPUs across all nodes
    ncclComm_t  comm;
    std::string data_dir;
    int         local_batch;
    int         num_epochs;
    float       base_lr;
    int         num_classes;
    float       momentum;
    float       weight_decay;
    int         warmup_epochs;
    int         patience;
    int         decode_threads;
    std::string checkpoint_dir;
    const std::vector<ImageRecord>* train_dataset;
    const std::vector<ImageRecord>* val_dataset;
    const std::vector<std::string>* wnids;
};

void run_rank(RankArgs args) {
    const int rank      = args.global_rank;   // global rank: sharding/seed/log gate
    const int num_gpus  = args.world_size;    // GLOBAL world size for batch math + sharding
    // Eval-mode BN inference (miopenBatchNormalizationForwardInference) is not
    // implemented for BF16 and errors ("Empty code object") for FP32 on this
    // MIOpen build, so the post-epoch train-metrics + validation phases crash.
    // They are not needed for throughput/energy runs and are local-only (no
    // cross-rank collectives), so skip them by default (RN50_SKIP_EVAL=0 re-enables).
    const bool skip_eval = EnvInt("RN50_SKIP_EVAL", 1) != 0;

    CUDA_CHECK(cudaSetDevice(args.local_rank));  // bind node-local GPU (NOT global rank)

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    ResNet50 model(args.num_classes);
    model.SetStream(stream);
    model.InitWeights();

#ifndef USE_HIP
    cublasSetMathMode(model.Cublas(), CUBLAS_TF32_TENSOR_OP_MATH);
#endif

    // ── Broadcast weights from global rank 0 to all GPUs ───────────────────────
    model.BroadcastWeights(args.comm, stream);
    if (rank == 0) printf("[Rank 0] Weights broadcast to all %d GPUs\n", num_gpus);

    // ── Data loaders — shard by GLOBAL rank / world_size; decode on LOCAL GPU ───
    // DEFAULT is the CPU turbojpeg path (PrefetchLoader<MTImageLoader>): it is
    // ~2.1x faster than rocAL on MI300X (2780 vs 1327 img/s, bench_loaders job
    // 52946477) and prefetch_loader.h is present in the repo. rocAL (USE_ROCAL)
    // remains an opt-in for GPU-decode experiments, run per-process single-GPU
    // to sidestep its known multi-GPU thread race. The ONLY multinode change vs
    // train_multigpu: device_id is the node-local GPU (args.local_rank), while
    // shard_id/shard_count are the GLOBAL rank/world.
    printf("[Rank %d] Creating data loaders (local_gpu=%d) ...\n", rank, args.local_rank);
    int decode_threads = args.decode_threads;
#if defined(USE_ROCAL)
    // rocAL scans the source dir tree and load-balances across (shard_id, shard_count).
    const std::string train_dir = args.data_dir + "/train";
    const std::string val_dir   = args.data_dir + "/val";
    const size_t train_samples  = ShardedSampleCount(args.train_dataset->size(), rank, num_gpus);
    const size_t val_samples    = ShardedSampleCount(args.val_dataset->size(),   rank, num_gpus);
    RocalImageNetLoader train_loader(
        train_dir, train_samples, args.local_batch, decode_threads,
        /*device_id=*/args.local_rank, /*shard_id=*/rank, /*shard_count=*/num_gpus,
        /*prefetch_queue_depth=*/1, /*training=*/true);
    RocalImageNetLoader val_loader(
        val_dir, val_samples, args.local_batch, decode_threads,
        /*device_id=*/args.local_rank, /*shard_id=*/rank, /*shard_count=*/num_gpus,
        /*prefetch_queue_depth=*/1, /*training=*/false);
    printf("[Rank %d] rocAL train loader ready: %zu samples\n", rank, train_loader.NumSamples());
    printf("[Rank %d] rocAL val   loader ready: %zu samples\n", rank, val_loader.NumSamples());
#else
    // CPU/turbojpeg path (default) — PrefetchLoader<MTImageLoader> overlaps CPU
    // JPEG decode with GPU compute. Fastest loader on MI300X (see bench above).
    MTImageLoader train_loader_base(*args.train_dataset, *args.wnids,
                                    args.local_batch, "train", true, rank, num_gpus,
                                    decode_threads);
    printf("[Rank %d] Train loader ready: %d samples (%d decode threads)\n",
           rank, (int)train_loader_base.NumSamples(), decode_threads);
    MTImageLoader val_loader(*args.val_dataset, *args.wnids,
                             args.local_batch, "val", false, rank, num_gpus,
                             decode_threads);
    printf("[Rank %d] Val loader ready: %d samples\n", rank, (int)val_loader.NumSamples());
    PrefetchLoader<MTImageLoader> train_loader(train_loader_base, /*queue_depth=*/3);
#endif

    // ── GPU buffers ───────────────────────────────────────────────────────────
    float* d_images = nullptr;
    int*   d_labels = nullptr;
    CUDA_CHECK(cudaMalloc(&d_images, args.local_batch * 3 * 224 * 224 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, args.local_batch * sizeof(int)));

    int* d_stop_flag = nullptr;
    int* d_batch_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_stop_flag, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_batch_count, sizeof(int)));

    std::vector<float> h_images;
    std::vector<int>   h_labels;
#ifndef RN50_GPU_LOADER
    float* h_images_pinned = nullptr;
    int* h_labels_pinned = nullptr;
    const size_t image_elements = static_cast<size_t>(args.local_batch) * 3 * 224 * 224;
    const size_t label_elements = static_cast<size_t>(args.local_batch);
    CUDA_CHECK(hipHostMalloc(&h_images_pinned, image_elements * sizeof(float),
                             hipHostMallocPortable | hipHostMallocMapped));
    CUDA_CHECK(hipHostMalloc(&h_labels_pinned, label_elements * sizeof(int),
                             hipHostMallocPortable | hipHostMallocMapped));

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

    // ── Ring-buffered staging (RN50_RINGBUF=1) ─────────────────────────────────
    // Port of experiments/findmode/ringbuf_concept_test.cu (+8.8% single-GPU):
    // a ring of pinned-host + device buffer slots, a dedicated copy stream, and
    // two events per slot let the CPU run up to ring_depth steps ahead with no
    // per-step host sync. Pair with RN50_NO_STEP_SYNC=1, which skips the stream
    // sync at the end of SyncGradients — without the ring, that sync is what
    // makes the single pinned buffer above safe to overwrite.
    const bool ringbuf = EnvInt("RN50_RINGBUF", 0) != 0;
    const int ring_depth = std::max(2, EnvInt("RN50_RING_DEPTH", 3));
    struct StageSlot { float* h_img; int* h_lbl; float* d_img; int* d_lbl;
                       cudaEvent_t e_copy, e_consumed; };
    std::vector<StageSlot> ring;
    cudaStream_t copy_stream = nullptr;
    if (ringbuf) {
        CUDA_CHECK(cudaStreamCreate(&copy_stream));
        ring.resize(ring_depth);
        for (auto& s : ring) {
            CUDA_CHECK(hipHostMalloc(&s.h_img, image_elements * sizeof(float),
                                     hipHostMallocPortable | hipHostMallocMapped));
            CUDA_CHECK(hipHostMalloc(&s.h_lbl, label_elements * sizeof(int),
                                     hipHostMallocPortable | hipHostMallocMapped));
            CUDA_CHECK(cudaMalloc(&s.d_img, image_elements * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&s.d_lbl, label_elements * sizeof(int)));
            CUDA_CHECK(cudaEventCreateWithFlags(&s.e_copy,     cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&s.e_consumed, cudaEventDisableTiming));
            // Pre-record so the first ring_depth iterations' waits are no-ops.
            CUDA_CHECK(cudaEventRecord(s.e_copy,     stream));
            CUDA_CHECK(cudaEventRecord(s.e_consumed, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (rank == 0) printf("[Rank 0] Ring-buffer staging ON (depth=%d, no_step_sync=%s)\n",
                              ring_depth, EnvInt("RN50_NO_STEP_SYNC", 0) ? "yes" : "NO — set RN50_NO_STEP_SYNC=1");
    }
#endif

    std::ofstream csv_log;
    if (rank == 0) {
        fs::create_directories(args.checkpoint_dir);
        csv_log.open(args.checkpoint_dir + "/training_log_multinode.csv");
        csv_log << "epoch,lr,train_loss,train_acc,val_loss,val_acc,epoch_time_s,images_per_sec\n";
    }

    float best_val_loss = FLT_MAX;
    int   patience_counter = 0;
    int   start_epoch = 0;

    if (rank == 0) {
        printf("\nMulti-NODE Training: %d GPUs | Global Batch: %d | Local per GPU: %d\n",
               num_gpus, num_gpus * args.local_batch, args.local_batch);
        printf("Precision: %s\n", rn50_precision_label());
        printf("%-6s %-8s %-12s %-12s %-12s %-12s %-10s %-12s\n",
               "Epoch", "LR", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "Time(s)", "Img/sec");
        printf("--------------------------------------------------------------------------------------------\n");
    }

    const int progress_interval = ProgressLogInterval();

    for (int epoch = start_epoch; epoch < args.num_epochs && !g_stop.load(); ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        // wall-clock epoch-boundary marker for epoch-based power measurement.
        // Only global rank 0 emits it so the aggregated markers file stays clean.
        if (rank == 0) {
            double _pw_now = std::chrono::duration<double>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            printf("[PWRMARK] epoch %d %.3f\n", epoch, _pw_now);
            fflush(stdout);
        }
        float lr = cosine_lr(args.base_lr, epoch, args.num_epochs, args.warmup_epochs);

        train_loader.Reset();
        model.SetTraining(true);
    #ifndef RN50_GPU_LOADER
        train_loader.StartPrefetch();
    #endif

        // ── Synchronize batch count across ALL ranks (avoid NCCL deadlock) ──────
        int my_batches = (int)train_loader.NumSamples() / args.local_batch;
        CUDA_CHECK(cudaMemcpy(d_batch_count, &my_batches, sizeof(int), cudaMemcpyHostToDevice));
        NCCL_CHECK(ncclAllReduce(d_batch_count, d_batch_count, 1, ncclInt32, ncclMin, args.comm, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int synced_batches = 0;
        CUDA_CHECK(cudaMemcpy(&synced_batches, d_batch_count, sizeof(int), cudaMemcpyDeviceToHost));
        if (synced_batches < 1) synced_batches = 1;

        int batches = 0;
        nvtxRangePushA("Epoch");
        for (int step = 0; step < synced_batches && train_loader.HasNextBatch(); ++step) {
            nvtxRangePushA("DataLoad_H2D");
            float* step_images = d_images;
            int*   step_labels = d_labels;
#ifdef RN50_GPU_LOADER
            train_loader.GetNextBatch(d_images, d_labels, stream);
#else
            StageSlot* slot = nullptr;
            if (ringbuf) {
                slot = &ring[step % ring_depth];
                // Host guard: pinned buffers free only after their last H2D done.
                CUDA_CHECK(cudaEventSynchronize(slot->e_copy));
                train_loader.GetNextBatch(h_images, h_labels);
                std::memcpy(slot->h_img, h_images.data(),
                            std::min(image_elements, h_images.size()) * sizeof(float));
                std::memcpy(slot->h_lbl, h_labels.data(),
                            std::min(label_elements, h_labels.size()) * sizeof(int));
                // Device guard: don't overwrite d_img until prior compute read it.
                CUDA_CHECK(cudaStreamWaitEvent(copy_stream, slot->e_consumed, 0));
                CUDA_CHECK(cudaMemcpyAsync(slot->d_img, slot->h_img,
                                           image_elements * sizeof(float),
                                           cudaMemcpyHostToDevice, copy_stream));
                CUDA_CHECK(cudaMemcpyAsync(slot->d_lbl, slot->h_lbl,
                                           label_elements * sizeof(int),
                                           cudaMemcpyHostToDevice, copy_stream));
                CUDA_CHECK(cudaEventRecord(slot->e_copy, copy_stream));
                CUDA_CHECK(cudaStreamWaitEvent(stream, slot->e_copy, 0));
                step_images = slot->d_img;
                step_labels = slot->d_lbl;
            } else {
                train_loader.GetNextBatch(h_images, h_labels);
                stage_batch_to_device(h_images, h_labels);
            }
#endif
            nvtxRangePop();

            nvtxRangePushA("TrainStep");
            model.ZeroGrad();
            model.Forward(step_images, args.local_batch, 224, 224);
            model.Backward(step_labels, args.local_batch);
            nvtxRangePop();
#ifndef RN50_GPU_LOADER
            // Backward is the last reader of d_img (conv1 wgrad) and d_lbl.
            if (ringbuf) CUDA_CHECK(cudaEventRecord(slot->e_consumed, stream));
#endif

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

    #ifndef RN50_GPU_LOADER
        // With ring staging the host runs up to ring_depth steps ahead of the
        // GPU; drain before stopping the clock so epoch_sec times real work.
        if (ringbuf) CUDA_CHECK(cudaStreamSynchronize(stream));
        train_loader.StopPrefetch();
    #endif

        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_sec = std::chrono::duration<double>(epoch_end - epoch_start).count();
        double img_per_sec = (double)(num_gpus * batches * args.local_batch) / epoch_sec;

        float avg_train_loss = std::numeric_limits<float>::quiet_NaN();
        float avg_train_acc  = std::numeric_limits<float>::quiet_NaN();
        if (!skip_eval && batches > 0) {
            nvtxRangePushA("TrainMetricsSample");
            model.SetTraining(false);
            train_loader.Reset();
#ifndef RN50_GPU_LOADER
            train_loader.StartPrefetch();
#endif
            const int metric_batches = std::min(synced_batches, kTrainMetricSampleBatches);
            float sampled_loss = 0.0f, sampled_acc = 0.0f;
            int sampled_batches = 0;
            for (int step = 0; step < metric_batches && train_loader.HasNextBatch(); ++step) {
#ifdef RN50_GPU_LOADER
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
#ifndef RN50_GPU_LOADER
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

        // ── Validation (ALL ranks participate to stay collective-synchronized) ──
        float val_loss = 0.0f, val_acc = 0.0f;
        if (!skip_eval && val_loader.NumSamples() > 0) {
            model.SetTraining(false);
            val_loader.Reset();
            int my_val_batches = (int)val_loader.NumSamples() / args.local_batch;
            float vl = 0, va = 0; int vb = 0;
            for (int vs = 0; vs < my_val_batches && val_loader.HasNextBatch(); ++vs) {
#ifdef RN50_GPU_LOADER
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
            if (vb > 0) { val_loss = vl / vb; val_acc = va / vb; }
            model.SetTraining(true);
        }

        // ── Only global rank 0 prints, logs, and saves checkpoints ─────────────
        if (rank == 0) {
            printf("%-6d %-8.5f %-12.4f %-12.2f %-12.4f %-12.2f %-10.1f %-12.0f\n",
                   epoch + 1, lr, avg_train_loss, avg_train_acc * 100.0f,
                   val_loss, val_acc * 100.0f, epoch_sec, img_per_sec);
            csv_log << epoch + 1 << "," << lr << ","
                    << avg_train_loss << "," << avg_train_acc << ","
                    << val_loss << "," << val_acc << ","
                    << epoch_sec << "," << img_per_sec << "\n";
            csv_log.flush();

            std::string pt_path = args.checkpoint_dir + "/resnet50_multinode_epoch" + std::to_string(epoch) + ".bin";
            model.SaveWeights(pt_path);

            if (val_loader.NumSamples() > 0 && val_loss < best_val_loss) {
                best_val_loss = val_loss;
                patience_counter = 0;
                model.SaveWeights(args.checkpoint_dir + "/resnet50_multinode_best.bin");
                printf("  >> New best model (val_loss=%.4f)\n", val_loss);
            } else {
                patience_counter++;
                if (patience_counter >= args.patience) {
                    printf("\n=== Early stopping at epoch %d ===\n", epoch + 1);
                    g_stop.store(true);
                }
            }
        }

        // Broadcast stop signal from rank 0 to all ranks via DEVICE memory.
        int h_stop = g_stop.load() ? 1 : 0;
        CUDA_CHECK(cudaMemcpy(d_stop_flag, &h_stop, sizeof(int), cudaMemcpyHostToDevice));
        NCCL_CHECK(ncclBcast(d_stop_flag, 1, ncclInt32, 0, args.comm, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(&h_stop, d_stop_flag, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_stop) { g_stop.store(true); break; }
    }

    CUDA_CHECK(cudaFree(d_images));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_stop_flag));
    CUDA_CHECK(cudaFree(d_batch_count));
#ifndef RN50_GPU_LOADER
    if (h_images_pinned) hipHostFree(h_images_pinned);
    if (h_labels_pinned) hipHostFree(h_labels_pinned);
    for (auto& s : ring) {
        hipHostFree(s.h_img); hipHostFree(s.h_lbl);
        cudaFree(s.d_img);    cudaFree(s.d_lbl);
        cudaEventDestroy(s.e_copy); cudaEventDestroy(s.e_consumed);
    }
    if (copy_stream) CUDA_CHECK(cudaStreamDestroy(copy_stream));
#endif
    CUDA_CHECK(cudaStreamDestroy(stream));
    if (rank == 0 && csv_log.is_open()) csv_log.close();
}

// ── Main (one process == one rank) ────────────────────────────────────────────
int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

#ifdef USE_MPI
    MPI_Init(&argc, &argv);
#endif

    std::string data_dir  = (argc > 1) ? argv[1] : "./imagenet";
    int global_batch      = (argc > 2) ? std::atoi(argv[2]) : 4096;
    int num_epochs        = (argc > 3) ? std::atoi(argv[3]) : 5;
    float base_lr         = (argc > 4) ? std::atof(argv[4]) : 0.1f;
    int num_classes       = 1000;
    float momentum        = 0.9f;
    float weight_decay    = 1e-4f;
    int warmup_epochs     = 1;
    int patience          = num_epochs;   // early stop disabled for energy runs

    // ── Rank topology from SLURM (no MPI needed) ──────────────────────────────
    int global_rank = EnvInt("SLURM_PROCID",  0);
    int local_rank  = EnvInt("SLURM_LOCALID", 0);
    int world_size  = EnvInt("SLURM_NTASKS",  1);
    // Decode threads per rank: explicit RN50_DECODE_THREADS wins (tunable
    // independently of the task's cgroup size), else the task's CPU allowance.
    // One task == one GPU here, so no division by num_gpus (unlike multigpu).
    int decode_threads = EnvInt("RN50_DECODE_THREADS", 0);
    if (decode_threads < 1) decode_threads = EnvInt("SLURM_CPUS_PER_TASK", 8);
    if (decode_threads < 1) decode_threads = 8;
#ifdef USE_MPI
    { int mr=0, ms=0; MPI_Comm_rank(MPI_COMM_WORLD,&mr); MPI_Comm_size(MPI_COMM_WORLD,&ms);
      if (ms > 0) { global_rank = mr; world_size = ms; } }
#endif

    int local_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&local_gpus));
    if (local_gpus == 0) { std::cerr << "[rank] no GPUs visible to this task\n"; return 1; }

    // Pick this rank's node-local GPU, robust to either launch convention:
    //   - all node GPUs visible to every task  -> device == local_rank
    //   - SLURM bound exactly one GPU per task  -> it appears as device 0
    int local_dev = (local_gpus == 1) ? 0 : local_rank;
    if (local_dev >= local_gpus) {
        fprintf(stderr, "[rank %d] local_rank %d >= visible GPUs %d — launch with all "
                "node GPUs visible (no per-task gpu-bind) or exactly 1 GPU/task\n",
                global_rank, local_rank, local_gpus);
        return 1;
    }

    // Bind the GPU BEFORE ncclCommInitRank — RCCL associates the communicator
    // with the current device, so it must be set here (not later in run_rank),
    // else every rank would init its comm on device 0.
    CUDA_CHECK(cudaSetDevice(local_dev));

    int local_batch = global_batch / world_size;
    if (local_batch < 1) local_batch = 1;

    const char* idf = std::getenv("NCCL_ID_FILE");
    std::string id_file = (idf && idf[0]) ? idf : (fs::current_path() / "nccl_id.bin").string();

    if (global_rank == 0) {
        std::cout << "=== ResNet-50 MULTI-NODE Training (DDP via RCCL) ===\n";
        std::cout << "world_size       : " << world_size << " GPUs\n";
        std::cout << "global batch     : " << global_batch << " | per-GPU: " << local_batch << "\n";
        std::cout << "epochs           : " << num_epochs << " | base lr: " << base_lr << "\n";
        std::cout << "data dir         : " << data_dir << "\n";
        std::cout << "nccl id file     : " << id_file << "\n";
        std::cout << "decode threads   : " << decode_threads << "\n\n";
    }
    printf("[rank %d] local_rank=%d world_size=%d host-visible-gpus=%d\n",
           global_rank, local_rank, world_size, local_gpus);

    // ── RCCL communicator: shared-file id handshake + ncclCommInitRank ─────────
    ncclUniqueId id;
    exchange_nccl_id(&id, global_rank, id_file);
    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, world_size, id, global_rank));
    if (global_rank == 0) printf("[rank 0] ncclCommInitRank OK across %d ranks\n", world_size);

    // ── Pre-scan dataset paths (deterministic; identical across ranks) ─────────
    if (global_rank == 0) std::cout << "Pre-scanning dataset paths...\n";
    auto [train_dataset, train_wnids] = ImageNetLoader::PreScanPaths(data_dir, "train");
    auto [val_dataset,   val_wnids]   = ImageNetLoader::PreScanPaths(data_dir, "val");
    const auto& wnids = train_wnids;

    if (global_rank == 0) fs::create_directories("checkpoints");

    RankArgs args;
    args.global_rank   = global_rank;
    args.local_rank    = local_dev;   // node-local GPU index (for cudaSetDevice + rocAL device_id)
    args.world_size    = world_size;
    args.comm          = comm;
    args.data_dir      = data_dir;
    args.local_batch   = local_batch;
    args.num_epochs    = num_epochs;
    args.base_lr       = base_lr;
    args.num_classes   = num_classes;
    args.momentum      = momentum;
    args.weight_decay  = weight_decay;
    args.warmup_epochs = warmup_epochs;
    args.patience      = patience;
    args.decode_threads = decode_threads;
    args.checkpoint_dir = "checkpoints";
    args.train_dataset = &train_dataset;
    args.val_dataset   = &val_dataset;
    args.wnids         = &wnids;

    run_rank(args);

    ncclCommDestroy(comm);
    if (global_rank == 0) {
        std::remove(id_file.c_str());   // tidy up the rendezvous file
        std::cout << "\nMulti-node training complete. Checkpoints in: checkpoints/\n";
    }
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
