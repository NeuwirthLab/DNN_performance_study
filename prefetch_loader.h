#pragma once
// =============================================================================
// prefetch_loader.h — Double-buffered prefetch wrapper for data loaders
//
// Wraps any loader that has GetNextBatch()/HasNextBatch()/Reset() and runs
// a background thread that decodes batch N+1 while the GPU trains on batch N.
//
// Before prefetch:
//   CPU: [===LOAD===]............[===LOAD===]............
//   GPU: ............[==TRAIN==]............[==TRAIN==]
//   Utilization: ~28%
//
// After prefetch:
//   CPU: [===LOAD===][===LOAD===][===LOAD===]
//   GPU: ............[==TRAIN==][==TRAIN==][==TRAIN==]
//   Utilization: ~60-80% (overlap once pipeline fills)
//
// Usage:
//   ImageNetLoader base_loader(data_dir, batch_size, "train", true, rank, ws);
//   PrefetchLoader<ImageNetLoader> loader(base_loader, /*queue_depth=*/3);
//   loader.StartPrefetch();
//   while (loader.HasNextBatch()) {
//       auto batch = loader.GetNextBatch();
//       hipMemcpyAsync(d_images, batch.images.data(), ...);
//       // train...
//   }
//   loader.StopPrefetch();
//   loader.Reset();  // for next epoch
//
// Thread safety: GetNextBatch() is called from the training thread.
// The background thread only touches the internal loader and queue.
// =============================================================================

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <stdexcept>

// A single prefetched batch
struct PrefetchedBatch {
    std::vector<float> images;
    std::vector<int>   labels;
    bool               valid = false;  // false = end-of-data sentinel
};

template <typename LoaderT>
class PrefetchLoader {
public:
    // queue_depth: how many batches to decode ahead (2-3 is optimal)
    PrefetchLoader(LoaderT& loader, int queue_depth = 3)
        : loader_(loader)
        , queue_depth_(queue_depth)
        , running_(false)
        , epoch_done_(false)
    {}

    ~PrefetchLoader() {
        StopPrefetch();
    }

    // ── Start the background prefetch thread ─────────────────────────────
    void StartPrefetch() {
        if (running_.load()) return;  // already running
        running_  = true;
        epoch_done_ = false;
        prefetch_thread_ = std::thread(&PrefetchLoader::PrefetchWorker, this);
    }

    // ── Stop the background thread (call at end of epoch or on error) ────
    void StopPrefetch() {
        if (!running_.load()) return;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            running_ = false;
        }
        cv_space_.notify_all();  // wake worker if blocked on full queue
        cv_ready_.notify_all();  // wake consumer if blocked on empty queue
        if (prefetch_thread_.joinable()) {
            prefetch_thread_.join();
        }
        // Drain any remaining batches
        std::lock_guard<std::mutex> lk(mtx_);
        while (!queue_.empty()) queue_.pop();
    }

    // ── Check if more batches are available ──────────────────────────────
    bool HasNextBatch() {
        std::unique_lock<std::mutex> lk(mtx_);
        // Wait until there's something in the queue OR epoch is done with empty queue
        cv_ready_.wait(lk, [this] {
            return !queue_.empty() || (epoch_done_ && queue_.empty());
        });
        if (queue_.empty()) return false;
        return queue_.front().valid;
    }

    // ── Get the next batch (blocks until a prefetched batch is ready) ────
    // Returns immediately if the background thread has already loaded it.
    void GetNextBatch(std::vector<float>& out_images, std::vector<int>& out_labels) {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_ready_.wait(lk, [this] {
            return !queue_.empty() || !running_.load();
        });

        if (queue_.empty()) {
            throw std::runtime_error("PrefetchLoader: no more batches");
        }

        PrefetchedBatch batch = std::move(queue_.front());
        queue_.pop();
        lk.unlock();
        cv_space_.notify_one();  // signal worker: room in queue

        if (!batch.valid) {
            throw std::runtime_error("PrefetchLoader: end of data");
        }

        out_images = std::move(batch.images);
        out_labels = std::move(batch.labels);
    }

    // ── Reset for next epoch ─────────────────────────────────────────────
    void Reset() {
        StopPrefetch();
        loader_.Reset();
    }

    // ── Passthrough to underlying loader ─────────────────────────────────
    size_t NumSamples() const { return loader_.NumSamples(); }
    int    NumClasses() const { return loader_.NumClasses(); }

private:
    // ── Background worker: loads batches into the queue ──────────────────
    void PrefetchWorker() {
        while (running_.load() && loader_.HasNextBatch()) {
            PrefetchedBatch batch;
            batch.valid = true;

            // This is the expensive call — JPEG decode + resize + normalize
            // Runs on the background thread, NOT blocking the training thread
            try {
                loader_.GetNextBatch(batch.images, batch.labels);
            } catch (...) {
                batch.valid = false;
            }

            // Push into queue, blocking if queue is full
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_space_.wait(lk, [this] {
                    return (int)queue_.size() < queue_depth_ || !running_.load();
                });
                if (!running_.load()) break;
                queue_.push(std::move(batch));
            }
            cv_ready_.notify_one();
        }

        // Signal end-of-data
        {
            std::lock_guard<std::mutex> lk(mtx_);
            PrefetchedBatch sentinel;
            sentinel.valid = false;
            queue_.push(std::move(sentinel));
            epoch_done_ = true;
        }
        cv_ready_.notify_all();
    }

    LoaderT&                    loader_;
    int                         queue_depth_;
    std::atomic<bool>           running_;
    bool                        epoch_done_;

    std::thread                 prefetch_thread_;
    std::queue<PrefetchedBatch> queue_;
    std::mutex                  mtx_;
    std::condition_variable     cv_ready_;   // consumer waits: batch available
    std::condition_variable     cv_space_;   // producer waits: room in queue
};
