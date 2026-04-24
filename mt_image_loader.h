#include "hip/hip_runtime.h"
#pragma once
// =============================================================================
// mt_image_loader.h — Multi-threaded ImageNet data loader
//
// Replaces the single-threaded stb_image decode with a thread pool that
// decodes multiple images in parallel. This is the most impactful change
// for GPU utilization — a single decode thread cannot feed an H100.
//
// Performance model (H100, ImageNet 224x224, batch=128):
//   1 thread:  ~110ms/batch → ~1,160 img/s → GPU util 28-42%
//   8 threads: ~15ms/batch  → ~8,500 img/s → GPU util 80-95%
//
// Usage:
//   MTImageLoader loader(dataset, wnids, batch_size, "train", true, rank, ws, 8);
//   while (loader.HasNextBatch()) {
//       loader.GetNextBatch(h_images, h_labels);
//       hipMemcpyAsync(d_images, h_images.data(), ...);
//       // train
//   }
//   loader.Reset();
//
// Can be combined with PrefetchLoader for double-buffered + multi-threaded:
//   MTImageLoader base(dataset, wnids, bs, "train", true, rank, ws, 8);
//   PrefetchLoader<MTImageLoader> loader(base, 3);
// =============================================================================

#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <thread>
#include <functional>

// ImageRecord must be defined before including this header.
// Include imagenet_loader.h (or equivalent) before mt_image_loader.h so
// the STB declarations and ImageRecord are available exactly once per TU.

class MTImageLoader {
public:
    MTImageLoader(const std::vector<ImageRecord>& full_dataset,
                  const std::vector<std::string>& wnids,
                  int batch_size, const std::string& split, bool augment,
                  int rank = 0, int world_size = 1,
                  int num_threads = 8)
        : batch_size_(batch_size)
        , current_idx_(0)
        , augment_(augment)
        , num_threads_(num_threads)
        , rng_(42 + rank)
    {
        (void)wnids;  // kept for interface compatibility
        num_classes_ = 0;

        // Shard dataset across GPUs
        for (size_t i = 0; i < full_dataset.size(); ++i) {
            if ((int)(i % world_size) == rank) {
                dataset_.push_back(full_dataset[i]);
            }
            if (full_dataset[i].label + 1 > num_classes_)
                num_classes_ = full_dataset[i].label + 1;
        }

        // Shuffle for training
        if (augment) {
            std::shuffle(dataset_.begin(), dataset_.end(), rng_);
        }
    }

    bool HasNextBatch() const {
        return current_idx_ + batch_size_ <= (int)dataset_.size();
    }

    void GetNextBatch(std::vector<float>& out_images, std::vector<int>& out_labels) {
        const int B = std::min(batch_size_, (int)(dataset_.size() - current_idx_));
        const int C = 3, H = 224, W = 224;

        out_images.resize(batch_size_ * C * H * W, 0.0f);
        out_labels.resize(batch_size_, 0);

        // Fill labels
        for (int b = 0; b < B; ++b) {
            out_labels[b] = dataset_[current_idx_ + b].label;
        }

        // Pre-generate random values for augmentation (RNG is not thread-safe)
        std::vector<float> flip_vals(B), scale_vals(B), aspect_vals(B);
        std::vector<float> crop_x_vals(B), crop_y_vals(B);
        if (augment_) {
            std::uniform_real_distribution<float> u01(0.0f, 1.0f);
            std::uniform_real_distribution<float> scale_dist(0.08f, 1.0f);
            std::uniform_real_distribution<float> aspect_dist(0.75f, 1.333333f);
            for (int b = 0; b < B; ++b) {
                flip_vals[b] = u01(rng_);
                scale_vals[b] = scale_dist(rng_);
                aspect_vals[b] = aspect_dist(rng_);
                crop_x_vals[b] = u01(rng_);
                crop_y_vals[b] = u01(rng_);
            }
        }

        // Decode images in parallel using a thread pool
        int actual_threads = std::min(num_threads_, B);
        std::vector<std::thread> threads(actual_threads);

        auto worker = [&](int tid) {
            for (int b = tid; b < B; b += actual_threads) {
                DecodeOneImage(
                    dataset_[current_idx_ + b].path,
                    out_images.data() + b * C * H * W,
                    augment_,
                    augment_ ? flip_vals[b] > 0.5f : false,
                    augment_ ? scale_vals[b] : 0.0f,
                    augment_ ? aspect_vals[b] : 0.0f,
                    augment_ ? crop_x_vals[b] : 0.0f,
                    augment_ ? crop_y_vals[b] : 0.0f
                );
            }
        };

        for (int t = 0; t < actual_threads; ++t) {
            threads[t] = std::thread(worker, t);
        }
        for (auto& t : threads) {
            t.join();
        }

        current_idx_ += B;
    }

    void Reset() {
        current_idx_ = 0;
        if (augment_) {
            std::shuffle(dataset_.begin(), dataset_.end(), rng_);
        }
    }

    size_t NumSamples() const { return dataset_.size(); }
    int    NumClasses() const { return num_classes_; }

private:
    static void DecodeOneImage(const std::string& path, float* dst,
                               bool augment, bool do_hflip,
                               float scale_val, float aspect_val,
                               float crop_x_val, float crop_y_val) {
        const int H = 224, W = 224, C = 3;

        int orig_w, orig_h, orig_comp;
        unsigned char* img = stbi_load(path.c_str(), &orig_w, &orig_h, &orig_comp, 3);
        if (!img) {
            std::memset(dst, 0, C * H * W * sizeof(float));
            return;
        }

        std::vector<unsigned char> crop_buf(H * W * C, 0);

        if (augment) {
            // RandomResizedCrop
            float target_area = orig_w * orig_h * scale_val;
            float target_aspect = aspect_val;
            int crop_w = (int)std::round(std::sqrt(target_area * target_aspect));
            int crop_h = (int)std::round(std::sqrt(target_area / target_aspect));
            if (crop_w > orig_w) crop_w = orig_w;
            if (crop_h > orig_h) crop_h = orig_h;

            int start_x = (int)(crop_x_val * (orig_w - crop_w));
            int start_y = (int)(crop_y_val * (orig_h - crop_h));

            std::vector<unsigned char> src_crop(crop_w * crop_h * 3);
            for (int y = 0; y < crop_h; ++y) {
                std::memcpy(src_crop.data() + y * crop_w * 3,
                           img + ((start_y + y) * orig_w + start_x) * 3,
                           crop_w * 3);
            }

            stbir_resize_uint8_linear(src_crop.data(), crop_w, crop_h, 0,
                                      crop_buf.data(), W, H, 0, (stbir_pixel_layout)3);
        } else {
            // Validation: resize shortest side to 256, center crop 224
            float scale = 256.0f / std::min(orig_w, orig_h);
            int rw = (int)(orig_w * scale);
            int rh = (int)(orig_h * scale);

            std::vector<unsigned char> resized(rw * rh * 3);
            stbir_resize_uint8_linear(img, orig_w, orig_h, 0,
                                      resized.data(), rw, rh, 0, (stbir_pixel_layout)3);

            int sx = std::max(0, (rw - W) / 2);
            int sy = std::max(0, (rh - H) / 2);

            for (int y = 0; y < H && (sy + y) < rh; ++y) {
                std::memcpy(crop_buf.data() + y * W * 3,
                           resized.data() + ((sy + y) * rw + sx) * 3,
                           std::min(W, rw - sx) * 3);
            }
        }
        stbi_image_free(img);

        // HWC→CHW + ImageNet normalization
        static const float mean[3] = {0.485f, 0.456f, 0.406f};
        static const float istd[3] = {1.0f/0.229f, 1.0f/0.224f, 1.0f/0.225f};

        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    int src_w = do_hflip ? (W - 1 - w) : w;
                    float val = crop_buf[(h * W + src_w) * 3 + c] / 255.0f;
                    dst[(c * H + h) * W + w] = (val - mean[c]) * istd[c];
                }
            }
        }
    }

    std::vector<ImageRecord> dataset_;
    int batch_size_;
    int current_idx_;
    bool augment_;
    int num_threads_;
    int num_classes_;
    std::mt19937 rng_;
};
