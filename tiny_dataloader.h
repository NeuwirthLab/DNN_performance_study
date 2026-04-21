#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>
#include <random>

// Define this ONLY ONCE in one .cpp/.cu file before including stb_image.h there
// #define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// ─────────────────────────────────────────────────────────────────────────────
// Tiny ImageNet DataLoader
//
// Reads JPEG images from the Tiny ImageNet directory structure.
// Expects the path to point to 'tiny-imagenet-200'.
// Tiny ImageNet images are 64x64 pixels.
//
// Supports:
//   - Training data augmentation (random horizontal flip)
//   - Validation set loading (val_annotations.txt)
// ─────────────────────────────────────────────────────────────────────────────

struct ImageRecord {
    std::string path;
    int label;
};

class TinyImageNetLoader {
public:
    // split: "train" or "val"
    // augment: enable random horizontal flip (only for training)
    TinyImageNetLoader(const std::string& data_dir, int batch_size,
                       const std::string& split = "train", bool augment = false,
                       int rank = 0, int world_size = 1)
        : data_dir_(data_dir), batch_size_(batch_size), current_idx_(0),
          augment_(augment), rank_(rank), world_size_(world_size),
          rng_(42 + rank)
    {
        LoadWnids();
        if (split == "val")
            LoadValPaths();
        else
            LoadTrainPaths();
        
        ShardDataset();
        Shuffle();
    }

    bool HasNextBatch() const {
        return current_idx_ + batch_size_ <= dataset_.size();
    }

    void Reset() {
        current_idx_ = 0;
        Shuffle();
    }

    int NumSamples() const {
        return dataset_.size();
    }

    int NumClasses() const {
        return wnids_.size();
    }

    // Fills out_images with [batch_size * 3 * 64 * 64] floats [-1, 1]
    // Fills out_labels with [batch_size] ints
    void GetNextBatch(std::vector<float>& out_images, std::vector<int>& out_labels) {
        if (!HasNextBatch()) {
            throw std::runtime_error("No more full batches available.");
        }

        const int H = 64;
        const int W = 64;
        const int C = 3;

        out_images.resize(batch_size_ * C * H * W);
        out_labels.resize(batch_size_);

        std::uniform_real_distribution<float> flip_dist(0.0f, 1.0f);

        for (int b = 0; b < batch_size_; ++b) {
            int idx = current_idx_ + b;
            const auto& record = dataset_[idx];
            out_labels[b] = record.label;

            int width, height, channels;
            // Force 3 channels (RGB)
            unsigned char* img = stbi_load(record.path.c_str(), &width, &height, &channels, 3);
            if (!img) {
                std::cerr << "Failed to load image: " << record.path << "\n";
                // Fill with zeros on failure
                std::fill(out_images.begin() + b * C * H * W,
                          out_images.begin() + (b + 1) * C * H * W, 0.0f);
                continue;
            }

            bool do_hflip = augment_ && (flip_dist(rng_) > 0.5f);

            // Convert to CHW format and normalize to [-1.0, 1.0]
            float* dst = out_images.data() + b * C * H * W;
            for (int c = 0; c < 3; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        int src_w = do_hflip ? (W - 1 - w) : w;
                        int src_idx = (h * W + src_w) * 3 + c;
                        int dst_idx = (c * H + h) * W + w;
                        dst[dst_idx] = (img[src_idx] / 127.5f) - 1.0f;
                    }
                }
            }
            stbi_image_free(img);
        }
        current_idx_ += batch_size_;
    }

private:
    std::string data_dir_;
    int batch_size_;
    size_t current_idx_;
    bool augment_;
    int rank_;
    int world_size_;
    std::mt19937 rng_;

    std::vector<std::string> wnids_;
    std::unordered_map<std::string, int> wnid_to_label_;
    std::vector<ImageRecord> dataset_;

    void LoadWnids() {
        std::string wnids_path = data_dir_ + "/wnids.txt";
        std::ifstream file(wnids_path);
        if (!file.is_open()) {
            std::cerr << "Warning: Could not open " << wnids_path << ". Ensure directory is correct.\n";
            return;
        }
        std::string wnid;
        while (file >> wnid) {
            int label = (int)wnids_.size();
            wnid_to_label_[wnid] = label;
            wnids_.push_back(wnid);
        }
    }

    void LoadTrainPaths() {
        for (int label = 0; label < (int)wnids_.size(); ++label) {
            for (int img_id = 0; img_id < 500; ++img_id) {
                std::string path = data_dir_ + "/train/" + wnids_[label] + "/images/" +
                                   wnids_[label] + "_" + std::to_string(img_id) + ".JPEG";
                dataset_.push_back({path, label});
            }
        }
        std::cout << "Discovered " << dataset_.size() << " training images across " << wnids_.size() << " classes.\n";
    }

    void LoadValPaths() {
        std::string ann_path = data_dir_ + "/val/val_annotations.txt";
        std::ifstream file(ann_path);
        if (!file.is_open()) {
            std::cerr << "Warning: Could not open " << ann_path << "\n";
            return;
        }
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string filename, wnid;
            if (!(iss >> filename >> wnid)) continue;
            auto it = wnid_to_label_.find(wnid);
            if (it == wnid_to_label_.end()) continue;
            std::string path = data_dir_ + "/val/images/" + filename;
            dataset_.push_back({path, it->second});
        }
        std::cout << "Discovered " << dataset_.size() << " validation images.\n";
    }

    void ShardDataset() {
        if (world_size_ <= 1) return;
        std::sort(dataset_.begin(), dataset_.end(),
                  [](const ImageRecord& a, const ImageRecord& b) {
                      return a.path < b.path;
                  });
        std::vector<ImageRecord> shard;
        shard.reserve(dataset_.size() / world_size_ + 1);
        for (size_t i = rank_; i < dataset_.size(); i += world_size_) {
            shard.push_back(std::move(dataset_[i]));
        }
        dataset_ = std::move(shard);
        std::cout << "[Rank " << rank_ << "] Data shard: " << dataset_.size() << " samples\n";
    }

    void Shuffle() {
        std::shuffle(dataset_.begin(), dataset_.end(), rng_);
    }
};
