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
#include <filesystem>

// Define this ONLY ONCE in one .cpp/.cu file before including stb_image.h there
// #define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Define this ONLY ONCE in one .cpp/.cu file before including stb_image_resize2.h there
// #define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// ImageNet-1k DataLoader
//
// Reads JPEG images from the standard ImageNet directory structure.
// Expects:
//   data_dir/train/<wnid>/*.JPEG
//   data_dir/val/<wnid>/*.JPEG  OR a flat folder structure handled via val_annotations.
// 
// Images are variable size.
// Supports:
//   - Training: RandomResizedCrop(224) + horizontal flip
//   - Validation: Resize shortest side to 256, then CenterCrop(224)
// ─────────────────────────────────────────────────────────────────────────────

struct ImageRecord {
    std::string path;
    int label;
};

class ImageNetLoader {
public:
    // split: "train" or "val"
    // augment: enable RandomResizedCrop + random horizontal flip (only for training)
    ImageNetLoader(const std::string& data_dir, int batch_size,
                   const std::string& split = "train", bool augment = false)
        : data_dir_(data_dir), split_(split), batch_size_(batch_size), current_idx_(0),
          augment_(augment), rng_(std::random_device{}())
    {
        LoadPaths();
        Shuffle();
    }

    size_t NumSamples() const { return dataset_.size(); }
    int NumClasses() const { return (int)wnids_.size(); }
    void Reset() {
        current_idx_ = 0;
        Shuffle();
    }
    bool HasNextBatch() const {
        return current_idx_ < dataset_.size();
    }

    void GetNextBatch(std::vector<float>& out_images, std::vector<int>& out_labels) {
        int B = std::min(batch_size_, (int)(dataset_.size() - current_idx_));
        int C = 3, H = 224, W = 224;

        // Reset if we need a full batch but can't get it (or let caller handle partial batches)
        out_images.resize(batch_size_ * C * H * W, 0.0f); // pad with 0s if partial
        out_labels.resize(batch_size_, 0);

        std::uniform_real_distribution<float> flip_dist(0.0f, 1.0f);
        std::uniform_real_distribution<float> scale_dist(0.08f, 1.0f);
        std::uniform_real_distribution<float> aspect_dist(3.0f/4.0f, 4.0f/3.0f);
        std::uniform_real_distribution<float> crop_dist(0.0f, 1.0f);

        for (int b = 0; b < B; ++b) {
            int idx = current_idx_ + b;
            const auto& record = dataset_[idx];
            out_labels[b] = record.label;

            int orig_w, orig_h, orig_comp;
            unsigned char* img = stbi_load(record.path.c_str(), &orig_w, &orig_h, &orig_comp, 3);
            if (!img) {
                std::cerr << "Failed to load image: " << record.path << "\n";
                continue;
            }

            // Target buffer for 224x224 RGB
            std::vector<unsigned char> crop_buf(224 * 224 * 3, 0);
            bool do_hflip = augment_ && (flip_dist(rng_) > 0.5f);

            if (augment_) {
                // Training: RandomResizedCrop(224)
                float target_area = orig_w * orig_h * scale_dist(rng_);
                float target_aspect = aspect_dist(rng_);

                int crop_w = (int)std::round(std::sqrt(target_area * target_aspect));
                int crop_h = (int)std::round(std::sqrt(target_area / target_aspect));

                if (crop_w > orig_w) crop_w = orig_w;
                if (crop_h > orig_h) crop_h = orig_h;

                int start_x = (int)(crop_dist(rng_) * (orig_w - crop_w));
                int start_y = (int)(crop_dist(rng_) * (orig_h - crop_h));

                // Extract crop logic
                std::vector<unsigned char> src_crop(crop_w * crop_h * 3);
                for (int y = 0; y < crop_h; ++y) {
                    for (int x = 0; x < crop_w; ++x) {
                        int src_idx = ((start_y + y) * orig_w + (start_x + x)) * 3;
                        int dst_idx = (y * crop_w + x) * 3;
                        src_crop[dst_idx] = img[src_idx];
                        src_crop[dst_idx+1] = img[src_idx+1];
                        src_crop[dst_idx+2] = img[src_idx+2];
                    }
                }

                // Resize to 224x224
                stbir_resize_uint8_linear(src_crop.data(), crop_w, crop_h, 0,
                                          crop_buf.data(), 224, 224, 0, (stbir_pixel_layout)3); // STBIR_RGB = 3 channels

            } else {
                // Validation: shortest side resize to 256, then CenterCrop(224)
                float scale = 256.0f / std::min(orig_w, orig_h);
                int resize_w = (int)(orig_w * scale);
                int resize_h = (int)(orig_h * scale);

                std::vector<unsigned char> resized(resize_w * resize_h * 3);
                stbir_resize_uint8_linear(img, orig_w, orig_h, 0,
                                          resized.data(), resize_w, resize_h, 0, (stbir_pixel_layout)3);

                int start_x = (resize_w - 224) / 2;
                int start_y = (resize_h - 224) / 2;

                if (start_x < 0) start_x = 0;
                if (start_y < 0) start_y = 0;

                for (int y = 0; y < 224 && (start_y + y) < resize_h; ++y) {
                    for (int x = 0; x < 224 && (start_x + x) < resize_w; ++x) {
                        int src_idx = ((start_y + y) * resize_w + (start_x + x)) * 3;
                        int dst_idx = (y * 224 + x) * 3;
                        crop_buf[dst_idx] = resized[src_idx];
                        crop_buf[dst_idx+1] = resized[src_idx+1];
                        crop_buf[dst_idx+2] = resized[src_idx+2];
                    }
                }
            }
            stbi_image_free(img);

            // Convert to CHW and normalize to [0, 1] then use ImageNet mean/std
            float* dst = out_images.data() + b * C * H * W;
            for (int c = 0; c < 3; ++c) {
                for (int h = 0; h < 224; ++h) {
                    for (int w = 0; w < 224; ++w) {
                        int src_w = do_hflip ? (223 - w) : w;
                        int src_idx = (h * 224 + src_w) * 3 + c;
                        int dst_idx = (c * 224 + h) * 224 + w;
                        
                        // ImageNet standard mean/std normalization logic typical for ResNet
                        // mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
                        float val = crop_buf[src_idx] / 255.0f;
                        if (c == 0) val = (val - 0.485f) / 0.229f;
                        else if (c == 1) val = (val - 0.456f) / 0.224f;
                        else if (c == 2) val = (val - 0.406f) / 0.225f;
                        
                        dst[dst_idx] = val;
                    }
                }
            }
        }

        current_idx_ += batch_size_;
    }

private:
    std::string data_dir_;
    std::string split_;
    int batch_size_;
    size_t current_idx_;
    bool augment_;
    std::mt19937 rng_;

    std::vector<std::string> wnids_;
    std::unordered_map<std::string, int> wnid_to_label_;
    std::vector<ImageRecord> dataset_;

    void LoadPaths() {
        std::string split_path = data_dir_ + "/" + split_;
        
        if (split_ == "train") {
            // Find all WNID folders in train directory
            for (const auto& entry : fs::directory_iterator(split_path)) {
                if (entry.is_directory()) {
                    wnids_.push_back(entry.path().filename().string());
                }
            }
            std::sort(wnids_.begin(), wnids_.end());
            for (int i = 0; i < (int)wnids_.size(); ++i) {
                wnid_to_label_[wnids_[i]] = i;
            }

            // Load all images
            for (int label = 0; label < (int)wnids_.size(); ++label) {
                std::string class_dir = split_path + "/" + wnids_[label];
                if (!fs::exists(class_dir)) continue;
                for (const auto& entry : fs::directory_iterator(class_dir)) {
                    if (entry.is_regular_file() && (entry.path().extension() == ".JPEG" || entry.path().extension() == ".jpg")) {
                        dataset_.push_back({entry.path().string(), label});
                    }
                }
            }
            std::cout << "Discovered " << dataset_.size() << " training images across " << wnids_.size() << " classes.\n";
        } else if (split_ == "val") {
            // Assume PyTorch ImageFolder structure exists (val/<wnid>/*.JPEG) for simpler parsing.
            // Requires mapping from train directory.
            std::string train_path = data_dir_ + "/train";
            if (fs::exists(train_path)) {
                for (const auto& entry : fs::directory_iterator(train_path)) {
                    if (entry.is_directory()) {
                        wnids_.push_back(entry.path().filename().string());
                    }
                }
                std::sort(wnids_.begin(), wnids_.end());
                for (int i = 0; i < (int)wnids_.size(); ++i) {
                    wnid_to_label_[wnids_[i]] = i;
                }
            }

            if (fs::exists(split_path)) {
                for (const auto& entry : fs::directory_iterator(split_path)) {
                    if (entry.is_directory()) {
                        std::string wnid = entry.path().filename().string();
                        auto it = wnid_to_label_.find(wnid);
                        if (it != wnid_to_label_.end()) {
                            int label = it->second;
                            for (const auto& img_entry : fs::directory_iterator(entry.path())) {
                                if (img_entry.is_regular_file() && (img_entry.path().extension() == ".JPEG" || img_entry.path().extension() == ".jpg")) {
                                    dataset_.push_back({img_entry.path().string(), label});
                                }
                            }
                        }
                    } else if (entry.is_regular_file() && (entry.path().extension() == ".JPEG" || entry.path().extension() == ".jpg")) {
                        // Fallback: If it's a flat directory, we would need val_annotations.
                        // Here we just skip, but in reality you'd parse ILSVRC annotations here.
                        std::cerr << "Warning: Flat validation directory not supported without mapping.\n";
                        break;
                    }
                }
            }
            std::cout << "Discovered " << dataset_.size() << " validation images.\n";
        }
    }

    void Shuffle() {
        if (split_ == "train") {
            std::shuffle(dataset_.begin(), dataset_.end(), rng_);
        }
    }
};
