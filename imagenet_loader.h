#include "hip/hip_runtime.h"
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
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <utility>

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
// Supports TWO input modes:
//
//   Mode 1 — Extracted directory (recommended for SLURM TMPDIR staging):
//     Pass a directory path containing the standard ImageNet layout:
//       data_dir/train/<wnid>/*.JPEG
//       data_dir/val/<wnid>/*.JPEG
//
//   Mode 2 — ZIP file (for standalone/local use):
//     Pass a .zip file path (e.g., imagenet-object-localization-challenge.zip).
//     The loader will auto-extract to a temporary directory, arrange the
//     validation set into per-class subdirectories, then load normally.
//
// Images are variable size.
// Augmentation:
//   - Training: RandomResizedCrop(224) + horizontal flip
//   - Validation: Resize shortest side to 256, then CenterCrop(224)
// ─────────────────────────────────────────────────────────────────────────────

struct ImageRecord {
    std::string path;
    int label;
};

class ImageNetLoader {
public:
    // data_path: either a directory (extracted ImageNet) or a .zip file path
    // split: "train" or "val"
    // augment: enable RandomResizedCrop + random horizontal flip (only for training)
    // rank/world_size: for multi-GPU data sharding (each GPU gets a unique subset)
    ImageNetLoader(const std::string& data_path, int batch_size,
                   const std::string& split = "train", bool augment = false,
                   int rank = 0, int world_size = 1)
        : split_(split), batch_size_(batch_size), current_idx_(0),
          augment_(augment), rank_(rank), world_size_(world_size),
          rng_(42 + rank)  // deterministic seed per rank for reproducibility
    {
        // ── Auto-detect: ZIP file or extracted directory ──────────────────────
        data_dir_ = ResolveDataPath(data_path);
        LoadPaths();
        ShardDataset();  // partition data across GPUs
        Shuffle();
    }

    // ── Pre-scanned constructor (for multi-GPU: scan once, shard per rank) ──
    // Takes a pre-built dataset + class list instead of scanning the filesystem.
    // Use PreScanPaths() in main() to build these, then pass to each GPU worker.
    ImageNetLoader(const std::vector<ImageRecord>& full_dataset,
                   const std::vector<std::string>& wnids,
                   int batch_size, const std::string& split, bool augment,
                   int rank, int world_size)
        : split_(split), batch_size_(batch_size), current_idx_(0),
          augment_(augment), rank_(rank), world_size_(world_size),
          rng_(42 + rank), wnids_(wnids), dataset_(full_dataset)
    {
        for (int i = 0; i < (int)wnids_.size(); ++i)
            wnid_to_label_[wnids_[i]] = i;
        ShardDataset();
        Shuffle();
    }

    // ── Static helper: scan filesystem ONCE from main thread ────────────────
    // Returns { dataset, wnids } that can be passed to the pre-scanned constructor.
    static std::pair<std::vector<ImageRecord>, std::vector<std::string>>
    PreScanPaths(const std::string& data_path, const std::string& split) {
        std::string data_dir = ResolveDataPath(data_path);
        std::string split_path = data_dir + "/" + split;
        std::vector<std::string> wnids;
        std::vector<ImageRecord> dataset;

        // Discover class directories from train/ (used for both splits)
        std::string class_src = (split == "val") ? data_dir + "/train" : split_path;
        if (fs::exists(class_src)) {
            for (const auto& entry : fs::directory_iterator(class_src)) {
                if (entry.is_directory())
                    wnids.push_back(entry.path().filename().string());
            }
        }
        std::sort(wnids.begin(), wnids.end());

        std::unordered_map<std::string, int> wnid_to_label;
        for (int i = 0; i < (int)wnids.size(); ++i)
            wnid_to_label[wnids[i]] = i;

        std::cout << "[PreScan] Scanning " << split << " from: " << split_path << "\n";
        auto t0 = std::chrono::high_resolution_clock::now();

        if (split == "train") {
            for (int label = 0; label < (int)wnids.size(); ++label) {
                std::string class_dir = split_path + "/" + wnids[label];
                if (!fs::exists(class_dir)) continue;
                for (const auto& entry : fs::directory_iterator(class_dir)) {
                    if (entry.is_regular_file()) {
                        auto ext = entry.path().extension().string();
                        if (ext == ".JPEG" || ext == ".jpg")
                            dataset.push_back({entry.path().string(), label});
                    }
                }
            }
        } else {
            // val split
            if (fs::exists(split_path)) {
                for (const auto& entry : fs::directory_iterator(split_path)) {
                    if (entry.is_directory()) {
                        std::string wnid = entry.path().filename().string();
                        auto it = wnid_to_label.find(wnid);
                        if (it == wnid_to_label.end()) continue;
                        int label = it->second;
                        for (const auto& img : fs::directory_iterator(entry.path())) {
                            if (img.is_regular_file()) {
                                auto ext = img.path().extension().string();
                                if (ext == ".JPEG" || ext == ".jpg")
                                    dataset.push_back({img.path().string(), label});
                            }
                        }
                    }
                }
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "[PreScan] Found " << dataset.size() << " " << split
                  << " images across " << wnids.size() << " classes in "
                  << std::fixed << std::setprecision(1) << sec << "s\n";

        return {dataset, wnids};
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
    int rank_;
    int world_size_;
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

    // Partition dataset so each GPU gets a unique, non-overlapping subset.
    // All GPUs must call LoadPaths() first to get the SAME full dataset,
    // then each takes its own slice based on rank.
    void ShardDataset() {
        if (world_size_ <= 1) return;

        // Sort to ensure identical ordering across all ranks before sharding
        std::sort(dataset_.begin(), dataset_.end(),
                  [](const ImageRecord& a, const ImageRecord& b) {
                      return a.path < b.path;
                  });

        // Each rank takes every world_size-th sample starting at rank_
        std::vector<ImageRecord> shard;
        shard.reserve(dataset_.size() / world_size_ + 1);
        for (size_t i = rank_; i < dataset_.size(); i += world_size_) {
            shard.push_back(std::move(dataset_[i]));
        }
        dataset_ = std::move(shard);

        std::cout << "[Rank " << rank_ << "] Data shard: "
                  << dataset_.size() << " samples (of ~"
                  << dataset_.size() * world_size_ << " total)\n";
    }

    void Shuffle() {
        if (split_ == "train") {
            std::shuffle(dataset_.begin(), dataset_.end(), rng_);
        }
    }

    // ── ZIP / directory auto-detection ───────────────────────────────────────
    // Returns the resolved data_dir (either the original path or extracted dir)
    static std::string ResolveDataPath(const std::string& data_path) {
        // Check if it's a ZIP file
        bool is_zip = false;
        if (fs::is_regular_file(data_path)) {
            std::string ext = fs::path(data_path).extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            is_zip = (ext == ".zip");
        }

        if (!is_zip) {
            // It's a directory path — use as-is (TMPDIR staging or local extracted)
            if (!fs::is_directory(data_path)) {
                std::cerr << "ERROR: Data path is neither a ZIP file nor a directory: "
                          << data_path << "\n";
                std::exit(1);
            }
            std::cout << "[ImageNetLoader] Using extracted directory: " << data_path << "\n";
            return data_path;
        }

        // ── ZIP mode: extract and arrange ────────────────────────────────────
        std::cout << "[ImageNetLoader] ZIP file detected: " << data_path << "\n";
        return ExtractZipAndPrepare(data_path);
    }

    // Extract a ZIP file and prepare the ImageNet directory structure.
    // Returns the path to the CLS-LOC directory (containing train/ and val/).
    static std::string ExtractZipAndPrepare(const std::string& zip_path) {
        // Determine extraction target: use TMPDIR if available, else a sibling dir
        std::string tmpdir;
        const char* env_tmpdir = std::getenv("TMPDIR");
        if (env_tmpdir && std::strlen(env_tmpdir) > 0) {
            tmpdir = std::string(env_tmpdir) + "/imagenet_extracted";
        } else {
            // Fallback: extract next to the ZIP file
            tmpdir = fs::path(zip_path).parent_path().string() + "/imagenet_extracted";
        }

        std::string cls_loc_dir = tmpdir + "/ILSVRC/Data/CLS-LOC";

        // Skip extraction if already done
        if (fs::is_directory(cls_loc_dir) &&
            fs::is_directory(cls_loc_dir + "/train")) {
            std::cout << "[ImageNetLoader] Found existing extraction at: " << cls_loc_dir << "\n";

            // Still arrange val set if needed (idempotent)
            ArrangeValSet(tmpdir);
            return cls_loc_dir;
        }

        // Extract ZIP using system unzip (much faster than in-process for 160GB)
        std::cout << "[ImageNetLoader] Extracting ZIP to: " << tmpdir << "\n";
        std::cout << "[ImageNetLoader] This may take 10-15 minutes for full ImageNet...\n";

        fs::create_directories(tmpdir);

        std::string cmd = "unzip -q -n \"" + zip_path + "\" -d \"" + tmpdir + "\"";
        int ret = std::system(cmd.c_str());

        if (ret != 0) {
            std::cerr << "ERROR: ZIP extraction failed (exit code " << ret << ")\n";
            std::cerr << "Command was: " << cmd << "\n";
            std::exit(1);
        }

        // Verify extraction produced expected structure
        if (!fs::is_directory(cls_loc_dir + "/train")) {
            std::cerr << "ERROR: Expected directory not found after extraction: "
                      << cls_loc_dir << "/train\n";
            std::cerr << "Contents of " << tmpdir << ":\n";
            for (const auto& e : fs::directory_iterator(tmpdir)) {
                std::cerr << "  " << e.path().filename().string() << "\n";
            }
            std::exit(1);
        }

        std::cout << "[ImageNetLoader] Extraction complete.\n";

        // Arrange validation set from flat Kaggle format into class subfolders
        ArrangeValSet(tmpdir);

        return cls_loc_dir;
    }

    // Arrange the Kaggle-format flat val/ directory into per-class subfolders.
    // Uses LOC_val_solution.csv (included in the Kaggle ZIP) to map images to WNIDs.
    // This is idempotent — skips images that are already in subdirectories.
    static void ArrangeValSet(const std::string& extract_root) {
        std::string val_dir = extract_root + "/ILSVRC/Data/CLS-LOC/val";
        std::string csv_path = extract_root + "/LOC_val_solution.csv";

        if (!fs::is_directory(val_dir)) {
            std::cerr << "[ImageNetLoader] Warning: val/ directory not found, skipping arrangement.\n";
            return;
        }

        if (!fs::is_regular_file(csv_path)) {
            std::cout << "[ImageNetLoader] LOC_val_solution.csv not found — "
                      << "assuming val set is already arranged.\n";
            return;
        }

        // Check if val images are already in subdirectories (first JPEG in a subdir = done)
        bool already_arranged = false;
        for (const auto& entry : fs::directory_iterator(val_dir)) {
            if (entry.is_directory()) {
                // Found a subdirectory — likely already arranged
                already_arranged = true;
                break;
            }
        }

        // Quick heuristic: if subdirs exist AND no loose JPEGs remain, skip
        if (already_arranged) {
            bool has_loose_jpegs = false;
            for (const auto& entry : fs::directory_iterator(val_dir)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    if (ext == ".JPEG" || ext == ".jpg") {
                        has_loose_jpegs = true;
                        break;
                    }
                }
            }
            if (!has_loose_jpegs) {
                std::cout << "[ImageNetLoader] Validation set already arranged into class folders.\n";
                return;
            }
        }

        // Parse CSV and move images into class subdirectories
        std::cout << "[ImageNetLoader] Arranging validation images into class subfolders...\n";
        std::ifstream csv_file(csv_path);
        if (!csv_file.is_open()) {
            std::cerr << "[ImageNetLoader] Warning: Cannot open " << csv_path << "\n";
            return;
        }

        std::string line;
        std::getline(csv_file, line); // skip header

        int moved = 0;
        while (std::getline(csv_file, line)) {
            if (line.empty()) continue;

            // CSV format: ImageId,PredictionString
            // PredictionString: "n01440764 0 0 500 400" — we want just the WNID
            size_t comma_pos = line.find(',');
            if (comma_pos == std::string::npos) continue;

            std::string img_id = line.substr(0, comma_pos);
            std::string pred_str = line.substr(comma_pos + 1);

            // Extract WNID (first token of PredictionString)
            size_t space_pos = pred_str.find(' ');
            std::string wnid = (space_pos != std::string::npos)
                                ? pred_str.substr(0, space_pos)
                                : pred_str;

            // Remove any surrounding quotes
            if (!wnid.empty() && wnid.front() == '"') wnid.erase(0, 1);
            if (!wnid.empty() && wnid.back() == '"') wnid.pop_back();

            std::string src = val_dir + "/" + img_id + ".JPEG";
            if (!fs::exists(src)) continue;

            std::string dst_dir = val_dir + "/" + wnid;
            fs::create_directories(dst_dir);

            std::string dst = dst_dir + "/" + img_id + ".JPEG";
            fs::rename(src, dst);
            moved++;
        }

        std::cout << "[ImageNetLoader] Moved " << moved
                  << " validation images into class subfolders.\n";
    }
};
