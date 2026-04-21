#include "dali_imagenet_loader.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            throw std::runtime_error(std::string("CUDA error: ") +              \
                                     cudaGetErrorString(_e));                   \
        }                                                                       \
    } while (0)

namespace {

size_t CountFileListSamples(const std::string& file_list_path) {
    std::ifstream input(file_list_path);
    if (!input.is_open()) {
        throw std::runtime_error("Unable to open file list: " + file_list_path);
    }

    size_t lines = 0;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty()) {
            ++lines;
        }
    }
    return lines;
}

void PrintUsage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0
        << " <pipeline.dali> <filelist.txt> <batch_size> [device_id] [num_threads] [max_batches]\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4) {
            PrintUsage(argv[0]);
            return 1;
        }

        const std::string pipeline_path = argv[1];
        const std::string file_list_path = argv[2];
        const int batch_size = std::stoi(argv[3]);
        const int device_id = (argc > 4) ? std::stoi(argv[4]) : 0;
        const int num_threads = (argc > 5) ? std::stoi(argv[5]) : 8;
        const int requested_batches = (argc > 6) ? std::stoi(argv[6]) : -1;

        if (batch_size <= 0) {
            throw std::runtime_error("batch_size must be positive");
        }

        const size_t sample_count = CountFileListSamples(file_list_path);
        if (sample_count < static_cast<size_t>(batch_size)) {
            throw std::runtime_error("file list does not contain a full batch");
        }

        CUDA_CHECK(cudaSetDevice(device_id));
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));

        const size_t image_elements = static_cast<size_t>(batch_size) * 3 * 224 * 224;
        float* d_images = nullptr;
        int* d_labels = nullptr;
        CUDA_CHECK(cudaMalloc(&d_images, image_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_labels, static_cast<size_t>(batch_size) * sizeof(int)));

        DaliImageNetLoader loader(pipeline_path,
                                  sample_count,
                                  batch_size,
                                  num_threads,
                                  device_id,
                                  2);

        const int available_batches = static_cast<int>(sample_count / static_cast<size_t>(batch_size));
        const int measured_batches = (requested_batches > 0)
            ? std::min(requested_batches, available_batches)
            : available_batches;
        const int warmup_batches = std::min(5, measured_batches);

        std::cout << "=== Native DALI C API ImageNet Smoke Test ===\n";
        std::cout << "Pipeline        : " << pipeline_path << "\n";
        std::cout << "File list       : " << file_list_path << "\n";
        std::cout << "Device          : " << device_id << "\n";
        std::cout << "Batch size      : " << batch_size << "\n";
        std::cout << "Samples         : " << sample_count << "\n";
        std::cout << "Full batches    : " << available_batches << "\n";
        std::cout << "Measured batches: " << measured_batches << "\n";

        std::vector<int> host_labels(static_cast<size_t>(batch_size));
        std::vector<float> host_first_image(3 * 224 * 224);
        bool printed_sample = false;

        loader.Reset();
        for (int step = 0; step < warmup_batches && loader.HasNextBatch(); ++step) {
            loader.GetNextBatch(d_images, d_labels, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        const auto start = std::chrono::high_resolution_clock::now();
        int completed_batches = 0;
        while (completed_batches < measured_batches && loader.HasNextBatch()) {
            loader.GetNextBatch(d_images, d_labels, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));

            if (!printed_sample) {
                CUDA_CHECK(cudaMemcpy(host_labels.data(), d_labels,
                                      static_cast<size_t>(batch_size) * sizeof(int),
                                      cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(host_first_image.data(), d_images,
                                      host_first_image.size() * sizeof(float),
                                      cudaMemcpyDeviceToHost));

                float min_value = std::numeric_limits<float>::infinity();
                float max_value = -std::numeric_limits<float>::infinity();
                bool all_finite = true;
                for (float value : host_first_image) {
                    min_value = std::min(min_value, value);
                    max_value = std::max(max_value, value);
                    all_finite = all_finite && std::isfinite(value);
                }

                std::cout << "First batch labels: ";
                const int labels_to_print = std::min(batch_size, 8);
                for (int index = 0; index < labels_to_print; ++index) {
                    if (index) std::cout << ", ";
                    std::cout << host_labels[static_cast<size_t>(index)];
                }
                std::cout << "\n";
                std::cout << "First image min/max: " << min_value << " / " << max_value << "\n";
                std::cout << "First image finite : " << (all_finite ? "yes" : "no") << "\n";
                printed_sample = true;
            }

            ++completed_batches;
        }
        const auto end = std::chrono::high_resolution_clock::now();

        const double seconds = std::chrono::duration<double>(end - start).count();
        const double images_per_sec = (seconds > 0.0)
            ? (static_cast<double>(completed_batches) * static_cast<double>(batch_size) / seconds)
            : 0.0;

        std::cout << "Completed batches: " << completed_batches << "\n";
        std::cout << "Elapsed seconds  : " << seconds << "\n";
        std::cout << "Images / second  : " << images_per_sec << "\n";

        CUDA_CHECK(cudaFree(d_images));
        CUDA_CHECK(cudaFree(d_labels));
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[dali_imagenet_smoke] ERROR: " << ex.what() << "\n";
        return 1;
    }
}