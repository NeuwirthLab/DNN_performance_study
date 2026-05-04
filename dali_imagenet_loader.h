#pragma once

#ifdef USE_DALI

#if defined(USE_HIP) && !defined(USE_ROCAL)
#error "DALI is NVIDIA-only. On AMD/HIP builds, either omit -DUSE_DALI=ON \
(the build will use mt_image_loader.h + prefetch_loader.h) or implement \
a rocAL-backed loader and gate it on USE_ROCAL."
#endif

#include <cuda_runtime.h>

#include <dali/c_api.h>

#include <cstdlib>
#include <cstddef>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

class DaliImageNetLoader {
public:
    DaliImageNetLoader(const std::string& pipeline_path,
                       size_t sample_count,
                       int batch_size,
                       int num_threads,
                       int device_id,
                       int prefetch_queue_depth = 2)
        : sample_count_(sample_count)
        , batch_size_(batch_size)
        , num_batches_(batch_size > 0 ? static_cast<int>(sample_count / batch_size) : 0)
    {
        if (batch_size_ <= 0) {
            throw std::runtime_error("DaliImageNetLoader: batch_size must be positive");
        }
        if (num_batches_ < 1) {
            throw std::runtime_error("DaliImageNetLoader: shard has no full batches to process");
        }

        daliInitialize();
        EnsureOperatorsLoaded();
        const std::vector<char> serialized = LoadFile(pipeline_path);
        if (serialized.empty()) {
            throw std::runtime_error("DaliImageNetLoader: serialized pipeline is empty: " + pipeline_path);
        }

        daliCreatePipeline(&pipe_handle_,
                           serialized.data(),
                           static_cast<int>(serialized.size()),
                           batch_size_,
                           num_threads,
                           device_id,
                           0,
                           prefetch_queue_depth,
                           prefetch_queue_depth,
                           prefetch_queue_depth,
                           0);

        daliPrefetchUniform(&pipe_handle_, prefetch_queue_depth);
    }

    ~DaliImageNetLoader() {
        if (pipe_handle_) {
            daliDeletePipeline(&pipe_handle_);
        }
    }

    void Reset() {
        current_batch_ = 0;
    }

    bool HasNextBatch() const {
        return current_batch_ < num_batches_;
    }

    size_t NumSamples() const {
        return sample_count_;
    }

    void GetNextBatch(float* d_images, int* d_labels, cudaStream_t stream) {
        if (!HasNextBatch()) {
            throw std::runtime_error("DaliImageNetLoader: no more batches in current epoch");
        }

        daliOutput(&pipe_handle_);
        if (daliGetNumOutput(&pipe_handle_) < 2) {
            throw std::runtime_error("DaliImageNetLoader: pipeline returned fewer than 2 outputs");
        }

        daliOutputCopy(&pipe_handle_, d_images, 0, static_cast<device_type_t>(1), stream, 0);
        daliOutputCopy(&pipe_handle_, d_labels, 1, static_cast<device_type_t>(1), stream, 0);
        daliOutputRelease(&pipe_handle_);

        current_batch_++;
        daliRun(&pipe_handle_);
    }

private:
    static void EnsureOperatorsLoaded() {
        static std::once_flag once;
        std::call_once(once, []() {
            const char* operator_lib = std::getenv("RN50_DALI_OPERATOR_LIB");
            if (operator_lib && operator_lib[0] != '\0') {
                daliLoadLibrary(operator_lib);
            } else {
                // Fall back to loader search paths if the caller didn't provide an absolute path.
                daliLoadLibrary("libdali_operators.so");
            }
        });
    }

    static std::vector<char> LoadFile(const std::string& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            throw std::runtime_error("DaliImageNetLoader: unable to open pipeline file: " + path);
        }

        const std::streamsize size = file.tellg();
        if (size < 0) {
            throw std::runtime_error("DaliImageNetLoader: unable to determine file size: " + path);
        }

        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(static_cast<size_t>(size));
        if (size > 0 && !file.read(buffer.data(), size)) {
            throw std::runtime_error("DaliImageNetLoader: unable to read pipeline file: " + path);
        }
        return buffer;
    }

    daliPipelineHandle pipe_handle_{nullptr};
    size_t sample_count_{0};
    int batch_size_{0};
    int num_batches_{0};
    int current_batch_{0};
};

#endif  // USE_DALI