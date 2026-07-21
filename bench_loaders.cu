// bench_loaders.cu — fair, isolated throughput comparison of the two data
// loaders, with NO model compute. Each loader must deliver a training-ready
// FP32 NCHW 224x224 batch to the GPU; we time pure GetNextBatch (+ H2D staging
// for the CPU loader, since rocAL delivers straight to device).
//
//   TURBOJPEG : MTImageLoader (CPU decode + CPU resize/crop/normalize) → host
//               → pinned → hipMemcpyAsync H2D
//   ROCAL     : RocalImageNetLoader (CPU decode + GPU resize/crop via RPP +
//               copy_data + crop/normalize HIP kernel) → device directly
//
// Usage: bench_loaders <data_dir(with train/)> [batch=128] [nbatch=100]
// stb implementation is compiled exactly once, via imagenet_loader.h — mirror
// train_multigpu.cu: define the impl macros, then let the loader header pull in
// stb (do NOT include stb_image.h directly, or it compiles the impl twice).
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "imagenet_loader.h"     // ImageRecord, ImageNetLoader::PreScanPaths, stb
#include "mt_image_loader.h"     // MTImageLoader (CPU / turbojpeg)

#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#ifdef USE_ROCAL
#include "rocal_imagenet_loader.h"
#endif

#define HIP_CHECK(x) do { hipError_t e=(x); if(e!=hipSuccess){ \
    std::fprintf(stderr,"[HIP] %s: %s\n",#x,hipGetErrorString(e)); } } while(0)

static double now_s() { return std::chrono::duration<double>(
    std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    const std::string data_dir = argc > 1 ? argv[1] : "rocal_research/in1k_mini";
    const int batch  = argc > 2 ? std::atoi(argv[2]) : 128;
    const int nbatch = argc > 3 ? std::atoi(argv[3]) : 100;
    const int warmup = 10;
    HIP_CHECK(hipSetDevice(0));

    std::fprintf(stderr, "[bench] data=%s batch=%d nbatch=%d (warmup=%d)\n",
                 data_dir.c_str(), batch, nbatch, warmup);

    auto pr = ImageNetLoader::PreScanPaths(data_dir, "train");
    std::vector<ImageRecord>& ds = pr.first;
    std::vector<std::string>& wnids = pr.second;
    std::fprintf(stderr, "[bench] dataset: %zu images, %zu classes\n", ds.size(), wnids.size());
    if ((int)ds.size() < batch * (warmup + 5)) {
        std::fprintf(stderr, "[bench] WARNING: dataset small vs batch*nbatch; loaders will Reset()\n");
    }

    const size_t img_elems = (size_t)batch * 3 * 224 * 224;
    float* d_img = nullptr; int* d_lab = nullptr; float* h_pin = nullptr;
    HIP_CHECK(hipMalloc(&d_img, img_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_lab, (size_t)batch * sizeof(int)));
    HIP_CHECK(hipHostMalloc((void**)&h_pin, img_elems * sizeof(float)));
    hipStream_t stream; HIP_CHECK(hipStreamCreate(&stream));

    // ── TURBOJPEG (MTImageLoader) ──────────────────────────────────────────
    {
        int bthreads = std::getenv("BENCH_THREADS") ? std::atoi(std::getenv("BENCH_THREADS")) : 8;
        MTImageLoader loader(ds, wnids, batch, "train", /*augment=*/true, 0, 1, bthreads);
        std::vector<float> h_img; std::vector<int> h_lab;
        int done = 0; double sec = 0;
        for (int b = 0; b < nbatch + warmup; ++b) {
            if (!loader.HasNextBatch()) loader.Reset();
            double t0 = now_s();
            loader.GetNextBatch(h_img, h_lab);                       // CPU decode+aug
            std::memcpy(h_pin, h_img.data(), img_elems * sizeof(float));
            HIP_CHECK(hipMemcpyAsync(d_img, h_pin, img_elems * sizeof(float),
                                     hipMemcpyHostToDevice, stream));
            HIP_CHECK(hipStreamSynchronize(stream));
            if (b >= warmup) { sec += now_s() - t0; ++done; }
        }
        std::fprintf(stderr, "[TURBOJPEG] %d batches x %d = %d imgs in %.3fs = %.0f img/s\n",
                     done, batch, done * batch, sec, done * batch / sec);
    }

#ifdef USE_ROCAL
    // ── ROCAL (RocalImageNetLoader) ────────────────────────────────────────
    {
        const std::string train_dir = data_dir + "/train";
        RocalImageNetLoader loader(train_dir, ds.size(), batch, /*threads=*/8,
                                   /*device=*/0, /*shard=*/0, /*shards=*/1,
                                   /*prefetch=*/3, /*training=*/true);
        int done = 0; double sec = 0;
        for (int b = 0; b < nbatch + warmup; ++b) {
            if (!loader.HasNextBatch()) loader.Reset();
            double t0 = now_s();
            loader.GetNextBatch(d_img, d_lab, stream);               // device-direct
            if (b >= warmup) { sec += now_s() - t0; ++done; }
        }
        std::fprintf(stderr, "[ROCAL]     %d batches x %d = %d imgs in %.3fs = %.0f img/s\n",
                     done, batch, done * batch, sec, done * batch / sec);
    }
#else
    std::fprintf(stderr, "[ROCAL] not built (USE_ROCAL off)\n");
#endif
    std::fprintf(stderr, "[bench] done\n");
    return 0;
}
