# ResNet-50 CUDA → HIP/ROCm Port

Branch: `amd/hip-runtime`.
Target hardware: AMD MI300X (`gfx942`) or MI250 (`gfx90a`).
Strategy: single source tree, selected by `-DUSE_HIP=1`. Tiny ImageNet first,
then ImageNet-1k. Data pipeline on AMD uses **MIVisionX / rocAL** (the AMD
equivalent of NVIDIA DALI).

---

## What is already in the repo

```
hip/
├── gpu_compat.h          runtime + cuBLAS + NCCL shim (CUDA ↔ HIP)
├── dnn_compat.h          cuDNN ↔ MIOpen shim (descriptors macroed,
│                         conv algo-selection needs real wrappers)
└── CMakeLists.txt        standalone HIP build; leaves root CUDA build alone
scripts/
└── port_to_hip.sh        idempotent hipify-perl sweep over the .cu files
smoke_test_amd.slurm      mirror of smoke_test_helma.slurm for AMD nodes
```

Nothing in the original CUDA source has been modified yet. The root
`CMakeLists.txt` still builds the CUDA binaries on HELMA unchanged.

---

## Library mapping

| CUDA (HELMA) | AMD replacement | Mechanism |
|---|---|---|
| `cuda_runtime.h`, streams, events | `hip/hip_runtime.h` | `gpu_compat.h` macros |
| `cublas_v2.h` | `hipblas.h` | `gpu_compat.h` macros |
| `cudnn.h` | `miopen/miopen.h` | `dnn_compat.h` + inline wrappers |
| `nccl.h` | `rccl/rccl.h` | drop-in (same symbols), just link `-lrccl` |
| `nvtx3` | roctx (optional) | no-op macro for now |
| **DALI** (GPU data loader) | **rocAL (MIVisionX)** | `-DUSE_ROCAL=ON` CMake option |
| `nvcc -arch=sm_90` | `hipcc --offload-arch=gfx942` | CMake HIP language |

---

## Why MIVisionX/rocAL (not DALI)

DALI is NVIDIA-only and will not run on AMD. The AMD-native equivalent is
**rocAL** (ROCm Augmentation Library), shipped as part of **MIVisionX**.
It matches the DALI feature set we actually use:

- JPEG decode (CPU fallback on older ROCm; **rocJPEG** on GPU for MI300)
- ImageNet-style `<class>/<image>.JPEG` folder reader
- Random crop / resize / horizontal flip / normalize
- Output as NCHW float32 directly into HIP device memory
- Multi-reader sharding keyed by `(shard_id, num_shards)` — plug straight
  into our NCCL/RCCL per-rank split

On the Tiny ImageNet smoke test we **don't need rocAL yet** — the existing
CPU `tiny_dataloader.h` handles 100k images comfortably. Bring rocAL in only
once Tiny is passing and we move to ImageNet-1k.

---

## Step-by-step porting workflow

### Phase A — local, already done
1. `hip/gpu_compat.h` + `hip/dnn_compat.h` written.
2. `hip/CMakeLists.txt` standalone HIP build written.
3. `smoke_test_amd.slurm` + `scripts/port_to_hip.sh` written.

### Phase B — on the AMD cluster, one-time port

```bash
git fetch && git checkout amd/hip-runtime
module load rocm cmake        # names vary per site; check `module avail rocm`
bash scripts/port_to_hip.sh   # runs hipify-perl in-place, keeps *.cuda.bak
```

Then hand-edit `resnet50.cu` / `resnet50.cuh` to:
1. Replace the top-of-file `#include <cuda_runtime.h>` / `<cublas_v2.h>` /
   `<cudnn.h>` / `<nccl.h>` block with a single `#include "hip/gpu_compat.h"`
   and `#include "hip/dnn_compat.h"`.
2. Swap the ~7 `cudnnFindConvolution*Algorithm` + `cudnnGet*WorkspaceSize`
   call sites for `miopenFindConvolution*Algorithm` wrappers (stub spots are
   marked with `TODO(amrit)` in `hip/dnn_compat.h`).
3. Leave every custom `__global__` kernel untouched — HIP accepts CUDA
   kernel syntax verbatim. Do `grep -n '\b32\b' resnet50.cu` and check that
   any literal `32` meaning "warp size" is replaced with `warpSize`, because
   AMD wavefronts are 64.

### Phase C — build and smoke test

```bash
# from repo root
cmake -S hip -B build-hip \
      -DCMAKE_HIP_ARCHITECTURES=gfx942 \
      -DCMAKE_BUILD_TYPE=Release \
      -DUSE_NCCL=ON
cmake --build build-hip -j16 --target train_multigpu_tiny
sbatch smoke_test_amd.slurm
```

Smoke-test success criteria (same as HELMA):
1. 2 GPUs detected via `rocm-smi`
2. RCCL link + AllReduce works (2-GPU broadcast of weights on rank 0)
3. Loss decreases over 2 epochs on Tiny ImageNet
4. Checkpoint written to `checkpoints/*.bin`

### Phase D — scale to ImageNet-1k with rocAL

Only after Phase C passes on Tiny:

1. Install/load MIVisionX: `module load mivisionx` (or check
   `/opt/rocm/include/rocal/rocal_api.h`).
2. Create `rocal_loader.h` next to `dali_loader.h`. Mirror the class
   interface exactly (`init(path, batch, shard_id, num_shards)`,
   `next_batch(float* d_images, int* d_labels)`) so `train_multigpu.cu`
   doesn't need to know which backend it's using. Under the hood:
   - `rocalCreate(batch, processingMode=ROCAL_PROCESS_GPU, tensor=ROCAL_FP32, numThreads)`
   - `rocalJpegFileSource(...)` (reader) with `shard_id, shard_count`
   - `rocalRandomCrop`, `rocalResize`, `rocalCropMirrorNormalize` (mean/std)
   - `rocalRun()` / `rocalGetOutputTensors()` returning device pointers.
3. Wire it behind a `#ifdef USE_ROCAL` guard in `train_multigpu.cu` (the
   same place `USE_DALI` sits today).
4. Build with `-DUSE_ROCAL=ON` and adapt `smoke_test_amd.slurm` into a
   full ImageNet SLURM (mirror `run_multigpu_tmpdir_v2.slurm`).

---

## AMD gotchas to watch for

- **Wavefront size 64, not 32.** Any `__shfl_sync(0xffffffff, ...)` mask
  assumes 32; fix using `warpSize`. Grep for `0xffffffff` and `__shfl`.
- **MIOpen algorithm search needs workspace allocated first.** Unlike cuDNN's
  "tell me the size, then allocate", MIOpen wants you to allocate a workspace
  and pass it in. The wrapper stubs in `dnn_compat.h` expect you to query
  `miopenConvolutionForwardGetWorkSpaceSize` first.
- **`find_package(Torch)` pulls CUDA LibTorch.** If the CUDA build only uses
  Torch for checkpoint utilities, the HIP CMakeLists above drops Torch
  entirely. If a `train_*.cu` actually needs LibTorch tensors, install the
  ROCm build of LibTorch and add it back.
- **`-arch=native` doesn't exist in hipcc.** Use
  `--offload-arch=gfx942` (or whatever `rocminfo` reports). The HIP
  CMakeLists exposes this via `CMAKE_HIP_ARCHITECTURES`.
- **NCCL env vars don't all port.** `NCCL_DEBUG=INFO` works (RCCL honors it),
  but `NCCL_P2P_LEVEL` and a few others have different defaults on Infinity
  Fabric. Start with defaults; only tune if bandwidth is low.
- **Module names vary per cluster.** `module avail rocm` / `module avail
  mivisionx` / `module avail rccl` first — the SLURM script has placeholders.

---

## What is explicitly deferred

- **Full MIOpen conv wrappers.** The algorithm-selection calls in
  `dnn_compat.h` are stubs (marked `TODO(amrit)`). These must be written on
  an AMD machine where you can actually run `miopenFindConvolutionForward
  Algorithm` and see which algo is fastest for our tensor shapes.
- **rocAL loader.** Not written yet; Phase D.
- **roctx profiling.** NVTX is macroed to no-ops. Add `roctx_range_push/pop`
  later if you need profiling traces.
- **fp16/bf16 mixed precision.** Stay in fp32 for the first pass; MIOpen
  handles dtypes differently and it's a separate tuning exercise.

---

## TL;DR

Clone branch, run `scripts/port_to_hip.sh` on the AMD node, do the two
header-swap edits in `resnet50.cu`, fill in the MIOpen conv wrappers in
`hip/dnn_compat.h`, `sbatch smoke_test_amd.slurm`. That gets Tiny ImageNet
training on AMD. rocAL/MIVisionX comes in for ImageNet-1k.
