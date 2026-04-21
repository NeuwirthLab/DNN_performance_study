# AMD GPU Porting Plan for This Repo

## Executive summary

This repository is **not** just raw CUDA kernels. It is a full NVIDIA stack application:

- CUDA runtime + custom kernels
- cuBLAS
- cuDNN
- NCCL
- NVIDIA DALI
- Apptainer / Slurm launch flow tuned for H100

That means the fastest safe way to port it to AMD is:

1. **Get a functional ROCm build first** using HIP + rocBLAS + MIOpen + RCCL
2. **Disable the DALI path initially** and use the existing CPU loader path
3. **Only after correctness is restored**, optimize for AMD performance

---

## Where the NVIDIA lock-in currently lives

### 1. Build system
- `CMakeLists.txt`
- CUDA is the build language and it links directly against cuBLAS, cuDNN, NCCL, and optional DALI.

### 2. Model core
- `resnet50.cuh`
- `resnet50.cu`

These files depend on:
- `cuda_runtime.h`
- `cuda_bf16.h`
- `cublas_v2.h`
- `cudnn.h`
- NCCL under `USE_NCCL`
- CUDA kernels via `__global__`

### 3. Multi-GPU training
- `train_multigpu.cu`

This file uses:
- CUDA streams and memory APIs
- NCCL AllReduce
- TF32-specific tuning

### 4. NVIDIA-only input pipeline
- `dali_imagenet_loader.h`
- `run_multigpu_tmpdir_dali.slurm`
- `tools/serialize_imagenet_dali.py`

This path is the least portable part of the repo.

---

## Recommended migration method

## Phase 1 — Create a backend split

Goal: keep one codebase, but allow both NVIDIA and AMD builds.

### What to add
Create a portability layer with two backends:

- NVIDIA backend: CUDA, cuBLAS, cuDNN, NCCL
- AMD backend: HIP, rocBLAS, MIOpen, RCCL

### Practical structure
Introduce a small compatibility header, for example:

- `gpu_backend.h`
- `gpu_backend_blas.h`
- `gpu_backend_dnn.h`
- `gpu_backend_comm.h`

These wrappers should hide:
- runtime allocation / memcpy / streams
- BLAS handle and GEMM calls
- DNN descriptors and conv / BN / pooling
- collectives

This prevents the whole model from being rewritten twice.

---

## Phase 2 — Build on AMD without DALI first

Goal: get the trainer running correctly on AMD before touching the input pipeline.

### Why
The repo already has a non-DALI path using:
- `mt_image_loader.h`
- `prefetch_loader.h`
- `imagenet_loader.h`

That path is much easier to port than the DALI branch.

### Recommendation
For the first ROCm milestone:
- keep `USE_DALI=OFF`
- keep single-node training
- then re-enable multi-GPU via RCCL

---

## Phase 3 — Mechanical HIP conversion

Use HIPIFY on these files first:

- `resnet50.cu`
- `resnet50.cuh`
- `train_multigpu.cu`
- `train.cu`
- `main.cu`
- `benchmark.cu`

### What HIPIFY usually handles well
- CUDA runtime calls such as malloc, free, memcpy, streams, events
- kernel launch syntax
- many CUDA type names

### What it does not finish for you
- cuDNN -> MIOpen API differences
- cuBLAS -> rocBLAS API differences in detail
- NCCL -> RCCL setup and linking
- BF16 / math-mode differences
- DALI replacement

So use HIPIFY as a starting point, not as the whole port.

---

## Phase 4 — Replace NVIDIA math libraries

## BLAS
Map:
- cuBLAS -> rocBLAS

Used heavily in:
- FC forward and backward in `resnet50.cu`

## DNN primitives
Map:
- cuDNN -> MIOpen

Used heavily in:
- `ConvBnLayer`
- convolution forward/backward
- batch norm forward/backward
- max pooling

This is the most important and highest-effort library replacement.

## Communication
Map:
- NCCL -> RCCL

Used in:
- gradient synchronization in `train_multigpu.cu`
- `SyncGradients()` in `resnet50.cu`

---

## Phase 5 — Precision path cleanup

Your current code uses NVIDIA-specific precision features:
- TF32 enablement in multi-GPU training
- BF16 via `cuda_bf16.h`
- tensor-op math selection in cuDNN

On AMD:
- remove or guard the TF32 path
- replace BF16 intrinsics/types with HIP/ROCm equivalents
- retune mixed precision behavior after correctness is verified

Do **not** optimize this first.
Get correctness first in FP32, then restore BF16.

---

## Phase 6 — Re-enable multi-GPU

Once single-GPU ROCm training works:

1. enable RCCL
2. validate weight broadcast
3. validate gradient all-reduce
4. verify loss parity against single-GPU
5. then scale to 2 GPUs

This reduces debugging complexity drastically.

---

## Phase 7 — Replace or redesign the DALI path

This is the hardest repo-specific portability issue.

### Best short-term solution
Use the existing CPU-based loaders on AMD and skip DALI.

### Later options
- keep a CPU preprocessing path with overlap and pinned memory
- write a ROCm-native image pipeline
- consider a framework-based loader for AMD experiments

Do not block the main ROCm port on DALI.

---

## Suggested order of work for this repo

### Milestone A — Functional single-GPU AMD build
- add CMake backend switch
- compile with hipcc
- port runtime APIs
- keep CPU data loader only
- run one batch forward + backward

### Milestone B — Full single-GPU training
- restore all conv, BN, pool, FC paths with MIOpen and rocBLAS
- train for 1 epoch
- compare loss curve with NVIDIA

### Milestone C — Multi-GPU AMD training
- port NCCL pieces to RCCL
- verify gradient sync correctness
- run 2-GPU DDP training without DALI

### Milestone D — Performance tuning
- mixed precision
- stream tuning
- workspace tuning
- ROCm profiler pass

---

## What I would do first in this repo

1. Modify `CMakeLists.txt` to support a ROCm/HIP build option
2. Add a portability shim header for CUDA vs HIP runtime calls
3. Build the non-DALI single-GPU path on AMD first
4. Only after that, port the multi-GPU path

---

## Risk assessment

### Low risk
- CUDA runtime -> HIP runtime migration
- custom kernel syntax conversion
- disabling DALI for first AMD version

### Medium risk
- rocBLAS replacement for FC head
- BF16 re-enablement
- Slurm/container environment updates for ROCm nodes

### High risk
- cuDNN -> MIOpen parity and performance tuning
- DALI-equivalent AMD pipeline
- achieving NVIDIA-level throughput immediately

---

## Bottom line

For this repo, the correct method is **not** “convert everything at once”.

The best method is:

- first build an AMD-compatible non-DALI single-GPU version
- then restore multi-GPU via RCCL
- then optimize precision and throughput

That gives you the highest chance of a clean, debuggable port.
