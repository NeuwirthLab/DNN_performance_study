# ResNet50 Repo Architecture

This repo is easiest to understand if you separate it into four layers:

1. Entry points: binaries and Slurm scripts that decide what runs.
2. Data pipeline: CPU loaders or DALI feed batches into GPU memory.
3. Model core: ResNet-50 forward, backward, loss, optimizer, checkpointing, NCCL sync.
4. Build/runtime glue: CMake, CUDA, cuDNN, NCCL, DALI, Apptainer, TMPDIR staging.

## 1. Main Code Paths

### Main training path

- `run_multigpu_tmpdir_v2.slurm`
- `train_multigpu.cu`
- `mt_image_loader.h` + `prefetch_loader.h`
- `resnet50.cu` / `resnet50.cuh`

This is the current main ImageNet multi-GPU path.

### Optional DALI training path

- `run_multigpu_tmpdir_dali.slurm`
- `train_multigpu.cu` with `USE_DALI=ON`
- `dali_imagenet_loader.h`
- `tools/generate_imagenet_dali_filelists.py`
- `tools/serialize_imagenet_dali.py`

This keeps the model code unchanged and swaps only the input path.

### Standalone DALI validation path

- `run_dali_imagenet_smoke.slurm`
- `dali_imagenet_smoke.cu`
- `dali_imagenet_loader.h`

This is the smallest native DALI path in the repo. Use it to verify that DALI C API works on staged ImageNet data before debugging the full trainer.

## 2. File Roles

### Build and feature selection

- `CMakeLists.txt`

Defines all binaries, optional `USE_NCCL`, optional `USE_DALI`, and external dependencies.

### Model core

- `resnet50.cuh`
- `resnet50.cu`

These files implement the actual network and most GPU-side computation:

- `ConvBnLayer`: cuDNN convolution + batch norm + optional ReLU
- `BottleneckBlock`: residual bottleneck block
- `ResNet50`: full model, loss, accuracy, backward pass, SGD, checkpointing, NCCL sync

If you understand these two files, you understand the model implementation.

### Multi-GPU training control plane

- `train_multigpu.cu`

This is the best file for understanding how the system fits together. It handles:

- GPU discovery
- NCCL communicator setup
- dataset pre-scan
- one worker thread per GPU
- loader creation
- training loop
- validation loop
- logging and checkpointing

### Data loading

- `imagenet_loader.h`: baseline single-threaded ImageNet loader
- `mt_image_loader.h`: multi-threaded CPU decode/transform loader
- `prefetch_loader.h`: background prefetch queue around a loader
- `tiny_dataloader.h`: Tiny ImageNet loader
- `dali_imagenet_loader.h`: thin native wrapper around serialized DALI pipelines

The important design choice is that the model does not care where batches come from. The loader boundary is where CPU and DALI paths differ.

### Helper scripts

- `tools/generate_imagenet_dali_filelists.py`
- `tools/serialize_imagenet_dali.py`
- `tools/generate_tiny_imagenet_filelists.py`
- `tools/dali_tiny_imagenet_bench.py`

These scripts prepare external assets. They are not the trainer itself.

### Slurm launchers

- `run_multigpu_tmpdir_v2.slurm`: main non-DALI ImageNet job
- `run_multigpu_tmpdir_dali.slurm`: DALI-enabled multi-GPU ImageNet job
- `run_dali_imagenet_smoke.slurm`: standalone DALI smoke test
- other `.slurm` files: experiments, older runs, or narrower launch paths

## 3. End-to-End Flow

### Non-DALI multi-GPU training

1. Slurm stages ImageNet ZIP into node-local TMPDIR.
2. `train_multigpu.cu` pre-scans dataset paths once.
3. One worker thread is launched per GPU.
4. Each worker creates a model replica and a shard-aware loader.
5. CPU loader decodes and transforms images.
6. Host buffers are copied to GPU.
7. `ResNet50::Forward()` runs.
8. `ResNet50::Backward()` runs.
9. `ResNet50::SyncGradients()` averages gradients across GPUs using NCCL.
10. `ResNet50::UpdateSGD()` updates weights.

### DALI path

1. Slurm stages ImageNet into TMPDIR.
2. Python scripts generate per-rank file lists and serialized `.dali` pipelines.
3. `dali_imagenet_loader.h` loads one serialized pipeline per rank.
4. DALI outputs batches directly into device buffers.
5. The model path remains the same from forward pass onward.

## 4. Reading Order

Use this order if you want the big picture quickly:

1. `CMakeLists.txt`
2. `train_multigpu.cu`
3. `resnet50.cuh`
4. `resnet50.cu`
5. `imagenet_loader.h`
6. `mt_image_loader.h`
7. `prefetch_loader.h`
8. `run_multigpu_tmpdir_v2.slurm`

If you want the DALI branch after that:

1. `dali_imagenet_loader.h`
2. `dali_imagenet_smoke.cu`
3. `tools/generate_imagenet_dali_filelists.py`
4. `tools/serialize_imagenet_dali.py`
5. `run_dali_imagenet_smoke.slurm`

## 5. How To Build A Mental Model

Trace one batch only. Do not read every file linearly.

For the current main path, trace:

1. `run_multigpu_tmpdir_v2.slurm`
2. `train_multigpu.cu:main()`
3. `gpu_worker()` in `train_multigpu.cu`
4. `train_loader.GetNextBatch(...)`
5. `model.Forward(...)`
6. `model.Backward(...)`
7. `model.SyncGradients(...)`
8. `model.UpdateSGD(...)`

If you can follow that single path, you already understand the repo at a system level.

## 6. Current Architecture Summary

The repo is not primarily a DALI repo. It is a native CUDA/cuDNN/NCCL ResNet-50 training repo with interchangeable input paths.

The most important split is:

- control flow and orchestration in `train_multigpu.cu`
- math and state in `resnet50.cu` / `resnet50.cuh`
- input pipeline in loader headers or DALI wrapper
- environment and staging in Slurm scripts

That is the big picture.