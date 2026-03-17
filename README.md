# Performance Analysis of Deep Learning Models: ResNet-50
A C++/CUDA implementation of the ResNet-50 architecture. This project is built to benchmark, profile, and deeply analyze deep learning training performance on High-Performance Computing (HPC) clusters utilizing NVIDIA and AMD GPUs.

This performance analysis is comparing implementation in CUDA vs PyTorch across different GPUs, working as a profiling testbed to understand GPU utilization, memory bandwidth bottlenecks, and Tensor Core utilization.( To be inclusded soon)

> **Note:** A very known issue is having problems with Libtorch compatibility. For that, look into the architecture you want to build it on using libtorch, and check the correct versions at the [PyTorch installation page](https://pytorch.org/get-started/locally/).

## 🚀 Key Features

*   **Pure C++/CUDA Implementation:** Core neural network logic is implemented directly using the `cuDNN` API (for convolutions, batch norm, and pooling) and `cuBLAS` (for fully connected layers).
*   **TF32 Acceleration:** Automatically enables Tensor Float 32 (TF32) math for Ampere and Hopper GPUs, significantly accelerating matrix multiplications while maintaining FP32 dynamic range.
*   **Advanced Training Loop:** Includes a Cosine Annealing learning rate schedule with linear warmup, validation pass monitoring, and early stopping based on generalization performance.
*   **Custom C++ DataLoader:** A lightweight, highly efficient `TinyImageNetLoader` that handles on-the-fly data augmentation (random horizontal flipping), normalization, and batching without the Python Global Interpreter Lock (GIL) overhead.
*   **Built for Profiling:** The codebase is heavily instrumented with NVTX (NVIDIA Tools Extension) markers, allowing for fine-grained timeline analysis using **NVIDIA Nsight Systems (`nsys`)**.
*   **Binary Checkpointing:** Features a robust, framework-independent binary serialization format. Training jobs gracefully save state at the end of each epoch and can seamlessly resume—critical for HPC environments with strict job time limits.

## 📋 Requirements

*   **Compiler:** GCC 11+
*   **Build System:** CMake 3.26+
*   **NVIDIA Toolkit:** CUDA Toolkit 12.6+ (recommended)
*   **Deep Learning Primitives:** cuDNN 9.8+ (CUDA 12 build recommended)
*   **Hardware:** A CUDA-capable GPU (NVIDIA Ampere, Ada Lovelace, or Hopper architectures strongly recommended to benefit from TF32).

## 🛠️ Installation & Building

1. **Clone the repository:**
   ```bash
   git clone https://gitlab.rlp.net/averma/performance-analysis-dl-models-resnet50.git
   cd performance-analysis-dl-models-resnet50
   ```

2. **Configure and Build:**
   Ensure your CUDA and cuDNN environments are loaded, then build using CMake:
   ```bash
   mkdir -p build && cd build
   # CMake will automatically detect the best CUDA architecture for your GPU
   # You can override it via -DCMAKE_CUDA_ARCHITECTURES="90" (for H100)
   cmake .. 
   make -j$(nproc)
   cd ..
   ```

## 💻 Usage

### 1. Data Preparation

**For Tiny ImageNet:**
```bash
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
```

**For ImageNet-1k:**
The highest performance, natively supported structure for C++ DataLoaders is the raw JPEG folder hierarchy. The easiest way to get this on an HPC is via **Kaggle**:
1. Make sure your `kaggle.json` API token is inside `~/.kaggle/kaggle.json`.
   > **Note:** If `~/.kaggle` does not work on your HPC, export your credentials directly in your run session:
   > `export KAGGLE_USERNAME="your_username" && export KAGGLE_KEY="your_api_key"`
2. Run the following:
```bash
pip install kaggle
mkdir -p imagenet && cd imagenet
kaggle competitions download -c imagenet-object-localization-challenge

# ⚠️ Troubleshooting: If you get "bash: kaggle: command not found" on your HPC,
# run the python module directly instead of the kaggle CLI wrapper:
# python -m kaggle competitions download -c imagenet-object-localization-challenge

unzip imagenet-object-localization-challenge.zip
mv ILSVRC/Data/CLS-LOC/train .
mv ILSVRC/Data/CLS-LOC/val .
```

### 2. Local Training
To run a training session locally (e.g., Batch Size 32, 50 Epochs, Base LR 0.01):
```bash
./build/train_tinyimagenet ./tiny-imagenet-200 32 50 0.01
```
Checkpoints and CSV training logs will be dumped into the `checkpoints/` directory.

### 3. Running on HPC (SLURM)
Two robust SLURM scripts are provided to dispatch jobs to a cluster scheduler.

*   **Standard Training:**
    `sbatch run_resnet.slurm` 
    This script automatically loads the necessary modules (CUDA 12.6, cuDNN 9.8), compiles the code, and launches a long-running training job utilizing a single H100 GPU. If resumed, it picks up from the last epoch saved in `checkpoints/`.

*   **Nsight Systems Profiling:**
    `sbatch profile_resnet.slurm`
    This script compiles the code and executes a shortened training loop under the `nsys profile` wrapper. It traces CUDA API calls, cuDNN/cuBLAS usage, and custom NVTX ranges. Once complete, it generates a `.nsys-rep` file which you can download and visualize using the Nsight Systems GUI on your local workstation.

## 📊 File Structure

*   `resnet50.cu` / `.cuh`: The core Deep Learning architecture, containing forward/backward passes and the custom binary checkpointing logic.
*   `train_tinyimagenet.cu`: The C++ entry point containing the training, validation, and serialization loops.
*   `tiny_dataloader.h`: Custom header-only C++ data loader utilizing `stb_image` for fast JPEG decoding.
*   `run_resnet.slurm`: Deployment script for cluster execution.
*   `profile_resnet.slurm`: Deployment script tuned for Nsight Systems hardware profiling.

## 📫 Acknowledgments
Developed for performance analysis and profiling of Deep Learning workloads on the NHR@FAU Helma cluster.


## Future works
This implementation is being tested with Imagenet1k, having appx. 1.2M training images and 50k validation images. Heavily utilising the GPUs.  