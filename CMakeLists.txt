cmake_minimum_required(VERSION 3.18)
project(ResNet50_CUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure CUDA architecture
# You can override this from the command line: cmake -DCMAKE_CUDA_ARCHITECTURES=80 ..
# (e.g., 90 for H100, 80 for A100, 70 for V100, 89 for RTX 500/4090)
set(CMAKE_CUDA_ARCHITECTURES "90" CACHE STRING "CUDA architecture")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -diag-suppress=170,177,550,1675")

# LibTorch and NVTX requirements
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(CUDAToolkit REQUIRED)

# CUDA 12+ removed nvToolsExt (replaced by header-only nvtx3).
# PyTorch's cmake config still references CUDA::nvToolsExt, so create a
# compatibility target before find_package(Torch) to prevent a fatal error.
if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
    target_include_directories(CUDA::nvToolsExt INTERFACE ${CUDAToolkit_INCLUDE_DIRS})
endif()

find_package(Torch REQUIRED)

# ── Find cuDNN library ────────────────────────────────────────────────────────
find_library(CUDNN_LIB cudnn
    HINTS
        ${CUDNN_PATH}/lib
        ${CUDNN_PATH}/lib64
        ${CUDAToolkit_LIBRARY_DIR}
        $ENV{CUDNN_PATH}/lib
        $ENV{CUDNN_PATH}/lib64
        /usr/local/cuda/lib64
        /usr/lib/x86_64-linux-gnu
)

# ── Find cuDNN include (cudnn.h) ──────────────────────────────────────────────
find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS
        ${CUDNN_PATH}/include
        ${CUDAToolkit_INCLUDE_DIRS}
        $ENV{CUDNN_PATH}/include
        $ENV{CUDNN_PATH}
        /usr/local/cuda/include
        /usr/include
)

if(NOT CUDNN_LIB)
    message(FATAL_ERROR
        "cuDNN library not found.\n"
        "  → Run: module avail cudnn  (on the HPC) to find the right module name\n"
        "  → Or:  cmake .. -DCUDNN_PATH=/path/to/cudnn")
endif()
if(NOT CUDNN_INCLUDE_DIR)
    message(FATAL_ERROR
        "cudnn.h not found.\n"
        "  → Run: find /usr /opt -name cudnn.h 2>/dev/null  to locate it\n"
        "  → Or:  cmake .. -DCUDNN_PATH=/path/to/cudnn")
endif()

message(STATUS "cuDNN library : ${CUDNN_LIB}")
message(STATUS "cuDNN include : ${CUDNN_INCLUDE_DIR}")

# ── NVTX: prefer nvtx3 (header-only) on CUDA 12+, fall back to old nvToolsExt
if(TARGET CUDA::nvToolsExt)
    set(NVTX_LIB CUDA::nvToolsExt)
elseif(TARGET CUDA::nvtx3)
    set(NVTX_LIB CUDA::nvtx3)
else()
    # Header-only nvtx3 ships with CUDA 12+ in include/nvtx3/
    # Create an interface target so we don't need a library
    add_library(nvtx3_headers INTERFACE)
    target_include_directories(nvtx3_headers INTERFACE ${CUDAToolkit_INCLUDE_DIRS})
    set(NVTX_LIB nvtx3_headers)
endif()

# ── Core library ──────────────────────────────────────────────────────────────
add_library(resnet50_lib STATIC resnet50.cu)
target_include_directories(resnet50_lib PUBLIC ${CUDNN_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})
target_link_libraries(resnet50_lib PUBLIC CUDA::cublas ${NVTX_LIB} ${CUDNN_LIB} ${TORCH_LIBRARIES})

# ── Inference demo ────────────────────────────────────────────────────────────
add_executable(main main.cu)
target_link_libraries(main PRIVATE resnet50_lib CUDA::cublas ${NVTX_LIB} ${CUDNN_LIB} ${TORCH_LIBRARIES})

# ── Training on synthetic data ────────────────────────────────────────────────
add_executable(train train.cu)
target_link_libraries(train PRIVATE resnet50_lib CUDA::cublas ${NVTX_LIB} ${CUDNN_LIB} ${TORCH_LIBRARIES})

# ── Training on Tiny ImageNet ─────────────────────────────────────────────────
add_executable(train_tinyimagenet train_tinyimagenet.cu)
target_link_libraries(train_tinyimagenet PRIVATE resnet50_lib CUDA::cublas ${NVTX_LIB} ${CUDNN_LIB} ${TORCH_LIBRARIES})

# ── Training on ImageNet-1k ───────────────────────────────────────────────────
add_executable(train_imagenet train_imagenet.cu)
target_link_libraries(train_imagenet PRIVATE resnet50_lib CUDA::cublas ${NVTX_LIB} ${CUDNN_LIB} ${TORCH_LIBRARIES})

# ── Benchmark ─────────────────────────────────────────────────────────────────
add_executable(benchmark benchmark.cu)
target_link_libraries(benchmark PRIVATE resnet50_lib CUDA::cublas ${NVTX_LIB} ${CUDNN_LIB} ${TORCH_LIBRARIES})

# ── Training benchmark ────────────────────────────────────────────────────────
add_executable(train_bench train_bench.cu)
target_link_libraries(train_bench PRIVATE resnet50_lib CUDA::cublas ${NVTX_LIB} ${CUDNN_LIB} ${TORCH_LIBRARIES})

# ── Multi-GPU DDP training (requires NCCL) ────────────────────────────────────
# Enable with:  cmake -DUSE_NCCL=ON ..
option(USE_NCCL "Build the multi-GPU DDP training binary using NCCL" OFF)

if(USE_NCCL)
    # 1. Prioritize variables passed from command line (-DNCCL_LIB=... -DNCCL_INCLUDE=...)
    if(NOT NCCL_LIB OR NOT NCCL_INCLUDE)
        find_library(NCCL_LIB nccl
            HINTS
                ${CUDAToolkit_LIBRARY_DIR}
                $ENV{NCCL_HOME}/lib
                $ENV{EBROOTNCCL}/lib
                $ENV{NVHPC_ROOT}/Linux_x86_64/$ENV{NVHPC_VERSION}/comm_libs/nccl/lib
                /usr/local/cuda/lib64
        )
        find_path(NCCL_INCLUDE nccl.h
            HINTS
                ${CUDAToolkit_INCLUDE_DIRS}
                $ENV{NCCL_HOME}/include
                $ENV{EBROOTNCCL}/include
                $ENV{NVHPC_ROOT}/Linux_x86_64/$ENV{NVHPC_VERSION}/comm_libs/nccl/include
                /usr/local/cuda/include
        )
    endif()

    if(NCCL_LIB AND NCCL_INCLUDE)
        message(STATUS "NCCL library  : ${NCCL_LIB}")
        message(STATUS "NCCL include  : ${NCCL_INCLUDE}")

        # train_multigpu recompiles resnet50.cu directly with USE_NCCL defined,
        # so SyncGradients (guarded by #ifdef USE_NCCL in resnet50.cu) is compiled in.
        # resnet50_lib stays NCCL-free → single-GPU targets never see ncclAllReduce.
        add_executable(train_multigpu train_multigpu.cu resnet50.cu)
        target_compile_definitions(train_multigpu PRIVATE USE_NCCL)
        target_include_directories(train_multigpu PRIVATE
            ${NCCL_INCLUDE} ${CUDNN_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})
        target_link_libraries(train_multigpu PRIVATE
            CUDA::cublas ${NVTX_LIB} ${CUDNN_LIB} ${TORCH_LIBRARIES} ${NCCL_LIB})

        message(STATUS "Multi-GPU DDP target 'train_multigpu' enabled.")
    else()
        message(FATAL_ERROR "USE_NCCL=ON but nccl.h or libnccl not found. \n"
                "Please provide -DNCCL_INCLUDE=/path/to/inc -DNCCL_LIB=/path/to/libnccl.so")
    endif()
endif()

