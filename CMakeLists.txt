cmake_minimum_required(VERSION 3.18)
project(ResNet50_CUDA LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure CUDA architecture
# By passing -arch=native, nvcc directly reads the GPU hardware on the node
# during the SLURM build phase, bypassing older CMake limitations.
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

#find_package(Torch REQUIRED)

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

# ── Miniz (pure-C library, always compiled with the C compiler) ──────────────
add_library(miniz_lib STATIC miniz.c)
set_target_properties(miniz_lib PROPERTIES LINKER_LANGUAGE C)

# ── Core library ──────────────────────────────────────────────────────────────
add_library(resnet50_lib STATIC resnet50.cu)
target_include_directories(resnet50_lib PUBLIC ${CUDNN_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})
target_link_libraries(resnet50_lib PUBLIC miniz_lib CUDA::cublas ${NVTX_LIB} ${CUDNN_LIB} ${TORCH_LIBRARIES})

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
target_link_libraries(train_imagenet PRIVATE resnet50_lib CUDA::cublas ${NVTX_LIB} ${CUDNN_LIB} ${TORCH_LIBRARIES} stdc++fs)

# ── Benchmark ─────────────────────────────────────────────────────────────────
add_executable(benchmark benchmark.cu)
target_link_libraries(benchmark PRIVATE resnet50_lib CUDA::cublas ${NVTX_LIB} ${CUDNN_LIB} ${TORCH_LIBRARIES})

# ── Training benchmark ────────────────────────────────────────────────────────
add_executable(train_bench train_bench.cu)
target_link_libraries(train_bench PRIVATE resnet50_lib CUDA::cublas ${NVTX_LIB} ${CUDNN_LIB} ${TORCH_LIBRARIES})

# ── Optional DALI integration for the ImageNet multi-GPU path ────────────────
option(USE_DALI "Enable NVIDIA DALI for ImageNet multi-GPU training" OFF)

if(USE_DALI)
    find_path(DALI_INCLUDE_DIR dali/c_api.h
        HINTS
            ${DALI_ROOT}/include
            $ENV{DALI_ROOT}/include
            /usr/local/include
            /usr/include
    )
    find_library(DALI_LIBRARY NAMES dali
        HINTS
            ${DALI_ROOT}/lib
            ${DALI_ROOT}/lib64
            $ENV{DALI_ROOT}/lib
            $ENV{DALI_ROOT}/lib64
            /usr/local/lib
            /usr/local/lib64
            /usr/lib/x86_64-linux-gnu
    )

    if(NOT DALI_INCLUDE_DIR OR NOT DALI_LIBRARY)
        message(FATAL_ERROR
            "USE_DALI=ON but DALI headers/library were not found.\n"
            "Provide -DDALI_ROOT=/path/to/dali or export DALI_ROOT to the DALI install prefix.")
    endif()

    message(STATUS "DALI include   : ${DALI_INCLUDE_DIR}")
    message(STATUS "DALI library   : ${DALI_LIBRARY}")

    add_executable(dali_imagenet_smoke dali_imagenet_smoke.cu)
    target_compile_definitions(dali_imagenet_smoke PRIVATE USE_DALI)
    target_include_directories(dali_imagenet_smoke PRIVATE ${DALI_INCLUDE_DIR})
    target_link_libraries(dali_imagenet_smoke PRIVATE CUDA::cudart ${DALI_LIBRARY})
endif()

# ── Multi-GPU DDP training (requires NCCL) ────────────────────────────────────
# Enable with:  cmake -DUSE_NCCL=ON ..
option(USE_NCCL "Build the multi-GPU DDP training binary using NCCL" OFF)

if(USE_NCCL)
    # If specific NCCL library/include explicitly provided via CMake options
    if(DEFINED NCCL_LIB AND DEFINED NCCL_INCLUDE)
        set(NCCL_LIBRARY_ACTUAL "-L${NCCL_LIB} -lnccl")
        set(NCCL_INCLUDE_ACTUAL "${NCCL_INCLUDE}")
    else()
        # Find paths specifically for nvhpc which nests nccl inside a cuda version dir.
        set(EXTRA_HINTS_LIB "")
        set(EXTRA_HINTS_INC "")
        if(DEFINED ENV{NVHPC_ROOT})
            # NVHPC 24.11 format: NVHPC_ROOT/Linux_x86_64/24.11/comm_libs/12.6/nccl
            file(GLOB NVHPC_NCCL_PATHS "$ENV{NVHPC_ROOT}/Linux_x86_64/*/comm_libs/*/nccl")
            foreach(p IN LISTS NVHPC_NCCL_PATHS)
                list(APPEND EXTRA_HINTS_LIB "${p}/lib")
                list(APPEND EXTRA_HINTS_INC "${p}/include")
            endforeach()
        endif()

        # Robustly find NCCL using hints from command line (-DNCCL_ROOT=...) or env
        find_library(NCCL_LIBRARY NAMES nccl
            HINTS
                ${NCCL_ROOT}/lib
                ${NCCL_ROOT}/lib64
                $ENV{NCCL_HOME}/lib
                $ENV{EBROOTNCCL}/lib
                ${EXTRA_HINTS_LIB}
                ${CUDAToolkit_LIBRARY_DIR}
                /usr/local/cuda/lib64
        )
        find_path(NCCL_INCLUDE_DIR NAMES nccl.h
            HINTS
                ${NCCL_ROOT}/include
                $ENV{NCCL_HOME}/include
                $ENV{EBROOTNCCL}/include
                ${EXTRA_HINTS_INC}
                ${CUDAToolkit_INCLUDE_DIRS}
                /usr/local/cuda/include
        )

        # If NVHPC module or similar is loaded, the compiler might find it natively
        if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
            find_path(NCCL_INCLUDE_DIR_FALLBACK NAMES nccl.h)
            if(NCCL_INCLUDE_DIR_FALLBACK)
                set(NCCL_INCLUDE_DIR ${NCCL_INCLUDE_DIR_FALLBACK})
                # If it's in a standard include path, link against nccl natively using ENV variables
                set(NCCL_LIBRARY "nccl") 
            endif()
        endif()

        # In case find_library erroneously returned a directory path instead of library file
        if(IS_DIRECTORY "${NCCL_LIBRARY}")
            set(NCCL_LIBRARY_ACTUAL "-L${NCCL_LIBRARY} -lnccl")
        else()
            set(NCCL_LIBRARY_ACTUAL "${NCCL_LIBRARY}")
        endif()
        
        set(NCCL_INCLUDE_ACTUAL "${NCCL_INCLUDE_DIR}")
    endif()

    if(NCCL_LIBRARY_ACTUAL AND NCCL_INCLUDE_ACTUAL)
        message(STATUS "NCCL linking : ${NCCL_LIBRARY_ACTUAL}")
        message(STATUS "NCCL include : ${NCCL_INCLUDE_ACTUAL}")

        # train_multigpu recompiles resnet50.cu directly with USE_NCCL defined,
        # so SyncGradients (guarded by #ifdef USE_NCCL in resnet50.cu) is compiled in.
        # resnet50_lib stays NCCL-free → single-GPU targets never see ncclAllReduce.
        add_executable(train_multigpu train_multigpu.cu resnet50.cu)
        target_compile_definitions(train_multigpu PRIVATE USE_NCCL)
        target_include_directories(train_multigpu PRIVATE
            ${NCCL_INCLUDE_ACTUAL} ${CUDNN_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})
        target_link_libraries(train_multigpu PRIVATE
            CUDA::cublas ${NVTX_LIB} ${CUDNN_LIB} ${TORCH_LIBRARIES} ${NCCL_LIBRARY_ACTUAL} stdc++fs)

        if(USE_DALI)
            target_compile_definitions(train_multigpu PRIVATE USE_DALI)
            target_include_directories(train_multigpu PRIVATE ${DALI_INCLUDE_DIR})
            target_link_libraries(train_multigpu PRIVATE ${DALI_LIBRARY})
        endif()

        add_executable(train_multigpu_tiny train_multigpu_tiny.cu resnet50.cu)
        target_compile_definitions(train_multigpu_tiny PRIVATE USE_NCCL)
        target_include_directories(train_multigpu_tiny PRIVATE
            ${NCCL_INCLUDE_ACTUAL} ${CUDNN_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})
        target_link_libraries(train_multigpu_tiny PRIVATE
            CUDA::cublas ${NVTX_LIB} ${CUDNN_LIB} ${TORCH_LIBRARIES} ${NCCL_LIBRARY_ACTUAL} stdc++fs)

        message(STATUS "Multi-GPU DDP target 'train_multigpu' and 'train_multigpu_tiny' enabled.")
    else()
        message(FATAL_ERROR "USE_NCCL=ON but nccl.h or libnccl not found. \n"
                "Please provide -DNCCL_ROOT=/path/to/nccl, -DNCCL_LIB=/path/to/lib, or load nvhpc module")
    endif()
endif()

