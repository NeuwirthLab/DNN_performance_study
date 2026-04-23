# hip/CMakeLists.txt — Standalone HIP/ROCm build for ResNet-50.
#
# Invoke from the repo root on an AMD cluster:
#   cmake -S hip -B build-hip \
#         -DCMAKE_HIP_ARCHITECTURES=gfx942 \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DUSE_NCCL=ON
#   cmake --build build-hip -j16 --target train_multigpu_tiny
#
# Leaves the original CUDA CMakeLists.txt at the repo root untouched, so the
# CUDA build on HELMA keeps working unchanged.
cmake_minimum_required(VERSION 3.21)   # 3.21+ has first-class HIP language
project(ResNet50_HIP LANGUAGES C CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to MI300X if the user didn't specify.
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "gfx942")
endif()
message(STATUS "HIP arch(s): ${CMAKE_HIP_ARCHITECTURES}")

set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3")

# ── Paths: sources live one level up ─────────────────────────────────────
set(SRC_ROOT "${CMAKE_SOURCE_DIR}/..")
include_directories(${CMAKE_SOURCE_DIR} ${SRC_ROOT})

# ── ROCm libraries ───────────────────────────────────────────────────────
# Most ROCm installs set CMAKE_PREFIX_PATH=/opt/rocm. On HPC, the rocm
# module does this for you. If not, pass -DCMAKE_PREFIX_PATH=/opt/rocm.
find_package(hip REQUIRED)
find_package(hipblas REQUIRED)
find_package(miopen REQUIRED)

option(USE_NCCL "Build multi-GPU DDP targets (needs RCCL)" ON)
if(USE_NCCL)
    find_package(rccl REQUIRED)
endif()

# ── Tell hipcc to treat *.cu files as HIP source ─────────────────────────
# The repo keeps file extensions as .cu/.cuh for minimal churn; instruct
# the HIP language driver to compile them.
set(CUDA_SOURCES
    ${SRC_ROOT}/resnet50.cu
    ${SRC_ROOT}/main.cu
    ${SRC_ROOT}/train.cu
    ${SRC_ROOT}/train_tinyimagenet.cu
    ${SRC_ROOT}/train_multigpu.cu
    ${SRC_ROOT}/train_multigpu_tiny.cu
    ${SRC_ROOT}/benchmark.cu
)
set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE HIP)

# ── miniz (pure C) ───────────────────────────────────────────────────────
add_library(miniz_lib STATIC ${SRC_ROOT}/miniz.c)
set_target_properties(miniz_lib PROPERTIES LINKER_LANGUAGE C)

# ── Core library ─────────────────────────────────────────────────────────
add_library(resnet50_lib STATIC ${SRC_ROOT}/resnet50.cu)
target_compile_definitions(resnet50_lib PUBLIC USE_HIP)
target_link_libraries(resnet50_lib PUBLIC
    miniz_lib
    hip::device
    roc::hipblas
    MIOpen
)

# ── Single-GPU training on Tiny ImageNet (fastest smoke target) ─────────
add_executable(train_tinyimagenet ${SRC_ROOT}/train_tinyimagenet.cu)
target_compile_definitions(train_tinyimagenet PRIVATE USE_HIP)
target_link_libraries(train_tinyimagenet PRIVATE resnet50_lib)

# ── Multi-GPU (RCCL) training on Tiny ImageNet ──────────────────────────
if(USE_NCCL)
    add_executable(train_multigpu_tiny
        ${SRC_ROOT}/train_multigpu_tiny.cu
        ${SRC_ROOT}/resnet50.cu)
    target_compile_definitions(train_multigpu_tiny PRIVATE USE_HIP USE_NCCL)
    target_link_libraries(train_multigpu_tiny PRIVATE
        miniz_lib hip::device roc::hipblas MIOpen rccl)

    add_executable(train_multigpu
        ${SRC_ROOT}/train_multigpu.cu
        ${SRC_ROOT}/resnet50.cu)
    target_compile_definitions(train_multigpu PRIVATE USE_HIP USE_NCCL)
    target_link_libraries(train_multigpu PRIVATE
        miniz_lib hip::device roc::hipblas MIOpen rccl stdc++fs)
endif()

# ── Optional MIVisionX / rocAL GPU data loader ──────────────────────────
# rocAL is AMD's DALI equivalent (part of MIVisionX). Ships with ROCm on
# most installs under /opt/rocm/{include,lib}. Enable for ImageNet-scale
# runs; the Tiny smoke target runs fine with the CPU loader.
option(USE_ROCAL "Build with rocAL (MIVisionX) GPU data loader" OFF)
if(USE_ROCAL)
    find_path(ROCAL_INCLUDE_DIR rocal/rocal_api.h
        HINTS ${ROCAL_ROOT}/include $ENV{ROCM_PATH}/include /opt/rocm/include)
    find_library(ROCAL_LIBRARY rocal
        HINTS ${ROCAL_ROOT}/lib $ENV{ROCM_PATH}/lib /opt/rocm/lib)
    if(ROCAL_INCLUDE_DIR AND ROCAL_LIBRARY)
        message(STATUS "rocAL include : ${ROCAL_INCLUDE_DIR}")
        message(STATUS "rocAL library : ${ROCAL_LIBRARY}")
        if(TARGET train_multigpu)
            target_compile_definitions(train_multigpu PRIVATE USE_ROCAL)
            target_include_directories(train_multigpu PRIVATE ${ROCAL_INCLUDE_DIR})
            target_link_libraries(train_multigpu PRIVATE ${ROCAL_LIBRARY})
        endif()
    else()
        message(WARNING
            "USE_ROCAL=ON but rocal_api.h or librocal not found.\n"
            "  Install: module load mivisionx  (or apt install rocal-dev)\n"
            "  Or: cmake -S hip -B build-hip -DUSE_ROCAL=ON -DROCAL_ROOT=/opt/rocm")
    endif()
endif()

message(STATUS "HIP build configured. Targets: train_tinyimagenet"
               "$<$<BOOL:${USE_NCCL}>:, train_multigpu_tiny, train_multigpu>")
