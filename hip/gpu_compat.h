// gpu_compat.h — Runtime + cuBLAS + NCCL compatibility shim for CUDA/HIP.
//
// Use:
//   - CUDA build (default): do NOT define USE_HIP. Symbols stay CUDA-native.
//   - HIP build  : add -DUSE_HIP=1 on the compile line (or target_compile_definitions).
//                  The header remaps cuda*/cublas* tokens to hip*/hipblas*
//                  equivalents so the rest of the codebase compiles unchanged.
//
// Scope: runtime, cuBLAS, and NCCL only. cuDNN ↔ MIOpen differ enough in the
// algorithm-selection API that they live in dnn_compat.h.
#pragma once

#ifdef USE_HIP
  #include <hip/hip_runtime.h>
  #include <hipblas/hipblas.h>
  // RCCL is a drop-in: headers still expose the nccl* symbol names.
  #include <rccl/rccl.h>

  // ── runtime ────────────────────────────────────────────────────────────
  #define cudaError_t                 hipError_t
  #define cudaSuccess                 hipSuccess
  #define cudaGetErrorString          hipGetErrorString
  #define cudaGetLastError            hipGetLastError
  #define cudaDeviceSynchronize       hipDeviceSynchronize
  #define cudaDeviceReset             hipDeviceReset
  #define cudaSetDevice               hipSetDevice
  #define cudaGetDevice               hipGetDevice
  #define cudaGetDeviceCount          hipGetDeviceCount
  #define cudaMalloc                  hipMalloc
  #define cudaFree                    hipFree
  #define cudaMallocHost              hipHostMalloc
  #define cudaFreeHost                hipHostFree
  #define cudaMemcpy                  hipMemcpy
  #define cudaMemcpyAsync             hipMemcpyAsync
  #define cudaMemset                  hipMemset
  #define cudaMemsetAsync             hipMemsetAsync
  #define cudaMemcpyHostToDevice      hipMemcpyHostToDevice
  #define cudaMemcpyDeviceToHost      hipMemcpyDeviceToHost
  #define cudaMemcpyDeviceToDevice    hipMemcpyDeviceToDevice
  #define cudaMemcpyHostToHost        hipMemcpyHostToHost

  #define cudaStream_t                hipStream_t
  #define cudaStreamCreate            hipStreamCreate
  #define cudaStreamCreateWithFlags   hipStreamCreateWithFlags
  #define cudaStreamDestroy           hipStreamDestroy
  #define cudaStreamSynchronize       hipStreamSynchronize
  #define cudaStreamNonBlocking       hipStreamNonBlocking
  #define cudaStreamWaitEvent         hipStreamWaitEvent

  #define cudaEvent_t                 hipEvent_t
  #define cudaEventCreate             hipEventCreate
  #define cudaEventCreateWithFlags    hipEventCreateWithFlags
  #define cudaEventDestroy            hipEventDestroy
  #define cudaEventRecord             hipEventRecord
  #define cudaEventSynchronize        hipEventSynchronize
  #define cudaEventElapsedTime        hipEventElapsedTime
  #define cudaEventDisableTiming      hipEventDisableTiming

  #define cudaDeviceProp              hipDeviceProp_t
  #define cudaGetDeviceProperties     hipGetDeviceProperties

  // ── cuBLAS → hipBLAS ──────────────────────────────────────────────────
  #define cublasHandle_t              hipblasHandle_t
  #define cublasStatus_t              hipblasStatus_t
  #define CUBLAS_STATUS_SUCCESS       HIPBLAS_STATUS_SUCCESS
  #define cublasCreate                hipblasCreate
  #define cublasDestroy               hipblasDestroy
  #define cublasSetStream             hipblasSetStream
  #define cublasSgemm                 hipblasSgemm
  #define cublasSnrm2                 hipblasSnrm2
  #define cublasSaxpy                 hipblasSaxpy
  #define cublasSscal                 hipblasSscal
  #define CUBLAS_OP_N                 HIPBLAS_OP_N
  #define CUBLAS_OP_T                 HIPBLAS_OP_T
  #define CUBLAS_OP_C                 HIPBLAS_OP_C

  // NVTX → no-op on HIP (swap to roctx later if profiling is needed).
  #define nvtxRangePushA(x)           ((void)0)
  #define nvtxRangePop()              ((void)0)

  // NCCL: RCCL keeps the nccl* symbol names, so no remap is required.
  // Just link -lrccl in place of -lnccl.
#else
  #include <cuda_runtime.h>
  #include <cublas_v2.h>
  #include <nccl.h>
  #include <nvtx3/nvToolsExt.h>
#endif
