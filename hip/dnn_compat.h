// dnn_compat.h — Compile-time Compatibility Shim: cuDNN ↔ MIOpen
#include <vector>
//
// ROCm/HIP COMPATIBILITY DESIGN:
// Rather than permanently modifying the original CUDA codebase using hipify,
// this shim intercepts standard cuDNN calls (via -DUSE_HIP) and maps them
// to their AMD MIOpen equivalents. 
//
// Key Responsibilities:
// 1. Remapping missing enums and algorithm types.
// 2. Struct adaptation (e.g., matching cuDNN's algo perf structs).
// 3. Inline wrappers to handle API differences (e.g., argument ordering, 
//    const cast removal for MIOpen BN, and thread-local workspace management 
//    for MIOpen Max Pooling backward and algorithm Find passes).
#pragma once
#include "gpu_compat.h"

#ifdef USE_HIP
  #include <miopen/miopen.h>

  // ── handle / status ───────────────────────────────────────────────────
  #define cudnnHandle_t                          miopenHandle_t
  #define cudnnStatus_t                          miopenStatus_t
  #define CUDNN_STATUS_SUCCESS                   miopenStatusSuccess
  #define cudnnCreate                            miopenCreate
  #define cudnnDestroy                           miopenDestroy
  #define cudnnSetStream                         miopenSetStream
  #define cudnnGetErrorString                    miopenGetErrorString

  // ── tensor descriptors ────────────────────────────────────────────────
  #define cudnnTensorDescriptor_t                miopenTensorDescriptor_t
  #define cudnnCreateTensorDescriptor            miopenCreateTensorDescriptor
  #define cudnnDestroyTensorDescriptor           miopenDestroyTensorDescriptor

  // ── filter / conv / pool descriptors ──────────────────────────────────
  #define cudnnFilterDescriptor_t                miopenTensorDescriptor_t
  #define cudnnCreateFilterDescriptor            miopenCreateTensorDescriptor
  #define cudnnDestroyFilterDescriptor           miopenDestroyTensorDescriptor
  #define cudnnConvolutionDescriptor_t           miopenConvolutionDescriptor_t
  #define cudnnCreateConvolutionDescriptor       miopenCreateConvolutionDescriptor
  #define cudnnDestroyConvolutionDescriptor      miopenDestroyConvolutionDescriptor
  #define cudnnPoolingDescriptor_t               miopenPoolingDescriptor_t
  #define cudnnCreatePoolingDescriptor           miopenCreatePoolingDescriptor
  #define cudnnDestroyPoolingDescriptor          miopenDestroyPoolingDescriptor

  // ── algorithm types ───────────────────────────────────────────────────
  #define cudnnConvolutionFwdAlgo_t              miopenConvFwdAlgorithm_t
  #define cudnnConvolutionBwdFilterAlgo_t        miopenConvBwdWeightsAlgorithm_t
  #define cudnnConvolutionBwdDataAlgo_t          miopenConvBwdDataAlgorithm_t

  // ── perf structs ──────────────────────────────────────────────────────
  // cuDNN perf structs have .algo and .memory; MIOpen's miopenConvAlgoPerf_t
  // uses a union of fwd_algo/bwd_weights_algo/bwd_data_algo. We create
  // compat structs so source code can use .algo and .memory uniformly.
  struct cudnnConvolutionFwdAlgoPerf_t {
      miopenConvFwdAlgorithm_t algo;
      float time;
      size_t memory;
  };
  struct cudnnConvolutionBwdFilterAlgoPerf_t {
      miopenConvBwdWeightsAlgorithm_t algo;
      float time;
      size_t memory;
  };
  struct cudnnConvolutionBwdDataAlgoPerf_t {
      miopenConvBwdDataAlgorithm_t algo;
      float time;
      size_t memory;
  };

  // ── dtype / format / mode enums ───────────────────────────────────────
  #define CUDNN_DATA_FLOAT                       miopenFloat
  #define CUDNN_DATA_HALF                        miopenHalf
  #define CUDNN_TENSOR_NCHW                      0   // MIOpen is NCHW by default

  #define CUDNN_BATCHNORM_SPATIAL                miopenBNSpatial
  #define CUDNN_BATCHNORM_PER_ACTIVATION         miopenBNPerActivation
  #define cudnnBatchNormMode_t                   miopenBatchNormMode_t

  #define CUDNN_POOLING_MAX                      miopenPoolingMax
  #define CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING miopenPoolingAverageInclusive
  #define CUDNN_PROPAGATE_NAN                    0
  #define CUDNN_NOT_PROPAGATE_NAN                0

  #define CUDNN_CROSS_CORRELATION                miopenConvolution

  #define CUDNN_SOFTMAX_ACCURATE                 MIOPEN_SOFTMAX_ACCURATE
  #define CUDNN_SOFTMAX_MODE_INSTANCE            MIOPEN_SOFTMAX_MODE_INSTANCE

  // cuDNN's math-type selector has no direct MIOpen counterpart.
  #define cudnnSetConvolutionMathType(desc, mode) miopenStatusSuccess
  #define CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION  0
  #define CUBLAS_TF32_TENSOR_OP_MATH             0
  #define cublasSetMathMode(handle, mode)        HIPBLAS_STATUS_SUCCESS

  // ── inline wrappers: descriptor setters ───────────────────────────────
  static inline miopenStatus_t
  _hip_set_tensor_4d(miopenTensorDescriptor_t d, int /*format*/,
                     miopenDataType_t dt, int n, int c, int h, int w) {
      return miopenSet4dTensorDescriptor(d, dt, n, c, h, w);
  }
  #define cudnnSetTensor4dDescriptor(d, fmt, dt, n, c, h, w) \
      _hip_set_tensor_4d((d), (int)(fmt), (dt), (n), (c), (h), (w))

  static inline miopenStatus_t
  _hip_set_filter_4d(miopenTensorDescriptor_t d, miopenDataType_t dt,
                     int /*format*/, int k, int c, int h, int w) {
      return miopenSet4dTensorDescriptor(d, dt, k, c, h, w);
  }
  #define cudnnSetFilter4dDescriptor(d, dt, fmt, k, c, h, w) \
      _hip_set_filter_4d((d), (dt), (int)(fmt), (k), (c), (h), (w))

  static inline miopenStatus_t
  _hip_set_conv_2d(miopenConvolutionDescriptor_t d,
                   int pad_h, int pad_w, int u, int v,
                   int dil_h, int dil_w, int /*mode*/, int /*dtype*/) {
      return miopenInitConvolutionDescriptor(d, miopenConvolution,
                                             pad_h, pad_w, u, v, dil_h, dil_w);
  }
  #define cudnnSetConvolution2dDescriptor(d, ph, pw, u, v, dh, dw, mode, dt) \
      _hip_set_conv_2d((d), (ph), (pw), (u), (v), (dh), (dw), (int)(mode), (int)(dt))

  static inline miopenStatus_t
  _hip_set_pool_2d(miopenPoolingDescriptor_t d, int mode, int /*nanprop*/,
                   int wh, int ww, int ph, int pw, int sh, int sw) {
      return miopenSet2dPoolingDescriptor(d, (miopenPoolingMode_t)mode,
                                          wh, ww, ph, pw, sh, sw);
  }
  #define cudnnSetPooling2dDescriptor(d, mode, nanprop, wh, ww, ph, pw, sh, sw) \
      _hip_set_pool_2d((d), (int)(mode), (int)(nanprop), (wh), (ww), (ph), (pw), (sh), (sw))

  // ═══════════════════════════════════════════════════════════════════════
  // Algorithm selection wrappers
  // ═══════════════════════════════════════════════════════════════════════

  // cudnnFindConvolutionForwardAlgorithmEx → miopenFindConvolutionForwardAlgorithm
  // MIOpen's Find fills miopenConvAlgoPerf_t (with .fwd_algo); we convert
  // to our compat struct which has .algo.
  static inline miopenStatus_t
  cudnnFindConvolutionForwardAlgorithmEx(
      miopenHandle_t handle,
      miopenTensorDescriptor_t xDesc, const void* x,
      miopenTensorDescriptor_t wDesc, const void* w,
      miopenConvolutionDescriptor_t convDesc,
      miopenTensorDescriptor_t yDesc, void* y,
      int requestedAlgoCount, int* returnedAlgoCount,
      cudnnConvolutionFwdAlgoPerf_t* perfResults,
      void* workSpace, size_t workSpaceSizeInBytes)
  {
      // Use MIOpen's native perf struct
      std::vector<miopenConvAlgoPerf_t> mi_perfs(requestedAlgoCount);
      miopenStatus_t st = miopenFindConvolutionForwardAlgorithm(
          handle, xDesc, x, wDesc, w, convDesc, yDesc, y,
          requestedAlgoCount, returnedAlgoCount, mi_perfs.data(),
          workSpace, workSpaceSizeInBytes, /*exhaustiveSearch=*/true);
      // Convert to compat struct
      for (int i = 0; i < *returnedAlgoCount && i < requestedAlgoCount; ++i) {
          perfResults[i].algo   = mi_perfs[i].fwd_algo;
          perfResults[i].time   = mi_perfs[i].time;
          perfResults[i].memory = mi_perfs[i].memory;
      }
      return st;
  }

  // cudnnGetConvolutionBackwardFilterAlgorithm_v7
  // cuDNN v7 heuristic doesn't need data; MIOpen's Find does.
  // We just pick the first algorithm returned by MIOpen's immediate mode.
  static inline miopenStatus_t
  cudnnGetConvolutionBackwardFilterAlgorithm_v7(
      miopenHandle_t handle,
      miopenTensorDescriptor_t xDesc,
      miopenTensorDescriptor_t dyDesc,
      miopenConvolutionDescriptor_t convDesc,
      miopenTensorDescriptor_t dwDesc,
      int requestedAlgoCount, int* returnedAlgoCount,
      cudnnConvolutionBwdFilterAlgoPerf_t* perfResults)
  {
      // MIOpen requires Find to register invokers. Since cuDNN v7 heuristic
      // doesn't provide data pointers, we allocate temporary ones.
      size_t x_sz=0, dy_sz=0, dw_sz=0, ws_sz=0;
      miopenGetTensorNumBytes(xDesc, &x_sz);
      miopenGetTensorNumBytes(dyDesc, &dy_sz);
      miopenGetTensorNumBytes(dwDesc, &dw_sz);
      miopenConvolutionBackwardWeightsGetWorkSpaceSize(handle, dyDesc, xDesc, convDesc, dwDesc, &ws_sz);

      void *d_x=nullptr, *d_dy=nullptr, *d_dw=nullptr, *d_ws=nullptr;
      if (x_sz) cudaMalloc(&d_x, x_sz);
      if (dy_sz) cudaMalloc(&d_dy, dy_sz);
      if (dw_sz) cudaMalloc(&d_dw, dw_sz);
      if (ws_sz) cudaMalloc(&d_ws, ws_sz);

      std::vector<miopenConvAlgoPerf_t> mi_perfs(requestedAlgoCount);
      miopenStatus_t st = miopenFindConvolutionBackwardWeightsAlgorithm(
          handle, dyDesc, d_dy, xDesc, d_x, convDesc, dwDesc, d_dw,
          requestedAlgoCount, returnedAlgoCount, mi_perfs.data(),
          d_ws, ws_sz, /*exhaustiveSearch=*/false);

      if (st == miopenStatusSuccess) {
          for (int i = 0; i < *returnedAlgoCount && i < requestedAlgoCount; ++i) {
              perfResults[i].algo   = mi_perfs[i].bwd_weights_algo;
              perfResults[i].time   = mi_perfs[i].time;
              perfResults[i].memory = mi_perfs[i].memory;
          }
      }

      if (d_x) cudaFree(d_x);
      if (d_dy) cudaFree(d_dy);
      if (d_dw) cudaFree(d_dw);
      if (d_ws) cudaFree(d_ws);

      return st;
  }

  static inline miopenStatus_t
  cudnnGetConvolutionBackwardDataAlgorithm_v7(
      miopenHandle_t handle,
      miopenTensorDescriptor_t wDesc,
      miopenTensorDescriptor_t dyDesc,
      miopenConvolutionDescriptor_t convDesc,
      miopenTensorDescriptor_t dxDesc,
      int requestedAlgoCount, int* returnedAlgoCount,
      cudnnConvolutionBwdDataAlgoPerf_t* perfResults)
  {
      size_t w_sz=0, dy_sz=0, dx_sz=0, ws_sz=0;
      miopenGetTensorNumBytes(wDesc, &w_sz);
      miopenGetTensorNumBytes(dyDesc, &dy_sz);
      miopenGetTensorNumBytes(dxDesc, &dx_sz);
      miopenConvolutionBackwardDataGetWorkSpaceSize(handle, dyDesc, wDesc, convDesc, dxDesc, &ws_sz);

      void *d_w=nullptr, *d_dy=nullptr, *d_dx=nullptr, *d_ws=nullptr;
      if (w_sz) cudaMalloc(&d_w, w_sz);
      if (dy_sz) cudaMalloc(&d_dy, dy_sz);
      if (dx_sz) cudaMalloc(&d_dx, dx_sz);
      if (ws_sz) cudaMalloc(&d_ws, ws_sz);

      std::vector<miopenConvAlgoPerf_t> mi_perfs(requestedAlgoCount);
      miopenStatus_t st = miopenFindConvolutionBackwardDataAlgorithm(
          handle, dyDesc, d_dy, wDesc, d_w, convDesc, dxDesc, d_dx,
          requestedAlgoCount, returnedAlgoCount, mi_perfs.data(),
          d_ws, ws_sz, /*exhaustiveSearch=*/false);

      if (st == miopenStatusSuccess) {
          for (int i = 0; i < *returnedAlgoCount && i < requestedAlgoCount; ++i) {
              perfResults[i].algo   = mi_perfs[i].bwd_data_algo;
              perfResults[i].time   = mi_perfs[i].time;
              perfResults[i].memory = mi_perfs[i].memory;
          }
      }

      if (d_w) cudaFree(d_w);
      if (d_dy) cudaFree(d_dy);
      if (d_dx) cudaFree(d_dx);
      if (d_ws) cudaFree(d_ws);

      return st;
  }

  // ── Workspace size queries ────────────────────────────────────────────
  static inline miopenStatus_t
  cudnnGetConvolutionBackwardFilterWorkspaceSize(
      miopenHandle_t handle,
      miopenTensorDescriptor_t xDesc,
      miopenTensorDescriptor_t dyDesc,
      miopenConvolutionDescriptor_t convDesc,
      miopenTensorDescriptor_t dwDesc,
      miopenConvBwdWeightsAlgorithm_t /*algo*/,
      size_t* sizeInBytes)
  {
      return miopenConvolutionBackwardWeightsGetWorkSpaceSize(
          handle, dyDesc, xDesc, convDesc, dwDesc, sizeInBytes);
  }

  static inline miopenStatus_t
  cudnnGetConvolutionBackwardDataWorkspaceSize(
      miopenHandle_t handle,
      miopenTensorDescriptor_t wDesc,
      miopenTensorDescriptor_t dyDesc,
      miopenConvolutionDescriptor_t convDesc,
      miopenTensorDescriptor_t dxDesc,
      miopenConvBwdDataAlgorithm_t /*algo*/,
      size_t* sizeInBytes)
  {
      return miopenConvolutionBackwardDataGetWorkSpaceSize(
          handle, dyDesc, wDesc, convDesc, dxDesc, sizeInBytes);
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Execution function wrappers  (parameter order differs)
  // ═══════════════════════════════════════════════════════════════════════

  // cuDNN:  ...algo, ws, wsSize, beta, yDesc, y
  // MIOpen: ...algo, beta, yDesc, y, ws, wsSize
  static inline miopenStatus_t
  cudnnConvolutionForward(
      miopenHandle_t handle, const void* alpha,
      miopenTensorDescriptor_t xDesc, const void* x,
      miopenTensorDescriptor_t wDesc, const void* w,
      miopenConvolutionDescriptor_t convDesc,
      miopenConvFwdAlgorithm_t algo,
      void* workSpace, size_t workSpaceSizeInBytes,
      const void* beta,
      miopenTensorDescriptor_t yDesc, void* y)
  {
      return miopenConvolutionForward(handle, alpha,
          xDesc, x, wDesc, w, convDesc, algo,
          beta, yDesc, y, workSpace, workSpaceSizeInBytes);
  }

  // cuDNN:  ...convDesc, algo, ws, wsSize, beta, filterDesc, dw
  // MIOpen: ...convDesc, algo, beta, dwDesc, dw, ws, wsSize
  static inline miopenStatus_t
  cudnnConvolutionBackwardFilter(
      miopenHandle_t handle, const void* alpha,
      miopenTensorDescriptor_t xDesc, const void* x,
      miopenTensorDescriptor_t dyDesc, const void* dy,
      miopenConvolutionDescriptor_t convDesc,
      miopenConvBwdWeightsAlgorithm_t algo,
      void* workSpace, size_t workSpaceSizeInBytes,
      const void* beta,
      miopenTensorDescriptor_t dwDesc, void* dw)
  {
      return miopenConvolutionBackwardWeights(handle, alpha,
          dyDesc, dy, xDesc, x, convDesc, algo,
          beta, dwDesc, dw, workSpace, workSpaceSizeInBytes);
  }

  // cuDNN:  handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, ws, wsSize, beta, dxDesc, dx
  // MIOpen: handle, alpha, dyDesc, dy, wDesc, w, convDesc, algo, beta, dxDesc, dx, ws, wsSize
  static inline miopenStatus_t
  cudnnConvolutionBackwardData(
      miopenHandle_t handle, const void* alpha,
      miopenTensorDescriptor_t wDesc, const void* w,
      miopenTensorDescriptor_t dyDesc, const void* dy,
      miopenConvolutionDescriptor_t convDesc,
      miopenConvBwdDataAlgorithm_t algo,
      void* workSpace, size_t workSpaceSizeInBytes,
      const void* beta,
      miopenTensorDescriptor_t dxDesc, void* dx)
  {
      return miopenConvolutionBackwardData(handle, alpha,
          dyDesc, dy, wDesc, w, convDesc, algo,
          beta, dxDesc, dx, workSpace, workSpaceSizeInBytes);
  }

  // ── Batch Normalization (signatures match closely but MIOpen lacks const) ──
  static inline miopenStatus_t
  cudnnBatchNormalizationForwardTraining(
      miopenHandle_t handle, miopenBatchNormMode_t mode,
      const void *alpha, const void *beta,
      miopenTensorDescriptor_t xDesc, const void *x,
      miopenTensorDescriptor_t yDesc, void *y,
      miopenTensorDescriptor_t bnScaleBiasMeanVarDesc,
      const void *bnScale, const void *bnBias,
      double exponentialAverageFactor,
      void *resultRunningMean, void *resultRunningVariance,
      double epsilon,
      void *resultSaveMean, void *resultSaveInvVariance)
  {
      return miopenBatchNormalizationForwardTraining(
          handle, mode, (void*)alpha, (void*)beta,
          xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc,
          (void*)bnScale, (void*)bnBias,
          exponentialAverageFactor, resultRunningMean, resultRunningVariance,
          epsilon, resultSaveMean, resultSaveInvVariance);
  }

  static inline miopenStatus_t
  cudnnBatchNormalizationForwardInference(
      miopenHandle_t handle, miopenBatchNormMode_t mode,
      const void *alpha, const void *beta,
      miopenTensorDescriptor_t xDesc, const void *x,
      miopenTensorDescriptor_t yDesc, void *y,
      miopenTensorDescriptor_t bnScaleBiasMeanVarDesc,
      const void *bnScale, const void *bnBias,
      const void *estimatedMean, const void *estimatedVariance,
      double epsilon)
  {
      return miopenBatchNormalizationForwardInference(
          handle, mode, (void*)alpha, (void*)beta,
          xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc,
          (void*)bnScale, (void*)bnBias,
          (void*)estimatedMean, (void*)estimatedVariance, epsilon);
  }

  static inline miopenStatus_t
  cudnnBatchNormalizationBackward(
      miopenHandle_t handle, miopenBatchNormMode_t mode,
      const void *alphaDataDiff, const void *betaDataDiff,
      const void *alphaParamDiff, const void *betaParamDiff,
      miopenTensorDescriptor_t xDesc, const void *x,
      miopenTensorDescriptor_t dyDesc, const void *dy,
      miopenTensorDescriptor_t dxDesc, void *dx,
      miopenTensorDescriptor_t bnScaleBiasDiffDesc,
      const void *bnScale,
      void *resultBnScaleDiff, void *resultBnBiasDiff,
      double epsilon,
      const void *savedMean, const void *savedInvVariance)
  {
      return miopenBatchNormalizationBackward(
          handle, mode,
          (void*)alphaDataDiff, (void*)betaDataDiff,
          (void*)alphaParamDiff, (void*)betaParamDiff,
          xDesc, x, dyDesc, dy, dxDesc, dx,
          bnScaleBiasDiffDesc, (void*)bnScale,
          resultBnScaleDiff, resultBnBiasDiff,
          epsilon, (void*)savedMean, (void*)savedInvVariance);
  }

  // ── Pooling ───────────────────────────────────────────────────────────
  // MIOpen requires a workspace for max pooling backward indices.
  // cuDNN does not pass a workspace pointer, so we manage a thread-local one.
  static thread_local void* g_miopen_pool_ws = nullptr;
  static thread_local size_t g_miopen_pool_ws_size = 0;

  static inline miopenStatus_t
  cudnnPoolingForward(
      miopenHandle_t handle,
      miopenPoolingDescriptor_t poolDesc,
      const void* alpha,
      miopenTensorDescriptor_t xDesc, const void* x,
      const void* beta,
      miopenTensorDescriptor_t yDesc, void* y)
  {
      size_t ws_size = 0;
      miopenPoolingGetWorkSpaceSizeV2(poolDesc, yDesc, &ws_size);
      if (ws_size > g_miopen_pool_ws_size) {
          if (g_miopen_pool_ws) cudaFree(g_miopen_pool_ws);
          cudaMalloc(&g_miopen_pool_ws, ws_size);
          g_miopen_pool_ws_size = ws_size;
      }
      return miopenPoolingForward(handle, poolDesc, alpha,
          xDesc, x, beta, yDesc, y, /*do_backward=*/true,
          g_miopen_pool_ws, ws_size);
  }

  static inline miopenStatus_t
  cudnnPoolingBackward(
      miopenHandle_t handle,
      miopenPoolingDescriptor_t poolDesc,
      const void* alpha,
      miopenTensorDescriptor_t yDesc, const void* y,
      miopenTensorDescriptor_t dyDesc, const void* dy,
      miopenTensorDescriptor_t xDesc, const void* x,
      const void* beta,
      miopenTensorDescriptor_t dxDesc, void* dx)
  {
      return miopenPoolingBackward(handle, poolDesc, alpha,
          yDesc, y, dyDesc, dy, xDesc, x,
          beta, dxDesc, dx, g_miopen_pool_ws);
  }

  // ── Softmax ───────────────────────────────────────────────────────────
  // cuDNN:  handle, algo, mode, alpha, xDesc, x, beta, yDesc, y
  // MIOpen: handle, alpha, xDesc, x, beta, yDesc, y, algo, mode
  static inline miopenStatus_t
  cudnnSoftmaxForward(
      miopenHandle_t handle,
      int algo, int mode,
      const void* alpha,
      miopenTensorDescriptor_t xDesc, const void* x,
      const void* beta,
      miopenTensorDescriptor_t yDesc, void* y)
  {
      return miopenSoftmaxForward_V2(handle, alpha, xDesc, x,
          beta, yDesc, y,
          (miopenSoftmaxAlgorithm_t)algo,
          (miopenSoftmaxMode_t)mode);
  }

#else
  // CUDA path — real cuDNN.
  #include <cudnn.h>
#endif
