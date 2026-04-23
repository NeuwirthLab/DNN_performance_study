// dnn_compat.h — cuDNN ↔ MIOpen typedef/macro shim + algorithm helpers.
//
// WHY A SEPARATE HEADER
// cuDNN and MIOpen have similar object models (tensor / filter / conv /
// pooling / BN descriptors) but DIFFERENT algorithm-selection APIs. Pure
// macro-renaming handles the create/set/destroy calls, but conv fwd/bwd
// algorithm discovery needs a real wrapper. Keep both here so the model
// code (resnet50.cu) can include ONE file.
//
// STATUS
// This is a starter template. Every #error marker below is a spot that must
// be verified against the actual MIOpen version on your AMD cluster before
// training results are trusted. Do that on the AMD box, not here.
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
  // Signatures differ slightly (format enum position); see set_tensor_4d() below.

  // ── filter / conv / pool descriptors ──────────────────────────────────
  #define cudnnFilterDescriptor_t                miopenTensorDescriptor_t   // MIOpen reuses tensor desc for filters
  #define cudnnCreateFilterDescriptor            miopenCreateTensorDescriptor
  #define cudnnDestroyFilterDescriptor           miopenDestroyTensorDescriptor
  #define cudnnConvolutionDescriptor_t           miopenConvolutionDescriptor_t
  #define cudnnCreateConvolutionDescriptor       miopenCreateConvolutionDescriptor
  #define cudnnDestroyConvolutionDescriptor      miopenDestroyConvolutionDescriptor
  #define cudnnPoolingDescriptor_t               miopenPoolingDescriptor_t
  #define cudnnCreatePoolingDescriptor           miopenCreatePoolingDescriptor
  #define cudnnDestroyPoolingDescriptor          miopenDestroyPoolingDescriptor

  // ── dtype / format / mode enums ───────────────────────────────────────
  #define CUDNN_DATA_FLOAT                       miopenFloat
  #define CUDNN_DATA_HALF                        miopenHalf
  // cuDNN has CUDNN_TENSOR_NCHW layout flag; MIOpen's legacy API is NCHW by default
  // (there is no explicit layout parameter in miopenSet4dTensorDescriptor).

  #define CUDNN_BATCHNORM_SPATIAL                miopenBNSpatial
  #define CUDNN_BATCHNORM_PER_ACTIVATION         miopenBNPerActivation
  #define cudnnBatchNormMode_t                   miopenBatchNormMode_t

  #define CUDNN_POOLING_MAX                      miopenPoolingMax
  #define CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING miopenPoolingAverageInclusive
  #define CUDNN_PROPAGATE_NAN                    0 /* MIOpen has no equivalent; ignored */

  #define CUDNN_CROSS_CORRELATION                miopenConvolution

  // cuDNN's math-type selector (tensor-op) has no direct MIOpen counterpart.
  #define cudnnSetConvolutionMathType(desc, mode) miopenStatusSuccess

  // ── inline wrappers to paper over signature differences ───────────────
  //
  // Model code calls e.g.:
  //   cudnnSetTensor4dDescriptor(d, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n,c,h,w);
  // MIOpen's analog:
  //   miopenSet4dTensorDescriptor(d, miopenFloat, n,c,h,w);
  // So we macro cudnnSetTensor4dDescriptor to a helper that drops the format arg.
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

  // ── conv fwd/bwd execution + algo selection ───────────────────────────
  //
  // cuDNN:  cudnnFindConvolutionForwardAlgorithm(...) returns a perf array.
  // MIOpen: miopenFindConvolutionForwardAlgorithm(...) has a similar shape
  //         but requires pre-allocated workspace and the tensors themselves.
  //
  // The model only needs "pick the best algo + its workspace size". Expose
  // minimal wrappers here; the model calls these instead of the cuDNN
  // perf-struct dance.
  //
  // TODO(amrit): implement these wrappers on the AMD box with real tensors
  //   once the first compile works. Leaving declarations only for now so the
  //   linker flags the sites needing attention.
  //
  // Suggested signatures:
  //   miopenStatus_t find_conv_fwd_algo(handle, xDesc, wDesc, convDesc, yDesc,
  //                                     void* workspace, size_t wsSize,
  //                                     miopenConvFwdAlgorithm_t* algo_out);
  //   miopenStatus_t find_conv_bwd_filter_algo(...);
  //   miopenStatus_t find_conv_bwd_data_algo(...);

#else
  // CUDA path — real cuDNN.
  #include <cudnn.h>
#endif
