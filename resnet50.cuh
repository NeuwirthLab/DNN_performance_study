#pragma once
/*
 * resnet50.cuh  –  ResNet-50  (forward + backward, cuDNN + cuBLAS)
 *
 * Convolution  : cuDNN cudnnConvolutionForward / BackwardFilter / BackwardData
 * Batch Norm   : cuDNN cudnnBatchNormalizationForward* / Backward
 * Max Pooling  : cuDNN cudnnPoolingForward / Backward
 * FC layer     : cuBLAS cublasSgemm
 * Optimizer    : SGD with momentum + weight decay
 * Loss         : softmax cross-entropy (numerically stable)
 *
 * Compile:
 *   nvcc -O2 -arch=sm_75 -std=c++17 main.cu resnet50.cu \
 *        -o resnet50 -lcublas -lcudnn
 */

#include <cublas_v2.h>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#ifdef USE_NCCL
#include <nccl.h>
#endif
#include <vector>
#include <stdexcept>
#include <string>
#include <cfloat>
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
// Error helpers
// ─────────────────────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do { cudaError_t _e=(call);                                                 \
         if(_e!=cudaSuccess) throw std::runtime_error(                          \
             std::string("CUDA[")+__FILE__+":"+std::to_string(__LINE__)+"]: "   \
             +cudaGetErrorString(_e)); } while(0)

#define CUBLAS_CHECK(call)                                                      \
    do { cublasStatus_t _s=(call);                                              \
         if(_s!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error(                \
             std::string("cuBLAS[")+__FILE__+":"+std::to_string(__LINE__)+"]"); \
    } while(0)

#define CUDNN_CHECK(call)                                                       \
    do { cudnnStatus_t _s=(call);                                               \
         if(_s!=CUDNN_STATUS_SUCCESS) throw std::runtime_error(                 \
             std::string("cuDNN[")+__FILE__+":"+std::to_string(__LINE__)+"]: "  \
             +std::string(cudnnGetErrorString(_s))); } while(0)

// ─────────────────────────────────────────────────────────────────────────────
// Utility
// ─────────────────────────────────────────────────────────────────────────────
void gpu_fill_random  (float* d, int n, float scale=0.01f);
void gpu_fill_constant(float* d, int n, float val);

// ─────────────────────────────────────────────────────────────────────────────
// CUDA kernels still implemented manually (not in cuDNN)
// ─────────────────────────────────────────────────────────────────────────────

// ReLU forward / backward
__global__ void relu_kernel    (float* x, int n);
__global__ void relu_bwd_kernel(float* dX, const float* dY,
                                 const float* Y, int n);

// Element-wise utilities
__global__ void add_inplace_kernel(float* dst, const float* src, int n);
__global__ void add_bias_kernel   (float* out, const float* bias, int N, int K);

// Global average pool (custom – cuDNN pooling not used for global avg)
__global__ void gavgpool_kernel    (const float* in, float* out,
                                    int N, int C, int HW);
__global__ void gavgpool_bwd_kernel(float* dX, const float* dY,
                                    int N, int C, int HW);

// Loss / accuracy
__global__ void sce_bwd_kernel (float* dlogit, const float* prob,
                                 const int* labels, int N, int K);
__global__ void ce_loss_kernel (float* loss,    const float* prob,
                                 const int* labels, int N, int K);
__global__ void top1_kernel    (int*   correct, const float* prob,
                                 const int* labels, int N, int K);

// Optimiser
__global__ void sgd_kernel (float* param, float* vel, const float* grad,
                             float lr, float mom, float wd, int n);
__global__ void zero_kernel(float* x, int n);

// ─────────────────────────────────────────────────────────────────────────────
// ConvBnLayer  –  Conv2d + BatchNorm + optional ReLU  (cuDNN-backed)
// ─────────────────────────────────────────────────────────────────────────────
class ConvBnLayer {
    friend class BottleneckBlock;
    friend class ResNet50;
public:
    ConvBnLayer(cublasHandle_t cublas,
                cudnnHandle_t  cudnn,
                int in_c, int out_c,
                int kH,   int kW,
                int padH, int padW,
                int strH, int strW,
                bool relu = true);
    ~ConvBnLayer();

    // Forward  (training or inference based on training_ flag)
    const float* Forward(const float* d_in, int N, int H, int W,
                         int& out_H, int& out_W);

    // Backward – returns gradient w.r.t. input [N, in_c, H, W]
    const float* Backward(const float* dY, int N, int H, int W);

    void InitWeights(float scale = 0.01f);
    void ZeroGrad();
    void UpdateSGD(float lr, float mom, float wd);
    void SetTraining(bool t) { training_ = t; }

    int OutC() const { return out_c_; }
    int InC()  const { return in_c_;  }
    int KH()   const { return kH_;    }
    int KW()   const { return kW_;    }

    // Exposed for ResNet50 grad-norm / SGD step / checkpoint
    float* Weight()   { return d_weight_;   }
    float* BnScale()  { return d_bn_scale_; }
    float* BnBias()   { return d_bn_bias_;  }
    float* BnMean()   { return d_bn_mean_;  }
    float* BnVar()    { return d_bn_var_;   }
    float* VelW()     { return d_vel_w_;    }
    float* VelBnS()   { return d_vel_bns_;  }
    float* VelBnB()   { return d_vel_bnb_;  }
    float* dWeight()  { return d_dweight_;  }
    float* dBnScale() { return d_dbn_scale_; }
    float* dBnBias()  { return d_dbn_bias_;  }
    int    WeightN()  { return out_c_ * in_c_ * kH_ * kW_; }

private:
    cublasHandle_t cublas_;
    cudnnHandle_t  cudnn_;
    int in_c_, out_c_, kH_, kW_, padH_, padW_, strH_, strW_;
    bool relu_, training_{true};

    // ── cuDNN descriptors ─────────────────────────────────────────────────
    cudnnTensorDescriptor_t     in_desc_, out_desc_, bn_desc_;
    cudnnFilterDescriptor_t     filter_desc_;
    cudnnConvolutionDescriptor_t conv_desc_;

    // Algorithm selection (re-run when N / H / W changes)
    bool algos_valid_{false};
    cudnnConvolutionFwdAlgo_t       fwd_algo_;
    cudnnConvolutionBwdFilterAlgo_t bwd_f_algo_;
    cudnnConvolutionBwdDataAlgo_t   bwd_d_algo_;

    // Shared workspace (max of fwd / bwdf / bwdd requirements)
    float  *d_ws_{nullptr};
    size_t  ws_bytes_{0};

    // ── Parameters ────────────────────────────────────────────────────────
    float *d_weight_;                              // [out_c, in_c, kH, kW]
    float *d_bn_scale_, *d_bn_bias_;               // [out_c]
    float *d_bn_mean_,  *d_bn_var_;                // running stats [out_c]

    // BN statistics saved by forward training  (needed by BN backward)
    float *d_saved_mean_{nullptr};                 // [out_c]
    float *d_saved_inv_var_{nullptr};              // [out_c]

    // ── Forward buffers ───────────────────────────────────────────────────
    float *d_conv_out_{nullptr};   // conv output  = BN input [N,out_c,oH,oW]
    float *d_out_{nullptr};        // BN + ReLU output        [N,out_c,oH,oW]
    size_t conv_out_bytes_{0}, out_bytes_{0};

    // Saved pointer to caller's buffer (for conv bwd filter)
    const float* d_saved_in_{nullptr};

    // ── Backward buffers ──────────────────────────────────────────────────
    float *d_dconv_out_{nullptr};  // BN-bwd output           [N,out_c,oH,oW]
    float *d_din_{nullptr};        // conv-bwd-data output    [N,in_c,H,W]
    size_t dconv_out_bytes_{0}, din_bytes_{0};

    // ── Gradient buffers ──────────────────────────────────────────────────
    float *d_dweight_{nullptr};                    // [out_c, in_c, kH, kW]
    float *d_dbn_scale_{nullptr};                  // [out_c]
    float *d_dbn_bias_{nullptr};                   // [out_c]

    // ── SGD velocities ────────────────────────────────────────────────────
    float *d_vel_w_{nullptr}, *d_vel_bns_{nullptr}, *d_vel_bnb_{nullptr};

    // Cached dims (invalidate algo selection when they change)
    int cur_N_{0}, cur_H_{0}, cur_W_{0};

    // ── Helpers ───────────────────────────────────────────────────────────
    static int conv_out(int in, int pad, int k, int str) {
        return (in + 2*pad - k) / str + 1;
    }
    void refresh_descriptors_and_algos(int N, int H, int W,
                                        int out_H, int out_W);
    void ensure_fwd_buffers(int N, int out_H, int out_W);
    void ensure_bwd_buffers(int N, int H, int W, int out_H, int out_W);
};

// ─────────────────────────────────────────────────────────────────────────────
// BottleneckBlock
// ─────────────────────────────────────────────────────────────────────────────
class BottleneckBlock {
    friend class ResNet50;
public:
    BottleneckBlock(cublasHandle_t cublas,
                    cudnnHandle_t  cudnn,
                    int in_c, int bot_c, int out_c, int stride = 1);
    ~BottleneckBlock();

    const float* Forward(const float* d_in, int N, int H, int W,
                         int& out_H, int& out_W);
    const float* Backward(const float* dY, int N, int H, int W);

    void InitWeights(float scale = 0.01f);
    void ZeroGrad();
    void UpdateSGD(float lr, float mom, float wd);
    void SetTraining(bool t);

    // Accessors for checkpoint helpers
    ConvBnLayer* Conv1() { return conv1_; }
    ConvBnLayer* Conv2() { return conv2_; }
    ConvBnLayer* Conv3() { return conv3_; }
    ConvBnLayer* Proj()  { return proj_;  }
    bool HasProj() const { return proj_ != nullptr; }

private:
    cublasHandle_t cublas_;
    cudnnHandle_t  cudnn_;
    int in_c_, out_c_, stride_;
    bool need_proj_;

    ConvBnLayer *conv1_, *conv2_, *conv3_, *proj_;

    float *d_out_{nullptr};
    size_t out_bytes_{0};

    float *d_din_{nullptr};
    size_t din_bytes_{0};

    int saved_H_{0}, saved_W_{0};
    int out_H_{0}, out_W_{0};

    void add_relu(float* dst, const float* main, const float* sc,
                  int N, int C, int H, int W);
};

// ─────────────────────────────────────────────────────────────────────────────
// Profiling data
// ─────────────────────────────────────────────────────────────────────────────
struct LayerProfile {
    std::string name;
    float       time_ms;
    double      flops;
};

// ─────────────────────────────────────────────────────────────────────────────
// ResNet50
// ─────────────────────────────────────────────────────────────────────────────
class ResNet50 {
public:
    explicit ResNet50(int num_classes = 1000);
    ~ResNet50();

    // Forward → softmax probabilities [N, num_classes]
    const float* Forward(const float* d_input, int N, int H = 224, int W = 224);

    // Forward with per-stage profiling
    const float* ForwardProfile(const float* d_input, int N,
                                std::vector<LayerProfile>& profile,
                                int H = 224, int W = 224);

    // Backward (cross-entropy)
    void Backward(const int* d_labels, int N);

    void ZeroGrad();
    void UpdateSGD(float lr, float momentum = 0.9f, float weight_decay = 1e-4f);
    void SetTraining(bool t);

    float Loss       (const int* d_labels, int N);
    float Top1Accuracy(const int* d_labels, int N);
    void  InitWeights();

    size_t ParamCount() const;
    float  GradNorm();
    static double ResNet50FLOPS(int N, int num_classes);

    bool SaveWeights(const std::string& path) const;
    bool LoadWeights(const std::string& path);

#ifdef USE_NCCL
    // Multi-GPU: average all gradient buffers across devices using NCCL AllReduce
    void SyncGradients(ncclComm_t comm, cudaStream_t stream);
#endif

    cublasHandle_t Cublas() const { return cublas_; }
    cudnnHandle_t  Cudnn()  const { return cudnn_;  }

private:
    cublasHandle_t cublas_;
    cudnnHandle_t  cudnn_;
    int  num_classes_;
    bool training_{true};

    // ── cuDNN MaxPool ─────────────────────────────────────────────────────
    cudnnPoolingDescriptor_t maxpool_desc_;
    cudnnTensorDescriptor_t  mp_in_desc_, mp_out_desc_;
    int mp_prev_N_{0}, mp_prev_inH_{0}, mp_prev_inW_{0};

    // ── Stage 0 ───────────────────────────────────────────────────────────
    ConvBnLayer* conv1_;

    int inp_H_{224}, inp_W_{224};
    int conv1_oH_{0}, conv1_oW_{0};
    int mp_oH_{0},    mp_oW_{0};
    int final_HW_{0};

    // MaxPool input/output  (saved for cuDNN pooling backward)
    float *d_conv1_out_{nullptr};   // maxpool input   (= copy of conv1 output)
    float *d_mp_out_{nullptr};      // maxpool output
    size_t conv1_out_bytes_{0}, mp_out_bytes_{0};

    // ── Residual layers ───────────────────────────────────────────────────
    std::vector<BottleneckBlock*> layer1_, layer2_, layer3_, layer4_;

    // ── Head ──────────────────────────────────────────────────────────────
    float *d_ap_out_{nullptr};      // global avgpool [N, 2048]
    size_t ap_out_bytes_{0};

    float *d_fc_w_{nullptr};        // [num_classes, 2048]
    float *d_fc_b_{nullptr};        // [num_classes]
    float *d_fc_out_{nullptr};      // [N, num_classes]  (also holds softmax probs)
    size_t fc_out_bytes_{0};

    float *d_dfc_w_{nullptr};
    float *d_dfc_b_{nullptr};
    float *d_dap_{nullptr};
    size_t dap_bytes_{0};

    float *d_vel_fcw_{nullptr}, *d_vel_fcb_{nullptr};

    // Backward intermediates
    float *d_dl4_{nullptr};         // avgpool_bwd → layer4  [N,2048,7,7]
    size_t dl4_bytes_{0};
    float *d_dconv1_{nullptr};      // maxpool_bwd → conv1   [N,64,112,112]
    size_t dconv1_bytes_{0};

    float *d_loss_buf_{nullptr};
    int   *d_correct_buf_{nullptr};

    int cur_N_{0};

    // ── Private helpers ───────────────────────────────────────────────────
    void maxpool_fwd(const float* d_in, int N, int inH, int inW);
    void maxpool_bwd(const float* d_dout, int N, int inH, int inW);
    void avgpool_fwd(const float* d_in, int N, int HW);
    void avgpool_bwd(const float* d_dap_out, int N, int HW);
    void fc_fwd(int N);
    void fc_bwd(const float* d_dlogit, int N);
    void softmax_inplace(int N);
};
