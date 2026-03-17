/*
 * resnet50.cu  –  ResNet-50 forward + backward
 *                 Convolution / BatchNorm / MaxPool → cuDNN
 *                 FC layer                          → cuBLAS
 *                 ReLU / AvgPool / Loss / SGD       → custom CUDA kernels
 *
 * Compile:
 *   nvcc -O2 -arch=sm_75 -std=c++17 main.cu resnet50.cu \
 *        -o resnet50 -lcublas -lcudnn
 */

#include "resnet50.cuh"
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

// ─────────────────────────────────────────────────────────────────────────────
// Utility
// ─────────────────────────────────────────────────────────────────────────────

void gpu_fill_random(float* d, int n, float scale)
{
    std::vector<float> h(n);
    for (int i = 0; i < n; ++i)
        h[i] = scale * (2.f * (float)rand() / RAND_MAX - 1.f);
    CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
}

void gpu_fill_constant(float* d, int n, float val)
{
    std::vector<float> h(n, val);
    CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
}

// ─────────────────────────────────────────────────────────────────────────────
// Custom CUDA kernels  (ReLU, add, avgpool, loss, sgd – not in cuDNN)
// ─────────────────────────────────────────────────────────────────────────────

/* ReLU in-place */
__global__ void relu_kernel(float* __restrict__ x, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] = fmaxf(0.f, x[idx]);
}

/* ReLU backward:  dX[i] = dY[i] * (Y[i] > 0) */
__global__ void relu_bwd_kernel(float* __restrict__ dX,
                                 const float* __restrict__ dY,
                                 const float* __restrict__ Y, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dX[idx] = dY[idx] * (Y[idx] > 0.f ? 1.f : 0.f);
}

/* Element-wise add  dst += src */
__global__ void add_inplace_kernel(float* __restrict__ dst,
                                    const float* __restrict__ src, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] += src[idx];
}

/* Broadcast-add bias: out[n*K + k] += bias[k] */
__global__ void add_bias_kernel(float* __restrict__ out,
                                 const float* __restrict__ bias,
                                 int N, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * K) out[idx] += bias[idx % K];
}

/* Global average pool:  in[N,C,HW] → out[N,C] */
__global__ void gavgpool_kernel(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int N, int C, int HW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C) return;
    const float* p = in + idx * HW;
    float s = 0.f;
    for (int i = 0; i < HW; ++i) s += p[i];
    out[idx] = s / HW;
}

/* Global avgpool backward: dX[n,c,hw] = dY[n,c] / HW */
__global__ void gavgpool_bwd_kernel(float* __restrict__ dX,
                                     const float* __restrict__ dY,
                                     int N, int C, int HW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * HW) return;
    int nc = idx / HW;
    dX[idx] = dY[nc] / (float)HW;
}

/* Row-wise softmax: x[N,K] in-place. One block per row. */
__global__ void softmax_kernel(float* __restrict__ x, int N, int K)
{
    int n = blockIdx.x;
    if (n >= N) return;
    float* row = x + n * K;
    float mx = row[0];
    for (int k = 1; k < K; ++k) mx = fmaxf(mx, row[k]);
    float s = 0.f;
    for (int k = 0; k < K; ++k) { row[k] = expf(row[k] - mx); s += row[k]; }
    float inv = 1.f / s;
    for (int k = 0; k < K; ++k) row[k] *= inv;
}

/* Fused softmax + cross-entropy backward */
__global__ void sce_bwd_kernel(float* __restrict__ dlogit,
                                const float* __restrict__ prob,
                                const int* __restrict__ labels,
                                int N, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;
    int n = idx / K, k = idx % K;
    dlogit[idx] = (prob[idx] - (k == labels[n] ? 1.f : 0.f)) / (float)N;
}

/* Per-sample cross-entropy loss */
__global__ void ce_loss_kernel(float* __restrict__ loss,
                                const float* __restrict__ prob,
                                const int* __restrict__ labels,
                                int N, int K)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    loss[n] = -logf(fmaxf(prob[n * K + labels[n]], 1e-12f));
}

/* Top-1 correct count */
__global__ void top1_kernel(int* __restrict__ correct,
                             const float* __restrict__ prob,
                             const int* __restrict__ labels,
                             int N, int K)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float* row = prob + n * K;
    int best = 0; float best_v = row[0];
    for (int k = 1; k < K; ++k)
        if (row[k] > best_v) { best_v = row[k]; best = k; }
    correct[n] = (best == labels[n]) ? 1 : 0;
}

/* SGD with momentum and weight decay */
__global__ void sgd_kernel(float* __restrict__ param,
                            float* __restrict__ vel,
                            const float* __restrict__ grad,
                            float lr, float mom, float wd, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = mom * vel[idx] + grad[idx] + wd * param[idx];
    vel[idx]   = v;
    param[idx] -= lr * v;
}

/* Zero a buffer */
__global__ void zero_kernel(float* __restrict__ x, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] = 0.f;
}

/* FC bias gradient: dbias[k] += sum_n dY[n*K + k] */
static __global__ void bias_grad_kernel(float* __restrict__ dbias,
                                         const float* __restrict__ dY,
                                         int N, int K)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    float s = 0.f;
    for (int n = 0; n < N; ++n) s += dY[n * K + k];
    dbias[k] += s;
}

// ─────────────────────────────────────────────────────────────────────────────
// Inline dispatch helpers
// ─────────────────────────────────────────────────────────────────────────────

static void launch_relu(float* d, int n)
{
    int blk = 256, grd = (n + blk - 1) / blk;
    relu_kernel<<<grd, blk>>>(d, n);
    CUDA_CHECK(cudaGetLastError());
}

// launch_zero kept for potential external use
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4505) // unreferenced local function
#endif
static void launch_zero(float* d, int n)
{
    int blk = 256, grd = (n + blk - 1) / blk;
    zero_kernel<<<grd, blk>>>(d, n);
    CUDA_CHECK(cudaGetLastError());
}

// ─────────────────────────────────────────────────────────────────────────────
// ConvBnLayer  (cuDNN-backed: conv + BN, custom kernel: ReLU)
// ─────────────────────────────────────────────────────────────────────────────

ConvBnLayer::ConvBnLayer(cublasHandle_t cublas,
                         cudnnHandle_t  cudnn,
                         int in_c, int out_c,
                         int kH, int kW,
                         int padH, int padW,
                         int strH, int strW,
                         bool relu)
    : cublas_(cublas), cudnn_(cudnn),
      in_c_(in_c), out_c_(out_c),
      kH_(kH), kW_(kW),
      padH_(padH), padW_(padW),
      strH_(strH), strW_(strW),
      relu_(relu)
{
    int K = in_c_ * kH_ * kW_;

    // ── Parameters ────────────────────────────────────────────────────────
    CUDA_CHECK(cudaMalloc(&d_weight_,   (size_t)out_c_ * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bn_scale_, out_c_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bn_bias_,  out_c_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bn_mean_,  out_c_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bn_var_,   out_c_ * sizeof(float)));

    // BN saved statistics (fixed size, allocated once)
    CUDA_CHECK(cudaMalloc(&d_saved_mean_,    out_c_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_saved_inv_var_, out_c_ * sizeof(float)));

    // ── Gradient buffers ──────────────────────────────────────────────────
    CUDA_CHECK(cudaMalloc(&d_dweight_,   (size_t)out_c_ * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dbn_scale_, out_c_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dbn_bias_,  out_c_ * sizeof(float)));

    // ── SGD velocities ────────────────────────────────────────────────────
    CUDA_CHECK(cudaMalloc(&d_vel_w_,   (size_t)out_c_ * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vel_bns_, out_c_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vel_bnb_, out_c_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_vel_w_,   0, (size_t)out_c_ * K * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_vel_bns_, 0, out_c_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_vel_bnb_, 0, out_c_ * sizeof(float)));

    // ── cuDNN descriptors ─────────────────────────────────────────────────
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc_));

    // BN scale/bias/mean/var descriptor: [1, out_c, 1, 1] (spatial BN)
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bn_desc_));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(bn_desc_,
                    CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                    1, out_c_, 1, 1));

    // Filter descriptor: [out_c, in_c, kH, kW]
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc_));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc_,
                    CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                    out_c_, in_c_, kH_, kW_));

    // Convolution descriptor: pad, stride, dilation=1, cross-correlation
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc_,
                    padH_, padW_, strH_, strW_,
                    1, 1,                         // dilation
                    CUDNN_CROSS_CORRELATION,
                    CUDNN_DATA_FLOAT));
    // Enable tensor-core math if available
    CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc_,
                    CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
}

ConvBnLayer::~ConvBnLayer()
{
    // Parameters & BN stats
    cudaFree(d_weight_);
    cudaFree(d_bn_scale_); cudaFree(d_bn_bias_);
    cudaFree(d_bn_mean_);  cudaFree(d_bn_var_);
    cudaFree(d_saved_mean_); cudaFree(d_saved_inv_var_);

    // Gradient & velocity buffers
    cudaFree(d_dweight_); cudaFree(d_dbn_scale_); cudaFree(d_dbn_bias_);
    cudaFree(d_vel_w_);   cudaFree(d_vel_bns_);   cudaFree(d_vel_bnb_);

    // Forward / backward working buffers
    if (d_conv_out_)  cudaFree(d_conv_out_);
    if (d_out_)       cudaFree(d_out_);
    if (d_dconv_out_) cudaFree(d_dconv_out_);
    if (d_din_)       cudaFree(d_din_);
    if (d_ws_)        cudaFree(d_ws_);

    // cuDNN descriptors
    cudnnDestroyTensorDescriptor(in_desc_);
    cudnnDestroyTensorDescriptor(out_desc_);
    cudnnDestroyTensorDescriptor(bn_desc_);
    cudnnDestroyFilterDescriptor(filter_desc_);
    cudnnDestroyConvolutionDescriptor(conv_desc_);
}

void ConvBnLayer::InitWeights(float /*scale*/)
{
    int K    = in_c_ * kH_ * kW_;
    float he = sqrtf(2.f / (float)(in_c_ * kH_ * kW_));
    gpu_fill_random  (d_weight_,  out_c_ * K, he);
    gpu_fill_constant(d_bn_scale_, out_c_, 1.f);
    gpu_fill_constant(d_bn_bias_,  out_c_, 0.f);
    gpu_fill_constant(d_bn_mean_,  out_c_, 0.f);
    gpu_fill_constant(d_bn_var_,   out_c_, 1.f);
}

// ── Buffer allocation helpers ──────────────────────────────────────────────

void ConvBnLayer::ensure_fwd_buffers(int N, int out_H, int out_W)
{
    size_t need = (size_t)N * out_c_ * out_H * out_W * sizeof(float);
    if (need > conv_out_bytes_) {
        if (d_conv_out_) CUDA_CHECK(cudaFree(d_conv_out_));
        CUDA_CHECK(cudaMalloc(&d_conv_out_, need));
        conv_out_bytes_ = need;
    }
    if (need > out_bytes_) {
        if (d_out_) CUDA_CHECK(cudaFree(d_out_));
        CUDA_CHECK(cudaMalloc(&d_out_, need));
        out_bytes_ = need;
    }
}

void ConvBnLayer::ensure_bwd_buffers(int N, int H, int W,
                                      int out_H, int out_W)
{
    size_t dconv_need = (size_t)N * out_c_ * out_H * out_W * sizeof(float);
    if (dconv_need > dconv_out_bytes_) {
        if (d_dconv_out_) CUDA_CHECK(cudaFree(d_dconv_out_));
        CUDA_CHECK(cudaMalloc(&d_dconv_out_, dconv_need));
        dconv_out_bytes_ = dconv_need;
    }
    size_t din_need = (size_t)N * in_c_ * H * W * sizeof(float);
    if (din_need > din_bytes_) {
        if (d_din_) CUDA_CHECK(cudaFree(d_din_));
        CUDA_CHECK(cudaMalloc(&d_din_, din_need));
        din_bytes_ = din_need;
    }
}

// ── Algorithm selection (called when N/H/W changes) ───────────────────────

void ConvBnLayer::refresh_descriptors_and_algos(int N, int H, int W,
                                                  int out_H, int out_W)
{
    // Update spatial-dependent tensor descriptors
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(in_desc_,
                    CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                    N, in_c_, H, W));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(out_desc_,
                    CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                    N, out_c_, out_H, out_W));

    // --- Forward algorithm ---
    int returned = 0;
    cudnnConvolutionFwdAlgoPerf_t fwd_perf;
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
        cudnn_, in_desc_, filter_desc_, conv_desc_, out_desc_,
        1, &returned, &fwd_perf));
    fwd_algo_ = fwd_perf.algo;

    // --- Backward filter algorithm ---
    cudnnConvolutionBwdFilterAlgoPerf_t bwdf_perf;
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        cudnn_, in_desc_, out_desc_, conv_desc_, filter_desc_,
        1, &returned, &bwdf_perf));
    bwd_f_algo_ = bwdf_perf.algo;

    // --- Backward data algorithm ---
    cudnnConvolutionBwdDataAlgoPerf_t bwdd_perf;
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        cudnn_, filter_desc_, out_desc_, conv_desc_, in_desc_,
        1, &returned, &bwdd_perf));
    bwd_d_algo_ = bwdd_perf.algo;

    // --- Workspace: max of all three ---
    size_t fwd_ws = 0, bwdf_ws = 0, bwdd_ws = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn_, in_desc_, filter_desc_, conv_desc_, out_desc_,
        fwd_algo_, &fwd_ws));
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnn_, in_desc_, out_desc_, conv_desc_, filter_desc_,
        bwd_f_algo_, &bwdf_ws));
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn_, filter_desc_, out_desc_, conv_desc_, in_desc_,
        bwd_d_algo_, &bwdd_ws));

    size_t ws_need = std::max({fwd_ws, bwdf_ws, bwdd_ws});
    if (ws_need > ws_bytes_) {
        if (d_ws_) CUDA_CHECK(cudaFree(d_ws_));
        if (ws_need > 0)
            CUDA_CHECK(cudaMalloc(&d_ws_, ws_need));
        else
            d_ws_ = nullptr;
        ws_bytes_ = ws_need;
    }

    algos_valid_ = true;
}

// ── Forward ────────────────────────────────────────────────────────────────

const float* ConvBnLayer::Forward(const float* d_in,
                                   int N, int H, int W,
                                   int& out_H, int& out_W)
{
    out_H = conv_out(H, padH_, kH_, strH_);
    out_W = conv_out(W, padW_, kW_, strW_);

    // Re-select algorithms and resize buffers if input dims changed
    if (N != cur_N_ || H != cur_H_ || W != cur_W_) {
        refresh_descriptors_and_algos(N, H, W, out_H, out_W);
        cur_N_ = N; cur_H_ = H; cur_W_ = W;
    }

    ensure_fwd_buffers(N, out_H, out_W);

    // Save pointer to caller's input for conv backward-filter
    if (training_) d_saved_in_ = d_in;

    const float one = 1.f, zero = 0.f;

    // ── 1. Convolution (cuDNN) ────────────────────────────────────────────
    CUDNN_CHECK(cudnnConvolutionForward(
        cudnn_, &one,
        in_desc_,     d_in,
        filter_desc_, d_weight_,
        conv_desc_,   fwd_algo_,
        d_ws_,        ws_bytes_,
        &zero,
        out_desc_,    d_conv_out_));

    // ── 2. Batch Normalization (cuDNN) ────────────────────────────────────
    if (training_) {
        CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
            cudnn_,
            CUDNN_BATCHNORM_SPATIAL,
            &one, &zero,
            out_desc_, d_conv_out_,   // x  (conv output = BN input)
            out_desc_, d_out_,        // y  (BN output)
            bn_desc_,  d_bn_scale_, d_bn_bias_,
            0.1,          // exponentialAverageFactor (momentum for running stats)
            d_bn_mean_,   // running mean  (updated in-place)
            d_bn_var_,    // running var   (updated in-place)
            1e-5,         // epsilon
            d_saved_mean_,    // savedMean    (for backward)
            d_saved_inv_var_  // savedInvVar  (for backward)
        ));
    } else {
        CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
            cudnn_,
            CUDNN_BATCHNORM_SPATIAL,
            &one, &zero,
            out_desc_, d_conv_out_,   // x
            out_desc_, d_out_,        // y
            bn_desc_,  d_bn_scale_, d_bn_bias_,
            d_bn_mean_, d_bn_var_,    // estimated mean / var
            1e-5
        ));
    }

    // ── 3. ReLU (custom kernel, in-place on d_out_) ────────────────────
    if (relu_) launch_relu(d_out_, N * out_c_ * out_H * out_W);

    return d_out_;
}

// ── Backward ───────────────────────────────────────────────────────────────

const float* ConvBnLayer::Backward(const float* dY, int N, int H, int W)
{
    int out_H = conv_out(H, padH_, kH_, strH_);
    int out_W = conv_out(W, padW_, kW_, strW_);
    int total = N * out_c_ * out_H * out_W;

    ensure_bwd_buffers(N, H, W, out_H, out_W);

    const float one = 1.f, zero = 0.f;

    // ── 1. ReLU backward (in-place) ───────────────────────────────────────
    //   d_out_[i] currently holds the ReLU output from forward.
    //   After: d_out_[i] = dY[i] * (d_out_[i] > 0)   → gradient after ReLU
    if (relu_) {
        int grd = (total + 255) / 256;
        relu_bwd_kernel<<<grd, 256>>>(d_out_, dY, d_out_, total);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // No ReLU: copy incoming gradient so BN-bwd always reads from d_out_
        CUDA_CHECK(cudaMemcpy(d_out_, dY,
                              (size_t)total * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    }

    // ── 2. BatchNorm backward (cuDNN) ─────────────────────────────────────
    //   x   = d_conv_out_   (conv output saved during forward)
    //   dy  = d_out_        (gradient after ReLU backward)
    //   dx  = d_dconv_out_  (gradient w.r.t. conv output)
    //   betaParamDiff = 1 → accumulate into d_dbn_scale_ / d_dbn_bias_
    CUDNN_CHECK(cudnnBatchNormalizationBackward(
        cudnn_,
        CUDNN_BATCHNORM_SPATIAL,
        &one, &zero,    // alphaDataDiff, betaDataDiff
        &one, &one,     // alphaParamDiff, betaParamDiff (accumulate)
        out_desc_, d_conv_out_,   // x  (BN input from forward)
        out_desc_, d_out_,        // dy (gradient after ReLU bwd)
        out_desc_, d_dconv_out_,  // dx (gradient w.r.t. BN input)
        bn_desc_,  d_bn_scale_,   // scale (gamma)
        d_dbn_scale_, d_dbn_bias_,// dscale, dbias  (accumulated)
        1e-5,
        d_saved_mean_,            // savedMean from forward
        d_saved_inv_var_          // savedInvVar from forward
    ));

    // ── 3. Conv backward filter (accumulate dW) ───────────────────────────
    //   beta = 1 → accumulate (ZeroGrad zeros it before each backward)
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(
        cudnn_, &one,
        in_desc_,     d_saved_in_,   // x (original conv input)
        out_desc_,    d_dconv_out_,  // dy (gradient w.r.t. conv output)
        conv_desc_,   bwd_f_algo_,
        d_ws_,        ws_bytes_,
        &one,                        // beta = 1  → accumulate
        filter_desc_, d_dweight_
    ));

    // ── 4. Conv backward data (compute dX) ───────────────────────────────
    CUDNN_CHECK(cudnnConvolutionBackwardData(
        cudnn_, &one,
        filter_desc_, d_weight_,
        out_desc_,    d_dconv_out_,  // dy
        conv_desc_,   bwd_d_algo_,
        d_ws_,        ws_bytes_,
        &zero,
        in_desc_,     d_din_         // dx
    ));

    return d_din_;
}

void ConvBnLayer::ZeroGrad()
{
    int K = in_c_ * kH_ * kW_;
    CUDA_CHECK(cudaMemset(d_dweight_,   0, (size_t)out_c_ * K * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dbn_scale_, 0, out_c_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dbn_bias_,  0, out_c_ * sizeof(float)));
}

void ConvBnLayer::UpdateSGD(float lr, float mom, float wd)
{
    int blk = 256;
    int K   = in_c_ * kH_ * kW_;
    int wn  = out_c_ * K;
    {
        int grd = (wn + blk - 1) / blk;
        sgd_kernel<<<grd, blk>>>(d_weight_, d_vel_w_, d_dweight_,
                                  lr, mom, wd, wn);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        int grd = (out_c_ + blk - 1) / blk;
        sgd_kernel<<<grd, blk>>>(d_bn_scale_, d_vel_bns_, d_dbn_scale_,
                                  lr, mom, 0.f, out_c_);
        CUDA_CHECK(cudaGetLastError());
        sgd_kernel<<<grd, blk>>>(d_bn_bias_,  d_vel_bnb_, d_dbn_bias_,
                                  lr, mom, 0.f, out_c_);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BottleneckBlock
// ─────────────────────────────────────────────────────────────────────────────

BottleneckBlock::BottleneckBlock(cublasHandle_t cublas,
                                  cudnnHandle_t  cudnn,
                                  int in_c, int bot_c, int out_c,
                                  int stride)
    : cublas_(cublas), cudnn_(cudnn),
      in_c_(in_c), out_c_(out_c), stride_(stride)
{
    need_proj_ = (in_c != out_c) || (stride > 1);

    conv1_ = new ConvBnLayer(cublas, cudnn, in_c,  bot_c, 1,1, 0,0, 1,     1,     true);
    conv2_ = new ConvBnLayer(cublas, cudnn, bot_c, bot_c, 3,3, 1,1, stride,stride, true);
    conv3_ = new ConvBnLayer(cublas, cudnn, bot_c, out_c, 1,1, 0,0, 1,     1,     false);
    proj_  = need_proj_
             ? new ConvBnLayer(cublas, cudnn, in_c, out_c, 1,1, 0,0, stride,stride, false)
             : nullptr;
}

BottleneckBlock::~BottleneckBlock()
{
    delete conv1_; delete conv2_; delete conv3_; delete proj_;
    if (d_out_) cudaFree(d_out_);
    if (d_din_) cudaFree(d_din_);
}

void BottleneckBlock::InitWeights(float scale)
{
    conv1_->InitWeights(scale); conv2_->InitWeights(scale);
    conv3_->InitWeights(scale);
    if (proj_) proj_->InitWeights(scale);
}

void BottleneckBlock::add_relu(float* dst,
                                const float* main_br,
                                const float* sc,
                                int N, int C, int H, int W)
{
    size_t n = (size_t)N * C * H * W;
    CUDA_CHECK(cudaMemcpy(dst, main_br, n * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    int blk = 256, grd = ((int)n + blk - 1) / blk;
    add_inplace_kernel<<<grd, blk>>>(dst, sc, (int)n);
    CUDA_CHECK(cudaGetLastError());
    launch_relu(dst, (int)n);
}

const float* BottleneckBlock::Forward(const float* d_in,
                                       int N, int H, int W,
                                       int& out_H, int& out_W)
{
    saved_H_ = H; saved_W_ = W;

    int h1, w1, h2, w2, h3, w3;
    const float* x1 = conv1_->Forward(d_in, N, H,  W,  h1, w1);
    const float* x2 = conv2_->Forward(x1,   N, h1, w1, h2, w2);
    const float* x3 = conv3_->Forward(x2,   N, h2, w2, h3, w3);

    int hs, ws;
    const float* sc = need_proj_
                      ? proj_->Forward(d_in, N, H, W, hs, ws)
                      : d_in;

    out_H = h3; out_W = w3;
    out_H_ = h3; out_W_ = w3;

    size_t need = (size_t)N * out_c_ * out_H * out_W * sizeof(float);
    if (need > out_bytes_) {
        if (d_out_) CUDA_CHECK(cudaFree(d_out_));
        CUDA_CHECK(cudaMalloc(&d_out_, need));
        out_bytes_ = need;
    }

    add_relu(d_out_, x3, sc, N, out_c_, out_H, out_W);
    return d_out_;
}

const float* BottleneckBlock::Backward(const float* dY, int N, int H, int W)
{
    int total_out = N * out_c_ * H * W;
    int blk = 256;

    // Step 1: ReLU backward on the residual add
    {
        int grd = (total_out + blk - 1) / blk;
        relu_bwd_kernel<<<grd, blk>>>(d_out_, dY, d_out_, total_out);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 2: Main branch backward (conv3 → conv2 → conv1)
    const float* dx2   = conv3_->Backward(d_out_, N,
                                            conv3_->cur_H_, conv3_->cur_W_);
    const float* dx1   = conv2_->Backward(dx2,    N,
                                            conv2_->cur_H_, conv2_->cur_W_);
    const float* d_main = conv1_->Backward(dx1,   N,
                                            saved_H_, saved_W_);

    // Step 3: Shortcut backward
    const float* d_sc;
    if (need_proj_) {
        d_sc = proj_->Backward(d_out_, N, saved_H_, saved_W_);
    } else {
        d_sc = d_out_;   // identity shortcut: gradient passes through
    }

    // Step 4: Accumulate main + shortcut gradients into d_din_
    int total_in = N * in_c_ * saved_H_ * saved_W_;
    size_t din_need = (size_t)total_in * sizeof(float);
    if (din_need > din_bytes_) {
        if (d_din_) CUDA_CHECK(cudaFree(d_din_));
        CUDA_CHECK(cudaMalloc(&d_din_, din_need));
        din_bytes_ = din_need;
    }
    CUDA_CHECK(cudaMemcpy(d_din_, d_main, din_need, cudaMemcpyDeviceToDevice));
    {
        int grd = (total_in + blk - 1) / blk;
        add_inplace_kernel<<<grd, blk>>>(d_din_, d_sc, total_in);
        CUDA_CHECK(cudaGetLastError());
    }

    return d_din_;
}

void BottleneckBlock::ZeroGrad()
{
    conv1_->ZeroGrad(); conv2_->ZeroGrad(); conv3_->ZeroGrad();
    if (proj_) proj_->ZeroGrad();
}

void BottleneckBlock::UpdateSGD(float lr, float mom, float wd)
{
    conv1_->UpdateSGD(lr, mom, wd);
    conv2_->UpdateSGD(lr, mom, wd);
    conv3_->UpdateSGD(lr, mom, wd);
    if (proj_) proj_->UpdateSGD(lr, mom, wd);
}

void BottleneckBlock::SetTraining(bool t)
{
    conv1_->SetTraining(t); conv2_->SetTraining(t);
    conv3_->SetTraining(t);
    if (proj_) proj_->SetTraining(t);
}

// ─────────────────────────────────────────────────────────────────────────────
// ResNet50
// ─────────────────────────────────────────────────────────────────────────────

ResNet50::ResNet50(int num_classes) : num_classes_(num_classes)
{
    CUBLAS_CHECK(cublasCreate(&cublas_));
    CUDNN_CHECK(cudnnCreate(&cudnn_));

    // ── cuDNN MaxPool descriptor ──────────────────────────────────────────
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&maxpool_desc_));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(maxpool_desc_,
        CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
        3, 3,    // window H, W
        1, 1,    // padding H, W
        2, 2));  // stride H, W

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&mp_in_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&mp_out_desc_));

    // ── Stage 0: 7×7 conv, 3→64, stride=2, pad=3 ─────────────────────────
    conv1_ = new ConvBnLayer(cublas_, cudnn_, 3, 64, 7, 7, 3, 3, 2, 2, true);

    // ── Layer 1 (3 bottlenecks, no stride) ────────────────────────────────
    layer1_.push_back(new BottleneckBlock(cublas_, cudnn_,  64,  64, 256, 1));
    layer1_.push_back(new BottleneckBlock(cublas_, cudnn_, 256,  64, 256, 1));
    layer1_.push_back(new BottleneckBlock(cublas_, cudnn_, 256,  64, 256, 1));

    // ── Layer 2 (4 bottlenecks, stride=2 in first) ────────────────────────
    layer2_.push_back(new BottleneckBlock(cublas_, cudnn_, 256, 128, 512, 2));
    layer2_.push_back(new BottleneckBlock(cublas_, cudnn_, 512, 128, 512, 1));
    layer2_.push_back(new BottleneckBlock(cublas_, cudnn_, 512, 128, 512, 1));
    layer2_.push_back(new BottleneckBlock(cublas_, cudnn_, 512, 128, 512, 1));

    // ── Layer 3 (6 bottlenecks, stride=2 in first) ────────────────────────
    layer3_.push_back(new BottleneckBlock(cublas_, cudnn_,  512, 256, 1024, 2));
    layer3_.push_back(new BottleneckBlock(cublas_, cudnn_, 1024, 256, 1024, 1));
    layer3_.push_back(new BottleneckBlock(cublas_, cudnn_, 1024, 256, 1024, 1));
    layer3_.push_back(new BottleneckBlock(cublas_, cudnn_, 1024, 256, 1024, 1));
    layer3_.push_back(new BottleneckBlock(cublas_, cudnn_, 1024, 256, 1024, 1));
    layer3_.push_back(new BottleneckBlock(cublas_, cudnn_, 1024, 256, 1024, 1));

    // ── Layer 4 (3 bottlenecks, stride=2 in first) ────────────────────────
    layer4_.push_back(new BottleneckBlock(cublas_, cudnn_, 1024, 512, 2048, 2));
    layer4_.push_back(new BottleneckBlock(cublas_, cudnn_, 2048, 512, 2048, 1));
    layer4_.push_back(new BottleneckBlock(cublas_, cudnn_, 2048, 512, 2048, 1));

    // ── FC head ───────────────────────────────────────────────────────────
    CUDA_CHECK(cudaMalloc(&d_fc_w_, (size_t)num_classes_ * 2048 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc_b_, (size_t)num_classes_         * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dfc_w_, (size_t)num_classes_ * 2048 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dfc_b_, (size_t)num_classes_         * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vel_fcw_, (size_t)num_classes_ * 2048 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vel_fcb_, (size_t)num_classes_         * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_vel_fcw_, 0, (size_t)num_classes_ * 2048 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_vel_fcb_, 0, (size_t)num_classes_         * sizeof(float)));
}

ResNet50::~ResNet50()
{
    delete conv1_;
    for (auto* b : layer1_) delete b;
    for (auto* b : layer2_) delete b;
    for (auto* b : layer3_) delete b;
    for (auto* b : layer4_) delete b;

    if (d_conv1_out_) cudaFree(d_conv1_out_);
    if (d_mp_out_)    cudaFree(d_mp_out_);
    if (d_ap_out_)    cudaFree(d_ap_out_);
    if (d_fc_out_)    cudaFree(d_fc_out_);
    if (d_dap_)       cudaFree(d_dap_);
    if (d_dl4_)       cudaFree(d_dl4_);
    if (d_dconv1_)    cudaFree(d_dconv1_);
    if (d_loss_buf_)    cudaFree(d_loss_buf_);
    if (d_correct_buf_) cudaFree(d_correct_buf_);

    cudaFree(d_fc_w_); cudaFree(d_fc_b_);
    cudaFree(d_dfc_w_); cudaFree(d_dfc_b_);
    cudaFree(d_vel_fcw_); cudaFree(d_vel_fcb_);

    cudnnDestroyPoolingDescriptor(maxpool_desc_);
    cudnnDestroyTensorDescriptor(mp_in_desc_);
    cudnnDestroyTensorDescriptor(mp_out_desc_);

    cudnnDestroy(cudnn_);
    cublasDestroy(cublas_);
}

void ResNet50::InitWeights()
{
    conv1_->InitWeights();
    for (auto* b : layer1_) b->InitWeights();
    for (auto* b : layer2_) b->InitWeights();
    for (auto* b : layer3_) b->InitWeights();
    for (auto* b : layer4_) b->InitWeights();
    float he = sqrtf(2.f / 2048.f);
    gpu_fill_random  (d_fc_w_, num_classes_ * 2048, he);
    gpu_fill_constant(d_fc_b_, num_classes_, 0.f);
}

// ── MaxPool (cuDNN) : [N,64,112,112] → [N,64,56,56] ─────────────────────────

void ResNet50::maxpool_fwd(const float* d_in, int N, int inH, int inW)
{
    const int C = 64;
    int outH = (inH + 2*1 - 3) / 2 + 1;
    int outW = (inW + 2*1 - 3) / 2 + 1;
    mp_oH_ = outH; mp_oW_ = outW;

    size_t need = (size_t)N * C * outH * outW * sizeof(float);
    if (need > mp_out_bytes_) {
        if (d_mp_out_) CUDA_CHECK(cudaFree(d_mp_out_));
        CUDA_CHECK(cudaMalloc(&d_mp_out_, need));
        mp_out_bytes_ = need;
    }

    // Update tensor descriptors when batch/spatial dims change
    if (N != mp_prev_N_ || inH != mp_prev_inH_ || inW != mp_prev_inW_) {
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(mp_in_desc_,
            CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, inH, inW));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(mp_out_desc_,
            CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, outH, outW));
        mp_prev_N_ = N; mp_prev_inH_ = inH; mp_prev_inW_ = inW;
    }

    const float alpha = 1.f, beta = 0.f;
    CUDNN_CHECK(cudnnPoolingForward(cudnn_, maxpool_desc_,
        &alpha, mp_in_desc_,  d_in,
        &beta,  mp_out_desc_, d_mp_out_));
}

void ResNet50::maxpool_bwd(const float* d_dout, int N, int inH, int inW)
{
    const int C = 64;
    size_t need = (size_t)N * C * inH * inW * sizeof(float);
    if (need > dconv1_bytes_) {
        if (d_dconv1_) CUDA_CHECK(cudaFree(d_dconv1_));
        CUDA_CHECK(cudaMalloc(&d_dconv1_, need));
        dconv1_bytes_ = need;
    }

    // mp_in_desc_ / mp_out_desc_ still valid from forward
    const float alpha = 1.f, beta = 0.f;
    CUDNN_CHECK(cudnnPoolingBackward(cudnn_, maxpool_desc_,
        &alpha,
        mp_out_desc_, d_mp_out_,    // y   (maxpool output, saved)
        mp_out_desc_, d_dout,       // dy  (gradient w.r.t. output)
        mp_in_desc_,  d_conv1_out_, // x   (maxpool input, saved)
        &beta,
        mp_in_desc_,  d_dconv1_));  // dx  (gradient w.r.t. input)
}

// ── Global average pool (custom kernel) ──────────────────────────────────────

void ResNet50::avgpool_fwd(const float* d_in, int N, int HW)
{
    const int C = 2048;
    final_HW_ = HW;
    size_t need = (size_t)N * C * sizeof(float);
    if (need > ap_out_bytes_) {
        if (d_ap_out_) CUDA_CHECK(cudaFree(d_ap_out_));
        CUDA_CHECK(cudaMalloc(&d_ap_out_, need));
        ap_out_bytes_ = need;
    }
    int blk = 256, grd = (N * C + blk - 1) / blk;
    gavgpool_kernel<<<grd, blk>>>(d_in, d_ap_out_, N, C, HW);
    CUDA_CHECK(cudaGetLastError());
}

void ResNet50::avgpool_bwd(const float* d_dap_out, int N, int HW)
{
    const int C = 2048;
    size_t need = (size_t)N * C * HW * sizeof(float);
    if (need > dl4_bytes_) {
        if (d_dl4_) CUDA_CHECK(cudaFree(d_dl4_));
        CUDA_CHECK(cudaMalloc(&d_dl4_, need));
        dl4_bytes_ = need;
    }
    int total = N * C * HW;
    int blk = 256, grd = (total + blk - 1) / blk;
    gavgpool_bwd_kernel<<<grd, blk>>>(d_dl4_, d_dap_out, N, C, HW);
    CUDA_CHECK(cudaGetLastError());
}

// ── FC: d_ap_out_[N,2048] × W^T → d_fc_out_[N,K] ────────────────────────────

void ResNet50::fc_fwd(int N)
{
    const int M = 2048, K = num_classes_;
    size_t need = (size_t)N * K * sizeof(float);
    if (need > fc_out_bytes_) {
        if (d_fc_out_) CUDA_CHECK(cudaFree(d_fc_out_));
        CUDA_CHECK(cudaMalloc(&d_fc_out_, need));
        fc_out_bytes_ = need;
    }
    const float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasSgemm(cublas_,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              K, N, M,
                              &alpha,
                              d_fc_w_,   M,
                              d_ap_out_, M,
                              &beta,
                              d_fc_out_, K));
    int blk = 256, grd = (N * K + blk - 1) / blk;
    add_bias_kernel<<<grd, blk>>>(d_fc_out_, d_fc_b_, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void ResNet50::fc_bwd(const float* d_dlogit, int N)
{
    const int M = 2048, K = num_classes_;
    const float one = 1.f, zero_f = 0.f;

    size_t dap_need = (size_t)N * M * sizeof(float);
    if (dap_need > dap_bytes_) {
        if (d_dap_) CUDA_CHECK(cudaFree(d_dap_));
        CUDA_CHECK(cudaMalloc(&d_dap_, dap_need));
        dap_bytes_ = dap_need;
    }
    CUBLAS_CHECK(cublasSgemm(cublas_,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              M, N, K,
                              &one,
                              d_fc_w_,    M,
                              d_dlogit,   K,
                              &zero_f,
                              d_dap_,     M));

    CUBLAS_CHECK(cublasSgemm(cublas_,
                              CUBLAS_OP_N, CUBLAS_OP_T,
                              M, K, N,
                              &one,
                              d_ap_out_,  M,
                              d_dlogit,   K,
                              &one,
                              d_dfc_w_,   M));

    int blk = 256, grd = (K + blk - 1) / blk;
    bias_grad_kernel<<<grd, blk>>>(d_dfc_b_, d_dlogit, N, K);
    CUDA_CHECK(cudaGetLastError());
}

// ── Softmax ───────────────────────────────────────────────────────────────────

void ResNet50::softmax_inplace(int N)
{
    softmax_kernel<<<N, 1>>>(d_fc_out_, N, num_classes_);
    CUDA_CHECK(cudaGetLastError());
}

// ── Full forward pass ─────────────────────────────────────────────────────────

const float* ResNet50::Forward(const float* d_input, int N, int H, int W)
{
    nvtxRangePushA("ResNet50_Forward");
    inp_H_ = H; inp_W_ = W;

    // Stage 0: Conv1
    int h, w;
    const float* x = conv1_->Forward(d_input, N, H, W, h, w);
    conv1_oH_ = h; conv1_oW_ = w;

    // Save conv1 output for cuDNN maxpool backward (needs original input)
    if (training_) {
        size_t need = (size_t)N * 64 * h * w * sizeof(float);
        if (need > conv1_out_bytes_) {
            if (d_conv1_out_) CUDA_CHECK(cudaFree(d_conv1_out_));
            CUDA_CHECK(cudaMalloc(&d_conv1_out_, need));
            conv1_out_bytes_ = need;
        }
        CUDA_CHECK(cudaMemcpy(d_conv1_out_, x, need, cudaMemcpyDeviceToDevice));
    }

    // MaxPool
    maxpool_fwd(x, N, h, w);
    x = d_mp_out_;
    h = mp_oH_; w = mp_oW_;

    // Residual stages
    for (auto* blk : layer1_) x = blk->Forward(x, N, h, w, h, w);
    for (auto* blk : layer2_) x = blk->Forward(x, N, h, w, h, w);
    for (auto* blk : layer3_) x = blk->Forward(x, N, h, w, h, w);
    for (auto* blk : layer4_) x = blk->Forward(x, N, h, w, h, w);

    // Head
    avgpool_fwd(x, N, h * w);
    fc_fwd(N);
    softmax_inplace(N);

    cur_N_ = N;
    nvtxRangePop();
    return d_fc_out_;
}

// ── Profiled forward pass ─────────────────────────────────────────────────────

static double conv_flops(int N, int in_c, int out_c, int kH, int kW,
                          int out_H, int out_W)
{
    return (double)N * out_c * out_H * out_W * in_c * kH * kW * 2.0;
}
static double bn_flops(int N, int C, int HW)  { return (double)N*C*HW*4.0; }
static double fc_flops(int N, int in_f, int out_f) { return (double)N*in_f*out_f*2.0; }

static double bottleneck_flops(int N, int in_c, int bot_c, int out_c,
                                int stride, int H, int W)
{
    int h2 = (H + 2 - 3) / stride + 1, w2 = (W + 2 - 3) / stride + 1;
    double f = 0.0;
    f += conv_flops(N, in_c,  bot_c, 1, 1, H,  W);  f += bn_flops(N, bot_c, H*W);
    f += conv_flops(N, bot_c, bot_c, 3, 3, h2, w2); f += bn_flops(N, bot_c, h2*w2);
    f += conv_flops(N, bot_c, out_c, 1, 1, h2, w2); f += bn_flops(N, out_c, h2*w2);
    if (in_c != out_c || stride > 1) {
        int ph = H/stride, pw = W/stride;
        f += conv_flops(N, in_c, out_c, 1, 1, ph, pw);
        f += bn_flops(N, out_c, ph*pw);
    }
    f += (double)N * out_c * h2 * w2 * 2.0;
    return f;
}

const float* ResNet50::ForwardProfile(const float* d_input, int N,
                                       std::vector<LayerProfile>& profile,
                                       int H, int W)
{
    inp_H_ = H; inp_W_ = W;
    profile.clear();
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    auto time_stage = [&](const char* name, double flops, auto&& fn) {
        CUDA_CHECK(cudaEventRecord(t0));
        fn();
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        profile.push_back({name, ms, flops});
    };

    int h, w;
    const float* x = nullptr;
    time_stage("Conv1+BN (7x7/2)",
        conv_flops(N,3,64,7,7,(H+6-7)/2+1,(W+6-7)/2+1) +
        bn_flops(N,64,((H+6-7)/2+1)*((W+6-7)/2+1)),
        [&]{ x = conv1_->Forward(d_input, N, H, W, h, w); });
    conv1_oH_ = h; conv1_oW_ = w;

    if (training_) {
        size_t need = (size_t)N * 64 * h * w * sizeof(float);
        if (need > conv1_out_bytes_) {
            if (d_conv1_out_) CUDA_CHECK(cudaFree(d_conv1_out_));
            CUDA_CHECK(cudaMalloc(&d_conv1_out_, need));
            conv1_out_bytes_ = need;
        }
        CUDA_CHECK(cudaMemcpy(d_conv1_out_, x, need, cudaMemcpyDeviceToDevice));
    }

    int mpOutH = (h + 2*1 - 3)/2+1, mpOutW = (w + 2*1 - 3)/2+1;
    time_stage("MaxPool (3x3/2) [cuDNN]",
        (double)N * 64 * mpOutH * mpOutW * 9.0,
        [&]{ maxpool_fwd(x, N, h, w); x = d_mp_out_; });
    h = mp_oH_; w = mp_oW_;

    for (int i = 0; i < (int)layer1_.size(); ++i) {
        char name[32]; snprintf(name, sizeof(name), "Layer1.%d", i);
        int in_c = (i == 0) ? 64 : 256;
        double fl = bottleneck_flops(N, in_c, 64, 256, 1, h, w);
        time_stage(name, fl, [&]{ x = layer1_[i]->Forward(x, N, h, w, h, w); });
    }
    for (int i = 0; i < (int)layer2_.size(); ++i) {
        char name[32]; snprintf(name, sizeof(name), "Layer2.%d", i);
        int in_c = (i == 0) ? 256 : 512, stride = (i==0)?2:1;
        double fl = bottleneck_flops(N, in_c, 128, 512, stride, h, w);
        time_stage(name, fl, [&]{ x = layer2_[i]->Forward(x, N, h, w, h, w); });
    }
    for (int i = 0; i < (int)layer3_.size(); ++i) {
        char name[32]; snprintf(name, sizeof(name), "Layer3.%d", i);
        int in_c = (i == 0) ? 512 : 1024, stride = (i==0)?2:1;
        double fl = bottleneck_flops(N, in_c, 256, 1024, stride, h, w);
        time_stage(name, fl, [&]{ x = layer3_[i]->Forward(x, N, h, w, h, w); });
    }
    for (int i = 0; i < (int)layer4_.size(); ++i) {
        char name[32]; snprintf(name, sizeof(name), "Layer4.%d", i);
        int in_c = (i == 0) ? 1024 : 2048, stride = (i==0)?2:1;
        double fl = bottleneck_flops(N, in_c, 512, 2048, stride, h, w);
        time_stage(name, fl, [&]{ x = layer4_[i]->Forward(x, N, h, w, h, w); });
    }

    time_stage("GlobalAvgPool", (double)N*2048*h*w,
        [&]{ avgpool_fwd(x, N, h*w); });
    time_stage("FC",            fc_flops(N, 2048, num_classes_),
        [&]{ fc_fwd(N); });
    time_stage("Softmax",       (double)N*num_classes_*5.0,
        [&]{ softmax_inplace(N); });

    cur_N_ = N;
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return d_fc_out_;
}

// ── Full backward pass ────────────────────────────────────────────────────────

void ResNet50::Backward(const int* d_labels, int N)
{
    nvtxRangePushA("ResNet50_Backward");
    int blk = 256, K = num_classes_;

    // Step 1: Softmax + CE backward → dlogit
    float* d_dlogit = nullptr;
    size_t dl_need = (size_t)N * K * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_dlogit, dl_need));
    {
        int grd = (N * K + blk - 1) / blk;
        sce_bwd_kernel<<<grd, blk>>>(d_dlogit, d_fc_out_, d_labels, N, K);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 2: FC backward
    fc_bwd(d_dlogit, N);
    CUDA_CHECK(cudaFree(d_dlogit));

    // Step 3: AvgPool backward
    avgpool_bwd(d_dap_, N, final_HW_);

    // Step 4: Residual layers backward (reverse)
    const float* grad = d_dl4_;
    for (int i = (int)layer4_.size()-1; i >= 0; --i)
        grad = layer4_[i]->Backward(grad, N,
                                     layer4_[i]->out_H_, layer4_[i]->out_W_);
    for (int i = (int)layer3_.size()-1; i >= 0; --i)
        grad = layer3_[i]->Backward(grad, N,
                                     layer3_[i]->out_H_, layer3_[i]->out_W_);
    for (int i = (int)layer2_.size()-1; i >= 0; --i)
        grad = layer2_[i]->Backward(grad, N,
                                     layer2_[i]->out_H_, layer2_[i]->out_W_);
    for (int i = (int)layer1_.size()-1; i >= 0; --i)
        grad = layer1_[i]->Backward(grad, N,
                                     layer1_[i]->out_H_, layer1_[i]->out_W_);

    // Step 5: MaxPool backward (cuDNN)
    maxpool_bwd(grad, N, conv1_oH_, conv1_oW_);

    // Step 6: Conv1 backward (accumulates weight grads; input grad discarded)
    conv1_->Backward(d_dconv1_, N, inp_H_, inp_W_);
    nvtxRangePop();
}

// ── ZeroGrad ──────────────────────────────────────────────────────────────────

void ResNet50::ZeroGrad()
{
    conv1_->ZeroGrad();
    for (auto* b : layer1_) b->ZeroGrad();
    for (auto* b : layer2_) b->ZeroGrad();
    for (auto* b : layer3_) b->ZeroGrad();
    for (auto* b : layer4_) b->ZeroGrad();
    CUDA_CHECK(cudaMemset(d_dfc_w_, 0, (size_t)num_classes_ * 2048 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dfc_b_, 0, (size_t)num_classes_         * sizeof(float)));
}

// ── UpdateSGD ─────────────────────────────────────────────────────────────────

void ResNet50::UpdateSGD(float lr, float momentum, float weight_decay)
{
    nvtxRangePushA("ResNet50_UpdateSGD");
    conv1_->UpdateSGD(lr, momentum, weight_decay);
    for (auto* b : layer1_) b->UpdateSGD(lr, momentum, weight_decay);
    for (auto* b : layer2_) b->UpdateSGD(lr, momentum, weight_decay);
    for (auto* b : layer3_) b->UpdateSGD(lr, momentum, weight_decay);
    for (auto* b : layer4_) b->UpdateSGD(lr, momentum, weight_decay);
    int blk = 256, wn = num_classes_ * 2048;
    {
        int grd = (wn + blk-1)/blk;
        sgd_kernel<<<grd,blk>>>(d_fc_w_, d_vel_fcw_, d_dfc_w_,
                                  lr, momentum, weight_decay, wn);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        int grd = (num_classes_ + blk-1)/blk;
        sgd_kernel<<<grd,blk>>>(d_fc_b_, d_vel_fcb_, d_dfc_b_,
                                  lr, momentum, 0.f, num_classes_);
        CUDA_CHECK(cudaGetLastError());
    }
    nvtxRangePop();
}

// ── SetTraining ───────────────────────────────────────────────────────────────

void ResNet50::SetTraining(bool t)
{
    training_ = t;
    conv1_->SetTraining(t);
    for (auto* b : layer1_) b->SetTraining(t);
    for (auto* b : layer2_) b->SetTraining(t);
    for (auto* b : layer3_) b->SetTraining(t);
    for (auto* b : layer4_) b->SetTraining(t);
}

// ── Loss / Accuracy ───────────────────────────────────────────────────────────

float ResNet50::Loss(const int* d_labels, int N)
{
    if (!d_loss_buf_) CUDA_CHECK(cudaMalloc(&d_loss_buf_, N * sizeof(float)));
    int blk = 256, grd = (N + blk-1)/blk;
    ce_loss_kernel<<<grd,blk>>>(d_loss_buf_, d_fc_out_, d_labels, N, num_classes_);
    CUDA_CHECK(cudaGetLastError());
    std::vector<float> h(N);
    CUDA_CHECK(cudaMemcpy(h.data(), d_loss_buf_, N*sizeof(float), cudaMemcpyDeviceToHost));
    float sum = 0.f;
    for (int i = 0; i < N; ++i) sum += h[i];
    return sum / N;
}

float ResNet50::Top1Accuracy(const int* d_labels, int N)
{
    if (!d_correct_buf_) CUDA_CHECK(cudaMalloc(&d_correct_buf_, N * sizeof(int)));
    int blk = 256, grd = (N + blk-1)/blk;
    top1_kernel<<<grd,blk>>>(d_correct_buf_, d_fc_out_, d_labels, N, num_classes_);
    CUDA_CHECK(cudaGetLastError());
    std::vector<int> h(N);
    CUDA_CHECK(cudaMemcpy(h.data(), d_correct_buf_, N*sizeof(int), cudaMemcpyDeviceToHost));
    int sum = 0;
    for (int i = 0; i < N; ++i) sum += h[i];
    return (float)sum / (float)N;
}

// ── Param count ───────────────────────────────────────────────────────────────

static size_t conv_bn_params(int in_c, int out_c, int kH, int kW)
{
    return (size_t)out_c * in_c * kH * kW + out_c * 2;
}
static size_t bottleneck_params(int in_c, int bot_c, int out_c, bool proj)
{
    size_t p = conv_bn_params(in_c,  bot_c, 1, 1)
             + conv_bn_params(bot_c, bot_c, 3, 3)
             + conv_bn_params(bot_c, out_c, 1, 1);
    if (proj) p += conv_bn_params(in_c, out_c, 1, 1);
    return p;
}

size_t ResNet50::ParamCount() const
{
    size_t p = conv_bn_params(3, 64, 7, 7);
    p += bottleneck_params(64,  64,  256, true);
    p += bottleneck_params(256, 64,  256, false) * 2;
    p += bottleneck_params(256, 128, 512, true);
    p += bottleneck_params(512, 128, 512, false) * 3;
    p += bottleneck_params(512,  256, 1024, true);
    p += bottleneck_params(1024, 256, 1024, false) * 5;
    p += bottleneck_params(1024, 512, 2048, true);
    p += bottleneck_params(2048, 512, 2048, false) * 2;
    p += (size_t)num_classes_ * 2048 + num_classes_;
    return p;
}

// ── GradNorm ──────────────────────────────────────────────────────────────────

float ResNet50::GradNorm()
{
    struct GradBuf { float* ptr; int n; };
    std::vector<GradBuf> grads;

    auto add_conv = [&](ConvBnLayer* c) {
        grads.push_back({c->dWeight(),  c->WeightN()});
        grads.push_back({c->dBnScale(), c->OutC()});
        grads.push_back({c->dBnBias(),  c->OutC()});
    };
    auto add_block = [&](BottleneckBlock* b) {
        add_conv(b->conv1_); add_conv(b->conv2_); add_conv(b->conv3_);
        if (b->proj_) add_conv(b->proj_);
    };

    add_conv(conv1_);
    for (auto* b : layer1_) add_block(b);
    for (auto* b : layer2_) add_block(b);
    for (auto* b : layer3_) add_block(b);
    for (auto* b : layer4_) add_block(b);
    grads.push_back({d_dfc_w_, num_classes_ * 2048});
    grads.push_back({d_dfc_b_, num_classes_});

    double total_sq = 0.0;
    for (auto& g : grads) {
        float nrm = 0.f;
        CUBLAS_CHECK(cublasSnrm2(cublas_, g.n, g.ptr, 1, &nrm));
        total_sq += (double)nrm * nrm;
    }
    return sqrtf((float)total_sq);
}

// ── FLOPS ─────────────────────────────────────────────────────────────────────

double ResNet50::ResNet50FLOPS(int N, int num_classes)
{
    double f = conv_flops(N, 3, 64, 7, 7, 112, 112) + bn_flops(N, 64, 112*112);
    f += (double)N * 64 * 56 * 56 * 9.0;   // MaxPool
    f += bottleneck_flops(N,  64,  64, 256, 1, 56, 56);
    f += bottleneck_flops(N, 256,  64, 256, 1, 56, 56) * 2;
    f += bottleneck_flops(N, 256, 128, 512, 2, 56, 56);
    f += bottleneck_flops(N, 512, 128, 512, 1, 28, 28) * 3;
    f += bottleneck_flops(N,  512, 256, 1024, 2, 28, 28);
    f += bottleneck_flops(N, 1024, 256, 1024, 1, 14, 14) * 5;
    f += bottleneck_flops(N, 1024, 512, 2048, 2, 14, 14);
    f += bottleneck_flops(N, 2048, 512, 2048, 1,  7,  7) * 2;
    f += (double)N * 2048 * 49;             // AvgPool
    f += fc_flops(N, 2048, num_classes);
    f += (double)N * num_classes * 5.0;     // Softmax
    return f;
}

// ─────────────────────────────────────────────────────────────────────────────
// Checkpoint helpers (binary format – no LibTorch dependency)
//
// Format: [magic(4)] [num_classes(4)] [tensor_data...]
// Each ConvBnLayer writes: weight, bn_scale, bn_bias, bn_mean, bn_var,
//                           vel_w, vel_bns, vel_bnb
// ─────────────────────────────────────────────────────────────────────────────

#include <fstream>

static const uint32_t CKPT_MAGIC = 0x524E3530; // "RN50"

static bool write_gpu_to_file(std::ofstream& f, const float* d_ptr, size_t n)
{
    std::vector<float> buf(n);
    CUDA_CHECK(cudaMemcpy(buf.data(), d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
    f.write(reinterpret_cast<const char*>(buf.data()), n * sizeof(float));
    return f.good();
}

static bool read_file_to_gpu(std::ifstream& f, float* d_ptr, size_t n)
{
    std::vector<float> buf(n);
    f.read(reinterpret_cast<char*>(buf.data()), n * sizeof(float));
    if (!f.good()) return false;
    CUDA_CHECK(cudaMemcpy(d_ptr, buf.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    return true;
}

static bool save_conv(std::ofstream& f, ConvBnLayer* c)
{
    size_t wn = (size_t)c->OutC() * c->InC() * c->KH() * c->KW();
    size_t bn = c->OutC();
    bool ok = true;
    ok &= write_gpu_to_file(f, c->Weight(),  wn);
    ok &= write_gpu_to_file(f, c->BnScale(), bn);
    ok &= write_gpu_to_file(f, c->BnBias(),  bn);
    ok &= write_gpu_to_file(f, c->BnMean(),  bn);
    ok &= write_gpu_to_file(f, c->BnVar(),   bn);
    ok &= write_gpu_to_file(f, c->VelW(),    wn);
    ok &= write_gpu_to_file(f, c->VelBnS(),  bn);
    ok &= write_gpu_to_file(f, c->VelBnB(),  bn);
    return ok;
}

static bool load_conv(std::ifstream& f, ConvBnLayer* c)
{
    size_t wn = (size_t)c->OutC() * c->InC() * c->KH() * c->KW();
    size_t bn = c->OutC();
    bool ok = true;
    ok &= read_file_to_gpu(f, c->Weight(),  wn);
    ok &= read_file_to_gpu(f, c->BnScale(), bn);
    ok &= read_file_to_gpu(f, c->BnBias(),  bn);
    ok &= read_file_to_gpu(f, c->BnMean(),  bn);
    ok &= read_file_to_gpu(f, c->BnVar(),   bn);
    ok &= read_file_to_gpu(f, c->VelW(),    wn);
    ok &= read_file_to_gpu(f, c->VelBnS(),  bn);
    ok &= read_file_to_gpu(f, c->VelBnB(),  bn);
    return ok;
}

static bool save_block(std::ofstream& f, BottleneckBlock* b)
{
    bool ok = true;
    ok &= save_conv(f, b->Conv1());
    ok &= save_conv(f, b->Conv2());
    ok &= save_conv(f, b->Conv3());
    if (b->HasProj()) ok &= save_conv(f, b->Proj());
    return ok;
}

static bool load_block(std::ifstream& f, BottleneckBlock* b)
{
    bool ok = true;
    ok &= load_conv(f, b->Conv1());
    ok &= load_conv(f, b->Conv2());
    ok &= load_conv(f, b->Conv3());
    if (b->HasProj()) ok &= load_conv(f, b->Proj());
    return ok;
}

bool ResNet50::SaveWeights(const std::string& path) const
{
    try {
        std::ofstream f(path, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "[Checkpoint] Cannot open for writing: " << path << "\n";
            return false;
        }

        // Header
        f.write(reinterpret_cast<const char*>(&CKPT_MAGIC), 4);
        int32_t nc = num_classes_;
        f.write(reinterpret_cast<const char*>(&nc), 4);

        // Layers
        bool ok = save_conv(f, conv1_);
        for (auto* b : layer1_) ok &= save_block(f, b);
        for (auto* b : layer2_) ok &= save_block(f, b);
        for (auto* b : layer3_) ok &= save_block(f, b);
        for (auto* b : layer4_) ok &= save_block(f, b);

        // FC
        ok &= write_gpu_to_file(f, d_fc_w_,    (size_t)num_classes_ * 2048);
        ok &= write_gpu_to_file(f, d_fc_b_,    (size_t)num_classes_);
        ok &= write_gpu_to_file(f, d_vel_fcw_, (size_t)num_classes_ * 2048);
        ok &= write_gpu_to_file(f, d_vel_fcb_, (size_t)num_classes_);

        f.close();
        if (ok) std::cout << "[Checkpoint] Saved -> " << path << "\n";
        return ok;
    } catch (const std::exception& e) {
        std::cerr << "[Checkpoint] Save FAILED: " << e.what() << "\n";
        return false;
    }
}

bool ResNet50::LoadWeights(const std::string& path)
{
    try {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "[Checkpoint] Cannot open: " << path << "\n";
            return false;
        }

        // Header
        uint32_t magic = 0;
        f.read(reinterpret_cast<char*>(&magic), 4);
        if (magic != CKPT_MAGIC) {
            std::cerr << "[Checkpoint] Invalid magic (incompatible format): " << path << "\n";
            return false;
        }
        int32_t nc = 0;
        f.read(reinterpret_cast<char*>(&nc), 4);
        if (nc != num_classes_) {
            std::cerr << "[Checkpoint] num_classes mismatch: expected "
                      << num_classes_ << ", got " << nc << "\n";
            return false;
        }

        // Layers
        bool ok = load_conv(f, conv1_);
        for (auto* b : layer1_) ok &= load_block(f, b);
        for (auto* b : layer2_) ok &= load_block(f, b);
        for (auto* b : layer3_) ok &= load_block(f, b);
        for (auto* b : layer4_) ok &= load_block(f, b);

        // FC
        ok &= read_file_to_gpu(f, d_fc_w_,    (size_t)num_classes_ * 2048);
        ok &= read_file_to_gpu(f, d_fc_b_,    (size_t)num_classes_);
        ok &= read_file_to_gpu(f, d_vel_fcw_, (size_t)num_classes_ * 2048);
        ok &= read_file_to_gpu(f, d_vel_fcb_, (size_t)num_classes_);

        f.close();
        if (ok) std::cout << "[Checkpoint] Loaded <- " << path << "\n";
        else    std::cerr << "[Checkpoint] Load FAILED -> " << path << "\n";
        return ok;
    } catch (const std::exception& e) {
        std::cerr << "[Checkpoint] Load FAILED: " << e.what() << "\n";
        return false;
    }
}

#ifdef USE_NCCL
// ─────────────────────────────────────────────────────────────────────────────
// Multi-GPU gradient synchronization via NCCL
// Call this after Backward(), before UpdateSGD().
// ─────────────────────────────────────────────────────────────────────────────

static void allreduce_conv(ConvBnLayer* c, ncclComm_t comm, cudaStream_t stream) {
    size_t wn = (size_t)c->OutC() * c->InC() * c->KH() * c->KW();
    ncclAllReduce(c->dWeight(),  c->dWeight(),  wn,        ncclFloat, ncclAvg, comm, stream);
    ncclAllReduce(c->dBnScale(), c->dBnScale(), c->OutC(), ncclFloat, ncclAvg, comm, stream);
    ncclAllReduce(c->dBnBias(),  c->dBnBias(),  c->OutC(), ncclFloat, ncclAvg, comm, stream);
}

static void allreduce_block(BottleneckBlock* b, ncclComm_t comm, cudaStream_t stream) {
    allreduce_conv(b->Conv1(), comm, stream);
    allreduce_conv(b->Conv2(), comm, stream);
    allreduce_conv(b->Conv3(), comm, stream);
    if (b->HasProj()) allreduce_conv(b->Proj(), comm, stream);
}

void ResNet50::SyncGradients(ncclComm_t comm, cudaStream_t stream) {
    // FC layer
    ncclAllReduce(d_dfc_w_, d_dfc_w_, (size_t)num_classes_ * 2048, ncclFloat, ncclAvg, comm, stream);
    ncclAllReduce(d_dfc_b_, d_dfc_b_, (size_t)num_classes_,        ncclFloat, ncclAvg, comm, stream);

    // Stage 0
    allreduce_conv(conv1_, comm, stream);

    // Residual stages
    for (auto* b : layer1_) allreduce_block(b, comm, stream);
    for (auto* b : layer2_) allreduce_block(b, comm, stream);
    for (auto* b : layer3_) allreduce_block(b, comm, stream);
    for (auto* b : layer4_) allreduce_block(b, comm, stream);

    // Synchronize the stream so all AllReduce operations complete before UpdateSGD
    cudaStreamSynchronize(stream);
}
#endif // USE_NCCL
