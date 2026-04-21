# ResNet-50 Native CUDA — Optimization Guide

**Target hardware:** NVIDIA H100 (SM90), NVLink, 2-GPU single-node
**Current throughput:** ~2516 img/s (2x H100, global batch 256, DALI loader)
**Baseline:** CUDA 12.6, cuDNN 9.8, NCCL 2.22.3, all FP32 except FC head BF16
**Last updated:** 2026-04-13

---

## Table of Contents

1. [Quick Wins](#1-quick-wins)
2. [Gradient Bucketing](#2-gradient-bucketing)
3. [cuDNN Replacements for Custom Kernels](#3-cudnn-replacements-for-custom-kernels)
4. [BF16 Mixed Precision — Full Network](#4-bf16-mixed-precision--full-network)
5. [H100-Specific Features Not Yet Used](#5-h100-specific-features-not-yet-used)
6. [Priority Roadmap](#6-priority-roadmap)
7. [Implementation History](#7-implementation-history)
8. [Appendix: cuDNN/cuBLAS API Inventory](#appendix-current-cudnncublas-api-inventory)

---

## 1. Quick Wins

### 1.1 `CUDNN_BATCHNORM_SPATIAL_PERSISTENT`

**Status: IMPLEMENTED**

**What:** Switched BN mode from `CUDNN_BATCHNORM_SPATIAL` to `CUDNN_BATCHNORM_SPATIAL_PERSISTENT` in all 3 call sites.

**Where (all in `resnet50.cu`):**
- `ConvBnLayer::Forward()` line 469 — BN forward training
- `ConvBnLayer::Forward()` line 484 — BN forward inference
- `ConvBnLayer::Backward()` line 533 — BN backward

**Why:** The persistent variant uses a single-pass algorithm that keeps warps resident across the batch dimension. On H100 with large spatial maps (early layers: 112x112, 56x56), this avoids repeated kernel launch overhead.

**Expected gain:** 5-15% faster BN operations.

**Risk:** Falls back silently if input contains NaN/Inf. ResNet-50 with BN is always well-conditioned, so this is safe.

---

### 1.2 NCCL Environment Tuning

**Status: IMPLEMENTED**

**What:** Added NCCL performance tuning environment variables to both primary multi-GPU SLURM scripts.

**Variables added:**
```bash
export NCCL_ALGO=Tree          # Better for 2-GPU NVLink topology
export NCCL_P2P_LEVEL=NVL     # Force NVLink peer-to-peer
export NCCL_DEBUG=INFO         # Log what NCCL is doing (remove after tuning)
```

**Where:**
- `run_multigpu_tmpdir_v2.slurm` — after NCCL root setup (line ~44). This script uses `--export=ALL` and no container, so exports propagate directly to the training binary.
- `run_multigpu_tmpdir_dali.slurm` — inside the dynamically generated Apptainer container script (after line ~293). Must be inside the heredoc because the container uses `--cleanenv`, which strips host environment variables.

**For multi-node (future, not yet added):**
```bash
export NCCL_IB_HCA=mlx5_0            # Which InfiniBand HCA to use
export NCCL_NET_GDR_LEVEL=LOC        # GPU Direct RDMA level
export NCCL_CROSS_NIC=1              # Allow cross-NIC communication
```

**Expected gain:** 5-10% on 2-GPU communication; 5-20% on multi-node.

**Verification:** Check SLURM `.out` file for `NCCL INFO` lines confirming `Tree` algorithm and `NVLink` transport.

---

### 1.3 `cudnnFindConvolutionForwardAlgorithmEx` Instead of Heuristic Selection

**Status: IMPLEMENTED (forward convolution only)**

**What:** Replaced heuristic algorithm selection with exhaustive benchmarking for forward convolution. Backward filter and backward data remain heuristic (`cudnnGet*Algorithm_v7`).

**Changes made (3 locations):**

1. **`resnet50.cuh` line 207-209** — Added `const float* d_in` parameter to `refresh_descriptors_and_algos()` signature so FindEx has access to the input buffer.

2. **`resnet50.cu` `ConvBnLayer::Forward()` lines 453-460** — Reordered calls: `ensure_fwd_buffers()` now runs BEFORE `refresh_descriptors_and_algos()` so that `d_conv_out_` is allocated before FindEx needs it as the output buffer.

3. **`resnet50.cu` `refresh_descriptors_and_algos()` lines 386-407** — Replaced forward algo selection:
   - Old: `cudnnGetConvolutionForwardAlgorithm_v7` (heuristic, no GPU execution)
   - New: `cudnnFindConvolutionForwardAlgorithmEx` (benchmarks up to 8 algorithms with a 256 MB temporary workspace)
   - Forward workspace size now taken from `fwd_perfs[0].memory` (FindEx reports exact requirement) instead of a separate `cudnnGetConvolutionForwardWorkspaceSize` call.

**Why only forward?** Backward Find calls would require backward buffers (`d_din_`, `d_dconv_out_`) to exist, and those are only allocated in `ensure_bwd_buffers()` which is called from `Backward()`, not `Forward()`. Forward convolution is the dominant single-operation cost, so optimising only forward gives the best effort-to-benefit ratio.

**Why the reorder is safe:**
- `ensure_fwd_buffers` needs only `(N, out_H, out_W)`, which are computed at the top of `Forward()`
- `ensure_fwd_buffers` does not touch `d_ws_` (which `refresh_descriptors_and_algos` allocates)
- `d_weight_` is always available (allocated in constructor)
- After reorder, `d_conv_out_` is guaranteed non-null when FindEx runs

**Trade-off:** Adds ~5-10 seconds to first forward pass (benchmarks all algos once per unique layer shape). This is a one-time cost — subsequent batches with the same shape reuse cached selection. The 256 MB temporary workspace is freed immediately after FindEx returns.

**Expected gain:** 2-8% if heuristic was suboptimal for H100 Winograd/implicit-GEMM variants.

**Verification:** Add `fprintf(stderr, "FindEx selected algo %d\n", fwd_perfs[0].algo)` inside `refresh_descriptors_and_algos` and compare to old heuristic. Different algo numbers indicate FindEx found a better kernel.

---

### 1.4 Replace `softmax_kernel` with `cudnnSoftmaxForward`

**Status: NOT YET IMPLEMENTED**

**Problem:** Current `softmax_kernel` (line 104) launched as `<<<N, 1>>>` — 1 thread per row, serially loops K=1000. This is ~100x slower than a warp-parallel implementation.

**Fix:** Replace with `cudnnSoftmaxForward()`:
```cpp
// Add to ResNet50 private members:
cudnnTensorDescriptor_t softmax_desc_;   // already added in resnet50.cuh:353
int softmax_prev_N_{0};                  // already added in resnet50.cuh:354

// In constructor (after mp_out_desc_ creation):
CUDNN_CHECK(cudnnCreateTensorDescriptor(&softmax_desc_));

// In destructor (before maxpool descriptor destroy):
cudnnDestroyTensorDescriptor(softmax_desc_);

// Replace softmax_inplace() body:
void ResNet50::softmax_inplace(int N)
{
    if (N != softmax_prev_N_) {
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(softmax_desc_,
                        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                        N, num_classes_, 1, 1));
        softmax_prev_N_ = N;
    }
    const float alpha = 1.f, beta = 0.f;
    CUDNN_CHECK(cudnnSoftmaxForward(cudnn_,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha, softmax_desc_, d_fc_out_,
        &beta,  softmax_desc_, d_fc_out_));
}
```

**Note:** The `softmax_desc_` and `softmax_prev_N_` members have already been added to `resnet50.cuh` (lines 353-354) but the constructor/destructor/softmax_inplace changes in `.cu` have NOT been made yet.

**Impact:** Softmax is not on the critical compute path (tiny vs convolution time), but fixing it removes a performance antipattern. Matters more for large num_classes.

---

## 2. Gradient Bucketing

**Status: IMPLEMENTED (Phase A — pack/unpack approach)**

### What It Does

Instead of 161 individual `ncclAllReduce` calls (one per gradient tensor), all ~25.5M gradient floats are:
1. **Packed** into a single contiguous flat buffer via `cudaMemcpyAsync` (device-to-device)
2. **One `ncclAllReduce`** on the entire buffer — maximum bandwidth utilisation
3. **Unpacked** back to individual gradient tensors

### Files Changed

- `resnet50.cuh`: Added `GradSlice` struct at file scope (line 274, guarded by `#ifdef USE_NCCL`), bucket members (`d_grad_bucket_`, `grad_bucket_nelems_`, `bucket_initialized_`, `grad_slices_`), and method declarations (`BuildGradSlices()`, `InitGradBucket()`)
- `resnet50.cu`: Added `collect_conv_slices()`, `collect_block_slices()` (file-scope helpers), `BuildGradSlices()`, `InitGradBucket()`, rewrote `SyncGradients()`

### Key Implementation Details

- `GradSlice { float* ptr; size_t nelems; }` — describes one gradient tensor's location and size. Declared at file scope (not inside `ResNet50` class) to be accessible by the free helper functions.
- `grad_slices_` — vector of all 161 slices, built once in `BuildGradSlices()`
- `d_grad_bucket_` — flat GPU buffer (~97 MB for ResNet-50 with 1000 classes)
- Bucket is lazily initialised on first `SyncGradients()` call
- Assertion checks `offset == grad_bucket_nelems_` after pack/unpack
- Destructor frees `d_grad_bucket_`

### Bug fix history

- **`GradSlice` accessibility error:** Initially declared as `struct GradSlice` inside `ResNet50`'s `private:` section, causing compile errors when the free functions `collect_conv_slices()` / `collect_block_slices()` tried to use `ResNet50::GradSlice`. Fixed by moving `GradSlice` to file scope in `resnet50.cuh` and removing the `ResNet50::` prefix from all references in `resnet50.cu`.

### Phase B (Future Improvement)

Eliminate the pack/unpack overhead entirely by allocating all gradient buffers (`d_dweight_`, `d_dbn_scale_`, `d_dbn_bias_`) from sub-regions of the flat bucket at construction time. This requires:
- `ConvBnLayer` constructor to accept external gradient pointers instead of `cudaMalloc`-ing its own
- Careful offset management during model construction
- `SyncGradients()` becomes a single `ncclAllReduce` with zero copies

**Expected gain Phase A:** 5-10% throughput improvement
**Expected gain Phase B:** 10-15% (eliminates ~194 MB of extra D2D copies per step)

---

## 3. cuDNN Replacements for Custom Kernels

### Current Custom Kernel Inventory

| # | Kernel | Line | Replaces with cuDNN? | Priority | Status |
|---|--------|------|---------------------|----------|--------|
| 1 | `softmax_kernel` | 104 | **YES — `cudnnSoftmaxForward`** | **HIGH** | Header ready, `.cu` not yet |
| 2 | `gavgpool_kernel` | 80 | YES — `cudnnPoolingForward(AVG)` | Medium | Not started |
| 3 | `gavgpool_bwd_kernel` | 93 | YES — `cudnnPoolingBackward(AVG)` | Medium | Not started |
| 4 | `relu_kernel` | 47 | Possible (`cudnnActivationForward`) | Low | Not worth it |
| 5 | `relu_bwd_kernel` | 54 | Possible (`cudnnActivationBackward`) | Low | Not worth it |
| 6 | `add_inplace_kernel` | 63 | Possible (`cublasSaxpy`) | Low | Not worth it |
| 7 | `add_bias_kernel` | 71 | Possible (`cudnnAddTensor`) | Low | Not worth it |
| 8 | `sce_bwd_kernel` | 118 | No cuDNN equivalent | — | N/A |
| 9 | `ce_loss_kernel` | 130 | No cuDNN equivalent | — | N/A |
| 10 | `top1_kernel` | 141 | No cuDNN equivalent | — | N/A |
| 11 | `sgd_kernel` | 156 | No cuDNN equivalent | — | N/A |
| 12 | `zero_kernel` | 169 | Dead code (`cudaMemsetAsync` used) | — | N/A |
| 13 | `bias_grad_kernel` | 176 | Possible (`cublasSgemv`) | Low | Not worth it |
| 14 | `float_to_bf16_kernel` | 187 | No equivalent | — | N/A |

### 3.1 `softmax_kernel` — Critical Fix

**Problem:** Launched as `<<<N, 1>>>` — **1 thread per row**, serially loops K=1000. This is ~100x slower than cuDNN's warp-parallel implementation.

**Fix:** Replace with `cudnnSoftmaxForward()`. See Section 1.4 for implementation details.

Requires creating a `cudnnTensorDescriptor_t` for the `[N, num_classes, 1, 1]` logit tensor. Members already added to header (`softmax_desc_`, `softmax_prev_N_`); `.cu` changes pending.

### 3.2 `gavgpool_kernel` / `gavgpool_bwd_kernel` — Moderate Win

**Problem:** 1 thread per (n,c) pair, serially sums HW=49 elements. Works but doesn't use shared memory reductions.

**Fix:** Use cuDNN average pooling (already used for MaxPool — same descriptor pattern):
```cpp
// cudnnPoolingForward(cudnn_,
//     avgpool_desc_,  // CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, window=7x7
//     &alpha, in_desc_, d_in,
//     &beta,  out_desc_, d_ap_out_);
```

The MaxPool already demonstrates the full descriptor setup pattern in `resnet50.cu`.

### 3.3 ReLU, Add, BiasAdd — Not Worth Replacing

These are trivially parallel elementwise kernels. The cuDNN descriptor setup overhead (create, set, destroy) exceeds any kernel execution savings.

---

## 4. BF16 Mixed Precision — Full Network

### Current State

- **FC head:** BF16 path implemented and auto-enabled on SM >= 8.0 (H100). Uses `cublasGemmEx` with `CUDA_R_16BF` inputs, `CUBLAS_COMPUTE_32F` accumulation. Master weights in FP32, BF16 shadow refreshed after each SGD step. Controlled by `bf16_stage1_enabled_`, overridable via `RN50_ENABLE_BF16_STAGE1` env var.
- **Convolutions:** FP32 tensors with `CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION` — cuDNN internally downcasts for tensor cores. No memory bandwidth savings.
- **Custom kernels:** All FP32.
- **Loss scaling:** Not implemented.

### Why BF16 vs FP16

| | BF16 | FP16 |
|--|------|------|
| Exponent bits | 8 (same as FP32) | 5 |
| Mantissa bits | 7 | 10 |
| Dynamic range | Same as FP32 | Much smaller |
| Loss scaling needed? | Usually no | Almost always yes |
| H100 throughput | 2000 TFLOPS | 2000 TFLOPS |

**BF16 is the clear choice for H100** — same throughput as FP16 but matching FP32's dynamic range means less numerical fragility and simpler implementation (no loss scaling in most cases).

### Implementation Plan

#### Phase 1: BF16 Convolution Tensors

**Goal:** Store activations in BF16 to halve memory bandwidth. cuDNN computes in FP32 internally.

**Changes to `ConvBnLayer`:**

1. **Add BF16 flag and buffers:**
```cpp
// In ConvBnLayer (resnet50.cuh):
bool bf16_enabled_{false};
__nv_bfloat16 *d_weight_bf16_{nullptr};  // BF16 shadow of conv weight
```

2. **Change tensor descriptors:**
```cpp
// In refresh_descriptors_and_algos():
cudnnDataType_t tensor_dt = bf16_enabled_ ? CUDNN_DATA_BFLOAT16 : CUDNN_DATA_FLOAT;
cudnnSetTensor4dDescriptor(in_desc_,  CUDNN_TENSOR_NCHW, tensor_dt, N, in_c_, H, W);
cudnnSetTensor4dDescriptor(out_desc_, CUDNN_TENSOR_NCHW, tensor_dt, N, out_c_, out_H, out_W);
// Filter descriptor: keep CUDNN_DATA_FLOAT (cuDNN handles mixed types)
// BN descriptor: keep CUDNN_DATA_FLOAT (required by cuDNN for BN params)
```

3. **Activation buffers to BF16:**
```cpp
// d_conv_out_, d_out_ -> allocate as __nv_bfloat16 (half the memory)
// d_dconv_out_, d_din_ -> allocate as __nv_bfloat16
```

4. **Master weight pattern:**
```cpp
// Keep d_weight_ as FP32 for SGD
// Add d_weight_bf16_ shadow, refreshed after UpdateSGD()
// Same pattern as existing FC head BF16
```

5. **cuDNN mixed-type behaviour:**

| cuDNN call | Input type | Filter/Param type | Output type | Works? |
|-----------|-----------|-------------------|------------|--------|
| `ConvolutionForward` | BF16 | FP32 | BF16 | Yes (with `TENSOR_OP_MATH_ALLOW_CONVERSION`) |
| `BatchNormForwardTraining` | BF16 | FP32 (scale/bias/mean/var) | BF16 | Yes (cuDNN 8.1+) |
| `ConvolutionBackwardFilter` | BF16 (dY, X) | — | FP32 (dW) | Yes |
| `ConvolutionBackwardData` | BF16 (dY) | FP32 (W) | BF16 (dX) | Yes |
| `BatchNormBackward` | BF16 (dY, X) | FP32 (scale) | BF16 (dX), FP32 (dscale, dbias) | Yes |

cuDNN 9.8 fully supports BF16 input tensors with FP32 parameters. BN statistics (mean/var/scale/bias) and gradients (dscale/dbias) are always FP32 regardless of input type.

#### Phase 2: Custom Kernel BF16 Overloads

Kernels that operate on activations (now BF16) need overloads:

```cpp
// relu_kernel — BF16 version:
__global__ void relu_kernel_bf16(__nv_bfloat16* __restrict__ x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (__hle(x[idx], __float2bfloat16(0.f)))
            x[idx] = __float2bfloat16(0.f);
    }
}

// add_inplace_kernel — BF16 version:
__global__ void add_inplace_kernel_bf16(__nv_bfloat16* __restrict__ dst,
                                        const __nv_bfloat16* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __hadd(dst[idx], src[idx]);
}
```

Affected kernels:
- `relu_kernel` / `relu_bwd_kernel` — operate on layer activations
- `add_inplace_kernel` — residual add in BottleneckBlock
- `gavgpool_kernel` / `gavgpool_bwd_kernel` — between layer4 and FC head

**Kernels that stay FP32** (operate on gradients/params, not activations):
- `sgd_kernel` — always FP32 (master weights)
- `bias_grad_kernel` — FP32 gradient accumulation
- `sce_bwd_kernel` — FP32 logit gradients
- `ce_loss_kernel` / `top1_kernel` — FP32 probabilities

#### Phase 3: Loss Scaling (if needed)

BF16 has 8 exponent bits (same as FP32), so loss scaling is **usually unnecessary**. However, gradients in very deep early layers can be small.

**Test first without scaling:**
1. Train 5 epochs, compare val accuracy to FP32 baseline.
2. Monitor `GradNorm()` — if it drops to near-zero or training diverges, add scaling.

**If needed — static loss scaling (simplest):**
```cpp
// In Backward():
float loss_scale = 1024.f;
// Multiply into sce_bwd_kernel output:
//   dlogit[idx] = (prob[idx] - one_hot) / N * loss_scale;

// In UpdateSGD() or before it:
//   effective_lr = lr / loss_scale;
// Or fuse into sgd_kernel: grad /= loss_scale before the update.
```

**Dynamic loss scaling (robust, more complex):**
```cpp
struct GradScaler {
    float scale = 65536.f;
    float growth_factor = 2.f;
    float backoff_factor = 0.5f;
    int growth_interval = 2000;
    int steps_since_last_overflow = 0;

    void update(bool overflow_detected) {
        if (overflow_detected) {
            scale *= backoff_factor;
            steps_since_last_overflow = 0;
        } else if (++steps_since_last_overflow >= growth_interval) {
            scale *= growth_factor;
            steps_since_last_overflow = 0;
        }
    }
};
```

#### Memory Savings Estimate

| Buffer category | Count | FP32 size | BF16 size | Savings |
|----------------|-------|-----------|-----------|---------|
| Conv activation fwd | ~53 | ~1.2 GB | ~600 MB | 600 MB |
| Conv activation bwd | ~53 | ~1.2 GB | ~600 MB | 600 MB |
| Conv weights (master) | 53 | ~92 MB | stays FP32 | 0 |
| Conv weight BF16 shadow | 53 | — | ~46 MB | -46 MB |
| **Net saving** | | | | **~1.15 GB** |

This freed memory enables larger batch sizes (e.g., 256->384 per GPU), which further amortises communication overhead.

---

## 5. H100-Specific Features Not Yet Used

### 5.1 CUDA Graphs

**Status: NOT IMPLEMENTED**

**What:** Capture the forward-backward-SGD kernel sequence as a graph, replay it each step without CPU launch overhead.

**Why:** Each training step launches ~400+ kernels. On H100, kernel launch overhead is ~3-5us each = ~1.5ms/step of pure launch overhead.

**How:**
```cpp
cudaGraph_t graph;
cudaGraphExec_t graph_exec;

// Capture:
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
model.ZeroGrad();
model.Forward(d_images, N);
model.Backward(d_labels, N);
model.SyncGradients(comm, stream);
model.UpdateSGD(lr, momentum, wd);
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graph_exec, graph, ...);

// Replay each step:
cudaGraphLaunch(graph_exec, stream);
```

**Caveat:** Graphs require fixed tensor shapes (N, H, W). If batch size varies (last batch of epoch), you need a fallback non-graph path. Also, NCCL inside graphs requires NCCL 2.19+ (you have 2.22.3 -- OK).

**Expected gain:** 5-10% for small batch sizes; diminishes as compute dominates.

---

### 5.2 Shared Workspace Pool

**Status: NOT IMPLEMENTED**

**What:** Currently each of 53 ConvBnLayers allocates its own cuDNN workspace. Since only one layer executes at a time (single stream), one shared workspace suffices.

**How:**
```cpp
// At model init:
size_t max_ws = 0;
for (auto* layer : all_conv_bn_layers) {
    max_ws = std::max(max_ws, layer->required_workspace());
}
CUDA_CHECK(cudaMalloc(&d_shared_ws_, max_ws));

// Pass shared_ws to each layer instead of per-layer allocation
```

**Savings:** ~100-200 MB GPU memory (not throughput, but enables larger batches).

---

### 5.3 FP8 (H100 Exclusive)

**Status: NOT IMPLEMENTED**

**What:** H100 has native FP8 tensor cores — E4M3 (4 exponent, 3 mantissa) for forward, E5M2 (5 exponent, 2 mantissa) for backward.

**Throughput:** 4000 TFLOPS (4x FP32, 2x BF16).

**Complexity:** Very high. Requires:
- Per-tensor dynamic scaling (track amax history, compute scale factors)
- NVIDIA Transformer Engine-style infrastructure
- Separate E4M3/E5M2 paths for forward/backward
- Careful handling of BN (must stay in higher precision)

**Recommendation:** Only pursue after BF16 is fully working and validated. The 2x gain over BF16 is significant but the implementation cost is enormous for a hand-written engine.

---

### 5.4 Multi-Stream Compute/Communication Overlap

**Status: NOT IMPLEMENTED**

**What:** Use separate CUDA streams for compute and NCCL communication, with events for synchronisation.

**Currently:** Single stream — backward must fully complete before AllReduce starts.

**Ideal flow:**
```
Compute stream:  [layer4 bwd] -> [layer3 bwd] -> [layer2 bwd] -> [layer1 bwd]
                       |              |              |
                   event3          event2          event1
                       |              |              |
Comm stream:     [AllReduce bucket3] [AllReduce bucket2] [AllReduce bucket1]
```

**Requirements:**
- Restructure `Backward()` to expose per-stage boundaries
- Create a communication stream separate from compute stream
- Use `cudaEventRecord()` / `cudaStreamWaitEvent()` for ordering
- NCCL must use the comm stream: `ncclAllReduce(..., comm_stream)`
- SGD must wait for both streams to complete

**Expected gain:** 10-25% (overlap hides most communication latency).

**Effort:** High — invasive refactoring of backward pass and SyncGradients.

---

## 6. Priority Roadmap

### Tier 1: Quick Wins

| # | Optimization | Effort | Gain | Risk | Status |
|---|-------------|--------|------|------|--------|
| 1 | `CUDNN_BATCHNORM_SPATIAL_PERSISTENT` | 10 min | 5-15% BN | None | **DONE** |
| 2 | NCCL env tuning | 5 min | 5-10% comm | None | **DONE** |
| 3 | Replace `softmax_kernel` with `cudnnSoftmaxForward` | 30 min | Small | None | Header ready |
| 4 | `cudnnFindAlgorithmEx` for conv fwd algo | 1 hr | 2-8% conv | Startup cost | **DONE** (fwd only) |

### Tier 2: Moderate Effort (next sprint)

| # | Optimization | Effort | Gain | Risk | Status |
|---|-------------|--------|------|------|--------|
| 5 | Gradient bucketing Phase B (zero-copy) | 1 day | +5% over Phase A | Medium | Not started |
| 6 | Shared cuDNN workspace pool | 2 hr | Memory savings | Low | Not started |
| 7 | Replace gavgpool with cuDNN | 1 hr | Small | Low | Not started |

### Tier 3: Major Features (planned)

| # | Optimization | Effort | Gain | Risk | Status |
|---|-------------|--------|------|------|--------|
| 8 | BF16 full-network (Phases 1-3) | 2-3 days | 15-30% | Medium | Not started |
| 9 | CUDA Graphs | 1-2 days | 5-10% | Medium | Not started |
| 10 | Multi-stream overlap | 3-5 days | 10-25% | High | Not started |

### Tier 4: Advanced (future)

| # | Optimization | Effort | Gain | Risk | Status |
|---|-------------|--------|------|------|--------|
| 11 | FP8 (E4M3/E5M2) | 1-2 weeks | ~2x over BF16 | High | Not started |
| 12 | cuDNN Graph API (fused kernels) | 1 week | 10-20% | High | Not started |

---

## 7. Implementation History

### 2026-04-13 — Quick Wins Batch

**Opt 1: BN Persistent Mode**
- Changed `CUDNN_BATCHNORM_SPATIAL` -> `CUDNN_BATCHNORM_SPATIAL_PERSISTENT` in all 3 BN call sites (fwd train, fwd inference, backward) in `resnet50.cu`

**Opt 3: NCCL Tuning**
- Added `NCCL_ALGO=Tree`, `NCCL_P2P_LEVEL=NVL`, `NCCL_DEBUG=INFO` to:
  - `run_multigpu_tmpdir_v2.slurm` (after NCCL root setup)
  - `run_multigpu_tmpdir_dali.slurm` (inside container heredoc script)

**Opt 4: FindEx Algorithm Selection**
- `resnet50.cuh`: Added `const float* d_in` parameter to `refresh_descriptors_and_algos`
- `resnet50.cu` `Forward()`: Reordered `ensure_fwd_buffers()` before `refresh_descriptors_and_algos()` so `d_conv_out_` exists for FindEx
- `resnet50.cu` `refresh_descriptors_and_algos()`: Replaced `cudnnGetConvolutionForwardAlgorithm_v7` with `cudnnFindConvolutionForwardAlgorithmEx` (benchmarks 8 algos, 256 MB temp workspace). Backward algos unchanged. Forward workspace size from `fwd_perfs[0].memory`.

### 2026-04-13 — Gradient Bucketing (Phase A)

- Added `GradSlice` struct at file scope in `resnet50.cuh` (guarded by `#ifdef USE_NCCL`)
- Added bucket infrastructure: `d_grad_bucket_`, `grad_bucket_nelems_`, `bucket_initialized_`, `grad_slices_`, `BuildGradSlices()`, `InitGradBucket()`
- Rewrote `SyncGradients()`: pack all 161 gradient tensors into flat buffer, single `ncclAllReduce`, unpack
- Added `collect_conv_slices()` / `collect_block_slices()` file-scope helpers
- **Bug fix:** Moved `GradSlice` from `ResNet50::private` to file scope to fix accessibility error with free helper functions

### Pre-existing optimizations (before this session)

- **FC head BF16**: `cublasGemmEx` with `CUDA_R_16BF` inputs, `CUBLAS_COMPUTE_32F` accumulation. Auto-enabled on SM >= 8.0 via `configure_bf16_stage1()`. Master FP32 weights with BF16 shadow refreshed after each SGD step.
- **TF32 tensor cores**: `CUBLAS_TF32_TENSOR_OP_MATH` set in `train_multigpu.cu`
- **cuDNN tensor ops**: `CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION` on all 53 conv descriptors
- **NCCL grouped collectives**: All AllReduce calls wrapped in `ncclGroupStart()`/`ncclGroupEnd()` (now superseded by gradient bucketing)
- **DALI data loading**: GPU-accelerated decode/augmentation via NVIDIA DALI C API

---

## Appendix: Current cuDNN/cuBLAS API Inventory

### cuDNN Calls (per training step)

| API | Count | Location |
|-----|-------|----------|
| `cudnnConvolutionForward` | 53 | `ConvBnLayer::Forward` |
| `cudnnBatchNormalizationForwardTraining` | 53 | `ConvBnLayer::Forward` |
| `cudnnBatchNormalizationBackward` | 53 | `ConvBnLayer::Backward` |
| `cudnnConvolutionBackwardFilter` | 53 | `ConvBnLayer::Backward` |
| `cudnnConvolutionBackwardData` | 53 | `ConvBnLayer::Backward` |
| `cudnnPoolingForward` (MaxPool) | 1 | `ResNet50::maxpool_fwd` |
| `cudnnPoolingBackward` (MaxPool) | 1 | `ResNet50::maxpool_bwd` |
| **Total** | **267** | |

### cuBLAS Calls (per training step)

| API | Count | Location |
|-----|-------|----------|
| `cublasGemmEx` (BF16, FC fwd) | 1 | `ResNet50::fc_fwd` |
| `cublasGemmEx` (BF16, FC bwd dX) | 1 | `ResNet50::fc_bwd` |
| `cublasGemmEx` (BF16, FC bwd dW) | 1 | `ResNet50::fc_bwd` |
| **Total** | **3** | |

### Custom CUDA Kernels (per training step)

| Kernel | Fwd calls | Bwd calls | SGD calls | Total |
|--------|-----------|-----------|-----------|-------|
| `relu_kernel` | 49 | — | — | 49 |
| `relu_bwd_kernel` | — | 49+16 | — | 65 |
| `add_inplace_kernel` | 16 | 16 | — | 32 |
| `gavgpool_kernel` | 1 | — | — | 1 |
| `gavgpool_bwd_kernel` | — | 1 | — | 1 |
| `softmax_kernel` | 1 | — | — | 1 |
| `sce_bwd_kernel` | — | 1 | — | 1 |
| `float_to_bf16_kernel` | 1 | 2 | — | 3 |
| `add_bias_kernel` | 1 | — | — | 1 |
| `bias_grad_kernel` | — | 1 | — | 1 |
| `sgd_kernel` | — | — | 161 | 161 |
| `ce_loss_kernel` | — | — | — | eval only |
| `top1_kernel` | — | — | — | eval only |
| **Total** | **69** | **70** | **161** | **~316** |
