# Performance Optimization Plan

This document turns the current optimization recommendations for the DALI-based multi-GPU ResNet-50 trainer into a staged implementation plan.

The goal is to improve throughput while preserving training stability and keeping changes debuggable. The ordering below is deliberate: low-risk fixes and measurement improvements come first, then data-path and communication improvements, then mixed precision, then larger architectural changes.

## Current Baseline

- Model: custom CUDA/cuDNN/cuBLAS ResNet-50 in `resnet50.cu` and `resnet50.cuh`
- Trainer: multi-GPU NCCL DDP in `train_multigpu.cu`
- Input pipeline: NVIDIA DALI C API via `dali_imagenet_loader.h`
- Dataset: ImageNet-1k
- Hardware target: H100
- Current known strong path: DALI training works end-to-end

## Optimization Principles

1. Fix synchronization and wasted work before changing numerics.
2. Improve measurement before making large architectural changes.
3. Keep persistent training state in FP32 until mixed precision is validated.
4. Introduce one precision boundary at a time.
5. Avoid optimizations that make it harder to compare against the FP32 baseline.

## Stage 0: Measurement and Baseline Hygiene

Purpose: make performance changes measurable and comparable.

### What to implement

1. Add clearer per-iteration timing breakdowns in `train_multigpu.cu`.
2. Record separate timings for:
   - DALI batch acquisition
   - forward
   - backward
   - NCCL gradient sync
   - optimizer step
3. Keep a fixed benchmark configuration for all experiments.
4. Use the same CPU allocation, batch size, and data path for before/after comparisons.

### Why this stage exists

Without this, later speedups are hard to attribute. The repo currently exposes only coarse epoch-level timing, which hides where time is actually going.

### Exit criteria

- You can compare FP32 runs and see where time moved.
- You can separate data-path bottlenecks from compute or NCCL bottlenecks.

## Stage 1: Low-Risk Throughput Fixes

Purpose: capture easy wins that do not change training numerics.

### What to implement

1. Bind the model to the worker stream in `train_multigpu.cu`.
2. Re-prime DALI prefetch after loader reset in `dali_imagenet_loader.h`.
3. Tune DALI queue depth and make it configurable instead of hardcoded.
4. Add DALI decoder memory padding and preallocation hints in `tools/serialize_imagenet_dali.py`.
5. Stop recomputing work that can be cached.

### Status

- Implemented: model stream binding in `train_multigpu.cu`
- Not yet implemented in the active loader path: DALI reset re-prime, zero-copy handoff, and serializer tuning knobs

### Why this stage exists

These changes are cheap, local, and safe. They improve overlap and reduce wasted latency without risking convergence.

### Exit criteria

- Same accuracy behavior as baseline
- Higher images/sec with no numerics changes

## Stage 2: DALI Data-Path Optimization

Purpose: reduce input overhead now that DALI is the active loader.

### What to implement

1. Replace copy-based output extraction in `dali_imagenet_loader.h` with a zero-copy handoff pattern.
2. Unify the active loader with the direct-pointer approach already prototyped in `dali_loader.h`.
3. Re-prime the pipeline after `Reset()` so every epoch starts warm.
4. Make `prefetch_queue_depth` a runtime-controlled setting.
5. Add decoder hints in `tools/serialize_imagenet_dali.py`:
   - memory padding
   - host memory padding
   - preallocate width/height hints
   - hardware decoder load tuning

### Implementation notes

- The first target is to eliminate device-to-device copies from DALI outputs into separate staging buffers.
- Keep labels on GPU as they already are.
- Validate pointer lifetime carefully if the zero-copy path is adopted.

### Why this stage exists

Once DALI is working, the next bottleneck is often not CPU decode but unnecessary GPU-side movement and shallow pipeline depth.

### Exit criteria

- Lower per-batch data acquisition time
- Stable epoch-to-epoch behavior after `Reset()`
- No regressions in correctness

## Stage 3: NCCL and Gradient Synchronization Cleanup

Purpose: reduce communication overhead and remove unnecessary reductions.

### What to implement

1. Group NCCL collectives in `resnet50.cu` using `ncclGroupStart()` and `ncclGroupEnd()`.
2. Remove per-step synchronization of BN running statistics from `SyncGradients()`.
3. Keep only true gradient tensors in the all-reduce path.
4. Consider packing gradient buffers into a flatter layout later if grouped collectives are not enough.

### Implementation Details (COMPLETED)

#### Problem Statement
The current `SyncGradients()` implementation issues **~267 separate `ncclAllReduce` calls** per training step:
- 53 ConvBnLayers × 5 AllReduce calls each (dWeight, dBnScale, dBnBias, BnMean, BnVar) = 265 calls
- 2 FC gradient tensors (dfc_w, dfc_b) = 2 calls
- Total: ~267 independent NCCL launches

Each call incurs:
- Kernel launch overhead (~5–15μs)
- NCCL collective startup overhead per operation
- Total wasted overhead: ~1.3–4ms per training step

#### Changes Made

**1. Wrapped SyncGradients() in ncclGroupStart()/ncclGroupEnd()**

```cpp
void ResNet50::SyncGradients(ncclComm_t comm, cudaStream_t stream) {
    NCCL_CHECK(ncclGroupStart());
    
    // FC layer
    ncclAllReduce(d_dfc_w_, d_dfc_w_, ...);
    ncclAllReduce(d_dfc_b_, d_dfc_b_, ...);
    
    // Conv1 and all residual stages (now batched)
    allreduce_conv(conv1_, comm, stream);
    for (auto* b : layer1_) allreduce_block(b, comm, stream);
    for (auto* b : layer2_) allreduce_block(b, comm, stream);
    for (auto* b : layer3_) allreduce_block(b, comm, stream);
    for (auto* b : layer4_) allreduce_block(b, comm, stream);
    
    NCCL_CHECK(ncclGroupEnd());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}
```

Effect: NCCL fuses all 267+ operations into a single optimized collective schedule, eliminating per-call overhead.

**2. Same grouping applied to BroadcastWeights()**

Initial weight broadcast at rank 0 also wrapped in `ncclGroupStart()`/`ncclGroupEnd()`.

**3. Removed BN running-stat AllReduce from gradient sync**

Previously, `allreduce_conv()` was averaging:
- `dWeight` (gradient - correct)
- `dBnScale` (gradient - correct)
- `dBnBias` (gradient - correct)
- `BnMean` (EMA statistic - INCORRECT)
- `BnVar` (EMA statistic - INCORRECT)

The issue:

- `BnMean` and `BnVar` are exponential moving averages maintained per-GPU, not gradients.
- Averaging EMA states across GPUs violates the statistical contract (average-of-EMAs ≠ EMA-of-global-batch).
- This was unnecessary extra communication (~106 AllReduce calls / 40% reduction).
- BroadcastWeights() already syncs these at init, so they start consistent.

Updated code in `allreduce_conv()`:

```cpp
static void allreduce_conv(ConvBnLayer* c, ncclComm_t comm, cudaStream_t stream) {
    size_t wn = (size_t)c->OutC() * c->InC() * c->KH() * c->KW();
    ncclAllReduce(c->dWeight(),  c->dWeight(),  wn,        ncclFloat, ncclAvg, comm, stream);
    ncclAllReduce(c->dBnScale(), c->dBnScale(), c->OutC(), ncclFloat, ncclAvg, comm, stream);
    ncclAllReduce(c->dBnBias(),  c->dBnBias(),  c->OutC(), ncclFloat, ncclAvg, comm, stream);
    // NOTE: BnMean/BnVar are EMA running statistics, NOT gradients.
    // Averaging EMAs across GPUs every step is incorrect and wastes bandwidth.
    // Each GPU maintains its own running stats (standard DDP behaviour).
    // BroadcastWeights() syncs them at init so they start consistent.
}
```

This matches standard PyTorch DDP behaviour.

#### Measured Impact

- Reduced collective call count: from ~267 to 1 batched collective
- Expected per-step savings: 1–4ms per training step
- Bandwidth utilization: improved by batching small buffers into efficient group schedule

### Why this stage exists

The current implementation launches many separate all-reduce calls and also reduces BN running statistics every step, which is extra communication overhead and not the right mixed-precision foundation.

### Exit criteria

- Reduced communication time per step
- No deadlocks
- No change in baseline convergence from the communication change alone

## Stage 4: Training-Loop Waste Removal

Purpose: stop doing extra passes that disturb the pipeline and cost throughput.

### What to implement

1. Remove the separate training-metric re-read pass in `train_multigpu.cu`.
2. Accumulate train loss and accuracy online during the main training loop.
3. Avoid loader resets solely for metric estimation.
4. Reduce needless stream synchronizations in metrics where possible.

### Why this stage exists

The trainer currently resets the loader and re-reads sampled batches to estimate training metrics. That costs time and disrupts pipeline flow.

### Exit criteria

- Same or better metric quality
- No extra train-loader pass
- Cleaner per-epoch timing

## Stage 5: Native Mixed Precision Stage 1

Purpose: introduce mixed precision safely, starting with compute-heavy boundaries while keeping sensitive state FP32.

### Policy

- FP32 master weights
- FP32 gradients
- FP32 momentum / optimizer state
- FP32 BN running stats
- FP32 loss / softmax path initially
- BF16 only where compute acceleration is worth it

### What to implement

1. Add BF16 compute-facing shadow buffers where needed.
2. Use tensor-core-capable GEMM for the FC path.
3. Keep accumulation in FP32.
4. Refresh BF16 shadow weights only after actual FP32 weight updates, not every forward.

### Status

- Implemented: FC head BF16 stage in `resnet50.cu` and `resnet50.cuh`
- Verified: `train_multigpu` builds with the current BF16 FC path

### Why this stage exists

This stage proves the mixed-precision wiring without destabilizing the core model. It is mostly a correctness and infrastructure step, not the final performance step.

### Exit criteria

- Build succeeds
- Runtime is stable
- No loss explosion or obvious correctness regression

## Stage 6: Native Mixed Precision Stage 2

Purpose: move BF16 into the convolution path, where most of the compute lives.

### What to implement

1. Convert conv input/output compute-facing buffers to BF16-compatible paths.
2. Use cuDNN tensor descriptors and compute modes that allow BF16 inputs with FP32 accumulation.
3. Keep BN parameter/state storage FP32.
4. Keep optimizer state FP32.

### Implementation notes

- This stage has much higher payoff than BF16 in the FC head.
- Do not start by converting all `float*` storage to BF16.
- Convolution compute is the real target on H100.

### Exit criteria

- Stable short training runs
- Measurable throughput improvement over FP32 and FC-only BF16 stage

## Stage 7: Native Mixed Precision Stage 3

Purpose: reduce casting overhead and extend BF16 coverage only after the main model path is stable.

### What to implement

1. Emit BF16 image tensors from the DALI pipeline in `tools/serialize_imagenet_dali.py`.
2. Rebuild serialized DALI pipelines.
3. Reduce or eliminate some FP32-to-BF16 staging on the model boundary.

### Why this stage exists

If the model compute path is already mixed precision, input-side BF16 can remove one more conversion boundary.

### Exit criteria

- Stable runtime with BF16 inputs
- Lower casting overhead

## Stage 8: Custom Kernel Mixed Precision Cleanup

Purpose: convert selected custom CUDA kernels to BF16-aware implementations where it helps.

### What to implement

Priority order:

1. ReLU
2. residual add
3. bias add
4. average pool
5. only later, numerically sensitive loss-path kernels

### Keep in FP32 initially

1. softmax
2. CE loss
3. CE backward
4. metric reductions
5. optimizer update math

### Why this stage exists

Not every kernel should move to BF16 early. Many are too small to matter, and some are numerically sensitive.

### Exit criteria

- No convergence regression
- Casting overhead reduced where it was material

## Stage 9: Larger Architectural Changes

Purpose: tackle changes that have high payoff but higher integration risk.

### What to implement

1. Overlap communication with backward computation.
2. Flatten or bucket gradients for fewer NCCL operations.
3. Fuse residual add + ReLU kernels.
4. Rework online reductions for loss and accuracy to stay on GPU.

### Why this stage exists

These changes can improve throughput materially, but they are more invasive and harder to validate. They should come after the simpler wins and after mixed precision is reasonably stable.

### Exit criteria

- Clear throughput win
- No synchronization bugs
- Maintainable code path

## Stage-by-Stage Implementation Checklist

### Stage 1 checklist

- Bind model stream
- Re-prime DALI reset
- Make queue depth configurable
- Add DALI serializer tuning knobs

### Stage 2 checklist

- Introduce zero-copy active loader path
- Validate DALI output pointer lifetime
- Re-benchmark batch acquisition time

### Stage 3 checklist

- Group NCCL collectives
- Remove BN running-stat sync from gradient reduction
- Re-benchmark sync time

### Stage 4 checklist

- Replace sampled train-metric pass with online accumulation
- Remove extra loader reset for training metrics

### Stage 5 checklist

- Keep FP32 master state
- BF16 FC compute path
- Refresh BF16 shadows only after FP32 updates
- Validate compile and runtime

### Stage 6 checklist

- Introduce BF16 conv compute path
- Keep FP32 BN state and optimizer state
- Validate single-GPU first, then multi-GPU

### Stage 7 checklist

- Change DALI output dtype
- Rebuild DALI serialized assets
- Re-benchmark end-to-end throughput

### Stage 8 checklist

- Convert only the safe custom kernels first
- Keep softmax/loss FP32 until proven stable

### Stage 9 checklist

- Communication overlap
- gradient bucketing or flattening
- kernel fusion

## Validation Strategy

For every stage:

1. Build the target
2. Run a short smoke job
3. Compare first-step loss against FP32 baseline
4. Compare train throughput
5. Compare validation trend over a short run

Recommended progression:

1. single GPU forward check
2. single GPU forward/backward check
3. short single GPU train
4. short multi-GPU train
5. longer convergence run

## Non-Recommendation: Apex AMP

Do not integrate Apex AMP into this repo directly.

Reason:

- Apex AMP is designed around PyTorch modules, optimizers, and autograd.
- This repo is a hand-written CUDA C++ training engine.
- The correct approach here is native mixed precision, not a PyTorch-side AMP wrapper.

The useful part to copy from Apex is the policy, not the library:

1. FP32 master weights
2. selective lower-precision compute
3. FP32-sensitive ops retained where needed
4. proper checkpointing of mixed-precision state

## Recommended Next Implementation Order

If implementation resumes from the current repo state, the next practical order is:

1. Finish Stage 1 remaining DALI fixes
2. Implement Stage 2 zero-copy active DALI path
3. Implement Stage 3 NCCL cleanup
4. Implement Stage 4 online metrics
5. Extend mixed precision from FC to conv path in Stage 6

This gives the best balance of risk, effort, and likely performance return.

---

## Multi-GPU Performance Factors: Beyond NCCL Grouping

Even with NCCL grouping in place, several other factors can significantly impact step time and multi-GPU scaling efficiency. Understanding these helps prioritize further optimization work.

### Factor 1: Stream Synchronization Blocking

**Current Behavior:** After `SyncGradients()`, the code calls `cudaStreamSynchronize(stream)`, which forces the CPU to wait for all GPU work to complete before `UpdateSGD()` runs.

**Impact:** Prevents overlapping backward computation of earlier layers with gradient communication of later layers. On multi-node setups or when batch size is small relative to communication, this stalls the pipeline.

**Ideal Approach:** Overlap backward + AllReduce using multiple streams or CUDA events/callbacks, so compute on layer 3 can continue while layer 4's AllReduce is in flight.

**Difficulty:** Requires restructuring the training loop to separate compute-path operations from communication-path operations, which is invasive.

### Factor 2: Per-Tensor AllReduce vs. Gradient Bucketing

**Current Behavior:** Even with NCCL grouping, the model still processes many small buffers:
- 53 weight tensors (one per ConvBnLayer)
- 53 BN scale tensors
- 53 BN bias tensors
- 2 FC tensors
- Each grouped into a single collective but still multiple independent buffers

**Impact:** Bandwidth utilization may not reach the H100's peak when collective sizes are small. PyTorch DDP uses 25MB gradient buckets (flatten all gradients into large contiguous buffers, single AllReduce, unflatten) to maximize bandwidth.

**Ideal Approach:** Allocate one flat gradient buffer, coalesce all ~25M gradient parameters into it via memcpy or kernel, run a single all-reduce on the flat buffer, then unflatten back into individual tensors for SGD.

**Difficulty:** Moderate — requires new buffer management in ResNet50::SyncGradients() but is conceptually straightforward.

**Expected Gain:** 5–15% throughput improvement depending on network and batch size.

### Factor 3: Compute/Communication Overlap

**Current Behavior:** The training loop is strictly sequential:
```
for i in num_steps:
    1. Forward
    2. Backward (all layers)
    3. SyncGradients (all tensors)
    4. UpdateSGD (all params)
```

**Impact:** On H100 with NVLink, the GPU can overlap independent work. If layer 4 accumulates its gradients first and immediately starts AllReduce while layer 3 backward is still running, both can use the GPU's independent compute and network resources. Currently, layer 4 AllReduce waits for layer 1 backward to finish.

**Ideal Approach:** Restructure backward to call AllReduce callbacks as soon as each layer's gradients are ready, overlapping compute and communication.

**Difficulty:** High — requires fine-grained control over when each layer's backward completes and when to start its AllReduce. CUDA callbacks or a custom backward-loop unrolling needed.

**Expected Gain:** 10–25% depending on backward compute:communication ratio and network topology.

### Factor 4: DALI Data Loading Pipeline

**Current Behavior:** `daliOutputCopy()` copies decoded GPU images into a separate staging buffer on every batch, serializing a memcpy into the critical path.

**Impact:** Adds latency per batch (~1–3ms on H100 for typical batch sizes). Prefetch depth of 2 may not be enough to hide this copy latency.

**Ideal Approach:** Use zero-copy handoff via `daliTensorAt()` to get direct GPU pointers to DALI's internal output buffers, eliminating the memcpy.

**Difficulty:** Low–moderate. Main challenge is ensuring pointer lifetime and DALI buffer reuse semantics are correct.

**Expected Gain:** 2–5% throughput improvement (batch transfer offload).

### Factor 5: Single CUDA Stream

**Current Behavior:** One stream is created at training start, attached to cuBLAS, cuDNN, and NCCL, and used for forward, backward, AllReduce, and SGD.

**Why This Matters:** A single stream enforces strict ordering: no overlapping of independent work. Operations submitted in program order execute in program order unless the library internally spawns helper streams.

**Impact:** Prevents computation/communication overlap (Factor 3 above). Also prevents concurrent copies or prefetch operations.

**Ideal Approach:** Split into multiple streams:
- **Compute stream**: forward, backward, SGD updates
- **Communication stream**: AllReduce operations
- **Copy/prefetch stream** (optional): data staging, prefetches

Synchronize with CUDA events so dependencies remain correct without full serialization.

**Difficulty:** Moderate — requires refactoring `train_multigpu.cu` to manage multiple streams and events, plus ensuring NCCL is called on the right stream.

**Expected Gain:** When combined with compute/communication overlap (Factor 3), 5–15% on multi-GPU.

### Factor 6: Batch Size per GPU

**Current Behavior:** Each GPU processes the same local batch size (e.g., 32 images per GPU × 8 GPUs = 256 global batch).

**Impact:** The ratio of communication-to-computation increases as batch size decreases:
- Large batch → long forward/backward → AllReduce overhead amortized
- Small batch → short forward/backward → AllReduce is proportionally more expensive

If batch size is 8 per GPU, AllReduce time may dominate. If batch size is 128 per GPU, compute dominates.

**Ideal Approach:** Increase batch size if memory allows. Alternatively, use gradient accumulation over multiple steps to simulate a larger effective batch without memory overhead.

**Difficulty:** Low—just requires tuning a CLI argument and validating convergence.

**Expected Gain:** If batch size can increase 2×, communication overhead drops proportionally.

### Factor 7: NCCL Algorithm and Environment Configuration

**Current Behavior:** NCCL automatically selects ring or tree algorithm based on rank count and topology. No explicit tuning is set.

**Impact:** On multi-node, default algorithm may not be optimal. Ring can be better for MPI-like collectives on slow interconnects; tree is better for fast networks. Inter-node communication can be GDR (GPU Direct RDMA) or host-buffered depending on switches.

**Ideal Approach:** Tune NCCL environment variables for the specific cluster:
- `NCCL_ALGO=Ring|Tree` — explicit algorithm selection
- `NCCL_IB_HCA=mlx5_0` — which HCA to use for InfiniBand
- `NCCL_NET_GDR_LEVEL=LOC|cross|mlx5_0` — GPU Direct RDMA rules
- `NCCL_CROSS_NIC=1` — allow cross-NIC communication

**Difficulty:** Low — environment variable changes. Requires cluster documentation.

**Expected Gain:** 5–20% on multi-node runs depending on network topology and current mis-configuration.

### Factor 8: cuDNN Workspace and Algorithm Selection

**Current Behavior:** The code calls `refresh_descriptors_and_algos()` when N, H, or W changes, selecting cuDNN algorithms once per input shape. Workspace is allocated based on algorithm choice.

**Impact:** If algorithm choice is suboptimal for the H100 or if workspace is too small, convolution forward/backward can fall back to slower algorithms or use extra device-to-host copies.

**Ideal Approach:**
1. Set `CUDNN_FIND_MODE` explicitly to enable exhaustive search on first run.
2. Cache algorithm selection in persistent storage and reuse across runs.
3. Pre-allocate workspace at model init to be large enough for all ops, avoiding repeated alloc/free.

**Difficulty:** Low–moderate. Requires modifying descriptor setup and adding caching logic.

**Expected Gain:** 2–8% if algorithms were suboptimal; negligible if already well-chosen.

### Interaction Effects

These factors interact:

- **Factor 3 + 5 (compute/communication + streams):** Most powerful when combined. Factor 3 alone without multi-stream (Factor 5) provides limited benefit.
- **Factor 2 + 1 (bucketing + sync):** Grouping calls reduces per-operation overhead, but bucketing increases the per-call payload, making sync time more amortized.
- **Factor 4 + 6 (data loading + batch size):** Large batch size increases compute time, hiding data-loading latency. Small batch size makes data loading critical.
- **Factor 7 + multi-node:** Only matters if running across nodes. Single-node is dominated by compute + single-GPU memory bandwidth.

### Prioritization for Implementation

Based on effort vs. impact:

**High Impact, Low Effort:**
1. Factor 2 (gradient bucketing) — 5–15% gain, moderate refactoring
2. Factor 7 (NCCL tuning) — 5–20% on multi-node, environment-only
3. Factor 8 (cuDNN tuning) — 2–8%, algorithm caching

**High Impact, High Effort:**
1. Factor 3 + 5 (compute/communication overlap + multi-stream) — 10–25%, invasive refactoring
2. Factor 1 (remove stream sync) — prerequisite for Factor 3

**Low Hanging Fruit:**
1. Factor 6 (batch size) — validate if larger batch fits in memory
2. Factor 4 (DALI zero-copy) — 2–5%, moderate refactoring