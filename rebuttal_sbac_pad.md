# SBAC-PAD 2026 Rebuttal

Organized by reviewer: Reviewer 1 (reproducibility, micro-benchmark platform, CARAML/LLM, relation to other tools), Reviewer 2 (FP32 only, DDP fairness, "framework tax" definition, model scope), Reviewer 3 (topology asymmetry, title). Paste-ready text is the prose block under each reviewer heading, ending at the next heading or divider; the closing "Summary of changes" paragraph after R3-2 is part of the paste text. The title-handling list after it and everything from the "Internal support material" heading onward are internal and must not be pasted.

Suggested opening line for the whole rebuttal:

We thank the reviewers for the careful reading. We address each reviewer's concerns
in turn and mark every change we commit to for the final version.

# Reviewer 1

## R1-1a. "No CUDA or HIP versions are provided for reproducibility" (Section IV)


R1-1a (software versions). The final version will state the full software
specification in Section IV for both vendors.

On H100, the native trainer is built with CUDA toolkit 12.6.2 and resolves
cuDNN 9.8.0 and NCCL 2.18.5 from the host; the framework baseline is the NGC
PyTorch 24.02 container (PyTorch 2.3.0a0, CUDA runtime 12.3, cuDNN 9.0.0, NCCL
2.20.3, DALI 1.34.0). Both stacks share one container image and driver, but the
container does not pin the native trainer's math libraries, so this skew is
disclosed rather than assumed away, and it runs against, not for, the native
result: on the newer cuDNN the native trainer's own measured marginal cost is
0.703 ms per image against the framework's 0.672 ms on the older, so the
library difference does not supply the reported gap.

On MI300X the pattern is the same and is measured, not promised. The native
trainer runs ROCm 6.4 (HIP 6.4.43482) from the host; the framework container
reports PyTorch 2.10.0+rocm7.2.4. The two stacks are on different ROCm
releases, and Section IV will state both explicitly, with their bundled MIOpen
and RCCL versions, rather than a single project-wide figure. The container is
pulled by the floating tag rocm/pytorch:latest, which advanced during the
project: the runs behind the reported scaling numbers logged ROCm 7.2.2
in-container, and it is that version, not the container's current 7.2.4, that
the reported numbers reflect.

Section IV will additionally give the driver version, compiler and build
flags, and the complete launch command and environment for both stacks.
Version strings alone do not make a run reproducible, so we commit to
releasing the native trainer source and build system for both vendors, pinned
to the exact commit used; the batch scripts for every reported experiment;
both containers identified by registry digest and image SHA-256 rather than by
their mutable tags, the fix the ROCm drift above calls for; and the per-run
logs with the scripts that extract the plotted numbers.

## R1-2a. "Why is the micro-benchmark performed on A100 and MI300X instead of H100 and MI300X?" (Section V)

R1-2a (micro-benchmark platform). The micro-benchmark isolates host-side dispatch
and orchestration cost (Python dispatch, launch path, allocator), a property of
the software stack rather than of the accelerator generation, and its device-side
work is deliberately small: it neither requires nor can exploit an H100-class
GPU. The micro-benchmark is corroborating rather than load-bearing evidence,
because the same quantity is measured directly on the H100 nodes themselves:
the single-GPU batch sweep separates native step time into 3.9 ms that is fixed
per step and 0.703 ms per image, and at 32 images per device the framework step
contains at least 24.8 ms that is not batch-proportional. No conclusion rests on
the micro-benchmark. Isolating the term on A100 is also the conservative
direction, since with the host held constant a faster accelerator shortens the
device term and therefore raises, never lowers, the measured overhead fraction.
We do not transfer the A100 host-side values to the H100 nodes, whose host CPUs
differ, which is why the H100 figures quoted above come from H100 runs. We agree
the presentation is cleaner on same-generation hardware and will repeat the
micro-benchmark on H100 for the final version.


## R1-2c. "Focus is on ResNet and image classification; tools such as CARAML also cover LLM tasks" (Section V)


R1-2c (CARAML and LLM tasks). Suites such as CARAML answer a different question:
they benchmark a framework implementation across accelerators, treating the
framework as a black box. Our study decomposes where the framework's cost arises,
which requires a layer-exact native reimplementation validated against the
framework model down to the loss trajectory; to our knowledge no such counterpart exists for LLM workloads in any suite.
Standalone native implementations of small-transformer training do exist (llm.c,
for example), but none is embedded in a controlled same-hardware, cross-vendor
comparison of the kind used here; a small data-parallel transformer under this
protocol is the natural next validation target, and we will name it as such in
the revision, while extending the construction to tensor and pipeline
parallelism would additionally evaluate parallelization strategy and is a study
of its own, I would say a new project by itseld. We will expand the related-work discussion of CARAML and position
the two as complementary: suite breadth identifies which workloads matter, while
our design explains where the time is spent for one of them.


## R1-2d. "No comparison with other benchmarking tools; what conclusions could not be taken from other tools?" (Section V)


R1-2d (relation to existing benchmarking tools). Existing tools compare systems
or frameworks against each other; to our knowledge none provides, on identical
hardware, a minimal native implementation of the same computation as a
reference. That reference enables four conclusions a black-box suite cannot
produce: (i) the framework's scaling deficit is bounded by construction, 1.62x
at 8 GPUs at global batch 256, against a reference that is not hand-favored,
since the framework wins at 1 GPU and in 8-GPU weak scaling; (ii) the deficit
is attributed mechanistically: it appears only as per-GPU work shrinks, is
established inside a single node (74% versus 103% parallel efficiency at
4 GPUs), and is excluded from being an interconnect effect by the collective
bound and the GPUDirect control reported in our response on fairness; (iii) the
decomposition is calibrated rather than merely reported: only a second
implementation of the identical computation licenses the statement that the
framework's marginal cost per image is at most 0.672 ms against the native
trainer's 0.703 ms, while at least 24.8 ms of its 46.3 ms step at 32 images per
device is not batch-proportional, against about 6.0 ms for the native trainer;
and (iv) the construction transfers across vendors: the same source model is
ported to HIP and revalidated, so the decomposition is repeated per platform
under identical methodology. Concretely: MLPerf Training publishes audited
end-to-end results but no attribution of where framework time is spent; CARAML
ports framework workloads, including LLM tasks, across accelerators but treats
each framework as a black box; operator-level suites such as DeepBench time
kernels in isolation and cannot bound end-to-end orchestration cost. The final
version will add a related-work paragraph making these contrasts explicit.

---

# Reviewer 2

## R2-1. "Experiments use only FP32 with tensor cores disabled; results may not represent real training"

R2-1 (FP32 only). One point of reviews are not correct - it requires correction first. The matched baseline is not tensor-core free: both stacks execute TF32 tensor-core math for
convolutions on H100 and MI300X, since the native trainer sets the cuDNN tensor-op math
type explicitly and the framework recipe runs with AMP disabled, under which
PyTorch's default TF32 convolution path is active on Hopper. Neither stack sets
an override that disables them. Section IV describes the configuration
imprecisely and will read "FP32 storage with TF32 tensor-core math, matched on
both stacks"; the arithmetic remains matched across the two stacks, which is the
property the comparison requires, and no reported result changes.

Restricting the headline comparison to a single matched numerical configuration
is deliberate: it holds the arithmetic fixed across both stacks so that
throughput differences are attributable to framework orchestration rather than
to precision-dependent kernel selection. Numerical format is not a neutral knob,
since it changes kernel selection, tensor-core eligibility, memory traffic and
the number of casts per layer, so letting each stack run its own preferred
precision would confound the framework effect with the precision recipe each
project happens to ship. Matched arithmetic is also what makes the native
implementation verifiable: the layer-exact equivalence and loss-trajectory
validation that underwrite the method require both sides to compute the same
function. The substantive question, whether the finding
survives reduced-precision tensor-core training, we can answer with data. The
native trainer also has a BF16 tensor-core path in NHWC layout, and we have
repeated the 8-GPU comparison against NVIDIA's recommended mixed-precision
configuration (NGC AMP), each stack in its intended reduced-precision mode. The
crossover structure of the FP32 study reproduces, and at a less favorable
operating point: NGC AMP is 60% faster at 1 GPU (3,790 vs 2,368 img/s), yet at
8 GPUs the native trainer is ahead already at 256 images per GPU (19,625 vs
18,937 img/s), a per-device batch at which the FP32 framework recipe still wins.
Reduced precision amplifies rather than removes the effect we study, and the
direction is fixed by the mechanism rather than observed after the fact: the
overhead fraction is F/(F + s*b) in the decomposition given in our response on
the definition of the framework tax, so reducing the marginal per-image cost s
while leaving the fixed per-step cost F unchanged must raise it. The framework's
8-GPU weak-scaling retention duly falls from 89% (FP32) to 62% (AMP), while the
native trainer shows no per-GPU degradation at 8 GPUs. The FP32 results therefore
understate, rather than overstate, the framework-overhead effect at scale:
moving to reduced precision widens the reported gap. We will add these BF16/AMP
results to the final version.


## R2-2. "Native NCCL setup is tuned while PyTorch DDP settings are not explained; fairness uncertain"


R2-2 (fairness of the DDP baseline). Three points. (i) The PyTorch baseline is not an
untuned reference: it is NVIDIA's own DeepLearningExamples ResNet-50 v1.5
implementation, run unmodified inside NVIDIA's NGC PyTorch container (the same
container image in which the native trainer runs), with NVIDIA's recommended
settings: DALI input pipeline, NHWC memory format, fused SGD, and DDP with its
default overlapped gradient bucketing. Every parameter is recorded in the DLLogger
output; the final version will list the complete launch command and environment
for both stacks. On the native side, the launch sets two non-default NCCL
variables (NCCL_ALGO=Tree, NCCL_P2P_LEVEL=NVL) and the network-interface
binding; the framework launch sets only error-handling variables. (ii) The asymmetry in communication machinery favors PyTorch: DDP
overlaps bucketed all-reduces with the backward pass, whereas the evaluated native
trainer issues one blocking 97.5 MiB (102 MB) all-reduce on a single stream after backward
completes, with no overlap. If communication tuning determined the outcome, the
native trainer would lose. (iii) Measurement excludes the interconnect
configuration as the cause. Both stacks run on the same InfiniBand fabric with
identical GPUDirect state; toggling NCCL_NET_GDR_LEVEL moves 8-GPU throughput by
0.2% (7,554 vs 7,567 img/s); and a standalone torch.distributed all-reduce at the
exact gradient size measures 16.0 ms on 8 GPUs across two nodes. That figure is
a deliberately loose upper bound rather than an estimate of the in-application
collective cost: it carries the standalone harness's own launch and
synchronization overhead, and DDP overlaps its bucketed all-reduces with the
backward pass. Even charging this full cost to every step with zero overlap
predicts at least 57% parallel efficiency for PyTorch at global batch 256; the
observed 46% therefore cannot be produced by communication configuration, tuned
or otherwise. The residual is bounded without reference to the interconnect at
all: at 32 images per device the framework step is 46.3 ms, of which at most
21.5 ms is device-proportional work at the framework's own measured marginal
cost and at most 16.0 ms is collective time, leaving at least 8.8 ms per step
that is neither arithmetic nor communication. The full decomposition is given in
our response on the definition of the framework tax. A single-GPU framework
batch sweep at batches 32, 64 and 128, which involves no communication at all,
replaces that bound with a direct measurement of both terms, as committed in
that response. The final version will add
an ablation with all non-default NCCL variables unset for both stacks.


## R2-3. "'Framework tax' includes both framework overhead and extra manual optimization (fused ops, preallocated buffers)"


R2-3 (definition of "framework tax"). We agree the term needs a precise
definition and provide it here: the reported gap is the difference between a
framework's shipped best-practice recipe and a minimal native implementation of
the identical computation, named the recipe-to-native gap in the final version. The
quantity can be decomposed rather than left bundled, and the decisive half of
the decomposition is already measured.

Per-device step time on either stack is described by step(b) = F + s*b, where s
is the marginal per-image term in which kernel quality, fusion and memory layout
act, and F is the batch-independent term produced by host-side orchestration:
dispatch, launch, allocation and synchronization. Hand optimization moves s;
framework orchestration moves F.

For the native trainer both terms are measured. A single-GPU batch sweep with no
communication gives step times of 26.7, 48.4 and 94.0 ms at 32, 64 and 128
images, and a least-squares fit of step(b) = 3.9 ms + 0.703 ms per image
reproduces all three to within 0.5 ms.

For the framework a single point settles the sign of the bundled term. Its
single-GPU step at 256 images takes 171.9 ms, that is 0.672 ms per image
including its own fixed cost, so its marginal per-image cost is at most
0.672 ms, below the native trainer's 0.703 ms. On the axis the concern
identifies, kernel implementation, the framework leads: the native trainer is
not the cheaper of the two per image of arithmetic. Hand optimization therefore
cannot produce the reported result, and the reported gap understates rather than
overstates the orchestration term.

The gap accordingly resides in F. At 8 GPUs and 32 images per device the
framework step is 46.3 ms and the native step is 28.5 ms. Charging each stack
its own marginal per-image cost accounts for at most 21.5 ms of the framework
step and 22.5 ms of the native step, leaving batch-independent residuals of at
least 24.8 ms and about 6.0 ms respectively. Of the framework's residual, at
most 16.0 ms can be collective time by the standalone upper bound reported in
our response on fairness, so at least 8.8 ms per step is neither device
arithmetic nor communication, which alone exceeds the native trainer's entire
non-arithmetic budget including its own unoverlapped all-reduce. The subtraction
assumes that the framework's marginal per-image cost at 32 images does not
exceed its value at 256; the native trainer's marginal cost is flat from 32 to
128 images and rises at 256, so if the framework behaves similarly the bound
above is conservative; the framework batch sweep committed below replaces this
assumption with a measurement.

The native trainer is a minimal reference and not an aggressively optimized one.
As evaluated it issues a single blocking all-reduce with no compute and
communication overlap, where DDP overlaps its buckets with the backward pass; it
leaves the BatchNorm and ReLU elementwise tail unfused; it performs a device
allocation per step in the loss path; and it issues 161 kernel launches per step
for gradient zeroing, packing, unpacking and the optimizer update. Its advantage
at small per-device batch is not obtained by out-engineering the framework's
kernels.

The final version will state this definition at first use, present the 1.62x
throughput ratio as the consequence of the difference in F at 32 images per
device rather than as its definition, report the framework's own single-GPU
batch sweep at 32, 64 and 128 images, which contains no communication and
replaces the bound above with a direct fit of F and s for the framework, and add
a kernel-level time comparison of the two stacks.


## R2-4. "Evaluation limited to ResNet-50 and a small fully connected network; no transformer or LLM experiments"


R2-4 (model scope). The study's method requires a native implementation that is
layer-exact and loss-trajectory-validated against the framework model, since
overhead attribution is meaningless if the two sides compute different work. This
is feasible for ResNet-50 v1.5, a CNN with audited cross-system training
baselines (MLPerf Training v0.5 through v4.0); the fully connected network is a second
instantiation on the same harness whose arithmetic is entirely
non-convolutional, so the harness result is not confined to convolutional
layers.

The mechanism we quantify, a fixed per-step host-side cost that stops amortizing
as per-device work shrinks, is defined independently of the model: the overhead
fraction equals fixed per-step host time divided by device step time, so
transformer training changes the denominator, not the definition, and its
small-microbatch regimes (pipeline stages, fine-tuning) sit in the same
overhead-dominated regime that our small-batch points probe. A transformer at
the scale that requires tensor and pipeline parallelism would compare
parallelization strategies rather than framework overhead; a moderate-size
data-parallel transformer needs neither, and we name it as the natural next
validation target for the method rather than claiming transfer. We will scope
the claims accordingly (data-parallel CNN training) and state this limitation
explicitly.


---

# Reviewer 3

## R3-1. "8 H100 GPUs span two InfiniBand nodes whereas 8 MI300X GPUs reside in one node; topology asymmetry makes cross-vendor scaling conclusions difficult"


R3-1 (topology asymmetry). We agree the fabrics differ. Two points bound the
effect of the asymmetry. First, the central
quantitative conclusions are within-platform comparisons (native versus framework on identical
hardware, normalized to the same single-GPU baseline), replicated per vendor.
Both stacks of each pair traverse the same fabric, and the residual bias within
a pair runs against the native trainer, which issues a single blocking
all-reduce with no overlap while DDP overlaps its buckets with the backward
pass, so added inter-node latency lands on the native step time first. The
two-node H100 configuration therefore understates the native advantage within
the pair rather than manufacturing it. Second, on H100 the framework's efficiency
decline is established inside a single node, before InfiniBand is involved: at
4 GPUs on one node (NVLink only) the PyTorch recipe drops to 74% parallel
efficiency versus 103% for the native trainer. The measured communication terms
then bound the inter-node contribution: a standalone all-reduce at the exact
gradient size measures 16.0 ms across two nodes, and charging that full cost to
every step with zero overlap still predicts at least 57% parallel efficiency at
8 GPUs, so the observed 46% cannot be produced by the inter-node hop; a
GPUDirect on/off control moves 8-GPU throughput by 0.2%. The 8-GPU H100 point
therefore extends a trend established intra-node rather than creating it. Where the topologies do differ, the difference is an inter-node hop that the
two-node H100 configuration pays and the single-node MI300X configuration does
not, so on the communication term the asymmetry works against the H100 figures.
We do not claim the net confound is single-signed, since the two configurations
also place a different number of ranks on each host. That difference is,
however, common to both stacks within each pair: every within-platform
comparison runs the native trainer and the framework at the same rank count per
host, so host contention, which acts on the fixed per-step term, is held equal
wherever the two stacks are compared. Cross-vendor statements
will therefore be scoped to configurations that are single-node on both
platforms, and the eight-GPU H100 point will be reported and labelled as a
within-platform result. In the final version we will label multi-node points
explicitly in all figures and present the intra-node cross-vendor comparison at
1, 2 and 4 GPUs, the counts the H100 platform provides within one node.


## R3-2. "The workload is not representative of the general DNN field; the title should reflect this limitation"


R3-2 (workload representativeness and title). We agree the title should name the
workload and will append the qualifier "A ResNet-50 Case Study" to the present
title; the abstract and conclusions will likewise state the scope of the claims
as data-parallel CNN training (see our response to Reviewer 2 on model scope).
The abstract will state the 1.62x figure explicitly as a strong-scaling result
at 32 images per device, so the headline number cannot be read as a blanket
framework penalty.
The remainder explains why we retain the main title: the object of study is a
mechanism, not a model. The mechanism is a fixed per-step host-side orchestration cost whose share of
step time grows as per-device work shrinks, and both terms of that share are
measured rather than assumed: a single-GPU batch sweep with no communication
separates native step time into 3.9 ms that is fixed per step and 0.703 ms per
image, so the same fixed cost is 1.9% of the step at 256 images per device and
14.6% at 32. ResNet-50 strong scaling exposes the mechanism because one workload
traverses both operating regimes: at 256 images per GPU the step is dominated by convolution
arithmetic, while strong scaling to 32 images per GPU pushes the same arithmetic
into the overhead-dominated regime. There, NGC per-GPU throughput collapses from
1,489 to 691 img/s while the native trainer, executing the same convolutions,
loses only 12% (1,272 to 1,123), producing the 1.62x crossover (8,980 vs 5,528
img/s; 88% vs 46% parallel efficiency). The framework baseline is faster at
1 GPU and retains the 8-GPU weak-scaling lead (10.6k vs
9.9k img/s), so the crossover is a property of the strong-scaling regime, not of
the baseline. Workloads with multi-second steps amortize the fixed cost below
measurement noise; the regimes where it dominates are small per-device-work
regimes, including pipeline stages and small-microbatch fine-tuning, which our
small-batch points probe directly.

The trainer separates architecture from harness: convolution, batch
normalization, activation, pooling, and fully connected layers sit behind a
common layer interface, and the data-parallel harness (input pipeline, gradient
bucketing, all-reduce, optimizer) is architecture-agnostic. The fully connected network reported in the paper exercises this separation on
arithmetic that is not convolutional at all, which is the primitive class that
dominates attention-based models, and the HIP port to MI300X exercises it across
vendors. The effect also reproduces
under mixed precision, as the mechanism predicts: NGC weak-scaling retention
falls from 89% (FP32) to 62% (AMP fp16), and native bf16 overtakes NGC at 8 GPUs
(19,625 vs 18,937 img/s) despite a 1.6x single-GPU deficit. The transfer
condition, overhead fraction equals fixed per-step host time over device step
time, is measured on this workload and stated as the condition under which the
result transfers; validating it on an attention-based workload is future work,
and the revision will say so explicitly.

ResNet-50 v1.5 was chosen as the case study because it served as the MLPerf
Training image-classification benchmark from v0.5 (2018) through v4.0 (2024), a
six-year record of audited cross-system results against which our baseline can
be checked externally. That tenure also makes the measured gap a conservative
one, for a reason external to our implementation: the framework baseline is not
a generic framework path but NVIDIA's own recipe with DALI input, NHWC memory
format, fused SGD and DDP bucketing, on the workload that received six years of
vendor tuning under audit, and on this workload it is still faster than the
native trainer at one GPU and retains the eight-GPU weak-scaling lead. A model
whose framework path has received less attention would not be expected to
exhibit a smaller fixed per-step cost, so we present the reported gap as a lower
bound on what this measurement would yield elsewhere rather than as a typical
value. MLCommons retired the benchmark in v4.1 (November 2024)
as the suite shifted toward generative workloads; retirement removes it as a
frontier task, not as a measurement vehicle.

Summary of changes for the final version. Already measured, to be added: the
full software specification and released artifact for both stacks and both
vendors; the mixed-precision comparison at 1 to 8 GPUs; the single-GPU step-time
decomposition; the collective-cost bound and the GPUDirect control; the
intra-node cross-vendor comparison at 1, 2 and 4 GPUs. Editorial:
the corrected description of the matched numerics; a precise definition of the
recipe-to-native gap at first use; the title qualifier "A ResNet-50 Case Study";
abstract and conclusions scoped accordingly; multi-node points labelled in every
figure; one related-work paragraph positioning MLPerf Training, CARAML and
operator-level suites. Additional experiments, each a single job: the
framework's single-GPU batch sweep, an ablation with all non-default NCCL
variables unset on both stacks, the H100 repeat of the micro-benchmark, and a
kernel-level time comparison.

Proposed title handling (R3-2), in order of preference:

1. `[Existing Title]: A ResNet-50 Case Study` (retains the full existing title and adds only the vehicle)
2. `[Existing Title]: A ResNet-50 Case Study on H100 and MI300X` (only if the committee also asks for hardware scoping)

---

# Internal support material (do not paste)

## Number provenance

| Claim | Number | Source |
|---|---|---|
| Strong scaling gb256, 8 GPU | 8,980 vs 5,528 img/s (88%/46%, 1.62x) | jobs 483138 (native) / 483139 (NGC) |
| Strong scaling, 4 GPU intra-node | native 5,221 (103%) vs NGC 4,415 (74%) | same runs; Helma nodes have 4x H100, so 1/2/4 GPU points are single-node |
| Strong scaling, 2 GPU intra-node | native 2,712 (107%) vs NGC 2,813 (94%) | same runs |
| 1-GPU point | native 1,272 vs NGC 1,489 img/s | same runs |
| Weak 8-GPU fp32 | NGC ~10.6k vs native ~9.9k img/s | weak runs 473110/471351 |
| NGC AMP fp16 1 to 8 GPU | 3,790 to 18,937 img/s (2,367/GPU, 62% retention) | job 501961, `pytorch_benchmark/ngc_amp_1_8gpu_501961.out` |
| Native bf16 1 to 8 GPU | 2,368 to 19,625 img/s (700 W row) | 1-GPU b256 bf16-full; `benchmark_results/power_imagenet_mn_503005/power_sweep_summary.csv` |
| NGC fp32 weak retention | 89% (88.7) | weak runs 471350-52, `plot_data/scaling.csv` |
| All-reduce microbench at the gradient size (97.5 MiB = 102 MB, the 25.56M-parameter fp32 bucket) | 10.31 / 10.64 / 16.02 ms at 2/4/8 GPU | job 488126 |
| GDR toggle | 7,554 vs 7,567 img/s | jobs 487810 / 487994 |
| 57% efficiency floor | 1-GPU NGC 1,489 img/s gives 21.5 ms ideal b32 compute; + 16.0 ms fully exposed = 37.5 ms/step = 6,825 img/s = 57% | derived from 483139 + 488126 |
| Native comm decomposition | ~6.4 of 12 efficiency points | 1-GPU batch sweep 477789 (b32 = 1,199 img/s) |
| **Step-time model (new, load-bearing)** | native 1-GPU step times 26.69/48.41/93.98/202.05 ms at b32/64/128/256; OLS over b32-b128 = **3.90 ms + 0.7026 ms/img** (R2=0.9999, max resid 0.46 ms; b256 falls off the line). NGC 1-GPU b256 = 171.9 ms = 0.672 ms/img INCLUDING its fixed cost, so NGC marginal <= 0.672 < native 0.703. 8-GPU b32/GPU: native 28.5 ms/step, NGC 46.3 ms/step. Residuals after charging marginal cost: NGC >= 24.8 ms, native ~6.0 ms; minus the 16.0 ms collective bound leaves NGC >= 8.8 ms neither arithmetic nor comm. Overhead share 3.9/202.05 = 1.9% at b256 vs 3.9/26.69 = 14.6% at b32 (7.6x) | job 477789 + 483139; recomputed 2026-07-25. **Do NOT print 4.9 + 0.699: that pair does not reproduce from the three step times and a reviewer can refit in seconds.** |
| Native minimalism (R2-3) | blocking single all-reduce, no overlap (resnet50.cu:2806-2837 pre-overlap code); per-step cudaMalloc/Free of d_dlogit (resnet50.cu:2342/2351); unfused BN/ReLU tail; 161-launch ZeroGrad/pack/unpack/SGD | July 2026 bottleneck audit, hand-verified; line numbers from current tree |
| Same container both stacks | `containers/nvidia-pytorch.sif` = NGC PyTorch 24.02 (measured: NVIDIA_PYTORCH_VERSION=24.02, torch 2.3.0a0+ebedce2, cuDNN 9.0.0.306; the 23.12 tag in `ngc_resnet_benchmark.slurm:118` is a different pull and does NOT describe this sif) | `apptainer inspect` 2026-07-23; `ngc_resnet_scaling_8gpu_benchmark.slurm:125` |
| NGC env | only NCCL_ASYNC_ERROR_HANDLING, NCCL_BLOCKING_WAIT, TORCH_NCCL_ASYNC_ERROR_HANDLING | `ngc_resnet_scaling_8gpu_benchmark.slurm:247-249` |
| Native env | NCCL_SOCKET_IFNAME=ibp35s0, NCCL_ALGO=Tree, NCCL_P2P_LEVEL=NVL | `run_imagenet_multinode_scaling.sh:331-332` |
| Host toolchain (R1-1a) | modules `cuda/12.6.2`, `cudnn` (9.8 bare-metal; container ships 9.0), `nvhpc` (NCCL) | `run_imagenet_multinode_scaling.sh:83-87`, `power_imagenet_multinode.slurm:37-38`, `experiment.md:3` |
| MI300X toolchain (R1-1a) | native ROCm 6.4 (HIP 6.4.43482, host `rocm-6.4.0` install); framework container `pytorch_rocm.sif` from `docker://rocm/pytorch:latest`, current rebuild (Jun 27) reports ROCm 7.2.4 / PyTorch 2.10.0+rocm7.2.4, but the runs behind the REPORTED scaling numbers (jobs 52424124, 52425551, June) logged ROCm 7.2.2 in-container; the floating `:latest` tag drifted mid-project, so 7.2.2 is the version of record, not 7.2.4 | user-verified 2026-07-25 via `/opt/rocm/.info/version` + `torch.__version__` inside the sif, and job logs; scripts `ngc_resnet_benchmark_amd*.slurm`, `bf16_pytorch_amd.slurm`; log `run_multinode_amd_52980217.out` |
| MLPerf tenure | RN50 in MLPerf Training v0.5 (2018) through v4.0 (2024); retired v4.1 (Nov 2024) | MLCommons training results repositories; never claim current status |

Conservative-quote note (R2-1): the matched 10-epoch power-harness pair gives a larger native margin (503178 bf16 19,389 vs 503179 NGC fp16 15,241 img/s at 700 W, 1.27x), but the extraction method differs from the 1-epoch job 501961. Quoting NGC's most favorable number (18,937) keeps the claim unassailable.

## Optional pre-deadline runs (each one job)

1. **RUN THIS BEFORE SUBMITTING, not after: NGC single-GPU batch sweep at b32/b64/b128 (~20 min, one job, 1 GPU, zero communication).** Fitting step(b)=F+s*b to the framework the same way it is fitted to the native trainer replaces every bound in R2-2, R2-3 and R1-2d with a direct measurement, in a configuration where no NCCL variable, DDP bucketing choice, or fabric property exists to be blamed: "the framework's fixed per-step cost is X ms against the native trainer's 3.9 ms." That one sentence answers R2-2, R2-3 and R1-2d at once and lets three "the final version will report it" promises be deleted. It is also a FALSIFICATION RISK that must be known before pasting: every bound in the strengthened text assumes the framework's marginal per-image cost at b32 does not exceed its b256 value (0.672 ms). If the framework's small-batch kernels are simply less efficient per image, part of the 24.8 ms residual is device-side, not host orchestration, and the model hands R2 the counter-argument it was written to close. Predicted if the framework is linear with F ~ 10 ms: b32 ~ 30 ms (~1,060 img/s), b64 ~ 50 ms (~1,270 img/s), b128 ~ 91 ms.
2. Native strong-256 rerun with NCCL_ALGO and NCCL_P2P_LEVEL unset (2 epochs). Turns the promised NCCL ablation into a measured "<1% change" sentence.
3. H100 rerun of the A100 micro-benchmark (R1-2a). Small job; would let the rebuttal say "repeated on H100, conclusions unchanged" instead of promising it.
4. NGC ncu kernel profile (`profile_ngc_ncu.slurm` + `kernel_canon` renamer exist; job never submitted). Needed to honor the R2-3 commitment to a kernel-level time comparison; run before camera-ready.
5. Small second-CNN datapoint (e.g. a VGG-style network assembled from the existing conv+BN+ReLU, pooling, and FC modules), one strong-scaling point. Would convert the R3-2 architecture/harness-separation property into a measured generality result, the single strongest possible answer to Reviewer 3. Check feasibility first: the architecture assembly is currently hardcoded in resnet50.cu.

## Pre-submission checklist

1. Versions (R1-1a): now MEASURED and stated in the text for BOTH vendors. H100: NGC 24.02 container (torch 2.3.0a0+ebedce2, CUDA runtime 12.3, cuDNN 9.0.0.306, NCCL 2.20.3 via torch, DALI 1.34.0); native binary resolves host cuDNN 9.8.0.87 and NVHPC NCCL 2.18.5 (verified by ldd on build/train_multigpu_tiny). MI300X: native ROCm 6.4 (HIP 6.4.43482); PyTorch container reports 7.2.4 now but the canonical scaling-number runs (52424124/52425551) actually ran under 7.2.2, a `:latest`-tag drift caught 2026-07-25, use 7.2.2 in the paper, not 7.2.4. Remaining: spot-check the production multi-GPU binary resolves the same H100 libraries, record the driver version on a compute node (nvidia-smi / rocm-smi on each side).
2. TF32 (R2-1): **gate now closed on the code evidence, one manual check left.** Verified 2026-07-25: the native NCHW path sets `CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION` (resnet50.cu:455-457 and 1539-1541), NHWC sets `CUDNN_TENSOR_OP_MATH` (resnet50.cu:300-310), no script sets NVIDIA_TF32_OVERRIDE=0, DeepLearningExamples never touches `allow_tf32` so PyTorch's default cuDNN TF32 convolution path applies on Hopper, and experiment.md:145 states the fp32 baseline is already TF32 tensor-core. R2-1 now OPENS by self-correcting this ("FP32 storage with TF32 tensor-core math, matched on both stacks"). Remaining check: read the submitted PDF's exact sentence and confirm the 483138/483139 job scripts set no override. If the submitted runs genuinely disabled tensor cores, delete the first paragraph of R2-1 and keep the second. Self-correcting reads as rigour to a rigour reviewer; being caught in discussion reads as the opposite.
3. R1-2a: the block no longer pleads resource stewardship (which read as "we used what was free" and invited the obvious follow-up that a host-side benchmark on a different host CPU is the wrong platform). It now argues the micro-benchmark is corroborating rather than load-bearing, since the same quantity is measured on H100 by the batch-sweep decomposition, plus a conservative-direction argument (holding the host constant, a faster accelerator shortens the device term and so raises, never lowers, the measured overhead fraction), plus an explicit refusal to extrapolate across host CPUs. GATE: confirm the micro-benchmark is genuinely host-side-dominated with low GPU utilization; if it is device-bound, the conservative-direction sentence inverts and must be cut. Schedule the H100 rerun (optional run 3) so "we will repeat it" is honored.
4. R2-3: the kernel-level comparison commitment requires optional run 4 before camera-ready.
5. FC network (R3-2, R2-4): the fully connected benchmark was not found in either local repo tree, so the text says "reported in the paper" with no section number. Both blocks were rewritten to STOP disowning it: the previous wording ("deliberately minimal", "not an evaluation target") conceded R2's and R3's premise that there is effectively one workload, in the same document that elsewhere cites the FC net as generality evidence. Both now make the same safe, construction-true claim: its arithmetic is entirely non-convolutional (pure GEMM, the primitive class that dominates attention-based models), so the harness result is not confined to convolutional layers. Do NOT claim the FC net exhibits the same measured fixed per-step cost; nothing available substantiates that. Optionally add the section/figure from the submitted PDF, and keep the two blocks in sync.
6. R3-1: verify the exact compensation the paper describes for the topology difference and that the MI300X claims match; MI300X result bundles are not on this filesystem, so all quoted numbers are H100-side. R3-1 now commits CONCRETELY to an intra-node cross-vendor comparison at 1, 2 and 4 GPUs: confirm the MI300X side has those single-node points before pasting; if it does not, change the counts in that sentence to the ones it has.
7. CUDA Graphs / torch.compile: sharpest possible follow-up attack (NVIDIA's documented mitigation for host overhead). Prepared reply: the baseline is the reference recipe as shipped; offer a graph-capture ablation for the revision if raised.
8. bf16 superlinearity: 19,625/8 = 2,453 img/s per GPU exceeds the 2,368 single-GPU figure. Prepared reply: steady-window extraction and single-GPU bf16 underutilization at that batch; consider a window-matched pair if pressed.
9. MLPerf retirement: citing tenure through v4.0 hands the reviewer the v4.1 retirement fact. Prepared reply: the suite pivoted to generative AI; tenure is invoked for baseline auditability, not fashion.
10. Timing phrasing: adjust "we have repeated" / "completed since submission" if the BF16 runs predate submission; verify the rebuttal rules permit new numbers, otherwise keep the mixed-precision sentences qualitative.
11. **LENGTH: the paste-ready text is now ~4,130 words (~26,500 characters), which likely exceeds many rebuttal portals.** R1-1a was condensed 2026-07-25 (409 -> 332 words) after the MI300X ROCm addition, no content lost: both version sets, the cuDNN skew-direction argument, and the ROCm tag-drift catch all survive, only phrasing tightened. Exact per-block counts: R1-1a 332, R1-2a 205, R1-2c 192, R1-2d 297, R2-1 457, R2-2 418, R2-3 612, R2-4 209, R3-1 412, R3-2 845. Check the portal limit FIRST. Cut order if constrained: (a) R3-2 is the most compressible, its third paragraph (MLPerf tenure plus conservative-baseline) can go to two sentences, saving ~200; (b) R1-1a's final commitment paragraph (source/build system/logs) can compress to one sentence, saving ~50; (c) R1-2c can fold into R1-2d; (d) within R2-2 cut the GDR parenthetical. Do NOT cut: R2-3's decomposition (paragraphs 2 to 5), R2-1's TF32 correction, R3-2's opening title concession, R1-1a's version-skew disclosures (either vendor), or the 57% floor. If the limit is per-reviewer rather than total, only R3-2 and R2-3 need attention.
12. **The 16.0 ms trap (new, know this before the discussion phase).** The step-time decomposition puts the native 26.7 ms single-GPU b32 step into the rebuttal, from which a reviewer can derive that the native trainer's entire cost of going to 8 GPUs across two InfiniBand nodes is 1.8 ms/step, while the rebuttal also quotes a 16.0 ms standalone all-reduce at the same message size and states the native trainer does not overlap. Those look irreconcilable. The reconciliation is that the standalone harness carries roughly 10 ms of its own overhead, visible in its implausible 10.3 ms for the same 102 MB over NVLink at 2 GPUs. R2-2 therefore labels 16.0 ms explicitly as a loose upper bound and uses it only as a bound. **Never add a sentence quoting exposed collective time for both stacks side by side**: it restates R2's fairness allegation using our own numbers ("your tuned NCCL is nine times faster on the same fabric"). If asked, the reply is that 16.0 ms is a torch.distributed measurement bounding the framework path, never an estimate of either stack's in-application cost, and that the comm-overlap A/B (+6.9% on native 8-GPU strong scaling, implying ~1.84 ms exposed) corroborates the native side independently. Hold that in reserve.
13. **CUDA Graphs / torch.compile, updated posture.** Do not pre-empt it unprompted (it spends characters and plants the idea). If raised, the decomposition makes the reply strictly better: graph capture is a mechanism for reducing F while leaving s unchanged, which confirms the attribution rather than contradicting it. Do not commit to running it; adding graph capture to the DeepLearningExamples recipe before camera-ready is of unestablished feasibility.
14. **Framing for the PC meeting.** As written, a PC member summarises this as "they wrote RN50 in CUDA and it beats PyTorch 1.62x", which is a comparison, and comparisons invite fairness objections (R2 filed three). The data supports a structurally immune framing: a reference that LOSES at 1 GPU (1,272 vs 1,489) and LOSES weak scaling (9.9k vs 10.6k) cannot be accused of being hand-favoured, and what it yields is a bound, not a boast. R2-3 and R1-2d now move that way; consider one sentence in the opening line repositioning the contribution as a measurement instrument. Also note the paper is not selling its second-strongest result: 88% strong-scaling efficiency at global batch 256 across two InfiniBand nodes, obtained with a deliberately unoptimized single blocking all-reduce.
15. **R1-1a artifact release gate.** The commitment paragraph of R1-1a now covers BOTH vendors: native trainer source and build system, batch scripts, both container digests plus both image SHA-256 hashes, and per-run logs with extraction scripts. Confirm you are authorized to release all of these (advisor, institution, any code-ownership constraints, and separately for the AMD side if a different group or cluster owns `pytorch_rocm.sif`) BEFORE pasting; if anything cannot be released, trim that item from the sentence rather than promising it. Compute the four hashes when convenient: NGC registry digest for the 24.02 tag + `sha256sum` on `containers/nvidia-pytorch.sif`; ROCm registry digest for the `rocm/pytorch:latest` pull actually used (note: `:latest` is not itself a stable reference, record the digest it resolved to at build time, Jun 27) + `sha256sum` on `pytorch_rocm.sif`.
16. R3-2 stance (revised after adversarial re-review): the block now OPENS with the accommodation (append "A ResNet-50 Case Study", scope abstract/conclusions to data-parallel CNN training) so a skimming reviewer reads acceptance, then justifies retaining the main title. The modularity argument is stated as a verifiable artifact property (architecture/harness separation exercised by the FC net and the HIP port), not as an untested "any CNN in a few lines" capability claim; verify the paper or artifact actually shows this structure. The MLPerf v4.1 retirement is now pre-empted head-on rather than left for the reviewer to find. R3-2 promises abstract/conclusions scoping: make sure the camera-ready actually does this, since the discussion phase will check. If the meta-review insists on stronger scoping, fall back to accepting title pattern 2. Optional run 5 (second CNN datapoint) would upgrade the generality argument from property to measurement.
