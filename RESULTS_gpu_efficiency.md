# ResNet-50 on MI300X: custom HIP vs PyTorch — GPU-efficiency analysis

**Config (all numbers):** 1× MI300X (gfx942), ResNet-50 v1.5, ImageNet-1K
(224×224, 200-class subset — per-step conv shapes identical to the full set),
global batch 256, FP32 (no AMP), MIOpen FIND_MODE=3. Custom trainer =
`train_multigpu` (HIP/MIOpen/RCCL); reference = PyTorch-ROCm 2.10 (torchvision
ResNet-50, NCHW). GPU activity from `rocprofv3 --kernel-trace`; occupancy =
union of the device's hardware-queue busy intervals over wall time, steady
state (warm-up excluded).

## Headline

| Runtime | Throughput (img/s) | GPU busy (steady) |
|---|---:|---:|
| PyTorch-ROCm | 1497 | ~98% |
| Custom HIP | 1204 | ~84–90% |

The custom trainer reaches **0.80×** of PyTorch (a **1.24× gap**). Crucially,
**both runtimes keep the GPU busy** (≥84%), so the gap is **not** launch/idle
("dispatch-bound"). The MI300X is compute-bound under both; the difference is
how much useful work each kernel does per unit of GPU-busy time.

> Correction note: an earlier internal trace suggested the custom trainer was
> ~65% idle / dispatch-bound. That trace was mislabeled — it ran 8-GPU at
> per-GPU batch **32**, where tiny kernels cannot hide launch latency. At the
> production batch (256) the device is 84–90% busy; the dispatch-bound reading
> was a small-batch artifact.

## Gap decomposition (Fig. levers)

Using two env-gated levers in an isolated fork — bypassing the input pipeline
(`RN50_SYNTHETIC_DATA`, reuse one batch = the pure compute ceiling) and fusing
the residual add+ReLU to remove its device-to-device copy
(`RN50_FUSED_RESIDUAL`) — the 1.24× gap decomposes multiplicatively:

| Lever | img/s | Δ |
|---|---:|---:|
| base (real data, unfused) | 1204 | — |
| + data-pipeline overlap (synthetic) | 1290 | **+7.1%** |
| + residual fusion | 1334 | **+3.4%** |
| **HIP compute ceiling** | **1334** | |
| remaining → PyTorch | 1497 | **+12.2%** |

The two implementation levers together recover ~11%; a **~12%
kernel-efficiency gap remains** (PyTorch's fused/better-tuned conv and
elementwise kernels), which is the dominant residual term.

## Why fusion barely moves throughput (Fig. gpu_idle)

The residual-add fusion removes exactly the per-block residual copies
(~16k `copyBuffer` dispatches over the run = 16 bottlenecks × ~1k steps), but
those are only **~4%** of the **~376k** `copyBuffer` dispatches the trainer
issues; GPU occupancy is unchanged (83.9%→83.6%). At batch 256 these copies are
largely *hidden* behind compute, so removing them yields little. The remaining
~360k device copies (per-layer activation stashing for the hand-written
backward, layout transposes) are the larger structural overhead and the natural
next target.

## Figures
- `plots/mi300x/rn50_pt_vs_hip.{pdf,png}` — measured throughput + side-by-side GPU occupancy.
- `plots/mi300x/rn50_levers.{pdf,png}` — 2×2 lever matrix + gap-decomposition waterfall.
- `plots/mi300x/rn50_gpu_idle.{pdf,png}` — HIP occupancy, idle-gap spectrum, and per-kernel idle attribution at batch 256.

## Reproduce
```
sbatch hyp_imagenet_exp_amd.slurm          # native + PyTorch throughput & traces (GPUS, USE_FULL knobs)
sbatch hyp_levers_exp_amd.slurm            # 2x2 lever matrix + base/fused traces
python3 plot_pt_vs_hip.py                   # Fig. pt_vs_hip
python3 plot_levers.py                      # Fig. levers
python3 plot_gpu_idle.py                    # Fig. gpu_idle  (auto-picks busiest agent)
```

## Profiling-methodology caveat (rocprofv3 + PyTorch)
`rocprofv3` collects in-process; its interceptor is inherited by the PyTorch
DataLoader's forked workers (which do no GPU work) and **segfaults on long
runs**. Running with `--workers 0` avoids the crash but **serializes decode**,
starving the GPU (a 300-step trace ran 98 min at 0.9% busy) — so it is useless
for occupancy. The valid recipe: a **short run with workers on** (prefetch
active), traced with `--kernel-trace --marker-trace` and `roctx` TrainStep/op
ranges (`--profile-steps`/`--profile-emit-ops`), then occupancy computed
**per-TrainStep** over the steady steps. This both names PyTorch's GPU kernels
(via the marker→kernel correlation) and yields the 98% figure.
