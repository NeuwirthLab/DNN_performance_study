# Repo orientation

ResNet50 performance-analysis project on AMD MI300X: a hand-written native HIP/CUDA
trainer is benchmarked and energy-profiled against a stock PyTorch DDP trainer, feeding
an energy/power-modeling paper (HiPC 2026 target — see memory `project_hipc2026_target`).
This file is a **navigational map**, last surveyed 2026-07-03. It documents where things
live, not how to build/run them (no top-level README/CMakeLists exists — see below).

The repo root is cluttered with hundreds of one-off SLURM job dirs/caches from ~2 months
of experiments. Treat anything not listed below as ephemeral job output unless proven
otherwise.

## Core trainer source (root)

| File | Role |
|---|---|
| `resnet50.cu` / `resnet50.cuh` | Current ResNet-50 model (fwd+bwd): MIOpen/cuDNN-style conv/BN/pool, hipBLAS/cuBLAS FC, custom kernels for ReLU/avgpool/loss/SGD. CUDA syntax; `-DUSE_HIP` routes through `hip/gpu_compat.h` + `hip/dnn_compat.h` shims. |
| `resnet50.cu.regressed_0627` / `.cuh.regressed_0627` | **Legacy backup** snapshots (~1700-line diff from current) taken around a June 27 regression. Kept for diffing only — do not build from these. |
| `train_multigpu.cu` | Primary single-node multi-GPU DDP trainer (NCCL/RCCL AllReduce, one process/thread per GPU). Loader selected via `USE_DALI`/`USE_ROCAL` ifdefs, else `mt_image_loader.h`+`prefetch_loader.h`. |
| `train_multigpu_tiny.cu` | Same DDP pattern for Tiny-ImageNet (64×64), uses `tiny_dataloader.h`. Side-track, not the main line. |
| `train_multinode.cu` | Multi-node sibling of `train_multigpu.cu` (SLURM `srun`, RCCL bootstrap via shared-file `ncclGetUniqueId`, no MPI). **Default (no `USE_ROCAL`) is now the working/fastest path** — CPU turbojpeg via `PrefetchLoader<MTImageLoader>`, ~2.1× faster than rocAL (2780 vs 1327 img/s, job 52946477); `USE_ROCAL` defaults OFF in `hip/CMakeLists.txt` and is opt-in only, gated by a known multi-GPU thread race (see memory `project_multinode_cpu_starvation`, `project_rocal_gpu_remaining_issues`). This reverses the older `project_imagenet_zipmode_loader`/`project_multinode_gap` guidance — the file's own header comment calling `prefetch_loader.h` "absent from the repo" is stale, the file exists at root. |
| `imagenet_loader.h` | Base CPU ImageNet loader (dir or extracted-set mode), wraps `stb_image.h`/`stb_image_resize2.h`, optional `USE_TURBOJPEG`. |
| `mt_image_loader.h` | Multi-threaded decode wrapper around the above. Explained in `MT_IMAGE_LOADER_EXPLAINER.md` (architecture) and `MT_IMAGE_LOADER_TALK.md` (talk script). |
| `rocal_imagenet_loader.h` | rocAL/MIVisionX GPU-accelerated loader, gated by `#ifdef USE_ROCAL`; CPU/GPU backend toggle via `ROCAL_BACKEND` env var. See memory `project_rocal_WORKING_fix`, `project_rocal_gpu_remaining_issues`. |
| `prefetch_loader.h` | Generic double-buffered background-prefetch template wrapper over any CPU loader. |
| `tiny_dataloader.h` | Tiny-ImageNet-specific loader. |
| `bench_loaders.cu` (+ `bench_loaders.slurm`) | Standalone throughput bench: MTImageLoader (turbojpeg) vs rocAL loader. |
| `pytorch_resnet50_ddp.py` | Reference PyTorch DDP trainer — the comparison baseline. |
| `stb_image.h`, `stb_image_resize2.h` | Vendored public-domain single-header image decode/resize libs. |
| `miniz.c` | **Dead code** — placeholder restored 2026-06-27 after original was deleted; nothing references its symbols; kept only so a legacy `extract_coefficients.slurm` build step still links. |
| `kernel_canon.py` | Renames MI300X kernel labels to H100 canonical names for cross-GPU roofline plots. |
| `nhwc_bench.py` | Small NCHW-vs-NHWC conv timing microbenchmark. |

**Build system:** no root `CMakeLists.txt`/`Makefile`/`README.md`. Real build def is
`hip/CMakeLists.txt` (project `ResNet50_HIP`, HIP lang, default `gfx942`, sources pulled
from repo root via `SRC_ROOT=../`). `hip/` also holds the compat shims
(`dnn_compat.h`, `gpu_compat.h`, `turbojpeg_compat.h`).

Build output dirs (all from `hip/CMakeLists.txt` unless noted):
- `build-bf16-in1k/`, `build-stb/`, `build-tj/`, `build_profile_in1k/` — tuning/path variants of `train_multigpu`, each with a successful binary.
- `build_hyptest/` — **no working binary**. Its only build attempt targeted `train_multigpu_tiny` (not `train_multigpu`) and failed to compile (`gmake Error 2`); only `libminiz_lib.a` succeeded. Don't expect a runnable trainer here.
- `build_profile/` — also has stale leftovers `train_multigpu.pre_nhwc`, `train_multigpu.stale_0604` from an old NHWC refactor.
- `.build_multinode/` — builds `train_multinode`; has SLURM rank-launch scripts.
- `build-exp/` — **different project**, root is `experiments/findmode` (`ResNet50_HIP_EXP`), builds `train_multigpu_exp`/`train_multigpu_tiny_exp`/`ringbuf_concept_test` — an experimental sandbox, not the main line.

## Analysis, plotting & paper-writing subsystem

- `power_analysis/` — main power/energy harness (`gpu_power_sampler.sh`, `measure_power*.sh`, sweep/probe `.slurm`) + modeling (`analyze_power.py`, `compose_energy.py`, `fit_dvfs_cubic.py`, `additivity_analyze.py`) + in-progress HiPC 2026 draft (`.tex`, `.bib`, `HIPC2026_*.md`). `HIPC2026_gap_plan.md` tracks reproduced-vs-overstated claims. `results/` holds raw sweep output + `ENERGY_RESULTS.md`.
- `paper_data/` — portable bundle for scaling + per-kernel figures only (`scaling.csv`, `kernels.csv`, `plot_scaling.py`, `plot_kernels.py`, own `paper_style.py`). Has a README/manifest. See memory `project_paper_data_provenance`.
- `plot_bundle/` — **the master figure bundle**: one self-contained subfolder per paper figure (README maps each to its script + SLURM job provenance). Subfolders: `arch_diagrams/`, `gpu_idle/`, `hip_vs_pytorch/`, `kernels/`, `levers/`, `mi300x_master/` (combined 11-figure script), `power_energy/`, `reference/` (canonical CSVs), `roofline_raw/`, `roofline_svg/`, `scaling/`. See memory `project_plot_bundle`.
- `nccl_and_rccl/`, `imagenet1k_profile_bundle/`, `fig_traffic_decomposition_bundle/` (+ frozen `fig_traffic_decomposition_bundle.tar.gz` snapshot) — same self-contained-bundle pattern, one per figure family. `imagenet1k_profile_bundle/README` flags a real caveat: PyTorch PMC data is 224² only, native PMC data is 64² only.
- `rocal_research/` — rocAL/gfx942 decode investigation; subdirs are symlinked ImageNet subsets + benchmark binaries + vendored RPMs (`rpms/`), plus newer `gate1_smoke`/`gate2_smoke` binaries for the GATE-1/GATE-2 honesty-test phase (see SLURM table below). Tied to `SYSADMIN_TICKET_mivisionx_gfx942.md` (root).
- `tools/` — analysis grab-bag: `compare_roofline_cache.py`, `coeff_errorbars.py`, `residual_decomposition.py`, `roofline_single_column.py`, `analyze_bf16_trace.py`, plus a HIP microbenchmark (`component_loop.cu`).
- `scripts/` — just ImageNet setup helpers (`setup_imagenet.sh`, `reorganize_val.py`).
- `paper_style.py` + `styling.md` — **canonical figure-styling source of truth**; every plot script should call `paper_style.setup()`. See memory `feedback_figure_styling`.
- **Superseded by `plot_bundle/`:** `plots/` (subdirs `amd_mi300x`, `mi300x`, May–Jun timestamps) and `figures/` (subdirs `compare_in1k_224`, `traffic_decomposition_amd`) — older flat render dumps, content overlaps the maintained bundles.
- Root-level loose scripts (`analyze_roofline.py`, `plot_roofline.py`, `plot_gpu_idle.py`, `plot_head2head.py`, `plot_levers.py`, `plot_pt_vs_hip.py`, `plot_pt_vs_native.py`, `plot_activation_sparsity.py`, `fig_comm_overhead_amd.py`, `fig_rccl_overhead_strong_weak.py`, `fig_traffic_decomposition*.py`, `amd_to_ncu_raw.py`, `resnet50_mi300x_plots.py`) — working copies that mirror scripts already frozen into the bundles above; check the bundle first before editing the root copy.
- `RESULTS_gpu_efficiency.md` — headline result doc (native HIP trainer ≈0.80× PyTorch-ROCm throughput, both compute-bound).

## SLURM jobs, logs, and experiment results

~65 `*.slurm` files at root, grouped by prefix:

| Prefix | Purpose |
|---|---|
| `bf16_*` | bf16 mixed-precision training/smoke (native + PyTorch) |
| `hyp_*` | "Hypothesis" experiments isolating perf levers (residual fusion, MIOpen find-mode, ring buffers, RCCL, data pipeline) |
| `ngc_resnet_benchmark*` | NVIDIA NGC container ResNet50 reference baseline |
| `ncu_*` | Nsight/rocprof compute profiling launches |
| `profile_*`, `rocprof_compute_*`, `rocprofv3_*` | rocprof-based profiling passes (native vs PyTorch, MFMA, L1-focus) |
| `rocal_*`, `probe_rocal_container.slurm`, `verify_rocal.slurm` | rocAL data-loader testing/tuning |
| `run_multigpu*` / `run_multinode*` | Multi-GPU / multi-node distributed training launches |
| `scaling_test_amd.slurm` | GPU-count scaling sweep driver |
| `smoke_*` | Quick sanity runs before full jobs |
| `test_*` | Isolated micro-tests of specific kernels/ops (bf16 allreduce, maxpool, NHWC fusion, rocAL diag) |
| `validate_*` | Post-change correctness validation for bf16 |
| `captest_gaod0001.slurm`, `powercap500_probe.slurm` | Power-capping probes |
| `rocjpeg_perf.slurm`, `build_check.slurm` | rocJPEG decode perf; build sanity check |
| `gate1_arm0.slurm`, `gate2_rocal_phase.slurm`, `rocjpeg_gate1.slurm` | Newest group (added 2026-07-03) — staged rocJPEG/rocAL honesty-test phases (GATE-1 host-vs-container turbojpeg A/B, GATE-2 decode/augment/copy phase decomposition), successor to the GATE-0 retraction in memory `project_rocjpeg_vcn_gate0`; matching binaries in `rocal_research/gate1_smoke`, `gate2_smoke` |

`logs/` subdirs (job stdout/stderr, by category): `bf16`, `multigpu`, `multinode`,
`profiling` (largest by file count and size), `rocal`, `scaling`, `misc` (catch-all).
`hyp/` and `smoke/` exist but are unused placeholders — hyp-prefixed job logs actually
land in `misc/` only (not `profiling/`), and smoke logs land in `smoke_test_results/`.

`profiles/` — 58 job-ID-suffixed dirs plus one stray non-directory `log.txt` and one
non-suffixed `perfmon/` dir. Prefix key: `amd_*`/`native_in1k_*` = native
trainer rocprof runs, `pt_amd_*`/`pt_bf16_*` = PyTorch rocprof runs, `hyp_in1k_*`/
`hyp_matrix_*`/`hyp_trace_*` = hypothesis-test profiling, `bf16_in1k_*`/`bf16_prof_*` =
bf16-specific, `v3direct_*` = rocprofv3 direct-counter mode, `rcc_native_*` = RCCL+native
combined, `validate_bf16_*`/`validate_pt_bf16_*` = post-fix validation passes.

Other result dirs: `benchmark_results/` (NGC container outputs), `scaling_logs_<jobid>/`
(5, each with a matching root `scaling_results_<jobid>.csv`), `rocal_build_logs_<jobid>/`
(17 dirs, rocAL compile attempt logs), `experiments/findmode/` (source for the
`build-exp/` sandbox, not logs), `smoke_test_results/`, `build_check_results/`,
`_archive/` (manually archived superseded power-analysis scratch work — check here before
recreating something that looks already solved). One-off MIOpen find-db test dirs
`bf16ar_test_*`, `flip_test_*`, `mpbf16_test_*`, `nhwc_test_*` each hold `db_*` subdirs
of isolated kernel-cache test output.

Stray root-level `.out`/`.err` files (`rn50_rccl_bench_*`, `rn50_validate_bf16_*`,
`run_multinode_amd_52942400.*`) are SLURM output that escaped `logs/` — harmless clutter.

`ncu_comparison_latest/` — per-metric NCU/rocprof comparison output (`native_amd/`,
`pytorch_amd/`, `per_metric_compare/`) tied to the roofline/traffic-decomposition work
(`tools/compare_roofline_cache.py`, `fig_traffic_decomposition*.py`); not itself a
bundle, just labeled results.

`dataloader_research/` — early rocJPEG/rocAL loader-compile prototype
(`rocjpeg_imagenet_loader.h`, `test_rocal_compile.cu`, `test_rocjpeg_compile.cu`),
precursor to the more developed `rocal_research/` — check here before re-prototyping
a rocJPEG loader header.

**Empty/unused:** `h100_tests/`. `checkpoints/resnet50_container/` (subdir only) is
empty — but `checkpoints/` itself is NOT empty: it holds ~8.1GB of real content
(35 multigpu + 5 multinode epoch checkpoints, `resnet50_multigpu_best.bin`,
`resnet50_multinode_best.bin`, `training_log_multigpu.csv`, `training_log_multinode.csv`).

## Datasets & vendored third-party code

- `ImageNet-1K/` — real Kaggle ImageNet-1K download (own `.git`); `competitions/imagenet-object-localization-challenge/` unpacked to `ILSVRC/Data/CLS-LOC/{train,val}` per Kaggle layout. Also holds the original 155GiB source zip.
- `imagenet/` — `train`/`val` symlinks straight into the `ImageNet-1K` tree above. Convenience alias only.
- `imagenet_subset_4/`, `imagenet_subset_8/`, `imagenet_subset_200/` — symlink-only N-class quick-test subsets into the same tree.
- `tiny-imagenet-200/` — standard Stanford Tiny-ImageNet-200 layout, but **currently empty** (0 images across all 400 train/val class dirs; the 14G `du` size is just leftover empty-directory-block overhead, `--apparent-size` is 7.6M). Data was removed at some point after the rocAL reshape step (`.rocal-reshape.done` marker present, source zip no longer at root) — don't assume this dataset is usable without re-fetching it.
- `.imagenet_valset/` — hidden, 14G total, **materialized (real files, not symlinks)**, split into `train/` (1000 class dirs, 50,000 JPEGs, 6.7G) **and** `val/` (1000 class dirs, 50,000 JPEGs, 6.7G) — note both halves exist, not just a val-set; neither is the full 1.28M-image ImageNet train set. Distinct from the lightweight symlink-only subset dirs above.
- `imagenet_results/` — just 3 pairs of stray job logs, not a dataset despite the name.
- `DeepLearningExamples/` — genuine clone of NVIDIA/DeepLearningExamples. Only `PyTorch/Classification/ConvNets/resnet50v1.5/` is actually used; everything else (TensorFlow, JAX, Kaldi, MxNet, PaddlePaddle, FasterTransformer, etc.) is unused monorepo baggage.
- `MIVisionX/` — genuine clone of AMD/MIVisionX. rocAL itself is **not** vendored as a full subtree here (only a README pointer + a `rocal_setup.py` helper) — the real rocAL work happens via `rocal_sandbox/` and `rocal_research/`. `build/` is compiled output.
- `rccl-tests/` — standard RCCL benchmark suite (upstream now retired in favor of ROCm/rocm-systems). `build/` has compiled perf binaries already run.
- `rocal_sandbox/` — full container rootfs overlay, Ubuntu 24.04.4 LTS base. See memory `project_rocjpeg_vcn_gate0`, `project_rocal_gfx942_solved`.
- `rocal_overlay.img` — ext3 filesystem image, the sandbox's scratch/overlay layer.
- `pytorch_rocm.sif` — Singularity/Apptainer container image for PyTorch+ROCm (~9.6GiB).

## Disk hogs / cleanup candidates (flagged, not touched)

- **`gpucore.*` — 41 files, 433GB total.** Core dumps from crashed GPU processes (May–Jul).
  This is almost certainly the single biggest thing worth cleaning up in this repo — ask
  before deleting, but it's very unlikely any of these are still needed.
- Dozens of `.miopen_*_cache` / `.miopen_coeff_*` / `.miopen_native_*_cache` /
  `.miopen_pytorch_*_cache` dirs at root — MIOpen kernel-database caches from individual
  job runs, one per job ID. Safe-looking cleanup candidates once experiments are done.
- `rocal_build_logs_<jobid>/` (17 dirs), `scaling_logs_<jobid>/` (5 dirs) — superseded
  build-attempt logs, likely fine to prune the older ones.
- `.container_tmp/` (644 entries) — container runtime scratch state.
- `ImageNet-1K/.../imagenet-object-localization-challenge.zip` (155GiB) — the original
  Kaggle zip, kept alongside the already-extracted tree.
- `.claude/worktrees/agent-a220fdf7e4a9765ae/` — an old abandoned agent git worktree
  containing an early snapshot of this project (its own `train.cu`, `main.cu`,
  `benchmark.cu`, `README.md`, etc. — predates the current file layout). Likely stale;
  confirm before removing.

## Related standalone docs at root

`MT_IMAGE_LOADER_EXPLAINER.md`, `MT_IMAGE_LOADER_TALK.md` (explain `mt_image_loader.h`),
`SYSADMIN_TICKET_mivisionx_gfx942.md` (rocAL/gfx942 bug ticket, ties to `rocal_research/`),
`RESULTS_gpu_efficiency.md` (headline throughput result), `fig_overview_methodology.dot`
(+ rendered `.pdf`/`.png`) — Graphviz big-picture methodology figure for the paper.

For project history, decisions, and status (why things are the way they are, what's
blocked, what's next), see auto-memory — this file only answers "where is X."
