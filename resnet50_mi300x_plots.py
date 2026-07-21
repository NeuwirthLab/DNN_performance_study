"""
Scientific Plots: ResNet-50 on ImageNet-1k — MI300X (native HIP) and
PyTorch ROCm baseline on the same hardware.

All style/typography comes from paper_style.py (the H100 cross-vendor
template, §6 of the plot_style spec):
  * paper_style.apply_template()       — rcParams (§6a)
  * paper_style.VENDOR_{COLORS,MARKERS,LABELS}, IDEAL_COLOR, ANN_COLOR (§6b)
  * paper_style.AXIS                   — canonical axis-label strings (§6c)
  * paper_style.save_vendor()          — PDF + PNG @ 600 dpi, bbox=tight (§6d)
  * paper_style.series / annotate / shared_bottom_legend  — drawing helpers
This script holds only data loaders + per-plot composition.

Numbers come from real measurements in this repository:
  * scaling_results_<jobid>.csv              (native HIP weak + strong sweep)
  * benchmark_results/ngc_amd_<id>/          (PyTorch ROCm weak + strong sweep)
  * profiles/amd_<id>/roofline_summary.csv   (per-kernel SOL via rocprof)
"""

import csv
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import paper_style as ps


# ============================================================================
# Apply the H100 cross-vendor template (rcParams + canonical constants)
# ============================================================================
ps.apply_template()

# Pull the shared vocabulary into local names so plot bodies stay readable.
VENDOR_COLORS, VENDOR_MARKERS, VENDOR_LABELS = (
    ps.VENDOR_COLORS, ps.VENDOR_MARKERS, ps.VENDOR_LABELS)
IDEAL_COLOR, ANN_COLOR = ps.IDEAL_COLOR, ps.ANN_COLOR
AXIS = ps.AXIS

# This script is the MI300X family.
VENDOR    = "amd_mi300x"
HIP_COLOR  = VENDOR_COLORS[VENDOR]          # native HIP trainer — vermillion
HIP_MARKER = VENDOR_MARKERS[VENDOR]         # circle (AMD vendor marker)
HIP_NAME   = VENDOR_LABELS[VENDOR] + " (HIP native)"

# PyTorch ROCm on the same silicon — second AMD hue, same marker shape.
PT_COLOR  = VENDOR_COLORS["amd_mi250"]      # palette orange
PT_MARKER = VENDOR_MARKERS["amd_mi250"]     # also "o"
PT_NAME   = VENDOR_LABELS[VENDOR] + " (PyTorch ROCm DDP)"


# ============================================================================
# Paths & data sources
# ============================================================================
REPO     = Path(__file__).resolve().parent
OUT_DIR  = REPO / "plots" / VENDOR
OUT_DIR.mkdir(parents=True, exist_ok=True)

AMD_RUN_ID      = os.environ.get("AMD_RUN_ID",     "52178693")
AMD_PROFILE_ID  = os.environ.get("AMD_PROFILE_ID", "amd_52290776")
PYTORCH_RUN_ID  = os.environ.get("PYTORCH_RUN_ID", "52255683")

AMD_CSV          = REPO / f"scaling_results_{AMD_RUN_ID}.csv"
AMD_ROOFLINE_CSV = REPO / "profiles" / AMD_PROFILE_ID / "roofline_summary.csv"
PT_WEAK_CSV      = REPO / f"benchmark_results/ngc_amd_{PYTORCH_RUN_ID}/weak_scaling_summary.csv"
PT_STRONG_CSV    = REPO / f"benchmark_results/ngc_amd_{PYTORCH_RUN_ID}/strong_scaling_summary.csv"


# ============================================================================
# MI300X hardware constants
# ============================================================================
MI300_PEAK_FP32_VECTOR_TFLOPS = 163.0
MI300_PEAK_BF16_MFMA_TFLOPS   = 1307.0
MI300_BW_TBS = 5.3
RN50_FLOPS_PER_IMAGE = 8.2e9


# ============================================================================
# Thin save wrapper — delegates to ps.save_vendor() and prints the path
# ============================================================================
def save(fig, stem, out_dir=None):
    out_dir = Path(out_dir) if out_dir is not None else OUT_DIR
    ps.save_vendor(fig, stem, VENDOR, out_dir=out_dir)
    print(f"  -> {(out_dir / f'{stem}_{VENDOR}.pdf').relative_to(REPO)}")


# ============================================================================
# LOADERS (unchanged — input data plumbing)
# ============================================================================
def load_amd_scaling(csv_path: Path) -> dict:
    out = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                ips = float(row["img_per_sec"])
            except ValueError:
                continue
            mode = row["scaling_mode"]
            gpus = int(row["ngpus"])
            out[(mode, gpus)] = {
                "gpus":         gpus,
                "mode":         mode,
                "global_batch": int(row["global_batch"]),
                "local_batch":  int(row["local_batch"]),
                "epoch_time_s": float(row["epoch_time_s"]),
                "img_per_sec":  ips,
            }
    return out


def load_pytorch_scaling(csv_path: Path) -> list:
    rows = []
    if not csv_path.exists():
        return rows
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("status", "").strip() != "ok":
                continue
            try:
                rows.append({
                    "gpus":         int(row["gpus"]),
                    "local_batch":  int(row["local_batch"]),
                    "global_batch": int(row["global_batch"]),
                    "throughput":   float(row["throughput_img_per_sec"]),
                })
            except (ValueError, KeyError):
                continue
    return sorted(rows, key=lambda r: r["gpus"])


def load_amd_roofline(csv_path: Path) -> list:
    rows = []
    if not csv_path.exists():
        return rows

    def f(row, key, default=0.0):
        try:
            return float(row[key])
        except (ValueError, KeyError, TypeError):
            return default

    with open(csv_path) as f_:
        for r in csv.DictReader(f_):
            try:
                rows.append({
                    "name":       r["kernel"],
                    "dispatches": int(r["dispatches"]),
                    "avg_us":     float(r["avg_us"]),
                    "total_ms":   float(r["total_ms"]),
                    "tflops":     float(r["fp32_tflops"]),
                    "hbm_gbps":   float(r["hbm_gbps"]),
                    "ai":         float(r["ai_fp32"]),
                    "pct_peak":   float(r["pct_fp32_peak"]),
                    "l1_hit":     f(r, "l1_hit_pct"),
                    "l2_hit":     f(r, "l2_hit_pct"),
                    "sl1_hit":    f(r, "sl1_hit_pct"),
                    "mfma_fp32":  f(r, "mfma_fp32_mops"),
                    "mfma_fp16":  f(r, "mfma_fp16_mops"),
                    "mfma_bf16":  f(r, "mfma_bf16_mops"),
                    "mfma_i8":    f(r, "mfma_i8_mops"),
                    "gpu_active": f(r, "gpu_active_pct"),
                    "sm_busy":    f(r, "sm_busy_pct"),
                })
            except (ValueError, KeyError):
                continue
    return rows


# Canonical kernel label + category functions.
# *** Keep in sync with paper_data/kernel_labels.py — single source of truth. ***
def classify_amd_kernel(name: str) -> str:
    nl = name.lower()
    if "rccl" in nl or "allreduce" in nl or "onerankreduce" in nl:
        return "AllReduce"
    if "bn_" in nl or "batchnorm" in nl or "batch_norm" in nl:
        return "BatchNorm"
    if "relu" in nl:
        return "ReLU"
    if "pool" in nl:
        return "Pooling"
    if ("softmax" in nl or "fc_" in nl
            or nl.startswith("ce_loss") or "sce_" in nl):
        return "FC/Softmax"
    if "sgd" in nl:
        return "SGD update"
    if "add_inplace" in nl or "reduce_wgrad" in nl:
        return "Residual/Reduce"
    if any(k in nl for k in ("conv", "gemm", "wrw", "winograd",
                             "fprop", "wgrad", "dgrad", "xdlops")):
        return "Conv/GEMM"
    if nl.startswith("cijk_"):
        return "Conv/GEMM"
    if ("transpose" in nl or "nchw2" in nl or "nhwc2" in nl
            or "layout" in nl):
        return "Layout xform"
    return "Other"


def short_amd_kernel_label(name: str) -> str:
    n  = name
    nl = name.lower()
    if "naive_conv" in nl:
        if "bwd" in nl: return "Conv naive (bwd)"
        if "fwd" in nl: return "Conv naive (fwd)"
        if "wrw" in nl or "wgrad" in nl: return "Conv naive (wgrad)"
        return "Conv naive"
    if "Sp3AsmConv" in n:
        if "dilation2" in n: return "Conv 3x3 (d=2)"
        if "stride2"   in n: return "Conv 3x3 (s=2)"
        if "stride1"   in n: return "Conv 3x3 (s=1)"
        return "Conv 3x3"
    if "implicit_gemm" in nl:
        if "bwd" in nl: return "Conv impl-GEMM (bwd)"
        if "wrw" in nl or "wgrad" in nl: return "Conv impl-GEMM (wgrad)"
        if "fwd" in nl: return "Conv impl-GEMM (fwd)"
        return "Conv impl-GEMM"
    if n.startswith("Cijk_"):           return "GEMM (Tensile)"
    if "gemm" in nl:                    return "GEMM"
    if "batched_transpose" in n:        return "Layout transpose"
    if "nchw2nhwc" in nl:               return "Layout NCHW->NHWC"
    if "nhwc2nchw" in nl:               return "Layout NHWC->NCHW"
    if "BatchNormBwd"        in n:      return "BatchNorm bwd"
    if "BatchNormFwdInfer"   in n:      return "BatchNorm fwd (infer)"
    if "FinalMeanVariance"   in n:      return "BatchNorm mu/sigma (final)"
    if "MeanVariance"        in n:      return "BatchNorm mu/sigma"
    if "SpatialNorm"         in n:      return "BatchNorm norm"
    if "BatchNormFwd"        in n:      return "BatchNorm fwd"
    if "bn" in nl and "fwd" in nl:      return "BatchNorm fwd"
    if "bn" in nl and "bwd" in nl:      return "BatchNorm bwd"
    if "oneRankReduce"       in n:      return "RCCL reduce"
    if "rocclr_copyBuffer"   in n:      return "ROCm copyBuf"
    if "rocclr_fillBuffer"   in n:      return "ROCm fillBuf"
    if "rccl" in nl or "allreduce" in nl: return "RCCL AllReduce"
    if "relu_bwd" in nl:                return "ReLU bwd"
    if "relu" in nl:                    return "ReLU fwd"
    if "PoolingMax" in n or "MaxPool" in n: return "MaxPool"
    if "gavgpool_bwd" in nl:            return "AvgPool (bwd)"
    if "gavgpool"     in nl:            return "AvgPool"
    if "pool" in nl:                    return "Pooling"
    if "softmax" in nl:                 return "Softmax"
    if "fc_" in nl:                     return "FC"
    if "sce_bwd" in nl:                 return "Softmax-CE bwd"
    if "ce_loss" in nl:                 return "CE loss"
    if "sgd" in nl:                     return "SGD"
    if "add_inplace" in nl:             return "Add (residual)"
    if "add_bias"    in nl:             return "Bias add"
    if "bias_grad"   in nl:             return "Bias grad"
    if "top1"        in nl:             return "Top-1 acc"
    m = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)", n)
    return (m.group(1) if m else n)[:18]


# ============================================================================
# Load all data once
# ============================================================================
print(f"Loading AMD data from run {AMD_RUN_ID} ...")
SCALING = load_amd_scaling(AMD_CSV)
WEAK_ROWS   = sorted([v for (m, _), v in SCALING.items() if m == "weak"],
                     key=lambda r: r["gpus"])
STRONG_ROWS = sorted([v for (m, _), v in SCALING.items() if m == "strong"],
                     key=lambda r: r["gpus"])

WEAK_GPUS, WEAK_IPS, WEAK_T = (
    [r["gpus"] for r in WEAK_ROWS],
    [r["img_per_sec"] for r in WEAK_ROWS],
    [r["epoch_time_s"] for r in WEAK_ROWS],
)
STRONG_GPUS, STRONG_IPS, STRONG_T = (
    [r["gpus"] for r in STRONG_ROWS],
    [r["img_per_sec"] for r in STRONG_ROWS],
    [r["epoch_time_s"] for r in STRONG_ROWS],
)
STRONG_GLOBAL_BATCH = STRONG_ROWS[0]["global_batch"] if STRONG_ROWS else 256

ROOFLINE_KERNELS = load_amd_roofline(AMD_ROOFLINE_CSV)
PT_WEAK_ROWS     = load_pytorch_scaling(PT_WEAK_CSV)
PT_STRONG_ROWS   = load_pytorch_scaling(PT_STRONG_CSV)

print(f"  HIP weak:   {len(WEAK_ROWS)} rows ({WEAK_GPUS} GPUs)")
print(f"  HIP strong: {len(STRONG_ROWS)} rows ({STRONG_GPUS} GPUs)")
print(f"  rocprof:    {len(ROOFLINE_KERNELS)} kernel samples from {AMD_PROFILE_ID}")
print(f"  PyTorch (ngc_amd_{PYTORCH_RUN_ID}): "
      f"{len(PT_WEAK_ROWS)} weak, {len(PT_STRONG_ROWS)} strong")


# Local aliases for the paper_style helpers — keep plot bodies short
_series   = ps.series
_annotate = ps.annotate


# ============================================================================
# PLOT 2 — Strong-scaling pair  (§5.1)
# ============================================================================
def plot_2_strong_scaling():
    if not STRONG_ROWS:
        print("  -> skipping plot 2 (no strong-scaling rows)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4))

    # Plot PyTorch first so its handle leads the legend (matches the
    # visual top-down line order). Stagger annotation offsets so PT (top)
    # labels sit above markers and HIP (bottom) labels sit below.
    pt_gpus, pt_ips, pt_eff = [], [], []
    if PT_STRONG_ROWS:
        pt_gpus = [r["gpus"]       for r in PT_STRONG_ROWS]
        pt_ips  = [r["throughput"] for r in PT_STRONG_ROWS]

    # (a) throughput vs GPU count + dashed ideal-linear reference
    ideal = [STRONG_IPS[0] * (g / STRONG_GPUS[0]) for g in STRONG_GPUS]
    ax1.plot(STRONG_GPUS, ideal, "--", color=IDEAL_COLOR, alpha=0.7,
             label="Ideal (linear)")
    if PT_STRONG_ROWS:
        _series(ax1, pt_gpus, pt_ips, PT_COLOR, PT_MARKER, PT_NAME)
        for x, y in zip(pt_gpus, pt_ips):
            _annotate(ax1, x, y, f"{y:,.0f}", PT_COLOR, dy=10, fontsize=9)
    _series(ax1, STRONG_GPUS, STRONG_IPS, HIP_COLOR, HIP_MARKER, HIP_NAME)
    for x, y in zip(STRONG_GPUS, STRONG_IPS):
        _annotate(ax1, x, y, f"{y:,.0f}", HIP_COLOR, dy=-16, fontsize=9)
    ax1.set_xlabel(AXIS["gpus"])
    ax1.set_ylabel(AXIS["throughput"])
    ax1.set_title("(a) Strong scaling: Throughput")
    ax1.set_xticks(STRONG_GPUS)

    # (b) parallel efficiency vs GPU count + dashed 100% reference
    base = STRONG_IPS[0]
    eff  = [STRONG_IPS[i] / (base * STRONG_GPUS[i] / STRONG_GPUS[0]) * 100
            for i in range(len(STRONG_GPUS))]
    ax2.axhline(100, color=IDEAL_COLOR, linestyle="--", alpha=0.7,
                label="Ideal (100%)")
    if PT_STRONG_ROWS:
        pt_base = pt_ips[0]
        pt_eff  = [pt_ips[i] / (pt_base * pt_gpus[i] / pt_gpus[0]) * 100
                   for i in range(len(pt_gpus))]
        _series(ax2, pt_gpus, pt_eff, PT_COLOR, PT_MARKER, PT_NAME)
        for x, e in zip(pt_gpus, pt_eff):
            _annotate(ax2, x, e, f"{e:.1f}%", PT_COLOR, dy=10, fontsize=9)
    _series(ax2, STRONG_GPUS, eff, HIP_COLOR, HIP_MARKER, HIP_NAME)
    for x, e in zip(STRONG_GPUS, eff):
        _annotate(ax2, x, e, f"{e:.1f}%", HIP_COLOR, dy=-16, fontsize=9)
    ax2.set_xlabel(AXIS["gpus"])
    ax2.set_ylabel(AXIS["efficiency"])
    ax2.set_title("(b) Strong scaling: Scaling efficiency")
    ax2.set_xticks(STRONG_GPUS)
    ax2.set_ylim(0, 110)

    ps.shared_bottom_legend(fig, ax1, ax2)
    fig.suptitle(f"{VENDOR_LABELS[VENDOR]}, ResNet-50  ·  "
                 f"global batch {STRONG_GLOBAL_BATCH}", y=1.02)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "rn50_2_strong_scaling")


# ============================================================================
# PLOT 4 — Kernel-time decomposition  (§5.7)
# ============================================================================
def plot_4_kernel_breakdown():
    if not ROOFLINE_KERNELS:
        print("  -> skipping plot 4 (no rocprof data)")
        return

    by_cat = {}
    for k in ROOFLINE_KERNELS:
        c = classify_amd_kernel(k["name"])
        by_cat[c] = by_cat.get(c, 0.0) + k["total_ms"]

    categories = sorted(by_cat.keys(), key=lambda c: -by_cat[c])
    times      = [by_cat[c] for c in categories]
    total      = sum(times)
    pct        = [100.0 * t / total for t in times]

    # Muted palette for category bars (kept neutral so the vendor color stays
    # reserved for series lines elsewhere in the figure suite).
    cat_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
                  "#8172B3", "#937860", "#CCB974", "#64B5CD",
                  "#8C564B", "#E377C2"]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ypos = np.arange(len(categories))
    ax.barh(ypos, times, color=cat_colors[:len(categories)],
            edgecolor="white", height=0.62)
    for i, (t, p) in enumerate(zip(times, pct)):
        ax.text(t + max(times) * 0.01, i,
                f"{t / 1000:.2f}s ({p:.1f}%)",
                va="center", fontsize=9, color=ANN_COLOR)

    ax.set_yticks(ypos)
    ax.set_yticklabels(categories)
    ax.invert_yaxis()
    ax.set_xlabel(AXIS["kernel_time"])
    ax.set_xlim(0, max(times) * 1.30)
    ax.set_title(f"Kernel-time decomposition  "
                 f"({len(ROOFLINE_KERNELS)} kernels, rocprof)")
    fig.suptitle(f"{VENDOR_LABELS[VENDOR]}, ResNet-50", y=1.02)
    fig.tight_layout()
    save(fig, "rn50_4_kernel_breakdown")


# ============================================================================
# PLOT 9 — Aggregate roofline (FP32 vector + BF16 MFMA)  (§5.8)
# ============================================================================
def plot_9_roofline():
    if not WEAK_ROWS:
        print("  -> skipping plot 9 (no weak-scaling rows)")
        return

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ai = np.logspace(-1, 3, 500)
    roof_fp32 = np.minimum(MI300_PEAK_FP32_VECTOR_TFLOPS, MI300_BW_TBS * ai)
    roof_bf16 = np.minimum(MI300_PEAK_BF16_MFMA_TFLOPS,   MI300_BW_TBS * ai)

    ax.plot(ai, roof_bf16, "--", color=IDEAL_COLOR, alpha=0.7, linewidth=1.6,
            label=f"BF16 MFMA peak ({MI300_PEAK_BF16_MFMA_TFLOPS:.0f} TFLOPS)")
    ax.plot(ai, roof_fp32, "-", color="black", linewidth=2.0,
            label=f"FP32 vector peak ({MI300_PEAK_FP32_VECTOR_TFLOPS:.0f} TFLOPS)")
    ax.fill_between(ai, 0.05, roof_fp32, alpha=0.05, color="gray")

    ridge = MI300_PEAK_FP32_VECTOR_TFLOPS / MI300_BW_TBS
    ax.axvline(x=ridge, color=IDEAL_COLOR, linestyle=":", alpha=0.6)
    ax.text(ridge * 1.1, 0.15, f"Ridge: {ridge:.0f}",
            fontsize=9, color=IDEAL_COLOR)

    rn50_ai = 150.0
    per_gpu_vals = [(r["gpus"],
                     r["img_per_sec"] / r["gpus"] * RN50_FLOPS_PER_IMAGE / 1e12)
                    for r in WEAK_ROWS]
    achieved_mean = float(np.mean([v for _, v in per_gpu_vals]))
    ax.scatter([rn50_ai], [achieved_mean], s=90, zorder=5,
               color=HIP_COLOR, marker=HIP_MARKER,
               edgecolor="white", linewidth=0.8,
               label="ResNet-50 measured (FP32)")

    pct_fp32 = 100.0 * achieved_mean / MI300_PEAK_FP32_VECTOR_TFLOPS
    pct_bf16 = 100.0 * achieved_mean / MI300_PEAK_BF16_MFMA_TFLOPS
    ax.annotate(f"{pct_fp32:.1f}% of FP32\n{pct_bf16:.2f}% of BF16",
                (rn50_ai, achieved_mean),
                textcoords="offset points", xytext=(10, -4),
                fontsize=9, color=ANN_COLOR)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 1000)
    ax.set_ylim(0.5, MI300_PEAK_BF16_MFMA_TFLOPS * 1.5)
    ax.set_xlabel(AXIS["ai"])
    ax.set_ylabel(AXIS["performance"])
    ax.set_title(f"{VENDOR_LABELS[VENDOR]} — Peak: "
                 f"{MI300_PEAK_FP32_VECTOR_TFLOPS:.0f} TFLOPS (FP32), "
                 f"BW: {MI300_BW_TBS:.1f} TB/s")
    ps.shared_bottom_legend(fig, ax)
    fig.suptitle(f"{VENDOR_LABELS[VENDOR]}, ResNet-50  ·  Aggregate roofline",
                 y=1.02)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "rn50_9_roofline")


# ============================================================================
# PLOT 9b — Per-kernel roofline (§5.8)
# ============================================================================
def plot_9b_per_kernel_roofline():
    if not ROOFLINE_KERNELS:
        print("  -> skipping plot 9b (no rocprof data)")
        return

    # Aggregate per short label so duplicate dispatches collapse to one point.
    grouped = {}
    for k in ROOFLINE_KERNELS:
        lbl = short_amd_kernel_label(k["name"])
        g = grouped.setdefault(lbl, {"ai": [], "tf": [], "ms": 0.0})
        if k["ai"]     > 0: g["ai"].append(k["ai"])
        if k["tflops"] > 0: g["tf"].append(k["tflops"])
        g["ms"] += k["total_ms"]

    POINTS, n_skipped = {}, 0
    for lbl, g in grouped.items():
        if not g["ai"] or not g["tf"]:
            n_skipped += 1
            continue
        POINTS[lbl] = (float(np.mean(g["ai"])),
                       float(np.mean(g["tf"])),
                       g["ms"])

    PEAK_FP32 = MI300_PEAK_FP32_VECTOR_TFLOPS
    BW_TBS    = MI300_BW_TBS

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # 1. Roofline curve — black solid (template §5.8)
    ai = np.logspace(-1, 3, 600)
    roof = np.minimum(PEAK_FP32, BW_TBS * ai)
    ax.plot(ai, roof, color="black", linewidth=2.0, label="Roofline", zorder=3)
    ax.fill_between(ai, 0.05, roof, alpha=0.05, color="gray", zorder=1)

    # 2. Ridge marker — grey dotted vertical with bottom label
    ridge = PEAK_FP32 / BW_TBS
    ax.axvline(x=ridge, color=IDEAL_COLOR, linestyle=":", alpha=0.6, zorder=2)
    ax.text(ridge * 1.1, 0.15, f"Ridge: {ridge:.0f}",
            fontsize=9, color=IDEAL_COLOR)

    # 3. Scatter — vendor color, white halo edge per template
    valid = [(n, v) for n, v in POINTS.items() if v[0] > 0 and v[1] > 0]
    for _, (ai_v, perf, _) in valid:
        ax.scatter(ai_v, perf, s=60, zorder=5, color=HIP_COLOR,
                   marker=HIP_MARKER, edgecolor="white", linewidth=0.8)

    # 4. Collision-avoiding labels for top-N kernels by GPU time
    TOP_N = 9
    labeled = sorted(valid, key=lambda kv: -kv[1][2])[:TOP_N]

    LABEL_OVERRIDES = {
        "Conv 3x3 (s=1)": ( 0,  22, "center", "bottom"),
        "Conv 3x3 (s=2)": ( 0, -22, "center", "top"),
        "Conv 3x3 (d=2)": ( 0, -22, "center", "top"),
    }
    CANDIDATES = [
        ( 32,   8, "left",   "center"),
        ( 28,  28, "left",   "bottom"),
        ( 28, -28, "left",   "top"),
        (-32,   8, "right",  "center"),
        (-28,  28, "right",  "bottom"),
        (-28, -28, "right",  "top"),
        (  0,  32, "center", "bottom"),
        (  0, -32, "center", "top"),
        ( 64,   8, "left",   "center"),
        ( 56,  56, "left",   "bottom"),
        ( 56, -56, "left",   "top"),
        (-64,   8, "right",  "center"),
        (-56,  56, "right",  "bottom"),
        (-56, -56, "right",  "top"),
        (  0,  64, "center", "bottom"),
        (  0, -64, "center", "top"),
    ]

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed_bboxes = []

    def _place(name, ai_v, perf, dx, dy, ha, va):
        return ax.annotate(
            name, (ai_v, perf),
            textcoords="offset points", xytext=(dx, dy),
            fontsize=9, color=ANN_COLOR,
            ha=ha, va=va, zorder=6, annotation_clip=True,
            arrowprops=dict(arrowstyle="-", color="0.55",
                            lw=0.5, shrinkA=2, shrinkB=4),
            bbox=dict(boxstyle="round,pad=0.22",
                      facecolor="white", edgecolor="0.7",
                      linewidth=0.5, alpha=0.92),
        )

    for name, (ai_v, perf, _) in labeled:
        if name in LABEL_OVERRIDES:
            dx, dy, ha, va = LABEL_OVERRIDES[name]
            ann = _place(name, ai_v, perf, dx, dy, ha, va)
            placed_bboxes.append(ann.get_window_extent(renderer))
            continue
        chosen = None
        for dx, dy, ha, va in CANDIDATES:
            ann = _place(name, ai_v, perf, dx, dy, ha, va)
            bb = ann.get_window_extent(renderer)
            if not any(bb.overlaps(prev) for prev in placed_bboxes):
                chosen = ann; placed_bboxes.append(bb); break
            ann.remove()
        if chosen is None:
            dx, dy, ha, va = CANDIDATES[0]
            ann = _place(name, ai_v, perf, dx, dy, ha, va)
            placed_bboxes.append(ann.get_window_extent(renderer))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 1000)
    ax.set_ylim(0.1, PEAK_FP32 * 2)
    ax.set_xlabel(AXIS["ai"])
    ax.set_ylabel(AXIS["performance"])
    ax.set_title(f"{VENDOR_LABELS[VENDOR]} — Peak: {PEAK_FP32:.0f} TFLOPS, "
                 f"BW: {BW_TBS:.1f} TB/s")
    fig.suptitle(f"{VENDOR_LABELS[VENDOR]}, ResNet-50  ·  "
                 f"Per-kernel roofline (top {TOP_N} labeled)", y=1.02)
    fig.tight_layout()

    save(fig, "rn50_9b_per_kernel_roofline")
    print(f"     [9b] plotted: {len(valid)}  skipped (AI<=0 or TF<=0): "
          f"{n_skipped}  raw kernels: {len(ROOFLINE_KERNELS)}")


# ============================================================================
# PLOT 9c — Per-kernel cache hit rates  (grouped bars)
# ============================================================================
def plot_9c_cache_thrashing():
    if not ROOFLINE_KERNELS:
        print("  -> skipping plot 9c (no rocprof data)")
        return
    have_data = [k for k in ROOFLINE_KERNELS
                 if k["l2_hit"] > 0 or k["l1_hit"] > 0]
    if not have_data:
        print("  -> skipping plot 9c (profile predates the cache-hit pmc patch)")
        return

    TOP_N = 8
    ordered = sorted(have_data, key=lambda k: -k["total_ms"])[:TOP_N]
    labels  = [short_amd_kernel_label(k["name"]) for k in ordered]
    l1      = [k["l1_hit"]  for k in ordered]
    l2      = [k["l2_hit"]  for k in ordered]
    sl1     = [k["sl1_hit"] for k in ordered]

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    x = np.arange(len(labels))
    width = 0.26
    series = [
        ("L1 vector hit", l1,  "#4C72B0", -1.0),
        ("L2 hit",         l2,  "#DD8452",  0.0),
        ("Scalar L1 hit",  sl1, "#55A868",  1.0),
    ]
    for label, vals, color, off in series:
        offs = x + off * width
        ax.bar(offs, vals, width, color=color, edgecolor="white", label=label)
        for xi, v in zip(offs, vals):
            ax.text(xi, v + 1.5, f"{v:.0f}", ha="center",
                    fontsize=9, color=color)

    # Thrashing-warning band: L2 < 50% likely thrashing.
    ax.axhspan(0, 50, color="#C44E52", alpha=0.05)
    ax.text(len(labels) - 0.4, 25, "thrashing zone\n(L2 < 50%)",
            fontsize=10, color="#C44E52", style="italic", ha="right",
            alpha=0.75)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel(AXIS["cache_hit"])
    ax.set_ylim(0, 110)
    ax.set_title(f"Per-kernel cache hit rates (top {TOP_N} by GPU time)")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.22))
    fig.suptitle(f"{VENDOR_LABELS[VENDOR]}, ResNet-50  ·  Cache thrashing",
                 y=1.02)
    fig.subplots_adjust(bottom=0.28)
    save(fig, "rn50_9c_cache_thrashing")


# ============================================================================
# PLOT 9d — MFMA utilization by precision
# ============================================================================
def plot_9d_mfma_utilization():
    if not ROOFLINE_KERNELS:
        print("  -> skipping plot 9d (no rocprof data)")
        return
    have_data = [k for k in ROOFLINE_KERNELS
                 if (k["mfma_fp32"] + k["mfma_fp16"]
                     + k["mfma_bf16"] + k["mfma_i8"]) > 0]
    TOP_N = 8
    ordered = sorted(ROOFLINE_KERNELS, key=lambda k: -k["total_ms"])[:TOP_N]
    labels  = [short_amd_kernel_label(k["name"]) for k in ordered]

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    x = np.arange(len(labels))
    width = 0.20

    series = [
        ("FP32 MFMA", [k["mfma_fp32"] for k in ordered], "#4C72B0", -1.5),
        ("FP16 MFMA", [k["mfma_fp16"] for k in ordered], "#55A868", -0.5),
        ("BF16 MFMA", [k["mfma_bf16"] for k in ordered], "#DD8452",  0.5),
        ("INT8 MFMA", [k["mfma_i8"]   for k in ordered], "#C44E52",  1.5),
    ]
    max_val = max((max(vals) for _, vals, _, _ in series), default=0) or 1
    for label, vals, color, off in series:
        offs = x + off * width
        ax.bar(offs, vals, width, color=color, edgecolor="white", label=label)
        for xi, v in zip(offs, vals):
            if v > 0:
                ax.text(xi, v + max_val * 0.02, f"{v:,.0f}",
                        ha="center", fontsize=8, color=color)
            elif label == "BF16 MFMA":
                ax.text(xi, max_val * 0.04, "0",
                        ha="center", fontsize=9, color="#C44E52")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel(AXIS["mfma_ops"])
    if max_val > 1000:
        ax.set_yscale("log")
        ax.set_ylim(1, max_val * 5)
    note = "" if have_data else "  ·  ALL zero → no MFMA engaged (FP32 vector path)"
    ax.set_title(f"MFMA matrix-core utilization{note}")
    ps.shared_bottom_legend(fig, ax)
    fig.suptitle(f"{VENDOR_LABELS[VENDOR]}, ResNet-50", y=1.02)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "rn50_9d_mfma_utilization")


# ============================================================================
# PLOT 11 — Epoch-time curve  (§5.5)
# ============================================================================
def plot_11_epoch_time():
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    _series(ax, WEAK_GPUS, WEAK_T, HIP_COLOR, HIP_MARKER, HIP_NAME)
    for x, y in zip(WEAK_GPUS, WEAK_T):
        _annotate(ax, x, y, f"{y:.0f}s", HIP_COLOR)
    ax.set_xlabel(AXIS["gpus"])
    ax.set_ylabel(AXIS["epoch_time"])
    ax.set_title("Wall time per epoch (batch = 256)")
    ax.set_xticks(WEAK_GPUS)
    ps.shared_bottom_legend(fig, ax)
    fig.suptitle(f"{VENDOR_LABELS[VENDOR]}, ResNet-50", y=1.02)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "rn50_11_epoch_time")


# ============================================================================
# PLOT 12 — Speedup curve  (§5.4)
# ============================================================================
def plot_12_speedup():
    if not STRONG_ROWS:
        print("  -> skipping plot 12 (no strong-scaling data)")
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    t1 = STRONG_T[0]
    speedup = [t1 / t for t in STRONG_T]
    ax.plot(STRONG_GPUS, STRONG_GPUS, "--", color=IDEAL_COLOR, alpha=0.7,
            label="Ideal (linear)")
    _series(ax, STRONG_GPUS, speedup, HIP_COLOR, HIP_MARKER, HIP_NAME)
    for x, s in zip(STRONG_GPUS, speedup):
        _annotate(ax, x, s, rf"{s:.2f}$\times$", HIP_COLOR)
    ax.set_xlabel(AXIS["gpus"])
    ax.set_ylabel(AXIS["speedup"])
    ax.set_title("Multi-GPU speedup")
    ax.set_xticks(STRONG_GPUS)
    ps.shared_bottom_legend(fig, ax)
    fig.suptitle(f"{VENDOR_LABELS[VENDOR]}, ResNet-50  ·  "
                 f"global batch {STRONG_GLOBAL_BATCH}", y=1.02)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "rn50_12_speedup")


# ============================================================================
# PLOT 14 — Weak-scaling pair  (§5.2)
# ============================================================================
def plot_14_weak_scaling():
    if not WEAK_ROWS:
        print("  -> skipping plot 14 (no weak-scaling rows)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4))

    # Plot PyTorch FIRST so its handle leads the legend — matches the visual
    # top-down ordering of the lines (PT is the higher trace at every GPU
    # count). Stagger annotation offsets so PT (top) labels sit above markers
    # and HIP (bottom) labels sit below, eliminating overlap in the middle.
    pt_gpus, pt_ips = [], []
    if PT_WEAK_ROWS:
        pt_gpus = [r["gpus"]       for r in PT_WEAK_ROWS]
        pt_ips  = [r["throughput"] for r in PT_WEAK_ROWS]

    # (a) throughput vs GPU count (weak scaling: no ideal-linear line)
    if PT_WEAK_ROWS:
        _series(ax1, pt_gpus, pt_ips, PT_COLOR, PT_MARKER, PT_NAME)
        for x, y in zip(pt_gpus, pt_ips):
            _annotate(ax1, x, y, f"{y:,.0f}", PT_COLOR, dy=10, fontsize=9)
    _series(ax1, WEAK_GPUS, WEAK_IPS, HIP_COLOR, HIP_MARKER, HIP_NAME)
    for x, y in zip(WEAK_GPUS, WEAK_IPS):
        _annotate(ax1, x, y, f"{y:,.0f}", HIP_COLOR, dy=-16, fontsize=9)
    ax1.set_xlabel(AXIS["gpus"])
    ax1.set_ylabel(AXIS["throughput"])
    ax1.set_title("(a) Weak scaling: Throughput")
    ax1.set_xticks(WEAK_GPUS)

    # (b) throughput per GPU (flat = ideal — implied by axis; no drawn line)
    per_gpu = [WEAK_IPS[i] / WEAK_GPUS[i] for i in range(len(WEAK_GPUS))]
    if PT_WEAK_ROWS:
        pt_per_gpu = [r["throughput"] / r["gpus"] for r in PT_WEAK_ROWS]
        _series(ax2, pt_gpus, pt_per_gpu, PT_COLOR, PT_MARKER, PT_NAME)
        for x, y in zip(pt_gpus, pt_per_gpu):
            _annotate(ax2, x, y, f"{y:,.0f}", PT_COLOR, dy=10, fontsize=9)
    _series(ax2, WEAK_GPUS, per_gpu, HIP_COLOR, HIP_MARKER, HIP_NAME)
    for x, y in zip(WEAK_GPUS, per_gpu):
        _annotate(ax2, x, y, f"{y:,.0f}", HIP_COLOR, dy=-16, fontsize=9)
    ax2.set_xlabel(AXIS["gpus"])
    ax2.set_ylabel(AXIS["throughput_per_g"])
    ax2.set_title("(b) Weak scaling: Throughput per GPU")
    ax2.set_xticks(WEAK_GPUS)

    ps.shared_bottom_legend(fig, ax1)
    fig.suptitle(f"{VENDOR_LABELS[VENDOR]}, ResNet-50  ·  weak scaling",
                 y=1.02)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "rn50_14_weak_scaling")


# ============================================================================
# PLOT 15 — Gap-decomposition waterfall
# ============================================================================
def plot_15_gap_decomposition():
    """Measured throughput + roofline-derived BF16-MFMA ceiling.

    Replaces the earlier fabricated waterfall (which stacked hypothetical
    +%-uplifts whose product overshot PyTorch by 2x). Every number here is
    either measured or derived from MI300X published peaks; nothing is
    speculative beyond the explicit "kernel-time-only" ceiling.

    Bars
    ----
      1. HIP native (measured)       — from scaling_results_<id>.csv
      2. PyTorch ROCm DDP (measured) — from ngc_amd_<id>/weak_scaling_summary.csv
      3. BF16-MFMA ceiling (theory)  — Conv/GEMM time scaled by
         PEAK_BF16_MFMA / PEAK_FP32_VECTOR (= 8.0x on MI300X), all other
         kernel time held fixed. Upper bound only; needs measurement.
    """
    if not WEAK_ROWS or not PT_WEAK_ROWS or not ROOFLINE_KERNELS:
        print("  -> skipping plot 15 (need HIP + PT + rocprof data)")
        return

    hip = WEAK_IPS[0]
    pyt = PT_WEAK_ROWS[0]["throughput"]

    # Profile-derived kernel-time split (single source of truth: rocprof)
    total_ms = sum(k["total_ms"] for k in ROOFLINE_KERNELS)
    conv_ms  = sum(k["total_ms"] for k in ROOFLINE_KERNELS
                   if classify_amd_kernel(k["name"]) == "Conv/GEMM")
    conv_frac = conv_ms / total_ms

    # Roofline-derived ceiling: only Conv/GEMM benefits from MFMA.
    ratio          = MI300_PEAK_BF16_MFMA_TFLOPS / MI300_PEAK_FP32_VECTOR_TFLOPS
    conv_ms_ceil   = conv_ms / ratio
    total_ms_ceil  = conv_ms_ceil + (total_ms - conv_ms)
    kernel_speedup = total_ms / total_ms_ceil
    ceiling_ips    = hip * kernel_speedup

    bars = [
        ("HIP native\n(measured)",            hip,          HIP_COLOR,  False),
        ("PyTorch ROCm DDP\n(measured)",      pyt,          PT_COLOR,   False),
        ("BF16-MFMA ceiling\n(theoretical)",  ceiling_ips,  "#888888",  True),
    ]
    labels       = [b[0] for b in bars]
    values       = [b[1] for b in bars]
    colors       = [b[2] for b in bars]
    is_theory    = [b[3] for b in bars]
    x = np.arange(len(bars))

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for xi, v, c, theory in zip(x, values, colors, is_theory):
        ax.bar(xi, v, width=0.55, color=c,
               edgecolor=c if not theory else "0.35",
               linewidth=1.5,
               hatch="///" if theory else None,
               alpha=0.85 if theory else 1.0)

    # Top-of-bar labels (absolute img/s)
    ymax = max(values)
    for xi, v in zip(x, values):
        ax.text(xi, v + ymax * 0.015, f"{v:,.0f}",
                ha="center", fontsize=11, color=ANN_COLOR)

    # Headroom annotations under the labels: ratios vs HIP
    ratios = [v / hip for v in values]
    for xi, v, r in zip(x, values, ratios):
        if r == 1.0:
            tag = "baseline"
        else:
            tag = rf"{r:.2f}$\times$ HIP"
        ax.text(xi, v + ymax * 0.06, tag,
                ha="center", fontsize=10, color=ANN_COLOR, style="italic")

    # Headroom indicator: dashed line from PT to ceiling with "PT = N% of ceiling"
    pt_frac_of_ceiling = pyt / ceiling_ips
    ax.annotate(
        "", xy=(2, pyt), xytext=(2, ceiling_ips),
        arrowprops=dict(arrowstyle="<->", color="0.45", lw=1.0,
                        shrinkA=4, shrinkB=4),
    )
    ax.text(2.18, (pyt + ceiling_ips) / 2,
            f"PT reaches {100*pt_frac_of_ceiling:.0f}%\nof ceiling",
            fontsize=9, color="0.30", va="center")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(AXIS["throughput_per_g"])
    ax.set_ylim(0, ceiling_ips * 1.18)
    ax.set_title("Measured throughput + BF16-MFMA roofline ceiling (1 GPU)")

    # Honest footnote describing exactly how the ceiling is derived.
    note = (f"Ceiling = HIP $\\times$ kernel-time speedup if Conv/GEMM "
            f"({conv_frac*100:.1f}% of GPU time) ran at MI300X BF16 MFMA peak "
            f"({MI300_PEAK_BF16_MFMA_TFLOPS:.0f} TFLOPS, "
            f"{ratio:.1f}$\\times$ FP32 vector peak); all other kernel time "
            f"held fixed. Upper bound — not a measured BF16 run.")
    fig.text(0.5, -0.02, note, ha="center", va="top",
             fontsize=8, color="0.30", style="italic", wrap=True)

    fig.suptitle(f"{VENDOR_LABELS[VENDOR]}, ResNet-50  ·  "
                 f"HIP vs PyTorch ROCm with theoretical BF16 headroom",
                 y=1.02)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "rn50_15_gap_decomposition")


# ============================================================================
# PLOT 16 — HIP vs PyTorch ROCm head-to-head
# ============================================================================
def plot_16_hip_vs_pytorch():
    if not PT_WEAK_ROWS:
        print("  -> skipping plot 16 (no PyTorch sweep data)")
        return

    pt_map = {r["gpus"]: r["throughput"] for r in PT_WEAK_ROWS}
    common = sorted(set(WEAK_GPUS) & set(pt_map.keys()))
    if not common:
        print("  -> skipping plot 16 (no overlapping GPU counts)")
        return

    hip_ips     = [WEAK_IPS[WEAK_GPUS.index(g)] for g in common]
    pt_ips      = [pt_map[g] for g in common]
    ratio       = [p / h for p, h in zip(pt_ips, hip_ips)]
    hip_per_gpu = [h / g for h, g in zip(hip_ips, common)]
    pt_per_gpu  = [p / g for p, g in zip(pt_ips, common)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    x = np.arange(len(common))
    width = 0.32

    # (a) grouped bars
    ax1.bar(x - width / 2, hip_per_gpu, width, color=HIP_COLOR,
            edgecolor="white", label=HIP_NAME)
    ax1.bar(x + width / 2, pt_per_gpu,  width, color=PT_COLOR,
            edgecolor="white", label=PT_NAME)
    ymax = max(max(hip_per_gpu), max(pt_per_gpu))
    for xi, v in zip(x - width / 2, hip_per_gpu):
        ax1.text(xi, v + ymax * 0.015, f"{v:,.0f}",
                 ha="center", fontsize=9, color=HIP_COLOR)
    for xi, v in zip(x + width / 2, pt_per_gpu):
        ax1.text(xi, v + ymax * 0.015, f"{v:,.0f}",
                 ha="center", fontsize=9, color=PT_COLOR)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{g}" for g in common])
    ax1.set_xlabel(AXIS["gpus"])
    ax1.set_ylabel(AXIS["throughput_per_g"])
    ax1.set_title("(a) Per-GPU throughput")
    ax1.set_ylim(0, ymax * 1.18)

    # (b) ratio line
    ax2.axhline(1.0, color=IDEAL_COLOR, linestyle="--", alpha=0.7,
                label=r"Parity (1.0$\times$)")
    _series(ax2, common, ratio, PT_COLOR, PT_MARKER,
            "PyTorch / HIP throughput")
    for x_, r_ in zip(common, ratio):
        _annotate(ax2, x_, r_, rf"{r_:.2f}$\times$", PT_COLOR)
    ax2.set_xlabel(AXIS["gpus"])
    ax2.set_ylabel(AXIS["throughput_ratio"])
    ax2.set_title("(b) Framework speedup")
    ax2.set_xticks(common)
    ax2.set_ylim(0.9, max(ratio) * 1.25)

    ps.shared_bottom_legend(fig, ax1, ax2)
    fig.suptitle(f"{VENDOR_LABELS[VENDOR]}, ResNet-50  ·  "
                 f"HIP vs PyTorch ROCm (DDP)", y=1.02)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "rn50_16_hip_vs_pytorch")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print(f"Generating {VENDOR_LABELS[VENDOR]} ResNet-50 scientific plots...")
    plot_2_strong_scaling()
    plot_4_kernel_breakdown()
    plot_9_roofline()
    plot_9b_per_kernel_roofline()
    plot_9c_cache_thrashing()
    plot_9d_mfma_utilization()
    plot_11_epoch_time()
    plot_12_speedup()
    plot_14_weak_scaling()
    plot_15_gap_decomposition()
    plot_16_hip_vs_pytorch()
    print(f"\nAll plots saved to: {OUT_DIR.relative_to(REPO)}/")
