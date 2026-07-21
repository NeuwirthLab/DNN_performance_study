#!/usr/bin/env python3
"""
plot_roofline.py — Publication-grade roofline plot for ResNet-50 on MI300X.

Reads rocprofv3 counter_collection.csv files (handles multi-pass collection),
pivots long→wide, computes per-kernel arithmetic intensity (FLOP/byte) and
performance (GFLOP/s), and produces an ISC/SC-style roofline plot with:

  * Memory-bound region shaded.
  * VALU FP32, MFMA FP32, MFMA FP16/BF16 ceilings (CDNA3 MI300X).
  * Knee-point markers where each ceiling meets the HBM diagonal.
  * Kernels grouped into 7 semantic categories by colour + marker
    (Conv direct/Winograd, im2col / col2im, rocBLAS GEMM, BatchNorm,
    Element-wise, Reduction, Optimizer, Memory ops).
  * Friendly short names — long mangled rocBLAS/MIOpen names are
    decoded via the RENAME_RULES table.
  * Top-N kernels annotated with leader lines so labels don't overlap.

Usage:
    python3 plot_roofline.py --root profiles/v3direct_<jobid>
    python3 plot_roofline.py --root <dir> --out figures/roofline.pdf --top 15

Output: both <out> and <out>.png are written (vector for papers, raster
for slides). Default DPI 300, font sizes tuned for two-column LaTeX.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D


# ── MI300X spec (CDNA3) ─────────────────────────────────────────────────────
@dataclass
class Spec:
    name:                  str   = "AMD Instinct MI300X (gfx942)"
    peak_vec_fp32_tflops:  float = 75.8      # 304 CU × 64 lane × 2 (FMA) × 1.95 GHz
    peak_mfma_fp32_tflops: float = 163.4
    peak_mfma_bf16_tflops: float = 1307.4
    peak_mfma_fp16_tflops: float = 1307.4
    peak_mfma_fp64_tflops: float = 81.7
    peak_hbm_bw_gbps:      float = 5300.0    # HBM3 aggregate


# ── Kernel name → category + friendly label ────────────────────────────────
# Each entry: (regex, friendly_name, category). First match wins.
# Categories drive colour; the friendly name is what gets annotated.
RENAME_RULES: list[tuple[str, str, str]] = [
    # MIOpen Winograd convolution — F(m, r) algorithm, tile sizes m×r.
    (r"^miopenSp3AsmConv.*fp32_f3x2_stride1",
        "Conv 3×3 Winograd F(3×2)",     "Convolution"),
    (r"^miopenSp3AsmConv.*fp32_f2x3_stride1",
        "Conv 3×3 Winograd F(2×3)",     "Convolution"),
    (r"^miopenSp3AsmConv.*Winograd.*",
        "Conv Winograd",                "Convolution"),
    # MIOpen "naive" direct convolution paths (fwd / bwd-data / wrw).
    (r"naive_conv.*fwd.*",  "Conv fwd (direct)",        "Convolution"),
    (r"naive_conv.*bwd.*",  "Conv bwd-data (direct)",   "Convolution"),
    (r"naive_conv.*wrw.*",  "Conv wrw (direct)",        "Convolution"),
    # im2col / col2im preludes — separate category because they're memory-only.
    (r"^Im2d2?Col(_v\d+)?",  "im2col",   "Layout transform"),
    (r"^Col2Im2dU",          "col2im",   "Layout transform"),
    (r"batched_transpose.*", "Transpose","Layout transform"),
    # rocBLAS Tensile GEMM. Parse the MT (macro tile) for a meaningful label.
    (r"^Cijk_.*_MT(\d+)x(\d+)x(\d+).*",
        "GEMM tile {MT0}×{MT1}",        "GEMM"),
    # MIOpen BatchNorm.
    (r"^MIOpenBatchNormFwd.*Spatial",  "BatchNorm fwd",  "BatchNorm"),
    (r"^MIOpenBatchNormBwd.*Spatial",  "BatchNorm bwd",  "BatchNorm"),
    (r"^MIOpenBatchNorm.*",            "BatchNorm",      "BatchNorm"),
    # Custom kernels from resnet50.cu.
    (r"^relu_kernel.*",         "ReLU fwd",          "Element-wise"),
    (r"^relu_bwd_kernel.*",     "ReLU bwd",          "Element-wise"),
    (r"^add_inplace_kernel.*",  "Residual add",      "Element-wise"),
    (r"^add_bias_kernel.*",     "Bias add",          "Element-wise"),
    (r"^bias_grad_kernel.*",    "Bias grad",         "Element-wise"),
    (r"^float_to_bf16_kernel.*","FP32→BF16 cast",    "Element-wise"),
    (r"^zero_kernel.*",         "Zero",              "Element-wise"),
    (r"^gavgpool_kernel.*",     "Global avg pool",   "Reduction"),
    (r"^gavgpool_bwd_kernel.*", "Global avg pool bwd","Reduction"),
    (r"^softmax_kernel.*",      "Softmax",           "Reduction"),
    (r"^sce_bwd_kernel.*",      "Softmax-CE bwd",    "Reduction"),
    (r"^ce_loss_kernel.*",      "Cross-entropy loss","Reduction"),
    (r"^top1_kernel.*",         "Top-1 accuracy",    "Reduction"),
    (r"^sgd_kernel.*",          "SGD update",        "Optimizer"),
    # HIP runtime memory ops.
    (r"__amd_rocclr_copyBuffer.*",         "hipMemcpy",  "Memory op"),
    (r"__amd_rocclr_fillBufferAligned.*",  "hipMemset",  "Memory op"),

    # ══════════════════════════════════════════════════════════════════════
    # PyTorch / ATen + shared-library kernels.
    #
    # A PyTorch ResNet-50 run on MI300X dispatches a DIFFERENT kernel set than
    # the native HIP trainer: ATen elementwise/reduction kernels, fused-foreach
    # optimizer kernels, and MIOpen's implicit-GEMM (igemm)/CK conv paths rather
    # than the native Winograd+direct convs. The rules below map those names
    # onto the SAME friendly labels + categories the native run uses, so a
    # PyTorch roofline / cache plot lines up 1:1 with the already-profiled
    # native tests. Several rules ALSO catch native names that previously fell
    # through to "Other" (Winograd stride2/dilation2, pooling, SoftMaxBlock,
    # the NCHW<->CNHW transposes), which keeps native and PyTorch consistent.
    #
    # NB friendly_label matches the FULL name, so the distinguishing functor is
    # reached even when it sits after "(anonymous namespace)".
    # ──────────────────────────────────────────────────────────────────────

    # ── ATen elementwise: identify by the functor, map to the native concept ──
    (r".*vectorized_elementwise_kernel.*launch_clamp_scalar.*",
        "ReLU fwd",            "Element-wise"),   # relu = clamp_min(0)
    (r".*vectorized_elementwise_kernel.*threshold.*",
        "ReLU bwd",            "Element-wise"),   # threshold_backward
    (r".*vectorized_elementwise_kernel.*CUDAFunctor_add<float>.*",
        "Residual add",        "Element-wise"),   # tensor+tensor add
    (r".*vectorized_elementwise_kernel.*CUDAFunctorOnSelf_add<long>.*",
        "Add (int counter)",   "Element-wise"),   # e.g. num_batches_tracked
    (r".*vectorized_elementwise_kernel.*MulFunctor.*",
        "Scale (a·x)",         "Element-wise"),   # scalar*tensor
    (r".*vectorized_elementwise_kernel.*FillFunctor.*",
        "Fill",                "Element-wise"),
    # Any other ATen elementwise kernel — never let it collapse to "Other".
    (r".*(vectorized_)?elementwise_kernel(_manual_unroll)?.*",
        "Elementwise (misc)",  "Element-wise"),
    (r"^SubTensorOpWithScalar1d.*",
        "SubTensor set",       "Element-wise"),   # MIOpen set-scalar / zero

    # ── Fused optimizer (foreach SGD) ─────────────────────────────────────
    (r".*multi_tensor_apply_kernel.*",
        "SGD update",          "Optimizer"),

    # ── Implicit-GEMM (igemm) + Composable-Kernel convolutions ────────────
    (r"^igemm_fwd.*", "Conv fwd (impl. GEMM)",      "Convolution"),
    (r"^igemm_bwd.*", "Conv bwd-data (impl. GEMM)", "Convolution"),
    (r"^igemm_wrw.*", "Conv wrw (impl. GEMM)",      "Convolution"),
    (r".*kernel_batched_gemm_xdlops_bwd_weight.*",
        "Conv wrw (CK XDL)",   "Convolution"),

    # ── Winograd variants the stride1 rules above miss (native + PyTorch) ──
    (r"^miopenSp3AsmConv.*fp32_f3x2.*", "Conv 3×3 Winograd F(3×2)", "Convolution"),
    (r"^miopenSp3AsmConv.*fp32_f2x3.*", "Conv 3×3 Winograd F(2×3)", "Convolution"),
    (r"^miopenSp3AsmConv.*",            "Conv Winograd",            "Convolution"),

    # ── Pooling (spatial reduction) — ATen + MIOpen, aligned ──────────────
    (r".*max_pool_forward.*",  "Max pool fwd", "Reduction"),
    (r".*max_pool_backward.*", "Max pool bwd", "Reduction"),
    (r"^mloPoolingMaxBwd.*",   "Max pool bwd", "Reduction"),
    (r"^mloPoolingG.*",        "Pool fwd",     "Reduction"),

    # ── Softmax / loss reductions (ATen + native SoftMaxBlock) ────────────
    (r".*softmax_warp_forward.*",  "Softmax",            "Reduction"),
    (r".*softmax_warp_backward.*", "Softmax bwd",        "Reduction"),
    (r"^SoftMaxBlock.*",           "Softmax",            "Reduction"),
    (r".*nll_loss_forward.*",      "Cross-entropy loss", "Reduction"),
    (r".*nll_loss_backward.*",     "Softmax-CE bwd",     "Reduction"),
    (r".*reduce_kernel.*MeanOps.*","Mean reduce",        "Reduction"),
    (r".*reduce_kernel.*",         "Reduce",             "Reduction"),

    # ── Layout transforms (ATen concat + MIOpen NCHW<->CNHW) ──────────────
    (r".*CatArrayBatchedCopy.*", "Concat",     "Layout transform"),
    (r"^transpose_NCHW2CNHW.*",  "NCHW→CNHW",  "Layout transform"),
    (r"^transpose_CNHW2NCHW.*",  "CNHW→NCHW",  "Layout transform"),
]


# Category style — colourblind-safe palette (Okabe-Ito) + distinct markers.
CATEGORY_STYLE = {
    "Convolution":      {"color": "#0072B2", "marker": "o", "z": 6},
    "GEMM":             {"color": "#D55E00", "marker": "s", "z": 6},
    "BatchNorm":        {"color": "#009E73", "marker": "D", "z": 5},
    "Element-wise":     {"color": "#CC79A7", "marker": "^", "z": 4},
    "Reduction":        {"color": "#F0E442", "marker": "v", "z": 4},
    "Optimizer":        {"color": "#56B4E9", "marker": "P", "z": 4},
    "Layout transform": {"color": "#999999", "marker": "X", "z": 3},
    "Memory op":        {"color": "#666666", "marker": "x", "z": 2},
    "Other":            {"color": "#000000", "marker": ".", "z": 1},
}

# Multi-pass counter folding — see plot_roofline.py docstring for details.
COUNTER_GROUPS = {
    "flops_fp32_vec": [
        ("SQ_INSTS_VALU_ADD_F32",   1),
        ("SQ_INSTS_VALU_MUL_F32",   1),
        ("SQ_INSTS_VALU_FMA_F32",   2),
        ("SQ_INSTS_VALU_TRANS_F32", 1),
    ],
    "flops_mfma_fp32": [("SQ_INSTS_VALU_MFMA_MOPS_F32", 2)],
    "flops_mfma_fp16": [("SQ_INSTS_VALU_MFMA_MOPS_F16", 2)],
    "flops_mfma_bf16": [("SQ_INSTS_VALU_MFMA_MOPS_BF16", 2)],
    "flops_mfma_fp64": [("SQ_INSTS_VALU_MFMA_MOPS_F64", 2)],
    "bytes_read":  [("TCC_EA_RDREQ_32B_sum", 32),
                    ("TCC_EA_RDREQ_sum",     64)]
                   + [(f"TCC_EA{i}_RDREQ_sum",     64) for i in range(32)]
                   + [(f"TCC_EA{i}_RDREQ_32B_sum", 32) for i in range(32)],
    "bytes_write": [("TCC_EA_WRREQ_64B_sum", 64),
                    ("TCC_EA_WRREQ_sum",     64)]
                   + [(f"TCC_EA{i}_WRREQ_sum", 64) for i in range(32)],
}
WAVE_SIZE = 64


# ── Name lookup ─────────────────────────────────────────────────────────────
def _compile_rules() -> list[tuple[re.Pattern, str, str]]:
    return [(re.compile(p), name, cat) for p, name, cat in RENAME_RULES]


_COMPILED_RULES = _compile_rules()


def friendly_label(kernel_name: str) -> tuple[str, str]:
    """Return (short_label, category) for a raw kernel name.

    Rules are matched against the FULL kernel name. We deliberately do NOT
    pre-split on "(" any more: PyTorch/ATen kernels embed "(anonymous
    namespace)" *before* the functor that identifies them (e.g.
    `...at::native::(anonymous namespace)::launch_clamp_scalar(...`), so
    splitting on the first "(" truncated away the very token a rule needs and
    dumped every elementwise kernel into one meaningless bucket. Matching the
    full string is safe because every native rule is either ^-anchored or ends
    in ".*", so the trailing "(args)" signature is irrelevant to them.
    """
    name = kernel_name.strip()
    for pat, label, cat in _COMPILED_RULES:
        m = pat.match(name)
        if m:
            # Substitute {MT0}, {MT1}, … in label from regex captures.
            if "{MT" in label and m.groups():
                fmt = {f"MT{i}": g for i, g in enumerate(m.groups())}
                try:
                    return label.format(**fmt), cat
                except KeyError:
                    return label, cat
            return label, cat
    # Fallback: strip the trailing C++ argument signature only (the first "("
    # that opens "(int, ..." — keep internal "(anonymous namespace)" intact by
    # cutting at the last "(" instead of the first).
    base = name.split("(anonymous namespace)")[-1].split("(")[0].strip(": ")
    return (base[:40] or name[:40]), "Other"


# ── Data loading ────────────────────────────────────────────────────────────
def find_counter_csvs(root: str) -> list[str]:
    return sorted(glob.glob(os.path.join(root, "**", "*counter_collection.csv"),
                            recursive=True))


def load_pmc_wide(csv_paths: list[str]) -> pd.DataFrame:
    frames = []
    base = os.path.commonpath(csv_paths) if len(csv_paths) > 1 else ""
    for p in csv_paths:
        df = pd.read_csv(p)
        for c in df.select_dtypes(include=object).columns:
            df[c] = df[c].astype(str).str.strip('"')
        df["_pass"] = os.path.relpath(p, base) if base else os.path.basename(p)
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)

    idx_cols = [c for c in [
        "Dispatch_Id", "Agent_Id", "Kernel_Name",
        "Start_Timestamp", "End_Timestamp",
        "Grid_Size", "Workgroup_Size",
        "VGPR_Count", "SGPR_Count", "LDS_Block_Size",
        "_pass",
    ] if c in long.columns]
    wide = long.pivot_table(index=idx_cols, columns="Counter_Name",
                            values="Counter_Value", aggfunc="sum").reset_index()
    wide.columns.name = None
    for c in ("Start_Timestamp", "End_Timestamp"):
        if c in wide.columns:
            wide[c] = pd.to_numeric(wide[c], errors="coerce")
    return wide


def sum_group(df: pd.DataFrame, group) -> pd.Series:
    out = pd.Series(0.0, index=df.index)
    for col, mult in group:
        if col in df.columns:
            out = out + pd.to_numeric(df[col], errors="coerce").fillna(0.0) * mult
    return out


def compute_metrics(wide: pd.DataFrame) -> pd.DataFrame:
    out = wide.copy()
    out["duration_s"] = (out["End_Timestamp"] - out["Start_Timestamp"]) * 1e-9
    out["flops_fp32_vec"] = sum_group(out, COUNTER_GROUPS["flops_fp32_vec"]) * WAVE_SIZE
    for k in ("flops_mfma_fp32", "flops_mfma_fp16",
              "flops_mfma_bf16", "flops_mfma_fp64"):
        out[k] = sum_group(out, COUNTER_GROUPS[k])
    out["flops_total"] = (out["flops_fp32_vec"]
                          + out["flops_mfma_fp32"]
                          + out["flops_mfma_fp16"]
                          + out["flops_mfma_bf16"]
                          + out["flops_mfma_fp64"])
    out["bytes_read"]  = sum_group(out, COUNTER_GROUPS["bytes_read"])
    out["bytes_write"] = sum_group(out, COUNTER_GROUPS["bytes_write"])
    out["bytes_total"] = out["bytes_read"] + out["bytes_write"]
    out["gflops_per_s"]  = np.where(out["duration_s"] > 0,
                                    out["flops_total"] / out["duration_s"] / 1e9, 0.0)
    out["intensity_fpb"] = np.where(out["bytes_total"] > 0,
                                    out["flops_total"] / out["bytes_total"], np.nan)
    return out


def aggregate_by_kernel(disp: pd.DataFrame) -> pd.DataFrame:
    g = disp.groupby("Kernel_Name", as_index=False).agg(
        dispatches   = ("duration_s",  "size"),
        total_time_s = ("duration_s",  "sum"),
        total_flops  = ("flops_total", "sum"),
        total_bytes  = ("bytes_total", "sum"),
    )
    g["avg_gflops_per_s"] = np.where(g["total_time_s"] > 0,
                                     g["total_flops"] / g["total_time_s"] / 1e9, 0.0)
    g["intensity_fpb"]    = np.where(g["total_bytes"] > 0,
                                     g["total_flops"] / g["total_bytes"], np.nan)
    labels_cats = g["Kernel_Name"].apply(friendly_label)
    g["label"]    = [lc[0] for lc in labels_cats]
    g["category"] = [lc[1] for lc in labels_cats]
    # Collapse same friendly-label kernels (e.g. all GEMM tiles aggregate per tile size)
    g2 = g.groupby(["label", "category"], as_index=False).agg(
        dispatches   = ("dispatches",  "sum"),
        total_time_s = ("total_time_s","sum"),
        total_flops  = ("total_flops", "sum"),
        total_bytes  = ("total_bytes", "sum"),
    )
    g2["avg_gflops_per_s"] = np.where(g2["total_time_s"] > 0,
                                      g2["total_flops"] / g2["total_time_s"] / 1e9, 0.0)
    g2["intensity_fpb"]    = np.where(g2["total_bytes"] > 0,
                                      g2["total_flops"] / g2["total_bytes"], np.nan)
    return g2.sort_values("total_time_s", ascending=False).reset_index(drop=True)


# ── Plot ───────────────────────────────────────────────────────────────────
def style_axes_publication():
    """Conference-grade matplotlib defaults."""
    mpl.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["DejaVu Serif", "Liberation Serif", "Times New Roman"],
        "mathtext.fontset":   "dejavuserif",
        "font.size":          11,
        "axes.titlesize":     12,
        "axes.labelsize":     12,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    9,
        "legend.frameon":     True,
        "legend.framealpha":  0.92,
        "legend.edgecolor":   "0.6",
        "axes.linewidth":     0.9,
        "axes.grid":          True,
        "grid.linestyle":     ":",
        "grid.linewidth":     0.5,
        "grid.alpha":         0.45,
        "savefig.bbox":       "tight",
        "savefig.dpi":        300,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    })


def plot_roofline(by_kernel: pd.DataFrame, spec: Spec,
                  top_n: int, out_path: str):
    valid = by_kernel.dropna(subset=["intensity_fpb"]).copy()
    valid = valid[(valid["intensity_fpb"] > 0) & (valid["avg_gflops_per_s"] > 0)]
    if valid.empty:
        print("  ✗ No kernels with both non-zero FLOPs and bytes; cannot plot.")
        return False

    style_axes_publication()
    fig, ax = plt.subplots(figsize=(7.2, 5.4))   # ~2-column LaTeX width

    # Axis range — pad an order of magnitude either side of data.
    ai_min = max(0.05, valid["intensity_fpb"].min() / 5)
    ai_max = max(1e4,  valid["intensity_fpb"].max() * 5)
    perf_max = spec.peak_mfma_bf16_tflops * 1e3 * 1.5
    ai_grid = np.logspace(np.log10(ai_min), np.log10(ai_max), 500)
    bw_line = spec.peak_hbm_bw_gbps * ai_grid

    # Shade memory-bound region under HBM diagonal, lightly.
    knee_vec = spec.peak_vec_fp32_tflops * 1e3 / spec.peak_hbm_bw_gbps
    ax.fill_between(ai_grid, 1e-1, np.minimum(bw_line, perf_max),
                    color="0.93", zorder=0)

    # Ceilings (compute roof).
    ceilings = [
        ("FP16/BF16 MFMA peak", spec.peak_mfma_bf16_tflops * 1e3, "#D55E00"),
        ("FP32 MFMA peak",      spec.peak_mfma_fp32_tflops * 1e3, "#E69F00"),
        ("FP32 VALU peak",      spec.peak_vec_fp32_tflops  * 1e3, "#56B4E9"),
    ]
    for label, ceil, color in ceilings:
        ax.loglog(ai_grid, np.minimum(bw_line, ceil),
                  color=color, ls="--", lw=1.4, zorder=1,
                  label=f"{label} ({ceil/1000:.1f} TF/s)")
        # Knee marker where slope changes.
        knee = ceil / spec.peak_hbm_bw_gbps
        ax.plot(knee, ceil, marker="o", ms=4.5, mec=color, mfc="white",
                mew=1.2, zorder=2)

    ax.loglog(ai_grid, bw_line, color="black", lw=1.6, zorder=1,
              label=f"HBM peak ({spec.peak_hbm_bw_gbps/1000:.1f} TB/s)")

    # Per-category scatter.
    sizes_norm = valid["total_time_s"] / max(valid["total_time_s"].max(), 1e-12)
    valid["_size"] = 32 + 380 * sizes_norm

    for cat, style in CATEGORY_STYLE.items():
        sub = valid[valid["category"] == cat]
        if sub.empty:
            continue
        ax.scatter(sub["intensity_fpb"], sub["avg_gflops_per_s"],
                   s=sub["_size"], c=style["color"], marker=style["marker"],
                   edgecolor="black", linewidth=0.5,
                   alpha=0.85, zorder=style["z"] + 3,
                   label=None)   # we build a manual category legend below

    # Annotate top-N by total time, with leader lines that avoid overlapping the
    # ceilings. We just nudge labels up-right and let matplotlib draw the line.
    top = valid.head(top_n).reset_index(drop=True)
    for i, row in top.iterrows():
        # Stagger label offsets so they don't all crash into one spot.
        dx = (i % 4 + 1) * 16        # horizontal offset in points
        dy = ((i // 4) - 1) * 11     # vertical stagger
        ax.annotate(row["label"],
                    xy=(row["intensity_fpb"], row["avg_gflops_per_s"]),
                    xytext=(dx, 18 + dy), textcoords="offset points",
                    fontsize=8, color="black",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="0.65", lw=0.5, alpha=0.85),
                    arrowprops=dict(arrowstyle="-", color="0.45",
                                    lw=0.6, alpha=0.7,
                                    shrinkA=2, shrinkB=4),
                    zorder=20)

    # Manual legend: roofs + category swatches.
    roof_handles, roof_labels = ax.get_legend_handles_labels()
    cat_handles = []
    for cat, style in CATEGORY_STYLE.items():
        if (valid["category"] == cat).any():
            cat_handles.append(Line2D([0], [0],
                                      marker=style["marker"], linestyle="",
                                      color=style["color"], markersize=7,
                                      markeredgecolor="black",
                                      markeredgewidth=0.5, label=cat))
    leg1 = ax.legend(handles=roof_handles, loc="upper left",
                     title="Roof", title_fontsize=9)
    leg1.get_frame().set_linewidth(0.6)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=cat_handles, loc="lower right",
                     title="Kernel category", title_fontsize=9)
    leg2.get_frame().set_linewidth(0.6)

    # Axis limits + cosmetic.
    ax.set_xlim(ai_min, ai_max)
    ax.set_ylim(0.5, perf_max)
    ax.set_xlabel("Arithmetic intensity (FLOPs / byte)")
    ax.set_ylabel("Performance (GFLOPs / s)")
    ax.set_title(f"Roofline — ResNet-50 on {spec.name}")

    # Make tick labels use 10^x scientific notation cleanly.
    ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=10))

    fig.tight_layout()
    # Write both vector (for paper) and raster (for slides).
    base, ext = os.path.splitext(out_path)
    if ext.lower() not in (".pdf", ".svg"):
        # Default → PDF for the primary output.
        out_pdf = base + ".pdf"
    else:
        out_pdf = out_path
    out_png = base + ".png"
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"  ✓ Vector: {out_pdf}")
    print(f"  ✓ Raster: {out_png}")
    return True


# ── Console summary ────────────────────────────────────────────────────────
def print_summary(disp: pd.DataFrame, by_kernel: pd.DataFrame, top_n: int):
    total_t = disp["duration_s"].sum()
    flop_cov = (disp["flops_total"] > 0).sum() / max(len(disp), 1)
    byte_cov = (disp["bytes_total"] > 0).sum() / max(len(disp), 1)

    print("\n=== Coverage ===")
    print(f"  Dispatches profiled : {len(disp):,}")
    print(f"  Distinct kernels    : {by_kernel.shape[0]:,} (after collapsing tile variants)")
    print(f"  Total kernel time   : {total_t:.3f} s")
    print(f"  Dispatches w/ FLOPs : {flop_cov:.1%}")
    print(f"  Dispatches w/ bytes : {byte_cov:.1%}")
    if flop_cov < 0.5:
        print("  ⚠ Most dispatches show zero FLOPs. Add MFMA + FMA counters:")
        print("    SQ_INSTS_VALU_FMA_F32 SQ_INSTS_VALU_MFMA_MOPS_F32")
        print("    SQ_INSTS_VALU_MFMA_MOPS_BF16 SQ_INSTS_VALU_MFMA_MOPS_F16")
    if byte_cov < 0.5:
        print("  ⚠ Most dispatches show zero bytes. Use the unsuffixed counter")
        print("    TCC_EA_RDREQ_sum (sums all 32 channels) instead of TCC_EA0_*.")

    print(f"\n=== Top {top_n} kernels (by total time) ===")
    cols = ["label", "category", "dispatches", "total_time_s",
            "avg_gflops_per_s", "intensity_fpb"]
    rename = {"label": "Kernel",
              "category": "Category",
              "dispatches": "Dispatches",
              "total_time_s": "Time(s)",
              "avg_gflops_per_s": "GF/s",
              "intensity_fpb": "AI"}
    show = by_kernel.head(top_n).copy().rename(columns=rename)
    show.columns = [rename.get(c, c) for c in by_kernel.head(top_n).columns]
    show = by_kernel.head(top_n)[cols].rename(columns=rename)
    with pd.option_context("display.max_colwidth", 32,
                           "display.width", 140,
                           "display.float_format", lambda v: f"{v:,.2f}"):
        print(show.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="rocprofv3 output dir")
    ap.add_argument("--out", default="figures/roofline.pdf",
                    help="output path; both .pdf and .png are written")
    ap.add_argument("--top", type=int, default=12,
                    help="how many kernels to annotate / list")
    ap.add_argument("--csv-out", default=None,
                    help="optional CSV of the per-kernel table")
    args = ap.parse_args()

    csvs = find_counter_csvs(args.root)
    if not csvs:
        sys.exit(f"No counter_collection.csv files under {args.root}")
    print(f"Found {len(csvs)} counter CSV(s):")
    for p in csvs:
        print(f"  • {p}")

    print("\nPivoting long → wide …")
    wide = load_pmc_wide(csvs)
    counters = sorted(c for c in wide.columns
                      if c.startswith(("SQ_", "TCC_", "TCP_", "GRBM_")))
    print(f"  → {len(wide):,} dispatches; counters present:")
    for c in counters:
        print(f"      {c}")

    print("\nComputing per-dispatch FLOPs and bytes …")
    disp = compute_metrics(wide)

    print("\nAggregating by friendly kernel name …")
    by_kernel = aggregate_by_kernel(disp)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.csv_out:
        by_kernel.to_csv(args.csv_out, index=False)
        print(f"  ✓ per-kernel CSV → {args.csv_out}")

    print_summary(disp, by_kernel, args.top)

    print("\nPlotting roofline …")
    plot_roofline(by_kernel, Spec(), args.top, args.out)


if __name__ == "__main__":
    main()
