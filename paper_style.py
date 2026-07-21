"""
IEEEtran-friendly matplotlib style.

One-line use:
    import paper_style as ps
    ps.setup()
    fig, ax = plt.subplots(figsize=ps.figsize("col"))
    ax.plot(...); ps.cleanup(ax)
    ps.save(fig, "fig_overhead")    # writes .pdf + .png

Why these defaults
------------------
* IEEEtran columns are 3.50 in (single) / 7.16 in (double). Build figures at the
  *exact print width* — never \\scale/\\resizebox in LaTeX, that destroys
  typography. Heights default to the golden-ratio aspect; override only the
  height when stacking panels.
* Body text in IEEE is 10 pt; figure typography sits 1-2 pt below that so it
  reads as the same family without competing with the body.
* TrueType (fonttype=42) — some IEEE/ACM submission systems silently reject
  Type-3 PDF fonts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt


# ── Canonical widths (IEEEtran, inches) ─────────────────────────────────────
# Override these if you target a different venue (ACM: 3.33/7.00,
# LNCS: 4.80/4.80, arXiv revtex: 3.40, etc.).
WIDTH_COL  = 3.50      # single column
WIDTH_TEXT = 7.16      # double column / full text width
GOLDEN     = 0.618     # ratio for default heights


# ── Default heights by layout (good starting points) ────────────────────────
_DEFAULT_HEIGHT = {
    "col":  WIDTH_COL  * GOLDEN,    # ≈ 2.16 in — single col, single panel
    "half": WIDTH_COL  * GOLDEN,    # half text width = same as single col
    "text": 2.4,                    # double col, 1xN side-by-side panels
}

# ── Colour-blind-safe palette (Wong, 2011) ──────────────────────────────────
# Use deliberately instead of matplotlib's default tab10. If you change the
# cycler in one figure, do it consistently across the paper.
PALETTE = [
    "#0072B2",   # blue
    "#D55E00",   # vermillion
    "#009E73",   # bluish green
    "#CC79A7",   # reddish purple
    "#F0E442",   # yellow
    "#56B4E9",   # sky blue
    "#E69F00",   # orange
    "#000000",   # black
]


# ── Public API ──────────────────────────────────────────────────────────────
def setup() -> None:
    """Apply all the paper-typography + production rcParams in one shot."""
    mpl.rcParams.update({
        # Typography (figure text sits 1-2 pt below IEEE 10 pt body)
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size":        8.5,
        "axes.titlesize":   9.0,
        "axes.labelsize":   8.5,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",   # bold axis labels for paper readability
        "xtick.labelsize":  7.5,
        "ytick.labelsize":  7.5,
        "legend.fontsize":  8.0,
        # Lines/markers — visible at print scale, not chunky
        "lines.linewidth":  1.2,
        "lines.markersize": 3.4,
        "axes.linewidth":   0.7,
        # Spines: keep left+bottom, drop top+right (less ink, more signal)
        "axes.spines.top":   False,
        "axes.spines.right": False,
        # Grid: dashed, low-contrast, behind the data
        "axes.grid":        True,
        "axes.axisbelow":   True,
        "grid.linestyle":   "--",
        "grid.linewidth":   0.5,
        "grid.alpha":       0.35,
        # Ticks: minor ticks on, ticks point in (paper convention)
        "xtick.direction":  "in",
        "ytick.direction":  "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
        # Legend: thin frame, light, no shadow
        "legend.frameon":   True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#888888",
        "legend.fancybox":  False,
        "legend.borderpad": 0.35,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.2,
        # Production (non-negotiable for submission)
        "pdf.fonttype":     42,     # TrueType — some venues reject Type-3
        "ps.fonttype":      42,
        "figure.dpi":       150,    # screen preview
        "savefig.dpi":      600,    # raster fallback PNG quality
        "savefig.bbox":     "tight",
        "savefig.pad_inches": 0.03,
        # Colour cycler
        "axes.prop_cycle":  mpl.cycler(color=PALETTE),
    })


def figsize(layout: str = "col", height: Optional[float] = None,
            h: Optional[float] = None) -> Tuple[float, float]:
    """Return (width, height) in inches for a named layout.

    Parameters
    ----------
    layout : {"col", "half", "text"}
        col  -> 3.50 in (single column, IEEEtran)
        half -> 3.50 in (half text width; same as col)
        text -> 7.16 in (double column / full text width)
    height : float, optional
        Override the default height (golden ratio for col/half, 2.4 in for text).
    """
    if h is not None and height is None:   # back-compat alias used by some callers
        height = h
    if layout not in _DEFAULT_HEIGHT:
        raise ValueError(f"unknown layout {layout!r}; "
                         f"use one of {sorted(_DEFAULT_HEIGHT)}")
    width = WIDTH_TEXT if layout == "text" else WIDTH_COL
    return (width, float(height) if height is not None else _DEFAULT_HEIGHT[layout])


def cleanup(ax) -> None:
    """Standard per-axes cleanup: grid layering + visible minor ticks.

    Call after plotting and before legend(). Idempotent.
    """
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.20)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in")
    # Pull tick labels in slightly so they don't push the layout
    ax.tick_params(pad=2.5)


def save(fig, name: str, *, out_dir: str | Path = ".", formats: Iterable[str] = ("pdf", "png")) -> None:
    """Write fig to <out_dir>/<name>.<ext> for each requested format.

    Defaults to PDF (vector, paper-ready) + PNG (preview / talk slides).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(out / f"{name}.{ext}")


# ──────────────────────────────────────────────────────────────────────────
# Standardised plot heading
#
# Two-line title used across the paper's roofline / per-kernel / scaling plots
# so every figure carries the same "<vendor> <gpu>, <model>" identity in the
# same place + the same secondary tags below it.
#
# Line 1 (bold):  "AMD MI300X, ResNet 50 v1.5"
# Line 2 (light): "Peak: 163 TFLOPS · BW: 5.3 TB/s"
#
# Examples
# --------
# Roofline header from raw GPU peaks:
#     ps.plot_heading(ax, vendor="AMD", gpu="MI300X",
#                     peak_tflops=163, bw_tbs=5.3)
#
# Scaling-experiment header with a custom subtitle:
#     ps.plot_heading(ax, vendor="AMD", gpu="MI300X",
#                     sub_parts=["FP32", "8x SPX", "Tiny-ImageNet"])
#
# NVIDIA sibling roofline:
#     ps.plot_heading(ax, vendor="NVIDIA", gpu="H100",
#                     peak_tflops=67, bw_tbs=3.35)
# ──────────────────────────────────────────────────────────────────────────
def plot_heading(ax, *, vendor: str, gpu: str,
                 model: str = "ResNet 50 v1.5",
                 peak_tflops: Optional[float] = None,
                 bw_tbs: Optional[float] = None,
                 sub_parts: Optional[Iterable[str]] = None,
                 loc: str = "left") -> None:
    """Apply the standardised two-line heading to a matplotlib Axes.

    Parameters
    ----------
    ax            : the Axes to put the heading on.
    vendor, gpu   : identity tags (e.g. "AMD" + "MI300X").
    model         : workload identifier; default "ResNet 50 v1.5".
    peak_tflops   : if given, becomes "Peak: <n> TFLOPS" on line 2.
    bw_tbs        : if given, becomes "BW: <n> TB/s" on line 2.
    sub_parts     : explicit list of subtitle tags (overrides peak/bw).
                    Strings are joined with " · " for visual separators.
    loc           : "left" (paper default) / "center" / "right".
    """
    main = f"{vendor} {gpu}, {model}"

    if sub_parts is None:
        sub_parts = []
        if peak_tflops is not None:
            sub_parts.append(f"Peak: {peak_tflops:.0f} TFLOPS")
        if bw_tbs is not None:
            sub_parts.append(f"BW: {bw_tbs} TB/s")
    sub = "  ·  ".join(str(p) for p in sub_parts) if sub_parts else ""

    # Reserve extra pad above the axes for the subtitle line, then place the
    # subtitle inside that pad just below the bold main title.
    pad = 16 if sub else 4
    ax.set_title(main, loc=loc, fontweight="bold", pad=pad)
    if sub:
        x = {"left": 0.0, "center": 0.5, "right": 1.0}[loc]
        ha = loc
        sub_size = max(8, int(plt.rcParams.get("axes.titlesize", 13)) - 3)
        ax.text(x, 1.012, sub, transform=ax.transAxes,
                ha=ha, va="bottom",
                fontsize=sub_size, color="#555555", fontstyle="italic")


def figure_heading(fig, *, vendor: str, gpu: str,
                   model: str = "ResNet 50 v1.5",
                   peak_tflops: Optional[float] = None,
                   bw_tbs: Optional[float] = None,
                   sub_parts: Optional[Iterable[str]] = None,
                   y_main: Optional[float] = None,
                   y_sub:  Optional[float] = None,
                   gap: float = 0.012) -> None:
    """Centered two-line heading at the top of the whole FIGURE.

    The heading is placed *just above* `fig.subplotpars.top` so it hugs the
    plot area regardless of where you set `top=` — no large gap. Override
    `y_main` / `y_sub` only for unusual layouts.
    """
    main = f"{vendor} {gpu}, {model}"
    if sub_parts is None:
        sub_parts = []
        if peak_tflops is not None: sub_parts.append(f"Peak: {peak_tflops:.0f} TFLOPS")
        if bw_tbs      is not None: sub_parts.append(f"BW: {bw_tbs} TB/s")
    sub = "  ·  ".join(str(p) for p in sub_parts) if sub_parts else ""

    main_size = plt.rcParams.get("axes.titlesize", 13)
    sub_size  = max(8, int(main_size) - 3)

    # Anchor the heading to the actual plot top so the gap is small + stable.
    plot_top = float(fig.subplotpars.top)
    # Vertical space (in figure fractions) each text line consumes.
    main_line_h = (main_size + 2) / max(1.0, fig.get_figheight() * 72.0)
    sub_line_h  = (sub_size  + 2) / max(1.0, fig.get_figheight() * 72.0)
    if y_sub  is None:
        y_sub  = plot_top + gap
    if y_main is None:
        y_main = (y_sub + sub_line_h) if sub else (plot_top + gap)
    # Clamp so we never push outside the figure (top is 1.0).
    y_main = min(0.995, y_main)
    y_sub  = min(0.995 - main_line_h, y_sub)

    fig.suptitle(main, x=0.5, y=y_main, ha="center", va="top",
                 fontweight="bold", fontsize=main_size)
    if sub:
        fig.text(0.5, y_sub, sub, ha="center", va="top",
                 fontsize=sub_size, color="#555555", fontstyle="italic")


# ──────────────────────────────────────────────────────────────────────────
# Canonical axis labels — one string per concept, used across every plot so
# the same idea ("throughput", "scaling efficiency", "arithmetic intensity")
# reads identically in every figure. Add to this list as needed; never
# duplicate the concept with a different wording in a plot.
# ──────────────────────────────────────────────────────────────────────────
class LABELS:
    # Scaling
    GPUS               = "Number of GPUs"
    THROUGHPUT         = "Throughput (images/sec)"
    THROUGHPUT_PER_GPU = "Per-GPU throughput (images/sec)"
    SPEEDUP            = "Speedup"
    EFFICIENCY         = "Scaling efficiency (%)"
    EPOCH_TIME         = "Epoch time (s)"
    # Roofline / per-kernel
    ARITH_INTENSITY    = "Arithmetic intensity (FLOPs/byte)"
    PERFORMANCE_TF     = "Performance (TFLOPS)"
    KERNEL_TIME        = "GPU kernel time (ms)"
    CACHE_HIT          = "Cache hit rate (%)"
    MFMA_OPS           = "MFMA ops per dispatch"
    # Misc
    BATCH_SIZE         = "Batch size"
    EPOCH              = "Epoch"
    THROUGHPUT_RATIO   = "Throughput ratio (PyTorch / HIP)"
    # Sparsity characterization
    ACT_SPARSITY       = "Activation sparsity (% zeros)"
    RELU_LAYER         = "ReLU layer (forward order)"


# ════════════════════════════════════════════════════════════════════════════
# H100 cross-vendor template (§6 of the plot_style spec)
# ════════════════════════════════════════════════════════════════════════════
# `setup()` above is the IEEEtran spec (8.5 pt body) used by the
# paper_data/* portable plotters. `apply_template()` below is the
# cross-vendor H100 style (12 pt base, Wong palette, centered bold titles,
# 600-dpi PDF+PNG output) used by resnet50_<vendor>_plots.py.
# Pick one per script — don't mix.
# ────────────────────────────────────────────────────────────────────────────

# §6b — Vendor palette (Wong / Okabe-Ito, colorblind-safe)
VENDOR_COLORS = {
    "nvidia_h100":  "#0072B2",
    "nvidia_a100":  "#56B4E9",
    "amd_mi300x":   "#D55E00",
    "amd_mi250":    "#E69F00",
    "intel_gaudi3": "#009E73",
    "intel_gaudi2": "#33CC99",
}
VENDOR_MARKERS = {
    "nvidia_h100": "s", "nvidia_a100": "s",
    "amd_mi300x":  "o", "amd_mi250":  "o",
    "intel_gaudi3": "^", "intel_gaudi2": "^",
}
VENDOR_LABELS = {
    "nvidia_h100":  "NVIDIA H100",  "nvidia_a100":  "NVIDIA A100",
    "amd_mi300x":   "AMD MI300X",   "amd_mi250":    "AMD MI250",
    "intel_gaudi3": "Intel Gaudi3", "intel_gaudi2": "Intel Gaudi2",
}
IDEAL_COLOR = "#666666"     # ideal / 100% / peak / ridge reference lines
ANN_COLOR   = "#222222"     # free-floating annotations (titles outside data)


# §6c — Axis-label vocabulary (identical strings everywhere). Dict form per
# template spec; the LABELS class above stays as the legacy interface.
AXIS = {
    "gpus":             "Number of GPUs",
    "throughput":       "Throughput (images/sec)",
    "throughput_per_g": "Throughput per GPU (images/sec/GPU)",
    "efficiency":       "Parallel efficiency (%)",
    "speedup":          r"Speedup $T_1 / T_N$",
    "epoch_time":       "Wall time per epoch (s)",
    "ai":               "Arithmetic intensity (FLOPs/byte)",
    "performance":      r"Performance (TFLOP$\cdot$s$^{-1}$)",
    "local_batch":      "Per-GPU (local) batch size",
    "global_batch":     "Global batch size",
    "cache_hit":        "Cache hit rate (%)",
    "kernel_time":      "Cumulative GPU kernel time (ms)",
    "mfma_ops":         "MFMA ops per dispatch (avg)",
    "throughput_ratio": r"Throughput ratio (PyTorch $/$ HIP)",
}


def apply_template() -> None:
    """Apply the H100 cross-vendor rcParams (§6a of the plot_style spec).

    Differences from `setup()` (IEEEtran):
      - 12 pt base, 13 pt bold axis labels, 14 pt bold centered titles
      - Dotted grid (α 0.25) instead of dashed (α 0.35)
      - Larger marker size and line width (slides + paper)
      - 600 dpi savefig, bbox=tight, pad 0.05
    """
    mpl.rcParams.update({
        "font.family":         "serif",
        "font.serif":          ["DejaVu Serif", "Times New Roman"],
        "font.size":           12,
        "axes.labelsize":      13,
        "axes.labelweight":    "bold",
        "axes.titlesize":      14,
        "axes.titleweight":    "bold",
        "figure.titlesize":    13,
        "figure.titleweight":  "bold",
        "legend.fontsize":     10,
        "legend.frameon":      True,
        "legend.framealpha":   0.95,
        "legend.edgecolor":    "0.6",
        "xtick.labelsize":     11,
        "ytick.labelsize":     11,
        "xtick.direction":     "in",
        "ytick.direction":     "in",
        "xtick.major.size":    4,
        "ytick.major.size":    4,
        "axes.linewidth":      0.9,
        "lines.linewidth":     2.0,
        "lines.markersize":    8,
        "figure.dpi":          150,
        "savefig.dpi":         600,
        "savefig.pad_inches":  0.05,
        "savefig.bbox":        "tight",
        "axes.grid":           True,
        "grid.alpha":          0.25,
        "grid.linestyle":      ":",
        "grid.linewidth":      0.6,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "pdf.fonttype":        42,
        "ps.fonttype":         42,
    })


def save_vendor(fig, stem: str, vendor: str,
                out_dir: str | Path = ".") -> None:
    """Save figure as `<stem>_<vendor>.{pdf,png}` per the §6d filename rule.

    PDF (vector) + PNG (600 dpi raster), both bbox=tight pad=0.05. Closes
    the figure after writing.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{stem}_{vendor}"
    pdf  = out / f"{name}.pdf"
    png  = out / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def series(ax, x, y, color: str, marker: str, label: str, **kwargs) -> None:
    """Standard series line: marker with white halo edge (template default).

    Any extra `kwargs` pass through to `ax.plot`.
    """
    ax.plot(x, y, marker=marker, color=color,
            markeredgecolor="white", markeredgewidth=0.8,
            label=label, **kwargs)


def annotate(ax, x, y, text: str, color: str,
             dy: float = 10, fontsize: int = 10) -> None:
    """Vendor-coloured annotation above a marker (normal weight per spec)."""
    ax.annotate(text, (x, y), xytext=(0, dy),
                textcoords="offset points", ha="center",
                fontsize=fontsize, color=color)


def shared_bottom_legend(fig, *axes, fontsize: int = 9,
                         y: float = -0.04, ncol: Optional[int] = None) -> None:
    """One shared frameless legend below the figure (template convention).

    Collects handles/labels from all passed axes, de-duplicates by label
    while preserving order, then places them centered below the figure.
    The caller is responsible for reserving the bottom strip, e.g.
    `fig.tight_layout(rect=(0, 0.04, 1, 1))`.
    """
    seen: set = set()
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li in seen:
                continue
            seen.add(li); handles.append(hi); labels.append(li)
    if not handles:
        return
    fig.legend(handles, labels,
               loc="lower center", ncol=ncol or len(labels),
               bbox_to_anchor=(0.5, y), frameon=False, fontsize=fontsize)


# ===========================================================================
# Shared SCALING TOOLKIT (added 2026-06-03) — one source of truth so every
# scaling figure looks identical. See styling.md. Additive: nothing above is
# changed, so the existing plot_heading / shared_bottom_legend / save_vendor /
# VENDOR_* API used by the other figure scripts keeps working.
# ===========================================================================

# Semantic colour aliases (unified mapping): native/custom implementation =
# vermillion, PyTorch = blue. PALETTE cycle order is unchanged (blue first);
# only the semantic mapping is fixed here.
NATIVE  = PALETTE[1]    # custom CUDA / HIP / "native"  -> vermillion (#D55E00)
PYTORCH = PALETTE[0]    # PyTorch / NGC                 -> blue (#0072B2)
GREEN, PURPLE, YELLOW, SKY, ORANGE, BLACK = PALETTE[2:8]

# scaling mode -> line/marker encoding (map your data's mode names onto these)
MODE_STYLES = {
    "strong": dict(ls="-",  marker="D", fill=True,  label="Strong"),
    "weak":   dict(ls="--", marker="o", fill=False, label="Weak"),
}
# ideal-linear guide (throughput panel) / 100% line (efficiency panel)
IDEAL = dict(color="0.45", ls=":", lw=1.4)


def big_fonts() -> None:
    """Bump sizes on top of setup() for a roomier double-column 2-panel canvas."""
    mpl.rcParams.update({
        "font.size": 10.0, "axes.titlesize": 11.0, "axes.labelsize": 10.5,
        "xtick.labelsize": 9.0, "ytick.labelsize": 9.0, "legend.fontsize": 9.5,
        "lines.linewidth": 2.0, "lines.markersize": 6.5, "axes.linewidth": 0.9,
    })


def scaling_fonts() -> None:
    """big_fonts() + bold, slightly larger labels/titles/ticks — the shared
    scaling-figure typography. Call bold_ticks(ax) per axis for bold tick labels
    (matplotlib has no rcParam for tick-label weight)."""
    big_fonts()
    mpl.rcParams.update({
        "axes.labelsize": 12.5, "axes.labelweight": "bold",
        "axes.titlesize": 12.0, "axes.titleweight": "bold",
        "xtick.labelsize": 11.0, "ytick.labelsize": 11.0,
    })


def bold_ticks(ax):
    """Bold the tick labels (call AFTER set_xticklabels / set_xlim)."""
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    return ax


def series_kw(mode_style: dict, color: str, ms: float = 6.5) -> dict:
    """plot/errorbar kwargs for one scaling series in the shared encoding
    (open marker for weak, filled for strong; white face, 1.6 edge)."""
    return dict(
        color=color, linestyle=mode_style["ls"], marker=mode_style["marker"],
        markersize=ms, markeredgecolor=color, markeredgewidth=1.6,
        markerfacecolor=(color if mode_style["fill"] else "white"),
        clip_on=False,
    )


def draw_series(ax, xs, ys, color, mode_style, *, yerr=None, capsize=3, zorder=3, ms=6.5):
    """Draw one scaling curve (errorbar when yerr given) in the shared style."""
    kw = series_kw(mode_style, color, ms=ms)
    if yerr is not None and any(yerr):
        return ax.errorbar(xs, ys, yerr=yerr, capsize=capsize, zorder=zorder, **kw)
    return ax.plot(xs, ys, zorder=zorder, **kw)


def dual_legend(target, impls, *, modes=True, ideal=True, below=False,
                loc="upper left", ncol=None, key_color="0.25", **kw):
    """One factored legend: implementation colour keys (+ strong/weak style keys
    + ideal). `impls` = [(color, label), ...]; `target` is an Axes or the Figure.
    `below=True` centres it under the panels; default keeps it in-panel.
    `modes=False` for a single-mode figure."""
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, lw=2.6, label=l) for c, l in impls]
    if modes:
        for key in ("strong", "weak"):
            m = MODE_STYLES[key]
            handles.append(Line2D(
                [], [], color=key_color, lw=2.0, ls=m["ls"], marker=m["marker"],
                markersize=6.5, markeredgecolor=key_color, markeredgewidth=1.6,
                markerfacecolor=(key_color if m["fill"] else "white"),
                label=m["label"]))
    if ideal:
        handles.append(Line2D([], [], label="Ideal", **IDEAL))
    labels = [h.get_label() for h in handles]
    if below:
        kw.setdefault("bbox_to_anchor", (0.5, -0.02))
        kw.setdefault("columnspacing", 1.1)
        kw.setdefault("handlelength", 1.9)
        return target.legend(handles, labels, loc="lower center",
                             ncol=ncol or len(handles), frameon=False, **kw)
    kw.setdefault("handlelength", 2.4)
    kw.setdefault("borderaxespad", 0.4)
    return target.legend(handles, labels, loc=loc, ncol=ncol or 1,
                        frameon=False, **kw)
