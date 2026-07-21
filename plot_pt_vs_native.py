#!/usr/bin/env python3
"""
plot_pt_vs_native.py — Native-HIP vs PyTorch-ROCm comparison on MI300X.

Two panels, paper-styled (paper_style.py, canonical Native=vermillion /
PyTorch=blue):

  (a) Roofline OVERLAY — every kernel of both implementations on one set of
      MI300X ceilings. Individual kernels are a faint cloud; each kernel
      *category* gets one bold marker (sized by its total GPU time) at the
      category's aggregate (AI, GFLOP/s), with a NAMED, non-overlapping label
      placed by a small force-directed de-overlap pass + leader lines.
  (b) Cache-thrashing view — time-weighted L2 hit-rate per kernel category,
      grouped bars. Low L2 hit % under high HBM traffic = thrashing.

Inputs (override on CLI):
  --native-bykernel  figures/by_kernel_native_52295979.csv     (roofline scatter)
  --pt-bykernel      profiles/pt_amd_52438059/by_kernel_pytorch.csv
  --native-summary   profiles/amd_52295979/roofline_summary.csv (cache + per-kernel)
  --pt-summary       profiles/pt_amd_52438059/roofline_summary.csv

Output: figures/pt_vs_native_<tag>.{pdf,png}
"""
from __future__ import annotations
import argparse, csv, math
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import paper_style as ps

# ── MI300X ceilings (match plot_roofline.Spec / analyze_roofline) ───────────
HBM_GBPS        = 5300.0
VALU_FP32_TF    = 75.8
MFMA_FP32_TF    = 163.4
MFMA_BF16_TF    = 1307.4


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(r, k):
    try:
        return float(r.get(k, "") or "nan")
    except ValueError:
        return float("nan")


# ── Category aggregation ────────────────────────────────────────────────────
# Order = rough pipeline order; only categories present are drawn.
CAT_ORDER = ["Convolution", "GEMM", "BatchNorm", "Element-wise",
             "Reduction", "Layout transform", "Optimizer", "Memory op", "Other"]


def aggregate_by_category(rows):
    """Sum FLOPs/bytes/time per category, derive aggregate (AI, GFLOP/s, time).

    Aggregate AI   = Σflops / Σbytes   (FLOP/byte)
    Aggregate perf = (Σflops/1e9) / Σtime   (GFLOP/s)
    Only kernels with flops>0 and bytes>0 contribute (memory-only layout/
    fill kernels do no FLOPs, so they have no place on a FLOP roofline).
    """
    fl = defaultdict(float); by = defaultdict(float); tm = defaultdict(float)
    for r in rows:
        f, b, t = (fnum(r, "total_flops"), fnum(r, "total_bytes"),
                   fnum(r, "total_time_s"))
        if not (f > 0 and b > 0 and t > 0):
            continue
        c = r.get("category", "Other") or "Other"
        fl[c] += f; by[c] += b; tm[c] += t
    out = []
    for c in CAT_ORDER:
        if c not in fl:
            continue
        out.append(dict(cat=c, ai=fl[c] / by[c],
                        perf=(fl[c] / 1e9) / tm[c], time=tm[c]))
    return out


# ── Deterministic callout labels (no overlap, no clipping, dpi-independent) ──
def place_labels(ax, fig, items, *, fontsize=8.0):
    """Label each item from a tidy callout band over the empty upper plot area.

    items: list of dict(text, targets=[(x,y,color),...]). The markers cluster
    in the lower-left, leaving the upper plot empty — so labels are laid out
    there: sorted by arithmetic intensity and spread evenly across two
    staggered rows, with thin leader lines (coloured per implementation)
    dropping to each marker. All positions are in dpi-independent AXES-FRACTION
    coords, so they survive the 150→600 dpi save; even spacing => no overlap.
    """
    # sort by mean target AI so leader lines run left→right without crossing
    order = sorted(range(len(items)),
                   key=lambda i: sum(t[0] for t in items[i]["targets"])
                                 / len(items[i]["targets"]))
    n = len(order)
    xf = np.linspace(0.05, 0.95, n) if n > 1 else [0.5]
    row_yf = [0.72, 0.60]                       # two staggered rows (axes frac)

    # transform from axes-fraction → data, so leader lines (drawn in data
    # coords) are dpi-independent too.
    ax2data = lambda fx, fy: ax.transData.inverted().transform(
        ax.transAxes.transform((fx, fy)))

    for slot, idx in enumerate(order):
        it = items[idx]
        fx, fy = xf[slot], row_yf[slot % 2]
        ax.annotate(
            it["text"], xy=(fx, fy), xycoords="axes fraction",
            xytext=(fx, fy), textcoords="axes fraction",
            fontsize=fontsize, color="#1a1a1a", fontweight="bold",
            ha="center", va="center", zorder=21,
            bbox=dict(boxstyle="round,pad=0.22", fc="white",
                      ec="0.55", lw=0.8, alpha=0.96))
        x0, y0 = ax2data(fx, fy)
        for (tx, ty, tcol) in it["targets"]:
            ax.plot([x0, tx], [y0, ty], color=tcol, lw=0.6,
                    alpha=0.6, zorder=8, solid_capstyle="round")


# ── Panel (a): roofline overlay ─────────────────────────────────────────────
def panel_roofline(ax, fig, native, pt):
    ai_lo, ai_hi = 0.05, 1e4
    ai = np.logspace(math.log10(ai_lo), math.log10(ai_hi), 400)
    hbm = HBM_GBPS * ai                       # GFLOP/s on the memory diagonal

    # memory-bound shading + ceilings (GFLOP/s)
    perf_top = MFMA_BF16_TF * 1e3 * 1.4
    ax.fill_between(ai, 1e-1, np.minimum(hbm, perf_top), color="0.95", zorder=0)
    for tf, lab, c, ls in [
        (MFMA_BF16_TF, "FP16/BF16 MFMA 1307 TF", "0.35", "--"),
        (MFMA_FP32_TF, "FP32 MFMA 163 TF",       "0.5",  "--"),
        (VALU_FP32_TF, "FP32 VALU 76 TF",        "0.65", ":"),
    ]:
        ax.plot(ai, np.minimum(hbm, tf * 1e3), color=c, ls=ls, lw=1.1, zorder=1)
        ax.text(ai_hi / 1.5, min(tf * 1e3, perf_top) * 1.04, lab, ha="right",
                va="bottom", fontsize=6.6, color=c)
    ax.plot(ai, hbm, color="black", lw=1.3, zorder=1)
    ax.text(ai_lo * 1.3, HBM_GBPS * ai_lo * 1.3 * 1.15, f"HBM {HBM_GBPS/1000:.1f} TB/s",
            rotation=34, fontsize=6.8, color="black", va="bottom")

    # faint individual-kernel cloud (context, no labels)
    def cloud(rows, color, marker):
        xs, ys = [], []
        for r in rows:
            x, y = fnum(r, "intensity_fpb"), fnum(r, "avg_gflops_per_s")
            if x > 0 and y > 0:
                xs.append(x); ys.append(y)
        ax.scatter(xs, ys, s=14, c=color, marker=marker, edgecolor="none",
                   alpha=0.28, zorder=4)
    cloud(native, ps.NATIVE,  "o")
    cloud(pt,     ps.PYTORCH, "s")

    # bold category centroids (sized by total category time)
    nat_agg = {a["cat"]: a for a in aggregate_by_category(native)}
    pt_agg  = {a["cat"]: a for a in aggregate_by_category(pt)}
    tmax = max([a["time"] for a in list(nat_agg.values()) + list(pt_agg.values())],
               default=1.0) or 1.0
    for agg, color, marker in [(nat_agg, ps.NATIVE,  "o"),
                               (pt_agg,  ps.PYTORCH, "s")]:
        for a in agg.values():
            s = 70 + 620 * (a["time"] / tmax)
            ax.scatter([a["ai"]], [a["perf"]], s=s, c=color, marker=marker,
                       edgecolor="black", linewidth=0.7, alpha=0.95, zorder=10)

    # ONE label per category, anchored at the geo-mean of the impl(s) that have
    # it, pointing leader lines at each implementation's marker.
    label_items = []
    for cat in CAT_ORDER:
        targets = []
        if cat in nat_agg:
            targets.append((nat_agg[cat]["ai"], nat_agg[cat]["perf"], ps.NATIVE))
        if cat in pt_agg:
            targets.append((pt_agg[cat]["ai"], pt_agg[cat]["perf"], ps.PYTORCH))
        if not targets:
            continue
        ax_ = math.exp(sum(math.log(t[0]) for t in targets) / len(targets))
        ay_ = math.exp(sum(math.log(t[1]) for t in targets) / len(targets))
        label_items.append(dict(text=cat, ax_=ax_, ay_=ay_, targets=targets))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(ai_lo, ai_hi); ax.set_ylim(1.0, perf_top)
    ax.set_xlabel(ps.LABELS.ARITH_INTENSITY)
    ax.set_ylabel("Performance (GFLOPs/s)")
    ax.set_title("(a) Roofline overlay — by kernel category")
    ps.cleanup(ax)

    # legend: implementation key (marker size = GPU time of the category)
    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor=ps.NATIVE,
               markeredgecolor="black", markersize=8, label="Native HIP"),
        Line2D([], [], marker="s", color="none", markerfacecolor=ps.PYTORCH,
               markeredgecolor="black", markersize=8, label="PyTorch"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True,
              framealpha=0.92, fontsize=8,
              title="marker size ∝ category GPU time", title_fontsize=7)

    # de-overlap the named labels last (needs final axes geometry)
    place_labels(ax, fig, label_items)


# ── Panel (b): L2 cache-hit by category (thrashing) ─────────────────────────
def cache_by_category(summary_rows, metric="l2_hit_pct"):
    """time-weighted mean hit-rate per category (weight = total_ms)."""
    num = defaultdict(float); den = defaultdict(float)
    for r in summary_rows:
        cat = r.get("category", "Other")
        v, w = fnum(r, metric), fnum(r, "total_ms")
        if v == v and w == w and w > 0:        # skip NaN
            num[cat] += v * w; den[cat] += w
    return {c: num[c] / den[c] for c in num if den[c] > 0}


def panel_cache(ax, native_sum, pt_sum, metric="l2_hit_pct"):
    nat = cache_by_category(native_sum, metric)
    pt  = cache_by_category(pt_sum,  metric)
    cats = [c for c in CAT_ORDER if c in nat or c in pt]
    x = np.arange(len(cats)); w = 0.38
    nv = [nat.get(c, np.nan) for c in cats]
    pv = [pt.get(c,  np.nan) for c in cats]
    ax.bar(x - w/2, nv, w, color=ps.NATIVE,  edgecolor="black", lw=0.4,
           label="Native HIP")
    ax.bar(x + w/2, pv, w, color=ps.PYTORCH, edgecolor="black", lw=0.4,
           label="PyTorch")
    ax.axhline(50, **ps.IDEAL)                 # rough "thrashing below this" guide
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=8, rotation=28, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("L2 hit rate (%, time-weighted)")
    ax.set_title("(b) Cache thrashing — L2 hit by category")
    ps.cleanup(ax)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native-bykernel", default="figures/by_kernel_native_52295979.csv")
    ap.add_argument("--pt-bykernel",     default="profiles/pt_amd_52438059/by_kernel_pytorch.csv")
    ap.add_argument("--native-summary",  default="profiles/amd_52295979/roofline_summary.csv")
    ap.add_argument("--pt-summary",      default="profiles/pt_amd_52438059/roofline_summary.csv")
    ap.add_argument("--tag", default="52295979_vs_52438059")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()

    ps.setup(); ps.big_fonts()
    # Wider canvas + more room for panel (a) so the named labels are readable.
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(10.2, 4.6),
        gridspec_kw=dict(width_ratios=[1.55, 1.0], wspace=0.22,
                         left=0.07, right=0.985, top=0.88, bottom=0.18))
    panel_roofline(axL, fig, read_rows(args.native_bykernel),
                   read_rows(args.pt_bykernel))
    panel_cache(axR, read_rows(args.native_summary), read_rows(args.pt_summary))
    for a in (axL, axR):
        a.title.set_fontsize(10.5); a.title.set_fontweight("bold")
    fig.suptitle("ResNet-50 on MI300X — Native HIP vs PyTorch-ROCm",
                 fontweight="bold", fontsize=12, y=0.965)
    ps.save(fig, f"pt_vs_native_{args.tag}", out_dir=args.out_dir)
    print(f"  -> {args.out_dir}/pt_vs_native_{args.tag}.pdf (+ .png)")


if __name__ == "__main__":
    main()
