#!/usr/bin/env python3
"""Memory-hierarchy traffic decomposition for MI300X (gfx942) — the AMD analog
of the H100 `fig_traffic_decomposition.py`.

For each canonical kernel category × stack (Native HIP vs PyTorch-ROCm) we draw
one horizontal bar whose length is the total memory traffic, broken into three
stacked segments:

    [   L1-served   |   L2-served   |   DRAM-served   ]

On CDNA3 the cache hierarchy is TCP (vector L1) -> TCC (L2) -> EA (HBM/DRAM),
so the NVIDIA l1tex/lts sector decomposition maps 1:1 onto rocprofv3 counters:

  L1-served   = (TCP_TOTAL_CACHE_ACCESSES - TCP->TCC requests) x 64 B   (L1 hits)
  L2-served   = (TCP->TCC requests        - TCC->EA  requests) x 64 B   (hit in L2)
  DRAM-served =  TCC->EA requests                              x 64 B   (L2 miss -> HBM)

where
  TCP->TCC requests = TCP_TCC_READ_REQ_sum + TCP_TCC_WRITE_REQ_sum
                      (+ TCP_TCC_ATOMIC_WITH_RET_REQ_sum when collected)
  TCC->EA  requests = TCC_EA0_RDREQ_sum    + TCC_EA0_WRREQ_sum

The 64 B/request convention is the same one analyze_roofline.py uses for HBM
bytes (matches rocprof-compute / AMD's own roofline scripts), so the absolute
totals here are consistent with every other AMD number in this repo.

Raw data is the rocprofv3 multi-pass `*_counter_collection.csv` already present
under profiles/. Each requested counter lands in exactly one pmc_* pass; the
shared loader (plot_roofline.load_pmc_wide) pivots long->wide and we sum each
counter's full-run total per kernel, then canonicalise with the SAME
friendly_label() used by the roofline plots so categories line up 1:1 with the
H100 figure and with native-vs-PyTorch.

Output: figures/traffic_decomposition_amd/fig_traffic_decomposition_amd.{pdf,png}
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paper_style as ps                                          # noqa: E402
from plot_roofline import find_counter_csvs, load_pmc_wide, friendly_label  # noqa: E402

REQ_BYTES = 64                      # CDNA3 cache-line / request granularity (B)

# rocprofv3 counter names (gfx942). The ATOMIC counter is optional — some pmc
# sets in this repo collected it, some did not; treat as 0 when absent.
C_L1_TOTAL = "TCP_TOTAL_CACHE_ACCESSES_sum"
C_L1_TO_L2 = ["TCP_TCC_READ_REQ_sum", "TCP_TCC_WRITE_REQ_sum",
              "TCP_TCC_ATOMIC_WITH_RET_REQ_sum"]
C_L2_TO_EA = ["TCC_EA0_RDREQ_sum", "TCC_EA0_WRREQ_sum"]   # legacy (single EA chan)
C_L2_HIT   = "TCC_HIT_sum"                                 # direct L2 hit/miss
C_L2_MISS  = "TCC_MISS_sum"


def _col(df, name):
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def load_stack(root):
    """Return per-canonical-category L1/L2/DRAM served bytes for one profile root.

    Aggregates every pmc_* pass under `root`, derives per-dispatch L1/L2/DRAM
    bytes, then sums by friendly_label() category.
    """
    csvs = find_counter_csvs(root)
    if not csvs:
        raise SystemExit(f"no *_counter_collection.csv under {root}")
    wide = load_pmc_wide(csvs)

    # rocprofv3 splits the requested counters across several pmc_* passes, so a
    # single pivoted row (one dispatch in one pass) only carries the counters
    # *its* pass collected — the others are NaN/0. The nested L1>L2>DRAM
    # subtraction is therefore only valid on FULL-RUN TOTALS, not per row. So we
    # first sum every raw counter per kernel (each counter's non-zero values all
    # live in its own pass, so the sum is that counter's run total), then derive
    # the three served-byte segments. This mirrors analyze_roofline.py.
    raw_cols = [C_L1_TOTAL] + C_L1_TO_L2 + C_L2_TO_EA + [C_L2_HIT, C_L2_MISS]
    for c in raw_cols:
        wide[c] = _col(wide, c)
    wide["category"] = [friendly_label(k)[1] for k in wide["Kernel_Name"]]
    g = wide.groupby("category")[raw_cols].sum()

    tcp_total = g[C_L1_TOTAL]
    tcp_to_l2 = g[C_L1_TO_L2].sum(axis=1)       # L1 misses → reach L2

    # L1-served = L1 hits. For the L1-miss traffic that reaches L2, split it into
    # L2-hits vs DRAM using the DIRECT L2 hit rate (TCC_HIT/MISS) — NOT the old
    # `tcp_to_l2 − TCC_EA0` subtraction. The EA0 path counts only one memory
    # channel at a different granularity than the TCP request counters, so the
    # subtraction badly under-reported L2 (e.g. BatchNorm L2 came out ≈2% when
    # the measured L2 hit rate is ≈22%). TCC_HIT/MISS is the authoritative L2
    # metric (matches the cache_compare figure).
    l2_hit_rate = (g[C_L2_HIT] / (g[C_L2_HIT] + g[C_L2_MISS])).clip(0, 1).fillna(0.0)
    l1_served   = (tcp_total - tcp_to_l2).clip(lower=0) * REQ_BYTES
    l2_served   = (tcp_to_l2 * l2_hit_rate)            * REQ_BYTES
    dram_served = (tcp_to_l2 * (1.0 - l2_hit_rate))    * REQ_BYTES

    return {cat: {"l1": float(l1_served[cat]),
                  "l2": float(l2_served[cat]),
                  "dr": float(dram_served[cat])} for cat in g.index}


def fmt_bytes(b):
    if b >= 1e12: return f"{b/1e12:.2f} TB"
    if b >= 1e9:  return f"{b/1e9:.0f} GB"
    if b >= 1e6:  return f"{b/1e6:.0f} MB"
    if b >= 1e3:  return f"{b/1e3:.0f} kB"
    return f"{b:.0f} B"


def plot_traffic(native, torch, cats, out_dir, *, col=False, top_n=8, fname=None):
    ps.setup()
    (ps.scaling_fonts if col else ps.big_fonts)()
    plt.rcParams.update({
        "text.color": "#0a0a0a", "axes.labelcolor": "#0a0a0a",
        "xtick.color": "#0a0a0a", "ytick.color": "#0a0a0a",
        "axes.labelweight": "bold", "axes.titleweight": "bold",
        "axes.linewidth": 1.1,
    })
    if col:
        cats = cats[:top_n]
    n_cat = len(cats)
    if col:
        fig, ax = plt.subplots(figsize=ps.figsize("col", max(3.0, 0.32 * n_cat + 0.9)))
        min_pct = 12
    else:
        fig, ax = plt.subplots(figsize=ps.figsize("text", max(3.0, 0.34 * n_cat + 1.4)))
        min_pct = 6

    y = np.arange(n_cat); h = 0.36
    yN = y - h / 2 - 0.02
    yT = y + h / 2 + 0.02
    C_L1, C_L2, C_DRAM = "#a6d96a", "#fdae61", "#d7191c"

    def _b(stack, c, key):
        return float(stack.get(c, {}).get(key, 0.0))

    def _stacked(yarr, stack, edge_color, prefix):
        l1 = np.array([_b(stack, c, "l1") for c in cats])
        l2 = np.array([_b(stack, c, "l2") for c in cats])
        dr = np.array([_b(stack, c, "dr") for c in cats])
        total = l1 + l2 + dr
        with np.errstate(invalid="ignore", divide="ignore"):
            p1 = np.where(total > 0, l1 / total * 100, 0)
            p2 = np.where(total > 0, l2 / total * 100, 0)
            p3 = np.where(total > 0, dr / total * 100, 0)
        ax.barh(yarr, p1, h, color=C_L1, edgecolor=edge_color, linewidth=1.0, zorder=3)
        ax.barh(yarr, p2, h, left=p1, color=C_L2, edgecolor=edge_color, linewidth=1.0, zorder=3)
        ax.barh(yarr, p3, h, left=p1 + p2, color=C_DRAM, edgecolor=edge_color, linewidth=1.0, zorder=3)
        anno_fs = 8.5 if col else 7.5
        for yi, (a, b, c, t) in enumerate(zip(p1, p2, p3, total)):
            if t <= 0:
                ax.text(50, yarr[yi], "—", ha="center", va="center",
                        fontsize=anno_fs, fontweight="bold", color="#555")
                continue
            if a >= min_pct:
                ax.text(a / 2, yarr[yi], f"{a:.0f}%", ha="center", va="center",
                        fontsize=anno_fs, fontweight="bold", color="#0a3300")
            if b >= min_pct:
                ax.text(a + b / 2, yarr[yi], f"{b:.0f}%", ha="center", va="center",
                        fontsize=anno_fs, fontweight="bold", color="#5a2a00")
            if c >= min_pct:
                ax.text(a + b + c / 2, yarr[yi], f"{c:.0f}%", ha="center", va="center",
                        fontsize=anno_fs, fontweight="bold", color="white")
            # absolute traffic at the right of every bar (both modes) — % alone
            # hides that e.g. Convolution moves ~6 TB while Optimizer moves MB.
            ax.text(101.5, yarr[yi], f"{prefix} {fmt_bytes(t)}",
                    ha="left", va="center", fontsize=(8.0 if col else 7.5),
                    fontweight="bold", color=edge_color)

    _stacked(yN, native, ps.NATIVE, "N")
    _stacked(yT, torch, ps.PYTORCH, "T")

    for yi in range(n_cat):
        if yi % 2 == 0:
            ax.axhspan(yi - 0.5, yi + 0.5, color="0.96", zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels(cats, fontsize=(11 if col else 10), fontweight="bold", color="#0a0a0a")
    ps.bold_ticks(ax)
    for lbl in ax.get_yticklabels():
        lbl.set_color("#0a0a0a")
    ax.invert_yaxis()
    ax.set_ylim(n_cat + 0.45, -0.6)
    ax.set_xlim(0, 150 if col else 145)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"], fontweight="bold")
    if col:
        ax.set_title("Memory traffic per cache level (%)\n"
                     "Native (N, vermillion) vs PyTorch (T, blue) · top "
                     f"{n_cat} kernels · MI300X",
                     loc="center", fontweight="bold", fontsize=10, pad=6)
    else:
        ax.set_title("Memory traffic per cache level (%)  —  "
                     "Native (N) vs PyTorch (T), MI300X",
                     loc="center", fontweight="bold", fontsize=11, pad=8)
    ax.grid(True, axis="x", alpha=0.30, zorder=0)
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch
    color_handles = [
        Patch(facecolor=C_L1,   edgecolor="0.4", label="served by L1"),
        Patch(facecolor=C_L2,   edgecolor="0.4", label="served by L2"),
        Patch(facecolor=C_DRAM, edgecolor="0.4", label="from DRAM"),
    ]
    nt_handles = [
        Patch(facecolor="white", edgecolor=ps.NATIVE,  linewidth=1.5, label="N : Native"),
        Patch(facecolor="white", edgecolor=ps.PYTORCH, linewidth=1.5, label="T : PyTorch"),
    ]
    if col:
        # col title already names N/T; the missing key is the cache-level colours.
        leg = ax.legend(handles=color_handles, loc="lower center",
                        bbox_to_anchor=(0.5, -0.155), ncol=3, frameon=False,
                        fontsize=9, handlelength=1.1, handletextpad=0.4,
                        columnspacing=1.3, borderpad=0.2)
    else:
        leg = ax.legend(handles=color_handles + nt_handles, loc="center",
                        bbox_to_anchor=(0.5, 0.04), ncol=5, frameon=False,
                        fontsize=8.5, handlelength=1.1, handletextpad=0.4,
                        columnspacing=1.4, borderpad=0.2, borderaxespad=0.0)
    for t in leg.get_texts():
        t.set_color("#0a0a0a"); t.set_fontweight("bold")

    ps.cleanup(ax)
    fig.tight_layout()
    ps.save(fig, fname or "fig_traffic_decomposition_amd", out_dir=out_dir)
    plt.close(fig)


def plot_traffic_detail(native, torch, cats, out_dir,
                        fname="fig_traffic_decomposition_amd_detail"):
    """Big, fully-annotated decomposition: ALL categories, both stacks, every
    segment %% labelled, absolute L1/L2/DRAM bytes + total on the right, and the
    direct L2 hit rate. L1 and L2 are both visible in this single graph."""
    ps.setup(); ps.big_fonts()
    plt.rcParams.update({
        "text.color": "#0a0a0a", "axes.labelcolor": "#0a0a0a",
        "xtick.color": "#0a0a0a", "ytick.color": "#0a0a0a",
        "axes.labelweight": "bold", "axes.titleweight": "bold", "axes.linewidth": 1.1,
    })
    C_L1, C_L2, C_DRAM = "#1a9850", "#fdae61", "#d7191c"

    def tot(stack, c):
        d = stack.get(c, {}); return d.get("l1", 0)+d.get("l2", 0)+d.get("dr", 0)
    cats = [c for c in sorted(set(native) | set(torch),
                              key=lambda c: -(tot(native, c)+tot(torch, c)))
            if (tot(native, c)+tot(torch, c)) > 0]
    n = len(cats)
    fig, ax = plt.subplots(figsize=(13.5, max(5.0, 0.74*n + 1.4)))
    y = np.arange(n); h = 0.38

    def seg(stack, c, k): return float(stack.get(c, {}).get(k, 0.0))

    def row(yarr, stack, edge, sym):
        l1 = np.array([seg(stack, c, "l1") for c in cats])
        l2 = np.array([seg(stack, c, "l2") for c in cats])
        dr = np.array([seg(stack, c, "dr") for c in cats])
        t  = l1 + l2 + dr
        with np.errstate(invalid="ignore", divide="ignore"):
            p1 = np.where(t > 0, l1/t*100, 0); p2 = np.where(t > 0, l2/t*100, 0)
            p3 = np.where(t > 0, dr/t*100, 0)
        ax.barh(yarr, p1, h, color=C_L1, edgecolor=edge, lw=1.1, zorder=3)
        ax.barh(yarr, p2, h, left=p1, color=C_L2, edgecolor=edge, lw=1.1, zorder=3)
        ax.barh(yarr, p3, h, left=p1+p2, color=C_DRAM, edgecolor=edge, lw=1.1, zorder=3)
        for i in range(n):
            if t[i] <= 0:
                continue
            if p1[i] >= 5: ax.text(p1[i]/2, yarr[i], f"{p1[i]:.0f}%", ha="center", va="center", fontsize=8, fontweight="bold", color="#06250a")
            if p2[i] >= 5: ax.text(p1[i]+p2[i]/2, yarr[i], f"{p2[i]:.0f}%", ha="center", va="center", fontsize=8, fontweight="bold", color="#4a2200")
            if p3[i] >= 5: ax.text(p1[i]+p2[i]+p3[i]/2, yarr[i], f"{p3[i]:.0f}%", ha="center", va="center", fontsize=8, fontweight="bold", color="white")
            ax.text(102, yarr[i],
                    f"{sym}  L1 {fmt_bytes(l1[i])} · L2 {fmt_bytes(l2[i])} · "
                    f"DRAM {fmt_bytes(dr[i])}   (Σ {fmt_bytes(t[i])})",
                    ha="left", va="center", fontsize=7.6, fontweight="bold", color=edge)

    row(y - h/2 - 0.02, native, ps.NATIVE, "N")
    row(y + h/2 + 0.02, torch,  ps.PYTORCH, "T")
    for yi in range(n):
        if yi % 2 == 0:
            ax.axhspan(yi-0.5, yi+0.5, color="0.965", zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels(cats, fontsize=11, fontweight="bold", color="#0a0a0a")
    ax.invert_yaxis(); ax.set_ylim(n+0.4, -0.7)
    ax.set_xlim(0, 250)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"], fontweight="bold")
    ax.grid(True, axis="x", alpha=0.30, zorder=0); ax.set_axisbelow(True)
    ax.set_title("Memory-traffic decomposition by cache level — ResNet-50 v1.5, MI300X (224², matched)\n"
                 "stack = % of each category's traffic served by L1 / L2 / DRAM   ·   "
                 "L2 split uses the direct TCC hit/miss rate   ·   N=Native  T=PyTorch",
                 loc="left", fontweight="bold", fontsize=11, pad=10)
    from matplotlib.patches import Patch
    leg = ax.legend(handles=[
        Patch(facecolor=C_L1, edgecolor="0.4", label="served by L1 (L1 hit)"),
        Patch(facecolor=C_L2, edgecolor="0.4", label="served by L2 (L1 miss, L2 hit)"),
        Patch(facecolor=C_DRAM, edgecolor="0.4", label="from DRAM (L1+L2 miss)"),
        Patch(facecolor="white", edgecolor=ps.NATIVE, linewidth=1.6, label="N : Native HIP"),
        Patch(facecolor="white", edgecolor=ps.PYTORCH, linewidth=1.6, label="T : PyTorch"),
    ], loc="lower right", ncol=1, frameon=True, framealpha=0.9, edgecolor="0.7",
        fontsize=9, handlelength=1.2, handletextpad=0.5, borderpad=0.5)
    for t in leg.get_texts():
        t.set_color("#0a0a0a"); t.set_fontweight("bold")
    ps.cleanup(ax)
    fig.tight_layout()
    ps.save(fig, fname, out_dir=out_dir)
    plt.close(fig)


def main():
    P = "profiles"
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", default=os.path.join(HERE, P, "native_in1k_retry_52488168"),
                    help="rocprofv3 profile root for the Native HIP trainer "
                         "(matched ImageNet-1K 224² run; was tiny-64² amd_52295979)")
    ap.add_argument("--torch", default=os.path.join(HERE, P, "pt_amd_52438059"),
                    help="rocprofv3 profile root for the PyTorch-ROCm run")
    ap.add_argument("--out", default=os.path.join(HERE, "figures", "traffic_decomposition_amd"))
    ap.add_argument("--mode", choices=("text", "col", "detail", "both", "all"), default="both")
    ap.add_argument("--top-n", type=int, default=8)
    args = ap.parse_args()

    print(f"Native  root : {args.native}")
    print(f"PyTorch root : {args.torch}")
    native = load_stack(args.native)
    torch  = load_stack(args.torch)

    cats = sorted(set(native) | set(torch),
                  key=lambda k: -max(sum(native.get(k, {}).get(x, 0) for x in "l1 l2 dr".split()),
                                     sum(torch.get(k, {}).get(x, 0) for x in "l1 l2 dr".split())))

    os.makedirs(args.out, exist_ok=True)
    if args.mode in ("text", "both", "all"):
        plot_traffic(native, torch, cats, args.out, col=False,
                     fname="fig_traffic_decomposition_amd")
    if args.mode in ("col", "both", "all"):
        plot_traffic(native, torch, cats, args.out, col=True, top_n=args.top_n,
                     fname="fig_traffic_decomposition_amd_col")
    if args.mode in ("detail", "all"):
        plot_traffic_detail(native, torch, cats, args.out)

    def fmt(m):
        if not m:
            return f"{'--':>8} {'--':>8} {'--':>8} {'--':>10}"
        tot = m["l1"] + m["l2"] + m["dr"]
        return (f"{fmt_bytes(m['l1']):>8} {fmt_bytes(m['l2']):>8} "
                f"{fmt_bytes(m['dr']):>8} {fmt_bytes(tot):>10}")

    print()
    print(f"{'category':<20} | {'N: L1      L2      DRAM     total':<46} "
          f"| {'T: L1      L2      DRAM     total':<46}")
    print("-" * 120)
    for c in cats:
        print(f"{c:<20} | {fmt(native.get(c, {})):<46} | {fmt(torch.get(c, {})):<46}")


if __name__ == "__main__":
    main()
