#!/usr/bin/env python3
"""Memory-hierarchy traffic decomposition — the architecturally rigorous view.

For each kernel category × stack we draw one horizontal bar whose total length
is the total global-load memory traffic, broken into three stacked segments:

    [   L1-served   |   L2-served   |   DRAM-served   ]

L1-served   = l1tex sectors that hit in L1                       (×32 B)
L2-served   = sectors that missed L1, then hit in L2             (×32 B)
DRAM-served = sectors that missed both L1 and L2                 (×32 B)

This is the canonical Memory Workload Analysis chart that NCU produces per
kernel, here aggregated by canonical category and shown for both stacks.

Output:  ncu_comparison_latest/per_metric_compare/fig_traffic_decomposition.{pdf,png}
"""
from __future__ import annotations
import os, sys, csv, argparse
import numpy as np
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))
import paper_style as ps                                          # noqa: E402
import kernel_canon                                               # noqa: E402

SECTOR_BYTES = 32                # NVIDIA cache line in bytes


def _num(s):
    if s is None or s == "":
        return 0.0
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return 0.0


L1_HIT_COUNTERS = [
    "l1tex__t_sectors_pipe_lsu_mem_global_op_atom_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_red_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st_lookup_hit.sum",
    "l1tex__t_sectors_pipe_tex_mem_surface_op_atom_lookup_hit.sum",
    "l1tex__t_sectors_pipe_tex_mem_surface_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_tex_mem_surface_op_red_lookup_hit.sum",
    "l1tex__t_sectors_pipe_tex_mem_surface_op_st_lookup_hit.sum",
    "l1tex__t_sectors_pipe_tex_mem_texture_lookup_hit.sum",
]
L1_MISS_COUNTERS = [
    "l1tex__t_sectors_pipe_lsu_mem_dshared_op_ld_lookup_miss.sum",
    "l1tex__t_sectors_pipe_lsu_mem_dshared_op_redas_lookup_miss.sum",
    "l1tex__t_sectors_pipe_lsu_mem_dshared_op_st_lookup_miss.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_atom_lookup_miss.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_red_lookup_miss.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_miss.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_miss.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st_lookup_miss.sum",
    "l1tex__t_sectors_pipe_tex_mem_surface_op_atom_lookup_miss.sum",
    "l1tex__t_sectors_pipe_tex_mem_surface_op_ld_lookup_miss.sum",
    "l1tex__t_sectors_pipe_tex_mem_surface_op_red_lookup_miss.sum",
    "l1tex__t_sectors_pipe_tex_mem_texture_lookup_miss.sum",
]


def parse_raw(path):
    """Return list of (canonical_label, l1_hit_B, l2_served_B, dram_B, dur_ms).

    L1 hits  = sum over all 11 hit sub-counters (global, local, surface, tex,
               atomic, reduction); captures every read path including TMA via
               its tex-pipe transit.
    L2 hits  = lts__t_sectors_lookup_hit.sum
    DRAM     = lts__t_sectors_lookup_miss.sum     (everything that escaped L2)
    """
    with open(path) as fh:
        rd = csv.reader(fh); hdr = next(rd); rows = list(rd)
    units = rows[0]; data = rows[1:]

    def col(name):
        return hdr.index(name) if name in hdr else None

    iname = col("Kernel Name")
    hit_idx  = [col(c) for c in L1_HIT_COUNTERS  if col(c) is not None]
    miss_idx = [col(c) for c in L1_MISS_COUNTERS if col(c) is not None]
    iL2h  = col("lts__t_sectors_lookup_hit.sum")
    iL2m  = col("lts__t_sectors_lookup_miss.sum")
    idur  = col("gpu__time_duration.sum")
    udur  = units[idur] if idur is not None else "ms"

    def to_ms(v):
        if udur == "us": return v / 1e3
        if udur == "ns": return v / 1e6
        return v

    out = []
    for r in data:
        name = r[iname]
        label, _cat = kernel_canon.canon(name)
        l1h = sum(_num(r[i]) for i in hit_idx)
        l1m = sum(_num(r[i]) for i in miss_idx)   # unused (only for hit-rate)
        l2h = _num(r[iL2h]); l2m = _num(r[iL2m])
        dur = to_ms(_num(r[idur]))
        out.append((label,
                    l1h * SECTOR_BYTES,                       # L1 served
                    l2h * SECTOR_BYTES,                       # L2 served
                    l2m * SECTOR_BYTES,                       # DRAM
                    dur))
    return out


def aggregate(rows):
    """Sum per canonical label."""
    agg = {}
    for label, l1, l2, dr, dur in rows:
        d = agg.setdefault(label,
                           {"l1": 0.0, "l2": 0.0, "dr": 0.0, "dur": 0.0, "n": 0})
        d["l1"] += l1; d["l2"] += l2; d["dr"] += dr
        d["dur"] += dur; d["n"] += 1
    return agg


def fmt_bytes(b):
    if b >= 1e9:  return f"{b/1e9:.1f} GB"
    if b >= 1e6:  return f"{b/1e6:.0f} MB"
    if b >= 1e3:  return f"{b/1e3:.0f} kB"
    return f"{b:.0f} B"


def plot_traffic(cuda, torch, cats, out_dir, *, col=False, top_n=8,
                 fname=None, col_cats=None):
    """Render the traffic-decomposition figure.

    col=False  -> IEEE double-column (text-width 7.16 in), all categories
    col=True   -> IEEE single-column (3.5 in), top-N categories only,
                  bigger fonts, byte totals dropped, legend inlined.
    """
    ps.setup()
    if col:
        ps.scaling_fonts()              # bigger + bold for tiny prints
    else:
        ps.big_fonts()
    plt.rcParams.update({
        "text.color":       "#0a0a0a",
        "axes.labelcolor":  "#0a0a0a",
        "xtick.color":      "#0a0a0a",
        "ytick.color":      "#0a0a0a",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.linewidth":   1.1,
    })

    # Single-column: either pin to an explicit kernel set/order (col_cats — used
    # to mirror the H100 figure row-for-row so the two are directly comparable),
    # or fall back to the heaviest top-N by traffic.
    if col:
        if col_cats:
            cats = [c for c in col_cats if c in set(cuda) | set(torch)]
        else:
            cats = cats[:top_n]

    n_cat = len(cats)
    if col:
        fig, ax = plt.subplots(figsize=ps.figsize("col", h=max(3.2,
                                                  0.55 * n_cat + 1.2)))
        min_pct = 15                    # suppress small in-bar % labels
    else:
        fig, ax = plt.subplots(figsize=ps.figsize("text",
                                                  h=max(3.0, 0.34 * n_cat + 1.4)))
        min_pct = 6

    y = np.arange(n_cat); h = 0.36
    yN = y - h / 2 - 0.02
    yT = y + h / 2 + 0.02

    # Universal level colours (apply to both stacks).
    C_L1   = "#a6d96a"           # light green
    C_L2   = "#fdae61"           # orange
    C_DRAM = "#d7191c"           # red

    def _bytes(stack, c, key):
        return float(stack.get(c, {}).get(key, 0.0))

    def _stacked(yarr, stack, edge_color, label_prefix):
        l1 = np.array([_bytes(stack, c, "l1") for c in cats])
        l2 = np.array([_bytes(stack, c, "l2") for c in cats])
        dr = np.array([_bytes(stack, c, "dr") for c in cats])
        total = l1 + l2 + dr
        # 100% normalisation per bar -- proportions, not absolute bytes
        with np.errstate(invalid="ignore", divide="ignore"):
            p1 = np.where(total > 0, l1 / total * 100, 0)
            p2 = np.where(total > 0, l2 / total * 100, 0)
            p3 = np.where(total > 0, dr / total * 100, 0)
            # Textbook L2 hit rate = L2_hits / (L2_hits + L2_misses)
            l2_input = l2 + dr
            l2_hit_pct = np.where(l2_input > 0, l2 / l2_input * 100, np.nan)
            # L1 hit rate as conventionally defined = L1_hits / total looking
            # at L1 (same as p1 here since DRAM access also looked at L1 first).
            l1_hit_pct = p1.copy()
        ax.barh(yarr, p1, h, color=C_L1,   edgecolor=edge_color, linewidth=1.4,
                zorder=3)
        ax.barh(yarr, p2, h, left=p1, color=C_L2,
                edgecolor=edge_color, linewidth=1.4, zorder=3)
        ax.barh(yarr, p3, h, left=p1 + p2, color=C_DRAM,
                edgecolor=edge_color, linewidth=1.4, zorder=3)
        # In-bar % labels (only if segment >= threshold, else it would overflow)
        anno_fs = 7.5 if col else 7.5
        for yi, (a, b, c, t) in enumerate(zip(p1, p2, p3, total)):
            if t <= 0:
                ax.text(50, yarr[yi], "—", ha="center", va="center",
                        fontsize=anno_fs, fontweight="bold", color="#555")
                continue
            if a >= min_pct:
                ax.text(a / 2, yarr[yi], f"{a:.0f}%", ha="center",
                        va="center", fontsize=anno_fs, fontweight="bold",
                        color="#0a3300")
            if b >= min_pct:
                ax.text(a + b / 2, yarr[yi], f"{b:.0f}%", ha="center",
                        va="center", fontsize=anno_fs, fontweight="bold",
                        color="#5a2a00")
            if c >= min_pct:
                ax.text(a + b + c / 2, yarr[yi], f"{c:.0f}%", ha="center",
                        va="center", fontsize=anno_fs, fontweight="bold",
                        color="white")
            # Absolute total at end of bar -- only in the wide layout; the
            # single-column figure relies on the caption to carry totals.
            if not col:
                ax.text(101.5, yarr[yi],
                        f"{label_prefix}  {fmt_bytes(t)}",
                        ha="left", va="center", fontsize=7.5,
                        fontweight="bold", color=edge_color)

    # Distinguish the two stacks by OUTLINE colour (no legend): black around
    # the native (CUDA/HIP) bars, a contrasting blue around the PyTorch bars.
    NATIVE_EDGE  = "#000000"        # black  (Native)
    PYTORCH_EDGE = "#0072B2"        # blue   (PyTorch) — high contrast vs black
    _stacked(yN, cuda,  NATIVE_EDGE,  "N")
    _stacked(yT, torch, PYTORCH_EDGE, "T")

    # Faint alternating rows so the eye snaps to the right category
    for yi in range(n_cat):
        if yi % 2 == 0:
            ax.axhspan(yi - 0.5, yi + 0.5, color="0.96", zorder=0)

    ax.set_yticks(y)
    if col:
        import textwrap
        ylabels = ["\n".join(textwrap.wrap(c, 16)) for c in cats]
        yt_fs = 8.5
    else:
        ylabels = cats
        yt_fs = 10
    ax.set_yticklabels(ylabels, fontsize=yt_fs, fontweight="bold",
                       color="#0a0a0a")
    ps.bold_ticks(ax)
    for lbl in ax.get_yticklabels():
        lbl.set_color("#0a0a0a")
    ax.invert_yaxis()
    # Reserve a strip below the last bar for the legend (col wraps to 3 cols).
    ax.set_ylim((n_cat + 0.95) if col else (n_cat + 0.45), -0.6)
    # Wide layout: leave 45 % extra x-room for end-of-bar byte annotations.
    # Single-column: bar ends at 100 %; no extra room needed.
    ax.set_xlim(0, 100 if col else 145)
    if col:
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(["0", "50%", "100%"], fontweight="bold")
    else:
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"], fontweight="bold")
    # (no title / heading)

    ax.grid(True, axis="x", alpha=0.30, zorder=0)
    ax.set_axisbelow(True)

    # Legend: cache levels (L1/L2/DRAM fills) + the two stacks (outline colour).
    from matplotlib.patches import Patch
    leg_handles = [
        Patch(facecolor=C_L1,   edgecolor="0.4", label="L1"),
        Patch(facecolor=C_L2,   edgecolor="0.4", label="L2"),
        Patch(facecolor=C_DRAM, edgecolor="0.4", label="DRAM"),
        Patch(facecolor="white", edgecolor=NATIVE_EDGE,  linewidth=1.6,
              label="N : Native"),
        Patch(facecolor="white", edgecolor=PYTORCH_EDGE, linewidth=1.6,
              label="T : PyTorch"),
    ]
    leg = ax.legend(
        handles=leg_handles, loc="center", bbox_to_anchor=(0.5, 0.045),
        ncol=(3 if col else 5), frameon=False,
        fontsize=(7.5 if col else 8.5),
        handlelength=1.1, handletextpad=0.4,
        columnspacing=(1.0 if col else 1.4), borderpad=0.2, borderaxespad=0.0,
    )
    for t in leg.get_texts():
        t.set_color("#0a0a0a"); t.set_fontweight("bold")

    ps.cleanup(ax)
    fig.tight_layout()
    ps.save(fig, fname or "fig_traffic_decomposition", out_dir=out_dir)
    plt.close(fig)


def main() -> None:
    H = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda",  default=os.path.join(
        H, "ncu_comparison_latest/native_amd/data/raw.csv"))
    ap.add_argument("--torch", default=os.path.join(
        H, "ncu_comparison_latest/pytorch_amd/data/raw.csv"))
    ap.add_argument("--out",   default=os.path.join(
        H, "ncu_comparison_latest/per_metric_compare"))
    ap.add_argument("--mode", choices=("text", "col", "both"), default="both",
                    help="text=IEEE 2-column figure, col=IEEE 1-column figure, "
                         "both=render both (default)")
    ap.add_argument("--top-n", type=int, default=8,
                    help="Categories shown in --col mode (default 8)")
    args = ap.parse_args()

    print(f"Native raw.csv : {args.cuda}")
    print(f"PyTorch raw.csv: {args.torch}")
    cuda  = aggregate(parse_raw(args.cuda))
    torch = aggregate(parse_raw(args.torch))

    # Order categories by larger of two total traffics
    cats = sorted(
        set(cuda) | set(torch),
        key=lambda k: -max(
            cuda.get(k, {}).get("l1", 0) + cuda.get(k, {}).get("l2", 0)
                + cuda.get(k, {}).get("dr", 0),
            torch.get(k, {}).get("l1", 0) + torch.get(k, {}).get("l2", 0)
                + torch.get(k, {}).get("dr", 0)),
    )

    os.makedirs(args.out, exist_ok=True)
    if args.mode in ("text", "both"):
        plot_traffic(cuda, torch, cats, args.out,
                     col=False, fname="fig_traffic_decomposition")
    if args.mode in ("col", "both"):
        # Same 8 kernels, same top-to-bottom order as the H100 col figure, so
        # the MI300X and H100 single-column plots are row-equivalent.
        H100_COL_ORDER = ["bn_bwd", "conv_wgrad", "conv_dgrad", "bn_fwd",
                          "relu", "gemm", "add_residual", "conv_fprop"]
        plot_traffic(cuda, torch, cats, args.out,
                     col=True, top_n=args.top_n, col_cats=H100_COL_ORDER,
                     fname="fig_traffic_decomposition_col")

    print()
    print(f"{'category':<18} | {'N: L1     L2     DRAM    total':<46} "
          f"| {'T: L1     L2     DRAM    total':<46}")
    print("-" * 116)
    def fmt(m):
        if not m:
            return f"{'--':>8} {'--':>8} {'--':>8} {'--':>10}"
        tot = m['l1'] + m['l2'] + m['dr']
        return (f"{fmt_bytes(m['l1']):>8} {fmt_bytes(m['l2']):>8} "
                f"{fmt_bytes(m['dr']):>8} {fmt_bytes(tot):>10}")
    for c in cats:
        print(f"{c:<18} | {fmt(cuda.get(c, {})):<46} "
              f"| {fmt(torch.get(c, {})):<46}")


if __name__ == "__main__":
    main()
