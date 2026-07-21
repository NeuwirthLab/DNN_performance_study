#!/usr/bin/env python3
"""Side-by-side cache traffic-decomposition: MI300X vs H100.

Two panels share the kernel axis. Within each panel, every canonical kernel gets
a Native bar (black outline, top) and a PyTorch bar (blue outline, bottom), each
split into L1 / L2 / DRAM shares (normalised to 100%). Reads the NCU-style
raw.csv from two bundles and reuses the exact parser + canonical kernel names of
the single-arch figure, so rows line up 1:1 across architectures.

Defaults point at the MI300X bundle and the (to-be-populated) H100 bundle. The
H100 raw.csv must be a real NVIDIA NCU export (l1tex/lts sector counters from
cudnn/cutlass kernels); until that is provided the right panel will simply
duplicate the left.

    python3.11 fig_traffic_decomposition_compare.py
"""
import os, sys, csv, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paper_style as ps                                          # noqa: E402
from fig_traffic_decomposition import parse_raw, aggregate        # exact reuse  # noqa: E402

H100_COL_ORDER = ["bn_bwd", "conv_wgrad", "conv_dgrad", "bn_fwd",
                  "relu", "gemm", "add_residual", "layout_transform",
                  "conv_fprop"]
C_L1, C_L2, C_DRAM = "#a6d96a", "#fdae61", "#d7191c"
NATIVE_EDGE, PYTORCH_EDGE = "#000000", "#0072B2"


def load(nat_csv, pt_csv):
    return aggregate(parse_raw(nat_csv)), aggregate(parse_raw(pt_csv))


def load_h100_aligned(path):
    """Reconstruct stacked L1/L2/DRAM traffic shares for H100 from the bundle's
    aligned_metrics.csv (per-kernel L1 and L2 hit rates), using the SAME nested
    model the MI300X figure uses: L1 = hit_L1, L2 = (1-hit_L1)*hit_L2,
    DRAM = (1-hit_L1)*(1-hit_L2). Skips kernel/stack cells that are NaN (the H100
    CUDA and NGC runs profiled complementary kernel subsets). Returns shares
    (summing to 100); the panel re-normalises, so units are irrelevant."""
    def _f(s):
        try:
            v = float(s)
            return v if v == v else None        # None on NaN
        except (TypeError, ValueError):
            return None
    native, torch = {}, {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            k = row["category_label"]
            for pfx, d in (("cuda", native), ("torch", torch)):
                l1 = _f(row.get(f"{pfx}_l1_hit"))
                l2 = _f(row.get(f"{pfx}_l2_hit"))
                if l1 is None or l2 is None:
                    continue
                a, b = l1 / 100.0, l2 / 100.0
                d[k] = {"l1": a * 100,
                        "l2": (1 - a) * b * 100,
                        "dr": (1 - a) * (1 - b) * 100}
    return native, torch


def _panel(ax, native, torch, cats, title, *, show_ylabels):
    y = np.arange(len(cats)); h = 0.36
    yN, yT = y - h / 2 - 0.02, y + h / 2 + 0.02

    def stk(yarr, stack, edge):
        l1 = np.array([stack.get(c, {}).get("l1", 0.0) for c in cats])
        l2 = np.array([stack.get(c, {}).get("l2", 0.0) for c in cats])
        dr = np.array([stack.get(c, {}).get("dr", 0.0) for c in cats])
        t = l1 + l2 + dr
        with np.errstate(invalid="ignore", divide="ignore"):
            p1 = np.where(t > 0, l1 / t * 100, 0)
            p2 = np.where(t > 0, l2 / t * 100, 0)
            p3 = np.where(t > 0, dr / t * 100, 0)
        ax.barh(yarr, p1, h, color=C_L1, edgecolor=edge, linewidth=1.3, zorder=3)
        ax.barh(yarr, p2, h, left=p1, color=C_L2, edgecolor=edge, linewidth=1.3, zorder=3)
        ax.barh(yarr, p3, h, left=p1 + p2, color=C_DRAM, edgecolor=edge, linewidth=1.3, zorder=3)
        for yi, (a, b, c, tt) in enumerate(zip(p1, p2, p3, t)):
            if tt <= 0:
                ax.text(50, yarr[yi], "—", ha="center", va="center",
                        fontsize=7, fontweight="bold", color="#555"); continue
            for val, xc, col in [(a, a / 2, "#0a3300"), (b, a + b / 2, "#5a2a00"),
                                 (c, a + b + c / 2, "white")]:
                if val >= 15:
                    ax.text(xc, yarr[yi], f"{val:.0f}", ha="center", va="center",
                            fontsize=6.5, fontweight="bold", color=col)

    for yi in range(len(cats)):
        if yi % 2 == 0:
            ax.axhspan(yi - 0.5, yi + 0.5, color="0.96", zorder=0)
    stk(yN, native, NATIVE_EDGE)
    stk(yT, torch, PYTORCH_EDGE)
    ax.set_yticks(y)
    if show_ylabels:
        ax.set_yticklabels(cats, fontweight="bold", fontsize=10)
    else:
        ax.tick_params(labelleft=False)
    ax.invert_yaxis(); ax.set_ylim(len(cats) - 0.4, -0.6)
    ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(["0", "50%", "100%"], fontweight="bold")
    ax.set_title(title, fontweight="bold", fontsize=12, pad=6)
    ax.grid(True, axis="x", alpha=0.30, zorder=0); ax.set_axisbelow(True)
    ps.cleanup(ax)


def main():
    B = os.path.join(HERE, "fig_traffic_decomposition_bundle")
    H = os.path.join(B, "h100/roofline_cache_compare_bundle-h100")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mi300-native", default=os.path.join(B, "ncu_comparison_latest/native_amd/data/raw.csv"))
    ap.add_argument("--mi300-torch",  default=os.path.join(B, "ncu_comparison_latest/pytorch_amd/data/raw.csv"))
    ap.add_argument("--h100-aligned", default=os.path.join(H, "figures/aligned_metrics.csv"),
                    help="H100 aligned_metrics.csv (L1/L2 hit rates -> reconstructed shares)")
    ap.add_argument("--out", default=os.path.join(HERE, "figures"))
    args = ap.parse_args()

    m_n, m_t = load(args.mi300_native, args.mi300_torch)
    h_n, h_t = load_h100_aligned(args.h100_aligned)

    cats = [c for c in H100_COL_ORDER
            if c in set(m_n) | set(m_t) | set(h_n) | set(h_t)]

    ps.setup(); ps.big_fonts()
    plt.rcParams.update({"axes.labelweight": "bold", "axes.titleweight": "bold",
                         "axes.linewidth": 1.1})
    fig, (axL, axR) = plt.subplots(1, 2, sharey=True,
                                   figsize=(7.16, 0.5 * len(cats) + 1.7))
    _panel(axL, m_n, m_t, cats, "MI300X (CDNA3)", show_ylabels=True)
    _panel(axR, h_n, h_t, cats, "H100 (Hopper)", show_ylabels=False)

    handles = [
        Patch(facecolor=C_L1, edgecolor="0.4", label="L1"),
        Patch(facecolor=C_L2, edgecolor="0.4", label="L2"),
        Patch(facecolor=C_DRAM, edgecolor="0.4", label="DRAM"),
        Patch(facecolor="white", edgecolor=NATIVE_EDGE, linewidth=1.6, label="Native"),
        Patch(facecolor="white", edgecolor=PYTORCH_EDGE, linewidth=1.6, label="PyTorch"),
    ]
    leg = fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
                     bbox_to_anchor=(0.5, -0.01), fontsize=9, columnspacing=1.4,
                     handlelength=1.1, handletextpad=0.4)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    ps.save(fig, "fig_traffic_decomposition_compare", out_dir=args.out)
    print("wrote", os.path.join(args.out, "fig_traffic_decomposition_compare.{pdf,png}"))

    # console table
    print(f"\n{'kernel':<13} | {'MI300X N L1/L2/DRAM':>22} | {'H100 N L1/L2/DRAM':>22}")
    for c in cats:
        def s(m):
            t = m.get(c, {}).get("l1", 0) + m.get(c, {}).get("l2", 0) + m.get(c, {}).get("dr", 0)
            if t <= 0: return "   --"
            g = m[c]
            return f"{100*g['l1']/t:.0f}/{100*g['l2']/t:.0f}/{100*g['dr']/t:.0f}"
        print(f"{c:<13} | {s(m_n):>22} | {s(h_n):>22}")


if __name__ == "__main__":
    main()
