#!/usr/bin/env python3
"""Optimization-lever decomposition: how the custom-HIP ResNet-50 closes (part
of) the gap to PyTorch on ImageNet-1K (1 GPU, batch 256, 224x224, FP32).

2x2 matrix measured on the isolated fork (train_multigpu_exp), job 52391399:
    base_real / fused_real / base_synth(compute ceiling) / fused_synth
PyTorch reference from profiles/<job>/summary.csv.

Panel (a): 2x2 grouped bars (real vs synthetic data) x (baseline vs fused
           residual), with the PyTorch reference line.
Panel (b): waterfall — start at base_real, add the data-pipeline lever and the
           residual-fusion lever, then show the residual kernel-efficiency gap
           that still separates the HIP ceiling from PyTorch.

House style: paper_style.py.
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import paper_style as ps
except ImportError:
    sys.exit("paper_style.py must sit next to this script (or be on PYTHONPATH).")

PT_IPS = 1516.0  # PyTorch fp32, 1 GPU batch 256 — full ImageNet-1K (job 52393128)


def load(levers_csv):
    d = {}
    with open(levers_csv) as f:
        for row in csv.DictReader(f):
            d[row["arm"]] = float(row["img_per_sec"])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levers", default="profiles/levers_52391399/summary.csv")
    ap.add_argument("--pt", type=float, default=PT_IPS)
    ap.add_argument("--out", default="plots/mi300x")
    ap.add_argument("--name", default="rn50_levers")
    args = ap.parse_args()

    d = load(args.levers)
    br, fr = d["base_real"], d["fused_real"]
    bs, fs = d["base_synth"], d["fused_synth"]
    pt = args.pt

    ps.setup()
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=ps.figsize("text", height=2.8),
        gridspec_kw=dict(width_ratios=[1.0, 1.15], wspace=0.34))

    BASE_C, FUSE_C, PT_C, GAP_C = ps.PALETTE[1], ps.PALETTE[2], ps.PALETTE[0], "#cfcfcf"

    # ── (a) 2x2 grouped bars ────────────────────────────────────────────────
    groups = ["Real data", "Synthetic\n(compute ceiling)"]
    base = [br, bs]; fused = [fr, fs]
    x = np.arange(2); w = 0.36
    axA.bar(x - w/2, base, w, color=BASE_C, label="baseline (copy+add+relu)")
    axA.bar(x + w/2, fused, w, color=FUSE_C, label="fused residual")
    for xi, v in zip(x - w/2, base):
        axA.text(xi, v + 8, f"{v:.0f}", ha="center", va="bottom", fontsize=6.6)
    for xi, v in zip(x + w/2, fused):
        axA.text(xi, v + 8, f"{v:.0f}", ha="center", va="bottom", fontsize=6.6)
    axA.axhline(pt, color=PT_C, ls="--", lw=1.0)
    axA.text(1.45, pt + 6, f"PyTorch {pt:.0f}", ha="right", va="bottom",
             fontsize=6.8, color=PT_C, fontweight="bold")
    axA.set_xticks(x); axA.set_xticklabels(groups, fontsize=7.2)
    axA.set_ylabel(ps.LABELS.THROUGHPUT)
    axA.set_ylim(0, pt * 1.12)
    axA.set_title("(a) Levers: data pipeline × residual fusion", fontsize=8.0)
    axA.legend(loc="lower center", bbox_to_anchor=(0.5, -0.36), ncol=1,
               fontsize=6.4, frameon=False)

    # ── (b) gap-decomposition waterfall ─────────────────────────────────────
    pipe = bs - br        # data-pipeline lever (real -> synthetic)
    fuse = fs - bs        # residual-fusion lever
    keff = pt - fs        # remaining kernel-efficiency gap to PyTorch
    labels = ["HIP\nbase_real", "+ data\npipeline", "+ residual\nfusion",
              "kernel-eff\ngap", "PyTorch"]
    xs = np.arange(5)
    # full bars at the ends, floating deltas in the middle
    axB.bar(0, br, color=BASE_C, width=0.62)
    axB.bar(1, pipe, bottom=br, color=FUSE_C, width=0.62)
    axB.bar(2, fuse, bottom=bs, color=FUSE_C, width=0.62, alpha=0.7)
    axB.bar(3, keff, bottom=fs, color=GAP_C, width=0.62, hatch="///",
            edgecolor="#888", linewidth=0.5)
    axB.bar(4, pt, color=PT_C, width=0.62)
    # connector lines + delta labels
    for xi, top in [(0, br), (1, bs), (2, fs), (3, pt)]:
        axB.plot([xi + 0.31, xi + 0.69], [top, top], color="#999", lw=0.6, ls=":")
    axB.text(0, br + 12, f"{br:.0f}", ha="center", fontsize=6.6, fontweight="bold")
    axB.text(1, bs + 12, f"+{pipe:.0f}", ha="center", fontsize=6.6, color="#222")
    axB.text(2, fs + 12, f"+{fuse:.0f}", ha="center", fontsize=6.6, color="#222")
    axB.text(3, pt + 12, f"+{keff:.0f}", ha="center", fontsize=6.6, color="#555")
    axB.text(4, pt + 12, f"{pt:.0f}", ha="center", fontsize=6.6, fontweight="bold")
    axB.set_xticks(xs); axB.set_xticklabels(labels, fontsize=6.6)
    axB.set_ylabel(ps.LABELS.THROUGHPUT)
    axB.set_ylim(0, pt * 1.16)
    axB.set_title("(b) Decomposing the 1.24× gap to PyTorch", fontsize=8.0)

    for ax in (axA, axB):
        ps.cleanup(ax)

    ps.figure_heading(
        fig, vendor="AMD", gpu="MI300X",
        sub_parts=["ResNet-50 ImageNet-1K, 1 GPU, batch 256, FP32",
                   "levers recover ~11%; ~12% kernel-efficiency gap remains"])
    fig.subplots_adjust(top=0.80, bottom=0.22, left=0.085, right=0.985)
    ps.save(fig, args.name, out_dir=args.out)
    print(f"base_real {br:.0f} | +pipeline {bs:.0f} | +fusion {fs:.0f} | "
          f"PyTorch {pt:.0f}")
    print(f"  pipeline +{100*pipe/br:.1f}% | fusion +{100*fuse/bs:.1f}% | "
          f"remaining kernel-eff {100*keff/fs:.1f}% | total gap {pt/br:.2f}x")
    print(f"wrote {args.out}/{args.name}.pdf + .png")


if __name__ == "__main__":
    main()
