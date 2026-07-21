#!/usr/bin/env python3
"""
plot_activation_sparsity.py — styled per-layer activation-sparsity figure.

Reads the CSV emitted by the RN50_SPARSITY_LOG probe in resnet50.cu
(columns: layer_ordinal,numel,zeros,zero_fraction) and renders a paper-ready
bar chart of post-ReLU zero fraction per layer, in forward order.

Follows styling.md: paper_style.setup(), ps.NATIVE (vermillion) for the native
HIP implementation, ps.IDEAL for the mean reference line, ps.save() → PDF+PNG.

Usage:
    /usr/bin/python3.11 plot_activation_sparsity.py \
        --csv activation_sparsity.csv --out fig_activation_sparsity
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

import paper_style as ps


def read_csv(path: Path):
    """Return (ordinals, numels, zero_pct) from the probe CSV."""
    ordinals, numels, zero_pct = [], [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            ordinals.append(int(row["layer_ordinal"]))
            numels.append(int(row["numel"]))
            zero_pct.append(100.0 * float(row["zero_fraction"]))
    if not ordinals:
        sys.exit(f"error: no data rows in {path}")
    return ordinals, numels, zero_pct


def stage_spans(numels):
    """Infer ResNet stage boundaries from the tensor element count.

    Activation numel drops at each stage transition (spatial resolution halves
    faster than channels double), so a strict decrease between consecutive ReLU
    outputs marks a stage start. Returns the ordinals at which a new stage opens
    (used only for light visual demarcation — never load-bearing).
    """
    starts = [0]
    for i in range(1, len(numels)):
        if numels[i] < numels[i - 1]:
            starts.append(i)
    return starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="activation_sparsity.csv",
                    help="probe CSV (default: activation_sparsity.csv)")
    ap.add_argument("--out", default="fig_activation_sparsity",
                    help="output filename stem (default: fig_activation_sparsity)")
    ap.add_argument("--out-dir", default=".", help="output directory")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"error: {csv_path} not found — run with "
                 f"RN50_SPARSITY_LOG={csv_path} first.")

    ordinals, numels, zero_pct = read_csv(csv_path)
    mean_pct = sum(zero_pct) / len(zero_pct)

    ps.setup()
    # Double-column width: ~53 bars read better than at single-column.
    fig, ax = plt.subplots(figsize=ps.figsize("text", height=2.6))

    ax.bar(ordinals, zero_pct, width=0.82, color=ps.NATIVE,
           edgecolor="white", linewidth=0.3, zorder=3, label="ReLU output")

    # Mean reference line (grey dotted, canonical ideal/reference style).
    ax.axhline(mean_pct, **ps.IDEAL, zorder=4,
               label=f"mean {mean_pct:.1f}%")

    # Light stage demarcation from numel drops (visual only).
    for s in stage_spans(numels)[1:]:
        ax.axvline(s - 0.5, color="0.7", ls="-", lw=0.6, zorder=1)

    ax.set_xlabel(ps.LABELS.RELU_LAYER)
    ax.set_ylabel(ps.LABELS.ACT_SPARSITY)
    ax.set_xlim(-0.7, len(ordinals) - 0.3)
    ax.set_ylim(0, max(100.0, max(zero_pct) * 1.08))
    ax.set_xticks([o for o in ordinals if o % 5 == 0])

    ps.plot_heading(ax, vendor="AMD", gpu="MI300X",
                    sub_parts=["FP32", "post-ReLU zero fraction",
                               f"{len(ordinals)} ReLU layers"])
    ps.cleanup(ax)
    ax.legend(loc="upper right", ncol=2)

    ps.save(fig, args.out, out_dir=args.out_dir)
    print(f"wrote {Path(args.out_dir) / args.out}.pdf / .png  "
          f"(mean activation sparsity {mean_pct:.1f}% over {len(ordinals)} layers)")


if __name__ == "__main__":
    main()
