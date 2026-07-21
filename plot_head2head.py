#!/usr/bin/env python3
"""
plot_head2head.py — Native-HIP vs PyTorch-ROCm throughput head-to-head on MI300X.

Plots the img/s produced by hyp_imagenet_exp_amd.slurm (RESULTS_DIR/summary.csv).
Two panels, paper-styled (Native=vermillion, PyTorch=blue):

  (a) Throughput (img/s) vs #GPUs, with the ideal-linear guide.
  (b) Strong-scaling efficiency (%) vs #GPUs, with the 100% line.

Default source is the canonical full-1K run (job 52393128); the earlier 8-GPU
native points (6161-6333 img/s) were on degraded data and are intentionally
NOT used. Override with --summary <path>.
"""
from __future__ import annotations
import argparse, csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import paper_style as ps

ARM_LABEL = {"N0_baseline_fm3": "Native HIP", "N1_fixed_fm1": "Native HIP (fm1)",
             "PyTorch_fp32": "PyTorch"}
ARM_COLOR = {"Native HIP": ps.NATIVE, "Native HIP (fm1)": ps.GREEN,
             "PyTorch": ps.PYTORCH}


def load(path):
    series = defaultdict(dict)            # label -> {gpus: img/s}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            lab = ARM_LABEL.get(r["arm"], r["arm"])
            try:
                series[lab][int(r["gpus"])] = float(r["img_per_sec"])
            except (ValueError, KeyError):
                continue
    return series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="profiles/hyp_in1k_52393128/summary.csv")
    ap.add_argument("--tag", default="52393128")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()

    series = load(args.summary)
    if not series:
        raise SystemExit(f"no usable rows in {args.summary}")

    ps.setup(); ps.scaling_fonts()
    fig, (axT, axE) = plt.subplots(
        1, 2, figsize=(7.0, 3.4),
        gridspec_kw=dict(wspace=0.28, left=0.105, right=0.99, top=0.85, bottom=0.22))

    # union of GPU counts present, sorted
    gpus_all = sorted({g for d in series.values() for g in d})
    style = ps.MODE_STYLES["strong"]

    impls = []
    for lab, d in series.items():
        gs = sorted(d)
        tput = [d[g] for g in gs]
        color = ARM_COLOR.get(lab, ps.BLACK)
        ps.draw_series(axT, gs, tput, color, style)
        # efficiency relative to this series' own 1-GPU point
        if 1 in d:
            base = d[1]
            eff = [100.0 * d[g] / (base * g) for g in gs]
            ps.draw_series(axE, gs, eff, color, style)
        impls.append((color, lab))

    # ideal-linear guide on throughput (scale each series' own 1-GPU baseline —
    # use the fastest 1-GPU as the reference line so both sit under it)
    base1 = max((d[1] for d in series.values() if 1 in d), default=None)
    if base1:
        axT.plot(gpus_all, [base1 * g for g in gpus_all], **ps.IDEAL)
    axE.axhline(100, **ps.IDEAL)

    axT.set_xlabel(ps.LABELS.GPUS);  axT.set_ylabel(ps.LABELS.THROUGHPUT)
    axT.set_title("(a) Throughput")
    axE.set_xlabel(ps.LABELS.GPUS);  axE.set_ylabel(ps.LABELS.EFFICIENCY)
    axE.set_title("(b) Strong-scaling efficiency")
    for a in (axT, axE):
        a.set_xticks(gpus_all); a.set_xlim(min(gpus_all) - 0.4, max(gpus_all) + 0.4)
        ps.bold_ticks(a)
    axE.set_ylim(0, 110)

    ps.dual_legend(axT, impls, modes=False, ideal=True, loc="upper left")
    ps.cleanup(axT); ps.cleanup(axE)
    fig.suptitle("ResNet-50 on MI300X — Native HIP vs PyTorch (strong scaling)",
                 fontweight="bold", fontsize=12)
    ps.save(fig, f"head2head_{args.tag}", out_dir=args.out_dir)

    # console recap
    for lab, d in series.items():
        gs = sorted(d)
        if 1 in d and len(gs) > 1:
            g = gs[-1]; sp = d[g] / d[1]
            print(f"  {lab:18s} 1GPU={d[1]:.0f}  {g}GPU={d[g]:.0f}  "
                  f"speedup={sp:.2f}x  eff={100*sp/g:.1f}%")
    print(f"  -> {args.out_dir}/head2head_{args.tag}.pdf (+ .png)")


if __name__ == "__main__":
    main()
