#!/usr/bin/env python3
"""PyTorch vs custom-HIP side-by-side at the PRODUCTION config (measured).

1 GPU, batch 256, FP32 (amp off), MI300X. All four bars are MEASURED:
  - throughput      <- profiles/<job>/summary.csv          (img/s)
  - GPU occupancy   <- rocprofv3 kernel traces, steady state, warmup excluded
        HIP : full-epoch kernel-trace, busiest-agent union busy% (warmup
              trimmed) -> ~90% busy
        PT  : per-TrainStep occupancy from the marker trace, median over the
              steady steps (the 12s TrainStep0 autotune is excluded) -> ~98%

This REPLACES the earlier version, which used a bogus 65%-idle number from a
mislabeled batch-32 trace and a projected "idle removed" bar. At batch 256 BOTH
runtimes keep the GPU busy; the 1.26x gap is mostly kernel efficiency, not idle.

House style: paper_style.py.

Usage:
    python3 plot_pt_vs_hip.py [--job profiles/hyp_in1k_52387299]
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

# Measured steady-state GPU occupancy (busy %), 1 GPU / batch 256.
# Updated: job 52393128 — full ImageNet-1K, 1.28 M images, 224², FP32.
# HIP : warmup-trimmed union busy% of the full-epoch kernel trace.
# PT  : median per-TrainStep occupancy over the steady steps (marker trace).
HIP_BUSY, PT_BUSY = 90.0, 98.0


def throughput(job):
    # Pick the 1-GPU rows (gpus == 1) so the plot always shows 1-GPU numbers.
    rows_1gpu = {}
    with open(Path(job) / "summary.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("gpus", "1") == "1":
                rows_1gpu[row["arm"]] = float(row["img_per_sec"])
    hip = next(v for k, v in rows_1gpu.items() if k.startswith("N0") or "baseline" in k)
    pt  = next(v for k, v in rows_1gpu.items() if k.startswith("PyTorch"))
    return pt, hip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="profiles/hyp_in1k_52387299")
    ap.add_argument("--out", default="plots/mi300x")
    ap.add_argument("--name", default="rn50_pt_vs_hip")
    args = ap.parse_args()

    pt_ips, hip_ips = throughput(args.job)
    gap = pt_ips / hip_ips
    occ_ratio = PT_BUSY / HIP_BUSY
    eff_ratio = gap / occ_ratio          # kernel-efficiency component
    print(f"PyTorch {pt_ips:.0f} | HIP {hip_ips:.0f} img/s | gap {gap:.2f}x  "
          f"(occupancy {occ_ratio:.2f}x x efficiency {eff_ratio:.2f}x)")

    ps.setup()
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=ps.figsize("text", height=2.7),
        gridspec_kw=dict(width_ratios=[1.0, 1.0], wspace=0.33))

    PT_C, HIP_C = ps.PALETTE[0], ps.PALETTE[1]
    BUSY, IDLE = ps.PALETTE[2], "#d9d9d9"

    # ── (a) throughput ──────────────────────────────────────────────────────
    x = np.arange(2)
    vals = [pt_ips, hip_ips]
    axA.bar(x, vals, color=[PT_C, HIP_C], width=0.6)
    for xi, v in zip(x, vals):
        axA.text(xi, v + max(vals) * 0.015, f"{v:,.0f}", ha="center",
                 va="bottom", fontsize=8, fontweight="bold")
    axA.annotate("", xy=(0, pt_ips), xytext=(1, hip_ips),
                 arrowprops=dict(arrowstyle="<->", color="#444", lw=0.8))
    axA.text(0.5, (pt_ips + hip_ips) / 2 + max(vals) * 0.04,
             f"{gap:.2f}× gap", ha="center", fontsize=7.5,
             color="#444", fontweight="bold")
    axA.set_xticks(x); axA.set_xticklabels(["PyTorch", "HIP custom"])
    axA.set_ylabel(ps.LABELS.THROUGHPUT)
    axA.set_ylim(0, max(vals) * 1.22)
    axA.set_title("(a) Throughput @ 1×MI300X (batch 256, FP32)", fontsize=8.0)

    # ── (b) measured GPU occupancy, side by side ────────────────────────────
    busy = [PT_BUSY, HIP_BUSY]
    idle = [100 - PT_BUSY, 100 - HIP_BUSY]
    axB.bar(x, busy, color=BUSY, width=0.6, label="GPU busy")
    axB.bar(x, idle, bottom=busy, color=IDLE, width=0.6, label="GPU idle")
    for xi, b in zip(x, busy):
        axB.text(xi, b / 2, f"{b:.0f}%", ha="center", va="center",
                 color="white", fontsize=8.5, fontweight="bold")
        axB.text(xi, b + (100 - b) / 2, f"{100-b:.0f}%", ha="center",
                 va="center", color="#555", fontsize=7)
    axB.set_xticks(x); axB.set_xticklabels(["PyTorch", "HIP custom"])
    axB.set_ylim(0, 100)
    axB.set_ylabel("Steady-state GPU busy (%)")
    axB.set_title("(b) Both keep the GPU busy — gap isn't idle", fontsize=8.0)
    axB.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30),
               ncol=2, fontsize=6.8, frameon=False)

    for ax in (axA, axB):
        ps.cleanup(ax)

    ps.figure_heading(
        fig, vendor="AMD", gpu="MI300X",
        sub_parts=["PyTorch vs custom HIP (measured)",
                   f"1.26× gap ≈ occupancy {occ_ratio:.2f}× × kernel-eff {eff_ratio:.2f}×"])
    fig.subplots_adjust(top=0.80, bottom=0.20, left=0.09, right=0.985)
    ps.save(fig, args.name, out_dir=args.out)
    print(f"wrote {args.out}/{args.name}.pdf + .png")


if __name__ == "__main__":
    main()
