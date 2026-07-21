#!/usr/bin/env python3.11
"""
Plot measured vs. predicted results from the analytical power/energy model.

Reads the validation_*.json files produced by the model validation step (each
holds, per implementation, the predicted per-GPU dynamic power broken down by
component plus the measured per-GPU power, and predicted/measured total energy).

Produces a two-panel figure:
  (a) Per-GPU dynamic power (W): a "Measured" bar beside a "Predicted" bar that
      is stacked by component (MFMA-FP32 / MFMA-BF16 / GEMM / HBM / Comm).
  (b) Prediction error (%): dynamic-power and total-energy error per case, so the
      model accuracy is readable independent of the (duration-dependent) energy
      scale.

Usage
-----
    # default: all power_analysis/validation_*_final.json (1-GPU then 8-GPU)
    /usr/bin/python3.11 power_analysis/plot_validation.py

    # explicit files / output name
    /usr/bin/python3.11 power_analysis/plot_validation.py \
        power_analysis/validation_1gpu_final.json \
        power_analysis/validation_8gpu_final.json \
        -o power_analysis/fig_model_validation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# paper_style.py lives at the repo root (one level above this dir).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paper_style as ps  # noqa: E402

HERE = Path(__file__).resolve().parent

# Predicted-power components, in stacking order, with colours from the Wong
# palette + a stable display label.
COMPONENTS = [
    ("mfma_fp32", "MFMA FP32", ps.PALETTE[0]),  # blue
    ("mfma_bf16", "MFMA BF16", ps.PALETTE[5]),  # sky blue
    ("gemm",      "GEMM",      ps.PALETTE[6]),  # orange
    ("hbm",       "HBM",       ps.PALETTE[2]),  # green
    ("comm",      "Comm",      ps.PALETTE[3]),  # purple
]
MEAS_COLOR = "0.55"          # neutral grey for the measured bar
IDLE_COLOR = "0.80"          # light grey for the idle-energy slice (unused panel)


def short_impl(name: str) -> str:
    """'native (in-sample)' -> 'Native', 'pytorch (cross-stack)' -> 'PyTorch'."""
    base = name.split("(")[0].strip().lower()
    return {"native": "Native", "pytorch": "PyTorch"}.get(base, base.title())


def load_cases(files):
    """Flatten {file -> {impl -> result}} into an ordered list of cases."""
    cases = []
    for f in files:
        data = json.loads(Path(f).read_text())
        for impl_name, r in data["results"].items():
            ngpu = int(round(r.get("ngpu", 1)))
            cases.append({
                "impl":       short_impl(impl_name),
                "ngpu":       ngpu,
                "label":      f"{short_impl(impl_name)}\n{ngpu}-GPU",
                "pred":       r["pred_power_pergpu_W"],
                "meas_power": r["P_dyn_meas_pergpu_W"],
                "err_dyn":    r["err_dyn_pct"],
                "err_total":  r["err_total_pct"],
            })
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="validation_*.json files to plot")
    ap.add_argument("-o", "--out", default=str(HERE / "fig_model_validation"),
                    help="output path stem (no extension); writes .pdf + .png")
    args = ap.parse_args()

    files = args.files
    if not files:
        files = sorted(HERE.glob("validation_*_final.json"),
                       key=lambda p: 0 if "1gpu" in p.name else 1)
        files = [str(p) for p in files]
    if not files:
        ap.error("no validation_*.json files given or found in power_analysis/")
    print("Plotting:", *[Path(f).name for f in files], sep="\n  ")

    cases = load_cases(files)
    n = len(cases)
    x = np.arange(n)

    ps.setup()
    fig, (axP, axE) = plt.subplots(
        1, 2, figsize=ps.figsize("text", height=3.1),
        gridspec_kw=dict(wspace=0.30, left=0.075, right=0.995, top=0.82, bottom=0.20))

    # ── Panel (a): per-GPU dynamic power, measured vs predicted(stacked) ──────
    bw = 0.38
    xm, xp = x - bw / 2, x + bw / 2
    axP.bar(xm, [c["meas_power"] for c in cases], width=bw,
            color=MEAS_COLOR, edgecolor="black", linewidth=0.5, label="Measured")

    bottom = np.zeros(n)
    for key, lbl, col in COMPONENTS:
        vals = np.array([c["pred"].get(key, 0.0) for c in cases])
        if not np.any(vals):
            continue
        axP.bar(xp, vals, width=bw, bottom=bottom, color=col,
                edgecolor="white", linewidth=0.4, label=lbl)
        bottom += vals

    # annotate measured / predicted totals so the comparison reads at a glance
    ymax = max(bottom.max(), max(c["meas_power"] for c in cases))
    for xi, c in zip(xm, cases):
        axP.text(xi, c["meas_power"] + ymax * 0.01, f"{c['meas_power']:.0f}",
                 ha="center", va="bottom", fontsize=6.0, color="0.25")
    for xi, tot in zip(xp, bottom):
        axP.text(xi, tot + ymax * 0.01, f"{tot:.0f}",
                 ha="center", va="bottom", fontsize=6.0, color="0.25")

    axP.set_ylabel("Per-GPU dynamic power (W)")
    axP.set_ylim(0, ymax * 1.16)
    axP.set_xticks(x)
    axP.set_xticklabels([c["label"] for c in cases])
    axP.set_title("(a) Dynamic power: measured vs. predicted", fontsize=8.5)
    ps.cleanup(axP)
    axP.grid(axis="x", visible=False)

    # ── Panel (b): prediction error (%) for dynamic power & total energy ──────
    ew = 0.36
    e_dyn = [c["err_dyn"] for c in cases]
    e_tot = [c["err_total"] for c in cases]
    b1 = axE.bar(x - ew / 2, e_dyn, width=ew, color=ps.PALETTE[0],
                 edgecolor="black", linewidth=0.5, label="Dynamic power")
    b2 = axE.bar(x + ew / 2, e_tot, width=ew, color=ps.PALETTE[6],
                 edgecolor="black", linewidth=0.5, label="Total energy")
    axE.axhline(0, color="black", linewidth=0.8)

    lo = min(min(e_dyn), min(e_tot))
    hi = max(max(e_dyn), max(e_tot), 0.0)
    pad = (hi - lo) * 0.16 or 2.0
    for bars in (b1, b2):
        for rect in bars:
            v = rect.get_height()
            off = pad * 0.18
            axE.text(rect.get_x() + rect.get_width() / 2,
                     v - off if v < 0 else v + off,
                     f"{v:+.0f}%", ha="center",
                     va="top" if v < 0 else "bottom", fontsize=6.0, color="0.20")

    axE.set_ylabel("Prediction error (%)")
    axE.set_ylim(lo - pad, hi + pad)
    axE.set_xticks(x)
    axE.set_xticklabels([c["label"] for c in cases])
    axE.set_title("(b) Prediction error (pred $-$ meas)", fontsize=8.5)
    axE.legend(loc="lower left", fontsize=7.0)
    ps.cleanup(axE)
    axE.grid(axis="x", visible=False)

    # shared component legend for panel (a) below the figure
    handles = [Patch(facecolor=MEAS_COLOR, edgecolor="black", label="Measured")]
    handles += [Patch(facecolor=col, edgecolor="white", label=lbl)
                for key, lbl, col in COMPONENTS
                if any(c["pred"].get(key, 0.0) for c in cases)]
    fig.legend(handles, [h.get_label() for h in handles],
               loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=len(handles), frameon=False, fontsize=7.5,
               columnspacing=1.1, handlelength=1.3)

    fig.suptitle("Analytical power model — measured vs. predicted",
                 fontsize=10.5, fontweight="bold", y=0.985)

    out = Path(args.out)
    ps.save(fig, out.name, out_dir=out.parent)
    print(f"figure name: {out.name}")
    print(f"wrote {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()
