#!/usr/bin/env python3
"""Paper-ready power-cap row figure (Native HIP vs NGC PyTorch on MI300X).

A cleaner, IEEE-textwidth replacement for the busy --ipw-on-first overlay:
  (a) Energy per image   (J/image, lower=better) with the energy-optimal cap
  (b) Throughput         (img/s, the performance cost of capping)
  (c) Energy-delay product (per-image EDP, normalised to each stack's own min)

What changed vs plot_powercap_overlay.py --ipw-on-first:
  * Panel (a) no longer overlays img/s/W on a twin axis (that dashed curve is
    ~1/J/image, so it duplicated panel (a) and crossed it in a hard-to-read X).
    The efficiency/return story is carried by panels (a)+(c).
  * Optimal-cap markers no longer collide: both stacks share the 550 W energy
    optimum and the 700-750 W EDP optimum, so each is shown once as a guide line
    + single label rather than two overlapping annotations.
  * Canonical palette (Native vermillion / PyTorch blue), one shared legend,
    compact fonts sized for a \\textwidth figure, vector PDF companion.

Usage:
  python3 plot_powercap_paper.py \
     --cuda results/native/power_sweep_summary_overlay.csv \
     --ngc  results/pytorch/power_sweep_summary_overlay.csv \
     --out-dir results/figures --out-name powercap_row
"""
from __future__ import annotations
import argparse, csv, os, sys
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

# shared paper style (serif, TrueType) when available
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import paper_style as ps
except Exception:
    ps = None

NATIVE, PYTORCH = "#D55E00", "#0072B2"          # canonical palette
GUIDE = "#888888"


def load(path):
    rows = []
    for r in csv.DictReader(open(path, newline="")):
        lab = r.get("label", "")
        if not lab.startswith("cap_") or lab == "cap_default":
            continue
        try:
            rows.append((float(lab[4:]), float(r["img_per_sec"]),
                         float(r["j_per_image"]), float(r["img_per_sec_per_watt"])))
        except (ValueError, KeyError):
            continue
    rows.sort()
    return rows


def _argmin(xs, ys):
    i = min(range(len(ys)), key=lambda k: ys[k])
    return xs[i], ys[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", required=True)
    ap.add_argument("--ngc", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--out-name", default="powercap_row")
    a = ap.parse_args()

    series = [("Native HIP", NATIVE, "o", load(a.cuda)),
              ("PyTorch (NGC)", PYTORCH, "s", load(a.ngc))]

    if ps is not None:
        try: ps.setup()
        except Exception: pass
    mpl.rcParams.update({
        "font.size": 8.0, "axes.titlesize": 9.0, "axes.titleweight": "bold",
        "axes.labelsize": 8.5, "axes.labelweight": "bold",
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "lines.linewidth": 1.5, "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.16, 2.55))
    MS = 3.6

    # ---- (a) Energy per image ----
    for name, c, mk, rows in series:
        xs = [r[0] for r in rows]; jpi = [r[2] for r in rows]
        axA.plot(xs, jpi, marker=mk, color=c, ms=MS, label=name)
        ox, oy = _argmin(xs, jpi)
        axA.scatter([ox], [oy], s=58, facecolors="none", edgecolors=c,
                    linewidths=1.6, zorder=6)
    # both stacks bottom out at 550 W -> one guide line + one label
    axA.axvline(550, color=GUIDE, ls=":", lw=1.0, zorder=1)
    axA.annotate("energy-opt\n550 W", (550, axA.get_ylim()[1]),
                 xytext=(2, -2), textcoords="offset points",
                 ha="left", va="top", fontsize=6.8, color="#333333",
                 fontweight="bold")
    axA.set_title("(a) Energy per image")
    axA.set_xlabel("power cap (W / GPU)")
    axA.set_ylabel("energy  (J / image)")
    axA.grid(True, color="#dddddd", linewidth=0.7)

    # ---- (b) Throughput ---- (carries the one shared legend; top-left is empty)
    for name, c, mk, rows in series:
        axB.plot([r[0] for r in rows], [r[1] for r in rows], marker=mk,
                 color=c, ms=MS, label=name)
    axB.set_title("(b) Throughput")
    axB.set_xlabel("power cap (W / GPU)")
    axB.set_ylabel("throughput  (img / s)")
    axB.grid(True, color="#dddddd", linewidth=0.7)
    axB.set_ylim(4000, 11000)
    axB.legend(loc="upper left", frameon=False, handlelength=1.4)

    # ---- (c) Energy-delay product (per-stack normalised) ----
    edp_opt = []
    for name, c, mk, rows in series:
        xs = [r[0] for r in rows]
        edp = [r[2] * 1000.0 / r[1] for r in rows]
        m = min(edp); ys = [v / m for v in edp]
        axC.plot(xs, ys, marker=mk, color=c, ms=MS, label=name)
        ox, oy = _argmin(xs, ys)
        axC.scatter([ox], [oy], s=58, facecolors="none", edgecolors=c,
                    linewidths=1.6, zorder=6)
        edp_opt.append(ox)
    axC.axhline(1.0, color="#bbbbbb", ls=":", lw=1.0, zorder=1)
    lo, hi = int(min(edp_opt)), int(max(edp_opt))
    axC.annotate(f"EDP-opt\n{lo}-{hi} W", (hi, 1.0), xytext=(-2, 18),
                 textcoords="offset points", ha="right", va="bottom",
                 fontsize=6.8, color="#333333", fontweight="bold")
    axC.set_title("(c) Energy-delay product")
    axC.set_xlabel("power cap (W / GPU)")
    axC.set_ylabel("EDP / its minimum")
    axC.grid(True, color="#dddddd", linewidth=0.7)

    fig.tight_layout(w_pad=1.1)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = os.path.join(a.out_dir, f"{a.out_name}.{ext}")
        fig.savefig(out, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print("wrote", out)

    for name, _c, _mk, rows in series:
        xs = [r[0] for r in rows]; jpi = [r[2] for r in rows]
        ox, oy = _argmin(xs, jpi)
        print(f"  {name}: energy-opt {ox:.0f} W -> {oy:.3f} J/img")


if __name__ == "__main__":
    main()
