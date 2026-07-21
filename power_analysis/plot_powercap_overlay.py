#!/usr/bin/env python3
"""
Overlay two (or three) power-cap sweeps on shared axes -- Native CUDA(->HIP) vs
NGC PyTorch (optionally + a third stack).

Reads cap-sweep `power_sweep_summary.csv` files (rows labelled cap_<W>, with
columns img_per_sec, j_per_image, img_per_sec_per_watt, n_gpus) and draws a row
of panels vs the power cap:

  default          (a) J/image          (b) img/s/W          (c) throughput
  --ipw-on-first   (a) J/image + img/s/W (twin axis, merged) (b) throughput
                   (c) EDP knee (per-image EDP normalised to its own minimum)

Inter-panel spacing is handled automatically by constrained_layout, so there is
no manual gap to tune (the twin right axis in --ipw-on-first is accounted for).

  --ieee   render at IEEE two-column text width (7.16 in) with compact fonts,
           sized to stay legible inside a LaTeX figure* (in-figure suptitle is
           dropped -- put the description in the \\caption). Default is a large
           standalone figure.

Usage (works from any folder; writes <out-dir>/<out-name>.{png,pdf}):
    python3 plot_powercap_overlay.py \
        --cuda  path/to/cuda/power_sweep_summary.csv \
        --ngc   path/to/ngc/power_sweep_summary.csv \
        [--amd path/to/amd/power_sweep_summary.csv] \
        [--ipw-on-first] [--ieee] [--out-dir .] [--out-name my_powercap_sweep]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(".")   # default output directory (current folder -> portable)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Linux Libertine O", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11.0,
    "axes.titleweight": "bold",     # panel titles bold
    "axes.labelweight": "bold",     # x/y axis labels bold (incl. twin axis)
    "figure.titleweight": "bold",   # suptitle bold
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def load(path):
    """Return [(cap_w, img_per_sec, j_per_image, img_per_sec_per_watt), ...] sorted by cap."""
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            label = r.get("label", "")
            if not label.startswith("cap_"):
                continue
            tok = label[4:]
            if tok == "default":
                continue
            try:
                cap = float(tok)
                ips = float(r["img_per_sec"])
                jpi = float(r["j_per_image"])
                ipw = float(r["img_per_sec_per_watt"])
            except (ValueError, KeyError):
                continue
            rows.append((cap, ips, jpi, ipw))
    rows.sort(key=lambda x: x[0])
    return rows


def peek_ngpus(path):
    """Read the n_gpus column from the first data row of a summary CSV."""
    try:
        for r in csv.DictReader(open(path, newline="")):
            v = r.get("n_gpus", "")
            if v not in ("", None):
                return int(float(v))
    except (OSError, ValueError, TypeError):
        pass
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cuda", type=Path, required=True, help="Native CUDA power_sweep_summary.csv")
    ap.add_argument("--ngc", type=Path, required=True, help="NGC PyTorch power_sweep_summary.csv")
    ap.add_argument("--amd", type=Path, default=None, help="optional third-stack power_sweep_summary.csv")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR, help="output directory (default: .)")
    ap.add_argument("--out-name", default="powercap_sweep_cuda_vs_pytorch_mi300x",
                    help="output filename stem")
    ap.add_argument("--ipw-on-first", action="store_true",
                    help="merge img/s/W onto the first (J/image) panel via a twin "
                         "right axis (dashed); third panel becomes the EDP knee")
    ap.add_argument("--ieee", action="store_true",
                    help="render at IEEE two-column text width (7.16 in), compact fonts, "
                         "no in-figure suptitle (use the LaTeX caption instead)")
    ap.add_argument("--trial", action="store_true",
                    help="compact trial: SMALLER canvas with proportionally BIGGER text and "
                         "THINNER lines -- text stays legible when the figure is scaled down, "
                         "all data kept. No suptitle (use the caption).")
    ap.add_argument("--fig-w", type=float, default=None,
                    help="override figure WIDTH (in) -- e.g. 7.16 for IEEE textwidth")
    ap.add_argument("--fig-h", type=float, default=None,
                    help="override figure HEIGHT (in) -- shorten for a wide, low-profile row")
    args = ap.parse_args()

    # size preset: --ieee = compact, legible at \textwidth; default = large standalone.
    if args.trial:
        # smaller image + bigger relative fonts + thinner lines: trend stays clear
        # at a reduced print size without dropping any data points.
        mpl.rcParams.update({
            "font.size": 8.5, "axes.titlesize": 8.0, "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0, "ytick.labelsize": 8.0, "legend.fontsize": 8.0,
            "lines.linewidth": 0.9, "axes.linewidth": 0.6,
        })
        FIG_W, FIG_H, MS, RING_S, ANN_FS, TITLE_FS, LW = 5.6, 2.15, 2.3, 26, 7.0, 8.0, 0.9
    elif args.ieee:
        mpl.rcParams.update({
            "font.size": 7.5, "axes.titlesize": 8.0, "axes.labelsize": 8.0,
            "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.6,
            "lines.linewidth": 1.2, "axes.linewidth": 0.7,
        })
        FIG_W, FIG_H, MS, RING_S, ANN_FS, TITLE_FS, LW = 7.16, 2.55, 3.0, 42, 6.2, 8.0, 1.3
    else:
        FIG_W_IPW, FIG_W_STD, FIG_H = 15.0, 13.5, 4.0
        MS, RING_S, ANN_FS, TITLE_FS, LW = 5, 120, 9.0, 11.0, 1.9

    # MI300X (mixed precision): Native CUDA(->HIP) = orange (#D55E00), NGC = blue.
    series = [("HIP ported ", "#D55E00", "o", load(args.cuda)),
              ("NGC PyTorch ", "#1f6feb", "s", load(args.ngc))]
    if args.amd:
        series.append(("AMD ROCm", "#2f9e44", "^", load(args.amd)))

    if args.ipw_on_first:
        # (1) J/image + img/s/W merged (twin axis)  (2) throughput  (3) EDP knee
        panels = [
            ("J / image (solid) + img/s/W (dashed)", "J / image", 2, True),
            ("Throughput (performance cost)", "images / s", 1, False),
            ("EDP knee", "EDP / its minimum", 4, True),   # col 4 = per-image EDP (computed)
        ]
        fig_w = FIG_W if (args.ieee or args.trial) else FIG_W_IPW
    else:
        panels = [
            ("J / image (lower = better)", "J / image", 2, True),
            ("Energy efficiency", "images / s / W", 3, False),
            ("Throughput (performance cost)", "images / s", 1, False),
        ]
        fig_w = FIG_W if (args.ieee or args.trial) else FIG_W_STD
    if args.fig_w is not None and (args.ieee or args.trial):
        FIG_W = fig_w = args.fig_w          # explicit width override (compact modes)
    if args.fig_h is not None:
        FIG_H = args.fig_h                  # explicit height override (any mode)
    if args.trial:   # short titles so they don't collide at the smaller trial size
        _short = {"J / image (solid) + img/s/W (dashed)": "Energy & efficiency",
                  "Throughput (performance cost)": "Throughput",
                  "J / image (lower = better)": "Energy / image",
                  "Energy efficiency": "Efficiency"}
        panels = [(_short.get(t, t), y, c, m) for (t, y, c, m) in panels]
    fig_h = FIG_H
    # constrained_layout spaces panels automatically. The throughput column is a
    # BROKEN y-axis: a tall top (5000-12000) over a short 0-stub, joined by a kink,
    # so the 0 baseline is visible without a big empty 0-5000 region.
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    gs = fig.add_gridspec(1, 3)
    tput_pos = [p[2] for p in panels].index(1)   # column holding throughput (col == 1)
    axes, tput_bot = [], None
    for j in range(3):
        # every column uses the SAME 2-row subgridspec so all three panels are
        # exactly equal width; the throughput column splits it (data + 0-stub),
        # the other two span both rows (full height).
        sub = gs[0, j].subgridspec(2, 1, height_ratios=[22, 1], hspace=0.05)
        if j == tput_pos:
            axes.append(fig.add_subplot(sub[0, 0]))   # tall top axis (data)
            tput_bot = fig.add_subplot(sub[1, 0])     # thin bottom stub (the 0 baseline)
        else:
            axes.append(fig.add_subplot(sub[:, 0]))   # full-height panel

    for idx, (ax, (title, ylab, col, mark_min)) in enumerate(zip(axes, panels)):
        bot = tput_bot if col == 1 else None
        for name, color, marker, rows in series:
            if not rows:
                continue
            xs = [r[0] for r in rows]
            if col == 4:   # per-image EDP = J/img * delay(ms), normalised to its own min
                raw = [r[2] * (1000.0 / r[1]) for r in rows]
                mn = min(raw)
                ys = [v / mn for v in raw]
            else:
                ys = [r[col] for r in rows]
            ax.plot(xs, ys, marker=marker, color=color, label=name, markersize=MS)
            if bot is not None:        # mirror onto the 0-stub (data is >5000 -> clipped away)
                bot.plot(xs, ys, marker=marker, color=color, markersize=MS)
            if mark_min:
                i = min(range(len(ys)), key=lambda k: ys[k])
                ax.scatter([xs[i]], [ys[i]], s=RING_S, facecolors="none",
                           edgecolors=color, linewidths=1.6, zorder=5)
                ax.annotate(f"{int(xs[i])}W", (xs[i], ys[i]),
                            textcoords="offset points", xytext=(5, 6),
                            fontsize=ANN_FS, color=color)
        if col == 4:
            ax.axhline(1.0, color="#bbbbbb", ls=":", lw=1.0, zorder=1)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=TITLE_FS)
        ax.grid(True, color="#dddddd", linewidth=0.7)
        if col == 1:   # ONE shared legend in the throughput panel's empty top-left
            ax.legend(frameon=False, loc="upper left")   # curves rise L->R, so TL is clear

        if bot is not None:   # finish the broken throughput axis (top + 0-stub + kink)
            ax.set_ylim(4000, 12000)
            ax.set_yticks(range(4000, 12000, 1000))    # 4000 .. 11000 (top end unchanged)
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(bottom=False, labelbottom=False)
            bot.set_xlim(ax.get_xlim())
            bot.set_xticks([200, 400, 600])
            bot.set_ylim(0, 600); bot.set_yticks([0])
            bot.spines["top"].set_visible(False)
            bot.grid(True, color="#dddddd", linewidth=0.7)
            bot.set_xlabel("power cap (W / GPU)")
            # diagonal kink marks (fixed-size marker -> consistent across the two heights)
            kk = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=7, linestyle="none",
                      color="k", mec="k", mew=1, clip_on=False)
            ax.plot([0, 1], [0, 0], transform=ax.transAxes, **kk)
            bot.plot([0, 1], [1, 1], transform=bot.transAxes, **kk)
        else:
            ax.set_xlabel("power cap (W / GPU)")

    if args.ipw_on_first:   # overlay img/s/W on the first panel via a twin right axis
        axR = axes[0].twinx()
        for name, color, marker, rows in series:
            if not rows:
                continue
            xs = [r[0] for r in rows]
            axR.plot(xs, [r[3] for r in rows], color=color, marker=marker, ls="--",
                     markersize=MS, markerfacecolor="white", markeredgecolor=color, lw=LW)
        axR.set_ylabel("images / s / W")

    if not (args.ieee or args.trial):   # compact modes: description goes in the \caption
        ngpus = peek_ngpus(args.cuda) or peek_ngpus(args.ngc) or 8
        fig.suptitle(f"ResNet-50 v1.5 ImageNet-1K: power-cap sweep, {ngpus}×MI300X "
                     "(Native CUDA vs NGC PyTorch)", fontsize=12)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = args.out_dir / f"{args.out_name}.{ext}"
        fig.savefig(out, dpi=200)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
