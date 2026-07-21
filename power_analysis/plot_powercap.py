#!/usr/bin/env python3
"""
Portable power-cap energy plots (H100 / MI300X) -- canonical version.

Turns a power_sweep_summary.csv (one row per power cap) into the
energy-optimal-tuning figures. Vendor-agnostic: pass any mix of summary CSVs
(custom-HIP, ROCm-PyTorch, H100-CUDA) and they overlay on the same axes.

CSV schema (per cap row):
  label,cap_w,n_gpus,avg_power_w_per_gpu,mean_sm_clock_mhz,img_per_sec,
  j_per_image,img_per_sec_per_watt,edp,ed2p[,dynamic_j_per_image]

CLI (thin wrapper; the functions match the canonical .md verbatim):
  python plot_powercap.py --run "HIP custom=hip/power_sweep_summary.csv" \
                          --run "ROCm PyTorch=pt/power_sweep_summary.csv" \
                          --out-dir power_analysis/results/figures --tag powercap
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")          # drop this line inside a notebook to show inline
import matplotlib.pyplot as plt

# ---- styling: canonical paper palette (styling.md) — run 0 = native/custom
# (vermillion), run 1 = PyTorch (blue), run 2/3 = green/purple (e.g. H100, A100).
STYLES = [
    dict(color="#D55E00", marker="o", ls="-"),    # run 0  native / custom  (vermillion)
    dict(color="#0072B2", marker="s", ls="--"),   # run 1  PyTorch           (blue)
    dict(color="#009E73", marker="^", ls="-."),   # run 2                     (green)
    dict(color="#CC79A7", marker="D", ls=":"),    # run 3                     (purple)
]

# apply the shared paper style (Times serif, TrueType, etc.) when available
try:
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    import paper_style as _ps
except Exception:
    _ps = None


def _paper_style():
    if _ps is not None:
        try:
            _ps.setup()
        except Exception:
            pass


def _f(row, key):
    v = row.get(key, "")
    try:
        return float(v) if v not in ("", "NA", None) else None
    except (ValueError, TypeError):
        return None


def load_sweep(path):
    """power_sweep_summary.csv -> list of per-cap dicts, sorted by cap (W)."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            cap = _f(r, "cap_w")
            if cap is None:                       # skip the 'default' row
                continue
            rows.append(dict(
                cap=cap, jpi=_f(r, "j_per_image"), ipw=_f(r, "img_per_sec_per_watt"),
                tput=_f(r, "img_per_sec"), wpg=_f(r, "avg_power_w_per_gpu"),
                clk=_f(r, "mean_sm_clock_mhz"), dyn=_f(r, "dynamic_j_per_image"),
                ngpu=_f(r, "n_gpus")))
    rows.sort(key=lambda d: d["cap"])
    return rows


def _load_runs(runs):
    """runs = [(label, csv_path), ...] -> [(label, rows), ...] with >=2 caps."""
    out = [(lbl, load_sweep(p)) for lbl, p in runs]
    return [(l, r) for l, r in out if len(r) >= 2]


def plot_powercap(runs, out="powercap_sweep.png", title=None):
    """4-panel cap sweep: (a) J/image (b) img/s/W (c) throughput (d) EDP knee.
       runs = [(label, csv_path), ...].  Annotates the energy-optimal cap/run."""
    data = _load_runs(runs)
    if not data:
        raise SystemExit("no runs with >=2 caps")
    _paper_style()
    fig, (axJ, axW, axT, axE) = plt.subplots(1, 4, figsize=(17.5, 4.1))
    for i, (lbl, rows) in enumerate(data):
        st = STYLES[i % len(STYLES)]
        caps = [r["cap"] for r in rows]
        axJ.plot(caps, [r["jpi"] for r in rows], **st, label=lbl)
        jp = [(r["cap"], r["jpi"]) for r in rows if r["jpi"] is not None]
        if jp:
            oc, ov = min(jp, key=lambda t: t[1])      # energy-optimal cap = min J/img
            axJ.scatter([oc], [ov], s=140, facecolors="none",
                        edgecolors=st["color"], linewidths=1.8, zorder=5)
            axJ.annotate(f"{lbl}\nopt {oc:.0f} W\n{ov:.3f} J/img", (oc, ov),
                         textcoords="offset points", xytext=(6, 8),
                         fontsize=8, color=st["color"])
        axW.plot(caps, [r["ipw"] for r in rows], **st, label=lbl)
        axT.plot(caps, [r["tput"] for r in rows], **st, label=lbl)
        # EDP normalised to each run's own minimum (relative knee)
        edp = [(r["jpi"] * 1000.0 / r["tput"]) if (r["jpi"] and r["tput"]) else None
               for r in rows]
        vals = [x for x in edp if x is not None]
        m = min(vals) if vals else 1.0
        axE.plot(caps, [x / m if x is not None else None for x in edp], **st, label=lbl)
    for ax in (axJ, axW, axT, axE):
        ax.set_xlabel("power cap (W/GPU)")
        ax.grid(True, color="#dddddd", linewidth=0.8)
        ax.legend(loc="upper left", frameon=False, fontsize=9)
    axJ.set_ylabel("J / image");          axJ.set_title("(a) Energy per image")
    axW.set_ylabel("img/s / W");          axW.set_title("(b) Performance per watt")
    axT.set_ylabel("throughput (img/s)"); axT.set_title("(c) Throughput")
    axE.set_ylabel("EDP / its minimum");  axE.set_title("(d) EDP knee")
    axE.axhline(1.0, color="#bbbbbb", linewidth=0.8)
    if title:
        fig.suptitle(title, fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)
    for lbl, rows in data:
        jp = [(r["cap"], r["jpi"]) for r in rows if r["jpi"] is not None]
        if jp:
            oc, ov = min(jp, key=lambda t: t[1])
            print(f"  {lbl}: energy-optimal cap {oc:.0f} W -> {ov:.3f} J/img")
    return fig


def plot_combined(runs, out="powercap_combined.png", title=None, also_pdf=True):
    """Merge panels (a) J/image and (b) img/s/W onto ONE dual-y-axis figure.

    Left axis  = J/image (energy per image, lower=better)  -> solid + filled marker
    Right axis = img/s/W  (perf per watt,   higher=better)  -> dashed + open marker
    Colour encodes the stack (Native vermillion / PyTorch blue); the energy-optimal
    (min-J/image) cap per stack is ringed. With also_pdf, writes a vector PDF
    companion next to <out> (highest-quality, paper-ready)."""
    data = _load_runs(runs)
    if not data:
        raise SystemExit("no runs with >=2 caps")
    _paper_style()
    fig, axJ = plt.subplots(figsize=(7.2, 4.6))
    axW = axJ.twinx()
    axW.grid(False)                       # keep only the left axis' gridlines
    jcol, wcol = "#222222", "#222222"     # axis-label tint set per-metric below
    for i, (lbl, rows) in enumerate(data):
        st = STYLES[i % len(STYLES)]
        c = st["color"]
        caps = [r["cap"] for r in rows]
        # left: J/image — solid, filled
        axJ.plot(caps, [r["jpi"] for r in rows], color=c, marker=st["marker"],
                 ls="-", lw=1.6, label=lbl)
        jp = [(r["cap"], r["jpi"]) for r in rows if r["jpi"] is not None]
        if jp:
            oc, ov = min(jp, key=lambda t: t[1])     # energy-optimal cap
            axJ.scatter([oc], [ov], s=150, facecolors="none", edgecolors=c,
                        linewidths=1.8, zorder=6)
            axJ.annotate(f"{lbl}\nopt {oc:.0f} W\n{ov:.3f} J/img", (oc, ov),
                         textcoords="offset points", xytext=(6, -28),
                         fontsize=7.5, color=c, fontweight="bold")
        # right: img/s/W — dashed, open marker
        axW.plot(caps, [r["ipw"] for r in rows], color=c, marker=st["marker"],
                 ls="--", lw=1.4, alpha=0.9, markerfacecolor="none", markeredgecolor=c)
    axJ.set_xlabel("power cap (W / GPU)")
    axJ.set_ylabel("energy per image  (J / image)  ↓ better", color=jcol)
    axW.set_ylabel("performance per watt  (img/s / W)  ↑ better", color=wcol)
    axJ.grid(True, color="#dddddd", linewidth=0.8)
    # two-axis legend: colour = stack, line style = metric
    from matplotlib.lines import Line2D
    stack_h = [Line2D([0], [0], color=STYLES[i % len(STYLES)]["color"],
                      marker=STYLES[i % len(STYLES)]["marker"], ls="-", lw=1.6, label=lbl)
               for i, (lbl, _) in enumerate(data)]
    metric_h = [Line2D([0], [0], color="#555555", ls="-", lw=1.6, marker="o",
                       label="J/image (left)"),
                Line2D([0], [0], color="#555555", ls="--", lw=1.4, marker="o",
                       markerfacecolor="none", label="img/s/W (right)")]
    leg1 = axJ.legend(handles=stack_h, loc="upper left", frameon=False, fontsize=8.5)
    axJ.add_artist(leg1)
    axJ.legend(handles=metric_h, loc="lower right", frameon=False, fontsize=8.0)
    if title:
        fig.suptitle(title, fontsize=10, y=0.99)
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)
    if also_pdf:
        pdf = out.rsplit(".", 1)[0] + ".pdf"
        fig.savefig(pdf)
        print("wrote", pdf)
    return fig


def plot_single(runs, metric, out, title=None, also_pdf=True):
    """One standalone single-panel cap-sweep figure + vector PDF companion.

    metric = 'throughput' -> img/s vs cap   (panel c)
           = 'edpknee'    -> EDP / its own minimum vs cap   (panel d)
    Same palette/markers as the 4-panel sweep so the figures stay a matched set."""
    data = _load_runs(runs)
    if not data:
        raise SystemExit("no runs with >=2 caps")
    _paper_style()
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for i, (lbl, rows) in enumerate(data):
        st = STYLES[i % len(STYLES)]
        caps = [r["cap"] for r in rows]
        if metric == "throughput":
            ax.plot(caps, [r["tput"] for r in rows], **st, label=lbl)
        elif metric == "edpknee":
            edp = [(r["jpi"] * 1000.0 / r["tput"]) if (r["jpi"] and r["tput"]) else None
                   for r in rows]
            vals = [x for x in edp if x is not None]
            m = min(vals) if vals else 1.0
            ax.plot(caps, [x / m if x is not None else None for x in edp], **st, label=lbl)
        else:
            raise SystemExit(f"unknown metric {metric!r}")
    ax.set_xlabel("power cap (W / GPU)")
    ax.grid(True, color="#dddddd", linewidth=0.8)
    ax.legend(loc="upper left" if metric == "throughput" else "upper right",
              frameon=False, fontsize=9)
    if metric == "throughput":
        ax.set_ylabel("throughput (img / s)"); ax.set_title("Throughput")
    else:
        ax.set_ylabel("EDP / its minimum"); ax.set_title("EDP knee")
        ax.axhline(1.0, color="#bbbbbb", linewidth=0.8)
    if title:
        fig.suptitle(title, fontsize=10, y=0.99)
    fig.tight_layout()
    fig.savefig(out); print("wrote", out)
    if also_pdf:
        pdf = out.rsplit(".", 1)[0] + ".pdf"
        fig.savefig(pdf); print("wrote", pdf)
    return fig


def plot_row(runs, out="powercap_row.png", also_pdf=True):
    """The three paper panels side-by-side in ONE IEEE double-column (full-width)
    row: (a) energy+efficiency dual-axis, (b) throughput, (c) EDP knee.
    Width = 7.16in (IEEE \\textwidth); panel (a) gets a little extra room for its
    two y-axes. Vector PDF companion for the paper."""
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D
    data = _load_runs(runs)
    if not data:
        raise SystemExit("no runs with >=2 caps")
    _paper_style()
    fig = plt.figure(figsize=(7.16, 2.85))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.3, 1.0, 1.0], wspace=0.62,
                           left=0.062, right=0.95, bottom=0.21, top=0.88)
    axA, axAr = fig.add_subplot(gs[0]), None
    axB, axC = fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    # ---- (a) energy per image (left, solid) + img/s/W (right, dashed) ----
    axAr = axA.twinx(); axAr.grid(False)
    for i, (lbl, rows) in enumerate(data):
        st = STYLES[i % len(STYLES)]; c = st["color"]; caps = [r["cap"] for r in rows]
        axA.plot(caps, [r["jpi"] for r in rows], color=c, marker=st["marker"],
                 ls="-", lw=1.4, ms=3.0)
        jp = [(r["cap"], r["jpi"]) for r in rows if r["jpi"] is not None]
        if jp:
            oc, ov = min(jp, key=lambda t: t[1])
            axA.scatter([oc], [ov], s=70, facecolors="none", edgecolors=c, linewidths=1.5, zorder=6)
        axAr.plot(caps, [r["ipw"] for r in rows], color=c, marker=st["marker"],
                  ls="--", lw=1.2, ms=3.0, markerfacecolor="none", markeredgecolor=c)
    axA.set_title("(a) Energy & efficiency")
    axA.set_xlabel("power cap (W/GPU)")
    axA.set_ylabel("J / image  ↓", labelpad=1)
    axAr.set_ylabel("img/s / W  ↑", labelpad=2)
    axA.grid(True, color="#dddddd", linewidth=0.7)
    stack_h = [Line2D([0], [0], color=STYLES[i % len(STYLES)]["color"],
                      marker=STYLES[i % len(STYLES)]["marker"], ls="-", lw=1.4, ms=3.2,
                      label=lbl.replace(" MI300X", "")) for i, (lbl, _) in enumerate(data)]
    style_h = [Line2D([0], [0], color="#666666", ls="-", lw=1.4, label="J/img"),
               Line2D([0], [0], color="#666666", ls="--", lw=1.2, label="img/s/W")]
    l1 = axA.legend(handles=stack_h, loc="upper left", frameon=False, fontsize=6.6,
                    handlelength=1.3, borderaxespad=0.2)
    axA.add_artist(l1)
    axA.legend(handles=style_h, loc="lower left", frameon=False, fontsize=6.6,
               handlelength=1.5, borderaxespad=0.2)

    # ---- (b) throughput ----
    for i, (lbl, rows) in enumerate(data):
        st = STYLES[i % len(STYLES)]
        axB.plot([r["cap"] for r in rows], [r["tput"] for r in rows], **st,
                 ms=3.0, label=lbl.replace(" MI300X", ""))
    axB.set_title("(b) Throughput"); axB.set_xlabel("power cap (W/GPU)")
    axB.set_ylabel("throughput (img/s)", labelpad=1)
    axB.grid(True, color="#dddddd", linewidth=0.7)
    axB.legend(loc="upper left", frameon=False, fontsize=6.8, handlelength=1.3)

    # ---- (c) EDP knee ----
    for i, (lbl, rows) in enumerate(data):
        st = STYLES[i % len(STYLES)]
        edp = [(r["jpi"] * 1000.0 / r["tput"]) if (r["jpi"] and r["tput"]) else None
               for r in rows]
        vals = [x for x in edp if x is not None]; m = min(vals) if vals else 1.0
        axC.plot([r["cap"] for r in rows], [x / m if x is not None else None for x in edp],
                 **st, ms=3.0, label=lbl.replace(" MI300X", ""))
    axC.axhline(1.0, color="#bbbbbb", linewidth=0.7)
    axC.set_title("(c) EDP knee"); axC.set_xlabel("power cap (W/GPU)")
    axC.set_ylabel("EDP / its minimum", labelpad=1)
    axC.grid(True, color="#dddddd", linewidth=0.7)
    axC.legend(loc="upper right", frameon=False, fontsize=6.8, handlelength=1.3)

    fig.savefig(out); print("wrote", out)
    if also_pdf:
        pdf = out.rsplit(".", 1)[0] + ".pdf"; fig.savefig(pdf); print("wrote", pdf)
    return fig


def plot_edp(runs, out="edp_sweep.png", title=None, measured_x=False):
    """Energy-delay sweep: (a) per-image E-vs-D trajectory with iso-EDP curves,
       EDP-optimal cap ringed; (b) per-image EDP & ED^2P vs cap (log).

       measured_x=True keys panel (b) on the MEASURED per-GPU power
       (avg_power_w_per_gpu) instead of the nominal cap. Use this when the
       nominal cap is not actually enforced (MI300X clamps low caps to a
       hardware floor ~350 W), so 'cap' is a misleading independent variable."""
    data = _load_runs(runs)
    if not data:
        raise SystemExit("no runs with >=2 caps")
    _paper_style()
    fig, (axS, axB) = plt.subplots(1, 2, figsize=(12.4, 4.7))
    pts_all = [(1000.0 / r["tput"], r["jpi"]) for _, rs in data for r in rs
               if r["tput"] and r["jpi"]]
    dmin, dmax = min(p[0] for p in pts_all) * 0.85, max(p[0] for p in pts_all) * 1.12
    emin, emax = min(p[1] for p in pts_all) * 0.9, max(p[1] for p in pts_all) * 1.06
    best = min(d * e for d, e in pts_all)              # global best per-image EDP
    dg = np.linspace(dmin, dmax, 240)
    for mult in (1.0, 1.25, 1.6, 2.1):                 # iso-EDP hyperbolas E*D=k
        axS.plot(dg, best * mult / dg, ":", color="#aaaaaa", linewidth=0.9)
    for i, (lbl, rows) in enumerate(data):
        st = STYLES[i % len(STYLES)]
        p = [(1000.0 / r["tput"], r["jpi"], r["cap"]) for r in rows
             if r["tput"] and r["jpi"]]
        d = [x[0] for x in p]; e = [x[1] for x in p]
        axS.plot(d, e, "-", color=st["color"], lw=1.4, alpha=0.6)
        axS.scatter(d, e, s=46, color=st["color"], marker=st["marker"],
                    edgecolor="white", lw=0.6, label=lbl)
        bopt = min(p, key=lambda x: x[0] * x[1])       # min EDP point
        axS.scatter([bopt[0]], [bopt[1]], s=210, facecolor="none",
                    edgecolor=st["color"], lw=1.8)
        axS.annotate(f"EDP-opt {bopt[2]:.0f} W", (bopt[0], bopt[1]),
                     textcoords="offset points", xytext=(8, 4), fontsize=8,
                     color=st["color"], fontweight="bold")
        # x-axis: measured per-GPU power (honest) or nominal cap (legacy)
        xs = [(r["wpg"] if r["wpg"] is not None else r["cap"]) for r in rows] \
             if measured_x else [r["cap"] for r in rows]
        edp = [r["jpi"] * (1000.0 / r["tput"]) for r in rows]
        ed2 = [r["jpi"] * (1000.0 / r["tput"]) ** 2 for r in rows]
        axB.plot(xs, edp, "-" + st["marker"], color=st["color"], label=f"{lbl} EDP")
        axB.plot(xs, ed2, "--" + st["marker"], color=st["color"], alpha=0.55,
                 markerfacecolor="none", label=f"{lbl} ED$^2$P")
    axS.set_xlim(dmin, dmax); axS.set_ylim(emin, emax)
    axS.set_xlabel("Delay per image (ms/image)")
    axS.set_ylabel("Energy per image (J/image)")
    _sub = "higher cap dominates => up-right = worse" if measured_x else \
           "iso-EDP dotted; cap down => down-right"
    axS.set_title(f"(a) Energy-delay sweep  ({_sub})")
    axS.legend(loc="upper left")
    axB.set_yscale("log")
    _xl = "Measured power (W/GPU)" if measured_x else "Power cap (W/GPU)"
    axB.set_xlabel(_xl); axB.set_ylabel("EDP / ED$^2$P per image")
    axB.set_title(f"(b) EDP & ED$^2$P vs {'measured power' if measured_x else 'cap'}")
    axB.legend(loc="upper left", fontsize=8, ncol=2)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)
    return fig


# ---- thin CLI wrapper (functions above are the canonical .md, unchanged) ----
if __name__ == "__main__":
    import argparse
    import os
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], metavar="LABEL=CSV",
                    help="repeatable; a labelled power_sweep_summary.csv")
    ap.add_argument("--out-dir", default="power_analysis/results/figures")
    ap.add_argument("--title", default="ResNet-50 / ImageNet-1K power-cap sweep")
    ap.add_argument("--tag", default="powercap")
    args = ap.parse_args()
    runs = [(s.split("=", 1)[0], s.split("=", 1)[1]) for s in args.run if "=" in s]
    if not runs:
        raise SystemExit("pass at least one --run LABEL=CSV")
    os.makedirs(args.out_dir, exist_ok=True)
    plot_powercap(runs, out=os.path.join(args.out_dir, f"{args.tag}_sweep.png"),
                  title=args.title)
    plot_edp(runs, out=os.path.join(args.out_dir, f"{args.tag}_edp.png"),
             title=args.title)
