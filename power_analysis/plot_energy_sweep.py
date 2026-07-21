#!/usr/bin/env python3
"""
Energy analysis plots for the MI300X ResNet-50 native-HIP vs PyTorch sweep.

Reads the per-run artifacts written by measure_power.sh / analyze_power.py:
    <run>/power_samples.csv   raw per-(sample,GPU) socket power
    <run>/power_summary.csv   per-GPU idle/steady (used to pick ACTIVE GPUs)
    <run>/run_metrics.csv     one-row aggregate (throughput, energy, J/image, EDP)
    <run>/markers.csv         idle/train/cooldown timestamps

Two products:

  (A) Two-panel energy figure  (--compare), in the style of the H100 paper fig:
      (a) Total board power vs elapsed time  — sum over the ACTIVE GPUs only
          (idle neighbours on a whole-node alloc are excluded), steady mean dashed.
      (b) Cumulative energy vs images processed — trapezoid integral of (a) over
          the steady-state window; the SLOPE of each line is J/image.
      Overlays native (blue) vs PyTorch (orange), 4 GPU (solid) vs 8 GPU (dashed).

  (B) Scaling curves  (--sweep-dir): EDP, energy-to-solution, img/s/W and
      aggregate power vs GPU count for weak & strong, + a combined sweep_summary.csv.

Usage:
  # per-stack aggregate + scaling curves (the driver calls this):
  python plot_energy_sweep.py --sweep-dir results/native/sweep_fp32_<jid>

  # cross-stack two-panel comparison at 4 & 8 GPU (run after both sweeps):
  python plot_energy_sweep.py --compare \
      --native-sweep  results/native/sweep_fp32_<jidN> \
      --pytorch-sweep results/pytorch/sweep_fp32_<jidP> \
      --mode strong --gpus 4 8
"""
from __future__ import annotations
import argparse, csv, os, re, sys
from pathlib import Path

NATIVE_C, PYTORCH_C = "#D55E00", "#0072B2"   # canonical (styling.md): native=vermillion, pytorch=blue
POWER_RE = re.compile(r"(socket_power|power\.draw|graphics.?package.?power|^power$)", re.I)
GPU_RE   = re.compile(r"(^gpu$|^index$|^device$|^card)", re.I)
NUM_RE   = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _to_float(s):
    if s is None:
        return None
    m = NUM_RE.search(str(s))
    return float(m.group()) if m else None


def _read_csv_rows(path):
    with open(path, newline="") as f:
        return [r for r in csv.reader(f) if r]


def active_gpu_ids(run, thresh=25.0):
    """GPU ids whose steady power rose >= thresh W above idle (the run's GPUs)."""
    p = Path(run) / "power_summary.csv"
    if not p.is_file():
        return None
    ids = set()
    for row in csv.DictReader(open(p, newline="")):
        idle, steady = _to_float(row.get("idle_W")), _to_float(row.get("steady_W"))
        if idle is not None and steady is not None and steady - idle >= thresh:
            ids.add(str(row["gpu"]).strip())
    return ids


def markers(run):
    ev = {}
    p = Path(run) / "markers.csv"
    if p.is_file():
        for row in _read_csv_rows(p):
            if len(row) >= 2 and row[0] != "event":
                t = _to_float(row[1])
                if t is not None:
                    ev[row[0]] = t
    return ev


def run_metrics(run):
    p = Path(run) / "run_metrics.csv"
    if not p.is_file():
        return {}
    rows = _read_csv_rows(p)
    if len(rows) < 2:
        return {}
    return dict(zip(rows[0], rows[1]))


def power_trace(run, active):
    """(elapsed_from_train_start_s, total_active_power_W) summing only active GPUs."""
    import numpy as np
    p = Path(run) / "power_samples.csv"
    rows = _read_csv_rows(p)
    if len(rows) < 2:
        return np.array([]), np.array([])
    header = [h.strip() for h in rows[0]]
    idx = {h: i for i, h in enumerate(header)}
    ts_i = idx.get("timestamp_s", 0)
    pw_c = next((h for h in header if POWER_RE.search(h)), None)
    gp_c = next((h for h in header if GPU_RE.search(h)), None)
    if pw_c is None:
        return np.array([]), np.array([])
    pw_i, gp_i = idx[pw_c], (idx[gp_c] if gp_c else None)
    by_ts = {}
    for r in rows[1:]:
        if len(r) <= pw_i:
            continue
        if active and gp_i is not None and gp_i < len(r) and str(r[gp_i]).strip() not in active:
            continue
        t, w = _to_float(r[ts_i]), _to_float(r[pw_i])
        if t is None or w is None:
            continue
        by_ts[t] = by_ts.get(t, 0.0) + w
    if len(by_ts) < 2:
        return np.array([]), np.array([])
    t = np.array(sorted(by_ts))
    p_arr = np.array([by_ts[x] for x in t])
    tr0 = markers(run).get("train_start", t[0])
    return t - tr0, p_arr


def cumulative_energy(elapsed, power, warmup_s):
    """Trapezoid-cumulate power over the steady window (elapsed>=warmup), kJ."""
    import numpy as np
    mask = elapsed >= warmup_s
    t, p = elapsed[mask] - warmup_s, power[mask]
    if t.size < 2:
        return t, np.zeros_like(t)
    e = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(t))])
    return t, e / 1000.0


def _style():
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import paper_style as ps          # canonical paper style (styling.md)
        ps.setup()
    except Exception:
        import matplotlib
        matplotlib.rcParams.update({"font.size": 11, "axes.grid": True,
                                    "grid.alpha": 0.3, "figure.dpi": 110})


# ── (B) aggregate one sweep dir -> combined CSV + scaling curves ─────────────
def collect_points(sweep_dir):
    pts = []
    for sub in sorted(Path(sweep_dir).glob("*_gpu*")):
        m = re.match(r"(weak|strong)_gpu(\d+)", sub.name)
        if not m or not (sub / "run_metrics.csv").is_file():
            continue
        rec = run_metrics(sub)
        rec["mode"], rec["gpus"], rec["run_dir"] = m.group(1), int(m.group(2)), str(sub)
        pts.append(rec)
    return pts


def write_summary(sweep_dir, pts):
    out = Path(sweep_dir) / "sweep_summary.csv"
    cols = ["mode", "gpus", "throughput_img_s", "train_s", "steady_W", "peak_W",
            "E_train_J", "E_dyn_J", "j_per_image", "img_s_per_W", "EDP_Js", "ED2P_Js2"]
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in sorted(pts, key=lambda x: (x["mode"], x["gpus"])):
            w.writerow([r.get(c, "") for c in cols])
    print(f"  wrote {out}")
    return out


def scaling_figure(sweep_dir, pts):
    try:
        import numpy as np, matplotlib
        matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (scaling figure skipped: {e})"); return
    _style()
    panels = [("throughput_img_s", "Throughput (img/s)"),
              ("img_s_per_W", "Efficiency (img/s/W)"),
              ("E_train_J", "Energy to solution (J)"),
              ("EDP_Js", "EDP  E·t (J·s)")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, (key, ylab) in zip(axes.flat, panels):
        for mode, ls in (("strong", "-o"), ("weak", "--s")):
            xs, ys = [], []
            for r in sorted([p for p in pts if p["mode"] == mode], key=lambda x: x["gpus"]):
                v = _to_float(r.get(key))
                if v is not None:
                    xs.append(r["gpus"]); ys.append(v)
            if xs:
                ax.plot(xs, ys, ls, label=mode)
        ax.set_xlabel("GPUs"); ax.set_ylabel(ylab); ax.set_xticks([1, 2, 4, 8])
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(f"ResNet-50 / ImageNet-1K scaling — {Path(sweep_dir).name}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    stem = Path(sweep_dir) / "scaling_curves"
    fig.savefig(f"{stem}.png", dpi=300); plt.close(fig)
    print(f"  wrote {stem}.png")


# ── (A) two-panel cross-stack comparison ────────────────────────────────────
def comparison_figure(runs, out_stem, title):
    """runs: list of (label, run_dir, color, linestyle)."""
    try:
        import numpy as np, matplotlib
        matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"comparison figure skipped: {e}"); return
    _style()
    import matplotlib.pyplot as plt
    fig, (axP, axE) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    print("\n=== energy comparison ===")
    for label, run, color, ls in runs:
        active = active_gpu_ids(run)
        elapsed, power = power_trace(run, active)
        if elapsed.size == 0:
            print(f"  SKIP {label}: no power trace in {run}"); continue
        rec = run_metrics(run)
        ips = _to_float(rec.get("throughput_img_s"))
        steady = _to_float(rec.get("steady_W"))
        jpi = _to_float(rec.get("j_per_image"))
        warm = _to_float(rec.get("warmup_s")) or 45.0
        axP.plot(elapsed, power, color=color, linestyle=ls, lw=1.4, alpha=0.9, label=label)
        if steady:
            axP.axhline(steady, color=color, linestyle=ls, lw=0.8, alpha=0.45)
        t, e_kJ = cumulative_energy(elapsed, power, warm)
        if ips and t.size:
            images_M = (t * ips) / 1e6
            axE.plot(images_M, e_kJ, color=color, linestyle=ls, lw=2.0, label=label)
            if jpi:
                dy = 6 if ls == "--" else -12
                axE.annotate(f"{jpi:.3f} J/img", (images_M[-1], e_kJ[-1]),
                             textcoords="offset points", xytext=(-2, dy),
                             fontsize=8, color=color, ha="right", fontweight="bold")
        print(f"  {label}: steady {steady:.0f} W | {ips:.0f} img/s | {jpi:.3f} J/img"
              f" | trace {elapsed[-1]:.0f}s")
    axP.set_xlabel("Elapsed time (s)"); axP.set_ylabel("Total active-GPU power (W)")
    axP.set_title("(a) Power draw over time", loc="left"); axP.set_ylim(bottom=0)
    axP.legend(loc="upper left", fontsize=8)
    axE.set_xlabel("Images processed (millions)"); axE.set_ylabel("Cumulative energy (kJ)")
    axE.set_title("(b) Energy to process images  (slope = J/image)", loc="left")
    axE.legend(loc="upper left", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{out_stem}.png", dpi=600); fig.savefig(f"{out_stem}.pdf")
    plt.close(fig)
    print(f"\nwrote {out_stem}.png + {out_stem}.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", help="aggregate one stack's sweep + scaling curves")
    ap.add_argument("--compare", action="store_true", help="cross-stack two-panel figure")
    ap.add_argument("--native-sweep"); ap.add_argument("--pytorch-sweep")
    ap.add_argument("--mode", default="strong", choices=["strong", "weak"])
    ap.add_argument("--gpus", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    if args.sweep_dir:
        pts = collect_points(args.sweep_dir)
        if not pts:
            print(f"no completed points under {args.sweep_dir}"); return
        write_summary(args.sweep_dir, pts)
        scaling_figure(args.sweep_dir, pts)

    if args.compare:
        runs = []
        for stack, sweep, color in (("Native", args.native_sweep, NATIVE_C),
                                    ("PyTorch", args.pytorch_sweep, PYTORCH_C)):
            if not sweep:
                continue
            for n in args.gpus:
                d = Path(sweep) / f"{args.mode}_gpu{n}"
                ls = "--" if n == max(args.gpus) else "-"
                if (d / "run_metrics.csv").is_file():
                    runs.append((f"{stack} · {n} GPU", str(d), color, ls))
        if not runs:
            print("no runs found for comparison (check sweep dirs / mode / gpus)"); return
        outdir = Path(args.out_dir or "power_analysis/results")
        outdir.mkdir(parents=True, exist_ok=True)
        comparison_figure(
            runs, str(outdir / f"energy_comparison_{args.mode}"),
            f"ResNet-50 v1.5 · ImageNet-1K · MI300X · native HIP vs PyTorch "
            f"({args.mode} scaling)")


if __name__ == "__main__":
    main()
