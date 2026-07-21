#!/usr/bin/env python3
"""
Statistical aggregation of the energy sweep over REPEATED runs.

Each sweep point is run N times (sweep_energy.slurm REPEATS=5), producing
    <sweep>/<mode>_gpu<N>_r<k>/run_metrics.csv
This script groups the repeats by (mode, gpus), computes mean +/- sample std
for every metric, writes stats_summary.csv, and draws error-bar curves.

  --sweep-dir DIR            one stack: stats_summary.csv + per-stack error-bar fig
  --compare --native-sweep A --pytorch-sweep B
                            cross-stack error-bar figure (native vs PyTorch),
                            EDP / img-per-W / energy / throughput vs GPU count.

Robust to single runs too (a point with one repeat just gets std=0).
"""
from __future__ import annotations
import argparse, csv, math, re, sys
from pathlib import Path
from collections import defaultdict

# Canonical paper styling (styling.md): native = vermillion, pytorch = blue.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    import paper_style as ps
    NATIVE_C, PYTORCH_C = ps.NATIVE, ps.PYTORCH
except Exception:
    ps = None
    NATIVE_C, PYTORCH_C = "#D55E00", "#0072B2"
PT_RE = re.compile(r"(weak|strong)_gpu(\d+)(?:_r(\d+))?$")
METRICS = ["throughput_img_s", "train_s", "steady_W", "peak_W", "E_train_J",
           "E_dyn_J", "j_per_image", "img_s_per_W", "EDP_Js", "ED2P_Js2"]


def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def read_metrics(run):
    p = Path(run) / "run_metrics.csv"
    if not p.is_file():
        return None
    rows = [r for r in csv.reader(open(p)) if r]
    if len(rows) < 2:
        return None
    return dict(zip(rows[0], rows[1]))


def collect(sweep_dir):
    """{(mode,gpus): [metric-dict, ...]} over all repeats."""
    g = defaultdict(list)
    for sub in sorted(Path(sweep_dir).glob("*_gpu*")):
        m = PT_RE.match(sub.name)
        if not m:
            continue
        rec = read_metrics(sub)
        # only count a repeat that actually produced throughput (i.e. didn't crash)
        if rec and _f(rec.get("throughput_img_s")):
            g[(m.group(1), int(m.group(2)))].append(rec)
    return g


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    n = len(vals)
    mu = sum(vals) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return mu, sd, n


def stats_table(group):
    """{(mode,gpus): {metric:(mean,std,n)}} plus n_repeats."""
    out = {}
    for key, recs in group.items():
        d = {"n_repeats": len(recs)}
        for mt in METRICS:
            d[mt] = mean_std([_f(r.get(mt)) for r in recs])
        out[key] = d
    return out


def write_stats_csv(sweep_dir, st):
    p = Path(sweep_dir) / "stats_summary.csv"
    cols = ["mode", "gpus", "n_repeats"]
    for mt in METRICS:
        cols += [f"{mt}_mean", f"{mt}_std"]
    with open(p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for (mode, n) in sorted(st):
            d = st[(mode, n)]
            row = [mode, n, d["n_repeats"]]
            for mt in METRICS:
                mu, sd, _ = d[mt]
                row += ["" if mu is None else f"{mu:.6g}",
                        "" if sd is None else f"{sd:.4g}"]
            w.writerow(row)
    print(f"  wrote {p}")
    return p


def _series(st, mode, metric):
    xs, ys, es = [], [], []
    for (m, n) in sorted(st, key=lambda k: k[1]):
        if m != mode:
            continue
        mu, sd, k = st[(m, n)][metric]
        if mu is not None:
            xs.append(n); ys.append(mu); es.append(sd or 0.0)
    return xs, ys, es


def errorbar_figure(out_stem, title, stacks):
    """stacks: list of (label, stats_table, color). Styled via paper_style toolkit:
    colour = implementation, line/marker = scaling mode (strong solid-diamond,
    weak dashed-circle), one dual-encoded legend. See styling.md."""
    try:
        import matplotlib
        matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (figure skipped: {e})"); return
    if ps is not None:
        ps.setup(); ps.scaling_fonts()
    panels = [("throughput_img_s", "Throughput (images/s)", False),
              ("img_s_per_W", "Throughput per watt (img/s/W)", False),
              ("E_train_J", "Energy to solution (J)", False),
              ("EDP_Js", "EDP  E·t (J·s)", True)]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.6))
    for ax, (mt, ylab, logy) in zip(axes.flat, panels):
        for label, st, color in stacks:
            for mode in ("strong", "weak"):
                xs, ys, es = _series(st, mode, mt)
                if not xs:
                    continue
                if ps is not None:
                    ps.draw_series(ax, xs, ys, color, ps.MODE_STYLES[mode], yerr=es)
                else:
                    ax.errorbar(xs, ys, yerr=es, color=color, capsize=3, lw=1.6, ms=5,
                                ls="-" if mode == "strong" else "--",
                                marker="D" if mode == "strong" else "o")
        ax.set_xlabel(ps.LABELS.GPUS if ps else "Number of GPUs"); ax.set_ylabel(ylab)
        ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels([1, 2, 4, 8])
        if logy:
            ax.set_yscale("log")
        if ps is not None:
            ps.bold_ticks(ax); ps.cleanup(ax)
        else:
            ax.grid(alpha=0.3, which="both")
    # in-panel legend, top-left of the throughput panel (its top-left is empty —
    # throughput rises bottom-left -> top-right — so it never overlaps the data).
    impls = [(c, l) for (l, _st, c) in stacks if l]
    if ps is not None and impls:
        ps.dual_legend(axes.flat[0], impls, modes=True, ideal=False,
                       loc="upper left", ncol=1)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if ps is not None:
        ps.save(fig, Path(out_stem).name, out_dir=str(Path(out_stem).parent))
    else:
        fig.savefig(f"{out_stem}.png", dpi=300); fig.savefig(f"{out_stem}.pdf")
    plt.close(fig)
    print(f"  wrote {out_stem}.png + .pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--native-sweep"); ap.add_argument("--pytorch-sweep")
    ap.add_argument("--out-dir", default="power_analysis/results/figures")
    args = ap.parse_args()

    if args.sweep_dir:
        st = stats_table(collect(args.sweep_dir))
        if not st:
            print(f"no completed repeats under {args.sweep_dir}"); return
        write_stats_csv(args.sweep_dir, st)
        print("  points (mode, gpus -> n_repeats):")
        for k in sorted(st):
            print(f"    {k[0]:6s} {k[1]} GPU : n={st[k]['n_repeats']}")
        sd = args.sweep_dir
        lbl, col = (("Native", NATIVE_C) if "/native/" in sd
                    else ("PyTorch", PYTORCH_C) if "/pytorch/" in sd
                    else ("Run", NATIVE_C))
        errorbar_figure(str(Path(args.sweep_dir) / "stats_curves"),
                        f"ResNet-50/ImageNet-1K — {lbl} (mean ± std)",
                        [(lbl, st, col)])

    if args.compare:
        stacks = []
        if args.native_sweep:
            stacks.append(("Native", stats_table(collect(args.native_sweep)), NATIVE_C))
        if args.pytorch_sweep:
            stacks.append(("PyTorch", stats_table(collect(args.pytorch_sweep)), PYTORCH_C))
        if not any(s[1] for s in stacks):
            print("no data for comparison"); return
        out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
        errorbar_figure(str(out / "energy_stats_comparison"),
                        "ResNet-50/ImageNet-1K/MI300X — native vs PyTorch "
                        "(mean ± std over repeats)", stacks)


if __name__ == "__main__":
    main()
