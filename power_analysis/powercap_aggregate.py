#!/usr/bin/env python3
"""
Aggregate ALL power-cap points (across jobs / nodes / caps) per stack into one
combined power_sweep_summary, then redraw the plot_powercap figures.

Re-run this any time new cap runs finish — it rescans every
results/<stack>/powercap_*/cap_*/ dir, dedups by cap (latest job wins), and
regenerates results/<stack>/power_sweep_all.csv + the sweep/EDP figures. This is
the "keep appending the latest results" entry point.

    python3 powercap_aggregate.py
"""
import csv, glob, os, re, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_powercap as pc   # reuse the H100-style plotters

ROOT = "power_analysis/results"
COLS = ["label", "cap_w", "n_gpus", "avg_power_w_per_gpu", "mean_sm_clock_mhz",
        "img_per_sec", "j_per_image", "img_per_sec_per_watt", "edp", "ed2p",
        "dynamic_j_per_image"]


def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _metrics(run):
    p = Path(run) / "run_metrics.csv"
    if not p.is_file():
        return None
    rows = [r for r in csv.reader(open(p)) if r]
    return dict(zip(rows[0], rows[1])) if len(rows) >= 2 else None


def _mean_active_sclk(run, thresh=25.0):
    p = Path(run) / "power_summary.csv"
    if not p.is_file():
        return None
    v = []
    for r in csv.DictReader(open(p)):
        idle, steady, sclk = _f(r.get("idle_W")), _f(r.get("steady_W")), _f(r.get("sclk_MHz"))
        if idle is not None and steady is not None and sclk and steady - idle >= thresh:
            v.append(sclk)
    return sum(v) / len(v) if v else None


def build(stack, label):
    best = {}   # cap -> (jobid, row); latest jobid wins
    for d in sorted(glob.glob(f"{ROOT}/{stack}/powercap_*/cap_*")):
        m = re.search(r"powercap_(\d+)/cap_(\d+)$", d)
        if not m:
            continue
        jid, cap = int(m.group(1)), int(m.group(2))
        rec = _metrics(d)
        if not rec or not _f(rec.get("throughput_img_s")):
            continue            # failed/incomplete run
        if cap in best and best[cap][0] >= jid:
            continue            # keep the newer job's point
        n = _f(rec.get("gpus_active")) or 1
        steady = _f(rec.get("steady_W")); tput = _f(rec.get("throughput_img_s"))
        ts = _f(rec.get("train_s")); edyn = _f(rec.get("E_dyn_J"))
        dyn = (edyn / (tput * ts)) if (edyn and tput and ts) else None
        best[cap] = (jid, {
            "label": label, "cap_w": cap, "n_gpus": int(n),
            "avg_power_w_per_gpu": f"{steady / n:.2f}" if steady else "",
            "mean_sm_clock_mhz": f"{(_mean_active_sclk(d) or 0):.0f}",
            "img_per_sec": f"{tput:.2f}", "j_per_image": rec.get("j_per_image", ""),
            "img_per_sec_per_watt": rec.get("img_s_per_W", ""),
            "edp": rec.get("EDP_Js", ""), "ed2p": rec.get("ED2P_Js2", ""),
            "dynamic_j_per_image": f"{dyn:.4f}" if dyn else ""})
    rows = [best[c][1] for c in sorted(best)]
    out = f"{ROOT}/{stack}/power_sweep_all.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    print(f"  {stack:8s}: {len(rows)} cap point(s) {sorted(best)} -> {out}")
    return out, len(rows)


def main():
    print("=== aggregating power-cap points ===")
    runs = []
    for stack, label in (("native", "Native MI300X"), ("pytorch", "PyTorch MI300X")):
        out, n = build(stack, label)
        if n >= 2:
            runs.append((label, out))
    if not runs:
        print("\nNeed >=2 cap points in at least one stack to plot. "
              "Add more caps and re-run.")
        return
    fig = f"{ROOT}/figures"; os.makedirs(fig, exist_ok=True)
    print("\n=== plotting ===")
    pc.plot_powercap(runs, out=os.path.join(fig, "powercap_sweep.png"),
                     title="ResNet-50 / ImageNet-1K / MI300X — power-cap sweep (epoch-based)")
    pc.plot_edp(runs, out=os.path.join(fig, "powercap_edp.png"),
                title="ResNet-50 / ImageNet-1K / MI300X — power-cap EDP (nominal cap)")
    # honest variant: panel (b) keyed on MEASURED per-GPU power, because the
    # MI300X clamps low nominal caps to a hardware floor (~350 W) -> the nominal
    # cap is not the true independent variable below ~400 W.
    pc.plot_edp(runs, out=os.path.join(fig, "powercap_edp_measured.png"),
                title="ResNet-50 / ImageNet-1K / MI300X — EDP vs MEASURED power",
                measured_x=True)
    # merged panel (a)+(b): J/image (left) and img/s/W (right) on one dual axis,
    # with a vector PDF companion (paper-ready, highest quality).
    pc.plot_combined(runs, out=os.path.join(fig, "powercap_combined.png"),
                     title="ResNet-50 / ImageNet-1K / MI300X — power-cap energy vs efficiency")
    # remaining two panels as standalone paper-ready figures (+ vector PDF)
    pc.plot_single(runs, "throughput", out=os.path.join(fig, "powercap_throughput.png"),
                   title="ResNet-50 / ImageNet-1K / MI300X — throughput vs power cap")
    pc.plot_single(runs, "edpknee", out=os.path.join(fig, "powercap_edpknee.png"),
                   title="ResNet-50 / ImageNet-1K / MI300X — EDP knee vs power cap")
    # all three panels in one IEEE double-column (full-width) row (+ vector PDF)
    pc.plot_row(runs, out=os.path.join(fig, "powercap_row.png"))


if __name__ == "__main__":
    main()
