#!/usr/bin/env python3
"""
Adapter: turn this repo's per-cap run dirs into a power_sweep_summary.csv
(the schema plot_powercap.py expects).

A powercap sweep writes one measure_power.sh run dir per cap, named cap_<W>/
(each with run_metrics.csv + power_summary.csv). This reads them and emits:
  label,cap_w,n_gpus,avg_power_w_per_gpu,mean_sm_clock_mhz,img_per_sec,
  j_per_image,img_per_sec_per_watt,edp,ed2p,dynamic_j_per_image

Usage:
  python build_powercap_summary.py --sweep-dir <dir-with-cap_*> --label "MI300X HIP"
"""
import argparse, csv, os, re
from pathlib import Path

COLS = ["label", "cap_w", "n_gpus", "avg_power_w_per_gpu", "mean_sm_clock_mhz",
        "img_per_sec", "j_per_image", "img_per_sec_per_watt", "edp", "ed2p",
        "dynamic_j_per_image"]


def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def metrics(run):
    p = Path(run) / "run_metrics.csv"
    if not p.is_file():
        return None
    rows = [r for r in csv.reader(open(p)) if r]
    return dict(zip(rows[0], rows[1])) if len(rows) >= 2 else None


def mean_active_sclk(run, thresh=25.0):
    """Mean gfx clock over the GPUs that actually did work this run."""
    p = Path(run) / "power_summary.csv"
    if not p.is_file():
        return None
    vals = []
    for r in csv.DictReader(open(p)):
        idle, steady, sclk = _f(r.get("idle_W")), _f(r.get("steady_W")), _f(r.get("sclk_MHz"))
        if idle is not None and steady is not None and sclk and steady - idle >= thresh:
            vals.append(sclk)
    return sum(vals) / len(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.sweep_dir, "power_sweep_summary.csv")

    rows = []
    for sub in sorted(Path(args.sweep_dir).glob("cap_*")):
        m = re.match(r"cap_(\d+)", sub.name)
        if not m:
            continue
        rec = metrics(sub)
        if not rec or not _f(rec.get("throughput_img_s")):
            print(f"  skip {sub.name}: no throughput (run failed)")
            continue
        cap = int(m.group(1))
        n = _f(rec.get("gpus_active")) or 1
        steady = _f(rec.get("steady_W"))
        tput = _f(rec.get("throughput_img_s"))
        train_s = _f(rec.get("train_s"))
        e_dyn = _f(rec.get("E_dyn_J"))
        dyn_jpi = (e_dyn / (tput * train_s)) if (e_dyn and tput and train_s) else None
        rows.append({
            "label": args.label, "cap_w": cap, "n_gpus": int(n),
            "avg_power_w_per_gpu": f"{steady / n:.2f}" if steady else "",
            "mean_sm_clock_mhz": f"{(mean_active_sclk(sub) or 0):.0f}",
            "img_per_sec": f"{tput:.2f}" if tput else "",
            "j_per_image": rec.get("j_per_image", ""),
            "img_per_sec_per_watt": rec.get("img_s_per_W", ""),
            "edp": rec.get("EDP_Js", ""), "ed2p": rec.get("ED2P_Js2", ""),
            "dynamic_j_per_image": f"{dyn_jpi:.4f}" if dyn_jpi else "",
        })
    rows.sort(key=lambda r: r["cap_w"])
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} caps)")


if __name__ == "__main__":
    main()
