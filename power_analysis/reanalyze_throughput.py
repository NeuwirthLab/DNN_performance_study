#!/usr/bin/env python3
"""
Re-derive throughput as the AVERAGE per-epoch throughput over the MEASURED epochs
(skip the first WARMUP_EPOCHS) and re-run analyze_power.py for every saved cap
run. Fixes the earlier bug where throughput was the LAST epoch (or NGC's
train.total_ips, which is a copy of the last epoch) -> a single outlier final
epoch skewed J/image and img/s/W. Uses only saved command.log + power_samples,
so NO GPU job is re-run.

    python3 power_analysis/reanalyze_throughput.py
"""
import csv, glob, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYZER = os.path.join(HERE, "analyze_power.py")
ROOT = "power_analysis/results"


def avg_throughput(cmdlog, warmup_epochs):
    txt = open(cmdlog, encoding="utf-8", errors="ignore").read()
    # (a) PyTorch: per-epoch "epoch <n> train.compute_ips : <val>"  (n 0-indexed)
    pt = [(int(e), float(v)) for e, v in
          re.findall(r"epoch (\d+) train\.compute_ips\s*:\s*([\d.]+)", txt)]
    if pt:
        vals = [v for e, v in pt if e >= warmup_epochs]
        return (sum(vals) / len(vals)) if vals else None
    # (b) native summary table: "<ep> <lr 0..1> ... <Time> <Img/sec>" (ep 1-indexed)
    nat = []
    for line in txt.splitlines():
        f = line.split()
        if (len(f) >= 8 and re.fullmatch(r"\d+", f[0])
                and re.fullmatch(r"0\.\d+", f[1]) and re.fullmatch(r"[\d.]+", f[-1])):
            nat.append((int(f[0]), float(f[-1])))
    if nat:
        vals = [v for e, v in nat if e > warmup_epochs]   # row 1 == warmup
        return (sum(vals) / len(vals)) if vals else None
    return None


def main():
    dirs = sorted(glob.glob(f"{ROOT}/*/powercap_*/cap_*"))
    n_ok = n_skip = 0
    for d in dirs:
        rm = os.path.join(d, "run_metrics.csv")
        cl = os.path.join(d, "command.log")
        if not (os.path.isfile(rm) and os.path.isfile(cl)):
            n_skip += 1; continue
        old = list(csv.DictReader(open(rm)))[0]
        W = int(float(old.get("warmup_epochs") or 0))
        ws = old.get("warmup_s") or "45"
        new_t = avg_throughput(cl, W)
        if new_t is None:
            print(f"  SKIP (no per-epoch tput): {d}"); n_skip += 1; continue
        old_t = float(old.get("throughput_img_s") or 0)
        r = subprocess.run([sys.executable, ANALYZER, "--run-dir", d,
                            "--warmup", str(ws), "--warmup-epochs", str(W),
                            "--throughput", f"{new_t:.2f}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FAIL analyze: {d}\n{r.stderr[-300:]}"); n_skip += 1; continue
        nm = list(csv.DictReader(open(rm)))[0]
        print(f"  {d.split('/results/')[1]:46s} tput {old_t:7.0f}->{new_t:7.1f}  "
              f"J/img {float(old.get('j_per_image') or 0):.3f}->{float(nm['j_per_image']):.3f}")
        n_ok += 1
    print(f"\n re-analyzed {n_ok}, skipped {n_skip}")


if __name__ == "__main__":
    main()
