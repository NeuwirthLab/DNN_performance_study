#!/usr/bin/env python3
# ============================================================================
# analyze_multinode_power.py  --  aggregate per-node power CSVs into one
#                                 run_metrics.csv for the multi-node energy model.
# ----------------------------------------------------------------------------
# Called by measure_power_multinode.sh after the srun-sidecar samplers finish.
# Layout expected under --run-dir:
#
#     run_dir/
#       markers.csv              # idle_start/idle_end/train_start/train_end events
#       command.log              # stdout+stderr of the training command
#       node_<hostname>/
#         power_samples.csv      # per-node amd-smi CSV (timestamp_s,gpu,power,...)
#         [power_summary.csv]    # written HERE by this script for that node
#         [run_metrics.csv]      # written HERE
#       run_metrics.csv          # AGGREGATE (written by this script)
#       power_summary.csv        # AGGREGATE (written by this script)
#
# Strategy: run analyze_power.py as a subprocess for EACH node (it handles
# all the amd-smi CSV parsing / marker windowing), then sum the per-node
# run_metrics.csv fields into a combined aggregate, and write it back to
# run_dir/run_metrics.csv  (same schema, plus n_nodes / gpus_per_node).
#
# Runs under /usr/bin/python3.11 (pure stdlib).
# ============================================================================
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


def _f(s, default=None):
    try:
        v = float(s)
        return v if v == v else default
    except (TypeError, ValueError):
        return default


def find_node_dirs(run_dir: Path) -> list[Path]:
    return sorted(d for d in run_dir.iterdir()
                  if d.is_dir() and d.name.startswith("node_")
                  and (d / "power_samples.csv").exists())


def prepare_node_dir(node_dir: Path, run_dir: Path) -> None:
    """Symlink shared files into the node dir so analyze_power.py can find them."""
    for fname in ("markers.csv", "command.log"):
        src = run_dir / fname
        dst = node_dir / fname
        if src.exists() and not dst.exists():
            try:
                dst.symlink_to(src)
            except OSError:
                pass  # already there or can't symlink (NFS) — fine


def run_analyze_power(node_dir: Path, warmup: float,
                      throughput: float | None, warmup_epochs: int | None,
                      analyzer_py: Path) -> dict | None:
    """Run analyze_power.py for one node's sub-dir and return its run_metrics dict."""
    cmd = [sys.executable, str(analyzer_py),
           "--run-dir", str(node_dir),
           "--warmup", str(warmup),
           "--no-plot"]
    if throughput is not None:
        cmd += ["--throughput", str(throughput)]
    if warmup_epochs is not None:
        cmd += ["--warmup-epochs", str(warmup_epochs)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(f"  [{node_dir.name}] analyze_power exit={proc.returncode}", flush=True)
    if proc.stdout:
        for line in proc.stdout.splitlines():
            print(f"  [{node_dir.name}]   {line}", flush=True)
    if proc.stderr and proc.returncode != 0:
        for line in proc.stderr.splitlines()[-20:]:
            print(f"  [{node_dir.name}] ERR {line}", flush=True)

    rm = node_dir / "run_metrics.csv"
    if not rm.exists():
        print(f"  [{node_dir.name}] WARNING: no run_metrics.csv produced", flush=True)
        return None

    with open(rm, newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        return None
    return dict(zip(rows[0], rows[1]))


def aggregate_node_metrics(per_node: list[dict]) -> dict:
    """Sum energy / power terms across all nodes."""

    def s(key):
        vals = [_f(d.get(key)) for d in per_node]
        valids = [v for v in vals if v is not None]
        return sum(valids) if valids else None

    def avg(key):
        vals = [_f(d.get(key)) for d in per_node]
        valids = [v for v in vals if v is not None]
        return (sum(valids) / len(valids)) if valids else None

    def first(key):
        for d in per_node:
            v = d.get(key)
            if v not in (None, ""):
                return v
        return None

    gpus_active = s("gpus_active")
    n_nodes = len(per_node)

    return dict(
        n_nodes=n_nodes,
        gpus_per_node=(int(gpus_active) // n_nodes) if gpus_active else None,
        gpus_active=gpus_active,
        gpus_sampled=s("gpus_sampled"),
        warmup_s=first("warmup_s"),
        warmup_epochs=first("warmup_epochs"),
        n_measured_epochs=first("n_measured_epochs"),
        train_s=avg("train_s"),
        throughput_img_s=first("throughput_img_s"),
        # Power: SUM across nodes (total socket power)
        idle_W=s("idle_W"),
        steady_W=s("steady_W"),
        peak_W=s("peak_W"),
        # Energy: SUM across nodes
        E_train_J=s("E_train_J"),
        E_dyn_J=s("E_dyn_J"),
        E_hw_accum_J=s("E_hw_accum_J"),
        # Derived efficiency (recompute from aggregate)
        j_per_image=None,
        img_s_per_W=None,
        EDP_Js=None,
        ED2P_Js2=None,
    )


def recompute_efficiency(agg: dict) -> dict:
    """Fill j_per_image, img_s_per_W from aggregate totals."""
    e = _f(agg.get("E_train_J"))
    t = _f(agg.get("throughput_img_s"))
    p = _f(agg.get("steady_W"))
    dur = _f(agg.get("train_s"))

    if e and t and t > 0:
        n_img_est = t * (dur or 0)
        if n_img_est > 0:
            agg["j_per_image"] = e / n_img_est
    if t and p and p > 0:
        agg["img_s_per_W"] = t / p
    e_dyn = _f(agg.get("E_dyn_J"))
    if e_dyn and dur and dur > 0:
        agg["EDP_Js"] = e_dyn * dur
        agg["ED2P_Js2"] = e_dyn * dur * dur
    return agg


def write_aggregate(run_dir: Path, agg: dict) -> None:
    rm = run_dir / "run_metrics.csv"
    keys = list(agg.keys())
    with open(rm, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        w.writerow(["" if v is None else (f"{v:.6g}" if isinstance(v, float) else v)
                    for v in agg.values()])
    print(f"\n  aggregate run_metrics: {rm}", flush=True)


def print_summary(agg: dict, per_node: list[dict]) -> None:
    def f(v, nd=1):
        return f"{v:.{nd}f}" if isinstance(v, float) and v is not None else str(v or "n/a")

    print("\n=== multi-node power summary ===")
    print(f"  nodes          : {agg['n_nodes']}")
    print(f"  gpus_active    : {agg['gpus_active']}")
    print(f"  train_s        : {f(agg.get('train_s'))} s (mean across nodes)")
    print(f"  idle_W  (sum)  : {f(agg.get('idle_W'))} W")
    print(f"  steady_W (sum) : {f(agg.get('steady_W'))} W")
    print(f"  E_train_J(sum) : {f(agg.get('E_train_J'), 0)} J")
    print(f"  E_dyn_J  (sum) : {f(agg.get('E_dyn_J'), 0)} J")
    if agg.get("j_per_image"):
        print(f"  J/image        : {f(agg['j_per_image'], 3)}")
    print("\n  per-node breakdown:")
    for d in per_node:
        nd = d.get("_node", "?")
        print(f"    {nd:20s}  idle={f(_f(d.get('idle_W')))}W  "
              f"steady={f(_f(d.get('steady_W')))}W  "
              f"E_dyn={f(_f(d.get('E_dyn_J')), 0)}J")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Aggregate per-node power CSVs into one run_metrics.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="top-level measure_power_multinode.sh output dir")
    ap.add_argument("--warmup", type=float, default=45.0,
                    help="steady-state warmup trim seconds (passed to analyze_power.py)")
    ap.add_argument("--warmup-epochs", type=int, default=None)
    ap.add_argument("--throughput", type=float, default=None,
                    help="img/s hint for J/image metrics")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        sys.exit(f"run-dir not found: {run_dir}")

    # Locate analyze_power.py (sibling of this script)
    analyzer_py = Path(__file__).parent / "analyze_power.py"
    if not analyzer_py.exists():
        sys.exit(f"analyze_power.py not found at {analyzer_py}")

    node_dirs = find_node_dirs(run_dir)
    if not node_dirs:
        sys.exit(f"no node_* dirs with power_samples.csv found under {run_dir}")

    print(f"=== analyze_multinode_power: {run_dir} ===")
    print(f"  found {len(node_dirs)} node dir(s): {[d.name for d in node_dirs]}")

    per_node_metrics = []
    for nd in node_dirs:
        prepare_node_dir(nd, run_dir)
        m = run_analyze_power(nd, args.warmup, args.throughput,
                              args.warmup_epochs, analyzer_py)
        if m is not None:
            m["_node"] = nd.name
            per_node_metrics.append(m)

    if not per_node_metrics:
        sys.exit("no per-node metrics extracted — check node_*/run_metrics.csv")

    agg = aggregate_node_metrics(per_node_metrics)
    agg = recompute_efficiency(agg)
    print_summary(agg, per_node_metrics)
    write_aggregate(run_dir, agg)


if __name__ == "__main__":
    main()
