#!/usr/bin/env python3
"""GPU-idle / dispatch-bound figure — HIP custom trainer vs PyTorch-ROCm.

Three panels, both implementations shown side-by-side on every panel:

  (a) Device occupancy   — stacked busy/idle bar for each runtime
  (b) Idle-gap spectrum  — grouped bars (HIP | PyTorch) per gap-size bucket
  (c) What the GPU waits — top-N idle contributors, paired bars HIP vs PyTorch

Updated to full ImageNet-1K (1.28M images, job 52393128, batch 256, FP32).
House style: paper_style.py.

Usage:
    python3 plot_gpu_idle.py
    python3 plot_gpu_idle.py \\
        --trace-hip  profiles/.../hip_kernel_trace.csv \\
        --trace-pt   profiles/.../pt_kernel_trace.csv  \\
        --out plots/mi300x
"""
import argparse
import bisect
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    import paper_style as ps
except ImportError:
    sys.exit("paper_style.py must sit next to this script (or be on PYTHONPATH).")

# Full ImageNet-1K traces (job 52393128, 1 GPU, batch 256, 224², FP32)
DEFAULT_HIP = ("profiles/hyp_in1k_52393128/N0_baseline_fm3_g1/"
               "trace/gaod0002/678789_kernel_trace.csv")
DEFAULT_PT  = ("profiles/hyp_in1k_52393128/PT_g1/"
               "trace/gaod0002/806896_kernel_trace.csv")

BUCKETS = [1e3, 5e3, 1e4, 5e4, 1e5, 1e6]
BLABELS = ["<1µs", "1–5µs", "5–10µs", "10–50µs", "50–100µs", "0.1–1ms", ">1ms"]
TOP_N = 7   # top idle contributors shown in panel (c)


def pretty(name: str):
    n = name.split("(")[0]
    table = {
        "__amd_rocclr_copyBuffer":            ("D2D copy", True),
        "__amd_rocclr_fillBufferAligned":     ("memset", True),
        "__amd_rocclr_fillBuffer":            ("memset", True),
        "MIOpenBatchNormBwdSpatial":          ("BatchNorm bwd", False),
        "MIOpenBatchNormFwdTrainSpatialNorm": ("BatchNorm fwd", False),
        "MIOpenBatchNormFwdTrainSpatial":     ("BatchNorm fwd", False),
        "MIOpenBatchNormFwdTrainSpatialMeanVariance": ("BatchNorm fwd", False),
        "sgd_kernel":                         ("SGD update", False),
        "relu_kernel":                        ("ReLU fwd", False),
        "relu_bwd_kernel":                    ("ReLU bwd", False),
        "add_inplace_kernel":                 ("residual add", False),
        "gavgpool_kernel":                    ("global avgpool", False),
    }
    if n in table:
        return table[n]
    if n.startswith("Cijk_") or "_MI16x16" in n or n.startswith("Cgemm"):
        return ("GEMM (rocBLAS/Tensile)", False)
    if "miopenSp3AsmConv" in n or "Winograd" in n or "ImplicitGemm" in n:
        return ("Conv (MIOpen)", False)
    if n.startswith("ncclDevKernel") or n.startswith("rccl"):
        return ("RCCL AllReduce", False)
    low = n.lower()
    pats = [
        ("copybuffer",   "D2D copy",          True),
        ("fillbuffer",   "memset",             True),
        ("memcpy",       "memcpy",             True),
        ("memset",       "memset",             True),
        ("nccl",         "RCCL AllReduce",     False),
        ("rccl",         "RCCL AllReduce",     False),
        ("batchnorm",    "BatchNorm",          False),
        ("batch_norm",   "BatchNorm",          False),
        ("winograd",     "Conv (MIOpen)",      False),
        ("implicitgemm", "Conv (MIOpen)",      False),
        ("conv",         "Conv",               False),
        ("gemm",         "GEMM",               False),
        ("transpose",    "transpose",          False),
        ("elementwise",  "elementwise",        False),
        ("reduce",       "reduce",             False),
        ("softmax",      "softmax",            False),
        ("relu",         "ReLU",               False),
        ("pool",         "pooling",            False),
        ("sgd",          "SGD update",         False),
        ("adam",         "optimizer",          False),
    ]
    for key, lbl, mem in pats:
        if key in low:
            return (lbl, mem)
    base = n.replace("void ", "").split("<")[0].split("::")[-1]
    return (base[:22] if base else n[:22], False)


def load_agent(trace, agent=-1):
    def aid(v):
        return int("".join(ch for ch in v if ch.isdigit()))
    rows = []
    with open(trace, newline="") as f:
        r = csv.reader(f)
        h = next(r)
        ix = {nm: i for i, nm in enumerate(h)}
        ai = ix["Agent_Id"]; si = ix["Start_Timestamp"]
        ei = ix["End_Timestamp"]; ni = ix["Kernel_Name"]
        for row in r:
            rows.append((aid(row[ai]), int(row[si]), int(row[ei]), row[ni]))
    if agent < 0:
        cnt = defaultdict(int)
        for a, *_ in rows: cnt[a] += 1
        agent = max(cnt, key=cnt.get)
        print(f"    auto-selected agent {agent} ({cnt[agent]:,} kernels)")
    ev = [(s, e, nm) for a, s, e, nm in rows if a == agent]
    ev.sort()
    return ev


def analyse(ev):
    merged = []
    cs, ce = ev[0][0], ev[0][1]
    for s, e, _ in ev[1:]:
        if s > ce:
            merged.append((cs, ce)); cs, ce = s, e
        else:
            ce = max(ce, e)
    merged.append((cs, ce))
    union_gaps = [(merged[i][0] - merged[i-1][1], merged[i-1][1])
                  for i in range(1, len(merged))]
    g0, at0 = max(union_gaps)
    start = at0 + g0
    tN = merged[-1][1]
    span = tN - start
    busy = sum(e - max(s, start) for s, e in merged if e > start)
    occ = dict(busy_pct=100*busy/span, idle_pct=100*(span-busy)/span, span=span)
    hist_t = [0.0] * (len(BUCKETS) + 1)
    attrib = defaultdict(float)
    prev = ev[0][1]
    for s, e, nm in ev[1:]:
        g = s - prev
        if g > 0 and s >= start:
            hist_t[bisect.bisect_right(BUCKETS, g)] += g
            lbl, mem = pretty(nm)
            attrib[(lbl, mem)] += g
        prev = max(prev, e)
    return occ, hist_t, attrib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-hip", default=DEFAULT_HIP)
    ap.add_argument("--trace-pt",  default=DEFAULT_PT)
    ap.add_argument("--out",  default="plots/mi300x")
    ap.add_argument("--name", default="rn50_gpu_idle")
    args = ap.parse_args()

    for path, label in [(args.trace_hip, "HIP"), (args.trace_pt, "PyTorch")]:
        if not Path(path).exists():
            sys.exit(f"trace not found ({label}): {path}")

    print("loading HIP trace ...")
    ev_hip = load_agent(args.trace_hip)
    print(f"  {len(ev_hip):,} kernels")
    occ_h, hist_h, att_h = analyse(ev_hip)
    print(f"  HIP  busy {occ_h['busy_pct']:.1f}%  idle {occ_h['idle_pct']:.1f}%")

    print("loading PyTorch trace ...")
    ev_pt = load_agent(args.trace_pt)
    print(f"  {len(ev_pt):,} kernels")
    occ_p, hist_p, att_p = analyse(ev_pt)
    print(f"  PT   busy {occ_p['busy_pct']:.1f}%  idle {occ_p['idle_pct']:.1f}%")

    # ── Union of labels for panel (c) — top-N by total idle across both ──────
    all_keys = set(att_h) | set(att_p)
    ranked   = sorted(all_keys, key=lambda k: -(att_h.get(k,0)+att_p.get(k,0)))[:TOP_N]
    c_lbls   = [k[0] for k in ranked]
    c_mem    = [k[1] for k in ranked]
    c_hip    = [att_h.get(k, 0)/1e9 for k in ranked]
    c_pt     = [att_p.get(k, 0)/1e9 for k in ranked]

    ps.setup()
    HIP_C = ps.PALETTE[1]   # orange-ish  — matches the rest of the paper figures
    PT_C  = ps.PALETTE[0]   # blue
    BUSY  = ps.PALETTE[2]
    IDLE_H = "#f4a460"       # sandy-orange for HIP idle
    IDLE_P = "#aec6e8"       # light-blue for PT idle
    MEM_C  = ps.PALETTE[3]   # highlight for memsync bars

    fig, (axA, axB, axC) = plt.subplots(
        1, 3,
        figsize=ps.figsize("text", height=2.9),
        gridspec_kw=dict(width_ratios=[0.85, 1.15, 1.55], wspace=0.52))

    # ────────────────────────────────────────────────────────────────────────
    # (a) Device occupancy — two stacked bars side by side
    # ────────────────────────────────────────────────────────────────────────
    for xi, (occ, bc, ic, lbl) in enumerate([
            (occ_h, HIP_C, IDLE_H, "HIP"),
            (occ_p, PT_C,  IDLE_P, "PyTorch")]):
        axA.bar(xi, occ["busy_pct"], color=bc,  width=0.62, label=f"{lbl} busy")
        axA.bar(xi, occ["idle_pct"], bottom=occ["busy_pct"],
                color=ic, width=0.62, label=f"{lbl} idle")
        axA.text(xi, occ["busy_pct"]/2,
                 f"{occ['busy_pct']:.0f}%", ha="center", va="center",
                 color="white", fontsize=8.5, fontweight="bold")
        axA.text(xi, occ["busy_pct"] + occ["idle_pct"]/2,
                 f"{occ['idle_pct']:.0f}%", ha="center", va="center",
                 color="#333", fontsize=7.5)

    axA.set_xticks([0, 1]); axA.set_xticklabels(["HIP", "PyTorch"], fontsize=8)
    axA.set_ylim(0, 100)
    axA.set_ylabel("Steady-state wall time (%)")
    axA.set_title("(a) Device occupancy", fontsize=8.5)

    # ────────────────────────────────────────────────────────────────────────
    # (b) Idle-gap spectrum — grouped bars
    # ────────────────────────────────────────────────────────────────────────
    x  = np.arange(len(BLABELS))
    w  = 0.38
    h_s = [t/1e9 for t in hist_h]
    p_s = [t/1e9 for t in hist_p]
    axB.bar(x - w/2, h_s, w, color=HIP_C, alpha=0.88, label="HIP")
    axB.bar(x + w/2, p_s, w, color=PT_C,  alpha=0.88, label="PyTorch")
    axB.set_xticks(x)
    axB.set_xticklabels(BLABELS, rotation=42, ha="right", fontsize=6.8)
    axB.set_ylabel("Idle time (s)")
    axB.set_xlabel("Inter-kernel gap size")
    axB.set_title("(b) Idle-gap spectrum", fontsize=8.5)
    axB.legend(loc="upper left", fontsize=6.5, framealpha=0.85)

    # ────────────────────────────────────────────────────────────────────────
    # (c) Idle attribution — paired horizontal bars
    # ────────────────────────────────────────────────────────────────────────
    nh     = len(c_lbls)
    ybase  = np.arange(nh) * 2.2     # spacing between groups
    bh     = 0.80
    axC.barh(ybase + bh/2, c_hip, bh, color=HIP_C, alpha=0.88, label="HIP")
    axC.barh(ybase - bh/2, c_pt,  bh, color=PT_C,  alpha=0.88, label="PyTorch")
    xmax   = max(max(c_hip), max(c_pt)) if (c_hip or c_pt) else 1.0
    for i, (vh, vp) in enumerate(zip(c_hip, c_pt)):
        if vh > 0:
            axC.text(vh + xmax*0.01, ybase[i]+bh/2, f"{vh:.1f}s",
                     va="center", fontsize=5.8, color=HIP_C)
        if vp > 0:
            axC.text(vp + xmax*0.01, ybase[i]-bh/2, f"{vp:.1f}s",
                     va="center", fontsize=5.8, color=PT_C)
        # memsync marker on the label
        if c_mem[i]:
            axC.text(-xmax*0.02, ybase[i], "*", ha="right",
                     va="center", fontsize=7, color=MEM_C, fontweight="bold")
    axC.set_yticks(ybase)
    axC.set_yticklabels(c_lbls, fontsize=6.5)
    axC.set_xlim(-xmax*0.08, xmax*1.22)
    axC.set_xlabel("Idle time before launch (s)")
    axC.set_title("(c) What the GPU waits to launch", fontsize=8.5)
    axC.legend(loc="lower right", fontsize=6.5, framealpha=0.85)
    axC.text(0.98, -0.13, "* = device memcpy / memset",
             transform=axC.transAxes, ha="right", fontsize=5.8,
             color=MEM_C, style="italic")

    for ax in (axA, axB, axC):
        ps.cleanup(ax)

    ps.figure_heading(
        fig, vendor="AMD", gpu="MI300X",
        sub_parts=[
            "custom HIP vs PyTorch-ROCm — full ImageNet-1K, batch 256, FP32",
            f"HIP {occ_h['busy_pct']:.0f}% busy / {occ_h['idle_pct']:.0f}% idle  |  "
            f"PyTorch {occ_p['busy_pct']:.0f}% busy / {occ_p['idle_pct']:.0f}% idle"])
    fig.subplots_adjust(top=0.80, bottom=0.27, left=0.065, right=0.99)
    ps.save(fig, args.name, out_dir=args.out)
    print(f"wrote {args.out}/{args.name}.pdf + .png")


if __name__ == "__main__":
    main()
