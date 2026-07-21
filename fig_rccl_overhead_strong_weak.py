"""
8-GPU scaling loss at 8x MI300X (single node, XGMI): HIP native vs PyTorch-ROCm
DDP, for BOTH strong and weak scaling. ResNet-50, FP32.

scaling_loss(%) = 100 * (1 - speedup/N),  speedup = ips_8 / ips_1,  N = 8
This is (1 - parallel efficiency); see the CAVEAT below -- it is NOT mostly RCCL.

Strong scaling (global batch fixed 256 -> per-GPU shrinks to 32 at 8 GPU):
    native job 52178693, PyTorch job 52424124
Weak scaling (per-GPU batch fixed 256 -> global grows to 2048 at 8 GPU):
    native job 52178693, PyTorch job 52425551 (FP32; the AMP run 52255683 was
    wrongly used before and inflated the weak loss to 31.7%).

Mirror of the 8xH100 nccl_overhead_strong_weak figure, same metric/layout, on
AMD data. Data source: paper_data/scaling.csv (MI300X, gfx942 SPX).

CAVEAT: this "overhead" is scaling LOSS (1 - efficiency), which the direct
rccl-tests measurement (job 52385119) shows is NOT mostly RCCL on XGMI -- the
~97 MB gradient AllReduce is ~0.6 ms (~0.2% of a step). The 8-GPU loss here is
dominated by the CPU data pipeline + per-step cudaStreamSynchronize lockstep.
See comm_overhead.csv for the true RCCL AllReduce ms/step.

Produces: fig_rccl_overhead_strong_weak.{png,pdf}
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paper_style as ps

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCALING_CSV = os.path.join(_SCRIPT_DIR, "paper_data", "scaling.csv")


def read_curve(impl, mode):
    """{gpus: img_per_sec} for one implementation+mode from scaling.csv."""
    pts = {}
    try:
        with open(SCALING_CSV, newline="") as fh:
            for row in csv.DictReader(fh):
                if row["implementation"] == impl and row["mode"] == mode:
                    pts[int(row["gpus"])] = float(row["img_per_sec"])
    except FileNotFoundError:
        return {}
    return pts


def nat(mode):
    return read_curve("HIP-native", mode)


def pyt(mode):
    return read_curve("PyTorch-ROCm-DDP", mode)


def overhead(curve, n=8):
    return 100.0 * (1 - (curve[n] / curve[1]) / n)


data = {
    "Strong": {"Native HIP":   nat("strong") or {1: 1217, 8: 6539},
               "PyTorch ROCm": pyt("strong") or {1: 1497.73, 8: 7326.58}},
    "Weak":   {"Native HIP":   nat("weak")   or {1: 1207, 8: 8446},
               "PyTorch ROCm": pyt("weak")   or {1: 1485.68, 8: 9622.39}},
}

regimes = ["Strong", "Weak"]
impls = ["Native HIP", "PyTorch ROCm"]
colors = {"Native HIP": ps.NATIVE, "PyTorch ROCm": ps.PYTORCH}
oh = {r: {im: overhead(data[r][im]) for im in impls} for r in regimes}

ps.setup()
fig, ax = plt.subplots(figsize=ps.figsize("col", h=2.9))

x = np.arange(len(regimes))
w = 0.36
for i, im in enumerate(impls):
    vals = [oh[r][im] for r in regimes]
    bars = ax.bar(x + (i - 0.5) * w, vals, w, label=im, color=colors[im],
                  edgecolor="black", linewidth=0.6, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f}%",
                ha="center", va="bottom", fontweight="bold", fontsize=8.5)

ax.set_xticks(x)
ax.set_xticklabels(["Strong scaling\n(per-GPU batch 256 to 32)",
                    "Weak scaling\n(per-GPU batch fixed 256)"])
ax.set_ylabel("Scaling loss  $1-S_8/8$  (%)")
ax.set_ylim(0, max(v for r in regimes for v in oh[r].values()) * 1.4)
ax.set_title("8-GPU scaling loss, MI300X (single node)")
# The bars are scaling loss (1 - efficiency), NOT communication: the measured
# RCCL all-reduce is only ~0.6 ms/step. Annotate that so the figure cannot be
# misread as a communication cost.
ax.text(0.97, 0.97,
        "RCCL all-reduce (measured):\n0.6 ms/step, <2% of step",
        transform=ax.transAxes, va="top", ha="right", fontsize=8.0, color="0.30",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", lw=0.6))
ax.yaxis.grid(True, ls=":", color="0.8", zorder=0)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left")
ps.cleanup(ax)
fig.tight_layout()
ps.save(fig, "fig_rccl_overhead_strong_weak", out_dir=_SCRIPT_DIR)

print("8-GPU (single-node XGMI) scaling loss (1 - S8/8):")
for r in regimes:
    for im in impls:
        c = data[r][im]
        print(f"  {r:<6} {im:<13}: {oh[r][im]:5.1f}%  "
              f"(ips_1={c[1]:.0f}, ips_8={c[8]:.0f})")
print(f"\nSaved -> {os.path.join(_SCRIPT_DIR, 'fig_rccl_overhead_strong_weak.png')}")
