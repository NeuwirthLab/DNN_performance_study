#!/usr/bin/env python3.11
"""RCCL AllReduce communication overhead per step — MI300X (8 GPU, XGMI).

Simple 2-bar comparison: Native (single 97 MB AllReduce) vs PyTorch DDP
(4x bucketed AllReduce). Data + provenance in comm_overhead.csv; both bars are
derived from the SAME measured busbw curve (rccl-tests job 52385119, tuned
auto-algo). PyTorch error cap shows the 4->7 bucket range.

NOTE: AMD MI300X / RCCL data, labelled as such. The H100/NCCL equivalent was
not measured (no H100 traces in repo) — see comm_overhead.csv header.

    /usr/bin/python3.11 fig_comm_overhead_amd.py
"""
import csv
import paper_style as ps
import matplotlib.pyplot as plt

ROWS = {}
with open("comm_overhead.csv") as f:
    for r in csv.DictReader(filter(lambda l: not l.startswith("#"), f)):
        ROWS[r["implementation"]] = r

nat, pt = ROWS["native"], ROWS["pytorch_ddp"]
vals   = [float(nat["comm_ms_per_step"]), float(pt["comm_ms_per_step"])]
uppers = [float(nat["comm_ms_upper"]),    float(pt["comm_ms_upper"])]
labels = [nat["label"].replace("\\n", "\n"), pt["label"].replace("\\n", "\n")]
colors = [ps.NATIVE, ps.PYTORCH]
# asymmetric upper-only error: PyTorch range = 4 buckets (bar) -> 7 buckets (cap)
yerr = [[0.0, 0.0], [u - v for u, v in zip(uppers, vals)]]

ps.setup()
ps.big_fonts()
fig, ax = plt.subplots(figsize=ps.figsize("col", height=2.9))

x = range(len(vals))
bars = ax.bar(x, vals, width=0.62, color=colors, edgecolor="black", linewidth=0.7,
              yerr=yerr, capsize=4, error_kw=dict(elinewidth=1.0, ecolor="0.25"))

# value labels above each bar
for xi, v, u in zip(x, vals, uppers):
    txt = (f"{v:.2f}–{u:.2f} ms" if u > v + 1e-3 else f"{v:.2f} ms")
    ax.text(xi, u + 0.025, txt, ha="center", va="bottom",
            fontsize=9, fontweight="bold")

# annotate the structural cause (AllReduce calls/step)
for xi, impl in zip(x, (nat, pt)):
    ax.text(xi, 0.03, f"{impl['allreduce_calls_per_step']}× AllReduce",
            ha="center", va="bottom", fontsize=8.5, color="white", fontweight="bold")

ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("AllReduce time per step (ms)")
ax.set_ylim(0, max(uppers) * 1.22)
ax.set_title("RCCL gradient AllReduce overhead\nMI300X · 8 GPU XGMI · FP32 · 97 MB grads",
             fontsize=10.5, fontweight="bold")

ps.cleanup(ax)
ps.bold_ticks(ax)
ps.save(fig, "fig_comm_overhead_amd", out_dir=".")
print("wrote fig_comm_overhead_amd.pdf / .png  | native %.2f ms, pytorch %.2f-%.2f ms"
      % (vals[0], vals[1], uppers[1]))
