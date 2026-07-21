#!/usr/bin/env python3.11
"""
plot_cap_power_clock.py — the "requested vs. realized" figure.

Three series vs. the requested power cap while training ResNet-50 / ImageNet-1K
on 8× MI300X (native HIP, FP32):
  • requested cap        (the y=x ideal — what you ASK for)
  • actual power drawn    (measured avg W/GPU — what the silicon DELIVERS)
  • mean clock            (right axis — the DVFS state behind the power)

The gap between the dashed y=x line and the measured-power curve below ~400 W
is the cap-enforcement floor: the management interface accepts a 200 W cap, but
each GPU still draws ~354 W at ~558 MHz. That shaded band is the paper's
headline measurement-validity result.

Data: power_analysis/results/native/power_sweep_all.csv
Out : figures/cap_power_clock.{pdf,png}
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))            # paper_style.py lives at repo root
import paper_style as ps                 # noqa: E402

CSV = ROOT / "power_analysis/results/native/power_sweep_all.csv"
OUT_DIR = ROOT / "figures"


def load(path: Path):
    cap, pwr, clk = [], [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            cap.append(float(r["cap_w"]))
            pwr.append(float(r["avg_power_w_per_gpu"]))
            clk.append(float(r["mean_sm_clock_mhz"]))
    return cap, pwr, clk


def main() -> None:
    ps.setup()
    cap, pwr, clk = load(CSV)

    # the realized-power floor = the measured draw at the lowest requested cap
    floor = pwr[0]
    # last cap whose measured power is still pinned to the floor (within 5%)
    knee = max(c for c, p in zip(cap, pwr) if p <= floor * 1.12)

    fig, axL = ps.plt.subplots(figsize=ps.figsize("text", height=3.7))
    axR = axL.twinx()

    # ── shaded "cap not enforced" region (label sits above the plot frame) ──
    axL.axvspan(min(cap), knee, color="0.86", alpha=0.6, lw=0, zorder=0)
    axL.text(0.5 * (min(cap) + knee), 812,
             "cap not enforced", ha="center", va="bottom",
             fontsize=8, color="0.35", style="italic", zorder=1)

    # ── ideal y = x: requested == realized ──────────────────────────────────
    axL.plot([min(cap), max(cap)], [min(cap), max(cap)],
             ls=(0, (4, 3)), lw=1.5, color=ps.BLACK, zorder=2,
             label="requested cap (ideal)")

    # ── actual power drawn (left axis, vermillion = native) ─────────────────
    axL.plot(cap, pwr, marker="o", ms=5.5, lw=2.2,
             color=ps.NATIVE, zorder=4, label="actual power drawn")

    # ── mean clock (right axis, blue) ───────────────────────────────────────
    axR.plot(cap, clk, marker="^", ms=5.0, lw=2.0, ls="-",
             color=ps.PYTORCH, zorder=3, label="mean clock")

    # ── single consolidated floor callout (below the flat part, in the band) ─
    axL.annotate(f"~{floor:.0f} W / ~{clk[0]:.0f} MHz floor",
                 xy=(cap[1], pwr[1]), xytext=(min(cap) + 18, 232),
                 fontsize=8, color=ps.NATIVE, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=ps.NATIVE, lw=1.1,
                                 connectionstyle="arc3,rad=-0.2"))

    # ── energy-optimal cap marker (550 W from the J/img valley) ─────────────
    axL.axvline(550, color=ps.GREEN, lw=1.4, ls=":", zorder=2)
    axL.text(560, 185, "energy-optimal\n550 W", ha="left", va="bottom",
             fontsize=8, color=ps.GREEN, linespacing=1.1)

    axL.set_xlabel("Requested power cap (W / GPU)", fontsize=9.5)
    axL.set_ylabel("Power drawn (W / GPU)", color=ps.NATIVE, fontsize=9.5)
    axR.set_ylabel("Mean clock (MHz)", color=ps.PYTORCH, fontsize=9.5)
    axL.tick_params(axis="y", colors=ps.NATIVE, labelsize=8.5)
    axR.tick_params(axis="y", colors=ps.PYTORCH, labelsize=8.5)
    axL.tick_params(axis="x", labelsize=8.5)
    axL.set_xlim(min(cap) - 20, max(cap) + 20)
    axL.set_ylim(150, 820)
    axR.set_ylim(400, 1500)
    ps.cleanup(axL)

    # combined legend — upper-left, the genuinely empty quadrant
    hL, lL = axL.get_legend_handles_labels()
    hR, lR = axR.get_legend_handles_labels()
    axL.legend(hL + hR, lL + lR, loc="upper left", fontsize=8.5,
               frameon=False, handlelength=2.2, borderaxespad=0.6,
               labelspacing=0.4)

    fig.suptitle("Requested vs. realized power and clock\n"
                 "ResNet-50 / ImageNet-1K · 8× MI300X · FP32",
                 fontsize=9.5, y=1.0, linespacing=1.3)
    fig.tight_layout()
    ps.save(fig, "cap_power_clock", out_dir=OUT_DIR)
    print(f"  knee (floor extends to) = {knee:.0f} W | floor = {floor:.0f} W "
          f"@ {clk[0]:.0f} MHz")
    print(f"  -> {OUT_DIR}/cap_power_clock.pdf (+ .png)")


if __name__ == "__main__":
    main()
