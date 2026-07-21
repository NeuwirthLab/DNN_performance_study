# Plot styling reference (ResNet-50 MI300X paper)

**Single source of truth for every figure in this repo.** Before plotting
anything, read this and use [`paper_style.py`](paper_style.py) — do not hand-roll
colours, fonts, or legends. The scaling figure
[`plot_bundle/scaling/plot_scaling_combined.py`](plot_bundle/scaling/plot_scaling_combined.py)
(→ `rn50_scaling_combined.pdf`) is the reference look.

`paper_style.py` carries both the legacy figure API (`plot_heading`,
`shared_bottom_legend`, `save_vendor`, `VENDOR_*` … — used by ~24 scripts) **and**
the shared **scaling toolkit** added 2026-06-03. Use the toolkit for anything new.

---

## 1. Start every figure the same way

```python
import paper_style as ps
import matplotlib.pyplot as plt

ps.setup()                 # rcParams + Okabe-Ito palette + TrueType (fonttype 42)
# scaling figures only — roomier, bold typography:
ps.scaling_fonts()         # = big_fonts() + bold labels 12.5 / titles 12 / ticks 11

fig, ax = plt.subplots(figsize=ps.figsize("col"))   # "col"=3.5in, "text"=7.16in
# ... draw with ps.draw_series (see §3) ...
ps.cleanup(ax)             # despine top/right (accepts one Axes or many)
ps.save(fig, "fig_name", out_dir=".")   # writes fig_name.pdf + .png (600 dpi)
```

`ps.setup()` sets: serif, bold axis labels/titles, dashed low-contrast grid,
ticks-in, no top/right spines, TrueType fonts, savefig 600 dpi + tight bbox.

---

## 2. Colours (Okabe-Ito, colour-blind-safe — `ps.PALETTE`)

`#0072B2` blue · `#D55E00` vermillion · `#009E73` green · `#CC79A7` purple ·
`#F0E442` yellow · `#56B4E9` sky · `#E69F00` orange · `#000000` black.

### Implementation → colour (CANONICAL — use the named aliases)
| Series | Use | Hex |
|---|---|---|
| **Native HIP / custom** | `ps.NATIVE` | `#D55E00` (vermillion) |
| **PyTorch (ROCm/NGC)** | `ps.PYTORCH` | `#0072B2` (blue) |

> ⚠️ The earlier energy figures used the **swapped** mapping. New figures must use
> `ps.NATIVE` / `ps.PYTORCH`. (Re-run the energy figures to flip them — they
> currently don't import paper_style correctly.)

Reference lines: `ps.IDEAL` = grey dotted (`color="0.45", ls=":", lw=1.4`).

---

## 3. Scaling figures — the shared dual encoding

Four series read from **two** channels, covered by one legend:
- **Colour = implementation** (`ps.NATIVE`, `ps.PYTORCH`).
- **Line+marker = mode** (`ps.MODE_STYLES`): **Strong** = solid + filled diamond `D`;
  **Weak** = dashed + open circle `o` (white face, 1.6 edge, `clip_on=False`).

Draw and legend with the toolkit (handles error bars automatically):

```python
ps.draw_series(axT, gpus, tput,  ps.NATIVE,  ps.MODE_STYLES["strong"])
ps.draw_series(axT, gpus, tput,  ps.PYTORCH, ps.MODE_STYLES["weak"], yerr=std)  # 5x stats
axT.plot(gpus, ideal, **ps.IDEAL)                     # ideal-linear guide
axE.axhline(100, **ps.IDEAL)                          # 100% efficiency line

ps.bold_ticks(axT)                                    # call AFTER set_xticks/lim
# LEGEND: in-panel, TOP-LEFT of the throughput panel — its top-left corner is
# empty (curves rise bottom-left -> top-right) so it never overlaps the data.
ps.dual_legend(axT, [(ps.NATIVE,"Native HIP"), (ps.PYTORCH,"PyTorch")],
               modes=True, ideal=True, loc="upper left", ncol=1)
               # modes=False = single-mode figure. Avoid below=True (bottom strip).
```

Layout for the combined 2-panel fig: `figsize=(7.0, 3.4)`,
`gridspec_kw=dict(wspace=0.28, left=0.105, right=0.99, top=0.85, bottom=0.22)`,
panels `(a) Throughput` / `(b) Scaling efficiency`, suptitle bold ~12.5.

---

## 4. Axis labels — use `ps.LABELS` (never hand-type)

`ps.LABELS.GPUS / THROUGHPUT / THROUGHPUT_PER_GPU / SPEEDUP / EFFICIENCY /
EPOCH_TIME / ARITH_INTENSITY / PERFORMANCE_TF / CACHE_HIT / MFMA_OPS /
BATCH_SIZE / EPOCH / THROUGHPUT_RATIO`. Add new ones to the `LABELS` class rather
than typing strings inline (energy labels: "Energy per image (J)",
"Throughput per watt (img/s/W)", "EDP (J·s)", "Power cap (W/GPU)").

---

## 5. Sizes & output
- `ps.figsize("col")` = 3.50 in (1-col) · `ps.figsize("text")` = 7.16 in (2-col).
  Build at final print size — never `\resizebox` in LaTeX.
- **Always** `ps.save(fig, name)` → both `.pdf` (vector) + `.png` (600 dpi),
  TrueType, tight bbox. Don't call `fig.savefig` directly for paper figures.

---

## 6. Checklist for any new figure
- [ ] `ps.setup()` (+ `ps.scaling_fonts()` for scaling plots)
- [ ] `ps.NATIVE` (vermillion) / `ps.PYTORCH` (blue); mode via `ps.MODE_STYLES`
- [ ] draw with `ps.draw_series` (pass `yerr=` for the 5× stats figures)
- [ ] `ps.IDEAL` for ideal/100 % lines; labels from `ps.LABELS`; `ps.cleanup(ax)`
- [ ] one `ps.dual_legend(...)` **in-panel, top-left** (throughput panel), not a bottom strip; `ps.bold_ticks(ax)` after ticks set
- [ ] `ps.save(fig, name)` → PDF + 600-dpi PNG
