#!/usr/bin/env python3
"""
Extract + plot for profile amd_52290776 (MI300X / gfx942), dependency-free.

Reads the already-generated roofline_summary.csv (does NOT regenerate it) and
writes, into plots/mi300x/:
  1. rn50_52290776_extracted.csv   — tidy per-kernel table (the extracted data)
  2. rn50_52290776_cache_hit.svg   — per-kernel L1/L2 hit rate (real data)
  3. rn50_52290776_roofline.svg    — FP32 VALU roofline (AI vs achieved TFLOP/s)
Also prints an ASCII table to stdout.

Collected passes (11/15): VALU FP32/FP16, MFMA, HBM, L1 (TCP), L2 (TCC_HIT/MISS).
Not collected: GRBM activity, SQ_BUSY, LDS, scalar-L1 (SQC), MFMA F64.
"""
import csv
import math
import os

CSV = "profiles/amd_52290776/roofline_summary.csv"
OUT = "plots/mi300x"
os.makedirs(OUT, exist_ok=True)

PEAK_FP32_TFLOPS = 163.4
PEAK_HBM_GBPS    = 5325.0


def fnum(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (ValueError, TypeError):
        return None


def short(name, n=22):
    name = name.split("(")[0]
    if name.startswith("_ZN"):
        name = "NCCL_reduce"
    return name if len(name) <= n else name[:n - 1] + "…"


rows = list(csv.DictReader(open(CSV)))
for r in rows:
    r["total_ms"] = float(r["total_ms"])
rows.sort(key=lambda r: -r["total_ms"])

# ── 1. Extracted tidy CSV + ASCII table ─────────────────────────────────────
cols = ["kernel", "dispatches", "total_ms", "fp32_tflops", "hbm_gbps",
        "ai_fp32", "l1_hit_pct", "l2_hit_pct", "pct_fp32_peak", "pct_bw_peak"]
ext = os.path.join(OUT, "rn50_52290776_extracted.csv")
with open(ext, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(cols)
    for r in rows:
        w.writerow([r.get(c, "") for c in cols])

hdr = f'{"kernel":26}{"ms":>8}{"fp32TF":>8}{"HBMgbs":>8}{"AI":>7}{"L1%":>6}{"L2%":>6}'
print(hdr); print("-" * len(hdr))
for r in rows[:12]:
    l1 = fnum(r["l1_hit_pct"]); l2 = fnum(r["l2_hit_pct"])
    print(f'{short(r["kernel"],26):26}{r["total_ms"]:>8.1f}'
          f'{float(r["fp32_tflops"]):>8.2f}{float(r["hbm_gbps"]):>8.0f}'
          f'{float(r["ai_fp32"]):>7.1f}'
          f'{("-" if l1 is None else f"{l1:.0f}"):>6}'
          f'{("-" if l2 is None else f"{l2:.0f}"):>6}')


# ── tiny SVG helpers ────────────────────────────────────────────────────────
SERIF = "Times New Roman, DejaVu Serif, serif"
NATIVE_COLOR = "#d62728"     # MI300 native HIP red — same as plot 9b
CAT_COLOR = {                # mirror plot 9b's category palette
    "Conv/GEMM":       NATIVE_COLOR,
    "BatchNorm":       "#dd8452",
    "Layout xform":    "#64b5cd",
    "ReLU":            "#55a868",
    "Pooling":         "#8172b3",
    "FC/Softmax":      "#c44e52",
    "AllReduce":       "#888888",
    "SGD update":      "#937860",
    "Residual/Reduce": "#ccb974",
    "Other":           "#666666",
}


# Canonical kernel label + category — keep in sync with paper_data/kernel_labels.py.
def classify(n):
    nl = n.lower()
    if "rccl" in nl or "allreduce" in nl or "onerankreduce" in nl: return "AllReduce"
    if "bn_" in nl or "batchnorm" in nl or "batch_norm" in nl: return "BatchNorm"
    if "relu" in nl: return "ReLU"
    if "pool" in nl: return "Pooling"
    if ("softmax" in nl or "fc_" in nl
            or nl.startswith("ce_loss") or "sce_" in nl): return "FC/Softmax"
    if "sgd" in nl: return "SGD update"
    if "add_inplace" in nl or "reduce_wgrad" in nl: return "Residual/Reduce"
    if any(k in nl for k in ("conv","gemm","wrw","winograd",
                             "fprop","wgrad","dgrad","xdlops")):
        return "Conv/GEMM"
    if nl.startswith("cijk_"): return "Conv/GEMM"
    if ("transpose" in nl or "nchw2" in nl or "nhwc2" in nl
            or "layout" in nl): return "Layout xform"
    return "Other"


def label_amd(n):
    nl = n.lower()
    if "naive_conv" in nl:
        if "bwd" in nl: return "Conv naive (bwd)"
        if "fwd" in nl: return "Conv naive (fwd)"
        if "wrw" in nl or "wgrad" in nl: return "Conv naive (wgrad)"
        return "Conv naive"
    if "Sp3AsmConv" in n:
        if "dilation2" in n: return "Conv 3x3 (d=2)"
        if "stride2"   in n: return "Conv 3x3 (s=2)"
        if "stride1"   in n: return "Conv 3x3 (s=1)"
        return "Conv 3x3"
    if "implicit_gemm" in nl:
        if "bwd" in nl: return "Conv impl-GEMM (bwd)"
        if "wrw" in nl or "wgrad" in nl: return "Conv impl-GEMM (wgrad)"
        if "fwd" in nl: return "Conv impl-GEMM (fwd)"
        return "Conv impl-GEMM"
    if n.startswith("Cijk_"):           return "GEMM (Tensile)"
    if "gemm" in nl:                    return "GEMM"
    if "batched_transpose" in n:        return "Layout transpose"
    if "nchw2nhwc" in nl:               return "Layout NCHW->NHWC"
    if "nhwc2nchw" in nl:               return "Layout NHWC->NCHW"
    if "BatchNormBwd"        in n:      return "BatchNorm bwd"
    if "BatchNormFwdInfer"   in n:      return "BatchNorm fwd (infer)"
    if "FinalMeanVariance"   in n:      return "BatchNorm mu/sigma (final)"
    if "MeanVariance"        in n:      return "BatchNorm mu/sigma"
    if "SpatialNorm"         in n:      return "BatchNorm norm"
    if "BatchNormFwd"        in n:      return "BatchNorm fwd"
    if "bn" in nl and "fwd" in nl:      return "BatchNorm fwd"
    if "bn" in nl and "bwd" in nl:      return "BatchNorm bwd"
    if "oneRankReduce"       in n:      return "RCCL reduce"
    if "rocclr_copyBuffer"   in n:      return "ROCm copyBuf"
    if "rocclr_fillBuffer"   in n:      return "ROCm fillBuf"
    if "rccl" in nl or "allreduce" in nl: return "RCCL AllReduce"
    if "relu_bwd" in nl:                return "ReLU bwd"
    if "relu" in nl:                    return "ReLU fwd"
    if "PoolingMax" in n or "MaxPool" in n: return "MaxPool"
    if "gavgpool_bwd" in nl:            return "AvgPool (bwd)"
    if "gavgpool"     in nl:            return "AvgPool"
    if "pool" in nl:                    return "Pooling"
    if "softmax" in nl:                 return "Softmax"
    if "fc_" in nl:                     return "FC"
    if "sce_bwd" in nl:                 return "Softmax-CE bwd"
    if "ce_loss" in nl:                 return "CE loss"
    if "sgd" in nl:                     return "SGD"
    if "add_inplace" in nl:             return "Add (residual)"
    if "add_bias"    in nl:             return "Bias add"
    if "bias_grad"   in nl:             return "Bias grad"
    if "top1"        in nl:             return "Top-1 acc"
    import re
    m = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)", n)
    return (m.group(1) if m else n)[:18]


def svg_open(w, h):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'font-family="{SERIF}" font-size="11">',
            f'<rect width="{w}" height="{h}" fill="white"/>']


def txt(x, y, s, size=12, anchor="start", color="#222", rot=None, weight="normal"):
    t = f' transform="rotate({rot} {x} {y})"' if rot is not None else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="{weight}"{t}>{s}</text>')


# ── 2. Per-kernel cache hit BAR chart, styled like plot 9b ─────────────────
# Top 8 kernels by GPU time, each with paired L1 and L2 bars.
# Chrome (serif, dashed grid, no top/right spines, title/footnote format)
# matches plot 9b so the figure pair reads as one family.
bars = []
for r in rows:
    l1 = fnum(r["l1_hit_pct"]); l2 = fnum(r["l2_hit_pct"])
    if l1 is None or l2 is None:
        continue
    bars.append({"l1": l1, "l2": l2, "ms": r["total_ms"],
                 "name": r["kernel"], "cat": classify(r["kernel"]),
                 "label": label_amd(r["kernel"])})
top = sorted(bars, key=lambda p: -p["ms"])[:8]

W, H = 960, 560
ml, mr, mt, mb = 80, 40, 78, 170
pw, ph = W - ml - mr, H - mt - mb
def Yb(v): return mt + ph * (1 - v / 100.0)
COL_L1 = "#4c72b0"     # blue:   L1 vector hit
COL_L2 = "#dd8452"     # orange: L2 hit (matches BatchNorm hue family for continuity)

s = svg_open(W, H)
s.append(txt(W/2, 32,
             f"ResNet-50 on MI300x: Per-Kernel Cache Hit Rate  "
             f"(rocprof, top {len(top)} of {len(rows)} kernels by GPU time)",
             14, "middle", weight="bold"))

# Cache-pressure band: bars under 50% land in this region
s.append(f'<rect x="{ml:.1f}" y="{Yb(50):.1f}" width="{pw:.1f}" '
         f'height="{mt+ph-Yb(50):.1f}" fill="#c44e52" opacity="0.06"/>')
s.append(txt(ml+pw-6, Yb(50)+14, "cache-pressure zone (hit &lt; 50%)",
             10, "end", "#9b3a3a", weight="bold"))

# Dashed gridlines + y ticks (matches 9b grid feel)
for g in (0, 20, 40, 60, 80, 100):
    yy = Yb(g)
    s.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw:.1f}" y2="{yy:.1f}" '
             f'stroke="#bbb" stroke-dasharray="2,3" opacity="0.55"/>')
    s.append(txt(ml-8, yy+3, f"{g}", 10, "end"))

# Axes (left & bottom only — top/right spines off)
s.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph:.1f}" stroke="#222"/>')
s.append(f'<line x1="{ml}" y1="{mt+ph:.1f}" x2="{ml+pw:.1f}" y2="{mt+ph:.1f}" '
         f'stroke="#222"/>')
s.append(txt(ml-54, mt+ph/2, "Cache hit rate (%)", 12, "middle", rot=-90))

# Bars: grouped pairs (L1, L2) per kernel, with value labels above each bar
N = len(top)
slot = pw / N
bw   = slot * 0.32
gap  = 3
for i, p in enumerate(top):
    cx = ml + slot * (i + 0.5)
    cat_col = CAT_COLOR.get(p["cat"], "#666")
    for (val, col, ox) in [(p["l1"], COL_L1, -bw - gap/2),
                           (p["l2"], COL_L2,        gap/2)]:
        bx = cx + ox
        bh = ph * val / 100.0
        by = mt + ph - bh
        s.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" '
                 f'height="{bh:.1f}" fill="{col}" fill-opacity="0.88" '
                 f'stroke="white" stroke-width="0.7"/>')
        s.append(txt(bx + bw/2, by - 4, f"{val:.0f}", 9, "middle",
                     col, weight="bold"))

    # Tick mark + kernel label (rotated) + GPU-time annotation underneath
    s.append(f'<line x1="{cx:.1f}" y1="{mt+ph:.1f}" '
             f'x2="{cx:.1f}" y2="{mt+ph+4:.1f}" stroke="#222"/>')
    # Small category dot beside the label so the kernel class is visible
    s.append(f'<circle cx="{cx-3:.1f}" cy="{mt+ph+14:.1f}" r="3.5" '
             f'fill="{cat_col}" stroke="white"/>')
    s.append(f'<text x="{cx+4:.1f}" y="{mt+ph+18:.1f}" font-size="10" '
             f'fill="#222" text-anchor="end" '
             f'transform="rotate(-28 {cx:.1f} {mt+ph+18:.1f})">'
             f'{p["label"]}</text>')
    s.append(f'<text x="{cx:.1f}" y="{mt+ph+78:.1f}" font-size="9" '
             f'fill="#666" text-anchor="middle" font-style="italic">'
             f'{p["ms"]:.0f} ms</text>')

# Legend (L1 / L2), top-right inside plot area
lx = ml + pw - 200; ly = mt + 6
s.append(f'<rect x="{lx-8:.1f}" y="{ly-14:.1f}" width="200" height="42" '
         f'rx="4" ry="4" fill="white" stroke="#999" stroke-width="0.6" '
         f'opacity="0.92"/>')
s.append(f'<rect x="{lx:.1f}" y="{ly-7:.1f}" width="14" height="11" '
         f'fill="{COL_L1}"/>')
s.append(txt(lx + 20, ly + 2, "L1 vector cache hit", 10, "start"))
s.append(f'<rect x="{lx:.1f}" y="{ly+9:.1f}" width="14" height="11" '
         f'fill="{COL_L2}"/>')
s.append(txt(lx + 20, ly + 18, "L2 cache hit", 10, "start"))

# Italic footnote (9b style)
foot = (f"Bars = L1 (blue) and L2 (orange) hit rate per kernel  ·  "
        f"category dot below each tick  ·  italic = total GPU time")
s.append(f'<text x="{W/2}" y="{H-10}" font-size="9" font-style="italic" '
         f'fill="#444" text-anchor="middle">{foot}</text>')

s.append("</svg>")
p1 = os.path.join(OUT, "rn50_52290776_cache_hit.svg")
open(p1, "w").write("\n".join(s))

# ── 3. FP32 VALU roofline, styled like plot 9b ─────────────────────────────
# Each kernel = one point; color = category; size ∝ sqrt(GPU time).
# Three marker conventions (same as 9b):
#   filled            → both AI > 0 and TFLOPS > 0
#   open at ridge     → TFLOPS measured, no HBM traffic (cache-resident)
#   open on floor     → AI measured, no FP ops (memory-only kernel)
BW_TBS = PEAK_HBM_GBPS / 1000.0
PEAK   = PEAK_FP32_TFLOPS

kernels = []
for r in rows:
    ai = fnum(r["ai_fp32"]) or 0.0
    tf = fnum(r["fp32_tflops"]) or 0.0
    if ai <= 0 and tf <= 0:
        continue                       # nothing measurable — skip silently
    kernels.append({"ai": ai, "tf": tf, "ms": r["total_ms"],
                    "name": r["kernel"], "cat": classify(r["kernel"]),
                    "label": label_amd(r["kernel"])})
total_dur = sum(p["ms"] for p in kernels) or 1.0

W, H = 820, 600
ml, mr, mt, mb = 80, 50, 78, 100
pw, ph = W - ml - mr, H - mt - mb
XLO, XHI = 0.05, 2000
YLO, YHI = 0.3, PEAK * 2

lx0, lx1 = math.log10(XLO), math.log10(XHI)
ly0, ly1 = math.log10(YLO), math.log10(YHI)
def Xr(ai): return ml + pw * (math.log10(max(ai, XLO)) - lx0) / (lx1 - lx0)
def Yr(tf): return mt + ph * (1 - (math.log10(max(tf, YLO)) - ly0) / (ly1 - ly0))


s = svg_open(W, H)
s.append(txt(W/2, 32,
             f"ResNet-50 on MI300x: Per-Kernel Roofline  "
             f"(rocprof, {len(kernels)} of {len(rows)} kernels)",
             14, "middle", weight="bold"))

# Decade gridlines (dashed, like 9b)
for ai_v in (0.1, 1, 10, 100, 1000):
    if XLO <= ai_v <= XHI:
        xx = Xr(ai_v)
        s.append(f'<line x1="{xx:.1f}" y1="{mt}" x2="{xx:.1f}" y2="{mt+ph:.1f}" '
                 f'stroke="#bbb" stroke-dasharray="2,3" opacity="0.55"/>')
        lab = f"{ai_v:g}"
        s.append(txt(xx, mt+ph+16, lab, 10, "middle"))
for tf_v in (1, 10, 100):
    if YLO <= tf_v <= YHI:
        yy = Yr(tf_v)
        s.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw:.1f}" y2="{yy:.1f}" '
                 f'stroke="#bbb" stroke-dasharray="2,3" opacity="0.55"/>')
        s.append(txt(ml-8, yy+3, f"{tf_v:g}", 10, "end"))

# Roofline: gray fill under the roof, then the black line on top
start_ai = max(XLO, YLO / BW_TBS)
start_y  = start_ai * BW_TBS
ridge_ai = PEAK / BW_TBS
fill_pts = [(Xr(start_ai), Yr(YLO)),
            (Xr(start_ai), Yr(start_y)),
            (Xr(ridge_ai), Yr(PEAK)),
            (Xr(XHI),      Yr(PEAK)),
            (Xr(XHI),      Yr(YLO))]
s.append('<polygon points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in fill_pts) +
         '" fill="#888" opacity="0.05"/>')
roof_pts = [(Xr(start_ai), Yr(start_y)),
            (Xr(ridge_ai), Yr(PEAK)),
            (Xr(XHI),      Yr(PEAK))]
s.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in roof_pts) +
         '" fill="none" stroke="#222" stroke-width="2"/>')
s.append(txt(Xr(XHI)-6, Yr(PEAK)-6,
             f"MI300x FP32 roof ({PEAK:.0f} TFLOPS)", 10, "end", "#222"))

# Ridge: vertical dotted line + small label
s.append(f'<line x1="{Xr(ridge_ai):.1f}" y1="{mt}" '
         f'x2="{Xr(ridge_ai):.1f}" y2="{mt+ph:.1f}" '
         f'stroke="#888" stroke-dasharray="1,4" opacity="0.7"/>')
s.append(txt(Xr(ridge_ai)+4, mt+ph-26, "Ridge",       9, "start", "#666"))
s.append(txt(Xr(ridge_ai)+4, mt+ph-14, f"{ridge_ai:.0f}", 9, "start", "#666"))

# Axes (left & bottom only)
s.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph:.1f}" stroke="#222"/>')
s.append(f'<line x1="{ml}" y1="{mt+ph:.1f}" x2="{ml+pw:.1f}" y2="{mt+ph:.1f}" stroke="#222"/>')
s.append(txt(ml+pw/2, H-50, "Arithmetic Intensity (FLOPs/byte, measured)",
             12, "middle"))
s.append(txt(ml-58, mt+ph/2, "Performance (TFLOPS)", 12, "middle", rot=-90))

# Scatter
def radius_r(ms):
    return max(5.0, 4.0 + 22.0 * math.sqrt(ms / total_dur))

floor_y = YLO * 1.4
n_full = n_cache = n_mem = 0
def pos(p):
    if p["ai"] > 0 and p["tf"] > 0:
        return Xr(p["ai"]), Yr(p["tf"]), "full"
    if p["tf"] > 0:
        return Xr(ridge_ai), Yr(p["tf"]), "cache"
    return Xr(p["ai"]), Yr(floor_y), "mem"

# Smallest first so big ones render on top
for p in sorted(kernels, key=lambda p: p["ms"]):
    col = CAT_COLOR.get(p["cat"], "#666")
    cx, cy, kind = pos(p)
    r_ = radius_r(p["ms"])
    if kind == "full":
        n_full += 1
        s.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r_:.1f}" '
                 f'fill="{col}" fill-opacity="0.85" stroke="white" stroke-width="1.2"/>')
    else:                              # open marker: empty fill, category-colored ring
        if kind == "cache": n_cache += 1
        else:               n_mem += 1
        s.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r_:.1f}" '
                 f'fill="white" stroke="{col}" stroke-width="1.5"/>')

# Annotate top 8 by GPU time, rounded white box, category-colored border
def bbox_label_r(x, y, text, color):
    w_ = 8 + 6.0 * len(text)
    bx = x + 9; by = y - 16
    return (f'<rect x="{bx:.1f}" y="{by:.1f}" width="{w_:.1f}" height="16" '
            f'rx="4" ry="4" fill="white" stroke="{color}" stroke-width="0.7" '
            f'opacity="0.92"/>'
            + txt(bx+4, by+12, text, 9, "start", color, weight="bold"))

for p in sorted(kernels, key=lambda p: -p["ms"])[:8]:
    col = CAT_COLOR.get(p["cat"], "#666")
    cx, cy, _ = pos(p)
    s.append(bbox_label_r(cx, cy, p["label"], col))

# Category legend (bottom-left, boxed) — matches the cache plot exactly
cats_present = sorted({p["cat"] for p in kernels},
                      key=lambda c: -sum(q["ms"] for q in kernels if q["cat"] == c))
lx, ly = ml + 12, mt + ph - 14*len(cats_present) - 8
s.append(f'<rect x="{lx-8:.1f}" y="{ly-14:.1f}" width="160" '
         f'height="{14*len(cats_present)+18:.1f}" rx="4" ry="4" '
         f'fill="white" stroke="#999" stroke-width="0.6" opacity="0.92"/>')
s.append(txt(lx, ly-2, "Category", 9, "start", "#222", weight="bold"))
for i, c in enumerate(cats_present):
    yy = ly + 12 + i*14
    s.append(f'<circle cx="{lx+4:.1f}" cy="{yy-4:.1f}" r="5" '
             f'fill="{CAT_COLOR[c]}" stroke="white"/>')
    s.append(txt(lx+14, yy, c, 9, "start"))

# Italic footnote matching 9b's wording
foot = (f"Filled = both AI and TFLOPS measured ({n_full})  ·  "
        f"open at ridge = compute-only / cache-resident ({n_cache})  ·  "
        f"open on slope = memory-only ({n_mem})")
s.append(f'<text x="{W/2}" y="{H-10}" font-size="9" font-style="italic" '
         f'fill="#444" text-anchor="middle">{foot}</text>')

s.append("</svg>")
p2 = os.path.join(OUT, "rn50_52290776_roofline.svg")
open(p2, "w").write("\n".join(s))

print(f"\nextracted data -> {ext}")
print(f"cache-hit plot -> {p1}")
print(f"roofline plot  -> {p2}")
print(f"kernels: {len(rows)} total, {len(top)} in cache plot, {len(kernels)} on roofline")
