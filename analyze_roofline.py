#!/usr/bin/env python3
"""
Roofline-style summary of a rocprofv3 multi-pass run produced by
profile_rn50_amd.slurm.

Reads every pmc_*/<host>/<pid>_counter_collection.csv under a profile root,
aggregates per-kernel averages of each counter, and prints:

    dispatches | avg_us | total_ms | FP32_TFLOPS | FP16_TFLOPS | HBM_GB/s
    arithmetic_intensity (FLOP/byte) | % of MI300X compute peak | % of HBM peak

Also writes the full per-kernel table to <profile_root>/roofline_summary.csv.

Usage:
    python3 analyze_roofline.py [PROFILE_ROOT]

If PROFILE_ROOT is omitted, prints a list of profile dirs that look usable
(under $PROJECT_DIR/profiles/).
"""

import csv
import glob
import os
import sys
from collections import defaultdict
from statistics import mean

# Share the SINGLE source of truth for kernel renaming with plot_roofline.py, so
# the cache-thrashing table here uses the exact same friendly labels + semantic
# categories as the roofline plot — and so a PyTorch profile lines up 1:1 with
# the native one. Fall back to identity if plot_roofline can't be imported
# (e.g. numpy/pandas/matplotlib not installed in this environment).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from plot_roofline import friendly_label
except Exception:  # pragma: no cover - keep analyzer usable without plot deps
    def friendly_label(kernel_name):
        return kernel_name.split("(")[0][:40], "Other"

# ──────────────────────────────────────────────────────────────────────────
# MI300X (gfx942) speed-of-light reference points
# Source: AMD spec sheet. These are theoretical peaks per GPU.
# ──────────────────────────────────────────────────────────────────────────
PEAK_FP32_TFLOPS    = 163.4   # VALU FMA, no MFMA
PEAK_FP16_TFLOPS    = 326.8   # VALU packed FP16
PEAK_FP16_MFMA_TF   = 1307.4  # Matrix core FP16 (not measured by VALU PMCs)
PEAK_HBM_GBPS       = 5325.0  # HBM3 aggregate
PEAK_L2_GBPS        = 17500.0 # rough; varies by SKU
WAVEFRONT_SIZE      = 64      # AMD wavefront width (lanes per VALU instr)

# ──────────────────────────────────────────────────────────────────────────
# FLOP / byte formulas
#
# Each SQ_INSTS_VALU_* count is the number of VALU instructions issued.
# Multiply by WAVEFRONT_SIZE to convert to per-lane operations:
#
#   FLOPs_fp32 = 64 * (2*FMA_F32 + ADD_F32 + MUL_F32 + TRANS_F32)
#   FLOPs_fp16 = 64 * (2*FMA_F16 + ADD_F16 + MUL_F16 + TRANS_F16)
#
# HBM bytes — TCC_EA0_*REQ_sum counters report request counts. Convention
# (matches rocprof-compute and AMD's own roofline scripts) is one cache line
# (64 B) per request:
#
#   HBM_bytes = 64 * (TCC_EA0_RDREQ_sum + TCC_EA0_WRREQ_sum)
#
# We also expose the more precise estimate built from the 32B/64B subset
# counters as `HBM_bytes_alt = 32*RDREQ_32B + 64*WRREQ_64B`. They differ a bit
# in practice but trend together.
# ──────────────────────────────────────────────────────────────────────────

COUNTERS_OF_INTEREST = {
    # ── Compute throughput (existing) ──
    "SQ_WAVES",
    "SQ_INSTS_VALU_FMA_F32",
    "SQ_INSTS_VALU_ADD_F32",
    "SQ_INSTS_VALU_MUL_F32",
    "SQ_INSTS_VALU_TRANS_F32",
    "SQ_INSTS_VALU_FMA_F16",
    "SQ_INSTS_VALU_ADD_F16",
    "SQ_INSTS_VALU_MUL_F16",
    "SQ_INSTS_VALU_TRANS_F16",
    # ── HBM traffic (existing) ──
    "TCC_EA0_RDREQ_sum",
    "TCC_EA0_WRREQ_sum",
    "TCC_EA0_RDREQ_32B_sum",
    "TCC_EA0_WRREQ_64B_sum",
    # ── L1 vector cache traffic (existing) ──
    "TCP_TCC_READ_REQ_sum",
    "TCP_TCC_WRITE_REQ_sum",
    "TCP_TOTAL_CACHE_ACCESSES_sum",
    # ── Direct L2 cache hit/miss (gfx942-verified) ──
    "TCC_HIT_sum",
    "TCC_MISS_sum",
    # Note: SQC_HITS/SQC_MISSES (scalar-L1 hit) do NOT exist on gfx942, so
    # scalar-L1 hit rate cannot be measured on this GPU.
    # ── MFMA / matrix-core ops (gfx942: F16/BF16/F32/F64 only, no I8) ──
    "SQ_INSTS_VALU_MFMA_MOPS_F32",
    "SQ_INSTS_VALU_MFMA_MOPS_F16",
    "SQ_INSTS_VALU_MFMA_MOPS_BF16",
    "SQ_INSTS_VALU_MFMA_MOPS_F64",
    # ── GPU activity + scheduler efficiency ──
    "GRBM_GUI_ACTIVE",
    "GRBM_COUNT",
    "SQ_BUSY_CU_CYCLES",   # gfx942 name (was SQ_BUSY_CYCLES; SQ_WAIT_ANY n/a)
    # ── LDS traffic ──
    "SQ_INSTS_LDS",        # SQ_WAIT_INST_LDS does not exist on gfx942
}


# ──────────────────────────────────────────────────────────────────────────
# Preflight: map each derived metric to the raw counters it needs, so we can
# tell which outputs are trustworthy for a given profile. rocprofv3 silently
# drops counters it does not recognise on the target GPU/build, so a counter
# listed in pmc.txt is not a guarantee it was collected.
# ──────────────────────────────────────────────────────────────────────────
DERIVED_METRIC_INPUTS = {
    "FP32 VALU TFLOPs": ["SQ_INSTS_VALU_FMA_F32", "SQ_INSTS_VALU_ADD_F32",
                         "SQ_INSTS_VALU_MUL_F32"],
    "HBM GB/s":         ["TCC_EA0_RDREQ_sum", "TCC_EA0_WRREQ_sum"],
    "L1 vector hit %":  ["TCP_TOTAL_CACHE_ACCESSES_sum", "TCP_TCC_READ_REQ_sum",
                         "TCP_TCC_WRITE_REQ_sum"],
    "L2 hit %":         ["TCC_HIT_sum", "TCC_MISS_sum"],
    "MFMA TFLOPs":      ["SQ_INSTS_VALU_MFMA_MOPS_F32", "SQ_INSTS_VALU_MFMA_MOPS_F16",
                         "SQ_INSTS_VALU_MFMA_MOPS_BF16", "SQ_INSTS_VALU_MFMA_MOPS_F64"],
    "GPU active %":     ["GRBM_GUI_ACTIVE", "GRBM_COUNT"],
}


def collect_present_counters(passes):
    """Union of counter names that actually appear in the raw pass data.

    parse_pass_csv only retains COUNTERS_OF_INTEREST, and a counter creates a
    key as soon as it is seen (even with value 0), so this set distinguishes
    "collected" from "silently dropped by rocprof".
    """
    present = set()
    for kernel_dict in passes.values():
        for entry in kernel_dict.values():
            present.update(entry["counters"].keys())
    return present


def preflight_counters(present):
    """Report requested-vs-collected counters and per-metric validity.

    Returns (missing_counters, invalid_metrics). invalid_metrics is the set of
    derived-metric names whose inputs were not collected; compute_roofline
    emits NaN for those so they are never mistaken for a real 0.0.
    """
    requested = set(COUNTERS_OF_INTEREST)
    missing = sorted(requested - present)

    print("=== Preflight: requested vs collected PMC counters ===")
    print(f"  requested : {len(requested)}")
    print(f"  collected : {len(requested & present)}")
    print(f"  MISSING   : {len(missing)}")
    if missing:
        print("  Requested but NOT collected by rocprofv3 (silently dropped —")
        print("  usually an unsupported counter name on this GPU/rocprof build).")
        print("  Confirm exact names with `rocprofv3 --list-avail`.")
        for c in missing:
            print(f"      [x] {c}")
    print()

    print("  Derived-metric validity:")
    invalid = set()
    for metric, inputs in DERIVED_METRIC_INPUTS.items():
        miss = [c for c in inputs if c not in present]
        if miss:
            invalid.add(metric)
            print(f"      [x] {metric:16s} INVALID -> emitted as NaN "
                  f"(missing {', '.join(miss)})")
        else:
            print(f"      [ok] {metric:16s} all inputs collected")
    print()
    return missing, invalid


def find_profile_dirs(project_dir):
    return sorted(glob.glob(os.path.join(project_dir, "profiles", "amd_*")))


def parse_pass_csv(path):
    """
    Returns: dict[kernel_name] -> {
        "dispatches": int,
        "duration_ns_sum": int,
        "counters": dict[counter_name] -> sum_value
    }
    """
    per_kernel = defaultdict(lambda: {
        "dispatches": 0,
        "duration_ns_sum": 0,
        "counters": defaultdict(float),
        "seen_dispatches": set(),
    })

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = row["Kernel_Name"]
            d = row["Dispatch_Id"]
            cn = row["Counter_Name"]
            if cn not in COUNTERS_OF_INTEREST:
                continue
            entry = per_kernel[k]
            try:
                cv = float(row["Counter_Value"])
            except ValueError:
                continue
            entry["counters"][cn] += cv
            # Count each dispatch only once even though each dispatch has one
            # row per counter (2 in our pmc.txt = 2 rows per dispatch).
            if d not in entry["seen_dispatches"]:
                entry["seen_dispatches"].add(d)
                entry["dispatches"] += 1
                try:
                    entry["duration_ns_sum"] += (
                        int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
                    )
                except (KeyError, ValueError):
                    pass

    # Drop the set; we have the count.
    for v in per_kernel.values():
        del v["seen_dispatches"]
    return dict(per_kernel)


def merge_passes(passes):
    """
    Merge per-pass parses into per-kernel aggregated counters with
    per-dispatch averages.

    Returns: dict[kernel] -> {
        "dispatches": <max across passes, treated as ground truth>,
        "avg_duration_ns": float (from pass with the most dispatches),
        "counter_avg_per_disp": dict[counter] -> float,
    }
    """
    all_kernels = set()
    for p in passes.values():
        all_kernels.update(p.keys())

    merged = {}
    for k in all_kernels:
        per_pass_dispatches = []
        per_pass_duration_ns = []
        counter_avg = {}
        for pass_name, kernel_dict in passes.items():
            entry = kernel_dict.get(k)
            if not entry or entry["dispatches"] == 0:
                continue
            per_pass_dispatches.append(entry["dispatches"])
            per_pass_duration_ns.append(
                entry["duration_ns_sum"] / entry["dispatches"]
            )
            for cn, total in entry["counters"].items():
                # Average value of this counter per dispatch in this pass.
                counter_avg[cn] = total / entry["dispatches"]

        if not per_pass_dispatches:
            continue
        merged[k] = {
            "max_dispatches": max(per_pass_dispatches),
            "avg_duration_ns": mean(per_pass_duration_ns),
            "counter_avg_per_disp": counter_avg,
        }
    return merged


def compute_roofline(merged, present=None):
    """
    For each kernel, compute the roofline-style metrics. We report *averages
    per dispatch*, which are well-defined regardless of how many dispatches
    each pass profiled.

    `present` is the set of counter names actually collected (from
    collect_present_counters). Derived metrics whose source counters were not
    collected are emitted as NaN rather than a misleading 0.0, so downstream
    consumers can tell "not measured" from "measured zero". If `present` is
    None, every counter is assumed available (legacy behaviour).
    """
    NAN = float("nan")
    if present is None:
        have_l1 = have_l2 = have_act = have_mfma = True
    else:
        have_l1   = {"TCP_TOTAL_CACHE_ACCESSES_sum", "TCP_TCC_READ_REQ_sum",
                     "TCP_TCC_WRITE_REQ_sum"} <= present
        have_l2   = {"TCC_HIT_sum", "TCC_MISS_sum"} <= present
        have_act  = {"GRBM_GUI_ACTIVE", "GRBM_COUNT"} <= present
        have_mfma = {"SQ_INSTS_VALU_MFMA_MOPS_F32", "SQ_INSTS_VALU_MFMA_MOPS_F16",
                     "SQ_INSTS_VALU_MFMA_MOPS_BF16",
                     "SQ_INSTS_VALU_MFMA_MOPS_F64"} <= present

    rows = []
    for k, v in merged.items():
        c = v["counter_avg_per_disp"]
        dur_s = v["avg_duration_ns"] * 1e-9

        # FLOPs per dispatch (averaged).
        flop_fp32 = WAVEFRONT_SIZE * (
            2 * c.get("SQ_INSTS_VALU_FMA_F32", 0)
            + c.get("SQ_INSTS_VALU_ADD_F32", 0)
            + c.get("SQ_INSTS_VALU_MUL_F32", 0)
            + c.get("SQ_INSTS_VALU_TRANS_F32", 0)
        )
        flop_fp16 = WAVEFRONT_SIZE * (
            2 * c.get("SQ_INSTS_VALU_FMA_F16", 0)
            + c.get("SQ_INSTS_VALU_ADD_F16", 0)
            + c.get("SQ_INSTS_VALU_MUL_F16", 0)
            + c.get("SQ_INSTS_VALU_TRANS_F16", 0)
        )

        # HBM bytes per dispatch.
        hbm_bytes = 64 * (
            c.get("TCC_EA0_RDREQ_sum", 0) + c.get("TCC_EA0_WRREQ_sum", 0)
        )
        hbm_bytes_alt = (
            32 * c.get("TCC_EA0_RDREQ_32B_sum", 0)
            + 64 * c.get("TCC_EA0_WRREQ_64B_sum", 0)
        )

        # Throughputs (per-dispatch averages converted to rates).
        tflops_fp32 = (flop_fp32 / dur_s) / 1e12 if dur_s > 0 else 0
        tflops_fp16 = (flop_fp16 / dur_s) / 1e12 if dur_s > 0 else 0
        hbm_gbps    = (hbm_bytes / dur_s) / 1e9  if dur_s > 0 else 0

        # Arithmetic intensity (FLOP/B). FP32-only for the standard plot.
        ai = flop_fp32 / hbm_bytes if hbm_bytes > 0 else 0

        # ── Cache hit rates (NEW) ─────────────────────────────────────
        # L1 vector cache: misses are the requests forwarded up to L2.
        # NaN => source counters not collected; 0.0 => collected but no traffic.
        tcp_total = c.get("TCP_TOTAL_CACHE_ACCESSES_sum", 0)
        tcp_to_l2 = (c.get("TCP_TCC_READ_REQ_sum", 0)
                     + c.get("TCP_TCC_WRITE_REQ_sum", 0))
        if not have_l1:
            l1_hit_pct = NAN
        else:
            l1_hit_pct = (100.0 * (1.0 - tcp_to_l2 / tcp_total)
                          if tcp_total > 0 else 0.0)
        # L2 cache: direct hit/miss counters.
        l2_hit  = c.get("TCC_HIT_sum",  0)
        l2_miss = c.get("TCC_MISS_sum", 0)
        if not have_l2:
            l2_hit_pct = NAN
        else:
            l2_hit_pct = (100.0 * l2_hit / (l2_hit + l2_miss)
                          if (l2_hit + l2_miss) > 0 else 0.0)
        # Scalar L1 (constant cache) hit rate: SQC_HITS/SQC_MISSES do not exist
        # on gfx942, so this is always unmeasurable (NaN) on MI300X.
        sl1_hit_pct = NAN

        # ── MFMA matrix-core ops (NEW) ────────────────────────────────
        # Matrix-multiply-add op counts per dispatch.  Each op = 1 FMA on
        # a tile of (M*N*K) elements; conversion to TFLOPS varies by
        # tile size and arch, so we expose the op rate (MOPS/dispatch)
        # rather than a synthetic TFLOPS for now.
        mfma_f32  = c.get("SQ_INSTS_VALU_MFMA_MOPS_F32",  0)
        mfma_f16  = c.get("SQ_INSTS_VALU_MFMA_MOPS_F16",  0)
        mfma_bf16 = c.get("SQ_INSTS_VALU_MFMA_MOPS_BF16", 0)
        mfma_f64  = c.get("SQ_INSTS_VALU_MFMA_MOPS_F64",  0)
        mfma_total = mfma_f32 + mfma_f16 + mfma_bf16 + mfma_f64
        # MfmaFlops (rocprof-compute): 512 FLOP per matrix-MOP. This is the
        # matrix-core compute the VALU FP32 counters cannot see, so it is the
        # number to put on the roofline for conv layers.
        if not have_mfma:
            mfma_tflops = NAN
        else:
            mfma_flop = 512.0 * mfma_total
            mfma_tflops = (mfma_flop / dur_s) / 1e12 if dur_s > 0 else 0.0

        # ── GPU activity + SM scheduler (NEW) ─────────────────────────
        grbm_act = c.get("GRBM_GUI_ACTIVE", 0)
        grbm_tot = c.get("GRBM_COUNT", 0)
        if not have_act:
            gpu_active_pct = NAN
        else:
            gpu_active_pct = (100.0 * grbm_act / grbm_tot
                              if grbm_tot > 0 else 0.0)
        # SM(CU)-busy stall fraction needed SQ_WAIT_ANY, which does not exist on
        # gfx942. Report raw CU-busy cycles/dispatch instead (SQ_BUSY_CU_CYCLES).
        sm_busy_pct = NAN
        sq_busy_cu = c.get("SQ_BUSY_CU_CYCLES", 0)

        # ── LDS traffic (SQ_WAIT_INST_LDS stall counter n/a on gfx942) ──
        lds_insts = c.get("SQ_INSTS_LDS", 0)

        label, category = friendly_label(k)
        rows.append({
            "kernel": k,
            "label": label,
            "category": category,
            "dispatches": v["max_dispatches"],
            "avg_us": v["avg_duration_ns"] / 1e3,
            "total_ms": v["max_dispatches"] * v["avg_duration_ns"] / 1e6,
            "fp32_tflops": tflops_fp32,
            "fp16_tflops": tflops_fp16,
            "hbm_gbps": hbm_gbps,
            "ai_fp32": ai,
            "pct_fp32_peak": 100 * tflops_fp32 / PEAK_FP32_TFLOPS,
            "pct_bw_peak":   100 * hbm_gbps    / PEAK_HBM_GBPS,
            "flop_fp32_per_disp": flop_fp32,
            "hbm_bytes_per_disp": hbm_bytes,
            "hbm_bytes_alt_per_disp": hbm_bytes_alt,
            "sq_waves_per_disp": c.get("SQ_WAVES", 0),
            # ── new columns (cache + MFMA + activity + LDS) ──
            "l1_hit_pct":      l1_hit_pct,
            "l2_hit_pct":      l2_hit_pct,
            "sl1_hit_pct":     sl1_hit_pct,
            "mfma_fp32_mops":  mfma_f32,
            "mfma_fp16_mops":  mfma_f16,
            "mfma_bf16_mops":  mfma_bf16,
            "mfma_fp64_mops":  mfma_f64,
            "mfma_i8_mops":    0.0,   # gfx942 has no MFMA INT8 MOPS counter
            "mfma_total_mops": mfma_total,
            "mfma_tflops":     mfma_tflops,   # 512 FLOP/MOP — real matrix-core rate
            "gpu_active_pct":  gpu_active_pct,
            "sm_busy_pct":     sm_busy_pct,
            "sq_busy_cu_per_disp": sq_busy_cu,
            "lds_insts_per_disp":  lds_insts,
        })
    rows.sort(key=lambda r: r["total_ms"], reverse=True)
    return rows


def truncate(s, n):
    return s if len(s) <= n else s[: n - 3] + "..."


def _fmt_pct(v):
    """Format a possibly-NaN percentage; show '   -' for unmeasured (NaN)."""
    return "   -" if v != v else f"{v:.1f}"  # v!=v is True only for NaN


def print_table(rows, top_n=15):
    # Kernel column now shows the FRIENDLY label (shared with plot_roofline), and
    # the L1/L2 hit-rate columns are included because cache-hit % IS the
    # cache-thrashing signal (low L2 hit % under high HBM traffic = thrashing).
    cols = [
        ("Kernel",         28, "label",          "{}",     "<"),
        ("Category",       16, "category",       "{}",     "<"),
        ("disp",            7, "dispatches",     "{:d}",   ">"),
        ("avg_us",          8, "avg_us",         "{:.1f}", ">"),
        ("tot_ms",          9, "total_ms",       "{:.1f}", ">"),
        ("AI_FP32",         8, "ai_fp32",        "{:.2f}", ">"),
        ("HBM_GB/s",        9, "hbm_gbps",       "{:.0f}", ">"),
        ("%BW",             5, "pct_bw_peak",    "{:.1f}", ">"),
        ("L1hit%",          7, "l1_hit_pct",     None,     ">"),
        ("L2hit%",          7, "l2_hit_pct",     None,     ">"),
    ]
    header = "  ".join(
        f"{name:{align}{w}}" for name, w, _, _, align in cols
    )
    print(header)
    print("-" * len(header))
    for r in rows[:top_n]:
        cells = []
        for name, w, k, fmt, align in cols:
            if name in ("Kernel", "Category"):
                val = truncate(str(r[k]), w)
            elif fmt is None:                       # cache % columns (NaN-aware)
                val = _fmt_pct(r[k])
            else:
                val = fmt.format(r[k])
            cells.append(f"{val:{align}{w}}")
        print("  ".join(cells))


def write_csv(rows, out_path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    project_dir = os.environ.get(
        "PROJECT_DIR",
        "/work/scratch/av30nuci/resent50/performance-analysis-dl-models-resnet50",
    )

    if len(sys.argv) > 1:
        profile_root = sys.argv[1]
    else:
        dirs = find_profile_dirs(project_dir)
        if not dirs:
            print(f"No profile dirs found under {project_dir}/profiles/")
            return 1
        print("Usage: analyze_roofline.py <profile_root>")
        print("Available profile dirs:")
        for d in dirs:
            csvs = glob.glob(os.path.join(d, "pmc_*", "*", "*_counter_collection.csv"))
            print(f"  {d}   ({len(csvs)} pass CSVs)")
        return 0

    csv_paths = sorted(glob.glob(os.path.join(
        profile_root, "pmc_*", "*", "*_counter_collection.csv"
    )))
    if not csv_paths:
        print(f"No *_counter_collection.csv found under {profile_root}")
        return 1

    print(f"Profile root : {profile_root}")
    print(f"Passes found : {len(csv_paths)}")
    print()

    passes = {}
    for p in csv_paths:
        pass_name = p.split("/pmc_")[1].split("/", 1)[0]
        passes[pass_name] = parse_pass_csv(p)
        print(f"  pmc_{pass_name}: {sum(v['dispatches'] for v in passes[pass_name].values())} dispatches, "
              f"{len(passes[pass_name])} unique kernels")
    print()

    present = collect_present_counters(passes)
    missing, invalid = preflight_counters(present)

    merged = merge_passes(passes)
    rows = compute_roofline(merged, present)

    print(f"=== Per-kernel roofline summary (top 15 by total_ms) ===")
    print(f"MI300X peaks: FP32 {PEAK_FP32_TFLOPS:.1f} TF | FP16 VALU {PEAK_FP16_TFLOPS:.1f} TF "
          f"| HBM {PEAK_HBM_GBPS:.0f} GB/s")
    print(f"(FP16 MFMA peak = {PEAK_FP16_MFMA_TF:.0f} TF — not captured by VALU counters)")
    print()
    print_table(rows, top_n=15)

    out_csv = os.path.join(profile_root, "roofline_summary.csv")
    write_csv(rows, out_csv)
    print()
    print(f"Full per-kernel table written to: {out_csv}")

    if invalid:
        print()
        print("!" * 70)
        print("WARNING: some derived metrics are INVALID for this profile because")
        print("their source counters were not collected. They are written as NaN")
        print("(not 0.0) so plots/consumers can skip them:")
        for m in sorted(invalid):
            print(f"    - {m}")
        print("Re-profile with valid counter names before trusting these metrics.")
        print("!" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
