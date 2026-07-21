#!/usr/bin/env python3
# ============================================================================
# compose_energy.py  --  Component-2 in-situ energy-coefficient pipeline.
# ----------------------------------------------------------------------------
# Extract per-operation energy coefficients for the HIP ResNet-50 from isolated
# component "replay" runs (matrix-engine / HBM-bandwidth / collective classes),
# then compose them with full-training hardware counters to PREDICT full-training
# energy -- in-sample on the native HIP trainer, cross-stack on PyTorch-ROCm.
#
# Model (per training step, aggregated over the active GPUs):
#     E_step ~= e_mfma(p)*F_mfma + e_hbm*B_hbm + e_comm*V_ar + P_idle*t_step
#       e_mfma(p) : J per matrix-engine FLOP, p in {FP32, BF16}
#       e_hbm     : J per HBM byte moved by bandwidth-bound (BN/act/...) kernels
#       e_comm    : J per all-reduced gradient byte (8-GPU bucket all-reduce)
#       P_idle    : ~140 W/GPU static floor  ->  P_idle*t_step is the idle energy
#
# A coefficient is  (per-iteration DYNAMIC energy) / (per-iteration work counted
# by rocprofv3).  Because the isolated replay loop is GPU-saturating and fires
# the SAME kernels every iteration, both the measured energy and the counted work
# scale linearly with iteration count, so the ratio of their *rates*
#     coeff = (E_dyn_J / train_s)  /  (work / busy_s)        [J per unit work]
# is clock/iteration-count independent.  busy_s is the rocprof GPU-busy time
# (sum of kernel total_ms); for a saturating replay it ~= train_s, and the ratio
# only depends on shapes/precision/kernels -- exactly what the replay contract
# guarantees is stable after iteration 1.
#
# ── STAGE 1  extract ────────────────────────────────────────────────────────
# For each component run dir (a measure_power.sh output with a rocprofv3 counter
# CSV alongside) read:
#   * the energy summary  : run_metrics.csv (or coefficients-style JSON) ->
#                           E_dyn_J, train_s, n_measured_epochs, idle_W ...
#   * the rocprofv3 work  : a roofline_summary.csv (preferred -- already friendly-
#                           labelled + per-disp MFMA mops / HBM bytes), OR raw
#                           *_counter_collection.csv passes (aggregated the same
#                           way analyze_roofline.py does it).
# and emit coefficients.json:
#   { e_mfma_fp32, e_mfma_bf16, e_hbm, e_comm, P_idle_W, per_cap{...}, provenance }
#
# ── STAGE 2  validate ───────────────────────────────────────────────────────
# Load coefficients.json + a full-training PMC CSV + that run's measured energy.
# Classify every kernel into {mfma, hbm, comm, other}; compute
#   F_mfma (FP32 & BF16 split), B_hbm, V_ar, t_step  PER STEP, then predict E_step.
# DOUBLE-COUNTING RULE: B_hbm counts ONLY non-mfma, non-comm kernels' bytes --
# conv/GEMM HBM is already inside e_mfma, the collective's inside e_comm; pricing
# those bytes again inflates every prediction.  Report predicted vs measured and
# percent error for native (in-sample) and pytorch (cross-stack).
#
# Pure standard library (csv/json/argparse/...). Runs under /usr/bin/python3.11.
#
# Counter conventions (gfx942), identical to analyze_roofline.py:
#   matrix FLOPs = 512 FLOP/mop * mfma_*_mops * dispatches
#   HBM bytes    = 64 B/req * (TCC_EA0_RDREQ_sum + TCC_EA0_WRREQ_sum)   (per disp)
# ============================================================================
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

# --- repo-relative defaults -------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                       # repo root (parent of power_analysis)


def _p(*parts):
    return os.path.join(_ROOT, *parts)


DEFAULT_NATIVE_PMC = _p("profiles", "native_in1k_retry_52488168", "roofline_summary.csv")
DEFAULT_PT_PMC     = _p("profiles", "pt_amd_52438059", "roofline_summary.csv")

# Matrix-core convention: 512 FLOP per MFMA matrix-MOP (rocprof-compute /
# analyze_roofline MfmaFlops).  HBM: 64 B per EA<->HBM request.
FLOP_PER_MOP = 512.0
BYTES_PER_REQ = 64.0
# Static idle floor per GPU (W) -- MI300X measured idle ~140 W/GPU; used only as a
# fallback when a run dir carries no measured idle_W.
DEFAULT_P_IDLE_W = 140.0


# ── component taxonomy ──────────────────────────────────────────────────────
# Map a rocprof friendly label / category to the energy-model component class.
# Four classes:  mfma  (matrix engine; HBM priced INSIDE e_mfma)
#                hbm   (bandwidth-bound; the ONLY bytes B_hbm prices)
#                comm  (collective all-reduce; bytes priced by e_comm)
#                other (memcpy/memset/optimizer/misc -- tracked, not priced in B_hbm)
def classify_component(label: str, category: str = "") -> str:
    l = (label or "").lower()
    c = (category or "").lower()
    # collective all-reduce first (its kernel name leaks into 'Other')
    if ("rankreduce" in l or "nccl" in l or "allreduce" in l or "all_reduce" in l
            or "rccl" in l or "reducekernel" in l):
        return "comm"
    # matrix engine, split: convolution (implicit-GEMM / Winograd) -> "mfma",
    # rocBLAS/CK GEMM (FC + GEMM-backed layers) -> "gemm". GEMM is ~1.6x less
    # FLOP-efficient than conv, so it carries its OWN coefficient (e_gemm).
    # Category wins over label so "Conv ... (impl. GEMM)" stays conv/mfma.
    if c == "convolution":
        return "mfma"
    if c == "gemm":
        return "gemm"
    if "winograd" in l or "conv" in l:
        return "mfma"
    if "gemm" in l:
        return "gemm"
    # explicit "other" buckets that must NOT be priced as HBM bandwidth
    if c in ("memory op", "optimizer") or "memcpy" in l or "memset" in l \
            or "fill" in l or "initheap" in l or "sgd" in l or "optim" in l \
            or "subtensor" in l:
        return "other"
    # everything bandwidth-bound: BN, ReLU, residual, bias, pool, softmax,
    # reductions, layout transforms (transpose/NCHW<->CNHW/concat), casts, scale.
    if c in ("batchnorm", "element-wise", "reduction", "layout transform"):
        return "hbm"
    # unknown -> treat as bandwidth-bound (it moves bytes); logged in provenance.
    return "hbm"


# ── rocprofv3 CSV readers ───────────────────────────────────────────────────
def _f(s, default=0.0):
    try:
        v = float(s)
        return v if v == v else default          # NaN -> default
    except (TypeError, ValueError):
        return default


def load_roofline_summary(path):
    """Read an analyze_roofline.py roofline_summary.csv into per-kernel rows.

    Returns list of dicts with keys: label, category, dispatches, busy_ms,
    mfma_mops_total, mfma_bf16_mops, mfma_fp16_mops, mfma_fp32_mops,
    hbm_bytes_per_disp.  Values are PER-DISPATCH (mops/bytes) as the file stores
    them; multiply by dispatches for run totals.
    """
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            disp = _f(r.get("dispatches"))
            if disp <= 0:
                continue
            rows.append(dict(
                label=r.get("label", "") or r.get("kernel", ""),
                category=r.get("category", ""),
                dispatches=disp,
                busy_ms=_f(r.get("total_ms")),
                mfma_mops_total=_f(r.get("mfma_total_mops")),
                mfma_fp32_mops=_f(r.get("mfma_fp32_mops")),
                mfma_fp16_mops=_f(r.get("mfma_fp16_mops")),
                mfma_bf16_mops=_f(r.get("mfma_bf16_mops")),
                hbm_bytes_per_disp=_f(r.get("hbm_bytes_per_disp")),
            ))
    return rows


# Raw long-format counter names (analyze_roofline.py vocabulary).
_MFMA_COUNTERS = {
    "fp32": "SQ_INSTS_VALU_MFMA_MOPS_F32",
    "fp16": "SQ_INSTS_VALU_MFMA_MOPS_F16",
    "bf16": "SQ_INSTS_VALU_MFMA_MOPS_BF16",
    "fp64": "SQ_INSTS_VALU_MFMA_MOPS_F64",
}
_RDREQ = "TCC_EA0_RDREQ_sum"
_WRREQ = "TCC_EA0_WRREQ_sum"


def _friendly_from_kernel(name):
    """Tiny fallback labeller for raw counter CSVs (roofline_summary already
    carries proper friendly labels; this only fires for the raw path)."""
    n = (name or "").lower()
    if "rankreduce" in n or "nccl" in n or "allreduce" in n or "rccl" in n:
        return "all-reduce", "Other"
    if "winograd" in n:
        return "Conv Winograd", "Convolution"
    if "conv" in n or "implicitgemm" in n or "miopen" in n and "conv" in n:
        return "Conv", "Convolution"
    if "gemm" in n or "cijk" in n:
        return "GEMM", "GEMM"
    if "batchnorm" in n or "bnorm" in n:
        return "BatchNorm", "BatchNorm"
    if "relu" in n:
        return "ReLU", "Element-wise"
    if "memset" in n or "fill" in n or "memcpy" in n:
        return "memory op", "Memory op"
    return name or "kernel", "Element-wise"


def load_raw_counter_csvs(run_dir):
    """Aggregate raw rocprofv3 *_counter_collection.csv passes (long format:
    one row per (dispatch, counter)) into the SAME per-kernel rows that
    load_roofline_summary returns.  Used when a component run emits raw passes
    instead of a derived roofline_summary.csv.

    Per analyze_roofline.py: a kernel's per-dispatch counter value is the mean
    over its dispatches; HBM bytes = 64*(RDREQ+WRREQ); mfma mops are the raw MOP
    counters.  busy_ms uses the dispatch wall durations when present.
    """
    csvs = sorted(glob.glob(os.path.join(run_dir, "**", "*_counter_collection.csv"),
                            recursive=True))
    if not csvs:
        return []
    # per kernel-name: counter-name -> [sum, ndisp]; plus dispatch count & busy ns
    agg = defaultdict(lambda: dict(counters=defaultdict(lambda: [0.0, 0]),
                                   disp=set(), busy_ns=0.0))
    for path in csvs:
        with open(path, newline="") as fh:
            seen_disp = set()
            for r in csv.DictReader(fh):
                kn = r.get("Kernel_Name", "")
                cn = r.get("Counter_Name", "")
                if not kn or not cn:
                    continue
                a = agg[kn]
                a["counters"][cn][0] += _f(r.get("Counter_Value"))
                # count a dispatch once per (kernel,dispatch_id), and only once
                # toward busy time per pass-file
                did = r.get("Dispatch_Id")
                key = (path, did)
                if key not in seen_disp:
                    seen_disp.add(key)
                    a["disp"].add(did)
                    st, en = _f(r.get("Start_Timestamp")), _f(r.get("End_Timestamp"))
                    if en > st:
                        a["busy_ns"] += (en - st)
                a["counters"][cn][1] += 1
    rows = []
    for kn, a in agg.items():
        ndisp = len(a["disp"]) or 1
        # per-dispatch average of each counter across all its sampled rows
        def cval(name):
            s, n = a["counters"].get(name, [0.0, 0])
            return (s / n) if n else 0.0
        mfma = {p: cval(c) for p, c in _MFMA_COUNTERS.items()}
        mfma_total = sum(mfma.values())
        hbm = BYTES_PER_REQ * (cval(_RDREQ) + cval(_WRREQ))
        label, category = _friendly_from_kernel(kn)
        rows.append(dict(
            label=label, category=category, dispatches=float(ndisp),
            busy_ms=a["busy_ns"] / 1e6,
            mfma_mops_total=mfma_total, mfma_fp32_mops=mfma["fp32"],
            mfma_fp16_mops=mfma["fp16"], mfma_bf16_mops=mfma["bf16"],
            hbm_bytes_per_disp=hbm))
    return rows


def load_pmc(path_or_dir):
    """Accept either a roofline_summary.csv file or a run dir containing one (or
    raw counter passes). Returns the per-kernel row list."""
    if os.path.isdir(path_or_dir):
        rs = os.path.join(path_or_dir, "roofline_summary.csv")
        if os.path.isfile(rs):
            return load_roofline_summary(rs)
        return load_raw_counter_csvs(path_or_dir)
    if path_or_dir.endswith(".csv"):
        # could be a roofline_summary or a raw counter CSV
        with open(path_or_dir, newline="") as fh:
            head = fh.readline()
        if "mfma_total_mops" in head or "hbm_bytes_per_disp" in head:
            return load_roofline_summary(path_or_dir)
        if "Counter_Name" in head:
            return load_raw_counter_csvs(os.path.dirname(path_or_dir) or ".")
        return load_roofline_summary(path_or_dir)
    raise SystemExit(f"unrecognised PMC source: {path_or_dir}")


# ── per-kernel work aggregation into model terms ────────────────────────────
def aggregate_work(rows):
    """Sum rocprof per-kernel rows into the model's work terms (RUN TOTALS, not
    per-step). Applies the component taxonomy + double-counting rule.

    Returns dict:
      F_mfma_fp32, F_mfma_bf16  : matrix FLOPs (FP32 / BF16 split)
      B_hbm                     : HBM bytes of NON-mfma, NON-comm kernels only
      V_ar                      : bytes moved by collective (comm) kernels
      busy_ms_*                 : GPU-busy time per class (rate denominators)
      by_class                  : {class: {flops, bytes, busy_ms, dispatches}}
      unknown_labels            : labels that fell through to the default class
    """
    out = dict(F_mfma_fp32=0.0, F_mfma_fp16=0.0, F_mfma_bf16=0.0,
               F_gemm=0.0, F_gemm_fp16=0.0, F_gemm_bf16=0.0,
               B_hbm=0.0, V_ar=0.0,
               busy_ms_total=0.0, busy_ms_mfma=0.0, busy_ms_gemm=0.0,
               busy_ms_hbm=0.0, busy_ms_comm=0.0)
    by_class = defaultdict(lambda: dict(flops=0.0, bytes=0.0, busy_ms=0.0,
                                        dispatches=0.0))
    unknown = []
    for r in rows:
        cls = classify_component(r["label"], r["category"])
        disp = r["dispatches"]
        flop = FLOP_PER_MOP * r["mfma_mops_total"] * disp
        bytes_ = r["hbm_bytes_per_disp"] * disp
        bc = by_class[cls]
        bc["flops"] += flop
        bc["bytes"] += bytes_
        bc["busy_ms"] += r["busy_ms"]
        bc["dispatches"] += disp
        out["busy_ms_total"] += r["busy_ms"]
        if cls == "mfma":
            # split matrix FLOPs by precision via the per-precision mop fractions.
            # FP16 and BF16 share the CDNA3 MFMA peak throughput, so FP16 is priced
            # at the BF16 rate (e_mfma_fp16 := e_mfma_bf16) but tracked separately.
            bf16 = r["mfma_bf16_mops"]
            fp16 = r["mfma_fp16_mops"]
            tot = r["mfma_mops_total"]
            frac_bf16 = (bf16 / tot) if tot > 0 else 0.0
            frac_fp16 = (fp16 / tot) if tot > 0 else 0.0
            out["F_mfma_bf16"] += flop * frac_bf16
            out["F_mfma_fp16"] += flop * frac_fp16
            out["F_mfma_fp32"] += flop * max(0.0, 1.0 - frac_bf16 - frac_fp16)
            out["busy_ms_mfma"] += r["busy_ms"]
        elif cls == "gemm":
            # split GEMM FLOPs by precision exactly like the MFMA term: the GEMM
            # class (rocBLAS/CK, incl. 1x1 convs) runs in whatever precision the
            # run uses, so reduced-precision (fp16/bf16) GEMM must be priced at
            # e_gemm_bf16, not the FP32 e_gemm. FP32 GEMM keeps F_gemm.
            bf16 = r["mfma_bf16_mops"]
            fp16 = r["mfma_fp16_mops"]
            tot = r["mfma_mops_total"]
            frac_bf16 = (bf16 / tot) if tot > 0 else 0.0
            frac_fp16 = (fp16 / tot) if tot > 0 else 0.0
            out["F_gemm_bf16"] += flop * frac_bf16
            out["F_gemm_fp16"] += flop * frac_fp16
            out["F_gemm"] += flop * max(0.0, 1.0 - frac_bf16 - frac_fp16)
            out["busy_ms_gemm"] += r["busy_ms"]
        elif cls == "comm":
            out["V_ar"] += bytes_                  # all-reduced gradient bytes
            out["busy_ms_comm"] += r["busy_ms"]
        elif cls == "hbm":
            out["B_hbm"] += bytes_                  # DOUBLE-COUNT RULE: only here
            out["busy_ms_hbm"] += r["busy_ms"]
            if r["category"] not in ("BatchNorm", "Element-wise", "Reduction",
                                     "Layout transform"):
                unknown.append(r["label"])
        # 'other' bytes are intentionally NOT added to B_hbm.
    out["by_class"] = {k: v for k, v in by_class.items()}
    out["unknown_labels"] = sorted(set(unknown))
    return out


# ── energy-summary reader (run_metrics.csv or JSON) ─────────────────────────
def load_energy_summary(run_dir):
    """Read measure_power.sh / analyze_power.py energy numbers for a run dir.

    Prefers run_metrics.csv (the one-row machine-readable summary). Falls back to
    a JSON with the same keys (energy.json / *_metrics.json).  Returns dict with
    E_dyn_J, E_train_J, train_s, idle_W, steady_W, n_measured_epochs,
    throughput_img_s, gpus_active (None where absent).
    """
    def _g(d, *keys):
        for k in keys:
            if k in d and d[k] not in (None, ""):
                return _f(d[k], None) if _is_num(d[k]) else d[k]
        return None

    rm = os.path.join(run_dir, "run_metrics.csv")
    if os.path.isfile(rm):
        with open(rm, newline="") as fh:
            rows = [r for r in csv.reader(fh) if r]
        d = dict(zip(rows[0], rows[1])) if len(rows) >= 2 else {}
    else:
        d = {}
        for cand in ("energy.json", "coefficients_run.json",
                     os.path.basename(run_dir) + "_metrics.json"):
            p = os.path.join(run_dir, cand)
            if os.path.isfile(p):
                with open(p) as fh:
                    d = json.load(fh)
                break
    return dict(
        E_dyn_J=_f(_g(d, "E_dyn_J", "e_dyn_j"), None),
        E_train_J=_f(_g(d, "E_train_J", "e_train_j"), None),
        train_s=_f(_g(d, "train_s", "train_seconds"), None),
        idle_W=_f(_g(d, "idle_W", "idle_w"), None),
        steady_W=_f(_g(d, "steady_W", "steady_w"), None),
        n_measured_epochs=_f(_g(d, "n_measured_epochs"), None),
        throughput_img_s=_f(_g(d, "throughput_img_s"), None),
        gpus_active=_f(_g(d, "gpus_active"), None),
        _raw=d,
    )


def _is_num(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


# ── STAGE 1: extract coefficients ───────────────────────────────────────────
def load_driver_rates(run_dir):
    """Parse component_loop's analytical [WORK]/[RATE] lines from command.log.

    The driver reports work_per_iter (exact conv FLOPs, or BN/ReLU bytes) and a
    per-block work_rate (work/sec), so we count work WITHOUT a profiler (rocprofv3
    GPU-hangs on the conv backward kernels). Only measured blocks (block>0; block 0
    is the dropped warmup) are used. Returns (work_per_iter, median_work_rate,
    unit) or (None, None, None) if no driver rates are present.
    """
    log = os.path.join(run_dir, "command.log")
    if not os.path.isfile(log):
        return None, None, None
    wpi = unit = None
    rates = []
    with open(log, errors="ignore") as f:
        for ln in f:
            m = re.search(r"\[WORK\].*work_per_iter=([0-9.eE+-]+)\s+unit=(\w+)", ln)
            if m:
                wpi, unit = float(m.group(1)), m.group(2)
            m = re.search(r"\[RATE\] block=(\d+).*work_rate=([0-9.eE+-]+)", ln)
            if m and int(m.group(1)) > 0:
                rates.append(float(m.group(2)))
    if not rates:
        return None, None, None
    rates.sort()
    n = len(rates)
    med = rates[n // 2] if n % 2 else 0.5 * (rates[n // 2 - 1] + rates[n // 2])
    return wpi, med, unit


def extract_coefficient(run_dir, work_term, energy, pmc_path=None):
    """Compute one energy coefficient = (dynamic power) / (work rate).

    work_term: which aggregate field carries this run's work
               ("F_mfma_fp32" | "F_mfma_bf16" | "B_hbm" | "V_ar").
    Returns (coeff [J per unit work], detail dict).

    PREFERRED path: the analytical driver rate (command.log [WORK]/[RATE]) with
    the measured dynamic power P_dyn = E_dyn_J/train_s, so coeff = P_dyn/work_rate
    needs no profiler. This is exact for conv FLOPs and robust to the intermittent
    MIOpen WrW GPU-hang (steady-state rate survives a partial run). Falls back to
    rocprofv3 counter work only when no driver rates are present.
    """
    E_dyn = energy.get("E_dyn_J")
    train_s = energy.get("train_s")
    wpi, drate, dunit = load_driver_rates(run_dir)
    if drate and E_dyn and train_s and train_s > 0:
        p_dyn = E_dyn / train_s                    # W (per active GPU)
        coeff = p_dyn / drate                      # J per unit work
        return coeff, dict(run_dir=run_dir, work_term=work_term,
                           work_total=drate * train_s, busy_s=train_s,
                           class_busy_frac=1.0, E_dyn_J=E_dyn, train_s=train_s,
                           P_dyn_W=p_dyn, work_rate=drate, work_per_iter=wpi,
                           work_unit=dunit, coeff=coeff,
                           method="driver-rate (analytical work, P_dyn/work_rate)")
    # ---- fallback: rocprofv3 counter-based work ----
    rows = load_pmc(pmc_path or run_dir)
    agg = aggregate_work(rows)
    work_total = agg.get(work_term, 0.0)
    busy_s = agg.get("busy_ms_total", 0.0) / 1e3
    class_key = {"F_mfma_fp32": "busy_ms_mfma", "F_mfma_bf16": "busy_ms_mfma",
                 "F_gemm": "busy_ms_gemm",
                 "B_hbm": "busy_ms_hbm", "V_ar": "busy_ms_comm"}[work_term]
    class_busy_s = agg.get(class_key, 0.0) / 1e3
    class_frac = (class_busy_s / busy_s) if busy_s > 0 else None
    E_dyn = energy.get("E_dyn_J")
    train_s = energy.get("train_s")
    detail = dict(run_dir=run_dir, work_term=work_term, work_total=work_total,
                  busy_s=busy_s, class_busy_frac=class_frac,
                  E_dyn_J=E_dyn, train_s=train_s)
    if not (work_total > 0) or not E_dyn or not train_s or not (train_s > 0):
        detail["coeff"] = None
        detail["note"] = "insufficient data (zero work or missing energy)"
        return None, detail
    p_dyn = E_dyn / train_s                       # W
    if busy_s > 0:
        wrate = work_total / busy_s
        coeff = p_dyn / wrate
        detail["method"] = "rate (P_dyn / work_rate)"
    else:
        coeff = E_dyn / work_total                # fallback: energy/work directly
        detail["method"] = "energy/work (no busy time)"
    detail["P_dyn_W"] = p_dyn
    detail["coeff"] = coeff
    return coeff, detail


# Component run dir layout the driver emits. A "spec" is (key, mode, precision,
# work_term).  key is the coefficients.json field it fills.  Directory naming is
# flexible: pass --run <key>=<dir> on the CLI, else we look for conventional
# subdir names under --components-dir.
COMPONENT_SPECS = [
    ("e_mfma_fp32",  "conv_fp32",       "FP32", "F_mfma_fp32"),
    ("e_mfma_bf16",  "conv_bf16",       "BF16", "F_mfma_bf16"),
    ("e_gemm",       "fc",              "FP32", "F_gemm"),
    ("e_hbm",        "bn_relu",         "FP32", "B_hbm"),
    ("e_comm",       "allreduce",       "FP32", "V_ar"),
    ("e_comm_2node", "allreduce_2node", "FP32", "V_ar"),
]
# conventional subdir name fragments to auto-discover a component run
_DIR_HINTS = {
    "e_gemm":       ["fc", "fc-gemm", "fcgemm", "gemm"],
    "e_mfma_fp32":  ["conv_fp32", "conv-fp32", "mfma_fp32", "convfp32", "fp32_conv"],
    "e_mfma_bf16":  ["conv_bf16", "conv-bf16", "mfma_bf16", "convbf16", "bf16_conv"],
    "e_hbm":        ["bn_relu", "bnrelu", "hbm", "bandwidth", "bn"],
    "e_comm":       ["allreduce", "all_reduce", "comm", "nccl", "rccl"],
    "e_comm_2node": ["allreduce_2node", "allreduce_2n", "comm_2node", "rccl_2node",
                     "rccl_2n", "nccl_2node"],
}


def _discover_component_dir(components_dir, key):
    if not components_dir or not os.path.isdir(components_dir):
        return None
    subs = [d for d in glob.glob(os.path.join(components_dir, "*"))
            if os.path.isdir(d)]
    for hint in _DIR_HINTS[key]:
        for d in subs:
            if hint in os.path.basename(d).lower():
                return d
    return None


def stage_extract(args):
    # resolve component run dirs: explicit --run key=dir wins, else discover.
    explicit = {}
    for spec in (args.run or []):
        if "=" not in spec:
            raise SystemExit(f"--run expects key=dir, got: {spec}")
        k, v = spec.split("=", 1)
        explicit[k.strip()] = v.strip()

    coeffs = {}
    provenance = {"runs": {}, "method": "coeff = P_dyn / work_rate (rate-based, "
                  "clock/iteration independent)",
                  "conventions": {"flop_per_mfma_mop": FLOP_PER_MOP,
                                  "bytes_per_hbm_req": BYTES_PER_REQ}}
    p_idle_list = []
    per_cap = defaultdict(dict)

    for key, mode, precision, work_term in COMPONENT_SPECS:
        run_dir = explicit.get(key) or _discover_component_dir(args.components_dir, key)
        if not run_dir or not os.path.isdir(run_dir):
            provenance["runs"][key] = {"status": "MISSING",
                                       "looked_for": _DIR_HINTS[key]}
            coeffs[key] = None
            continue
        energy = load_energy_summary(run_dir)
        coeff, detail = extract_coefficient(run_dir, work_term, energy)
        coeffs[key] = coeff
        detail["mode"] = mode
        detail["precision"] = precision
        provenance["runs"][key] = detail
        if energy.get("idle_W"):
            # idle_W in run_metrics is AGGREGATE over active GPUs; normalise to
            # per-GPU for P_idle_W.
            ngpu = energy.get("gpus_active") or 1
            p_idle_list.append(energy["idle_W"] / max(ngpu, 1))
        # per-cap capture: if the run dir name encodes a cap (cap_<W>), record it
        m = re.search(r"cap[_-]?(\d+)", os.path.basename(run_dir))
        if m:
            per_cap[int(m.group(1))][key] = coeff

    P_idle_W = (sum(p_idle_list) / len(p_idle_list)) if p_idle_list else DEFAULT_P_IDLE_W
    out = dict(
        e_mfma_fp32=coeffs.get("e_mfma_fp32"),
        e_mfma_bf16=coeffs.get("e_mfma_bf16"),
        e_gemm=coeffs.get("e_gemm"),
        e_hbm=coeffs.get("e_hbm"),
        e_comm=coeffs.get("e_comm"),
        e_comm_2node=coeffs.get("e_comm_2node"),
        P_idle_W=P_idle_W,
        units=dict(e_mfma_fp32="J/FLOP", e_mfma_bf16="J/FLOP", e_gemm="J/FLOP",
                   e_hbm="J/byte", e_comm="J/byte", e_comm_2node="J/byte",
                   P_idle_W="W per GPU"),
        provenance=provenance,
    )
    if per_cap:
        out["per_cap"] = {str(c): v for c, v in sorted(per_cap.items())}

    with open(args.coefficients, "w") as fh:
        json.dump(out, fh, indent=2)

    # human-readable echo
    print(f"=== extract -> {args.coefficients} ===")
    for key, _, _, _ in COMPONENT_SPECS:
        v = out.get(key)
        st = provenance["runs"].get(key, {})
        src = st.get("run_dir", st.get("status", "?"))
        print(f"  {key:14s} = {('%.4e' % v) if v else 'n/a':>12s}  "
              f"[{st.get('method','-')}]  {os.path.basename(str(src))}")
    print(f"  {'P_idle_W':14s} = {P_idle_W:8.1f} W/GPU")
    if out.get("per_cap"):
        print(f"  per-cap coefficients for caps: {list(out['per_cap'])}")
    return out


# ── STAGE 2: validate / compose ─────────────────────────────────────────────
def _steps_for(rows, energy, override):
    """Steps the PMC pass captured, for the PER-STEP *display* terms ONLY (the
    energy comparison itself is step-free, see predict_energy).

    rocprofv3 multiplexes counters across passes, so the roofline_summary
    `dispatches` column is a PER-PASS count, not a run total -- consistent only
    within a kernel's own pass. The loss / cross-entropy kernel fires exactly
    once per training step, so its dispatch count is the step count of the pass
    that profiled it; we use it purely to scale the reported per-step work.
    """
    if override:
        return float(override), "cli --steps"
    anchors = [("cross-entropy loss", 1.0), ("softmax-ce bwd", 1.0),
               ("top-1 accuracy", 1.0), ("global avg pool", 1.0)]
    for name, per_step in anchors:
        for r in rows:
            if name == (r["label"] or "").lower() and r["dispatches"] > 0:
                return r["dispatches"] / per_step, f"anchor '{r['label']}'"
    return 1.0, "none (per-step == per-pass total)"


def predict_energy(coeffs, pmc_rows, energy, steps_override=None, ngpu=None, nnodes=1):
    """Compose coefficients with full-training counters -> predicted energy.

    Core composition is RATE-based and STEP-COUNT-FREE, matching how the
    coefficients were extracted (each is J per unit work, == W per unit
    work-rate). Predicted mean per-GPU dynamic power:

        P_dyn_pred = e_mfma_fp32*R_fp32 + e_mfma_bf16*R_bf16
                     + e_hbm*R_hbm + e_comm*R_comm                       [W]

    where R_x = work_x / busy_s is the work-RATE over the profiled GPU-busy time
    (sum of kernel total_ms). Because rocprof multiplexes counters across passes,
    work_x and busy_s are scaled by the SAME pass-dispatch count, so each ratio
    R_x is pass-count invariant -- no fragile step inference enters the energy.

    Full-training predicted DYNAMIC energy (aggregate over active GPUs):
        E_dyn_pred = P_dyn_pred * ngpu * train_s
    compared against the measured E_dyn_J. Predicted TOTAL adds the idle floor
    P_idle*ngpu*train_s. Per-step terms (F_mfma, B_hbm, V_ar, t_step) are also
    reported, scaled by the loss-anchor step count, for the readable table.
    """
    agg = aggregate_work(pmc_rows)
    busy_s_total = agg["busy_ms_total"] / 1e3
    busy_mfma = agg["busy_ms_mfma"] / 1e3
    busy_hbm  = agg["busy_ms_hbm"] / 1e3
    busy_comm = agg["busy_ms_comm"] / 1e3

    # Resolve ngpu early — needed for the comm-term fix below.
    E_dyn = energy.get("E_dyn_J")
    train_s = energy.get("train_s")
    if ngpu is None:
        ngpu = energy.get("gpus_active") or 1
    ngpu = max(ngpu, 1)

    # work-RATES over the profiled GPU-busy window. A class's work is divided by
    # TOTAL busy time (not just its own class busy time): we want the contribution
    # to mean power across the whole step, i.e. work per wall-second of the run.
    # busy_s_total is the per-GPU GPU-busy time the profile spanned.
    bs = busy_s_total if busy_s_total > 0 else 1.0
    R_fp32 = agg["F_mfma_fp32"] / bs
    R_fp16 = agg.get("F_mfma_fp16", 0.0) / bs
    R_bf16 = agg["F_mfma_bf16"] / bs
    R_gemm = agg.get("F_gemm", 0.0) / bs                                 # FP32 GEMM
    R_gemm_bf16 = (agg.get("F_gemm_bf16", 0.0)
                   + agg.get("F_gemm_fp16", 0.0)) / bs                    # reduced-prec GEMM
    R_hbm  = agg["B_hbm"] / bs

    # Comm-term fix: at 1-GPU, oneRankReduce writes the gradient buffer to HBM
    # but is NOT an inter-GPU allreduce. The 1-GPU PMC's V_ar comes entirely from
    # that kernel, giving a spurious R_comm and ~7 W of phantom comm power.
    # At multi-GPU, use the actual gradient volume (97.49 MB for ResNet-50's 161
    # tensors) divided by the real per-step time inferred from measured throughput,
    # not the PMC's V_ar / busy_s (which uses 1-GPU kernel timing).
    _V_AR_REAL = 97.49e6   # ResNet-50 gradient buffer: 161 tensors = 97.49 MB
    _LOCAL_BATCH = 256      # per-GPU batch for all validation runs
    if ngpu <= 1:
        R_comm = 0.0
    else:
        tput = energy.get("throughput_img_s")
        if tput and tput > 0:
            t_step_real = _LOCAL_BATCH * ngpu / tput   # wall-time per allreduce
            R_comm = _V_AR_REAL / t_step_real
        else:
            R_comm = agg["V_ar"] / bs   # fallback if throughput unavailable

    e_mfma_fp32 = coeffs.get("e_mfma_fp32") or 0.0
    e_mfma_bf16 = coeffs.get("e_mfma_bf16") or 0.0
    # FP16 priced at the BF16 rate unless a dedicated e_mfma_fp16 is supplied
    # (FP16 and BF16 share the CDNA3 matrix-engine peak throughput).
    e_mfma_fp16 = coeffs.get("e_mfma_fp16") or e_mfma_bf16
    e_gemm = coeffs.get("e_gemm") or 0.0
    # reduced-precision (fp16/bf16) GEMM coefficient. If not separately measured,
    # fall back to the FP32 e_gemm scaled by the BF16/FP32 MFMA efficiency ratio
    # (GEMM and conv share the matrix engines, so the precision scaling matches).
    _gemm_ratio = (e_mfma_bf16 / e_mfma_fp32) if e_mfma_fp32 else 0.348
    e_gemm_bf16 = coeffs.get("e_gemm_bf16") or (e_gemm * _gemm_ratio)
    e_hbm = coeffs.get("e_hbm") or 0.0
    # Select the comm coefficient based on the number of training nodes.
    # At nnodes>=2 inter-node RCCL traffic (IB/RoCE) is more expensive per byte
    # than intra-node XGMI. Use e_comm_2node when available; fall back to e_comm.
    if nnodes >= 2 and coeffs.get("e_comm_2node"):
        e_comm = coeffs["e_comm_2node"]
    else:
        e_comm = coeffs.get("e_comm") or 0.0
    P_idle = coeffs.get("P_idle_W") or DEFAULT_P_IDLE_W

    # predicted per-GPU dynamic power, by component
    p_fp32 = e_mfma_fp32 * R_fp32
    p_fp16 = e_mfma_fp16 * R_fp16
    p_bf16 = e_mfma_bf16 * R_bf16
    p_gemm = e_gemm * R_gemm
    p_gemm_bf16 = e_gemm_bf16 * R_gemm_bf16
    p_hbm  = e_hbm * R_hbm
    p_comm = e_comm * R_comm
    P_dyn_pred_pergpu = (p_fp32 + p_fp16 + p_bf16
                         + p_gemm + p_gemm_bf16 + p_hbm + p_comm)

    # measured mean dynamic power (aggregate -> per GPU)
    P_dyn_meas_agg = (E_dyn / train_s) if (E_dyn and train_s) else None
    P_dyn_meas_pergpu = (P_dyn_meas_agg / ngpu) if P_dyn_meas_agg else None

    # full-training energy predictions (aggregate over active GPUs)
    E_dyn_pred = (P_dyn_pred_pergpu * ngpu * train_s) if train_s else None
    E_idle = (P_idle * ngpu * train_s) if train_s else None
    E_total_pred = (E_dyn_pred + E_idle) if (E_dyn_pred is not None and E_idle is not None) else None
    E_total_meas = (E_dyn + E_idle) if (E_dyn is not None and E_idle is not None) else None

    # per-step display terms (step anchor; does NOT affect the energy above)
    steps, steps_src = _steps_for(pmc_rows, energy, steps_override)
    steps = max(steps, 1.0)
    F_fp32 = agg["F_mfma_fp32"] / steps
    F_bf16 = agg["F_mfma_bf16"] / steps
    B_hbm  = agg["B_hbm"] / steps
    V_ar   = agg["V_ar"] / steps
    t_step = (busy_s_total / steps) if steps else None

    def err(pred, meas):
        if pred is None or meas in (None, 0):
            return None
        return 100.0 * (pred - meas) / meas

    return dict(
        ngpu=ngpu, busy_s=busy_s_total, train_s=train_s,
        busy_s_by_class=dict(mfma=busy_mfma, gemm=agg.get("busy_ms_gemm", 0.0) / 1e3,
                             hbm=busy_hbm, comm=busy_comm),
        work_rates=dict(R_fp32=R_fp32, R_fp16=R_fp16, R_bf16=R_bf16,
                        R_gemm=R_gemm, R_gemm_bf16=R_gemm_bf16,
                        R_hbm=R_hbm, R_comm=R_comm),
        pred_power_pergpu_W=dict(mfma_fp32=p_fp32, mfma_fp16=p_fp16,
                                 mfma_bf16=p_bf16, gemm=p_gemm,
                                 gemm_bf16=p_gemm_bf16,
                                 hbm=p_hbm, comm=p_comm,
                                 total=P_dyn_pred_pergpu),
        P_dyn_pred_pergpu_W=P_dyn_pred_pergpu,
        P_dyn_meas_pergpu_W=P_dyn_meas_pergpu,
        err_power_pct=err(P_dyn_pred_pergpu, P_dyn_meas_pergpu),
        E_dyn_pred_J=E_dyn_pred, E_dyn_meas_J=E_dyn,
        E_idle_J=E_idle, P_idle_W=P_idle,
        E_total_pred_J=E_total_pred, E_total_meas_J=E_total_meas,
        err_dyn_pct=err(E_dyn_pred, E_dyn),
        err_total_pct=err(E_total_pred, E_total_meas),
        steps=steps, steps_src=steps_src, t_step_s=t_step,
        work_per_step=dict(F_mfma_fp32=F_fp32, F_mfma_bf16=F_bf16,
                           B_hbm=B_hbm, V_ar=V_ar),
        unknown_labels=agg["unknown_labels"],
    )


def stage_validate(args):
    with open(args.coefficients) as fh:
        coeffs = json.load(fh)

    targets = []
    if args.native_pmc:
        targets.append(("native (in-sample)", args.native_pmc, args.native_energy,
                        args.native_steps, args.native_gpus))
    if args.pytorch_pmc:
        targets.append(("pytorch (cross-stack)", args.pytorch_pmc, args.pytorch_energy,
                        args.pytorch_steps, args.pytorch_gpus))
    if not targets:
        raise SystemExit("validate: give at least one of --native-pmc / --pytorch-pmc")

    nnodes = getattr(args, "nnodes", 1)
    print(f"=== coefficients in use (nnodes={nnodes}) ===")
    for k in ("e_mfma_fp32", "e_mfma_bf16", "e_hbm", "e_comm", "e_comm_2node", "P_idle_W"):
        v = coeffs.get(k)
        active = " [ACTIVE for inter-node]" if k == "e_comm_2node" and nnodes >= 2 else ""
        print(f"  {k:14s} = {('%.4e' % v) if isinstance(v,(int,float)) and v else v}{active}")

    def _n(v, fmt="%.1f"):
        return (fmt % v) if isinstance(v, (int, float)) else "n/a"

    results = {}
    print("\n=== validation: predicted vs measured (full-training energy) ===")
    hdr = (f"{'target':22s} {'Pdyn_pred':>10s} {'Pdyn_meas':>10s} {'errP%':>7s}  "
           f"{'Edyn_pred':>11s} {'Edyn_meas':>11s} {'errE%':>7s} {'errTot%':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for name, pmc, energy_dir, steps, gpus in targets:
        rows = load_pmc(pmc)
        en = load_energy_summary(energy_dir) if energy_dir else _energy_from_cli(args, name)
        res = predict_energy(coeffs, rows, en, steps_override=steps, ngpu=gpus,
                             nnodes=getattr(args, "nnodes", 1))
        results[name] = res
        print(f"{name:22s} {_n(res['P_dyn_pred_pergpu_W']):>10s} "
              f"{_n(res['P_dyn_meas_pergpu_W']):>10s} {_n(res['err_power_pct']):>7s}  "
              f"{_n(res['E_dyn_pred_J'],'%.0f'):>11s} {_n(res['E_dyn_meas_J'],'%.0f'):>11s} "
              f"{_n(res['err_dyn_pct']):>7s} {_n(res['err_total_pct']):>8s}")
        # per-GPU predicted dynamic power decomposition + per-step work
        c = res["pred_power_pergpu_W"]
        w = res["work_per_step"]
        fmf = w["F_mfma_fp32"] + w["F_mfma_bf16"]
        print(f"    ngpu={res['ngpu']:.0f} busy={res['busy_s']:.2f}s "
              f"train_s={_n(res['train_s'],'%.1f')}s P_idle={res['P_idle_W']:.0f}W/GPU "
              f"| pred per-GPU dyn W: mfma_fp32={c['mfma_fp32']:.1f} "
              f"mfma_fp16={c.get('mfma_fp16', 0.0):.1f} "
              f"mfma_bf16={c['mfma_bf16']:.1f} "
              f"gemm={c.get('gemm', 0.0):.1f} "
              f"gemm_bf16={c.get('gemm_bf16', 0.0):.1f} "
              f"hbm={c['hbm']:.1f} comm={c['comm']:.1f}")
        print(f"    per-step work (anchor={res['steps_src']}, steps={res['steps']:.0f}): "
              f"F_mfma={fmf:.3e} FLOP  B_hbm={w['B_hbm']/1e9:.2f} GB  "
              f"V_ar={w['V_ar']/1e9:.3f} GB  t_step={(res['t_step_s'] or 0)*1e3:.1f}ms")
        if res["E_total_pred_J"] is not None:
            print(f"    full-train energy (dyn+idle): pred={res['E_total_pred_J']:.0f} J  "
                  f"meas={res['E_total_meas_J']:.0f} J  (idle={res['E_idle_J']:.0f} J)")
        if res["unknown_labels"]:
            print(f"    [warn] labels defaulted to HBM class: {res['unknown_labels']}")

    out = dict(coefficients_file=os.path.abspath(args.coefficients), results=results)
    if args.validation_out:
        with open(args.validation_out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\n  wrote {args.validation_out}")
    return out


def _energy_from_cli(args, name):
    """Build a minimal energy summary from CLI scalars when no run dir is given."""
    if "native" in name:
        return dict(E_dyn_J=args.native_e_dyn, train_s=args.native_train_s,
                    gpus_active=args.native_gpus, n_measured_epochs=None,
                    throughput_img_s=None, idle_W=None, steady_W=None)
    return dict(E_dyn_J=args.pytorch_e_dyn, train_s=args.pytorch_train_s,
                gpus_active=args.pytorch_gpus, n_measured_epochs=None,
                throughput_img_s=None, idle_W=None, steady_W=None)


# ── CLI ─────────────────────────────────────────────────────────────────────
def build_parser():
    ap = argparse.ArgumentParser(
        description="Component-2 energy-coefficient extract + validate pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    # extract
    ex = sub.add_parser("extract", help="extract per-op energy coefficients")
    ex.add_argument("--components-dir", default=_p("power_analysis", "results", "components"),
                    help="dir containing one measure_power.sh run dir per component "
                         "(auto-discovered by name: conv_fp32/conv_bf16/bn_relu/allreduce)")
    ex.add_argument("--run", action="append", metavar="KEY=DIR",
                    help="explicit component run dir, e.g. e_mfma_fp32=/path/run "
                         "(repeatable; overrides discovery)")
    ex.add_argument("--coefficients", default=_p("power_analysis", "coefficients.json"),
                    help="output coefficients JSON")
    ex.set_defaults(func=stage_extract)

    # validate
    va = sub.add_parser("validate", help="compose coefficients -> predict full-training energy")
    va.add_argument("--coefficients", default=_p("power_analysis", "coefficients.json"))
    va.add_argument("--native-pmc", default=DEFAULT_NATIVE_PMC,
                    help="native full-training roofline_summary.csv (or run dir)")
    va.add_argument("--pytorch-pmc", default=DEFAULT_PT_PMC,
                    help="pytorch full-training roofline_summary.csv (or run dir)")
    va.add_argument("--native-energy", default=None,
                    help="run dir with run_metrics.csv for the native energy run")
    va.add_argument("--pytorch-energy", default=None,
                    help="run dir with run_metrics.csv for the pytorch energy run")
    # scalar energy fallbacks (when no run dir is available)
    va.add_argument("--native-e-dyn", type=float, default=None, help="native E_dyn_J (J)")
    va.add_argument("--native-train-s", type=float, default=None, help="native train_s (s)")
    va.add_argument("--pytorch-e-dyn", type=float, default=None, help="pytorch E_dyn_J (J)")
    va.add_argument("--pytorch-train-s", type=float, default=None, help="pytorch train_s (s)")
    # step / gpu overrides
    va.add_argument("--native-steps", type=float, default=None)
    va.add_argument("--pytorch-steps", type=float, default=None)
    va.add_argument("--native-gpus", type=float, default=None)
    va.add_argument("--pytorch-gpus", type=float, default=None)
    va.add_argument("--nnodes", type=int, default=1,
                    help="number of training nodes (1=intra-node e_comm; "
                         ">=2 uses e_comm_2node if available)")
    va.add_argument("--validation-out", default=_p("power_analysis", "validation.json"))
    va.set_defaults(func=stage_validate)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
