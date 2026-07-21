#!/usr/bin/env python3
# ============================================================================
# revalidate_fulldata.py -- Stage-2 re-validation of coefficients_final.json
# against the NEW full-ImageNet-1K runs (2026-07-01..03).
# ----------------------------------------------------------------------------
# Supersedes validation_2node_{strong,weak}_gpu*_r1.json, whose 2-node targets
# were 12-iter val-subset runs (window dominated by inter-epoch overhead at
# near-idle power -> the +102..110% "drift" was mostly a target artifact).
#
# Targets (all trained on the full 1.28M-image train split, verified via
# [PreScan] 1281167 / 312-iters-per-epoch in the run logs):
#   in-sample   : native FP32 1-node 8-GPU repeats
#                 mn_52980217/218/219/220 + mn_52981427   (3 measured epochs, n=5)
#                 mn_52991531/532/533/534                 (4 measured epochs, n=4)
#   cross-stack : PyTorch FP32 2-node 16-GPU powercap run 52942558, cap_750
#                 (other caps composed with the SAME 750W coefficients and
#                  reported separately -- expected to drift: coefficients are
#                  cap-specific and per-cap sets were not extracted)
#   2-node nat. : mn_52941669 (16 GPU, BF16 through-mode) -- APPROXIMATE:
#                 priced by swapping e_mfma_fp32->e_mfma_bf16, e_gemm->
#                 e_gemm_bf16 on the FP32-profile MOP counts (same matrix-op
#                 count either precision); B_hbm comes from the FP32 profile
#                 and overstates BF16-through activation bytes, so the hbm
#                 term is an upper bound for this row.
#
# Uses compose_energy.py as a library (load_pmc / load_energy_summary /
# predict_energy) so the composition logic is identical to the canonical tool.
# Native run_metrics carry an empty throughput_img_s; it is filled here as
# n_measured_epochs * 1281167 / train_s so the comm term uses the real step
# time instead of the phantom V_ar/busy fallback.
#
# Extra diagnostic per target: u_est = (per-GPU img/s) / (saturated 1-GPU
# img/s), the duty proxy for the missing stall/wait term; for u_est < 0.95
# the implied stall power P_stall = (P_meas - u*P_pred) / (1-u) is printed --
# the in-situ calibration number for a future e_wait class.
#
# Run:  /usr/bin/python3.11 revalidate_fulldata.py
# Out:  validation_fulldata_20260703.json  (+ table on stdout)
# ============================================================================
import importlib.util
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

spec = importlib.util.spec_from_file_location(
    "compose_energy", os.path.join(HERE, "compose_energy.py"))
ce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ce)

IMAGES_PER_EPOCH = 1281167.0
SAT_IMG_S = {"native": 1250.0, "pytorch": 1516.0}   # measured 1-GPU saturated rates

NATIVE_PMC = os.path.join(ROOT, "profiles", "native_in1k_retry_52488168",
                          "roofline_summary.csv")
PT_PMC = os.path.join(ROOT, "profiles", "pt_amd_52438059", "roofline_summary.csv")
MN = os.path.join(HERE, "results", "native_multinode")
PT_CAP = os.path.join(HERE, "results", "pytorch",
                      "powercap_2node_fp32_52942558", "resnet50")

NATIVE_FP32_RUNS = ["mn_52980217", "mn_52980218", "mn_52980219", "mn_52980220",
                    "mn_52981427", "mn_52991531", "mn_52991532", "mn_52991533",
                    "mn_52991534"]
NATIVE_BF16_2NODE = "mn_52941669"


def fill_throughput(energy):
    if not energy.get("throughput_img_s"):
        ep, ts = energy.get("n_measured_epochs"), energy.get("train_s")
        if ep and ts:
            energy["throughput_img_s"] = ep * IMAGES_PER_EPOCH / ts
    return energy


def validate_one(coeffs, pmc_rows, run_dir, nnodes, stack):
    en = fill_throughput(ce.load_energy_summary(run_dir))
    res = ce.predict_energy(coeffs, pmc_rows, en, nnodes=nnodes)
    tput = en.get("throughput_img_s") or 0.0
    ngpu = res["ngpu"]
    u = (tput / ngpu) / SAT_IMG_S[stack] if tput else None
    p_pred, p_meas = res["P_dyn_pred_pergpu_W"], res["P_dyn_meas_pergpu_W"]
    stall = None
    if u is not None and u < 0.95 and p_meas is not None:
        stall = (p_meas - u * p_pred) / (1.0 - u)
    return dict(run=os.path.basename(run_dir), ngpu=ngpu, nnodes=nnodes,
                tput_img_s=tput, u_est=u,
                P_dyn_pred_W=p_pred, P_dyn_meas_W=p_meas,
                err_power_pct=res["err_power_pct"],
                err_total_pct=res["err_total_pct"],
                E_dyn_pred_J=res["E_dyn_pred_J"], E_dyn_meas_J=res["E_dyn_meas_J"],
                P_stall_implied_W=stall,
                pred_power_pergpu_W=res["pred_power_pergpu_W"])


def main():
    with open(os.path.join(HERE, "coefficients_final.json")) as fh:
        coeffs = json.load(fh)
    native_rows = ce.load_pmc(NATIVE_PMC)
    pt_rows = ce.load_pmc(PT_PMC)

    out = {"coefficients_file": "coefficients_final.json",
           "native_insample_fp32_8gpu": [], "pytorch_crossstack_2node": [],
           "native_2node_bf16_approx": None}

    hdr = (f"{'target':38s} {'u_est':>5s} {'Ppred':>6s} {'Pmeas':>6s} "
           f"{'errP%':>7s} {'errTot%':>8s} {'Pstall':>6s}")
    print(hdr)
    print("-" * len(hdr))

    def row(tag, r):
        f = lambda v, fm="%.1f": (fm % v) if isinstance(v, (int, float)) else "  n/a"
        print(f"{tag:38s} {f(r['u_est'],'%.2f'):>5s} {f(r['P_dyn_pred_W']):>6s} "
              f"{f(r['P_dyn_meas_W']):>6s} {f(r['err_power_pct']):>7s} "
              f"{f(r['err_total_pct']):>8s} {f(r['P_stall_implied_W'],'%.0f'):>6s}")

    # -- in-sample: native FP32 repeats ------------------------------------
    errs = []
    for name in NATIVE_FP32_RUNS:
        r = validate_one(coeffs, native_rows, os.path.join(MN, name), 1, "native")
        out["native_insample_fp32_8gpu"].append(r)
        errs.append(r["err_power_pct"])
        row(f"native FP32 8GPU {name}", r)
    n = len(errs)
    mean = sum(errs) / n
    sd = math.sqrt(sum((e - mean) ** 2 for e in errs) / (n - 1))
    out["native_insample_stats"] = dict(n=n, err_power_mean_pct=mean,
                                        err_power_std_pct=sd)
    print(f"{'  -> in-sample mean +/- std (n=%d)' % n:38s} "
          f"{'':>5s} {'':>6s} {'':>6s} {mean:6.1f}+/-{sd:.1f}")

    # -- cross-stack: PyTorch 2-node, all caps (750 is the matched one) ----
    for cap in sorted(os.listdir(PT_CAP)):
        d = os.path.join(PT_CAP, cap)
        if not (cap.startswith("cap_") and os.path.isdir(d)):
            continue
        r = validate_one(coeffs, pt_rows, d, 2, "pytorch")
        r["cap_W"] = int(cap.split("_")[1])
        r["cap_matched"] = (r["cap_W"] == 750)
        out["pytorch_crossstack_2node"].append(r)
        tag = f"pytorch FP32 2node {cap}" + (" [MATCHED]" if r["cap_matched"] else "")
        row(tag, r)

    # -- native 2-node BF16 through-mode (approximate pricing) -------------
    cb = dict(coeffs)
    cb["e_mfma_fp32"] = coeffs["e_mfma_bf16"]
    cb["e_gemm"] = coeffs.get("e_gemm_bf16") or 4.55e-12
    r = validate_one(cb, native_rows, os.path.join(MN, NATIVE_BF16_2NODE), 2, "native")
    r["note"] = ("BF16 through-mode priced with e_mfma_bf16/e_gemm_bf16 on FP32-"
                 "profile MOP counts; B_hbm from FP32 profile = upper bound")
    out["native_2node_bf16_approx"] = r
    row(f"native BF16 2node {NATIVE_BF16_2NODE} ~", r)

    dst = os.path.join(HERE, "validation_fulldata_20260703.json")
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
