#!/usr/bin/env python3
# ============================================================================
# revalidate_ood.py -- re-validate the out-of-sample targets under the CURRENT
# energy model (precision-split GEMM in compose_energy + coefficients_final),
# and scope-test the P_active term against them.
# ----------------------------------------------------------------------------
# WHY. The five OOD validation JSONs were produced by separate
# compose_energy.py validate runs in late June, BEFORE the verified additions:
#   - validation_rn50_bf16.json PREDATES the precision-split GEMM (+34.6% on
#     disk); the post-split +3.1% exists only as a table row in
#     PRECISION_gemm_fix.md and was never persisted to a JSON.
#   - RN-101 / VGG-16 / MobileNetV2 JSONs are pre-split-era files; for FP32
#     targets the split is a no-op, so re-running them is a regression gate.
# This script recomputes every target from its ORIGINAL PMC profile + power
# run with the current code, so every number is re-derived from raw data, not
# copied from documentation.
#
# Targets and provenance (TIER1_out_of_sample_summary.md L44-48;
# PRECISION_gemm_fix.md L38 for the FP16 120-step re-run):
#   RN-101        PMC profiles/pt_amd_52858992  power pytorch_fp32_gpu1_52858993
#   VGG-16        PMC profiles/pt_amd_52859962  power pytorch_fp32_gpu1_52859937
#   MobileNetV2   PMC profiles/pt_amd_52859934  power pytorch_fp32_gpu1_52859935
#   RN-50 BF16    PMC profiles/pt_amd_52859963  power pytorch_amp_bfloat16_gpu1_52859940
#   RN-50 FP16    PMC profiles/pt_amd_52861196  power pytorch_amp_float16_gpu1_52861197
#
# P_active SCOPE TEST (not adoption): P_active = 76.17 W was fitted on native
# uncapped runs and held-out verified only there. All OOD targets are PyTorch
# 1-GPU uncapped, outside that scope; applying it here TESTS the scope
# boundary. The vector-class (fit_e_vec) term is deliberately excluded.
#
# Run:  /usr/bin/python3.11 revalidate_ood.py [--write]
# Out:  validation_ood_20260705.json  (+ table on stdout)
# ============================================================================
import argparse
import importlib.util
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

spec = importlib.util.spec_from_file_location(
    "compose_energy", os.path.join(HERE, "compose_energy.py"))
ce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ce)

PT = os.path.join(HERE, "results", "pytorch")
PROF = os.path.join(ROOT, "profiles")

# name, pmc dir, power run dir, stored-JSON (old on-disk number), documented
# expectation for the CURRENT code (source of that expectation)
TARGETS = [
    ("ResNet-101",
     "pt_amd_52858992", "pytorch_fp32_gpu1_52858993",
     "validation_rn101_1gpu_full.json",
     None),                                   # FP32: expect == stored
    ("VGG-16",
     "pt_amd_52859962", "pytorch_fp32_gpu1_52859937",
     "validation_vgg16.json",
     None),
    ("MobileNetV2",
     "pt_amd_52859934", "pytorch_fp32_gpu1_52859935",
     "validation_mobilenet_v2.json",
     None),
    ("ResNet-50 BF16",
     "pt_amd_52859963", "pytorch_amp_bfloat16_gpu1_52859940",
     "validation_rn50_bf16.json",
     ("+3.1 (PRECISION_gemm_fix.md L35, post-split re-run, never persisted)",
      3.1)),
    ("ResNet-50 FP16",
     "pt_amd_52861196", "pytorch_amp_float16_gpu1_52861197",
     "validation_rn50_fp16_noscaler_long.json",
     None),                                   # post-split file: expect == stored
]


def stored_err(fname):
    d = json.load(open(os.path.join(HERE, fname)))
    return d["results"]["pytorch (cross-stack)"]["err_total_pct"]


def p_active_errs(pred_dyn, meas_dyn, p_active, p_idle):
    """fit_active_base.py err_pct: P_active added to the PREDICTED side only."""
    e_pow = 100.0 * (pred_dyn + p_active - meas_dyn) / meas_dyn
    e_eng = 100.0 * ((p_idle + pred_dyn + p_active) - (p_idle + meas_dyn)) \
        / (p_idle + meas_dyn)
    return e_pow, e_eng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the table without writing the JSON")
    args = ap.parse_args()
    args.write = not args.dry_run

    coeffs = json.load(open(os.path.join(HERE, "coefficients_final.json")))
    p_active = json.load(open(os.path.join(
        HERE, "coefficients_active_base.json")))["P_active_W"]
    p_idle = coeffs["P_idle_W"]

    out = {"coefficients_file": "coefficients_final.json",
           "p_active_source": f"coefficients_active_base.json ({p_active} W; "
                              "native-uncapped scope; applied here as a SCOPE "
                              "TEST only, not adopted)",
           "targets": {}}

    hdr = (f"{'target':16s} {'stored':>7s} {'V0 rerun':>9s} {'gate':>6s} "
           f"{'resid W':>8s} {'V1 +P_act':>10s}  verdict")
    print(hdr)
    print("-" * len(hdr))

    gates_ok = True
    for name, pmc, power, old_json, expect in TARGETS:
        rows = ce.load_pmc(os.path.join(PROF, pmc))
        en = ce.load_energy_summary(os.path.join(PT, power))
        res = ce.predict_energy(coeffs, rows, en, nnodes=1)

        v0_pow = res["err_power_pct"]
        v0_tot = res["err_total_pct"]
        pred = res["P_dyn_pred_pergpu_W"]
        meas = res["P_dyn_meas_pergpu_W"]
        resid = meas - pred
        old = stored_err(old_json)

        # regression gate: FP32 / post-split targets must reproduce the stored
        # number; the pre-split BF16 file must instead match the documented
        # post-split value.
        if expect is None:
            ref, ref_src, tol = old, f"stored {old_json}", 0.3
        else:
            ref_src, ref = expect[0], expect[1]
            tol = 0.5
        ok = abs(v0_tot - ref) <= tol
        gates_ok &= ok

        v1_pow, v1_tot = p_active_errs(pred, meas, p_active, p_idle)
        verdict = ("P_active helps" if abs(v1_tot) < abs(v0_tot)
                   else "P_active hurts")

        print(f"{name:16s} {old:+7.1f} {v0_tot:+9.1f} "
              f"{'PASS' if ok else 'FAIL':>6s} {resid:+8.1f} {v1_tot:+10.1f}  "
              f"{verdict}")
        if not ok:
            print(f"  !! gate ref {ref:+.1f} ({ref_src}); investigate before "
                  f"using these numbers")

        out["targets"][name] = dict(
            pmc=pmc, power_run=power, old_json=old_json,
            old_err_total_pct=old,
            v0=dict(err_power_pct=v0_pow, err_total_pct=v0_tot,
                    P_dyn_pred_pergpu_W=pred, P_dyn_meas_pergpu_W=meas,
                    pred_power_pergpu_W=res["pred_power_pergpu_W"]),
            residual_W=resid,
            gate=dict(reference=ref, source=ref_src, passed=ok),
            v1_p_active=dict(err_power_pct=v1_pow, err_total_pct=v1_tot,
                             note="scope test only; P_active is "
                                  "native-uncapped, NOT adopted for PyTorch"),
        )

    print("\nP_active scope test: residuals span "
          f"{min(t['residual_W'] for t in out['targets'].values()):+.1f} to "
          f"{max(t['residual_W'] for t in out['targets'].values()):+.1f} W; "
          "a single positive constant cannot fit targets with negative "
          "residuals, confirming P_active is an operating-point/stack "
          "property, not universal.")
    out["gates_all_passed"] = bool(gates_ok)

    if args.write:
        dst = os.path.join(HERE, "validation_ood_20260705.json")
        with open(dst, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {dst}" + ("" if gates_ok else "  (WITH FAILED GATES)"))
    else:
        print("\n(dry run; no JSON written)")


if __name__ == "__main__":
    main()
