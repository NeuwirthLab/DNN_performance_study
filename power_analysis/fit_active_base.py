#!/usr/bin/env python3
# ============================================================================
# fit_active_base.py -- add an "active-base power" term to the component-2
#                       energy model and re-measure prediction error.
# ----------------------------------------------------------------------------
# WHY. coefficients_final.json prices dynamic power as compute only:
#     P_dyn_pred = e_mfma*F_mfma + e_gemm*F_gemm + e_hbm*B_hbm + e_comm*V_ar
# and adds a fixed P_idle (141 W) floor. Against the new full-data native runs
# it under-predicts steady dynamic power by -13.7 % +/- 2.3 % (n=9), and the
# gap GROWS the harder the chip is driven (uncapped -13.7 %, capped-8GPU -8 %).
# That residual is the "active-but-not-compute-attributable" power: CUs powered
# up, clock trees, L2, memory controllers -- present whenever kernels run but
# not carried by any per-FLOP / per-byte class.
#
# WHAT. Fit one constant P_active = mean(P_dyn_meas - P_dyn_pred) over the
# in-sample native runs (least-squares intercept; the component prediction is
# ~constant across these runs so the LSQ intercept == mean residual). Report
# in-sample error with the term, then CROSS-VALIDATE on other regimes to show
# where the constant is valid and where it is not.
#
# SCOPE (honest). P_active is calibrated at the NATIVE, UNCAPPED (~720 W/GPU)
# operating point. It is operating-point dependent: under a power cap the
# active base shrinks (capped-8GPU flips to +6.7 %), and the PyTorch stack sits
# at a different point (adding it there hurts). Do NOT apply this constant to
# capped or PyTorch runs. The physically-correct generalization is P_active(cap)
# -- blocked today because PMC counters were never co-collected at >1 cap
# (see project_dvfs_fit_status), so a single uncapped constant is the most we
# can fit rigorously. Cross-check: P_idle + P_active = 217 W independently
# matches the audit's P_stall ~197-216 W (project_energy_model_audit).
#
# Does NOT overwrite coefficients_final.json. Writes coefficients_active_base.json.
# Run:  /usr/bin/python3.11 fit_active_base.py [--write]
# ============================================================================
import argparse
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
COEF = os.path.join(HERE, "coefficients_final.json")
VALID = os.path.join(HERE, "validation_fulldata_20260703.json")
OUT = os.path.join(HERE, "coefficients_active_base.json")


def fit_p_active(insample_rows):
    """LSQ intercept on dynamic power = mean(meas - pred)."""
    res = [r["P_dyn_meas_W"] - r["P_dyn_pred_W"] for r in insample_rows]
    return st.mean(res), res


def err_pct(pred_dyn, meas_dyn, p_active, p_idle):
    e_pow = 100.0 * (pred_dyn + p_active - meas_dyn) / meas_dyn
    e_eng = 100.0 * ((p_idle + pred_dyn + p_active) - (p_idle + meas_dyn)) \
        / (p_idle + meas_dyn)
    return e_pow, e_eng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write coefficients_active_base.json")
    args = ap.parse_args()

    coef = json.load(open(COEF))
    vj = json.load(open(VALID))
    p_idle = coef["P_idle_W"]
    insample = vj["native_insample_fp32_8gpu"]

    p_active, res = fit_p_active(insample)
    print(f"in-sample native 8-GPU uncapped  n={len(insample)}")
    print(f"  residual dynamic W : min={min(res):.1f}  mean={p_active:.1f}  "
          f"max={max(res):.1f}  std={st.stdev(res):.1f}")
    print(f"  fitted P_active    : {p_active:.1f} W")
    print(f"  active floor       : P_idle + P_active = {p_idle + p_active:.1f} W "
          f"(audit P_stall ~197-216 W)\n")

    # in-sample before/after
    ep0 = [r["err_power_pct"] for r in insample]
    ee0 = [r["err_total_pct"] for r in insample]
    ep1, ee1 = [], []
    for r in insample:
        a, b = err_pct(r["P_dyn_pred_W"], r["P_dyn_meas_W"], p_active, p_idle)
        ep1.append(a); ee1.append(b)
    print("IN-SAMPLE (native 8-GPU uncapped):")
    print(f"  BEFORE power {st.mean(ep0):+.1f}+/-{st.stdev(ep0):.1f}%  "
          f"energy {st.mean(ee0):+.1f}+/-{st.stdev(ee0):.1f}%")
    print(f"  AFTER  power {st.mean(ep1):+.1f}+/-{st.stdev(ep1):.1f}%  "
          f"energy {st.mean(ee1):+.1f}+/-{st.stdev(ee1):.1f}%\n")

    # cross-validation table (pred_dyn, meas_dyn, regime)
    cf = coef
    checks = [
        ("native 1-GPU uncapped", 475.0, 549.2, "native-uncapped"),
        ("native 8-GPU CAP 52476068",
         cf["validation_8gpu"]["native_8gpu"]["P_pred_W"],
         cf["validation_8gpu"]["native_8gpu"]["P_meas_W"], "native-CAPPED"),
        ("pytorch 1-GPU uncapped",
         cf["validation_1gpu"]["pytorch_cross_stack"]["P_pred_W"],
         cf["validation_1gpu"]["pytorch_cross_stack"]["P_meas_W"], "pytorch"),
        ("pytorch 8-GPU CAP 52476069",
         cf["validation_8gpu"]["pytorch_8gpu"]["P_pred_W"],
         cf["validation_8gpu"]["pytorch_8gpu"]["P_meas_W"], "pytorch-CAPPED"),
    ]
    print(f"CROSS-VALIDATION  {'regime':16} {'errBEFORE':>9} {'errAFTER':>9}")
    for name, pred, meas, regime in checks:
        e0 = 100.0 * (pred - meas) / meas
        e1, _ = err_pct(pred, meas, p_active, p_idle)
        note = "APPLY" if regime == "native-uncapped" else "do NOT apply"
        print(f"  {name:30} {regime:16} {e0:+8.1f}% {e1:+8.1f}%   {note}")

    out = dict(coef)
    out["P_active_W"] = round(p_active, 2)
    out["P_active_note"] = (
        "Active-but-not-compute-attributable power, added to predicted DYNAMIC "
        "power (on top of the per-FLOP/byte classes). Fit as mean(P_dyn_meas - "
        "P_dyn_pred) over native 8-GPU uncapped full-data runs "
        "(validation_fulldata_20260703.json, n=9). SCOPE: native, UNCAPPED "
        "(~720 W/GPU) only. Do NOT apply to capped or PyTorch runs (active base "
        "is operating-point dependent; capped-8GPU flips to +6.7%). "
        "P_idle+P_active=217 W matches audit P_stall ~197-216 W. Generalization "
        "to P_active(cap) blocked: PMC not co-collected at >1 cap.")
    out["P_active_provenance"] = ("fit_active_base.py on "
                                  "validation_fulldata_20260703.json in-sample n=9")
    if args.write:
        json.dump(out, open(OUT, "w"), indent=2)
        print(f"\nwrote {OUT}")
    else:
        print("\n(dry run; pass --write to emit coefficients_active_base.json)")


if __name__ == "__main__":
    main()
