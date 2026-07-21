#!/usr/bin/env python3
# ============================================================================
# apply_evec.py -- VALU-mode (vector-FLOP class) post-hoc energy correction.
#
# SEPARATE FROM THE MAIN MODEL BY DESIGN. compose_energy.py::predict_energy()
# prices only the matrix-engine (MFMA/GEMM), HBM, and comm classes. For
# networks whose convolutions lower to non-MFMA (Winograd/direct) kernels
# (MobileNet / EfficientNet family), that omits the depthwise vector-ALU work
# and underprices them (the MobileNetV2 -30% boundary). This script adds the
# vector class e_vec * F_vec on top of an already-composed baseline prediction,
# WITHOUT touching compose_energy.py or coefficients_final.json.
#
# It must NOT be applied to matrix-engine-dominated networks (ResNet/VGG):
# their vector/BN work is already absorbed into the hybrid e_hbm. Guarded below.
#
# INPUTS
#   --baseline   validation JSON from `compose_energy.py validate` (reads the
#                "pytorch (cross-stack)" result: P_dyn_pred, P_dyn_meas, steps,
#                t_step_s, E_dyn_meas, E_idle, E_total_meas)
#   --coeffs     coefficients_vec.json (reads e_vec [J/FLOP])
#   F_vec, one of:
#     --f-vec <FLOP/window>              measured vector FLOPs over the power window
#     --profile <root> --fmfma-per-step <F> --steps <n>
#                                        compute F_vec via compute_fvec.py's method
#
# OUTPUT (--out)  a JSON mirroring validation_evec_heldout_*.json: baseline vs
#                 with_e_vec power/energy errors.
#
# Run (held-out mobilenet_v3_large, reproduces validation_evec_heldout_mnv3):
#   python3.11 apply_evec.py \
#     --baseline validation_mobilenet_v3_large_20260707.json \
#     --coeffs   coefficients_vec.json \
#     --profile  ../profiles/pt_amd_53074001 --fmfma-per-step 159276664382.439 \
#     --steps 41 --network mobilenet_v3_large \
#     --out validation_evec_heldout_mnv3_20260707.json
# ============================================================================
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def f_vec_from_profile(profile_root, fmfma_per_step, steps):
    """Reuse compute_fvec.py (single source of truth for Eq. fvec + scaling)."""
    out = subprocess.check_output(
        [sys.executable, os.path.join(HERE, "compute_fvec.py"),
         profile_root, str(fmfma_per_step), str(steps)],
        text=True)
    for line in out.splitlines():
        if "--f-vec" in line:
            return float(line.split("--f-vec")[1].strip())
    raise SystemExit("compute_fvec.py produced no --f-vec value:\n" + out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--coeffs", default=os.path.join(HERE, "coefficients_vec.json"))
    ap.add_argument("--f-vec", type=float, default=None)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--fmfma-per-step", type=float, default=None)
    ap.add_argument("--steps", type=float, default=None)
    ap.add_argument("--network", default="unknown")
    ap.add_argument("--result-key", default="pytorch (cross-stack)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force-dense", action="store_true",
                    help="override the dense-network guard (NOT recommended)")
    args = ap.parse_args()

    base = json.load(open(args.baseline))["results"][args.result_key]
    e_vec = json.load(open(args.coeffs))["e_vec"]        # J/FLOP

    # Dense-network guard: e_vec is scoped to non-MFMA-conv (depthwise) nets.
    dense_markers = ("resnet", "vgg", "rn50", "rn101")
    if any(m in args.network.lower() for m in dense_markers) and not args.force_dense:
        raise SystemExit(
            f"REFUSED: '{args.network}' looks matrix-engine-dominated. e_vec is "
            "scoped to depthwise/non-MFMA-conv networks; applying it to a dense "
            "net double-counts vector work already in e_hbm. Use --force-dense "
            "only for a deliberate diagnostic.")

    if args.f_vec is None:
        if not (args.profile and args.fmfma_per_step and args.steps):
            raise SystemExit("give --f-vec, or --profile + --fmfma-per-step + --steps")
        args.f_vec = f_vec_from_profile(args.profile, args.fmfma_per_step, args.steps)

    window = base["steps"] * base["t_step_s"]            # power-run window (s)
    P_vec = e_vec * args.f_vec / window                  # added per-GPU dyn W

    P_pred0 = base["P_dyn_pred_pergpu_W"]
    P_meas = base["P_dyn_meas_pergpu_W"]
    P_pred1 = P_pred0 + P_vec

    # dynamic energy scales with predicted power over the same GPU-seconds;
    # total adds the unchanged idle floor.
    t_gpus = base["E_dyn_meas_J"] / P_meas               # GPU-seconds in window basis
    E_dyn_pred1 = P_pred1 * t_gpus
    E_tot_pred1 = E_dyn_pred1 + base["E_idle_J"]
    E_tot_meas = base["E_total_meas_J"]

    def err(pred, meas):
        return 100.0 * (pred / meas - 1.0)

    out = {
        "held_out_network": args.network,
        "no_refit": True,
        "e_vec_J_per_FLOP": e_vec,
        "e_vec_source": os.path.basename(args.coeffs),
        "baseline_json": os.path.basename(args.baseline),
        "F_vec_window_FLOP": args.f_vec,
        "P_vec_W": P_vec,
        "baseline": {
            "P_dyn_pred_W": P_pred0,
            "err_power_pct": err(P_pred0, P_meas),
            "err_total_pct": base["err_total_pct"],
        },
        "with_e_vec": {
            "P_dyn_pred_W": P_pred1,
            "err_power_pct": err(P_pred1, P_meas),
            "err_total_pct": err(E_tot_pred1, E_tot_meas),
        },
        "P_dyn_meas_W": P_meas,
    }

    print(f"{args.network}: e_vec={e_vec*1e12:.1f} pJ/FLOP  F_vec={args.f_vec:.3e}  "
          f"P_vec={P_vec:.1f} W")
    print(f"  power err : {out['baseline']['err_power_pct']:+.1f}% -> "
          f"{out['with_e_vec']['err_power_pct']:+.1f}%")
    print(f"  total err : {out['baseline']['err_total_pct']:+.1f}% -> "
          f"{out['with_e_vec']['err_total_pct']:+.1f}%")

    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"  wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
