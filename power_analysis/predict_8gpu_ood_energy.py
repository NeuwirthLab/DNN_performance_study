#!/usr/bin/env python3
# ============================================================================
# predict_8gpu_ood_energy.py -- energy-model prediction vs measurement for the
# 8-GPU (single-node) OUT-OF-DISTRIBUTION architecture runs:
#   ResNet-101, VGG-16, MobileNet (v3_large or v2).
#
# WHY. We had OOD energy validation only at 1-GPU (RN-101/VGG/MobileNet) and at
# 2-node/16-GPU (VGG/MobileNetV2). This closes the single-node 8-GPU point, the
# scale the paper's scaling study actually runs at. Each target is a fresh
# PyTorch-DDP FP32 run measured with measure_power.sh (1 warmup epoch skipped,
# 2 measured epochs averaged, HW energy counters).
#
# METHOD (identical to predict_4gpu_energy.py / revalidate_fulldata.py):
#   * per-GPU work-rates come from each arch's OWN canonical 1-GPU PMC profile
#     (mfma/gemm/hbm rates are per-GPU, data-parallel-invariant);
#   * throughput/train_s come from the measured 8-GPU window, so the comm term
#     uses the real per-step time;
#   * ce.predict_energy(..., ngpu=8, nnodes=1) composes with coefficients_final.
#
# CAVEAT (comm term): compose_energy hardcodes ResNet-50's 97.49 MB gradient
# buffer for the intra-node comm rate. For OOD archs the true gradient volume
# differs (RN-101 ~178 MB, VGG-16 ~528 MB, MNv3-L ~22 MB), but the comm term is
# ~1-3 W out of ~500 W, so the effect on the reported error is <0.5 pt. We keep
# the shared function for methodological consistency with the 1-/2-node OOD
# validations and record the per-arch gradient volume below for the record.
#
# For MobileNet (depthwise, non-MFMA conv) the base model structurally
# under-prices the vector-ALU work; this script ALSO emits an apply_evec-ready
# baseline JSON and (if the profile constants are known) runs the e_vec
# correction, reproducing the held-out methodology at 8 GPU.
#
# Run:  /usr/bin/python3.11 predict_8gpu_ood_energy.py [--write]
# Out:  validation_ood_8gpu_<found>.json  (+ table on stdout)
# ============================================================================
import argparse
import glob
import importlib.util
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

spec = importlib.util.spec_from_file_location(
    "compose_energy", os.path.join(HERE, "compose_energy.py"))
ce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ce)

COEFFS = json.load(open(os.path.join(HERE, "coefficients_final.json")))
PROF = os.path.join(ROOT, "profiles")
PT_RESULTS = os.path.join(HERE, "results", "pytorch")

# arch tag (from the trainer's "[ARCH] model = <arch>" line) -> target spec.
#   pmc      : canonical per-GPU PMC roofline_summary.csv for this arch
#   grad_mb  : FP32 gradient-buffer size (record-only; see comm caveat)
#   evec     : (fmfma_per_step, steps) profile constants for apply_evec, or None
#              for dense nets (e_vec must NOT be applied to matrix-engine nets).
TARGETS = {
    "resnet101": dict(
        name="ResNet-101",
        pmc=os.path.join(PROF, "pt_amd_52858992", "roofline_summary.csv"),
        grad_mb=178.0, evec=None),
    "vgg16": dict(
        name="VGG-16",
        pmc=os.path.join(PROF, "pt_amd_52859962", "roofline_summary.csv"),
        grad_mb=528.0, evec=None),
    "mobilenet_v3_large": dict(
        name="MobileNetV3-Large",
        pmc=os.path.join(PROF, "pt_amd_53074001", "roofline_summary.csv"),
        grad_mb=22.0,
        evec=dict(profile="pt_amd_53074001",
                  fmfma_per_step=159276664382.439, steps=41)),
    "mobilenet_v2": dict(
        name="MobileNetV2",
        pmc=os.path.join(PROF, "pt_amd_52859934", "roofline_summary.csv"),
        grad_mb=56.0,
        evec=dict(profile="pt_amd_52859934",
                  fmfma_per_step=None, steps=None)),  # fill if v2 chosen
}


def discover_runs(gpus=8):
    """Map arch-tag -> newest 8-GPU run dir, by reading [ARCH] from command.log."""
    found = {}
    pat = os.path.join(PT_RESULTS, f"pytorch_fp32_gpu{gpus}_*")
    for d in sorted(glob.glob(pat), key=os.path.getmtime):
        cl = os.path.join(d, "command.log")
        rm = os.path.join(d, "run_metrics.csv")
        if not (os.path.isfile(cl) and os.path.isfile(rm)):
            continue
        arch = None
        with open(cl, errors="ignore") as fh:
            for line in fh:
                if "[ARCH] model =" in line:
                    arch = line.split("=", 1)[1].strip()
                    break
        if arch in TARGETS:
            found[arch] = d          # newest wins (sorted by mtime)
    return found


def predict_run(arch, run_dir):
    spec = TARGETS[arch]
    pmc = ce.load_pmc(spec["pmc"])
    en = ce.load_energy_summary(run_dir)
    p = ce.predict_energy(COEFFS, pmc, en, ngpu=8, nnodes=1)
    images = None
    if en.get("throughput_img_s") and en.get("train_s"):
        images = en["throughput_img_s"] * en["train_s"]
    rec = dict(
        arch=arch, name=spec["name"], run_dir=os.path.basename(run_dir),
        ngpu=p["ngpu"], nnodes=1,
        train_s=en.get("train_s"),
        throughput_img_s=en.get("throughput_img_s"),
        n_measured_epochs=en.get("n_measured_epochs"),
        gpus_active=en.get("gpus_active"),
        images_measured=images,
        grad_buffer_mb=spec["grad_mb"],
        P_dyn_pred_pergpu_W=p["P_dyn_pred_pergpu_W"],
        P_dyn_meas_pergpu_W=p["P_dyn_meas_pergpu_W"],
        err_power_pct=p["err_power_pct"],
        E_dyn_pred_J=p["E_dyn_pred_J"], E_dyn_meas_J=p["E_dyn_meas_J"],
        E_total_pred_J=p["E_total_pred_J"], E_total_meas_J=p["E_total_meas_J"],
        err_total_pct=p["err_total_pct"],
        j_per_image_pred=(p["E_total_pred_J"] / images) if images else None,
        j_per_image_meas=(p["E_total_meas_J"] / images) if images else None,
        steps=p["steps"], t_step_s=p["t_step_s"],
        pred_breakdown_W={k: round(v, 2) for k, v in p["pred_power_pergpu_W"].items()},
        _predict_energy=p,   # kept for the apply_evec baseline emit
    )
    if rec["j_per_image_pred"] and rec["j_per_image_meas"]:
        rec["err_jperimg_pct"] = 100.0 * (rec["j_per_image_pred"]
                                          / rec["j_per_image_meas"] - 1.0)
    else:
        rec["err_jperimg_pct"] = rec["err_total_pct"]
    return rec


def run_evec(arch, rec):
    """Emit an apply_evec-compatible baseline JSON and run the correction."""
    spec = TARGETS[arch]["evec"]
    if not spec or not spec.get("fmfma_per_step") or not spec.get("steps"):
        return None
    base_json = os.path.join(HERE, f"_baseline_8gpu_{arch}.json")
    json.dump({"results": {"pytorch (cross-stack)": rec["_predict_energy"]}},
              open(base_json, "w"), indent=2)
    out_json = os.path.join(HERE, f"validation_evec_8gpu_{arch}.json")
    cmd = [sys.executable, os.path.join(HERE, "apply_evec.py"),
           "--baseline", base_json,
           "--coeffs", os.path.join(HERE, "coefficients_vec.json"),
           "--profile", os.path.join(PROF, spec["profile"]),
           "--fmfma-per-step", str(spec["fmfma_per_step"]),
           "--steps", str(spec["steps"]),
           "--network", arch, "--out", out_json]
    try:
        print("\n--- e_vec (VALU) correction for %s ---" % rec["name"])
        subprocess.check_call(cmd)
        return json.load(open(out_json))
    except Exception as e:                                    # noqa: BLE001
        print("  e_vec correction failed: %s" % e)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="persist the combined validation JSON")
    args = ap.parse_args()

    runs = discover_runs(8)
    if not runs:
        print("No 8-GPU OOD run dirs found under", PT_RESULTS)
        print("(expected pytorch_fp32_gpu8_* dirs with [ARCH] in command.log)")
        return

    hdr = (f"{'target':22s}{'img/s':>9}{'P_pred':>8}{'P_meas':>8}{'errP%':>7}"
           f"{'J/img_pr':>9}{'J/img_me':>9}{'errE%':>7}")
    print(hdr); print("-" * len(hdr))

    out = {"coefficients_file": "coefficients_final.json",
           "ngpu": 8, "nnodes": 1, "precision": "fp32",
           "basis": "measure_power.sh, 1 warmup epoch skipped, 2 measured epochs",
           "results": {}, "evec": {}}
    order = ["resnet101", "vgg16", "mobilenet_v3_large", "mobilenet_v2"]
    for arch in [a for a in order if a in runs]:
        rec = predict_run(arch, runs[arch])
        def f(v, w=8, p=1):
            return (f"%{w}.{p}f" % v) if isinstance(v, (int, float)) else " " * (w - 3) + "n/a"
        print(f"{rec['name']:22s}{f(rec['throughput_img_s'],9,0)}"
              f"{f(rec['P_dyn_pred_pergpu_W'])}{f(rec['P_dyn_meas_pergpu_W'])}"
              f"{f(rec['err_power_pct'],7)}"
              f"{f(rec['j_per_image_pred'],9,3)}{f(rec['j_per_image_meas'],9,3)}"
              f"{f(rec['err_jperimg_pct'],7)}")
        ev = run_evec(arch, rec)
        rec.pop("_predict_energy", None)
        out["results"][arch] = rec
        if ev:
            out["evec"][arch] = ev

    if args.write:
        tag = "_".join(a[:4] for a in out["results"])
        dst = os.path.join(HERE, f"validation_ood_8gpu.json")
        json.dump(out, open(dst, "w"), indent=2, default=str)
        print("\nwrote", dst)


if __name__ == "__main__":
    main()
