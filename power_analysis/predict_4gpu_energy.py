#!/usr/bin/env python3
# ============================================================================
# predict_4gpu_energy.py -- energy-model prediction vs measurement for the
# 2026-07-06 4-GPU FP32 energy runs (jobs 53061751/753/754/755).
#
# Uses compose_energy.predict_energy (the canonical Stage-2 composition) with
# coefficients_final.json and the SAME canonical per-GPU PMC profiles the paper
# reuses at other GPU counts:
#     native  -> profiles/native_in1k_retry_52488168/roofline_summary.csv
#     pytorch -> profiles/pt_amd_52438059/roofline_summary.csv
# Method is identical to revalidate_fulldata.py: reuse the per-GPU work-rates,
# fill throughput from the measured 3-epoch window so the comm term uses the
# real step time. WEAK runs (256/GPU) match the profile batch exactly. STRONG
# runs (64/GPU) reuse the 256/GPU compute profile -> flagged approximate; the
# model has no small-batch stall term (documented limitation), so strong rows
# are expected to under-predict a touch more.
#
# Out: validation_4gpu_20260706.json  (+ table on stdout)
# ============================================================================
import importlib.util, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("compose_energy",
        os.path.join(HERE, "compose_energy.py"))
ce = importlib.util.module_from_spec(spec); spec.loader.exec_module(ce)

COEFFS = json.load(open(os.path.join(HERE, "coefficients_final.json")))
NATIVE_PMC = ce.load_pmc(os.path.join(ROOT, "profiles",
                         "native_in1k_retry_52488168", "roofline_summary.csv"))
PT_PMC = ce.load_pmc(os.path.join(ROOT, "profiles",
                     "pt_amd_52438059", "roofline_summary.csv"))

IMAGES = 3843072.0   # global_batch * steps * 3 measured epochs (same for all)

# measured 3-epoch-window energy (from run_metrics.csv) for each config
RUNS = [
 dict(impl="HIP-native", stack="native", mode="weak",  run="53061751",
      E_dyn=1.78155e6, E_train=2.22375e6, train_s=805.663, pmc=NATIVE_PMC),
 dict(impl="HIP-native", stack="native", mode="strong", run="53061753",
      E_dyn=1.88521e6, E_train=2.37724e6, train_s=884.494, pmc=NATIVE_PMC),
 dict(impl="PyTorch-ROCm-DDP", stack="pytorch", mode="weak",  run="53061754",
      E_dyn=1.59234e6, E_train=1.98037e6, train_s=689.006, pmc=PT_PMC),
 dict(impl="PyTorch-ROCm-DDP", stack="pytorch", mode="strong", run="53061755",
      E_dyn=1.66125e6, E_train=2.06667e6, train_s=730.189, pmc=PT_PMC),
]

out = {"coefficients_file": os.path.join(HERE, "coefficients_final.json"),
       "images_measured": IMAGES, "ngpu": 4, "nnodes": 1, "results": {}}
rows = []
for r in RUNS:
    tput = IMAGES / r["train_s"]                         # measured steady img/s (aggregate, 4 GPU)
    energy = dict(E_dyn_J=r["E_dyn"], train_s=r["train_s"],
                  gpus_active=4, throughput_img_s=tput)
    p = ce.predict_energy(COEFFS, r["pmc"], energy, ngpu=4, nnodes=1)
    # measured total = actual measured train energy (incl. real idle floor)
    jimg_pred  = p["E_total_pred_J"] / IMAGES
    jimg_meas  = r["E_train"] / IMAGES
    err_jimg   = 100.0 * (jimg_pred - jimg_meas) / jimg_meas
    rec = dict(run_id=r["run"], mode=r["mode"],
               throughput_img_s=round(tput, 1),
               P_dyn_pred_pergpu_W=round(p["P_dyn_pred_pergpu_W"], 1),
               P_dyn_meas_pergpu_W=round(p["P_dyn_meas_pergpu_W"], 1),
               err_power_pct=round(p["err_power_pct"], 1),
               E_total_pred_J=round(p["E_total_pred_J"]),
               E_total_meas_J=round(r["E_train"]),
               j_per_image_pred=round(jimg_pred, 4),
               j_per_image_meas=round(jimg_meas, 4),
               err_jperimg_pct=round(err_jimg, 1),
               pred_breakdown={k: round(v, 1) for k, v in p["pred_power_pergpu_W"].items()})
    out["results"][f"{r['stack']}_{r['mode']}"] = rec
    rows.append((r["impl"], r["mode"], rec))

with open(os.path.join(HERE, "validation_4gpu_20260706.json"), "w") as fh:
    json.dump(out, fh, indent=2)

print(f"{'config':26s}{'P_pred':>8}{'P_meas':>8}{'errP%':>7}   "
      f"{'J/img_pred':>11}{'J/img_meas':>11}{'errJ%':>7}")
for impl, mode, rec in rows:
    print(f"{impl+' '+mode:26s}{rec['P_dyn_pred_pergpu_W']:>8}{rec['P_dyn_meas_pergpu_W']:>8}"
          f"{rec['err_power_pct']:>7}   {rec['j_per_image_pred']:>11}{rec['j_per_image_meas']:>11}"
          f"{rec['err_jperimg_pct']:>7}")
print("\nwrote validation_4gpu_20260706.json")
