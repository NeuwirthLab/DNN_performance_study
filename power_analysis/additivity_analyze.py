#!/usr/bin/env python3
"""
additivity_analyze.py -- strict test of the energy-model additivity hypothesis.

Consumes the per-mode dynamic-energy runs produced by extract_coefficients.slurm
(RUNS="conv-fp32 bnrelu fc step") at one or more power caps, plus the existing
full-step PMC, and decides whether Eq. (7) holds per the pre-registered rule in
power_analysis/additivity_experiment.md.

Per cap c it computes, in Joules per step iteration:
    Delta      = e_step - (e_conv + e_bn + e_fc)              # co-location gap
    E_model    = e_mfma(c) F_mfma + e_gemm(c) F_gemm + e_hbm(c) B_hbm
    R_model    = e_step - E_model                             # clean in-sample residual
    E_other    = e_hbm(c) B_other                             # un-priced optimizer/memset
    R_add      = R_model - E_other                            # PURE additivity residual

Verdict (at 750 W): additivity ASSERTED if |R_add|/e_step <= THRESH and 0 in CI.

Usage:
    python3 additivity_analyze.py \
        --runs results/coeff/run_<j1> results/coeff/run_<j2> ... \
        --native-pmc profiles/native_in1k_retry_52488168/roofline_summary.csv \
        [--iters-per-run 1000] [--thresh 0.03] [--out verdict.json]

Notes / things to confirm against your data:
  * iters-per-run = CL_ITERS * (CL_BLOCKS - WARMUP_BLOCKS); default 200*5 = 1000.
  * E_dyn is read as the sum of the E_dyn_J column of run_metrics.csv (idle
    neighbours contribute ~0). If your metrics file uses another name/location,
    adjust read_e_dyn().
  * B_hbm from the per-kernel summary may undercount MIOpen BN (rocprofv3 faults
    on gfx942). Pass --b-hbm-bytes <B> to override with the device TCC_EA0 total
    used in coefficients_final.json provenance (R_hbm * t_step).
"""
import argparse, csv, glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compose_energy as ce  # classify_component, load_roofline_summary, FLOP_PER_MOP

MODES = ("conv-fp32", "bnrelu", "fc", "step")


# ---- energy side ----------------------------------------------------------
def read_e_dyn(mode_dir):
    """Aggregate dynamic energy (J) for one mode run. Sums the E_dyn_J column of
    run_metrics.csv (idle neighbours ~ 0). Returns None if not found."""
    cands = glob.glob(os.path.join(mode_dir, "**", "run_metrics.csv"), recursive=True) \
        or glob.glob(os.path.join(mode_dir, "**", "*metrics*.csv"), recursive=True)
    for path in cands:
        tot, seen = 0.0, False
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                v = r.get("E_dyn_J") or r.get("e_dyn_J") or r.get("E_dyn")
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    continue
                if f == f:               # not NaN
                    tot += max(f, 0.0); seen = True
        if seen:
            return tot
    return None


def collect(run_roots, iters_per_run):
    """-> {captag: {mode: [J_per_iter, ...]}} and {captag: coeff_path}."""
    eps, coeffs = {}, {}
    for root in run_roots:
        for capdir in sorted(glob.glob(os.path.join(root, "cap_*"))):
            tag = os.path.basename(capdir)
            eps.setdefault(tag, {m: [] for m in MODES})
            cpath = os.path.join(root, f"coefficients_{tag}.json")
            if os.path.isfile(cpath):
                coeffs[tag] = cpath
            for m in MODES:
                e = read_e_dyn(os.path.join(capdir, m))
                if e is not None:
                    eps[tag][m].append(e / iters_per_run)
                else:
                    print(f"  [warn] no E_dyn for {tag}/{m} under {root}")
    return eps, coeffs


# ---- work side (per step, from the existing PMC) --------------------------
def work_counts(pmc_path, b_hbm_override=None):
    rows = ce.load_roofline_summary(pmc_path)
    F_mfma = F_gemm = B_hbm = B_other = 0.0
    for r in rows:
        cls = ce.classify_component(r["label"], r["category"])
        disp = r["dispatches"]
        flop = r["mfma_fp32_mops"] * disp * ce.FLOP_PER_MOP
        byts = r["hbm_bytes_per_disp"] * disp
        if cls == "mfma":
            F_mfma += flop
        elif cls == "gemm":
            F_gemm += flop
        elif cls == "hbm":
            B_hbm += byts
        elif cls == "other":
            B_other += byts
    if b_hbm_override is not None:
        B_hbm = b_hbm_override
    return dict(F_mfma=F_mfma, F_gemm=F_gemm, B_hbm=B_hbm, B_other=B_other)


# ---- bootstrap ------------------------------------------------------------
def _resample(xs, picks):
    return sum(xs[i] for i in picks) / len(picks)


def bootstrap(eps_cap, coeff, work, n=10000, seed=12345):
    """Return mean and 95% CI for Delta, R_model, E_other, R_add, and ratios.
    Resamples replicate runs (paired across modes is not assumed; modes are
    resampled independently since they are separate runs)."""
    # tiny deterministic LCG so the script needs no Date/Random-restricted libs
    state = seed
    def rnd(mod):
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return state % mod

    e_mfma = coeff["e_mfma_fp32"]; e_gemm = coeff["e_gemm"]; e_hbm = coeff["e_hbm"]
    E_model0 = e_mfma * work["F_mfma"] + e_gemm * work["F_gemm"] + e_hbm * work["B_hbm"]
    E_other0 = e_hbm * work["B_other"]

    def one(sample):
        c, b, f, s = (sample[m] for m in MODES)
        delta = s - (c + b + f)
        r_model = s - E_model0
        r_add = r_model - E_other0
        return dict(e_step=s, Delta=delta, E_model=E_model0, R_model=r_model,
                    E_other=E_other0, R_add=r_add)

    point = one({m: sum(eps_cap[m]) / len(eps_cap[m]) for m in MODES})
    draws = {k: [] for k in ("Delta", "R_model", "E_other", "R_add", "e_step")}
    for _ in range(n):
        sample = {}
        for m in MODES:
            xs = eps_cap[m]
            picks = [rnd(len(xs)) for _ in xs]
            sample[m] = _resample(xs, picks)
        o = one(sample)
        for k in draws:
            draws[k].append(o[k])

    def ci(xs):
        xs = sorted(xs)
        lo = xs[int(0.025 * len(xs))]; hi = xs[int(0.975 * len(xs)) - 1]
        return lo, hi

    out = {"point": point, "ci": {}}
    for k in ("Delta", "R_model", "E_other", "R_add"):
        out["ci"][k] = ci(draws[k])
    # R_add as a fraction of e_step, with CI
    fr = [a / b for a, b in zip(draws["R_add"], draws["e_step"]) if b]
    out["R_add_frac_point"] = point["R_add"] / point["e_step"] if point["e_step"] else None
    out["R_add_frac_ci"] = ci(fr) if fr else None
    out["n_reps"] = {m: len(eps_cap[m]) for m in MODES}
    return out


# ---- main -----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run_<jobid> roots")
    ap.add_argument("--native-pmc", default=ce.DEFAULT_NATIVE_PMC)
    ap.add_argument("--iters-per-run", type=int, default=1000)
    ap.add_argument("--thresh", type=float, default=0.03)
    ap.add_argument("--b-hbm-bytes", type=float, default=None,
                    help="override B_hbm with device TCC_EA0 per-step total (bytes)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    work = work_counts(a.native_pmc, a.b_hbm_bytes)
    print(f"PMC per-step work: F_mfma={work['F_mfma']:.3e} F_gemm={work['F_gemm']:.3e} "
          f"B_hbm={work['B_hbm']:.3e} B_other={work['B_other']:.3e}")

    eps, coeffs = collect(a.runs, a.iters_per_run)
    report = {}
    for tag in sorted(eps):
        if tag not in coeffs:
            print(f"[skip] {tag}: no coefficients_{tag}.json found"); continue
        coeff = json.load(open(coeffs[tag]))
        nrep = min(len(eps[tag][m]) for m in MODES)
        if nrep < 2:
            print(f"[skip] {tag}: <2 replicate runs for some mode (have "
                  f"{ {m: len(eps[tag][m]) for m in MODES} }) -- CI not meaningful")
            continue
        res = bootstrap(eps[tag], coeff, work)
        report[tag] = res
        p, ci = res["point"], res["ci"]
        f = res["R_add_frac_point"]; fci = res["R_add_frac_ci"]
        print(f"\n=== {tag}  (n={res['n_reps']}) ===")
        print(f"  e_step      = {p['e_step']:.3f} J/step")
        print(f"  Delta       = {p['Delta']:.3f} J  CI[{ci['Delta'][0]:.3f},{ci['Delta'][1]:.3f}]")
        print(f"  R_model     = {p['R_model']:.3f} J  CI[{ci['R_model'][0]:.3f},{ci['R_model'][1]:.3f}]"
              f"  ({100*p['R_model']/p['e_step']:.1f}% of step)")
        print(f"  E_other_est = {p['E_other']:.3f} J  ({100*p['E_other']/p['e_step']:.1f}% of step)")
        print(f"  R_add       = {p['R_add']:.3f} J  CI[{ci['R_add'][0]:.3f},{ci['R_add'][1]:.3f}]")
        print(f"  R_add/e_step= {100*f:.2f}%  CI[{100*fci[0]:.2f}%,{100*fci[1]:.2f}%]")
        zero_in = ci["R_add"][0] <= 0.0 <= ci["R_add"][1]
        verdict = ("ADDITIVITY ASSERTED" if (abs(f) <= a.thresh and zero_in)
                   else "ADDITIVITY REJECTED (M1)" if not zero_in and abs(f) > a.thresh
                   else "INCONCLUSIVE (effect within noise but >thresh, or 0 in CI but >thresh)")
        print(f"  VERDICT     = {verdict}")

    # DVFS coupling: R_add(450) - R_add(750)
    caps = {t: report[t]["point"]["R_add"] for t in report}
    cap450 = next((t for t in caps if "450" in t), None)
    cap750 = next((t for t in caps if "750" in t or "default" in t), None)
    if cap450 and cap750:
        dR = caps[cap450] - caps[cap750]
        print(f"\nDVFS coupling  R_add({cap450}) - R_add({cap750}) = {dR:.3f} J "
              f"({'degrades' if dR > 0 else 'no degradation'} under binding cap)")
        report["dvfs_delta_R_add_J"] = dR

    if a.out:
        json.dump(report, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
