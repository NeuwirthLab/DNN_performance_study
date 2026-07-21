#!/usr/bin/env python3.11
"""
Fit the AccelWattch (MICRO 2021) DVFS power law to MI300X cap-sweep data:

    P(f) = beta*C * f^3 + tau * f + P_const          (cubic missing the quadratic)

Input : power_analysis/results/{native,pytorch}/power_sweep_all.csv
        columns used: cap_w, avg_power_w_per_gpu, mean_sm_clock_mhz
Output: form comparison (AIC), fitted coefficients with leave-one-cap-out
        spread, R^2, Pearson r, and P_const under three anchoring methods.

Two adaptations vs AccelWattch, in order of preference:

  1. PRIMARY  - constrained fit: P_const is FIXED to the directly measured
     idle floor (140 W/GPU) and only beta,tau are fit on active points.
     AccelWattch had to extrapolate f->0 to get Volta's 32.5 W; we measure
     the constant term directly, so we should use it as a constraint, not
     re-estimate it from data that can't see f<558 MHz.
  2. SENSITIVITY - idle-anchored: append the measured idle point
     (122 MHz, 140 W) and fit all three coefficients. The idle point has
     high leverage (it is far outside the active f-range), so this fit's
     R^2 is optimistic; report it only as a robustness check.
  3. DIAGNOSTIC - unconstrained active-only fit. A cap sweep CLAMPS power
     while holding clock at low caps, so the naive f->0 extrapolation is
     unphysical (negative tau, P_const ~780 W). Shown to document WHY the
     AccelWattch recipe cannot be applied verbatim to a cap sweep.

Form caveat (verified 2026-06-12): on the active points alone a plain
quadratic fits as well as the cubic-missing-quad (delta-AIC < 0.7). The
558-1332 MHz range cannot discriminate the form; the data is CONSISTENT
WITH the AccelWattch law but does not confirm it. Discrimination needs
lower-frequency points via direct clock control (see clock_probe.slurm).

Usage: /usr/bin/python3.11 power_analysis/fit_dvfs_cubic.py
"""
import csv, os, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
IDLE_F_GHZ = 0.122   # measured idle core clock
IDLE_P_W   = 140.0   # measured idle power per GPU (amd-smi)
CAP_FLOOR  = 450     # caps below ~400 W are silently floored -> exclude

def load(path):
    caps, f, p = [], [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            caps.append(float(r["cap_w"]))
            f.append(float(r["mean_sm_clock_mhz"]) / 1000.0)   # GHz
            p.append(float(r["avg_power_w_per_gpu"]))           # W/GPU
    return np.array(caps), np.array(f), np.array(p)

def lstsq_fit(A, p):
    coef, *_ = np.linalg.lstsq(A, p, rcond=None)
    pred = A @ coef
    rss = np.sum((p - pred) ** 2)
    r2 = 1 - rss / np.sum((p - p.mean()) ** 2)
    r = np.corrcoef(p, pred)[0, 1]
    n, k = len(p), A.shape[1]
    aic = n * np.log(rss / n) + 2 * k
    return coef, r2, r, aic

def fit_constrained(f, p):
    """Fix P_const = IDLE_P_W; fit beta,tau on active points only."""
    A = np.column_stack([f**3, f])
    coef, *_ = np.linalg.lstsq(A, p - IDLE_P_W, rcond=None)
    pred = A @ coef + IDLE_P_W
    r2 = 1 - np.sum((p - pred) ** 2) / np.sum((p - p.mean()) ** 2)
    return coef, r2

def loco_spread(f, p):
    """Leave-one-cap-out spread of the constrained coefficients."""
    betas, taus = [], []
    for i in range(len(f)):
        m = np.arange(len(f)) != i
        (b, t), _ = fit_constrained(f[m], p[m])
        betas.append(b); taus.append(t)
    return (min(betas), max(betas)), (min(taus), max(taus))

def report(name, path):
    caps, f, p = load(path)
    m = caps >= CAP_FLOOR
    fm, pm = f[m], p[m]
    print(f"\n===== {name}  ({len(f)} caps, {m.sum()} active >= {CAP_FLOOR} W, "
          f"{fm.min()*1e3:.0f}-{fm.max()*1e3:.0f} MHz, {pm.min():.0f}-{pm.max():.0f} W/GPU) =====")

    # form comparison on active points: can this f-range discriminate the law?
    print("  form comparison (active points):")
    for label, A in [("cubic-no-quad [f3,f,1]", np.column_stack([fm**3, fm, np.ones_like(fm)])),
                     ("quadratic     [f2,f,1]", np.column_stack([fm**2, fm, np.ones_like(fm)])),
                     ("linear           [f,1]", np.column_stack([fm, np.ones_like(fm)]))]:
        _, r2, _, aic = lstsq_fit(A, pm)
        print(f"    {label}: R2={r2:.4f}  AIC={aic:6.1f}")

    (beta, tau), r2 = fit_constrained(fm, pm)
    (blo, bhi), (tlo, thi) = loco_spread(fm, pm)
    print(f"  PRIMARY constrained (P_const={IDLE_P_W:.0f} W fixed):")
    print(f"    P = {beta:.1f} f^3 {tau:+.1f} f + {IDLE_P_W:.0f}   R2={r2:.4f}  tau>0: {tau > 0}")
    print(f"    leave-one-cap-out: beta in [{blo:.0f},{bhi:.0f}]  tau in [{tlo:.0f},{thi:.0f}]")

    fa, pa = np.append(fm, IDLE_F_GHZ), np.append(pm, IDLE_P_W)
    (a, c, d), r2a, ra, _ = lstsq_fit(np.column_stack([fa**3, fa, np.ones_like(fa)]), pa)
    print(f"  sensitivity idle-anchored   : P={a:.1f}f^3{c:+.1f}f{d:+.1f}  "
          f"R2={r2a:.4f} (leverage-inflated)  P_const={d:.0f}W")

    (a, c, d), r2u, _, _ = lstsq_fit(np.column_stack([fm**3, fm, np.ones_like(fm)]), pm)
    print(f"  diagnostic unconstrained    : P={a:.1f}f^3{c:+.1f}f{d:+.1f}  "
          f"R2={r2u:.4f}  P_const(f->0)={d:.0f}W [unphysical: cap sweep clamps P, not f]")

if __name__ == "__main__":
    report("NATIVE",  os.path.join(RESULTS, "native",  "power_sweep_all.csv"))
    report("PYTORCH", os.path.join(RESULTS, "pytorch", "power_sweep_all.csv"))
