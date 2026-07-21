#!/usr/bin/env python3
# ============================================================================
# analyze_fp16_probe.py <run_dir> -- window power_samples.csv by the probe's
# [PHASE_*] markers and report per-precision dynamic-power-per-image, which (for
# identical FLOP/image across fp16/bf16) IS the per-FLOP energy ratio.
#
# Outputs the two answers:
#   e_fp16/e_bf16  = (P_dyn_fp16 / img_s_fp16) / (P_dyn_bf16 / img_s_bf16)
#                    < 1  => FP16 MFMA genuinely cheaper than the shared 3.426
#                            pJ/FLOP coefficient (confirms coefficient bias)
#   scaler effect  = P_dyn(fp16_scaler) vs P_dyn(fp16_noscaler)
#                    scaler >> noscaler => unscaled grads were underflowing
#                    (data-dependent power); ~equal => pure coefficient effect
# ============================================================================
import csv, os, re, sys, statistics as st

run = sys.argv[1]
CL  = os.path.join(run, "command.log")
PS  = os.path.join(run, "power_samples.csv")

# --- phase windows + img/s from the probe log ------------------------------
starts, ends, imgs = {}, {}, {}
for ln in open(CL, errors="ignore"):
    m = re.search(r"\[PHASE_START\]\s+(\S+)\s+([\d.]+)", ln)
    if m: starts[m.group(1)] = float(m.group(2))
    m = re.search(r"\[PHASE_END\]\s+(\S+)\s+([\d.]+)", ln)
    if m: ends[m.group(1)] = float(m.group(2))
    m = re.search(r"\[PHASE_RESULT\]\s+(\S+).*img_s=([\d.]+)", ln)
    if m: imgs[m.group(1)] = float(m.group(2))
phases = [p for p in starts if p in ends]
compute = [p for p in phases if p != "idle"]

# --- load samples ----------------------------------------------------------
rows = []
with open(PS) as fh:
    for r in csv.DictReader(fh):
        try:
            rows.append((float(r["timestamp_s"]), int(r["gpu"]),
                         float(r["socket_power"]), float(r["gfx_activity"]), r))
        except (ValueError, KeyError):
            continue

def win(gpu, a, b):
    return [(p, act, r) for (t, g, p, act, r) in rows if g == gpu and a <= t <= b]

# active GPU = highest mean power over the union of compute windows
gpus = sorted({g for _, g, *_ in rows})
def mean_pow_compute(g):
    vals = [p for ph in compute for (p, _, _) in win(g, starts[ph], ends[ph])]
    return st.mean(vals) if vals else 0.0
active = max(gpus, key=mean_pow_compute)

idle_samples = [p for (p, _, _) in win(active, starts["idle"], ends["idle"])]
idle_W = st.mean(idle_samples) if idle_samples else 137.0

# --- per-phase aggregation (average the ABBA repeats by base name) ---------
def base(name): return name.split("#")[0]
agg = {}
for ph in compute:
    s = win(active, starts[ph], ends[ph])
    if not s: continue
    mp = st.mean(p for p, _, _ in s)
    ma = st.mean(a for _, a, _ in s)
    clk_col = f"gfx_{active}_clk"
    clk = [float(r[clk_col]) for _, _, r in s if r.get(clk_col, "N/A") not in ("N/A", "")]
    agg.setdefault(base(ph), []).append(
        dict(P=mp, Pdyn=mp - idle_W, act=ma,
             clk=(st.mean(clk) if clk else float("nan")), img_s=imgs.get(ph, float("nan"))))

def m(name, k):
    return st.mean(d[k] for d in agg[name]) if name in agg else float("nan")

print(f"\n=== FP16 vs BF16 per-FLOP energy probe (PyTorch, 1 GPU, active idx {active}) ===")
print(f"idle_W = {idle_W:.1f}   (dynamic = mean - idle)\n")
hdr = f"{'phase':16s}{'P_dyn W':>9}{'gfx_act%':>9}{'clk MHz':>9}{'img/s':>9}{'J/img_dyn':>11}"
print(hdr); print("-"*len(hdr))
res = {}
for name in ("bf16", "fp16_noscaler", "fp16_scaler"):
    if name not in agg: continue
    Pdyn, act, clk, ips = m(name, "Pdyn"), m(name, "act"), m(name, "clk"), m(name, "img_s")
    jimg = Pdyn / ips if ips else float("nan")
    res[name] = dict(Pdyn=Pdyn, ips=ips, jimg=jimg, act=act, clk=clk)
    print(f"{name:16s}{Pdyn:>9.1f}{act:>9.1f}{clk:>9.0f}{ips:>9.0f}{jimg:>11.4f}")

print()
if "bf16" in res and "fp16_noscaler" in res:
    ratio = res["fp16_noscaler"]["jimg"] / res["bf16"]["jimg"]
    e_bf16 = 3.426  # pJ/FLOP, coefficients_final e_mfma_bf16 (also applied to fp16)
    print(f"TEST 1 (coefficient bias):")
    print(f"  e_fp16/e_bf16 (dyn J/img ratio, identical FLOP/img) = {ratio:.3f}")
    print(f"  -> implied e_mfma_fp16 = {e_bf16*ratio:.3f} pJ/FLOP  vs the {e_bf16} shared coeff")
    verdict = ("FP16 CHEAPER -> shared coeff over-prices FP16 (confirms bias)" if ratio < 0.95
               else "FP16 ~= BF16 -> coefficient not the cause" if ratio < 1.05
               else "FP16 MORE expensive -> bias is the other sign")
    print(f"  verdict: {verdict}")
if "fp16_noscaler" in res and "fp16_scaler" in res:
    dP = res["fp16_scaler"]["Pdyn"] - res["fp16_noscaler"]["Pdyn"]
    print(f"\nTEST 2 (gradient-underflow / data-dependent power):")
    print(f"  P_dyn scaler - noscaler = {dP:+.1f} W "
          f"({100*dP/res['fp16_noscaler']['Pdyn']:+.1f}%)")
    print(f"  verdict: " + ("scaler draws MORE -> unscaled grads underflowed (data-dependent)"
                            if dP > 8 else "~equal -> underflow NOT a material power driver"))
