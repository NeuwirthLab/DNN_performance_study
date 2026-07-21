#!/usr/bin/env python3
# ============================================================================
# analyze_power.py  --  turn a power_profile.slurm run dir into power/energy
#                       numbers + an optional timeline plot.
# ----------------------------------------------------------------------------
# Pure standard library for the numbers (runs on a bare compute-node python3).
# matplotlib is optional -- if absent, the PNG is skipped and everything else
# still works.
#
# Consumes (written by power_profile.slurm):
#   <run-dir>/power_samples.csv   tidy: timestamp_s, <backend columns...>
#   <run-dir>/markers.csv         event,timestamp_s  (idle_start/end, train_*)
#   <run-dir>/energy_before.csv   optional energy-accumulator snapshot
#   <run-dir>/energy_after.csv    optional energy-accumulator snapshot
#
# Produces:
#   <run-dir>/power_summary.csv   per-GPU + aggregate stats
#   <run-dir>/power_timeline.png  (if matplotlib present)
#   ...and prints a report to stdout.
#
# Column matching is by REGEX on the header, because amd-smi and rocm-smi name
# things differently and versions drift. If detection misses, pass --power-col
# / --gpu-col with the exact header text from your power_samples.csv.
#
# Higher-rate alternative (for fwd/bwd/allreduce phase power): replace the CLI
# sampler with the amdsmi Python binding:
#     import amdsmi; amdsmi.amdsmi_init()
#     h = amdsmi.amdsmi_get_processor_handles()[gpu]
#     amdsmi.amdsmi_get_power_info(h)['current_socket_power']   # ~50-100 Hz OK
# ============================================================================
import argparse
import csv
import os
import re
import statistics
import sys

# Header matchers cover amd-smi, rocm-smi, and nvidia-smi naming.
POWER_RE = re.compile(r"(socket.?power|graphics.?package.?power|average.?socket|"
                      r"current.?socket|power\.draw|power.*\(?\s*w\s*\)?|^power)", re.I)
GPU_RE   = re.compile(r"(^gpu$|gpu.?id|^device$|^card|^id$|^index$|gpu.?index)", re.I)
SCLK_RE  = re.compile(r"(gfx_\d+_clk$|^sclk$|sclk|clocks\.?sm|gfx.?clock)", re.I)
MCLK_RE  = re.compile(r"(mem_\d+_clk$|^mclk$|mclk|clocks\.?mem|mem.?clock)", re.I)
TEMP_RE  = re.compile(r"(^hotspot$|junction|temperature\.gpu|edge.?temp|temperature|temp.*\(?c\)?)", re.I)
UTIL_RE  = re.compile(r"(gfx.?activity|utilization\.gpu|gfx.?util|gpu.?use|gpu_?util|^use)", re.I)
NUM_RE   = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def to_float(s):
    if s is None:
        return None
    m = NUM_RE.search(str(s))
    return float(m.group()) if m else None


def find_col(header, regex, prefer=None):
    """Return the header name matching regex; prefer an exact-ish name first."""
    cands = [h for h in header if regex.search(h or "")]
    if not cands:
        return None
    if prefer:
        for c in cands:
            if prefer.lower() in c.lower():
                return c
    # shortest name tends to be the plain value column, not a derived/limit one
    return min(cands, key=len)


def load_markers(path):
    ev = {}
    if not os.path.exists(path):
        return ev
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0] != "event":
                t = to_float(row[1])
                if t is not None:
                    ev[row[0]] = t
    return ev


def load_samples(path, gpu_col_override, power_col_override):
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if r]
    header = [h.strip() for h in header]
    idx = {h: i for i, h in enumerate(header)}

    ts_col   = "timestamp_s" if "timestamp_s" in idx else header[0]
    gpu_col  = gpu_col_override   or find_col(header, GPU_RE)
    pow_col  = power_col_override or find_col(header, POWER_RE, prefer="socket")
    sclk_col = find_col(header, SCLK_RE)
    mclk_col = find_col(header, MCLK_RE)
    temp_col = find_col(header, TEMP_RE)
    util_col = find_col(header, UTIL_RE)

    if pow_col is None:
        sys.exit(f"ERROR: no power column found in {path}\n"
                 f"  header was: {header}\n"
                 f"  re-run with --power-col '<exact name>'")
    if gpu_col is None:
        # single-GPU CSVs sometimes omit an index column; synthesize one
        gpu_col = "__gpu__"

    cols = dict(ts=ts_col, gpu=gpu_col, power=pow_col,
                sclk=sclk_col, mclk=mclk_col, temp=temp_col, util=util_col)

    samples = {}  # gpu -> list of dict(t, power, sclk, ...)
    for r in rows:
        def get(name):
            c = cols[name]
            if c is None or c == "__gpu__":
                return None
            i = idx.get(c)
            return r[i] if (i is not None and i < len(r)) else None
        t = to_float(get("ts") if cols["ts"] in idx else r[0])
        p = to_float(get("power"))
        if t is None or p is None:
            continue
        g = (get("gpu") or "0").strip() if cols["gpu"] != "__gpu__" else "0"
        samples.setdefault(g, []).append(dict(
            t=t, power=p,
            sclk=to_float(get("sclk")), mclk=to_float(get("mclk")),
            temp=to_float(get("temp")), util=to_float(get("util"))))
    for g in samples:
        samples[g].sort(key=lambda d: d["t"])
    return samples, cols


def window(series, t0, t1):
    return [d for d in series if t0 is not None and t1 is not None and t0 <= d["t"] <= t1]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = min(len(xs) - 1, int(round(q / 100.0 * (len(xs) - 1))))
    return xs[k]


def trapz_energy(series, t0, t1):
    """Integrate power(W) over [t0,t1] -> energy in Joules (W*s)."""
    w = window(series, t0, t1)
    if len(w) < 2:
        return None
    e = 0.0
    for a, b in zip(w, w[1:]):
        e += 0.5 * (a["power"] + b["power"]) * (b["t"] - a["t"])
    return e


def fmt(v, nd=1, unit=""):
    return f"{v:.{nd}f}{unit}" if v is not None else "n/a"


def energy_delta(run_dir, keep_ids=None):
    """Exact ΔE from the energy-accumulator snapshots, summed over GPUs.

    keep_ids: if given, only sum these GPU ids (the active ones on a shared node).

    amd-smi  -> column 'total_energy_consumption' in Joules
    nvidia-smi -> 'total_energy_consumption [mJ]' in milli-Joules
    Returned in Joules where the unit is known, else raw (see caller note).
    """
    def total(path):
        if not os.path.exists(path):
            return None
        with open(path) as f:
            rows = [r for r in csv.reader(f) if r]
        if len(rows) < 2:
            return None
        header = [h.strip() for h in rows[0]]
        ecol = next((i for i, h in enumerate(header)
                     if re.search(r"energy", h, re.I)), None)
        if ecol is None:
            return None
        gcol = next((i for i, h in enumerate(header) if GPU_RE.search(h)), None)
        scale = 1e-3 if re.search(r"\bmj\b|\[mj\]", header[ecol], re.I) else 1.0  # mJ->J
        vals = []
        for r in rows[1:]:
            if len(r) <= ecol:
                continue
            if keep_ids is not None and gcol is not None and len(r) > gcol \
                    and r[gcol].strip() not in keep_ids:
                continue   # skip idle/neighbour GPUs on a shared node
            v = to_float(r[ecol])
            if v is not None and v > 0:
                vals.append(v * scale)
        return sum(vals) if vals else None
    b = total(os.path.join(run_dir, "energy_before.csv"))
    a = total(os.path.join(run_dir, "energy_after.csv"))
    if a is None or b is None or a < b:
        return None
    return a - b  # Joules over the whole measurement (idle+run+cooldown)


def main():
    ap = argparse.ArgumentParser(description="Analyze MI300X power-profile run dir")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--warmup", type=float, default=45.0,
                    help="seconds trimmed from front of train window (steady-state); "
                         "ignored when --warmup-epochs is used and epoch markers exist")
    ap.add_argument("--warmup-epochs", type=int, default=0,
                    help="discard the first N epochs (epoch-based warmup) and measure "
                         "from the start of epoch N onward; needs [PWRMARK] epoch markers")
    ap.add_argument("--throughput", type=float, default=None,
                    help="img/s from trainer, for J/image and img/s/W")
    ap.add_argument("--min-dynamic-w", type=float, default=25.0,
                    help="a GPU counts as 'active' (part of this run) if its steady "
                         "power exceeds its idle by this many W (shared-node filter)")
    ap.add_argument("--gpu-col", default=None)
    ap.add_argument("--power-col", default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    rd = args.run_dir
    samples, cols = load_samples(os.path.join(rd, "power_samples.csv"),
                                 args.gpu_col, args.power_col)
    ev = load_markers(os.path.join(rd, "markers.csv"))

    idle0, idle1 = ev.get("idle_start"), ev.get("idle_end")
    tr0, tr1     = ev.get("train_start"), ev.get("train_end")

    # epoch markers: events named "epoch<n>" -> [(n, ts), ...] sorted by n
    epochs = sorted(((int(re.match(r"epoch(\d+)$", k).group(1)), v)
                     for k, v in ev.items() if re.match(r"epoch(\d+)$", k)),
                    key=lambda t: t[0])
    # per-epoch windows: epoch e spans [ts_e, ts_{e+1}) ; last -> train_end
    epoch_win = []
    for i, (n, ts) in enumerate(epochs):
        end = epochs[i + 1][1] if i + 1 < len(epochs) else tr1
        epoch_win.append((n, ts, end))
    measured = [w for w in epoch_win if w[0] >= args.warmup_epochs]

    if args.warmup_epochs > 0 and measured:
        steady0 = measured[0][1]      # start of the first MEASURED epoch
        warm_desc = (f"epoch-based: discard {args.warmup_epochs} warmup epoch(s); "
                     f"measure epochs {[m[0] for m in measured]}")
    else:
        steady0 = (tr0 + args.warmup) if tr0 is not None else None
        warm_desc = f"time-based: trim {args.warmup:.0f}s"

    print(f"\nColumns used  : power='{cols['power']}'  gpu='{cols['gpu']}'  "
          f"sclk='{cols['sclk']}'  temp='{cols['temp']}'")
    if tr0 and tr1:
        print(f"Train window  : {tr1 - tr0:.1f}s  ({warm_desc} -> "
              f"steady {max(0.0, tr1 - steady0):.1f}s)")

    # ---- per-GPU metrics ----------------------------------------------------
    rows_out = []
    # measurement window = [steady0, tr1] (measured epochs / post-warmup). ALL
    # aggregate metrics (power, energy, EDP, J/image) use this SAME window so they
    # stay consistent — energy must NOT include the warmup epoch while steady_W
    # excludes it. ms0 falls back to tr0 if steady0 is unset.
    ms0 = steady0 if steady0 is not None else tr0
    for g in sorted(samples, key=lambda x: (len(x), x)):
        s = samples[g]
        p_idle   = mean([d["power"] for d in window(s, idle0, idle1)])
        steady_w = window(s, ms0, tr1)
        p_steady = mean([d["power"] for d in steady_w])
        p_p95    = pct([d["power"] for d in steady_w], 95)
        p_peak   = max([d["power"] for d in window(s, ms0, tr1)] or [0]) or None
        sclk     = mean([d["sclk"] for d in steady_w])
        temp     = mean([d["temp"] for d in steady_w])
        e_total  = trapz_energy(s, ms0, tr1)
        e_dyn    = (e_total - (p_idle * (tr1 - ms0))) if (e_total is not None and
                    p_idle is not None and ms0 and tr1) else None
        rows_out.append(dict(gpu=g, idle_W=p_idle, steady_W=p_steady, p95_W=p_p95,
                             peak_W=p_peak, sclk_MHz=sclk, temp_C=temp,
                             E_train_J=e_total, E_dyn_J=e_dyn))

    # ---- pick the GPUs actually doing work ----------------------------------
    # On a shared node `-g all` also samples neighbours' (idle) GPUs, and SLURM's
    # job-local index need not match amd-smi's physical index. A GPU "belongs" to
    # this run if its steady power rose >= --min-dynamic-w above its own idle.
    def is_active(r):
        return (r["steady_W"] is not None and r["idle_W"] is not None
                and (r["steady_W"] - r["idle_W"]) >= args.min_dynamic_w)
    active = [r for r in rows_out if is_active(r)]
    if not active:                       # nothing crossed the bar -> treat all as the run
        active = rows_out
    active_ids = {r["gpu"] for r in active}

    print(f"\n{'GPU':>5} {'idle_W':>8} {'steady_W':>9} {'p95_W':>7} {'peak_W':>7} "
          f"{'sclk_MHz':>9} {'temp_C':>7} {'E_train_J':>10} {'E_dyn_J':>9}")
    print("-" * 84)
    for r in rows_out:
        star = "*" if r["gpu"] in active_ids else " "
        print(f"{star}{r['gpu']:>4} {fmt(r['idle_W']):>8} {fmt(r['steady_W']):>9} "
              f"{fmt(r['p95_W']):>7} {fmt(r['peak_W']):>7} {fmt(r['sclk_MHz'],0):>9} "
              f"{fmt(r['temp_C']):>7} {fmt(r['E_train_J'],0):>10} {fmt(r['E_dyn_J'],0):>9}")

    def asum(key):
        return sum(r[key] for r in active if r[key] is not None) or None
    agg = dict(idle=asum("idle_W"), steady=asum("steady_W"), peak=asum("peak_W"),
               e_total=asum("E_train_J"), e_dyn=asum("E_dyn_J"))
    print("-" * 84)
    print(f"{'ACT':>5} {fmt(agg['idle']):>8} {fmt(agg['steady']):>9} {'':>7} "
          f"{fmt(agg['peak']):>7} {'':>9} {'':>7} "
          f"{fmt(agg['e_total'],0):>10} {fmt(agg['e_dyn'],0):>9}")
    print(f"\n  active GPUs (*)     : {len(active)} of {len(rows_out)} sampled "
          f"(>= {args.min_dynamic_w:.0f} W dynamic; the rest are idle/neighbours)")

    # ---- efficiency metrics (same measured window as the energy above) ------
    dur = (tr1 - ms0) if (ms0 and tr1) else None
    print("\n=== Summary (over active GPUs) ===")
    print(f"  Active / sampled    : {len(active)} / {len(rows_out)} GPUs")
    print(f"  Aggregate idle      : {fmt(agg['idle'])} W")
    print(f"  Aggregate steady    : {fmt(agg['steady'])} W")
    print(f"  Aggregate peak      : {fmt(agg['peak'])} W")
    _ekwh = (agg['e_total'] / 3.6e6) if agg['e_total'] is not None else None
    print(f"  Total train energy  : {fmt(agg['e_total'],0)} J  ({fmt(_ekwh,4)} kWh)")
    print(f"  Dynamic energy      : {fmt(agg['e_dyn'],0)} J  (total minus idle floor)")
    de = energy_delta(rd, keep_ids=active_ids)
    if de is not None:
        print(f"  HW energy-accum ΔE  : {de:.0f} J  ({de/3.6e6:.4f} kWh)  "
              f"[full run: idle+work+cooldown; cross-check vs trapezoid]")
    j_per_img = ips_per_w = edp = ed2p = None
    if args.throughput and dur and agg["steady"] and agg["e_total"]:
        images = args.throughput * dur
        j_per_img = agg["e_total"] / images if images else None
        ips_per_w = args.throughput / agg["steady"]
        edp  = agg["e_total"] * dur            # energy-delay product (J*s)
        ed2p = agg["e_total"] * dur * dur       # energy-delay^2 product (J*s^2)
        print(f"  Throughput          : {args.throughput:.1f} img/s")
        print(f"  Energy / image      : {fmt(j_per_img,3)} J/image")
        print(f"  Efficiency          : {fmt(ips_per_w,3)} img/s/W")
        print(f"  EDP (E*t)           : {fmt(edp,0)} J*s")

    # ---- per-epoch breakdown + average over the MEASURED epochs -------------
    if measured and len(measured) >= 1:
        print(f"\n=== Per-epoch (measured, over active GPUs) ===")
        print(f"  {'epoch':>5} {'dur_s':>7} {'aggW':>8} {'E_J':>10}")
        per = []
        for (n, e0, e1) in measured:
            if e0 is None or e1 is None:
                continue
            pw = sum((mean([d["power"] for d in window(samples[r['gpu']], e0, e1)]) or 0)
                     for r in active)
            en = sum((trapz_energy(samples[r['gpu']], e0, e1) or 0) for r in active)
            per.append((n, e1 - e0, pw, en))
            print(f"  {n:>5} {e1 - e0:>7.1f} {pw:>8.1f} {en:>10.0f}")
        if per:
            mP = sum(p[2] for p in per) / len(per)
            mE = sum(p[3] for p in per) / len(per)
            print(f"  {'avg':>5} {sum(p[1] for p in per)/len(per):>7.1f} "
                  f"{mP:>8.1f} {mE:>10.0f}   <- averaged over {len(per)} measured epoch(s)")

    # ---- machine-readable one-row run summary (for sweep plots) -------------
    metrics = dict(
        gpus_active=len(active), gpus_sampled=len(rows_out),
        warmup_s=args.warmup, warmup_epochs=args.warmup_epochs,
        n_measured_epochs=len(measured), train_s=dur, throughput_img_s=args.throughput,
        idle_W=agg["idle"], steady_W=agg["steady"], peak_W=agg["peak"],
        E_train_J=agg["e_total"], E_dyn_J=agg["e_dyn"],
        E_hw_accum_J=de, j_per_image=j_per_img, img_s_per_W=ips_per_w,
        EDP_Js=edp, ED2P_Js2=ed2p)
    with open(os.path.join(rd, "run_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(metrics.keys()))
        w.writerow(["" if v is None else (f"{v:.6g}" if isinstance(v, float) else v)
                    for v in metrics.values()])

    # ---- write summary CSV --------------------------------------------------
    out_csv = os.path.join(rd, "power_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gpu", "idle_W", "steady_W", "p95_W",
                                          "peak_W", "sclk_MHz", "temp_C",
                                          "E_train_J", "E_dyn_J"])
        w.writeheader()
        for r in rows_out:
            w.writerow({k: (v if k == "gpu" else "" if v is None else f"{v:.3f}")
                        for k, v in r.items()})
    print(f"\n  wrote {out_csv}")

    # ---- optional timeline plot --------------------------------------------
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            t0ref = min(d["t"] for s in samples.values() for d in s)
            fig, ax = plt.subplots(figsize=(11, 5))
            for g in sorted(samples):
                xs = [d["t"] - t0ref for d in samples[g]]
                ys = [d["power"] for d in samples[g]]
                ax.plot(xs, ys, lw=1.0, alpha=0.8, label=f"GPU {g}")
            for name, color in (("idle_start", "#cccccc"), ("idle_end", "#cccccc"),
                                ("train_start", "#2c7fb8"), ("train_end", "#2c7fb8")):
                if name in ev:
                    ax.axvline(ev[name] - t0ref, color=color, ls="--", lw=1)
            if steady0:
                ax.axvspan(steady0 - t0ref, (tr1 - t0ref) if tr1 else (steady0 - t0ref),
                           color="#41ab5d", alpha=0.08, label="steady-state")
            ax.set_xlabel("time since first sample (s)")
            ax.set_ylabel("socket power (W)")
            ax.set_title(os.path.basename(rd))
            ax.legend(ncol=4, fontsize=8)
            ax.grid(alpha=0.3)
            png = os.path.join(rd, "power_timeline.png")
            fig.tight_layout(); fig.savefig(png, dpi=130)
            print(f"  wrote {png}")
        except Exception as e:
            print(f"  (plot skipped: {e})")


if __name__ == "__main__":
    main()
