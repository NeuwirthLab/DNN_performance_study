#!/usr/bin/env python3
"""Convert an AMD rocprofv3 profile root into the NCU-style raw.csv that the
H100 fig_traffic_decomposition.py reads — so the *unchanged* provided script
draws the figure from MI300X data.

The provided parser does, per kernel row:
    L1_served_B = (sum of l1tex hit sub-counters) * 32
    L2_served_B =  lts__t_sectors_lookup_hit.sum  * 32
    DRAM_B      =  lts__t_sectors_lookup_miss.sum * 32

We emit the MI300X L1 / L2 / DRAM served *bytes* / 32 into those columns; the
script's ×32 turns them back into the right bytes.

L2 / DRAM split — CACHE-BASED (direct TCC_HIT/MISS), not the old EA-subtraction.
The L1-miss traffic that reaches L2 (TCP->TCC requests) is split using the
measured L2 hit rate from the dedicated counters:

    L1_served = (TCP_TOTAL_CACHE_ACCESSES - TCP->TCC reqs) * 64      # L1 hits
    L2_hit_rate = TCC_HIT / (TCC_HIT + TCC_MISS)                     # direct L2 metric
    L2_served = TCP->TCC reqs *      L2_hit_rate  * 64               # hit in L2
    DRAM      = TCP->TCC reqs * (1 - L2_hit_rate) * 64               # miss -> HBM

Why not the old `L2 = TCP->TCC - TCC_EA0` subtraction? TCC_EA0 counts only one
memory channel at a different granularity and its request count exceeds the
real L2-miss count by 1.1-2.2x, which falsely drove L2 toward 0 (BatchNorm came
out ~2% when the measured L2 hit rate is ~22%). TCC_HIT/MISS is the
authoritative L2 metric and is internally consistent: (TCC_HIT+TCC_MISS) matches
the TCP->TCC request stream to ~1-3%.

Kernel_Name is the raw AMD name so kernel_canon.canon (-> friendly_label)
buckets it into the same canonical categories as every other plot.
"""
import os, sys, csv, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from plot_roofline import find_counter_csvs, load_pmc_wide

REQ_BYTES = 64       # CDNA3 request granularity (matches analyze_roofline.py)
SECTOR    = 32       # NCU sector size the consumer multiplies back by

L1_TOTAL = "TCP_TOTAL_CACHE_ACCESSES_sum"
L1_TO_L2 = ["TCP_TCC_READ_REQ_sum", "TCP_TCC_WRITE_REQ_sum",
            "TCP_TCC_ATOMIC_WITH_RET_REQ_sum"]
L2_HIT   = "TCC_HIT_sum"
L2_MISS  = "TCC_MISS_sum"

# One representative NCU column for each served level (the parser sums all 11 L1
# hit sub-counters, so a single populated one is enough).
COL_L1   = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum"
COL_L2   = "lts__t_sectors_lookup_hit.sum"
COL_DRAM = "lts__t_sectors_lookup_miss.sum"
COL_DUR  = "gpu__time_duration.sum"


def _col(df, name):
    return pd.to_numeric(df[name], errors="coerce").fillna(0.0) if name in df.columns \
        else pd.Series(0.0, index=df.index)


def convert(root, out_path):
    wide = load_pmc_wide(find_counter_csvs(root))
    for c in [L1_TOTAL, *L1_TO_L2, L2_HIT, L2_MISS]:
        wide[c] = _col(wide, c)
    wide["_dur_ms"] = (pd.to_numeric(wide["End_Timestamp"], errors="coerce")
                       - pd.to_numeric(wide["Start_Timestamp"], errors="coerce")) * 1e-6

    g = wide.groupby("Kernel_Name").agg(
        {**{c: "sum" for c in [L1_TOTAL, *L1_TO_L2, L2_HIT, L2_MISS]},
         "_dur_ms": "sum"})

    tot   = g[L1_TOTAL]
    to_l2 = g[L1_TO_L2].sum(axis=1)
    hit   = g[L2_HIT]
    miss  = g[L2_MISS]
    l2_rate = (hit / (hit + miss)).clip(0, 1).fillna(0.0)   # direct L2 hit rate
    l1_B  = (tot - to_l2).clip(lower=0) * REQ_BYTES
    l2_B  = (to_l2 * l2_rate)           * REQ_BYTES
    dr_B  = (to_l2 * (1.0 - l2_rate))   * REQ_BYTES

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        hdr = ["Kernel Name", COL_L1, COL_L2, COL_DRAM, COL_DUR]
        w.writerow(hdr)
        w.writerow(["", "sector", "sector", "sector", "ms"])   # units row
        for k in g.index:
            w.writerow([k, l1_B[k] / SECTOR, l2_B[k] / SECTOR,
                        dr_B[k] / SECTOR, g.loc[k, "_dur_ms"]])
    print(f"wrote {out_path}  ({len(g)} kernels)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    a = ap.parse_args()
    convert(a.root, a.out)
