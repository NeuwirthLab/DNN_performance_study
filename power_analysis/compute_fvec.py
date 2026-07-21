#!/usr/bin/env python3
"""
compute_fvec.py -- extract F_vec (vector-ALU FLOP) from a 15-pass rocprofv3
counter profile, following methodology Eq. (fvec):

    F_vec = W * (N_add + N_mul + N_trans + 2*N_fma),   W = 64

Counters (each measured in its own PMC pass, summed over all profiled
dispatches):
    N_add   = SQ_INSTS_VALU_ADD_F32     (pmc_2)
    N_mul   = SQ_INSTS_VALU_MUL_F32     (pmc_2)
    N_trans = SQ_INSTS_VALU_TRANS_F32   (pmc_3)
    N_fma   = SQ_INSTS_VALU_FMA_F32     (pmc_1)
    M_fp32  = SQ_INSTS_VALU_MFMA_MOPS_F32 (pmc_6)  -> F_mfma = 512 * M_fp32

The absolute counter totals are over the profiled window (profile-steps, KIR
1-50), which is NOT the same as the power-run deficit window. We normalise by
the MFMA ratio, which is window-independent:

    F_vec_window = (F_vec_total / F_mfma_total) * F_mfma_window

where F_mfma_window is the already-priced MFMA FLOP over the power-run window
from the validation JSON (F_mfma_per_step * steps). This ties F_vec to a
quantity the model already accounts for and cancels the step-count normalisation.

Usage:
    compute_fvec.py <profile_root>  (lives in power_analysis/)  [F_mfma_per_step]  [steps]
e.g. mobilenet: compute_fvec.py profiles/pt_amd_53070169 168588183901.65854 41
"""
import csv, glob, os, sys
from collections import defaultdict

CN, CV, KN = 15, 16, 8   # 0-indexed cols: Counter_Name, Counter_Value, Kernel_Name

WANT = ["SQ_INSTS_VALU_ADD_F32", "SQ_INSTS_VALU_MUL_F32",
        "SQ_INSTS_VALU_TRANS_F32", "SQ_INSTS_VALU_FMA_F32",
        "SQ_INSTS_VALU_MFMA_MOPS_F32", "SQ_INSTS_VALU_MFMA_MOPS_F16",
        "SQ_INSTS_VALU_MFMA_MOPS_BF16"]

def collect(root):
    tot = defaultdict(float)
    per_kernel = defaultdict(lambda: defaultdict(float))
    for csvf in glob.glob(os.path.join(root, "**", "*_counter_collection.csv"),
                          recursive=True):
        with open(csvf) as fh:
            rdr = csv.reader(fh)
            hdr = next(rdr, None)
            for row in rdr:
                if len(row) <= CV:
                    continue
                name = row[CN].strip().strip('"')
                if name not in WANT:
                    continue
                try:
                    val = float(row[CV])
                except ValueError:
                    continue
                tot[name] += val
                per_kernel[name][row[KN].strip().strip('"')] += val
    return tot, per_kernel

def main():
    root = sys.argv[1]
    f_mfma_per_step = float(sys.argv[2]) if len(sys.argv) > 2 else None
    steps = float(sys.argv[3]) if len(sys.argv) > 3 else None

    tot, per_kernel = collect(root)
    print(f"profile: {root}")
    for k in WANT:
        print(f"  {k:32s} = {tot[k]:.4e}")

    W = 64.0
    F_vec_total = W * (tot["SQ_INSTS_VALU_ADD_F32"]
                       + tot["SQ_INSTS_VALU_MUL_F32"]
                       + tot["SQ_INSTS_VALU_TRANS_F32"]
                       + 2.0 * tot["SQ_INSTS_VALU_FMA_F32"])
    F_mfma_total = 512.0 * tot["SQ_INSTS_VALU_MFMA_MOPS_F32"]
    print(f"\n  F_vec_total  (over profiled window) = {F_vec_total:.4e} FLOP")
    print(f"  F_mfma_total (512*M_fp32)           = {F_mfma_total:.4e} FLOP")
    if F_mfma_total > 0:
        print(f"  ratio F_vec/F_mfma                  = {F_vec_total/F_mfma_total:.4f}")

    if f_mfma_per_step and steps and F_mfma_total > 0:
        F_mfma_window = f_mfma_per_step * steps
        ratio = F_vec_total / F_mfma_total
        F_vec_window = ratio * F_mfma_window
        # cross-check: implied number of profiled steps
        n_prof = F_mfma_total / f_mfma_per_step
        print(f"\n  [cross-check] implied profiled steps = F_mfma_total/F_mfma_per_step "
              f"= {n_prof:.2f}")
        print(f"  F_mfma_window (per-step * {steps:.0f})     = {F_mfma_window:.4e} FLOP")
        print(f"  ==> F_vec_window (ratio-scaled)     = {F_vec_window:.4e} FLOP")
        print(f"\n  >>> pass to fit_e_vec.py:  --f-vec {F_vec_window:.6e}")

if __name__ == "__main__":
    main()
