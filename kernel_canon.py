"""kernel_canon for the AMD figure — emits the SAME canonical axis names as the
H100 figure so the two memory-traffic-decomposition plots are row-equivalent.

The provided fig_traffic_decomposition.py calls `canon(name) -> (label, cat)`
and groups bars by `label`. On H100 the labels are pass-level canonical names
(conv_fprop / conv_dgrad / conv_wgrad, bn_fwd / bn_bwd, relu, gemm,
add_residual, ...). We reach the same names on MI300X by first running this
repo's friendly_label() (which already decodes every rocBLAS/MIOpen/CK/ATen
kernel) and then renaming its output to the H100 vocabulary. The AMD kernel
names carry the conv pass explicitly (igemm_fwd/bwd/wrw, naive_conv ..._fwd/
bwd/wrw_, CK bwd_weight), so the mapping is exact, not guessed — except Winograd
(miopenSp3AsmConv), which MIOpen uses as a forward 3x3 algorithm -> conv_fprop.
"""
from plot_roofline import friendly_label

# AMD friendly_label  ->  H100 canonical axis name
_RENAME = {
    # ── Convolution, by pass (matches H100 conv_fprop/dgrad/wgrad) ──
    "Conv fwd (impl. GEMM)":     "conv_fprop",
    "Conv fwd (direct)":         "conv_fprop",
    "Conv 3×3 Winograd F(2×3)":  "conv_fprop",
    "Conv 3×3 Winograd F(3×2)":  "conv_fprop",
    "Conv bwd-data (impl. GEMM)":"conv_dgrad",
    "Conv bwd-data (direct)":    "conv_dgrad",
    "Conv wrw (impl. GEMM)":     "conv_wgrad",
    "Conv wrw (direct)":         "conv_wgrad",
    "Conv wrw (CK XDL)":         "conv_wgrad",
    # ── BatchNorm ──
    "BatchNorm fwd":             "bn_fwd",
    "BatchNorm fwd (infer)":     "bn_fwd",
    "BatchNorm bwd":             "bn_bwd",
    # ── Activation / residual ──
    "ReLU fwd":                  "relu",
    "ReLU bwd":                  "relu",
    "Residual add":              "add_residual",
    # ── Long tail (kept in the H100 lowercase snake_case style) ──
    "Bias add":                  "bias_add",
    "Bias grad":                 "bias_grad",
    "Pool fwd":                  "maxpool_fwd",
    "Max pool fwd":              "maxpool_fwd",
    "Max pool bwd":              "maxpool_bwd",
    "Global avg pool":           "avgpool",
    "Global avg pool bwd":       "avgpool_bwd",
    "Softmax":                   "softmax",
    "Softmax bwd":               "softmax_bwd",
    "Softmax-CE bwd":            "softmax_ce_bwd",
    "Cross-entropy loss":        "ce_loss",
    "SGD update":                "sgd",
    "Mean reduce":               "mean_reduce",
    "Reduce":                    "reduce",
    "Transpose":                 "transpose",
    "CNHW→NCHW":                 "layout_transform",
    "NCHW→CNHW":                 "layout_transform",
    "Concat":                    "concat",
    "Scale (a·x)":               "scale",
    "Elementwise (misc)":        "elementwise",
    "Add (int counter)":         "add",
    "Fill":                      "fill",
    "SubTensor set":             "set",
    "Top-1 accuracy":            "top1_acc",
    "hipMemcpy":                 "memcpy",
    "hipMemset":                 "memset",
    "__amd_rocclr_initHeap":     "init_heap",
}


def _to_h100(amd_label: str) -> str:
    if amd_label in _RENAME:
        return _RENAME[amd_label]
    if amd_label.startswith("GEMM"):          # every GEMM tile -> single bucket
        return "gemm"
    if amd_label.startswith("_ZN") and "oneRankReduce" in amd_label:
        return "allreduce"                    # RCCL all-reduce
    # Fallback: lowercase snake_case so it still reads like an H100 name.
    return amd_label.lower().replace(" ", "_").replace("(", "").replace(")", "")


def canon(name: str):
    label, cat = friendly_label(name)
    return _to_h100(label), cat
