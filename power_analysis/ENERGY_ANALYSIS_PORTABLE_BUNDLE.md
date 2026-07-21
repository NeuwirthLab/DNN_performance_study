# Portable Energy-Coefficient Analysis Bundle

**Self-contained.** This one file carries both the method *and* the full source
of every analysis script. Paste it onto the target system (NVIDIA CUDA, or any
accelerator with a power-telemetry CLI + a per-kernel counter profiler), then
extract each script from the fenced blocks in **Part 2** into a file of the name
in its heading. No other download is needed for the analysis toolkit.

**Goal.** Measure where a training run's energy goes, per operation class, so any
system can find and close its own efficiency gaps. You extract per-operation
energy coefficients from short isolated "replay" loops, then use them plus a
full-training kernel profile to predict full-training energy and validate.

Verified against `power_analysis/` source on 2026-07-04.

---

# PART 1 — METHOD & PORTING

## 1. The model

Per training step, aggregated over the active GPUs:

```
E_step  =  e_mfma(p)*F_mfma  +  e_hbm*B_hbm  +  e_comm*V_ar  +  P_idle*t_step
```

| Term | Meaning | Unit | Source of measurement |
|---|---|---|---|
| `e_mfma(p)` | energy per matrix-engine FLOP, precision `p in {FP32,BF16,FP16}` | J/FLOP | isolated conv/GEMM replay loop |
| `e_hbm` | energy per HBM byte moved by bandwidth-bound (BN/activation) kernels | J/byte | isolated BN+ReLU replay loop |
| `e_comm` | energy per all-reduced gradient byte | J/byte/GPU | isolated multi-GPU all-reduce loop |
| `P_idle` | static power floor | W/GPU | idle-baseline window |

A coefficient is **(per-iteration dynamic energy) / (per-iteration counted
work)**. The replay loop fires the SAME kernels every iteration and saturates the
GPU, so energy and work both scale linearly with iteration count; the ratio of
their rates is clock/iteration-count independent. That invariance is why the
method transports across hardware.

**Validation** = use the coefficients + a full-training per-kernel counter
profile to predict full-training energy, compared against separately measured
energy: in-sample on the native trainer, cross-stack on PyTorch. The final
MI300X coefficients are in `coefficients_final.json` (read its `units` and
`provenance` blocks — they record every calibration choice and two retracted
values).

## 2. What is already cross-architecture, and what you replace

The measurement and math layers were written to cover the native-HIP,
PyTorch-ROCm, AND CUDA builds with one code path.

**Already portable (no change):**
- `measure_power.sh` auto-selects the telemetry backend: `amd-smi` → `rocm-smi`
  → `nvidia-smi`. On NVIDIA it queries `power.draw` and reads
  `total_energy_consumption`.
- `gpu_power_sampler.sh` — same backend fallback.
- `analyze_power.py` — matches CSV columns by regex; the matchers already cover
  nvidia-smi naming. Pure standard library.
- `compose_energy.py` — the extract/validate arithmetic is hardware-agnostic and
  pure standard library.
- `fit_dvfs_cubic.py` — works on any `(cap_w, avg_power_w_per_gpu,
  mean_sm_clock_mhz)` CSV.

**Must be replaced for NVIDIA:**
1. **The counter profiler + counter names.** AMD uses `rocprofv3` with gfx942
   counters; NVIDIA uses **Nsight Compute (`ncu`)** / CUPTI. See the mapping in
   Section 4. `analyze_roofline.py` (the raw-counter → `roofline_summary.csv`
   converter) is the file whose counter vocabulary you re-target.
2. **The build recipe.** The driver + model are CUDA source with a `-DUSE_HIP`
   shim; build the native CUDA path (drop `-DUSE_HIP`, use `nvcc`, link
   cuBLAS + cuDNN + NCCL). See Section 5.

Everything else — run order, marker protocol, coefficient arithmetic,
validation — is unchanged.

## 3. Pipeline stages

The master orchestrator is `extract_coefficients.slurm` (embedded in Part 2). It
is the executable spec; the stages below map to its numbered sections.

- **Stage 1 — build the isolation driver** (`component_loop.cu`). It builds the
  model FP32, runs one real `Forward()+Backward()` warmup, then loops ONE kernel
  class. Each block emits `[PWRMARK] epoch <i> <unixtime>`; block 0 is warmup and
  discarded. Size each block for ~30–60 s of GPU load (the sampler is only
  1–2 Hz, so a block must last far longer than the sample interval).
- **Stage 2 — per-component runs**, two passes into the SAME dir:
  - **(a) energy** — `measure_power.sh` brackets the driver with
    `[idle] → [loop] → [cooldown]`, samples power, reads the HW energy counter,
    calls `analyze_power.py`. Contract windows: `-i 0.5 -b 15 -c 8 -E 1 -g all`.
  - **(b) counters** — `rocprofv3`/`ncu` over a SHORT kernel window (paired
    counters, retry on fault) → the per-kernel work.

  | Label | Driver mode | GPUs | Calibrates |
  |---|---|---|---|
  | `conv-fp32` | `conv` | 1 | `e_mfma_fp32` |
  | `conv-bf16` | `conv` + `ISO_BF16=1` | 1 | `e_mfma_bf16` |
  | `fc` | `fc` | 1 | `e_gemm` |
  | `bnrelu` | `bnrelu` | 1 | `e_hbm` |
  | `allreduce` | `allreduce` | N | `e_comm` |

- **Stage 3 — compose + validate** with `compose_energy.py extract` then
  `validate` (predict full-training energy, report percent error). The
  **double-counting rule** is enforced in code: `e_hbm` counts ONLY non-mfma,
  non-comm kernels' bytes.
- **Stage 4 — full-data re-validation** with `revalidate_fulldata.py` (imports
  `compose_energy.py` so the math is identical).
- **Stage 5 (optional) — DVFS / power-cap sweep**: `powercap_sweep.slurm` →
  `build_powercap_summary.py` → `fit_dvfs_cubic.py`. Use the **constrained fit**
  (fix `P_const` to the measured idle floor, fit only `β,τ`). On MI300X the
  558–1332 MHz range could not discriminate cubic from quadratic (ΔAIC < 0.7);
  on NVIDIA you can set the clock directly (`nvidia-smi -lgc`) for a wider,
  discriminable range — an improvement over the AMD path.

## 4. Counter mapping: rocprofv3 (gfx942) → Nsight Compute

The math needs three quantities per kernel: matrix FLOPs, HBM bytes, and L2
hit/miss (context). Re-target the counter names in `extract_coefficients.slurm`'s
`COUNTER_LINES` and in `analyze_roofline.py`.

| Quantity | MI300X (rocprofv3) | AMD derivation | NVIDIA (`ncu` metric) |
|---|---|---|---|
| Matrix FLOPs | `SQ_INSTS_VALU_MFMA_MOPS_F32/_BF16/_F16` | `512 FLOP/MOP × mops × dispatches` | `sm__ops_path_tensor_src_*` (Tensor-Core), or `smsp__sass_thread_inst_executed_op_hfma/ffma_pred_on.sum` for non-TC FMA |
| HBM bytes | `TCC_EA0_RDREQ_sum + TCC_EA0_WRREQ_sum` | `64 B/req × requests` | `dram__bytes_read.sum + dram__bytes_write.sum` (already bytes — **no ×64**) |
| L2 hit/miss | `TCC_HIT_sum`, `TCC_MISS_sum` | ratio → L2-hit context | `lts__t_sector_hit_rate.pct` or `lts__t_sectors_lookup_hit/miss.sum` |
| VALU FP32 (context) | `SQ_INSTS_VALU_FMA_F32/ADD_F32/MUL_F32` | `64×(2·FMA+ADD+MUL)` | `smsp__sass_thread_inst_executed_op_ffma/fadd/fmul_pred_on.sum` |

`amd_to_ncu_raw.py` (embedded) is the worked cross-reference: it documents
exactly which NCU counters correspond to the AMD ones and how the L1/L2/DRAM
byte split is derived. Read it before writing the `ncu` recipe.

Two rules that transport: (1) on NVIDIA `dram__bytes_*` are already bytes — drop
the AMD `×64`; (2) keep the SHORT kernel window (`ncu --launch-count` /
`--kernel-id`) so you profile a steady-state block, not warmup.

**AMD-only failure you will NOT hit:** rocprofv3 SIGSEGVs on MIOpen BN kernels
for all counter types (jobs 52612342), which forced a HYBRID `e_hbm` calibration
(`P_dyn_bnrelu_isolation / R_hbm_pmc_training`). On NVIDIA you can profile BN
kernels directly and get `e_hbm` the clean way. Record whichever path you use in
the output `provenance`, as `coefficients_final.json` does.

## 5. Building the driver + model for NVIDIA

The driver `component_loop.cu` (embedded in Part 2) `#include`s `resnet50.cuh`
and links `resnet50.o`, so you need the model source too. The model is large
(`resnet50.cu` 1859 lines, `resnet50.cuh` 432 lines) and is the trainer, not the
analysis — it is **NOT embedded here**. Two options:

- **A. Replicate exactly:** copy `resnet50.cu`, `resnet50.cuh` (and, for the
  legacy AMD link only, `miniz.c`) from the repo root, then build native CUDA:
  ```
  nvcc -std=c++17 -O3 --gpu-architecture=<sm_XX> -DUSE_NCCL -I <ROOT> \
       -c tools/component_loop.cu -o component_loop.o
  nvcc -std=c++17 -O3 --gpu-architecture=<sm_XX> -DUSE_NCCL -I <ROOT> \
       -c resnet50.cu -o resnet50.o
  nvcc component_loop.o resnet50.o -lcudnn -lcublas -lnccl -o component_loop
  ```
  Do **not** pass `-DUSE_HIP` (that routes through the HIP shim). `allreduce`
  mode compiles only under `-DUSE_NCCL`. `miniz.c` is dead code — droppable on
  NVIDIA.
- **B. Bring your own trainer (recommended on NVIDIA):** the analysis toolkit is
  fully decoupled from the trainer — it only needs the two CSV contracts in
  Section 6. Use torchvision / your own ResNet and emit an isolation loop that
  prints the `[PWRMARK] epoch` markers; the embedded `.py`/`.sh` do the rest.

The marker protocol is plain `printf`+`fflush` — arch-independent. The PyTorch
reference (`pytorch_resnet50_ddp.py`) runs unchanged on NVIDIA; `RN_ARCH` selects
the torchvision model for OOD tests.

## 6. The data-file contracts (the real portability interface)

The scripts talk through two CSVs. **Any system reproduces this study by
producing these two files** — trainer and profiler are interchangeable as long
as the columns exist. Names are matched exactly by `compose_energy.py`
(`load_energy_summary`, `load_roofline_summary`).

**`run_metrics.csv`** — one header + one data row, written by `analyze_power.py`,
read by `compose_energy.py`:
```
gpus_active,gpus_sampled,warmup_s,warmup_epochs,n_measured_epochs,train_s,
throughput_img_s,idle_W,steady_W,peak_W,E_train_J,E_dyn_J,E_hw_accum_J,
j_per_image,img_s_per_W,EDP_Js,ED2P_Js2
```
`compose_energy.py` consumes only **`E_dyn_J`**, **`train_s`**, **`idle_W`**,
**`n_measured_epochs`**. `E_dyn_J` = dynamic energy (steady − idle) over the
measured window; `idle_W` = aggregate idle floor.

**`roofline_summary.csv`** — one row per kernel, written by `analyze_roofline.py`,
read by `compose_energy.py`. Columns the loader reads (others ignored):
```
label, category, dispatches, total_ms,
mfma_total_mops, mfma_fp32_mops, mfma_fp16_mops, mfma_bf16_mops,
hbm_bytes_per_disp
```
`mfma_*_mops` and `hbm_bytes_per_disp` are **per-dispatch** (× `dispatches` for
totals). `category` drives the {mfma,hbm,comm,other} classification and the
double-counting rule. On NVIDIA emit this same schema from `ncu`: `mfma_*_mops`
from Tensor-Core op metrics, `hbm_bytes_per_disp` from `dram__bytes_*`.

## 7. Smallest end-to-end check on a new system

```bash
# 0. build ./component_loop for the target arch (Section 5)
# 1. idle floor
./measure_power.sh -b 60 -o out/idle -g all                        # → P_idle

# 2. one component (FP32 conv): energy + counters into ONE dir
./measure_power.sh -i 0.5 -b 15 -c 8 -E 1 -g all \
    -o out/conv-fp32 -- ./component_loop conv 256 200 6            # → run_metrics.csv
ncu --csv --set full -o out/conv-fp32/counters ./component_loop conv 256 1 2
python3 analyze_roofline.py out/conv-fp32/counters                 # → roofline_summary.csv

# 3. repeat step 2 for bnrelu, fc, conv-bf16, allreduce, then compose
python3 compose_energy.py extract --components-dir out --coefficients out/coefficients.json

# 4. validate against a full-training profile + its measured energy
python3 compose_energy.py validate \
    --coefficients out/coefficients.json \
    --native-pmc  out/full_native/roofline_summary.csv \
    --pytorch-pmc out/full_pytorch/roofline_summary.csv \
    --native-e-dyn <J> --native-train-s <S> \
    --pytorch-e-dyn <J> --pytorch-train-s <S> \
    --validation-out out/validation.json
```
Compare against the AMD references (`coefficients_final.json`,
`validation_fulldata_20260703.json`): same order of magnitude per term +
cross-stack error within ~±8–13% means the port is sound.

## 8. Honesty guardrails (carried over from the AMD study)

- **Report error bars.** Same-operating-point variance was 2–6%, comparable to
  some headline gaps; n=1 per point is not enough.
- **Coefficients are cap-specific.** A 750 W coefficient on a 300 W run degraded
  error 13% → 158%. Extract a set per cap, or scope the claim to the cap used.
- **`e_hbm` rests on an assumed L2-hit rate** (~42% on MI300X) — the softest
  coefficient. If you can measure BN-kernel bytes directly, do that.
- **Do not double-count bytes.** The `e_hbm` term is non-mfma, non-comm kernels
  only; conv/GEMM/collective bytes are priced in their own terms.
- **Two AMD values were retracted** (`e_comm_2node` bench value; driver-rate-only
  `e_hbm`) — both documented in `coefficients_final.json` provenance. Expect
  analogous traps (idle-spin power posing as comm energy; logical-rate vs
  PMC-rate mismatch).
- **Verify precision labels.** MI300X "FP32" is effectively pure FP32 because the
  FP16 FC path is NVIDIA-gated; on your NVIDIA build that gate flips ON, so
  re-check the precision mix before fitting `e_mfma(p)`.
- **Expect systematic underprediction** of saturated runs (MI300X: −13.7%±2.3) —
  the missing stall/wait power term (`P_stall≈197–216 W/GPU`), an open modelling
  gap on any stack.

## 9. Dependencies + how to extract the embedded files

| File | Runtime | Third-party deps |
|---|---|---|
| `measure_power.sh`, `gpu_power_sampler.sh` | bash + `nvidia-smi` | none |
| `analyze_power.py` | `python3` | stdlib only (matplotlib optional for PNG) |
| `compose_energy.py`, `revalidate_fulldata.py` | `python3` | **stdlib only** |
| `analyze_roofline.py` | `python3` | **numpy** |
| `fit_dvfs_cubic.py` | `python3` | **numpy** |
| `build_powercap_summary.py` | `python3` | stdlib only |
| `component_loop.cu` + model | `nvcc`, cuDNN, cuBLAS, NCCL | Section 5 |
| counter pass | `ncu` (or rocprofv3) | — |

Target `python3 ≥ 3.8` (the AMD side used `/usr/bin/python3.11`). Only two
scripts need `pip install numpy`.

**To extract:** each block in Part 2 is headed by its path. Save the block's
contents to that path (recreate the `power_analysis/`, `tools/`, and repo-root
layout). Make the shells executable (`chmod +x *.sh`). Or paste this whole file
and split on the `#### FILE:` markers.

---

# PART 2 — EMBEDDED SOURCE (verbatim)

Each section below is one file. The path in the heading is where it belongs
relative to the repo root. Contents are byte-exact copies from the AMD repo.


#### FILE: `power_analysis/measure_power.sh`

```bash
#!/usr/bin/env bash
# ============================================================================
# measure_power.sh  --  run a power measurement around ANY command (MI300X)
# ----------------------------------------------------------------------------
# Self-contained: the sampler loop is embedded, so this one file is all you
# need. It wraps whatever command you give it in:
#
#     [idle baseline] -> [your command] -> [cooldown]
#        ^ socket power sampled across the whole run, phases marked ^
#
# and then hands the CSVs to analyze_power.py (same dir) for the numbers.
#
# Run it INSIDE a GPU allocation (amd-smi/rocm-smi only exist on the compute
# node):
#     salloc -p gpu:mi300x --gres=gpu:mi300x:8 -A l0003099 -t 1:00:00
#     ./power_analysis/measure_power.sh -- python train.py ...
#
# ...or call it from inside an sbatch script (it inherits the allocation).
#
# Examples:
#   # measure a containerized training run
#   ./measure_power.sh -t 12000 -- \
#       apptainer exec --rocm pytorch_rocm.sif python -m torch.distributed.run ...
#
#   # measure just idle power (no command) for the baseline floor
#   ./measure_power.sh -b 60 -o results/idle_floor
#
#   # custom windows + 2 Hz sampling
#   ./measure_power.sh -i 0.5 -b 30 -c 15 -w 45 -- ./my_benchmark
# ============================================================================
set -uo pipefail

# ---- defaults --------------------------------------------------------------
INTERVAL=1.0          # seconds between samples (~1 Hz; CLI fork-limited)
IDLE_SECONDS=20       # idle baseline window before the command
COOLDOWN_SECONDS=10   # quiet window after the command
WARMUP_SECONDS=30     # trimmed from front of command window for steady-state
THROUGHPUT=""         # optional img/s hint -> enables J/image, img/s/W
OUT_DIR=""
GPU_TARGET=""         # which physical GPUs to sample; default = your allocation
WARMUP_EPOCHS=""      # if set, measure from epoch <N> onward (epoch-based warmup)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYZER="$SCRIPT_DIR/analyze_power.py"

usage() {
    sed -n '2,/^# ===.*$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  -o DIR     output directory (default: power_analysis/results/measure_<ts>)
  -i SEC     sample interval seconds        (default 1.0)
  -b SEC     idle baseline window seconds    (default 20)
  -c SEC     cooldown window seconds         (default 10)
  -w SEC     steady-state warmup trim seconds(default 30)
  -t IMG/S   throughput hint for efficiency metrics
  -g LIST    GPUs to sample, e.g. "0" or "0,1,2,3" (default: your allocation's
             visible devices; "all" samples every GPU on the node)
  -h         this help
  --         everything after is the command to measure (optional)
EOF
}

# ---- parse options ---------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        -o) OUT_DIR="$2"; shift 2;;
        -i) INTERVAL="$2"; shift 2;;
        -b) IDLE_SECONDS="$2"; shift 2;;
        -c) COOLDOWN_SECONDS="$2"; shift 2;;
        -w) WARMUP_SECONDS="$2"; shift 2;;
        -t) THROUGHPUT="$2"; shift 2;;
        -g) GPU_TARGET="$2"; shift 2;;
        -E) WARMUP_EPOCHS="$2"; shift 2;;
        -h|--help) usage; exit 0;;
        --) shift; break;;
        *)  echo "unknown option: $1" >&2; usage; exit 2;;
    esac
done
CMD=("$@")   # the workload (may be empty -> idle-only measurement)

# ---- pick a telemetry backend (AMD or NVIDIA, so the same harness measures
#      the native-HIP, PyTorch-ROCm, and CUDA builds) ------------------------
if command -v amd-smi >/dev/null 2>&1; then
    BACKEND="amd-smi"
elif command -v rocm-smi >/dev/null 2>&1; then
    BACKEND="rocm-smi"
elif command -v nvidia-smi >/dev/null 2>&1; then
    BACKEND="nvidia-smi"
else
    echo "ERROR: no GPU telemetry tool (amd-smi / rocm-smi / nvidia-smi) on PATH." >&2
    echo "       Are you inside a GPU allocation? (salloc -p gpu:... )" >&2
    exit 1
fi

# ---- which GPUs to sample --------------------------------------------------
# On a SHARED node, sampling "all" captures neighbours' GPUs, so we default to
# the allocation's visible-device mask. -g overrides.
if [ -z "$GPU_TARGET" ]; then
    case "$BACKEND" in
        amd-smi|rocm-smi) GPU_TARGET="${ROCR_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-all}}";;
        nvidia-smi)       GPU_TARGET="${CUDA_VISIBLE_DEVICES:-all}";;
    esac
fi
# only a numeric / comma-numeric mask is usable as device indices
if [ "$GPU_TARGET" != "all" ] && ! [[ "$GPU_TARGET" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "[measure] WARN: visible-device mask '$GPU_TARGET' is not numeric; sampling ALL GPUs" >&2
    GPU_TARGET="all"
fi
_ids_space() { echo "${GPU_TARGET//,/ }"; }   # "0,1" -> "0 1" for amd-smi/rocm-smi

# ---- output dir + markers --------------------------------------------------
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/results/measure_${SLURM_JOB_ID:-$TS}}"
mkdir -p "$OUT_DIR"
POWER_CSV="$OUT_DIR/power_samples.csv"
MARKERS="$OUT_DIR/markers.csv"
CMD_LOG="$OUT_DIR/command.log"
echo "event,timestamp_s" > "$MARKERS"
mark() { echo "$1,$(date +%s.%N)" >> "$MARKERS"; echo "[mark] $1"; }

echo "=== measure_power.sh ==="
echo "  backend     : $BACKEND"
echo "  out dir     : $OUT_DIR"
echo "  interval    : ${INTERVAL}s   idle: ${IDLE_SECONDS}s  cooldown: ${COOLDOWN_SECONDS}s  warmup-trim: ${WARMUP_SECONDS}s"
echo "  sampling GPU: $GPU_TARGET"
[ ${#CMD[@]} -gt 0 ] && echo "  command     : ${CMD[*]}" || echo "  command     : (none -- idle measurement only)"

# ---- embedded sampler ------------------------------------------------------
# amd-smi prints a "missing render/video groups" WARNING to STDOUT on some
# setups, which would corrupt the CSV -- strip any non-CSV noise lines.
_clean() { grep -v -E '^WARNING:|missing the following required groups|^$'; }
_snapshot() { _snapshot_raw | _clean; }
_energy_snapshot() { _energy_raw | _clean; }
_snapshot_raw() {
    case "$BACKEND" in
        amd-smi)
            if [ "$GPU_TARGET" = all ]; then amd-smi metric -g all -p -c -t -u --csv 2>/dev/null
            else amd-smi metric -g $(_ids_space) -p -c -t -u --csv 2>/dev/null; fi ;;
        rocm-smi)
            if [ "$GPU_TARGET" = all ]; then rocm-smi --showpower --showclocks --showtemp --showuse --csv 2>/dev/null
            else rocm-smi -d $(_ids_space) --showpower --showclocks --showtemp --showuse --csv 2>/dev/null; fi ;;
        nvidia-smi)
            local q="index,power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu"
            if [ "$GPU_TARGET" = all ]; then nvidia-smi --query-gpu="$q" --format=csv 2>/dev/null
            else nvidia-smi -i "$GPU_TARGET" --query-gpu="$q" --format=csv 2>/dev/null; fi ;;
    esac
}
_energy_raw() {
    case "$BACKEND" in
        amd-smi)
            if [ "$GPU_TARGET" = all ]; then amd-smi metric -g all --energy --csv 2>/dev/null
            else amd-smi metric -g $(_ids_space) --energy --csv 2>/dev/null; fi ;;
        rocm-smi) rocm-smi --showenergycounter --csv 2>/dev/null ;;
        nvidia-smi)
            if [ "$GPU_TARGET" = all ]; then nvidia-smi --query-gpu=index,total_energy_consumption --format=csv 2>/dev/null
            else nvidia-smi -i "$GPU_TARGET" --query-gpu=index,total_energy_consumption --format=csv 2>/dev/null; fi ;;
    esac
}
_sampler() {
    local header; header="$(_snapshot | head -1)"
    [ -z "$header" ] && { echo "ERROR: backend returned no data" >&2; exit 1; }
    echo "timestamp_s,${header}" > "$POWER_CSV"
    while :; do
        local ts; ts="$(date +%s.%N)"
        _snapshot | tail -n +2 | sed "s/^/${ts},/" >> "$POWER_CSV"
        sleep "$INTERVAL"
    done
}

# ---- energy accumulator BEFORE (exact ΔE ground-truth; best effort) --------
_energy_snapshot > "$OUT_DIR/energy_before.csv" 2>/dev/null || true

# ---- start sampler; guarantee it dies on any exit --------------------------
_sampler &
SPID=$!
cleanup() { [ -n "${SPID:-}" ] && kill -TERM "$SPID" 2>/dev/null; }
trap cleanup EXIT INT TERM
sleep 2   # let the header + first rows land

# ---- phase 1: idle baseline ------------------------------------------------
mark idle_start
sleep "$IDLE_SECONDS"
mark idle_end

# ---- phase 2: the command (train_* names so analyze_power.py picks them up) -
STATUS=0
if [ ${#CMD[@]} -gt 0 ]; then
    mark train_start
    "${CMD[@]}" 2>&1 | tee "$CMD_LOG"
    STATUS=${PIPESTATUS[0]}
    mark train_end
    echo "[measure] command exit status: $STATUS"
else
    # idle-only run: give the analyzer a (degenerate) train window = idle window
    mark train_start; mark train_end
fi

# ---- phase 3: cooldown, stop sampler, energy AFTER -------------------------
sleep "$COOLDOWN_SECONDS"
mark cooldown_end
kill -TERM "$SPID" 2>/dev/null || true
wait "$SPID" 2>/dev/null || true
SPID=""
_energy_snapshot > "$OUT_DIR/energy_after.csv" 2>/dev/null || true

# auto-detect throughput = AVERAGE per-epoch throughput over the MEASURED epochs
# (skip the first WARMUP_EPOCHS). NOTE: do NOT use the last epoch or NGC's
# "train.total_ips" summary -- that line is just a copy of the FINAL epoch, so a
# single outlier final epoch skews J/image and img/s/W (seen as a spurious 5-10%
# shift between 5- and 10-epoch runs at the same cap).
#  (a) PyTorch:  "epoch <n> train.compute_ips : <val>"   (n 0-indexed)
#  (b) native :  "<ep> <lr(0..1)> ... <Time(s)> <Img/sec>"  (ep 1-indexed)
_W="${WARMUP_EPOCHS:-0}"
if [ -z "$THROUGHPUT" ] && [ -f "$CMD_LOG" ]; then
    THROUGHPUT="$(awk -v w="$_W" '$1=="epoch" && $3=="train.compute_ips"{e=$2+0; if(e>=w){s+=$5;n++}} END{if(n)printf "%.2f",s/n}' "$CMD_LOG")"
fi
if [ -z "$THROUGHPUT" ] && [ -f "$CMD_LOG" ]; then
    THROUGHPUT="$(awk -v w="$_W" 'NF>=8 && $1 ~ /^[0-9]+$/ && $2 ~ /^0\.[0-9]+$/ {e=$1+0; if(e>w){s+=$NF;n++}} END{if(n)printf "%.2f",s/n}' "$CMD_LOG")"
fi

# ---- capture per-epoch wall-clock markers the trainer printed --------------
# Trainers emit "[PWRMARK] epoch <n> <unix_seconds>" at each epoch start; fold
# them into markers.csv so the analyzer can do epoch-based warmup/averaging.
if [ -f "$CMD_LOG" ]; then
    grep -aE '\[PWRMARK\] epoch ' "$CMD_LOG" 2>/dev/null \
        | awk '{print "epoch"$3","$4}' >> "$MARKERS"
fi

# ---- analyze ---------------------------------------------------------------
echo ""; echo "=== analysis ==="
THRU_ARG=(); [ -n "$THROUGHPUT" ] && THRU_ARG=(--throughput "$THROUGHPUT")
EPOCH_ARG=(); [ -n "$WARMUP_EPOCHS" ] && EPOCH_ARG=(--warmup-epochs "$WARMUP_EPOCHS")
if [ -f "$ANALYZER" ] && command -v python3 >/dev/null 2>&1; then
    python3 "$ANALYZER" --run-dir "$OUT_DIR" --warmup "$WARMUP_SECONDS" "${THRU_ARG[@]}" "${EPOCH_ARG[@]}" \
        || echo "  (analyzer error; raw CSVs in $OUT_DIR)"
else
    # minimal fallback: mean of the 3rd column (the prefixed power CSV) over all samples
    echo "  analyze_power.py / python3 unavailable -- quick fallback summary:"
    awk -F',' 'NR>1{n++; s+=$3; if($3>mx)mx=$3} END{if(n)printf "    samples=%d  mean(col3)=%.1f  max(col3)=%.1f\n", n, s/n, mx}' "$POWER_CSV"
    echo "    full data: $POWER_CSV  (run analyze_power.py later for proper numbers)"
fi

echo ""; echo "  outputs: $OUT_DIR"
exit "$STATUS"
```

#### FILE: `power_analysis/gpu_power_sampler.sh`

```bash
#!/usr/bin/env bash
# ============================================================================
# gpu_power_sampler.sh  --  background socket-power sampler for AMD Instinct
#                           (MI300X / gfx942), one tidy CSV row per (sample,GPU)
# ----------------------------------------------------------------------------
# WHY this exists:
#   amd-smi / rocm-smi only live on the compute node (and inside the rocm
#   container) -- never on the submit node. So power has to be measured *next
#   to the workload*, inside the SLURM allocation. This script is meant to be
#   launched in the background by a job wrapper, then SIGTERM'd when the
#   workload finishes (power_profile.slurm does exactly that).
#
# WHAT it logs (best-effort, whatever the backend exposes):
#   wall-clock timestamp, GPU index, socket power (W), gfx/mem clocks,
#   temperature, utilization. Column NAMES come straight from the backend's
#   own CSV header, so the analyzer matches them by regex rather than position.
#
# Backend choice:
#   amd-smi  -> supported tool on ROCm 6.x (preferred)
#   rocm-smi -> deprecated but still present; automatic fallback
#
# Sampling rate:
#   Each sample forks the SMI CLI once, so ~1-2 Hz is the practical ceiling.
#   That captures idle / steady-state / peak / scaling / energy fine. For
#   intra-iteration phase power (fwd vs bwd vs allreduce) you need the amdsmi
#   Python binding at 50-100 Hz instead -- see README note in analyze_power.py.
#
# Usage:
#   ./gpu_power_sampler.sh <out_csv> [interval_s]
#   kill -TERM <pid>     # stop cleanly (flushes + reports row count)
# ============================================================================
set -uo pipefail

OUT="${1:?usage: gpu_power_sampler.sh <out_csv> [interval_s]}"
INTERVAL="${2:-1.0}"
mkdir -p "$(dirname "$OUT")"

# --- pick a telemetry backend -----------------------------------------------
if command -v amd-smi >/dev/null 2>&1; then
    BACKEND="amd-smi"
elif command -v rocm-smi >/dev/null 2>&1; then
    BACKEND="rocm-smi"
else
    echo "[sampler] ERROR: neither amd-smi nor rocm-smi found on PATH" >&2
    exit 1
fi
echo "[sampler] backend=$BACKEND interval=${INTERVAL}s out=$OUT" >&2

# --- one CSV line per call, machine-readable --------------------------------
# -g all : every visible GPU. -p power, -c clocks, -t temperature, -u util.
sample_amdsmi() { amd-smi metric -g all -p -c -t -u --csv 2>/dev/null; }
sample_rocmsmi() { rocm-smi --showpower --showclocks --showtemp --showuse --csv 2>/dev/null; }

# amd-smi may print a render/video group WARNING to STDOUT -> strip non-CSV noise
_clean() { grep -v -E '^WARNING:|missing the following required groups|^$'; }
snapshot() { { if [ "$BACKEND" = "amd-smi" ]; then sample_amdsmi; else sample_rocmsmi; fi; } | _clean; }

# --- write header once, prefixed with our wall-clock column -----------------
HEADER="$(snapshot | head -1)"
if [ -z "$HEADER" ]; then
    echo "[sampler] ERROR: backend returned no data (check ROCR/HIP_VISIBLE_DEVICES)" >&2
    exit 1
fi
echo "timestamp_s,${HEADER}" > "$OUT"

# --- sample loop; SIGTERM/SIGINT breaks out cleanly -------------------------
STOP=0
trap 'STOP=1' TERM INT
while [ "$STOP" -eq 0 ]; do
    ts="$(date +%s.%N)"
    snapshot | tail -n +2 | sed "s/^/${ts},/" >> "$OUT"
    sleep "$INTERVAL"
done
echo "[sampler] stopped; wrote $(($(wc -l < "$OUT") - 1)) sample rows to $OUT" >&2
```

#### FILE: `power_analysis/analyze_power.py`

```python
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
```

#### FILE: `power_analysis/compose_energy.py`

```python
#!/usr/bin/env python3
# ============================================================================
# compose_energy.py  --  Component-2 in-situ energy-coefficient pipeline.
# ----------------------------------------------------------------------------
# Extract per-operation energy coefficients for the HIP ResNet-50 from isolated
# component "replay" runs (matrix-engine / HBM-bandwidth / collective classes),
# then compose them with full-training hardware counters to PREDICT full-training
# energy -- in-sample on the native HIP trainer, cross-stack on PyTorch-ROCm.
#
# Model (per training step, aggregated over the active GPUs):
#     E_step ~= e_mfma(p)*F_mfma + e_hbm*B_hbm + e_comm*V_ar + P_idle*t_step
#       e_mfma(p) : J per matrix-engine FLOP, p in {FP32, BF16}
#       e_hbm     : J per HBM byte moved by bandwidth-bound (BN/act/...) kernels
#       e_comm    : J per all-reduced gradient byte (8-GPU bucket all-reduce)
#       P_idle    : ~140 W/GPU static floor  ->  P_idle*t_step is the idle energy
#
# A coefficient is  (per-iteration DYNAMIC energy) / (per-iteration work counted
# by rocprofv3).  Because the isolated replay loop is GPU-saturating and fires
# the SAME kernels every iteration, both the measured energy and the counted work
# scale linearly with iteration count, so the ratio of their *rates*
#     coeff = (E_dyn_J / train_s)  /  (work / busy_s)        [J per unit work]
# is clock/iteration-count independent.  busy_s is the rocprof GPU-busy time
# (sum of kernel total_ms); for a saturating replay it ~= train_s, and the ratio
# only depends on shapes/precision/kernels -- exactly what the replay contract
# guarantees is stable after iteration 1.
#
# ── STAGE 1  extract ────────────────────────────────────────────────────────
# For each component run dir (a measure_power.sh output with a rocprofv3 counter
# CSV alongside) read:
#   * the energy summary  : run_metrics.csv (or coefficients-style JSON) ->
#                           E_dyn_J, train_s, n_measured_epochs, idle_W ...
#   * the rocprofv3 work  : a roofline_summary.csv (preferred -- already friendly-
#                           labelled + per-disp MFMA mops / HBM bytes), OR raw
#                           *_counter_collection.csv passes (aggregated the same
#                           way analyze_roofline.py does it).
# and emit coefficients.json:
#   { e_mfma_fp32, e_mfma_bf16, e_hbm, e_comm, P_idle_W, per_cap{...}, provenance }
#
# ── STAGE 2  validate ───────────────────────────────────────────────────────
# Load coefficients.json + a full-training PMC CSV + that run's measured energy.
# Classify every kernel into {mfma, hbm, comm, other}; compute
#   F_mfma (FP32 & BF16 split), B_hbm, V_ar, t_step  PER STEP, then predict E_step.
# DOUBLE-COUNTING RULE: B_hbm counts ONLY non-mfma, non-comm kernels' bytes --
# conv/GEMM HBM is already inside e_mfma, the collective's inside e_comm; pricing
# those bytes again inflates every prediction.  Report predicted vs measured and
# percent error for native (in-sample) and pytorch (cross-stack).
#
# Pure standard library (csv/json/argparse/...). Runs under /usr/bin/python3.11.
#
# Counter conventions (gfx942), identical to analyze_roofline.py:
#   matrix FLOPs = 512 FLOP/mop * mfma_*_mops * dispatches
#   HBM bytes    = 64 B/req * (TCC_EA0_RDREQ_sum + TCC_EA0_WRREQ_sum)   (per disp)
# ============================================================================
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

# --- repo-relative defaults -------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                       # repo root (parent of power_analysis)


def _p(*parts):
    return os.path.join(_ROOT, *parts)


DEFAULT_NATIVE_PMC = _p("profiles", "native_in1k_retry_52488168", "roofline_summary.csv")
DEFAULT_PT_PMC     = _p("profiles", "pt_amd_52438059", "roofline_summary.csv")

# Matrix-core convention: 512 FLOP per MFMA matrix-MOP (rocprof-compute /
# analyze_roofline MfmaFlops).  HBM: 64 B per EA<->HBM request.
FLOP_PER_MOP = 512.0
BYTES_PER_REQ = 64.0
# Static idle floor per GPU (W) -- MI300X measured idle ~140 W/GPU; used only as a
# fallback when a run dir carries no measured idle_W.
DEFAULT_P_IDLE_W = 140.0


# ── component taxonomy ──────────────────────────────────────────────────────
# Map a rocprof friendly label / category to the energy-model component class.
# Four classes:  mfma  (matrix engine; HBM priced INSIDE e_mfma)
#                hbm   (bandwidth-bound; the ONLY bytes B_hbm prices)
#                comm  (collective all-reduce; bytes priced by e_comm)
#                other (memcpy/memset/optimizer/misc -- tracked, not priced in B_hbm)
def classify_component(label: str, category: str = "") -> str:
    l = (label or "").lower()
    c = (category or "").lower()
    # collective all-reduce first (its kernel name leaks into 'Other')
    if ("rankreduce" in l or "nccl" in l or "allreduce" in l or "all_reduce" in l
            or "rccl" in l or "reducekernel" in l):
        return "comm"
    # matrix engine, split: convolution (implicit-GEMM / Winograd) -> "mfma",
    # rocBLAS/CK GEMM (FC + GEMM-backed layers) -> "gemm". GEMM is ~1.6x less
    # FLOP-efficient than conv, so it carries its OWN coefficient (e_gemm).
    # Category wins over label so "Conv ... (impl. GEMM)" stays conv/mfma.
    if c == "convolution":
        return "mfma"
    if c == "gemm":
        return "gemm"
    if "winograd" in l or "conv" in l:
        return "mfma"
    if "gemm" in l:
        return "gemm"
    # explicit "other" buckets that must NOT be priced as HBM bandwidth
    if c in ("memory op", "optimizer") or "memcpy" in l or "memset" in l \
            or "fill" in l or "initheap" in l or "sgd" in l or "optim" in l \
            or "subtensor" in l:
        return "other"
    # everything bandwidth-bound: BN, ReLU, residual, bias, pool, softmax,
    # reductions, layout transforms (transpose/NCHW<->CNHW/concat), casts, scale.
    if c in ("batchnorm", "element-wise", "reduction", "layout transform"):
        return "hbm"
    # unknown -> treat as bandwidth-bound (it moves bytes); logged in provenance.
    return "hbm"


# ── rocprofv3 CSV readers ───────────────────────────────────────────────────
def _f(s, default=0.0):
    try:
        v = float(s)
        return v if v == v else default          # NaN -> default
    except (TypeError, ValueError):
        return default


def load_roofline_summary(path):
    """Read an analyze_roofline.py roofline_summary.csv into per-kernel rows.

    Returns list of dicts with keys: label, category, dispatches, busy_ms,
    mfma_mops_total, mfma_bf16_mops, mfma_fp16_mops, mfma_fp32_mops,
    hbm_bytes_per_disp.  Values are PER-DISPATCH (mops/bytes) as the file stores
    them; multiply by dispatches for run totals.
    """
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            disp = _f(r.get("dispatches"))
            if disp <= 0:
                continue
            rows.append(dict(
                label=r.get("label", "") or r.get("kernel", ""),
                category=r.get("category", ""),
                dispatches=disp,
                busy_ms=_f(r.get("total_ms")),
                mfma_mops_total=_f(r.get("mfma_total_mops")),
                mfma_fp32_mops=_f(r.get("mfma_fp32_mops")),
                mfma_fp16_mops=_f(r.get("mfma_fp16_mops")),
                mfma_bf16_mops=_f(r.get("mfma_bf16_mops")),
                hbm_bytes_per_disp=_f(r.get("hbm_bytes_per_disp")),
            ))
    return rows


# Raw long-format counter names (analyze_roofline.py vocabulary).
_MFMA_COUNTERS = {
    "fp32": "SQ_INSTS_VALU_MFMA_MOPS_F32",
    "fp16": "SQ_INSTS_VALU_MFMA_MOPS_F16",
    "bf16": "SQ_INSTS_VALU_MFMA_MOPS_BF16",
    "fp64": "SQ_INSTS_VALU_MFMA_MOPS_F64",
}
_RDREQ = "TCC_EA0_RDREQ_sum"
_WRREQ = "TCC_EA0_WRREQ_sum"


def _friendly_from_kernel(name):
    """Tiny fallback labeller for raw counter CSVs (roofline_summary already
    carries proper friendly labels; this only fires for the raw path)."""
    n = (name or "").lower()
    if "rankreduce" in n or "nccl" in n or "allreduce" in n or "rccl" in n:
        return "all-reduce", "Other"
    if "winograd" in n:
        return "Conv Winograd", "Convolution"
    if "conv" in n or "implicitgemm" in n or "miopen" in n and "conv" in n:
        return "Conv", "Convolution"
    if "gemm" in n or "cijk" in n:
        return "GEMM", "GEMM"
    if "batchnorm" in n or "bnorm" in n:
        return "BatchNorm", "BatchNorm"
    if "relu" in n:
        return "ReLU", "Element-wise"
    if "memset" in n or "fill" in n or "memcpy" in n:
        return "memory op", "Memory op"
    return name or "kernel", "Element-wise"


def load_raw_counter_csvs(run_dir):
    """Aggregate raw rocprofv3 *_counter_collection.csv passes (long format:
    one row per (dispatch, counter)) into the SAME per-kernel rows that
    load_roofline_summary returns.  Used when a component run emits raw passes
    instead of a derived roofline_summary.csv.

    Per analyze_roofline.py: a kernel's per-dispatch counter value is the mean
    over its dispatches; HBM bytes = 64*(RDREQ+WRREQ); mfma mops are the raw MOP
    counters.  busy_ms uses the dispatch wall durations when present.
    """
    csvs = sorted(glob.glob(os.path.join(run_dir, "**", "*_counter_collection.csv"),
                            recursive=True))
    if not csvs:
        return []
    # per kernel-name: counter-name -> [sum, ndisp]; plus dispatch count & busy ns
    agg = defaultdict(lambda: dict(counters=defaultdict(lambda: [0.0, 0]),
                                   disp=set(), busy_ns=0.0))
    for path in csvs:
        with open(path, newline="") as fh:
            seen_disp = set()
            for r in csv.DictReader(fh):
                kn = r.get("Kernel_Name", "")
                cn = r.get("Counter_Name", "")
                if not kn or not cn:
                    continue
                a = agg[kn]
                a["counters"][cn][0] += _f(r.get("Counter_Value"))
                # count a dispatch once per (kernel,dispatch_id), and only once
                # toward busy time per pass-file
                did = r.get("Dispatch_Id")
                key = (path, did)
                if key not in seen_disp:
                    seen_disp.add(key)
                    a["disp"].add(did)
                    st, en = _f(r.get("Start_Timestamp")), _f(r.get("End_Timestamp"))
                    if en > st:
                        a["busy_ns"] += (en - st)
                a["counters"][cn][1] += 1
    rows = []
    for kn, a in agg.items():
        ndisp = len(a["disp"]) or 1
        # per-dispatch average of each counter across all its sampled rows
        def cval(name):
            s, n = a["counters"].get(name, [0.0, 0])
            return (s / n) if n else 0.0
        mfma = {p: cval(c) for p, c in _MFMA_COUNTERS.items()}
        mfma_total = sum(mfma.values())
        hbm = BYTES_PER_REQ * (cval(_RDREQ) + cval(_WRREQ))
        label, category = _friendly_from_kernel(kn)
        rows.append(dict(
            label=label, category=category, dispatches=float(ndisp),
            busy_ms=a["busy_ns"] / 1e6,
            mfma_mops_total=mfma_total, mfma_fp32_mops=mfma["fp32"],
            mfma_fp16_mops=mfma["fp16"], mfma_bf16_mops=mfma["bf16"],
            hbm_bytes_per_disp=hbm))
    return rows


def load_pmc(path_or_dir):
    """Accept either a roofline_summary.csv file or a run dir containing one (or
    raw counter passes). Returns the per-kernel row list."""
    if os.path.isdir(path_or_dir):
        rs = os.path.join(path_or_dir, "roofline_summary.csv")
        if os.path.isfile(rs):
            return load_roofline_summary(rs)
        return load_raw_counter_csvs(path_or_dir)
    if path_or_dir.endswith(".csv"):
        # could be a roofline_summary or a raw counter CSV
        with open(path_or_dir, newline="") as fh:
            head = fh.readline()
        if "mfma_total_mops" in head or "hbm_bytes_per_disp" in head:
            return load_roofline_summary(path_or_dir)
        if "Counter_Name" in head:
            return load_raw_counter_csvs(os.path.dirname(path_or_dir) or ".")
        return load_roofline_summary(path_or_dir)
    raise SystemExit(f"unrecognised PMC source: {path_or_dir}")


# ── per-kernel work aggregation into model terms ────────────────────────────
def aggregate_work(rows):
    """Sum rocprof per-kernel rows into the model's work terms (RUN TOTALS, not
    per-step). Applies the component taxonomy + double-counting rule.

    Returns dict:
      F_mfma_fp32, F_mfma_bf16  : matrix FLOPs (FP32 / BF16 split)
      B_hbm                     : HBM bytes of NON-mfma, NON-comm kernels only
      V_ar                      : bytes moved by collective (comm) kernels
      busy_ms_*                 : GPU-busy time per class (rate denominators)
      by_class                  : {class: {flops, bytes, busy_ms, dispatches}}
      unknown_labels            : labels that fell through to the default class
    """
    out = dict(F_mfma_fp32=0.0, F_mfma_fp16=0.0, F_mfma_bf16=0.0,
               F_gemm=0.0, F_gemm_fp16=0.0, F_gemm_bf16=0.0,
               B_hbm=0.0, V_ar=0.0,
               busy_ms_total=0.0, busy_ms_mfma=0.0, busy_ms_gemm=0.0,
               busy_ms_hbm=0.0, busy_ms_comm=0.0)
    by_class = defaultdict(lambda: dict(flops=0.0, bytes=0.0, busy_ms=0.0,
                                        dispatches=0.0))
    unknown = []
    for r in rows:
        cls = classify_component(r["label"], r["category"])
        disp = r["dispatches"]
        flop = FLOP_PER_MOP * r["mfma_mops_total"] * disp
        bytes_ = r["hbm_bytes_per_disp"] * disp
        bc = by_class[cls]
        bc["flops"] += flop
        bc["bytes"] += bytes_
        bc["busy_ms"] += r["busy_ms"]
        bc["dispatches"] += disp
        out["busy_ms_total"] += r["busy_ms"]
        if cls == "mfma":
            # split matrix FLOPs by precision via the per-precision mop fractions.
            # FP16 and BF16 share the CDNA3 MFMA peak throughput, so FP16 is priced
            # at the BF16 rate (e_mfma_fp16 := e_mfma_bf16) but tracked separately.
            bf16 = r["mfma_bf16_mops"]
            fp16 = r["mfma_fp16_mops"]
            tot = r["mfma_mops_total"]
            frac_bf16 = (bf16 / tot) if tot > 0 else 0.0
            frac_fp16 = (fp16 / tot) if tot > 0 else 0.0
            out["F_mfma_bf16"] += flop * frac_bf16
            out["F_mfma_fp16"] += flop * frac_fp16
            out["F_mfma_fp32"] += flop * max(0.0, 1.0 - frac_bf16 - frac_fp16)
            out["busy_ms_mfma"] += r["busy_ms"]
        elif cls == "gemm":
            # split GEMM FLOPs by precision exactly like the MFMA term: the GEMM
            # class (rocBLAS/CK, incl. 1x1 convs) runs in whatever precision the
            # run uses, so reduced-precision (fp16/bf16) GEMM must be priced at
            # e_gemm_bf16, not the FP32 e_gemm. FP32 GEMM keeps F_gemm.
            bf16 = r["mfma_bf16_mops"]
            fp16 = r["mfma_fp16_mops"]
            tot = r["mfma_mops_total"]
            frac_bf16 = (bf16 / tot) if tot > 0 else 0.0
            frac_fp16 = (fp16 / tot) if tot > 0 else 0.0
            out["F_gemm_bf16"] += flop * frac_bf16
            out["F_gemm_fp16"] += flop * frac_fp16
            out["F_gemm"] += flop * max(0.0, 1.0 - frac_bf16 - frac_fp16)
            out["busy_ms_gemm"] += r["busy_ms"]
        elif cls == "comm":
            out["V_ar"] += bytes_                  # all-reduced gradient bytes
            out["busy_ms_comm"] += r["busy_ms"]
        elif cls == "hbm":
            out["B_hbm"] += bytes_                  # DOUBLE-COUNT RULE: only here
            out["busy_ms_hbm"] += r["busy_ms"]
            if r["category"] not in ("BatchNorm", "Element-wise", "Reduction",
                                     "Layout transform"):
                unknown.append(r["label"])
        # 'other' bytes are intentionally NOT added to B_hbm.
    out["by_class"] = {k: v for k, v in by_class.items()}
    out["unknown_labels"] = sorted(set(unknown))
    return out


# ── energy-summary reader (run_metrics.csv or JSON) ─────────────────────────
def load_energy_summary(run_dir):
    """Read measure_power.sh / analyze_power.py energy numbers for a run dir.

    Prefers run_metrics.csv (the one-row machine-readable summary). Falls back to
    a JSON with the same keys (energy.json / *_metrics.json).  Returns dict with
    E_dyn_J, E_train_J, train_s, idle_W, steady_W, n_measured_epochs,
    throughput_img_s, gpus_active (None where absent).
    """
    def _g(d, *keys):
        for k in keys:
            if k in d and d[k] not in (None, ""):
                return _f(d[k], None) if _is_num(d[k]) else d[k]
        return None

    rm = os.path.join(run_dir, "run_metrics.csv")
    if os.path.isfile(rm):
        with open(rm, newline="") as fh:
            rows = [r for r in csv.reader(fh) if r]
        d = dict(zip(rows[0], rows[1])) if len(rows) >= 2 else {}
    else:
        d = {}
        for cand in ("energy.json", "coefficients_run.json",
                     os.path.basename(run_dir) + "_metrics.json"):
            p = os.path.join(run_dir, cand)
            if os.path.isfile(p):
                with open(p) as fh:
                    d = json.load(fh)
                break
    return dict(
        E_dyn_J=_f(_g(d, "E_dyn_J", "e_dyn_j"), None),
        E_train_J=_f(_g(d, "E_train_J", "e_train_j"), None),
        train_s=_f(_g(d, "train_s", "train_seconds"), None),
        idle_W=_f(_g(d, "idle_W", "idle_w"), None),
        steady_W=_f(_g(d, "steady_W", "steady_w"), None),
        n_measured_epochs=_f(_g(d, "n_measured_epochs"), None),
        throughput_img_s=_f(_g(d, "throughput_img_s"), None),
        gpus_active=_f(_g(d, "gpus_active"), None),
        _raw=d,
    )


def _is_num(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


# ── STAGE 1: extract coefficients ───────────────────────────────────────────
def load_driver_rates(run_dir):
    """Parse component_loop's analytical [WORK]/[RATE] lines from command.log.

    The driver reports work_per_iter (exact conv FLOPs, or BN/ReLU bytes) and a
    per-block work_rate (work/sec), so we count work WITHOUT a profiler (rocprofv3
    GPU-hangs on the conv backward kernels). Only measured blocks (block>0; block 0
    is the dropped warmup) are used. Returns (work_per_iter, median_work_rate,
    unit) or (None, None, None) if no driver rates are present.
    """
    log = os.path.join(run_dir, "command.log")
    if not os.path.isfile(log):
        return None, None, None
    wpi = unit = None
    rates = []
    with open(log, errors="ignore") as f:
        for ln in f:
            m = re.search(r"\[WORK\].*work_per_iter=([0-9.eE+-]+)\s+unit=(\w+)", ln)
            if m:
                wpi, unit = float(m.group(1)), m.group(2)
            m = re.search(r"\[RATE\] block=(\d+).*work_rate=([0-9.eE+-]+)", ln)
            if m and int(m.group(1)) > 0:
                rates.append(float(m.group(2)))
    if not rates:
        return None, None, None
    rates.sort()
    n = len(rates)
    med = rates[n // 2] if n % 2 else 0.5 * (rates[n // 2 - 1] + rates[n // 2])
    return wpi, med, unit


def extract_coefficient(run_dir, work_term, energy, pmc_path=None):
    """Compute one energy coefficient = (dynamic power) / (work rate).

    work_term: which aggregate field carries this run's work
               ("F_mfma_fp32" | "F_mfma_bf16" | "B_hbm" | "V_ar").
    Returns (coeff [J per unit work], detail dict).

    PREFERRED path: the analytical driver rate (command.log [WORK]/[RATE]) with
    the measured dynamic power P_dyn = E_dyn_J/train_s, so coeff = P_dyn/work_rate
    needs no profiler. This is exact for conv FLOPs and robust to the intermittent
    MIOpen WrW GPU-hang (steady-state rate survives a partial run). Falls back to
    rocprofv3 counter work only when no driver rates are present.
    """
    E_dyn = energy.get("E_dyn_J")
    train_s = energy.get("train_s")
    wpi, drate, dunit = load_driver_rates(run_dir)
    if drate and E_dyn and train_s and train_s > 0:
        p_dyn = E_dyn / train_s                    # W (per active GPU)
        coeff = p_dyn / drate                      # J per unit work
        return coeff, dict(run_dir=run_dir, work_term=work_term,
                           work_total=drate * train_s, busy_s=train_s,
                           class_busy_frac=1.0, E_dyn_J=E_dyn, train_s=train_s,
                           P_dyn_W=p_dyn, work_rate=drate, work_per_iter=wpi,
                           work_unit=dunit, coeff=coeff,
                           method="driver-rate (analytical work, P_dyn/work_rate)")
    # ---- fallback: rocprofv3 counter-based work ----
    rows = load_pmc(pmc_path or run_dir)
    agg = aggregate_work(rows)
    work_total = agg.get(work_term, 0.0)
    busy_s = agg.get("busy_ms_total", 0.0) / 1e3
    class_key = {"F_mfma_fp32": "busy_ms_mfma", "F_mfma_bf16": "busy_ms_mfma",
                 "F_gemm": "busy_ms_gemm",
                 "B_hbm": "busy_ms_hbm", "V_ar": "busy_ms_comm"}[work_term]
    class_busy_s = agg.get(class_key, 0.0) / 1e3
    class_frac = (class_busy_s / busy_s) if busy_s > 0 else None
    E_dyn = energy.get("E_dyn_J")
    train_s = energy.get("train_s")
    detail = dict(run_dir=run_dir, work_term=work_term, work_total=work_total,
                  busy_s=busy_s, class_busy_frac=class_frac,
                  E_dyn_J=E_dyn, train_s=train_s)
    if not (work_total > 0) or not E_dyn or not train_s or not (train_s > 0):
        detail["coeff"] = None
        detail["note"] = "insufficient data (zero work or missing energy)"
        return None, detail
    p_dyn = E_dyn / train_s                       # W
    if busy_s > 0:
        wrate = work_total / busy_s
        coeff = p_dyn / wrate
        detail["method"] = "rate (P_dyn / work_rate)"
    else:
        coeff = E_dyn / work_total                # fallback: energy/work directly
        detail["method"] = "energy/work (no busy time)"
    detail["P_dyn_W"] = p_dyn
    detail["coeff"] = coeff
    return coeff, detail


# Component run dir layout the driver emits. A "spec" is (key, mode, precision,
# work_term).  key is the coefficients.json field it fills.  Directory naming is
# flexible: pass --run <key>=<dir> on the CLI, else we look for conventional
# subdir names under --components-dir.
COMPONENT_SPECS = [
    ("e_mfma_fp32",  "conv_fp32",       "FP32", "F_mfma_fp32"),
    ("e_mfma_bf16",  "conv_bf16",       "BF16", "F_mfma_bf16"),
    ("e_gemm",       "fc",              "FP32", "F_gemm"),
    ("e_hbm",        "bn_relu",         "FP32", "B_hbm"),
    ("e_comm",       "allreduce",       "FP32", "V_ar"),
    ("e_comm_2node", "allreduce_2node", "FP32", "V_ar"),
]
# conventional subdir name fragments to auto-discover a component run
_DIR_HINTS = {
    "e_gemm":       ["fc", "fc-gemm", "fcgemm", "gemm"],
    "e_mfma_fp32":  ["conv_fp32", "conv-fp32", "mfma_fp32", "convfp32", "fp32_conv"],
    "e_mfma_bf16":  ["conv_bf16", "conv-bf16", "mfma_bf16", "convbf16", "bf16_conv"],
    "e_hbm":        ["bn_relu", "bnrelu", "hbm", "bandwidth", "bn"],
    "e_comm":       ["allreduce", "all_reduce", "comm", "nccl", "rccl"],
    "e_comm_2node": ["allreduce_2node", "allreduce_2n", "comm_2node", "rccl_2node",
                     "rccl_2n", "nccl_2node"],
}


def _discover_component_dir(components_dir, key):
    if not components_dir or not os.path.isdir(components_dir):
        return None
    subs = [d for d in glob.glob(os.path.join(components_dir, "*"))
            if os.path.isdir(d)]
    for hint in _DIR_HINTS[key]:
        for d in subs:
            if hint in os.path.basename(d).lower():
                return d
    return None


def stage_extract(args):
    # resolve component run dirs: explicit --run key=dir wins, else discover.
    explicit = {}
    for spec in (args.run or []):
        if "=" not in spec:
            raise SystemExit(f"--run expects key=dir, got: {spec}")
        k, v = spec.split("=", 1)
        explicit[k.strip()] = v.strip()

    coeffs = {}
    provenance = {"runs": {}, "method": "coeff = P_dyn / work_rate (rate-based, "
                  "clock/iteration independent)",
                  "conventions": {"flop_per_mfma_mop": FLOP_PER_MOP,
                                  "bytes_per_hbm_req": BYTES_PER_REQ}}
    p_idle_list = []
    per_cap = defaultdict(dict)

    for key, mode, precision, work_term in COMPONENT_SPECS:
        run_dir = explicit.get(key) or _discover_component_dir(args.components_dir, key)
        if not run_dir or not os.path.isdir(run_dir):
            provenance["runs"][key] = {"status": "MISSING",
                                       "looked_for": _DIR_HINTS[key]}
            coeffs[key] = None
            continue
        energy = load_energy_summary(run_dir)
        coeff, detail = extract_coefficient(run_dir, work_term, energy)
        coeffs[key] = coeff
        detail["mode"] = mode
        detail["precision"] = precision
        provenance["runs"][key] = detail
        if energy.get("idle_W"):
            # idle_W in run_metrics is AGGREGATE over active GPUs; normalise to
            # per-GPU for P_idle_W.
            ngpu = energy.get("gpus_active") or 1
            p_idle_list.append(energy["idle_W"] / max(ngpu, 1))
        # per-cap capture: if the run dir name encodes a cap (cap_<W>), record it
        m = re.search(r"cap[_-]?(\d+)", os.path.basename(run_dir))
        if m:
            per_cap[int(m.group(1))][key] = coeff

    P_idle_W = (sum(p_idle_list) / len(p_idle_list)) if p_idle_list else DEFAULT_P_IDLE_W
    out = dict(
        e_mfma_fp32=coeffs.get("e_mfma_fp32"),
        e_mfma_bf16=coeffs.get("e_mfma_bf16"),
        e_gemm=coeffs.get("e_gemm"),
        e_hbm=coeffs.get("e_hbm"),
        e_comm=coeffs.get("e_comm"),
        e_comm_2node=coeffs.get("e_comm_2node"),
        P_idle_W=P_idle_W,
        units=dict(e_mfma_fp32="J/FLOP", e_mfma_bf16="J/FLOP", e_gemm="J/FLOP",
                   e_hbm="J/byte", e_comm="J/byte", e_comm_2node="J/byte",
                   P_idle_W="W per GPU"),
        provenance=provenance,
    )
    if per_cap:
        out["per_cap"] = {str(c): v for c, v in sorted(per_cap.items())}

    with open(args.coefficients, "w") as fh:
        json.dump(out, fh, indent=2)

    # human-readable echo
    print(f"=== extract -> {args.coefficients} ===")
    for key, _, _, _ in COMPONENT_SPECS:
        v = out.get(key)
        st = provenance["runs"].get(key, {})
        src = st.get("run_dir", st.get("status", "?"))
        print(f"  {key:14s} = {('%.4e' % v) if v else 'n/a':>12s}  "
              f"[{st.get('method','-')}]  {os.path.basename(str(src))}")
    print(f"  {'P_idle_W':14s} = {P_idle_W:8.1f} W/GPU")
    if out.get("per_cap"):
        print(f"  per-cap coefficients for caps: {list(out['per_cap'])}")
    return out


# ── STAGE 2: validate / compose ─────────────────────────────────────────────
def _steps_for(rows, energy, override):
    """Steps the PMC pass captured, for the PER-STEP *display* terms ONLY (the
    energy comparison itself is step-free, see predict_energy).

    rocprofv3 multiplexes counters across passes, so the roofline_summary
    `dispatches` column is a PER-PASS count, not a run total -- consistent only
    within a kernel's own pass. The loss / cross-entropy kernel fires exactly
    once per training step, so its dispatch count is the step count of the pass
    that profiled it; we use it purely to scale the reported per-step work.
    """
    if override:
        return float(override), "cli --steps"
    anchors = [("cross-entropy loss", 1.0), ("softmax-ce bwd", 1.0),
               ("top-1 accuracy", 1.0), ("global avg pool", 1.0)]
    for name, per_step in anchors:
        for r in rows:
            if name == (r["label"] or "").lower() and r["dispatches"] > 0:
                return r["dispatches"] / per_step, f"anchor '{r['label']}'"
    return 1.0, "none (per-step == per-pass total)"


def predict_energy(coeffs, pmc_rows, energy, steps_override=None, ngpu=None, nnodes=1):
    """Compose coefficients with full-training counters -> predicted energy.

    Core composition is RATE-based and STEP-COUNT-FREE, matching how the
    coefficients were extracted (each is J per unit work, == W per unit
    work-rate). Predicted mean per-GPU dynamic power:

        P_dyn_pred = e_mfma_fp32*R_fp32 + e_mfma_bf16*R_bf16
                     + e_hbm*R_hbm + e_comm*R_comm                       [W]

    where R_x = work_x / busy_s is the work-RATE over the profiled GPU-busy time
    (sum of kernel total_ms). Because rocprof multiplexes counters across passes,
    work_x and busy_s are scaled by the SAME pass-dispatch count, so each ratio
    R_x is pass-count invariant -- no fragile step inference enters the energy.

    Full-training predicted DYNAMIC energy (aggregate over active GPUs):
        E_dyn_pred = P_dyn_pred * ngpu * train_s
    compared against the measured E_dyn_J. Predicted TOTAL adds the idle floor
    P_idle*ngpu*train_s. Per-step terms (F_mfma, B_hbm, V_ar, t_step) are also
    reported, scaled by the loss-anchor step count, for the readable table.
    """
    agg = aggregate_work(pmc_rows)
    busy_s_total = agg["busy_ms_total"] / 1e3
    busy_mfma = agg["busy_ms_mfma"] / 1e3
    busy_hbm  = agg["busy_ms_hbm"] / 1e3
    busy_comm = agg["busy_ms_comm"] / 1e3

    # Resolve ngpu early — needed for the comm-term fix below.
    E_dyn = energy.get("E_dyn_J")
    train_s = energy.get("train_s")
    if ngpu is None:
        ngpu = energy.get("gpus_active") or 1
    ngpu = max(ngpu, 1)

    # work-RATES over the profiled GPU-busy window. A class's work is divided by
    # TOTAL busy time (not just its own class busy time): we want the contribution
    # to mean power across the whole step, i.e. work per wall-second of the run.
    # busy_s_total is the per-GPU GPU-busy time the profile spanned.
    bs = busy_s_total if busy_s_total > 0 else 1.0
    R_fp32 = agg["F_mfma_fp32"] / bs
    R_fp16 = agg.get("F_mfma_fp16", 0.0) / bs
    R_bf16 = agg["F_mfma_bf16"] / bs
    R_gemm = agg.get("F_gemm", 0.0) / bs                                 # FP32 GEMM
    R_gemm_bf16 = (agg.get("F_gemm_bf16", 0.0)
                   + agg.get("F_gemm_fp16", 0.0)) / bs                    # reduced-prec GEMM
    R_hbm  = agg["B_hbm"] / bs

    # Comm-term fix: at 1-GPU, oneRankReduce writes the gradient buffer to HBM
    # but is NOT an inter-GPU allreduce. The 1-GPU PMC's V_ar comes entirely from
    # that kernel, giving a spurious R_comm and ~7 W of phantom comm power.
    # At multi-GPU, use the actual gradient volume (97.49 MB for ResNet-50's 161
    # tensors) divided by the real per-step time inferred from measured throughput,
    # not the PMC's V_ar / busy_s (which uses 1-GPU kernel timing).
    _V_AR_REAL = 97.49e6   # ResNet-50 gradient buffer: 161 tensors = 97.49 MB
    _LOCAL_BATCH = 256      # per-GPU batch for all validation runs
    if ngpu <= 1:
        R_comm = 0.0
    else:
        tput = energy.get("throughput_img_s")
        if tput and tput > 0:
            t_step_real = _LOCAL_BATCH * ngpu / tput   # wall-time per allreduce
            R_comm = _V_AR_REAL / t_step_real
        else:
            R_comm = agg["V_ar"] / bs   # fallback if throughput unavailable

    e_mfma_fp32 = coeffs.get("e_mfma_fp32") or 0.0
    e_mfma_bf16 = coeffs.get("e_mfma_bf16") or 0.0
    # FP16 priced at the BF16 rate unless a dedicated e_mfma_fp16 is supplied
    # (FP16 and BF16 share the CDNA3 matrix-engine peak throughput).
    e_mfma_fp16 = coeffs.get("e_mfma_fp16") or e_mfma_bf16
    e_gemm = coeffs.get("e_gemm") or 0.0
    # reduced-precision (fp16/bf16) GEMM coefficient. If not separately measured,
    # fall back to the FP32 e_gemm scaled by the BF16/FP32 MFMA efficiency ratio
    # (GEMM and conv share the matrix engines, so the precision scaling matches).
    _gemm_ratio = (e_mfma_bf16 / e_mfma_fp32) if e_mfma_fp32 else 0.348
    e_gemm_bf16 = coeffs.get("e_gemm_bf16") or (e_gemm * _gemm_ratio)
    e_hbm = coeffs.get("e_hbm") or 0.0
    # Select the comm coefficient based on the number of training nodes.
    # At nnodes>=2 inter-node RCCL traffic (IB/RoCE) is more expensive per byte
    # than intra-node XGMI. Use e_comm_2node when available; fall back to e_comm.
    if nnodes >= 2 and coeffs.get("e_comm_2node"):
        e_comm = coeffs["e_comm_2node"]
    else:
        e_comm = coeffs.get("e_comm") or 0.0
    P_idle = coeffs.get("P_idle_W") or DEFAULT_P_IDLE_W

    # predicted per-GPU dynamic power, by component
    p_fp32 = e_mfma_fp32 * R_fp32
    p_fp16 = e_mfma_fp16 * R_fp16
    p_bf16 = e_mfma_bf16 * R_bf16
    p_gemm = e_gemm * R_gemm
    p_gemm_bf16 = e_gemm_bf16 * R_gemm_bf16
    p_hbm  = e_hbm * R_hbm
    p_comm = e_comm * R_comm
    P_dyn_pred_pergpu = (p_fp32 + p_fp16 + p_bf16
                         + p_gemm + p_gemm_bf16 + p_hbm + p_comm)

    # measured mean dynamic power (aggregate -> per GPU)
    P_dyn_meas_agg = (E_dyn / train_s) if (E_dyn and train_s) else None
    P_dyn_meas_pergpu = (P_dyn_meas_agg / ngpu) if P_dyn_meas_agg else None

    # full-training energy predictions (aggregate over active GPUs)
    E_dyn_pred = (P_dyn_pred_pergpu * ngpu * train_s) if train_s else None
    E_idle = (P_idle * ngpu * train_s) if train_s else None
    E_total_pred = (E_dyn_pred + E_idle) if (E_dyn_pred is not None and E_idle is not None) else None
    E_total_meas = (E_dyn + E_idle) if (E_dyn is not None and E_idle is not None) else None

    # per-step display terms (step anchor; does NOT affect the energy above)
    steps, steps_src = _steps_for(pmc_rows, energy, steps_override)
    steps = max(steps, 1.0)
    F_fp32 = agg["F_mfma_fp32"] / steps
    F_bf16 = agg["F_mfma_bf16"] / steps
    B_hbm  = agg["B_hbm"] / steps
    V_ar   = agg["V_ar"] / steps
    t_step = (busy_s_total / steps) if steps else None

    def err(pred, meas):
        if pred is None or meas in (None, 0):
            return None
        return 100.0 * (pred - meas) / meas

    return dict(
        ngpu=ngpu, busy_s=busy_s_total, train_s=train_s,
        busy_s_by_class=dict(mfma=busy_mfma, gemm=agg.get("busy_ms_gemm", 0.0) / 1e3,
                             hbm=busy_hbm, comm=busy_comm),
        work_rates=dict(R_fp32=R_fp32, R_fp16=R_fp16, R_bf16=R_bf16,
                        R_gemm=R_gemm, R_gemm_bf16=R_gemm_bf16,
                        R_hbm=R_hbm, R_comm=R_comm),
        pred_power_pergpu_W=dict(mfma_fp32=p_fp32, mfma_fp16=p_fp16,
                                 mfma_bf16=p_bf16, gemm=p_gemm,
                                 gemm_bf16=p_gemm_bf16,
                                 hbm=p_hbm, comm=p_comm,
                                 total=P_dyn_pred_pergpu),
        P_dyn_pred_pergpu_W=P_dyn_pred_pergpu,
        P_dyn_meas_pergpu_W=P_dyn_meas_pergpu,
        err_power_pct=err(P_dyn_pred_pergpu, P_dyn_meas_pergpu),
        E_dyn_pred_J=E_dyn_pred, E_dyn_meas_J=E_dyn,
        E_idle_J=E_idle, P_idle_W=P_idle,
        E_total_pred_J=E_total_pred, E_total_meas_J=E_total_meas,
        err_dyn_pct=err(E_dyn_pred, E_dyn),
        err_total_pct=err(E_total_pred, E_total_meas),
        steps=steps, steps_src=steps_src, t_step_s=t_step,
        work_per_step=dict(F_mfma_fp32=F_fp32, F_mfma_bf16=F_bf16,
                           B_hbm=B_hbm, V_ar=V_ar),
        unknown_labels=agg["unknown_labels"],
    )


def stage_validate(args):
    with open(args.coefficients) as fh:
        coeffs = json.load(fh)

    targets = []
    if args.native_pmc:
        targets.append(("native (in-sample)", args.native_pmc, args.native_energy,
                        args.native_steps, args.native_gpus))
    if args.pytorch_pmc:
        targets.append(("pytorch (cross-stack)", args.pytorch_pmc, args.pytorch_energy,
                        args.pytorch_steps, args.pytorch_gpus))
    if not targets:
        raise SystemExit("validate: give at least one of --native-pmc / --pytorch-pmc")

    nnodes = getattr(args, "nnodes", 1)
    print(f"=== coefficients in use (nnodes={nnodes}) ===")
    for k in ("e_mfma_fp32", "e_mfma_bf16", "e_hbm", "e_comm", "e_comm_2node", "P_idle_W"):
        v = coeffs.get(k)
        active = " [ACTIVE for inter-node]" if k == "e_comm_2node" and nnodes >= 2 else ""
        print(f"  {k:14s} = {('%.4e' % v) if isinstance(v,(int,float)) and v else v}{active}")

    def _n(v, fmt="%.1f"):
        return (fmt % v) if isinstance(v, (int, float)) else "n/a"

    results = {}
    print("\n=== validation: predicted vs measured (full-training energy) ===")
    hdr = (f"{'target':22s} {'Pdyn_pred':>10s} {'Pdyn_meas':>10s} {'errP%':>7s}  "
           f"{'Edyn_pred':>11s} {'Edyn_meas':>11s} {'errE%':>7s} {'errTot%':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for name, pmc, energy_dir, steps, gpus in targets:
        rows = load_pmc(pmc)
        en = load_energy_summary(energy_dir) if energy_dir else _energy_from_cli(args, name)
        res = predict_energy(coeffs, rows, en, steps_override=steps, ngpu=gpus,
                             nnodes=getattr(args, "nnodes", 1))
        results[name] = res
        print(f"{name:22s} {_n(res['P_dyn_pred_pergpu_W']):>10s} "
              f"{_n(res['P_dyn_meas_pergpu_W']):>10s} {_n(res['err_power_pct']):>7s}  "
              f"{_n(res['E_dyn_pred_J'],'%.0f'):>11s} {_n(res['E_dyn_meas_J'],'%.0f'):>11s} "
              f"{_n(res['err_dyn_pct']):>7s} {_n(res['err_total_pct']):>8s}")
        # per-GPU predicted dynamic power decomposition + per-step work
        c = res["pred_power_pergpu_W"]
        w = res["work_per_step"]
        fmf = w["F_mfma_fp32"] + w["F_mfma_bf16"]
        print(f"    ngpu={res['ngpu']:.0f} busy={res['busy_s']:.2f}s "
              f"train_s={_n(res['train_s'],'%.1f')}s P_idle={res['P_idle_W']:.0f}W/GPU "
              f"| pred per-GPU dyn W: mfma_fp32={c['mfma_fp32']:.1f} "
              f"mfma_fp16={c.get('mfma_fp16', 0.0):.1f} "
              f"mfma_bf16={c['mfma_bf16']:.1f} "
              f"gemm={c.get('gemm', 0.0):.1f} "
              f"gemm_bf16={c.get('gemm_bf16', 0.0):.1f} "
              f"hbm={c['hbm']:.1f} comm={c['comm']:.1f}")
        print(f"    per-step work (anchor={res['steps_src']}, steps={res['steps']:.0f}): "
              f"F_mfma={fmf:.3e} FLOP  B_hbm={w['B_hbm']/1e9:.2f} GB  "
              f"V_ar={w['V_ar']/1e9:.3f} GB  t_step={(res['t_step_s'] or 0)*1e3:.1f}ms")
        if res["E_total_pred_J"] is not None:
            print(f"    full-train energy (dyn+idle): pred={res['E_total_pred_J']:.0f} J  "
                  f"meas={res['E_total_meas_J']:.0f} J  (idle={res['E_idle_J']:.0f} J)")
        if res["unknown_labels"]:
            print(f"    [warn] labels defaulted to HBM class: {res['unknown_labels']}")

    out = dict(coefficients_file=os.path.abspath(args.coefficients), results=results)
    if args.validation_out:
        with open(args.validation_out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\n  wrote {args.validation_out}")
    return out


def _energy_from_cli(args, name):
    """Build a minimal energy summary from CLI scalars when no run dir is given."""
    if "native" in name:
        return dict(E_dyn_J=args.native_e_dyn, train_s=args.native_train_s,
                    gpus_active=args.native_gpus, n_measured_epochs=None,
                    throughput_img_s=None, idle_W=None, steady_W=None)
    return dict(E_dyn_J=args.pytorch_e_dyn, train_s=args.pytorch_train_s,
                gpus_active=args.pytorch_gpus, n_measured_epochs=None,
                throughput_img_s=None, idle_W=None, steady_W=None)


# ── CLI ─────────────────────────────────────────────────────────────────────
def build_parser():
    ap = argparse.ArgumentParser(
        description="Component-2 energy-coefficient extract + validate pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    # extract
    ex = sub.add_parser("extract", help="extract per-op energy coefficients")
    ex.add_argument("--components-dir", default=_p("power_analysis", "results", "components"),
                    help="dir containing one measure_power.sh run dir per component "
                         "(auto-discovered by name: conv_fp32/conv_bf16/bn_relu/allreduce)")
    ex.add_argument("--run", action="append", metavar="KEY=DIR",
                    help="explicit component run dir, e.g. e_mfma_fp32=/path/run "
                         "(repeatable; overrides discovery)")
    ex.add_argument("--coefficients", default=_p("power_analysis", "coefficients.json"),
                    help="output coefficients JSON")
    ex.set_defaults(func=stage_extract)

    # validate
    va = sub.add_parser("validate", help="compose coefficients -> predict full-training energy")
    va.add_argument("--coefficients", default=_p("power_analysis", "coefficients.json"))
    va.add_argument("--native-pmc", default=DEFAULT_NATIVE_PMC,
                    help="native full-training roofline_summary.csv (or run dir)")
    va.add_argument("--pytorch-pmc", default=DEFAULT_PT_PMC,
                    help="pytorch full-training roofline_summary.csv (or run dir)")
    va.add_argument("--native-energy", default=None,
                    help="run dir with run_metrics.csv for the native energy run")
    va.add_argument("--pytorch-energy", default=None,
                    help="run dir with run_metrics.csv for the pytorch energy run")
    # scalar energy fallbacks (when no run dir is available)
    va.add_argument("--native-e-dyn", type=float, default=None, help="native E_dyn_J (J)")
    va.add_argument("--native-train-s", type=float, default=None, help="native train_s (s)")
    va.add_argument("--pytorch-e-dyn", type=float, default=None, help="pytorch E_dyn_J (J)")
    va.add_argument("--pytorch-train-s", type=float, default=None, help="pytorch train_s (s)")
    # step / gpu overrides
    va.add_argument("--native-steps", type=float, default=None)
    va.add_argument("--pytorch-steps", type=float, default=None)
    va.add_argument("--native-gpus", type=float, default=None)
    va.add_argument("--pytorch-gpus", type=float, default=None)
    va.add_argument("--nnodes", type=int, default=1,
                    help="number of training nodes (1=intra-node e_comm; "
                         ">=2 uses e_comm_2node if available)")
    va.add_argument("--validation-out", default=_p("power_analysis", "validation.json"))
    va.set_defaults(func=stage_validate)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
```

#### FILE: `analyze_roofline.py`

```python
#!/usr/bin/env python3
"""
Roofline-style summary of a rocprofv3 multi-pass run produced by
profile_rn50_amd.slurm.

Reads every pmc_*/<host>/<pid>_counter_collection.csv under a profile root,
aggregates per-kernel averages of each counter, and prints:

    dispatches | avg_us | total_ms | FP32_TFLOPS | FP16_TFLOPS | HBM_GB/s
    arithmetic_intensity (FLOP/byte) | % of MI300X compute peak | % of HBM peak

Also writes the full per-kernel table to <profile_root>/roofline_summary.csv.

Usage:
    python3 analyze_roofline.py [PROFILE_ROOT]

If PROFILE_ROOT is omitted, prints a list of profile dirs that look usable
(under $PROJECT_DIR/profiles/).
"""

import csv
import glob
import os
import sys
from collections import defaultdict
from statistics import mean

# Share the SINGLE source of truth for kernel renaming with plot_roofline.py, so
# the cache-thrashing table here uses the exact same friendly labels + semantic
# categories as the roofline plot — and so a PyTorch profile lines up 1:1 with
# the native one. Fall back to identity if plot_roofline can't be imported
# (e.g. numpy/pandas/matplotlib not installed in this environment).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from plot_roofline import friendly_label
except Exception:  # pragma: no cover - keep analyzer usable without plot deps
    def friendly_label(kernel_name):
        return kernel_name.split("(")[0][:40], "Other"

# ──────────────────────────────────────────────────────────────────────────
# MI300X (gfx942) speed-of-light reference points
# Source: AMD spec sheet. These are theoretical peaks per GPU.
# ──────────────────────────────────────────────────────────────────────────
PEAK_FP32_TFLOPS    = 163.4   # VALU FMA, no MFMA
PEAK_FP16_TFLOPS    = 326.8   # VALU packed FP16
PEAK_FP16_MFMA_TF   = 1307.4  # Matrix core FP16 (not measured by VALU PMCs)
PEAK_HBM_GBPS       = 5325.0  # HBM3 aggregate
PEAK_L2_GBPS        = 17500.0 # rough; varies by SKU
WAVEFRONT_SIZE      = 64      # AMD wavefront width (lanes per VALU instr)

# ──────────────────────────────────────────────────────────────────────────
# FLOP / byte formulas
#
# Each SQ_INSTS_VALU_* count is the number of VALU instructions issued.
# Multiply by WAVEFRONT_SIZE to convert to per-lane operations:
#
#   FLOPs_fp32 = 64 * (2*FMA_F32 + ADD_F32 + MUL_F32 + TRANS_F32)
#   FLOPs_fp16 = 64 * (2*FMA_F16 + ADD_F16 + MUL_F16 + TRANS_F16)
#
# HBM bytes — TCC_EA0_*REQ_sum counters report request counts. Convention
# (matches rocprof-compute and AMD's own roofline scripts) is one cache line
# (64 B) per request:
#
#   HBM_bytes = 64 * (TCC_EA0_RDREQ_sum + TCC_EA0_WRREQ_sum)
#
# We also expose the more precise estimate built from the 32B/64B subset
# counters as `HBM_bytes_alt = 32*RDREQ_32B + 64*WRREQ_64B`. They differ a bit
# in practice but trend together.
# ──────────────────────────────────────────────────────────────────────────

COUNTERS_OF_INTEREST = {
    # ── Compute throughput (existing) ──
    "SQ_WAVES",
    "SQ_INSTS_VALU_FMA_F32",
    "SQ_INSTS_VALU_ADD_F32",
    "SQ_INSTS_VALU_MUL_F32",
    "SQ_INSTS_VALU_TRANS_F32",
    "SQ_INSTS_VALU_FMA_F16",
    "SQ_INSTS_VALU_ADD_F16",
    "SQ_INSTS_VALU_MUL_F16",
    "SQ_INSTS_VALU_TRANS_F16",
    # ── HBM traffic (existing) ──
    "TCC_EA0_RDREQ_sum",
    "TCC_EA0_WRREQ_sum",
    "TCC_EA0_RDREQ_32B_sum",
    "TCC_EA0_WRREQ_64B_sum",
    # ── L1 vector cache traffic (existing) ──
    "TCP_TCC_READ_REQ_sum",
    "TCP_TCC_WRITE_REQ_sum",
    "TCP_TOTAL_CACHE_ACCESSES_sum",
    # ── Direct L2 cache hit/miss (gfx942-verified) ──
    "TCC_HIT_sum",
    "TCC_MISS_sum",
    # Note: SQC_HITS/SQC_MISSES (scalar-L1 hit) do NOT exist on gfx942, so
    # scalar-L1 hit rate cannot be measured on this GPU.
    # ── MFMA / matrix-core ops (gfx942: F16/BF16/F32/F64 only, no I8) ──
    "SQ_INSTS_VALU_MFMA_MOPS_F32",
    "SQ_INSTS_VALU_MFMA_MOPS_F16",
    "SQ_INSTS_VALU_MFMA_MOPS_BF16",
    "SQ_INSTS_VALU_MFMA_MOPS_F64",
    # ── GPU activity + scheduler efficiency ──
    "GRBM_GUI_ACTIVE",
    "GRBM_COUNT",
    "SQ_BUSY_CU_CYCLES",   # gfx942 name (was SQ_BUSY_CYCLES; SQ_WAIT_ANY n/a)
    # ── LDS traffic ──
    "SQ_INSTS_LDS",        # SQ_WAIT_INST_LDS does not exist on gfx942
}


# ──────────────────────────────────────────────────────────────────────────
# Preflight: map each derived metric to the raw counters it needs, so we can
# tell which outputs are trustworthy for a given profile. rocprofv3 silently
# drops counters it does not recognise on the target GPU/build, so a counter
# listed in pmc.txt is not a guarantee it was collected.
# ──────────────────────────────────────────────────────────────────────────
DERIVED_METRIC_INPUTS = {
    "FP32 VALU TFLOPs": ["SQ_INSTS_VALU_FMA_F32", "SQ_INSTS_VALU_ADD_F32",
                         "SQ_INSTS_VALU_MUL_F32"],
    "HBM GB/s":         ["TCC_EA0_RDREQ_sum", "TCC_EA0_WRREQ_sum"],
    "L1 vector hit %":  ["TCP_TOTAL_CACHE_ACCESSES_sum", "TCP_TCC_READ_REQ_sum",
                         "TCP_TCC_WRITE_REQ_sum"],
    "L2 hit %":         ["TCC_HIT_sum", "TCC_MISS_sum"],
    "MFMA TFLOPs":      ["SQ_INSTS_VALU_MFMA_MOPS_F32", "SQ_INSTS_VALU_MFMA_MOPS_F16",
                         "SQ_INSTS_VALU_MFMA_MOPS_BF16", "SQ_INSTS_VALU_MFMA_MOPS_F64"],
    "GPU active %":     ["GRBM_GUI_ACTIVE", "GRBM_COUNT"],
}


def collect_present_counters(passes):
    """Union of counter names that actually appear in the raw pass data.

    parse_pass_csv only retains COUNTERS_OF_INTEREST, and a counter creates a
    key as soon as it is seen (even with value 0), so this set distinguishes
    "collected" from "silently dropped by rocprof".
    """
    present = set()
    for kernel_dict in passes.values():
        for entry in kernel_dict.values():
            present.update(entry["counters"].keys())
    return present


def preflight_counters(present):
    """Report requested-vs-collected counters and per-metric validity.

    Returns (missing_counters, invalid_metrics). invalid_metrics is the set of
    derived-metric names whose inputs were not collected; compute_roofline
    emits NaN for those so they are never mistaken for a real 0.0.
    """
    requested = set(COUNTERS_OF_INTEREST)
    missing = sorted(requested - present)

    print("=== Preflight: requested vs collected PMC counters ===")
    print(f"  requested : {len(requested)}")
    print(f"  collected : {len(requested & present)}")
    print(f"  MISSING   : {len(missing)}")
    if missing:
        print("  Requested but NOT collected by rocprofv3 (silently dropped —")
        print("  usually an unsupported counter name on this GPU/rocprof build).")
        print("  Confirm exact names with `rocprofv3 --list-avail`.")
        for c in missing:
            print(f"      [x] {c}")
    print()

    print("  Derived-metric validity:")
    invalid = set()
    for metric, inputs in DERIVED_METRIC_INPUTS.items():
        miss = [c for c in inputs if c not in present]
        if miss:
            invalid.add(metric)
            print(f"      [x] {metric:16s} INVALID -> emitted as NaN "
                  f"(missing {', '.join(miss)})")
        else:
            print(f"      [ok] {metric:16s} all inputs collected")
    print()
    return missing, invalid


def find_profile_dirs(project_dir):
    return sorted(glob.glob(os.path.join(project_dir, "profiles", "amd_*")))


def parse_pass_csv(path):
    """
    Returns: dict[kernel_name] -> {
        "dispatches": int,
        "duration_ns_sum": int,
        "counters": dict[counter_name] -> sum_value
    }
    """
    per_kernel = defaultdict(lambda: {
        "dispatches": 0,
        "duration_ns_sum": 0,
        "counters": defaultdict(float),
        "seen_dispatches": set(),
    })

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = row["Kernel_Name"]
            d = row["Dispatch_Id"]
            cn = row["Counter_Name"]
            if cn not in COUNTERS_OF_INTEREST:
                continue
            entry = per_kernel[k]
            try:
                cv = float(row["Counter_Value"])
            except ValueError:
                continue
            entry["counters"][cn] += cv
            # Count each dispatch only once even though each dispatch has one
            # row per counter (2 in our pmc.txt = 2 rows per dispatch).
            if d not in entry["seen_dispatches"]:
                entry["seen_dispatches"].add(d)
                entry["dispatches"] += 1
                try:
                    entry["duration_ns_sum"] += (
                        int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
                    )
                except (KeyError, ValueError):
                    pass

    # Drop the set; we have the count.
    for v in per_kernel.values():
        del v["seen_dispatches"]
    return dict(per_kernel)


def merge_passes(passes):
    """
    Merge per-pass parses into per-kernel aggregated counters with
    per-dispatch averages.

    Returns: dict[kernel] -> {
        "dispatches": <max across passes, treated as ground truth>,
        "avg_duration_ns": float (from pass with the most dispatches),
        "counter_avg_per_disp": dict[counter] -> float,
    }
    """
    all_kernels = set()
    for p in passes.values():
        all_kernels.update(p.keys())

    merged = {}
    for k in all_kernels:
        per_pass_dispatches = []
        per_pass_duration_ns = []
        counter_avg = {}
        for pass_name, kernel_dict in passes.items():
            entry = kernel_dict.get(k)
            if not entry or entry["dispatches"] == 0:
                continue
            per_pass_dispatches.append(entry["dispatches"])
            per_pass_duration_ns.append(
                entry["duration_ns_sum"] / entry["dispatches"]
            )
            for cn, total in entry["counters"].items():
                # Average value of this counter per dispatch in this pass.
                counter_avg[cn] = total / entry["dispatches"]

        if not per_pass_dispatches:
            continue
        merged[k] = {
            "max_dispatches": max(per_pass_dispatches),
            "avg_duration_ns": mean(per_pass_duration_ns),
            "counter_avg_per_disp": counter_avg,
        }
    return merged


def compute_roofline(merged, present=None):
    """
    For each kernel, compute the roofline-style metrics. We report *averages
    per dispatch*, which are well-defined regardless of how many dispatches
    each pass profiled.

    `present` is the set of counter names actually collected (from
    collect_present_counters). Derived metrics whose source counters were not
    collected are emitted as NaN rather than a misleading 0.0, so downstream
    consumers can tell "not measured" from "measured zero". If `present` is
    None, every counter is assumed available (legacy behaviour).
    """
    NAN = float("nan")
    if present is None:
        have_l1 = have_l2 = have_act = have_mfma = True
    else:
        have_l1   = {"TCP_TOTAL_CACHE_ACCESSES_sum", "TCP_TCC_READ_REQ_sum",
                     "TCP_TCC_WRITE_REQ_sum"} <= present
        have_l2   = {"TCC_HIT_sum", "TCC_MISS_sum"} <= present
        have_act  = {"GRBM_GUI_ACTIVE", "GRBM_COUNT"} <= present
        have_mfma = {"SQ_INSTS_VALU_MFMA_MOPS_F32", "SQ_INSTS_VALU_MFMA_MOPS_F16",
                     "SQ_INSTS_VALU_MFMA_MOPS_BF16",
                     "SQ_INSTS_VALU_MFMA_MOPS_F64"} <= present

    rows = []
    for k, v in merged.items():
        c = v["counter_avg_per_disp"]
        dur_s = v["avg_duration_ns"] * 1e-9

        # FLOPs per dispatch (averaged).
        flop_fp32 = WAVEFRONT_SIZE * (
            2 * c.get("SQ_INSTS_VALU_FMA_F32", 0)
            + c.get("SQ_INSTS_VALU_ADD_F32", 0)
            + c.get("SQ_INSTS_VALU_MUL_F32", 0)
            + c.get("SQ_INSTS_VALU_TRANS_F32", 0)
        )
        flop_fp16 = WAVEFRONT_SIZE * (
            2 * c.get("SQ_INSTS_VALU_FMA_F16", 0)
            + c.get("SQ_INSTS_VALU_ADD_F16", 0)
            + c.get("SQ_INSTS_VALU_MUL_F16", 0)
            + c.get("SQ_INSTS_VALU_TRANS_F16", 0)
        )

        # HBM bytes per dispatch.
        hbm_bytes = 64 * (
            c.get("TCC_EA0_RDREQ_sum", 0) + c.get("TCC_EA0_WRREQ_sum", 0)
        )
        hbm_bytes_alt = (
            32 * c.get("TCC_EA0_RDREQ_32B_sum", 0)
            + 64 * c.get("TCC_EA0_WRREQ_64B_sum", 0)
        )

        # Throughputs (per-dispatch averages converted to rates).
        tflops_fp32 = (flop_fp32 / dur_s) / 1e12 if dur_s > 0 else 0
        tflops_fp16 = (flop_fp16 / dur_s) / 1e12 if dur_s > 0 else 0
        hbm_gbps    = (hbm_bytes / dur_s) / 1e9  if dur_s > 0 else 0

        # Arithmetic intensity (FLOP/B). FP32-only for the standard plot.
        ai = flop_fp32 / hbm_bytes if hbm_bytes > 0 else 0

        # ── Cache hit rates (NEW) ─────────────────────────────────────
        # L1 vector cache: misses are the requests forwarded up to L2.
        # NaN => source counters not collected; 0.0 => collected but no traffic.
        tcp_total = c.get("TCP_TOTAL_CACHE_ACCESSES_sum", 0)
        tcp_to_l2 = (c.get("TCP_TCC_READ_REQ_sum", 0)
                     + c.get("TCP_TCC_WRITE_REQ_sum", 0))
        if not have_l1:
            l1_hit_pct = NAN
        else:
            l1_hit_pct = (100.0 * (1.0 - tcp_to_l2 / tcp_total)
                          if tcp_total > 0 else 0.0)
        # L2 cache: direct hit/miss counters.
        l2_hit  = c.get("TCC_HIT_sum",  0)
        l2_miss = c.get("TCC_MISS_sum", 0)
        if not have_l2:
            l2_hit_pct = NAN
        else:
            l2_hit_pct = (100.0 * l2_hit / (l2_hit + l2_miss)
                          if (l2_hit + l2_miss) > 0 else 0.0)
        # Scalar L1 (constant cache) hit rate: SQC_HITS/SQC_MISSES do not exist
        # on gfx942, so this is always unmeasurable (NaN) on MI300X.
        sl1_hit_pct = NAN

        # ── MFMA matrix-core ops (NEW) ────────────────────────────────
        # Matrix-multiply-add op counts per dispatch.  Each op = 1 FMA on
        # a tile of (M*N*K) elements; conversion to TFLOPS varies by
        # tile size and arch, so we expose the op rate (MOPS/dispatch)
        # rather than a synthetic TFLOPS for now.
        mfma_f32  = c.get("SQ_INSTS_VALU_MFMA_MOPS_F32",  0)
        mfma_f16  = c.get("SQ_INSTS_VALU_MFMA_MOPS_F16",  0)
        mfma_bf16 = c.get("SQ_INSTS_VALU_MFMA_MOPS_BF16", 0)
        mfma_f64  = c.get("SQ_INSTS_VALU_MFMA_MOPS_F64",  0)
        mfma_total = mfma_f32 + mfma_f16 + mfma_bf16 + mfma_f64
        # MfmaFlops (rocprof-compute): 512 FLOP per matrix-MOP. This is the
        # matrix-core compute the VALU FP32 counters cannot see, so it is the
        # number to put on the roofline for conv layers.
        if not have_mfma:
            mfma_tflops = NAN
        else:
            mfma_flop = 512.0 * mfma_total
            mfma_tflops = (mfma_flop / dur_s) / 1e12 if dur_s > 0 else 0.0

        # ── GPU activity + SM scheduler (NEW) ─────────────────────────
        grbm_act = c.get("GRBM_GUI_ACTIVE", 0)
        grbm_tot = c.get("GRBM_COUNT", 0)
        if not have_act:
            gpu_active_pct = NAN
        else:
            gpu_active_pct = (100.0 * grbm_act / grbm_tot
                              if grbm_tot > 0 else 0.0)
        # SM(CU)-busy stall fraction needed SQ_WAIT_ANY, which does not exist on
        # gfx942. Report raw CU-busy cycles/dispatch instead (SQ_BUSY_CU_CYCLES).
        sm_busy_pct = NAN
        sq_busy_cu = c.get("SQ_BUSY_CU_CYCLES", 0)

        # ── LDS traffic (SQ_WAIT_INST_LDS stall counter n/a on gfx942) ──
        lds_insts = c.get("SQ_INSTS_LDS", 0)

        label, category = friendly_label(k)
        rows.append({
            "kernel": k,
            "label": label,
            "category": category,
            "dispatches": v["max_dispatches"],
            "avg_us": v["avg_duration_ns"] / 1e3,
            "total_ms": v["max_dispatches"] * v["avg_duration_ns"] / 1e6,
            "fp32_tflops": tflops_fp32,
            "fp16_tflops": tflops_fp16,
            "hbm_gbps": hbm_gbps,
            "ai_fp32": ai,
            "pct_fp32_peak": 100 * tflops_fp32 / PEAK_FP32_TFLOPS,
            "pct_bw_peak":   100 * hbm_gbps    / PEAK_HBM_GBPS,
            "flop_fp32_per_disp": flop_fp32,
            "hbm_bytes_per_disp": hbm_bytes,
            "hbm_bytes_alt_per_disp": hbm_bytes_alt,
            "sq_waves_per_disp": c.get("SQ_WAVES", 0),
            # ── new columns (cache + MFMA + activity + LDS) ──
            "l1_hit_pct":      l1_hit_pct,
            "l2_hit_pct":      l2_hit_pct,
            "sl1_hit_pct":     sl1_hit_pct,
            "mfma_fp32_mops":  mfma_f32,
            "mfma_fp16_mops":  mfma_f16,
            "mfma_bf16_mops":  mfma_bf16,
            "mfma_fp64_mops":  mfma_f64,
            "mfma_i8_mops":    0.0,   # gfx942 has no MFMA INT8 MOPS counter
            "mfma_total_mops": mfma_total,
            "mfma_tflops":     mfma_tflops,   # 512 FLOP/MOP — real matrix-core rate
            "gpu_active_pct":  gpu_active_pct,
            "sm_busy_pct":     sm_busy_pct,
            "sq_busy_cu_per_disp": sq_busy_cu,
            "lds_insts_per_disp":  lds_insts,
        })
    rows.sort(key=lambda r: r["total_ms"], reverse=True)
    return rows


def truncate(s, n):
    return s if len(s) <= n else s[: n - 3] + "..."


def _fmt_pct(v):
    """Format a possibly-NaN percentage; show '   -' for unmeasured (NaN)."""
    return "   -" if v != v else f"{v:.1f}"  # v!=v is True only for NaN


def print_table(rows, top_n=15):
    # Kernel column now shows the FRIENDLY label (shared with plot_roofline), and
    # the L1/L2 hit-rate columns are included because cache-hit % IS the
    # cache-thrashing signal (low L2 hit % under high HBM traffic = thrashing).
    cols = [
        ("Kernel",         28, "label",          "{}",     "<"),
        ("Category",       16, "category",       "{}",     "<"),
        ("disp",            7, "dispatches",     "{:d}",   ">"),
        ("avg_us",          8, "avg_us",         "{:.1f}", ">"),
        ("tot_ms",          9, "total_ms",       "{:.1f}", ">"),
        ("AI_FP32",         8, "ai_fp32",        "{:.2f}", ">"),
        ("HBM_GB/s",        9, "hbm_gbps",       "{:.0f}", ">"),
        ("%BW",             5, "pct_bw_peak",    "{:.1f}", ">"),
        ("L1hit%",          7, "l1_hit_pct",     None,     ">"),
        ("L2hit%",          7, "l2_hit_pct",     None,     ">"),
    ]
    header = "  ".join(
        f"{name:{align}{w}}" for name, w, _, _, align in cols
    )
    print(header)
    print("-" * len(header))
    for r in rows[:top_n]:
        cells = []
        for name, w, k, fmt, align in cols:
            if name in ("Kernel", "Category"):
                val = truncate(str(r[k]), w)
            elif fmt is None:                       # cache % columns (NaN-aware)
                val = _fmt_pct(r[k])
            else:
                val = fmt.format(r[k])
            cells.append(f"{val:{align}{w}}")
        print("  ".join(cells))


def write_csv(rows, out_path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    project_dir = os.environ.get(
        "PROJECT_DIR",
        "/work/scratch/av30nuci/resent50/performance-analysis-dl-models-resnet50",
    )

    if len(sys.argv) > 1:
        profile_root = sys.argv[1]
    else:
        dirs = find_profile_dirs(project_dir)
        if not dirs:
            print(f"No profile dirs found under {project_dir}/profiles/")
            return 1
        print("Usage: analyze_roofline.py <profile_root>")
        print("Available profile dirs:")
        for d in dirs:
            csvs = glob.glob(os.path.join(d, "pmc_*", "*", "*_counter_collection.csv"))
            print(f"  {d}   ({len(csvs)} pass CSVs)")
        return 0

    csv_paths = sorted(glob.glob(os.path.join(
        profile_root, "pmc_*", "*", "*_counter_collection.csv"
    )))
    if not csv_paths:
        print(f"No *_counter_collection.csv found under {profile_root}")
        return 1

    print(f"Profile root : {profile_root}")
    print(f"Passes found : {len(csv_paths)}")
    print()

    passes = {}
    for p in csv_paths:
        pass_name = p.split("/pmc_")[1].split("/", 1)[0]
        passes[pass_name] = parse_pass_csv(p)
        print(f"  pmc_{pass_name}: {sum(v['dispatches'] for v in passes[pass_name].values())} dispatches, "
              f"{len(passes[pass_name])} unique kernels")
    print()

    present = collect_present_counters(passes)
    missing, invalid = preflight_counters(present)

    merged = merge_passes(passes)
    rows = compute_roofline(merged, present)

    print(f"=== Per-kernel roofline summary (top 15 by total_ms) ===")
    print(f"MI300X peaks: FP32 {PEAK_FP32_TFLOPS:.1f} TF | FP16 VALU {PEAK_FP16_TFLOPS:.1f} TF "
          f"| HBM {PEAK_HBM_GBPS:.0f} GB/s")
    print(f"(FP16 MFMA peak = {PEAK_FP16_MFMA_TF:.0f} TF — not captured by VALU counters)")
    print()
    print_table(rows, top_n=15)

    out_csv = os.path.join(profile_root, "roofline_summary.csv")
    write_csv(rows, out_csv)
    print()
    print(f"Full per-kernel table written to: {out_csv}")

    if invalid:
        print()
        print("!" * 70)
        print("WARNING: some derived metrics are INVALID for this profile because")
        print("their source counters were not collected. They are written as NaN")
        print("(not 0.0) so plots/consumers can skip them:")
        for m in sorted(invalid):
            print(f"    - {m}")
        print("Re-profile with valid counter names before trusting these metrics.")
        print("!" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

#### FILE: `amd_to_ncu_raw.py`

```python
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
```

#### FILE: `power_analysis/revalidate_fulldata.py`

```python
#!/usr/bin/env python3
# ============================================================================
# revalidate_fulldata.py -- Stage-2 re-validation of coefficients_final.json
# against the NEW full-ImageNet-1K runs (2026-07-01..03).
# ----------------------------------------------------------------------------
# Supersedes validation_2node_{strong,weak}_gpu*_r1.json, whose 2-node targets
# were 12-iter val-subset runs (window dominated by inter-epoch overhead at
# near-idle power -> the +102..110% "drift" was mostly a target artifact).
#
# Targets (all trained on the full 1.28M-image train split, verified via
# [PreScan] 1281167 / 312-iters-per-epoch in the run logs):
#   in-sample   : native FP32 1-node 8-GPU repeats
#                 mn_52980217/218/219/220 + mn_52981427   (3 measured epochs, n=5)
#                 mn_52991531/532/533/534                 (4 measured epochs, n=4)
#   cross-stack : PyTorch FP32 2-node 16-GPU powercap run 52942558, cap_750
#                 (other caps composed with the SAME 750W coefficients and
#                  reported separately -- expected to drift: coefficients are
#                  cap-specific and per-cap sets were not extracted)
#   2-node nat. : mn_52941669 (16 GPU, BF16 through-mode) -- APPROXIMATE:
#                 priced by swapping e_mfma_fp32->e_mfma_bf16, e_gemm->
#                 e_gemm_bf16 on the FP32-profile MOP counts (same matrix-op
#                 count either precision); B_hbm comes from the FP32 profile
#                 and overstates BF16-through activation bytes, so the hbm
#                 term is an upper bound for this row.
#
# Uses compose_energy.py as a library (load_pmc / load_energy_summary /
# predict_energy) so the composition logic is identical to the canonical tool.
# Native run_metrics carry an empty throughput_img_s; it is filled here as
# n_measured_epochs * 1281167 / train_s so the comm term uses the real step
# time instead of the phantom V_ar/busy fallback.
#
# Extra diagnostic per target: u_est = (per-GPU img/s) / (saturated 1-GPU
# img/s), the duty proxy for the missing stall/wait term; for u_est < 0.95
# the implied stall power P_stall = (P_meas - u*P_pred) / (1-u) is printed --
# the in-situ calibration number for a future e_wait class.
#
# Run:  /usr/bin/python3.11 revalidate_fulldata.py
# Out:  validation_fulldata_20260703.json  (+ table on stdout)
# ============================================================================
import importlib.util
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

spec = importlib.util.spec_from_file_location(
    "compose_energy", os.path.join(HERE, "compose_energy.py"))
ce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ce)

IMAGES_PER_EPOCH = 1281167.0
SAT_IMG_S = {"native": 1250.0, "pytorch": 1516.0}   # measured 1-GPU saturated rates

NATIVE_PMC = os.path.join(ROOT, "profiles", "native_in1k_retry_52488168",
                          "roofline_summary.csv")
PT_PMC = os.path.join(ROOT, "profiles", "pt_amd_52438059", "roofline_summary.csv")
MN = os.path.join(HERE, "results", "native_multinode")
PT_CAP = os.path.join(HERE, "results", "pytorch",
                      "powercap_2node_fp32_52942558", "resnet50")

NATIVE_FP32_RUNS = ["mn_52980217", "mn_52980218", "mn_52980219", "mn_52980220",
                    "mn_52981427", "mn_52991531", "mn_52991532", "mn_52991533",
                    "mn_52991534"]
NATIVE_BF16_2NODE = "mn_52941669"


def fill_throughput(energy):
    if not energy.get("throughput_img_s"):
        ep, ts = energy.get("n_measured_epochs"), energy.get("train_s")
        if ep and ts:
            energy["throughput_img_s"] = ep * IMAGES_PER_EPOCH / ts
    return energy


def validate_one(coeffs, pmc_rows, run_dir, nnodes, stack):
    en = fill_throughput(ce.load_energy_summary(run_dir))
    res = ce.predict_energy(coeffs, pmc_rows, en, nnodes=nnodes)
    tput = en.get("throughput_img_s") or 0.0
    ngpu = res["ngpu"]
    u = (tput / ngpu) / SAT_IMG_S[stack] if tput else None
    p_pred, p_meas = res["P_dyn_pred_pergpu_W"], res["P_dyn_meas_pergpu_W"]
    stall = None
    if u is not None and u < 0.95 and p_meas is not None:
        stall = (p_meas - u * p_pred) / (1.0 - u)
    return dict(run=os.path.basename(run_dir), ngpu=ngpu, nnodes=nnodes,
                tput_img_s=tput, u_est=u,
                P_dyn_pred_W=p_pred, P_dyn_meas_W=p_meas,
                err_power_pct=res["err_power_pct"],
                err_total_pct=res["err_total_pct"],
                E_dyn_pred_J=res["E_dyn_pred_J"], E_dyn_meas_J=res["E_dyn_meas_J"],
                P_stall_implied_W=stall,
                pred_power_pergpu_W=res["pred_power_pergpu_W"])


def main():
    with open(os.path.join(HERE, "coefficients_final.json")) as fh:
        coeffs = json.load(fh)
    native_rows = ce.load_pmc(NATIVE_PMC)
    pt_rows = ce.load_pmc(PT_PMC)

    out = {"coefficients_file": "coefficients_final.json",
           "native_insample_fp32_8gpu": [], "pytorch_crossstack_2node": [],
           "native_2node_bf16_approx": None}

    hdr = (f"{'target':38s} {'u_est':>5s} {'Ppred':>6s} {'Pmeas':>6s} "
           f"{'errP%':>7s} {'errTot%':>8s} {'Pstall':>6s}")
    print(hdr)
    print("-" * len(hdr))

    def row(tag, r):
        f = lambda v, fm="%.1f": (fm % v) if isinstance(v, (int, float)) else "  n/a"
        print(f"{tag:38s} {f(r['u_est'],'%.2f'):>5s} {f(r['P_dyn_pred_W']):>6s} "
              f"{f(r['P_dyn_meas_W']):>6s} {f(r['err_power_pct']):>7s} "
              f"{f(r['err_total_pct']):>8s} {f(r['P_stall_implied_W'],'%.0f'):>6s}")

    # -- in-sample: native FP32 repeats ------------------------------------
    errs = []
    for name in NATIVE_FP32_RUNS:
        r = validate_one(coeffs, native_rows, os.path.join(MN, name), 1, "native")
        out["native_insample_fp32_8gpu"].append(r)
        errs.append(r["err_power_pct"])
        row(f"native FP32 8GPU {name}", r)
    n = len(errs)
    mean = sum(errs) / n
    sd = math.sqrt(sum((e - mean) ** 2 for e in errs) / (n - 1))
    out["native_insample_stats"] = dict(n=n, err_power_mean_pct=mean,
                                        err_power_std_pct=sd)
    print(f"{'  -> in-sample mean +/- std (n=%d)' % n:38s} "
          f"{'':>5s} {'':>6s} {'':>6s} {mean:6.1f}+/-{sd:.1f}")

    # -- cross-stack: PyTorch 2-node, all caps (750 is the matched one) ----
    for cap in sorted(os.listdir(PT_CAP)):
        d = os.path.join(PT_CAP, cap)
        if not (cap.startswith("cap_") and os.path.isdir(d)):
            continue
        r = validate_one(coeffs, pt_rows, d, 2, "pytorch")
        r["cap_W"] = int(cap.split("_")[1])
        r["cap_matched"] = (r["cap_W"] == 750)
        out["pytorch_crossstack_2node"].append(r)
        tag = f"pytorch FP32 2node {cap}" + (" [MATCHED]" if r["cap_matched"] else "")
        row(tag, r)

    # -- native 2-node BF16 through-mode (approximate pricing) -------------
    cb = dict(coeffs)
    cb["e_mfma_fp32"] = coeffs["e_mfma_bf16"]
    cb["e_gemm"] = coeffs.get("e_gemm_bf16") or 4.55e-12
    r = validate_one(cb, native_rows, os.path.join(MN, NATIVE_BF16_2NODE), 2, "native")
    r["note"] = ("BF16 through-mode priced with e_mfma_bf16/e_gemm_bf16 on FP32-"
                 "profile MOP counts; B_hbm from FP32 profile = upper bound")
    out["native_2node_bf16_approx"] = r
    row(f"native BF16 2node {NATIVE_BF16_2NODE} ~", r)

    dst = os.path.join(HERE, "validation_fulldata_20260703.json")
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
```

#### FILE: `power_analysis/fit_dvfs_cubic.py`

```python
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
```

#### FILE: `power_analysis/build_powercap_summary.py`

```python
#!/usr/bin/env python3
"""
Adapter: turn this repo's per-cap run dirs into a power_sweep_summary.csv
(the schema plot_powercap.py expects).

A powercap sweep writes one measure_power.sh run dir per cap, named cap_<W>/
(each with run_metrics.csv + power_summary.csv). This reads them and emits:
  label,cap_w,n_gpus,avg_power_w_per_gpu,mean_sm_clock_mhz,img_per_sec,
  j_per_image,img_per_sec_per_watt,edp,ed2p,dynamic_j_per_image

Usage:
  python build_powercap_summary.py --sweep-dir <dir-with-cap_*> --label "MI300X HIP"
"""
import argparse, csv, os, re
from pathlib import Path

COLS = ["label", "cap_w", "n_gpus", "avg_power_w_per_gpu", "mean_sm_clock_mhz",
        "img_per_sec", "j_per_image", "img_per_sec_per_watt", "edp", "ed2p",
        "dynamic_j_per_image"]


def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def metrics(run):
    p = Path(run) / "run_metrics.csv"
    if not p.is_file():
        return None
    rows = [r for r in csv.reader(open(p)) if r]
    return dict(zip(rows[0], rows[1])) if len(rows) >= 2 else None


def mean_active_sclk(run, thresh=25.0):
    """Mean gfx clock over the GPUs that actually did work this run."""
    p = Path(run) / "power_summary.csv"
    if not p.is_file():
        return None
    vals = []
    for r in csv.DictReader(open(p)):
        idle, steady, sclk = _f(r.get("idle_W")), _f(r.get("steady_W")), _f(r.get("sclk_MHz"))
        if idle is not None and steady is not None and sclk and steady - idle >= thresh:
            vals.append(sclk)
    return sum(vals) / len(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.sweep_dir, "power_sweep_summary.csv")

    rows = []
    for sub in sorted(Path(args.sweep_dir).glob("cap_*")):
        m = re.match(r"cap_(\d+)", sub.name)
        if not m:
            continue
        rec = metrics(sub)
        if not rec or not _f(rec.get("throughput_img_s")):
            print(f"  skip {sub.name}: no throughput (run failed)")
            continue
        cap = int(m.group(1))
        n = _f(rec.get("gpus_active")) or 1
        steady = _f(rec.get("steady_W"))
        tput = _f(rec.get("throughput_img_s"))
        train_s = _f(rec.get("train_s"))
        e_dyn = _f(rec.get("E_dyn_J"))
        dyn_jpi = (e_dyn / (tput * train_s)) if (e_dyn and tput and train_s) else None
        rows.append({
            "label": args.label, "cap_w": cap, "n_gpus": int(n),
            "avg_power_w_per_gpu": f"{steady / n:.2f}" if steady else "",
            "mean_sm_clock_mhz": f"{(mean_active_sclk(sub) or 0):.0f}",
            "img_per_sec": f"{tput:.2f}" if tput else "",
            "j_per_image": rec.get("j_per_image", ""),
            "img_per_sec_per_watt": rec.get("img_s_per_W", ""),
            "edp": rec.get("EDP_Js", ""), "ed2p": rec.get("ED2P_Js2", ""),
            "dynamic_j_per_image": f"{dyn_jpi:.4f}" if dyn_jpi else "",
        })
    rows.sort(key=lambda r: r["cap_w"])
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} caps)")


if __name__ == "__main__":
    main()
```

#### FILE: `power_analysis/extract_coefficients.slurm`

```bash
#!/bin/bash -l
# ============================================================================
# extract_coefficients.slurm  --  Component-2 in-situ energy-coefficient extractor
# ----------------------------------------------------------------------------
# Builds the marker-emitting component-isolation driver (tools/component_loop.cu)
# and, for each isolated component class, measures BOTH:
#   (a) dynamic energy        -> power_analysis/measure_power.sh + analyze_power.py
#   (b) per-kernel HW counters -> rocprofv3 (paired counters, gfx942 rule)
# then composes per-operation coefficients via compose_energy.py:
#
#   E_step ~= e_mfma(p)*F_mfma + e_hbm*B_hbm + e_comm*V_ar + P_idle*t_step
#
# A coefficient = (per-iteration dynamic energy) / (per-iteration counted work).
#
# COMPONENT RUNS (logical label -> component_loop driver mode + env):
#   conv-fp32  -> driver mode `conv`                : FP32 conv fwd+bwd  (matrix-engine)
#   conv-bf16  -> driver mode `conv`  ISO_BF16=1     : pure-BF16 conv     (matrix-engine BF16)
#   bnrelu     -> driver mode `bnrelu`              : BN+ReLU fwd+bwd     (HBM / bandwidth)
#   allreduce  -> driver mode `allreduce`           : 8-GPU bucket all-reduce (comm, needs RCCL)
# All conv runs build the model FP32 (RN50_ENABLE_BF16=0); BF16 conv is the
# pure-BF16 overlay (PrepareIsoBF16) selected by ISO_BF16=1, NOT a BF16 model.
# Driver argv:  component_loop <mode> <local_batch> <iters_per_block> <num_blocks>
#
# DEFAULT: 750 W cap (NO reservation needed). Set CAPS="750 550 450" to also
# sweep caps via rocm-smi --setpoweroverdrive (this path NEEDS the gaod0002
# AMDGPU reservation); a trap ALWAYS restores 750 W on exit.
#
# Submit (default, 750 W, 8 GPUs so allreduce works):
#   sbatch power_analysis/extract_coefficients.slurm
# Cap sweep (reservation required):
#   CAPS="750 550 450" sbatch --reservation=gaod0002 power_analysis/extract_coefficients.slurm
# Conv/bnrelu only on 1 GPU (skip allreduce -> request just 1 GPU at submit):
#   GPUS=1 RUNS="conv-fp32 conv-bf16 bnrelu" \
#     sbatch --gres=gpu:mi300x:1 --partition=gpu:mi300x:1 power_analysis/extract_coefficients.slurm
# ============================================================================
#SBATCH --job-name=rn50_coeff
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --account=l0003099
#SBATCH --partition=acc_short
#SBATCH --gres=gpu:mi300x:8
#SBATCH --time=02:00:00
#SBATCH --output=power_analysis/results/coeff/extract_coeff_%j.out
#SBATCH --error=power_analysis/results/coeff/extract_coeff_%j.err
set -uo pipefail

# --------------------------------------------------------------------------
# paths / env
# --------------------------------------------------------------------------
PROJECT_DIR="/work/scratch/av30nuci/resent50/performance-analysis-dl-models-resnet50"
cd "$PROJECT_DIR"
SIF="${SIF:-$PROJECT_DIR/pytorch_rocm.sif}"
APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"

# knobs ---------------------------------------------------------------------
GPUS="${GPUS:-8}"                                   # default 8 so allreduce works
LOCAL_BATCH="${LOCAL_BATCH:-256}"                   # per-GPU batch for replay shapes
RUNS="${RUNS:-conv-fp32 conv-bf16 bnrelu allreduce}"
CAPS="${CAPS:-}"                                    # empty -> single 750 W run, no reservation
WARMUP_BLOCKS="${WARMUP_BLOCKS:-1}"                 # measure_power.sh -E (discard 1st marked block)
# measure_power windows (match the contract: -i 0.5 -b 15 -c 8 -E 1 -g all)
PWR_IVAL="${PWR_IVAL:-0.5}"; PWR_IDLE="${PWR_IDLE:-15}"; PWR_COOL="${PWR_COOL:-8}"; PWR_WARMUP="${PWR_WARMUP:-30}"
# steady-state loop sizing (driver argv: <iters_per_block> <num_blocks>)
CL_ITERS="${CL_ITERS:-200}"                         # replay iters per [PWRMARK] block
CL_BLOCKS="${CL_BLOCKS:-6}"                         # number of [PWRMARK] blocks (incl. warmup)
PROF_KIR="${PROF_KIR:-1-200}"                       # rocprofv3 kernel window for the counter pass
PROF_TIMEOUT="${PROF_TIMEOUT:-300}"; MAX_RETRY="${MAX_RETRY:-4}"

OUTROOT="$PROJECT_DIR/power_analysis/results/coeff"
RESULTS_DIR="$OUTROOT/run_${SLURM_JOB_ID:-$$}"
mkdir -p "$RESULTS_DIR"

echo "========================================================================"
echo " Component-2 energy-coefficient extraction"
echo " Job $SLURM_JOB_ID  node $SLURM_NODELIST  $(date)"
echo " GPUS=$GPUS  LOCAL_BATCH=$LOCAL_BATCH  RUNS=[$RUNS]  CAPS=[${CAPS:-<none,750W>}]"
echo " CL_BLOCKS=$CL_BLOCKS  CL_ITERS=$CL_ITERS  results -> $RESULTS_DIR"
echo "========================================================================"

[ -f "$SIF" ] || { echo "FAIL: SIF not found at $SIF"; exit 1; }
command -v "$APPTAINER_BIN" >/dev/null 2>&1 || { echo "FAIL: apptainer not on PATH"; exit 1; }

# visible devices: indices 0..GPUS-1
GPU_IDS="$(seq -s, 0 $((GPUS-1)))"
export HIP_VISIBLE_DEVICES="$GPU_IDS"
export ROCR_VISIBLE_DEVICES="$GPU_IDS"

# per-job MIOpen cache so FindEx/BF16 autotune does not collide with other jobs
export MIOPEN_USER_DB_PATH="$PROJECT_DIR/.miopen_coeff_${SLURM_JOB_ID:-$$}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
# disk-backed container TMPDIR for MIOpen's LLVM JIT (apptainer tmpfs is tiny)
HOST_TMP="$PROJECT_DIR/.container_tmp"; mkdir -p "$HOST_TMP"

GFX_ARCH=$(rocminfo 2>/dev/null | awk '/gfx[0-9a-f]+/{for(i=1;i<=NF;i++) if($i ~ /^gfx/){print $i; exit}}')
[ -z "$GFX_ARCH" ] && GFX_ARCH="gfx942"
echo "  GPU arch : $GFX_ARCH"

# --------------------------------------------------------------------------
# 1) BUILD component_loop (contract hipcc recipe: USE_HIP + USE_NCCL)
# --------------------------------------------------------------------------
BUILD_DIR="$RESULTS_DIR/build"; mkdir -p "$BUILD_DIR"
BIN="$BUILD_DIR/component_loop"
SRC_LOOP="$PROJECT_DIR/tools/component_loop.cu"
[ -f "$SRC_LOOP" ] || { echo "FAIL: $SRC_LOOP missing (write the driver first)"; exit 1; }

echo ""; echo "=== [1] build component_loop ==="
"$APPTAINER_BIN" exec --rocm \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --bind "$HOST_TMP:/container_tmp" \
    --env TMPDIR=/container_tmp --env TMP=/container_tmp --env TEMP=/container_tmp \
    "$SIF" bash -lc "
        set -e
        export PATH=/opt/rocm/bin:\$PATH
        export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:\$LD_LIBRARY_PATH
        export ROCM_PATH=/opt/rocm
        HIPCC_FLAGS='-x hip -std=c++17 --offload-arch=$GFX_ARCH -O3 -ffast-math \
            -mwavefrontsize64 -munsafe-fp-atomics -DUSE_HIP -DUSE_NCCL \
            -I $PROJECT_DIR -I /opt/rocm/include'
        echo '  hipcc : '\$(hipcc --version 2>/dev/null | head -1)
        echo '  compile component_loop.cu'
        hipcc \$HIPCC_FLAGS -c '$SRC_LOOP'              -o '$BUILD_DIR/component_loop.o'
        echo '  compile resnet50.cu'
        hipcc \$HIPCC_FLAGS -c '$PROJECT_DIR/resnet50.cu' -o '$BUILD_DIR/resnet50.o'
        echo '  compile miniz.c (as C; -x hip would force C++ and break its tentative defs)'
        hipcc -O2 -c '$PROJECT_DIR/miniz.c'              -o '$BUILD_DIR/miniz.o'
        echo '  link component_loop'
        hipcc --offload-arch=$GFX_ARCH \
            '$BUILD_DIR/component_loop.o' '$BUILD_DIR/resnet50.o' '$BUILD_DIR/miniz.o' \
            -L/opt/rocm/lib -lMIOpen -lhipblas -lrccl -o '$BIN'
    "
RC=$?
[ $RC -eq 0 ] && [ -x "$BIN" ] || { echo "FAIL: build (rc=$RC)"; exit 1; }
echo "  BUILD OK: $BIN"

# --------------------------------------------------------------------------
# power-cap helpers (rocm-smi overdrive; default path skips this entirely)
# --------------------------------------------------------------------------
set_cap()    { /usr/bin/rocm-smi --setpoweroverdrive "$1" --autorespond yes >/dev/null 2>&1; }
restore_cap(){ /usr/bin/rocm-smi --setpoweroverdrive 750 --autorespond yes >/dev/null 2>&1 || true; }
# ALWAYS restore 750 W on any exit if we ever touched the cap.
CAP_TOUCHED=0
trap '[ "$CAP_TOUCHED" = 1 ] && { echo "[trap] restoring 750 W"; restore_cap; }' EXIT INT TERM

# --------------------------------------------------------------------------
# per-run mapping: logical label -> driver mode string + env + #GPUs needed
# (driver modes are conv|bnrelu|allreduce; bf16 conv = mode `conv` + ISO_BF16=1)
# --------------------------------------------------------------------------
run_env() {   # $1 = label -> sets DRV_MODE, MODE_ENV[], MODE_GPUS
    MODE_ENV=(RN50_ENABLE_BF16=0)                   # all runs build the model FP32
    case "$1" in
        conv-fp32) DRV_MODE=conv;       MODE_GPUS=1 ;;
        conv-bf16) DRV_MODE=conv;       MODE_ENV+=(ISO_BF16=1); MODE_GPUS=1 ;;
        bnrelu)    DRV_MODE=bnrelu;      MODE_GPUS=1 ;;
        fc)        DRV_MODE=fc;          MODE_GPUS=1 ;;
        fc-bf16)   DRV_MODE=fc;          MODE_ENV+=(ISO_BF16=1); MODE_GPUS=1 ;;
        step)      DRV_MODE=step;       MODE_GPUS=1 ;;   # full co-located step (additivity probe)
        allreduce) DRV_MODE=allreduce;  MODE_GPUS="$GPUS" ;;
        *) echo "  !! unknown run label: $1"; return 1 ;;
    esac
    return 0
}

# --------------------------------------------------------------------------
# rocprofv3 counter set (gfx942 rule: NEVER collect a counter solo, always pair)
#   matrix work  = mfma_total_mops (SQ_INSTS_VALU_MFMA_MOPS_*)   * 512 FLOP/mop
#   HBM bytes    = (TCC_EA0_RDREQ + TCC_EA0_WRREQ) * 64 B
#   VALU fp32    = 64*(2*FMA_F32 + ADD_F32 + MUL_F32)
# each "line" is one rocprofv3 process so a fault kills only that pass.
# --------------------------------------------------------------------------
COUNTER_LINES=(
  "SQ_INSTS_VALU_MFMA_MOPS_F32 SQ_INSTS_VALU_MFMA_MOPS_BF16"   # matrix work (FP32 + BF16)
  "SQ_INSTS_VALU_FMA_F32 SQ_INSTS_VALU_MFMA_MOPS_F16"          # VALU FMA + BF16/F16 matrix
  "SQ_INSTS_VALU_ADD_F32 SQ_INSTS_VALU_MUL_F32"                # VALU add/mul (fp32)
  "TCC_EA0_RDREQ_sum TCC_EA0_WRREQ_sum"                        # HBM bytes
  "TCC_HIT_sum TCC_MISS_sum"                                   # L2 (context for bandwidth class)
)

# --------------------------------------------------------------------------
# launch the driver inside the container with mode env applied
#   $1 = mode ; $2 = how many GPUs visible ; $3.. = wrapper prefix (e.g. rocprofv3)
# echoes the full apptainer command via "${LAUNCH[@]}"
# --------------------------------------------------------------------------
build_launch() {   # populates global LAUNCH[]
    local label="$1"; shift
    local gpus="$1"; shift
    local rewrite_marker="$1"; shift               # 1 = block->epoch rewrite (energy pass)
    local pre=("$@")                                # profiler/wrapper prefix (may be empty)
    run_env "$label" || return 1
    local ids; ids="$(seq -s, 0 $((gpus-1)))"
    # ALL apptainer flags (incl. every --env) MUST precede "$SIF"; anything after
    # the image is treated as the in-container command. So build the flag list
    # first, append the mode-specific --env, THEN the image, THEN the command.
    LAUNCH=( "$APPTAINER_BIN" exec --rocm --cleanenv
        --env HIP_VISIBLE_DEVICES="$ids" --env ROCR_VISIBLE_DEVICES="$ids"
        --env MIOPEN_USER_DB_PATH="$MIOPEN_USER_DB_PATH"
        --env MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_CUSTOM_CACHE_DIR"
        --env TMPDIR=/container_tmp --env TMP=/container_tmp --env TEMP=/container_tmp
        --bind "$PROJECT_DIR:$PROJECT_DIR"
        --bind "$HOST_TMP:/container_tmp"
        --bind "$MIOPEN_USER_DB_PATH:$MIOPEN_USER_DB_PATH" )
    # mode-specific env (RN50_ENABLE_BF16=0, optional ISO_BF16=1) -- still BEFORE the image
    local e; for e in "${MODE_ENV[@]}"; do LAUNCH+=( --env "$e" ); done
    LAUNCH+=( "$SIF" )
    # The driver emits "[PWRMARK] block <i> <ts>"; measure_power.sh -E only folds
    # "[PWRMARK] epoch ..." markers, so for the energy pass we rewrite block->epoch
    # in the live stream (stdbuf keeps it line-buffered for the 2 Hz sampler).
    local inner="${pre[*]} '$BIN' '$DRV_MODE' '$LOCAL_BATCH' '$CL_ITERS' '$CL_BLOCKS'"
    if [ "$rewrite_marker" = 1 ]; then
        inner="$inner | stdbuf -oL sed -u 's/\[PWRMARK\] block /[PWRMARK] epoch /'"
    fi
    LAUNCH+=( bash -lc "export PATH=/opt/rocm/bin:\$PATH; export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:\$LD_LIBRARY_PATH; \
                        $inner" )
    return 0
}

# --------------------------------------------------------------------------
# (a) ENERGY pass: run under measure_power.sh -> analyze_power.py JSON/CSV
# --------------------------------------------------------------------------
energy_pass() {   # $1 = label ; $2 = gpus ; $3 = out dir
    local mode="$1" gpus="$2" out="$3"
    mkdir -p "$out"
    build_launch "$mode" "$gpus" 1 || return 1      # 1 = rewrite block->epoch markers
    echo "  [energy] $mode  ($gpus GPU)  -> $out"
    bash "$PROJECT_DIR/power_analysis/measure_power.sh" \
        -o "$out" -i "$PWR_IVAL" -b "$PWR_IDLE" -c "$PWR_COOL" -w "$PWR_WARMUP" \
        -E "$WARMUP_BLOCKS" -g all \
        -- "${LAUNCH[@]}"
}

# --------------------------------------------------------------------------
# (b) COUNTER pass: rocprofv3 over a SHORT block (paired counters, retry on fault)
# --------------------------------------------------------------------------
counter_pass() {  # $1 = mode ; $2 = gpus ; $3 = out dir
    local mode="$1" gpus="$2" out="$3"
    mkdir -p "$out"
    if ! "$APPTAINER_BIN" exec --rocm "$SIF" bash -lc 'command -v rocprofv3 >/dev/null'; then
        echo "  [counters] rocprofv3 not in container -- skipping counter pass for $mode"; return 0
    fi
    local idx=0
    for line in "${COUNTER_LINES[@]}"; do
        idx=$((idx+1)); local got=0
        for try in $(seq 1 "$MAX_RETRY"); do
            local t="$out/_try_${idx}_${try}"; rm -rf "$t"; mkdir -p "$t"
            printf 'pmc: %s\n' "$line" > "$t/pmc.txt"
            # rocprofv3 wraps the binary INSIDE the container; restrict the kernel
            # window so a single SHORT block is profiled (values meaningless past iter1).
            # rewrite=0: keep raw output (no pipe) so rocprofv3 sees the real exit.
            build_launch "$mode" "$gpus" 0 \
                rocprofv3 -i "$t/pmc.txt" --kernel-iteration-range "$PROF_KIR" \
                          --output-format csv -d "$t/out" -- || { echo "  bad mode"; return 1; }
            timeout -k 15 "$PROF_TIMEOUT" "${LAUNCH[@]}" > "$t/log" 2>&1
            local rc=$?
            local csv; csv=$(find "$t/out" -name '*_counter_collection.csv' 2>/dev/null | head -1)
            if [ -n "$csv" ] && [ "$(wc -l < "$csv")" -gt 1 ]; then
                mkdir -p "$out/pmc_$idx"; cp -r "$(dirname "$csv")" "$out/pmc_$idx/"
                echo "  [counters] $mode line $idx ok (try $try): $line"; got=1; rm -rf "$t"; break
            fi
            echo "  [counters] $mode line $idx try $try faulted (rc=$rc), retrying: $line"
            cp "$t/log" "$out/fail_line${idx}_try${try}.log" 2>/dev/null || true
            pkill -9 -f "$BIN" 2>/dev/null || true; pkill -9 -f rocprofv3 2>/dev/null || true
            sleep 5; rm -rf "$t"
        done
        [ "$got" = 1 ] || echo "  [counters] $mode line $idx FAILED after $MAX_RETRY tries: $line"
    done
}

# --------------------------------------------------------------------------
# 2) per-component runs (x optional cap sweep)
# --------------------------------------------------------------------------
do_run() {        # $1 = mode ; $2 = cap-tag dir component
    local mode="$1" captag="$2"
    run_env "$mode" || return 1
    local need="$MODE_GPUS"
    if [ "$need" -gt "$GPUS" ]; then
        echo "  !! $mode needs $need GPUs but only $GPUS allocated -- skipping"; return 0
    fi
    local out="$RESULTS_DIR/$captag/$mode"
    # Co-locate: run_metrics.csv lands in $out/ (measure_power.sh -o root), counter
    # CSVs under $out/counters/. compose_energy.py `extract` reads run_metrics.csv
    # from the component run_dir AND recursively globs *_counter_collection.csv, so
    # one run_dir ($out) must hold both.
    energy_pass  "$mode" "$need" "$out"
    # Rate-based extraction (driver self-reports [WORK]/[RATE]) needs no profiler;
    # SKIP_COUNTERS=1 avoids the rocprofv3 GPU-hang on conv backward (WrW) kernels.
    [ "${SKIP_COUNTERS:-0}" = 1 ] || counter_pass "$mode" "$need" "$out/counters"
}

run_all_components() {   # $1 = cap-tag dir component (e.g. cap_750)
    local captag="$1"
    for mode in $RUNS; do
        echo ""; echo "==== component: $mode  [$captag] ===="
        do_run "$mode" "$captag"
    done
}

echo ""; echo "=== [2] component runs ==="
if [ -z "$CAPS" ]; then
    # default: single point at whatever cap is active (750 W); NO reservation needed.
    run_all_components "cap_default"
else
    for CAP in $CAPS; do
        echo ""; echo "######## power cap = ${CAP} W ########"
        CAP_TOUCHED=1
        if set_cap "$CAP"; then echo "  cap set to ${CAP} W"
        else echo "  !! could not set cap ${CAP} W (reservation/permission?) -- running at active cap"; fi
        run_all_components "cap_${CAP}"
    done
    restore_cap; CAP_TOUCHED=0
fi

# --------------------------------------------------------------------------
# 3) COMPOSE coefficients -> coefficients.json
# --------------------------------------------------------------------------
echo ""; echo "=== [3] compose coefficients ==="
COMPOSE="$PROJECT_DIR/power_analysis/compose_energy.py"
NAT_PMC="$PROJECT_DIR/profiles/native_in1k_retry_52488168/roofline_summary.csv"
PT_PMC="$PROJECT_DIR/profiles/pt_amd_52438059/roofline_summary.csv"
if [ ! -f "$COMPOSE" ]; then
    echo "  WARN: $COMPOSE missing -- skipping composition; raw data under $RESULTS_DIR"
else
    # One coefficients.json per cap-tag dir (cap_default, or cap_<W> for a sweep).
    for capdir in "$RESULTS_DIR"/cap_*; do
        [ -d "$capdir" ] || continue
        tag="$(basename "$capdir")"
        coeff="$RESULTS_DIR/coefficients_${tag}.json"
        echo "  [extract] $tag -> $coeff"
        "$APPTAINER_BIN" exec --rocm --bind "$PROJECT_DIR:$PROJECT_DIR" "$SIF" \
            python3 "$COMPOSE" extract \
                --components-dir "$capdir" \
                --coefficients "$coeff" \
            || echo "    (extract error for $tag)"
        # Validate against full-training PMC + measured energy. Single-GPU targets
        # exercise e_mfma_fp32 + e_hbm + P_idle only (e_comm/e_mfma_bf16 are
        # characterized but not exercised by these FP32 1-GPU traces -- scoped
        # honestly in the paper). Pass measured full-training energy via env if set.
        if [ -f "$coeff" ]; then
            echo "  [validate] $tag"
            "$APPTAINER_BIN" exec --rocm --bind "$PROJECT_DIR:$PROJECT_DIR" "$SIF" \
                python3 "$COMPOSE" validate \
                    --coefficients "$coeff" \
                    --native-pmc "$NAT_PMC" \
                    --pytorch-pmc "$PT_PMC" \
                    ${NATIVE_E_DYN:+--native-e-dyn "$NATIVE_E_DYN"} \
                    ${NATIVE_TRAIN_S:+--native-train-s "$NATIVE_TRAIN_S"} \
                    ${PYTORCH_E_DYN:+--pytorch-e-dyn "$PYTORCH_E_DYN"} \
                    ${PYTORCH_TRAIN_S:+--pytorch-train-s "$PYTORCH_TRAIN_S"} \
                    --validation-out "$RESULTS_DIR/validation_${tag}.json" \
                || echo "    (validate error for $tag)"
        fi
    done
fi

echo ""; echo "=== done  $(date) ==="
echo "  results : $RESULTS_DIR"
exit 0
```

#### FILE: `tools/component_loop.cu`

```cpp
// component_loop.cu
//
// Component-2 energy-coefficient driver. Re-fires ONE class of production
// kernels in a long, marker-partitioned steady-state loop so the power harness
// (power_analysis/measure_power.sh) can attribute dynamic energy to a single
// term of the model
//
//   E_step ~= e_mfma(p)*F_mfma + e_hbm*B_hbm + e_comm*V_ar + P_idle*t_step
//
// A coefficient is (per-iteration dynamic energy) / (per-iteration work counted
// by rocprofv3 over the SAME kernels this driver fires). It builds the model
// FP32, runs ONE real Forward()+Backward() warmup (sizes every buffer, selects
// the MIOpen algorithms, populates the saved tensors the Replay* methods read),
// then loops the selected isolation class. Replay values are meaningless after
// iter 1; the kernels, shapes, and precision are not, and energy depends only on
// those.
//
// CLI:
//   component_loop <mode> <local_batch> <iters_per_block> <num_blocks>
//     mode = conv | bnrelu | allreduce | step
//   env RN50_ENABLE_BF16=0   (REQUIRED for the isolation contract; default 0)
//   env ISO_BF16=1           mode=conv only -> pure-BF16 conv overlay path
//                            (PrepareIsoBF16 + ReplayConv*BF16); isolates
//                            e_mfma(BF16) on identical FP32-built layer shapes.
//
// Each block emits  [PWRMARK] epoch <i> <unixtime>  (exact printf+fflush format
// the harness parses with -E), then runs <iters_per_block> iterations of the
// mode's inner loop over all 53 conv layers (or SyncGradients for allreduce),
// then hipDeviceSynchronize() at the block boundary. Block 0 is warmup;
// measure_power.sh -E 1 discards it and averages the rest. Size <iters_per_block>
// so a block sustains ~30-60 s of GPU load at batch 256.
//
// Build (inside pytorch_rocm.sif), per the shared contract:
//   hipcc -x hip -std=c++17 --offload-arch=gfx942 -O3 -ffast-math \
//         -mwavefrontsize64 -munsafe-fp-atomics -DUSE_HIP -DUSE_NCCL \
//         -I <ROOT> -I /opt/rocm/include -c component_loop.cu -o component_loop.o
//   hipcc --offload-arch=gfx942 component_loop.o resnet50.o miniz.o \
//         -L/opt/rocm/lib -lMIOpen -lhipblas -lrccl -o component_loop
//
// allreduce mode is compiled only when USE_NCCL is defined (it is, per the
// recipe). conv/bnrelu/step run single-GPU on device 0.

#include "hip/hip_runtime.h"
#include "resnet50.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <atomic>

#ifdef USE_NCCL
#include <thread>
#endif

// ── HIP error check (driver-local; resnet50.cuh's CUDA_CHECK throws) ──────────
#define CK(call) do { hipError_t _e=(call); if(_e!=hipSuccess){                  \
    fprintf(stderr,"[HIP FAIL] %s:%d %s\n",__FILE__,__LINE__,                    \
            hipGetErrorString(_e)); std::exit(2); } } while(0)

// Wall-clock unix time in seconds (matches train_multigpu.cu:292 PWRMARK clock).
static double unix_now() {
    return std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// EXACT marker format the harness parses: measure_power.sh greps "[PWRMARK]
// epoch " and analyze_power.py matches /epoch(\d+)$/, so the literal token MUST
// be "epoch" (we index by block, but the emitted word is "epoch"). -E <n> drops
// the first n marked blocks.
static void emit_marker(int block) {
    printf("[PWRMARK] epoch %d %.3f\n", block, unix_now());
    fflush(stdout);
}

enum class Mode { Conv, BnRelu, Fc, AllReduce, Step };

static bool parse_mode(const char* s, Mode& m) {
    if (!std::strcmp(s, "conv"))      { m = Mode::Conv;      return true; }
    if (!std::strcmp(s, "bnrelu"))    { m = Mode::BnRelu;    return true; }
    if (!std::strcmp(s, "fc"))        { m = Mode::Fc;        return true; }
    if (!std::strcmp(s, "allreduce")) { m = Mode::AllReduce; return true; }
    if (!std::strcmp(s, "step"))      { m = Mode::Step;      return true; }
    return false;
}

static bool env_flag(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] && std::strcmp(v, "0") != 0;
}

static void usage(const char* prog) {
    fprintf(stderr,
        "usage: %s <mode> <local_batch> <iters_per_block> <num_blocks>\n"
        "  mode = conv | bnrelu | allreduce | step\n"
        "  env  RN50_ENABLE_BF16=0  (required; isolation contract)\n"
        "       ISO_BF16=1          (mode=conv: pure-BF16 conv overlay)\n",
        prog);
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared per-device setup: build FP32, one warmup Forward()+Backward() on
// synthetic input. Returns the conv-layer list and the synthetic buffers (kept
// alive for the whole run). Used by the single-GPU modes and by each allreduce
// worker thread.
// ─────────────────────────────────────────────────────────────────────────────
struct DeviceCtx {
    ResNet50*               model = nullptr;
    hipStream_t             stream = nullptr;
    float*                  d_img = nullptr;
    int*                    d_lab = nullptr;
    std::vector<ConvBnLayer*> convs;
    int N = 0;
};

// Build + warmup. Leaves model in training mode with grads populated by one
// real Backward(). num_classes fixed at 1000 (network shape the PMC data uses).
static void build_and_warmup(DeviceCtx& c, int dev, int N) {
    CK(hipSetDevice(dev));
    CK(hipStreamCreate(&c.stream));
    c.N = N;

    c.model = new ResNet50(1000);
    c.model->SetStream(c.stream);
    c.model->InitWeights();

    const size_t img_elems = (size_t)N * 3 * 224 * 224;
    CK(hipMalloc(&c.d_img, img_elems * sizeof(float)));
    CK(hipMalloc(&c.d_lab, (size_t)N * sizeof(int)));
    gpu_fill_random(c.d_img, (int)img_elems, 1.0f);   // values irrelevant
    CK(hipMemset(c.d_lab, 0, (size_t)N * sizeof(int)));// all label 0

    // ONE real step: sizes buffers, selects MIOpen algos, fills saved tensors.
    c.model->SetTraining(true);
    c.model->ZeroGrad();
    c.model->Forward(c.d_img, N, 224, 224);
    c.model->Backward(c.d_lab, N);
    CK(hipDeviceSynchronize());

    c.convs = c.model->AllConvLayers();
}

static void teardown(DeviceCtx& c) {
    if (c.d_img) hipFree(c.d_img);
    if (c.d_lab) hipFree(c.d_lab);
    delete c.model;
    if (c.stream) hipStreamDestroy(c.stream);
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-GPU isolation modes (conv / bnrelu / step) on device 0.
// ─────────────────────────────────────────────────────────────────────────────
static int run_single_gpu(Mode mode, int N, int iters_per_block, int num_blocks,
                          bool iso_bf16) {
    DeviceCtx c;
    build_and_warmup(c, /*dev=*/0, N);

    if (c.convs.size() != 53) {
        fprintf(stderr, "[WARN] AllConvLayers() returned %zu (expected 53)\n",
                c.convs.size());
    }

    // mode=conv + ISO_BF16: build the pure-BF16 conv overlay once after warmup.
    if (mode == Mode::Conv && iso_bf16) {
        for (auto* l : c.convs) l->PrepareIsoBF16();
        CK(hipDeviceSynchronize());
    }

    const char* tag =
        (mode == Mode::Conv) ? (iso_bf16 ? "conv(BF16-iso)" : "conv(FP32)") :
        (mode == Mode::BnRelu) ? "bnrelu(HBM)" :
        (mode == Mode::Fc) ? (iso_bf16 ? "fc(GEMM-BF16)" : "fc(GEMM-FP32)") :
        "step(full)";
    printf("[component_loop] mode=%s batch=%d iters/block=%d blocks=%d layers=%zu\n",
           tag, N, iters_per_block, num_blocks, c.convs.size());

    // Per-iteration analytical work for this mode (summed over all 53 layers).
    // Coefficient = (steady_W - idle_W) / (work_per_iter * iters_per_sec); both
    // factors are steady-state, so a partial run (intermittent MIOpen WrW hang)
    // still yields a valid rate. bnrelu = HBM bytes; conv/step = matrix FLOPs.
    double work_per_iter = 0.0;
    const char* work_unit;
    if (mode == Mode::BnRelu) {
        for (auto* l : c.convs) work_per_iter += l->ReplayBnReluWorkBytes();
        work_unit = "bytes";
    } else if (mode == Mode::Fc) {
        work_per_iter = c.model->ReplayFcWorkFlops();
        work_unit = "flops";
    } else {  // Conv (FP32 or BF16-iso) and Step both count conv FLOPs
        for (auto* l : c.convs) work_per_iter += l->ReplayConvWorkFlops();
        work_unit = "flops";
    }
    printf("[WORK] mode=%s work_per_iter=%.6e unit=%s\n", tag, work_per_iter, work_unit);
    fflush(stdout);

    for (int blk = 0; blk < num_blocks; ++blk) {
        emit_marker(blk);
        auto b0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iters_per_block; ++it) {
            switch (mode) {
                case Mode::Conv:
                    if (iso_bf16)
                        for (auto* l : c.convs) { l->ReplayConvFwdBF16();
                                                  l->ReplayConvBwdBF16(); }
                    else
                        for (auto* l : c.convs) { l->ReplayConvFwd();
                                                  l->ReplayConvBwd(); }
                    break;
                case Mode::BnRelu:
                    for (auto* l : c.convs) { l->ReplayBnReluFwd();
                                             l->ReplayBnReluBwd(); }
                    break;
                case Mode::Fc:
                    if (iso_bf16) { c.model->ReplayFcFwdBF16();
                                    c.model->ReplayFcBwdBF16(); }
                    else          { c.model->ReplayFcFwd();
                                    c.model->ReplayFcBwd(); }
                    break;
                case Mode::Step:
                    // Full synthetic train step, in-sample control.
                    c.model->ZeroGrad();
                    c.model->Forward(c.d_img, c.N, 224, 224);
                    c.model->Backward(c.d_lab, c.N);
                    c.model->UpdateSGD(0.1f, 0.9f, 1e-4f);
                    break;
                case Mode::AllReduce:  // unreachable here
                    break;
            }
        }
        CK(hipDeviceSynchronize());
        // Per-block rate from completed work (block 0 included; the composer uses
        // the measured blocks). Printed every block so a later hang cannot lose it.
        auto b1 = std::chrono::high_resolution_clock::now();
        double blk_s = std::chrono::duration<double>(b1 - b0).count();
        double rate = (blk_s > 0.0) ? work_per_iter * iters_per_block / blk_s : 0.0;
        printf("[RATE] block=%d iters=%d block_s=%.4f iters_per_s=%.4f work_rate=%.6e %s_per_s\n",
               blk, iters_per_block, blk_s, iters_per_block / blk_s, rate, work_unit);
        fflush(stdout);
    }

    teardown(c);
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// allreduce mode: replicate train_multigpu.cu's ncclCommInitAll over every
// visible GPU, one worker thread per GPU. Each builds a model and runs ONE
// warmup train step, then loops SyncGradients (the bucket lazy-inits on the
// first call). Only rank 0 emits PWRMARKs and prints. Guarded by USE_NCCL.
// ─────────────────────────────────────────────────────────────────────────────
#ifdef USE_NCCL
struct ArWorkerArgs {
    int rank;
    int num_gpus;
    ncclComm_t comm;
    int N;
    int iters_per_block;
    int num_blocks;
};

// All workers must enter each NCCL collective together, so the marker/block
// loop is identical across ranks; only rank 0 writes markers + progress.
static void ar_worker(ArWorkerArgs a) {
    DeviceCtx c;
    build_and_warmup(c, /*dev=*/a.rank, a.N);

    // One warmup SyncGradients so the bucket is built before the timed blocks
    // (its first call also does pack/allreduce/unpack + a stream sync).
    c.model->SyncGradients(a.comm, c.stream);
    CK(hipDeviceSynchronize());

    // V_ar = the all-reduced gradient volume in bytes (the bucket = one FP32
    // gradient per parameter). e_comm = P_dyn / (V_ar * iters_per_sec).
    const double work_per_iter = (double)c.model->ParamCount() * sizeof(float);
    if (a.rank == 0) {
        printf("[component_loop] mode=allreduce gpus=%d local_batch=%d "
               "iters/block=%d blocks=%d\n",
               a.num_gpus, a.N, a.iters_per_block, a.num_blocks);
        printf("[WORK] mode=allreduce work_per_iter=%.6e unit=bytes\n", work_per_iter);
        fflush(stdout);
    }

    for (int blk = 0; blk < a.num_blocks; ++blk) {
        if (a.rank == 0) emit_marker(blk);
        auto b0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < a.iters_per_block; ++it) {
            // SyncGradients re-fires pack(D2D) + bucket all-reduce + unpack(D2D)
            // and ends with a stream sync. This is exactly the e_comm class.
            c.model->SyncGradients(a.comm, c.stream);
        }
        CK(hipDeviceSynchronize());
        if (a.rank == 0) {
            auto b1 = std::chrono::high_resolution_clock::now();
            double blk_s = std::chrono::duration<double>(b1 - b0).count();
            double rate = (blk_s > 0.0) ? work_per_iter * a.iters_per_block / blk_s : 0.0;
            printf("[RATE] block=%d iters=%d block_s=%.4f iters_per_s=%.4f "
                   "work_rate=%.6e bytes_per_s\n",
                   blk, a.iters_per_block, blk_s, a.iters_per_block / blk_s, rate);
            fflush(stdout);
        }
    }

    teardown(c);
}

static int run_allreduce(int N, int iters_per_block, int num_blocks) {
    int num_gpus = 0;
    CK(hipGetDeviceCount(&num_gpus));
    if (num_gpus < 1) { fprintf(stderr, "no GPUs visible\n"); return 1; }

    std::vector<int>        devs(num_gpus);
    std::vector<ncclComm_t> comms(num_gpus);
    for (int i = 0; i < num_gpus; ++i) devs[i] = i;
    NCCL_CHECK(ncclCommInitAll(comms.data(), num_gpus, devs.data()));

    std::vector<std::thread> threads;
    threads.reserve(num_gpus);
    for (int rank = 0; rank < num_gpus; ++rank) {
        ArWorkerArgs a;
        a.rank            = rank;
        a.num_gpus        = num_gpus;
        a.comm            = comms[rank];
        a.N               = N;
        a.iters_per_block = iters_per_block;
        a.num_blocks      = num_blocks;
        threads.emplace_back(ar_worker, a);
    }
    for (auto& t : threads) t.join();

    for (int i = 0; i < num_gpus; ++i) ncclCommDestroy(comms[i]);
    return 0;
}
#endif // USE_NCCL

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    // Force the FP32 isolation build if the operator did not set it (overwrite=0
    // leaves an explicit value untouched). rn50_bf16_mode() treats unset as FP32
    // already, but pinning it removes any ambiguity about the build precision.
    setenv("RN50_ENABLE_BF16", "0", 0);

    if (argc != 5) { usage(argv[0]); return 1; }

    Mode mode;
    if (!parse_mode(argv[1], mode)) {
        fprintf(stderr, "unknown mode '%s'\n", argv[1]);
        usage(argv[0]);
        return 1;
    }
    const int N               = std::atoi(argv[2]);
    const int iters_per_block = std::atoi(argv[3]);
    const int num_blocks      = std::atoi(argv[4]);
    if (N < 1 || iters_per_block < 1 || num_blocks < 1) {
        fprintf(stderr, "local_batch, iters_per_block, num_blocks must be >= 1\n");
        return 1;
    }

    const bool iso_bf16 = env_flag("ISO_BF16");
    if (iso_bf16 && mode != Mode::Conv && mode != Mode::Fc)
        fprintf(stderr, "[WARN] ISO_BF16=1 ignored for mode != conv|fc\n");

    // The isolation contract requires an FP32 build (RN50_ENABLE_BF16=0). Warn
    // loudly if the env is set otherwise so coefficients are not mislabeled.
    if (env_flag("RN50_ENABLE_BF16"))
        fprintf(stderr, "[WARN] RN50_ENABLE_BF16 != 0: replay classes assume an "
                        "FP32 build; coefficients may be mislabeled\n");

    if (mode == Mode::AllReduce) {
#ifdef USE_NCCL
        return run_allreduce(N, iters_per_block, num_blocks);
#else
        fprintf(stderr, "allreduce mode requires a USE_NCCL build\n");
        return 1;
#endif
    }

    return run_single_gpu(mode, N, iters_per_block, num_blocks, iso_bf16);
}
```
