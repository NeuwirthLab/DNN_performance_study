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
#   bnrelu-early -> mode `bnrelu` RN50_BN_STAGE=early: BN on 56x56,28x28 maps (DRAM-stream)
#   bnrelu-late  -> mode `bnrelu` RN50_BN_STAGE=late : BN on 14x14,7x7 maps  (L2-resident)
#       -> the two cache-mix buckets test whether e_hbm is one number or shape-
#          dependent; compare each bucket's dynamic-W/byte AND its TCC_HIT rate.
#          Run alongside plain `bnrelu` (the all-layer control that must equal
#          the byte-weighted mean of the two buckets).
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
# Source tree for resnet50.{cu,cuh}. The root copy was reverted to the 1859-line
# trainer without the Replay*/BnFeatHW methods component_loop needs; the buildable
# copy with them is the isolation worktree (memory project_worktree_isolation_source).
# Override SRC_ROOT to point elsewhere. component_loop.cu + miniz.c stay at root.
SRC_ROOT="${SRC_ROOT:-$PROJECT_DIR/.claude/worktrees/agent-a220fdf7e4a9765ae}"
[ -f "$SRC_ROOT/resnet50.cu" ] || { echo "FAIL: $SRC_ROOT/resnet50.cu missing (set SRC_ROOT)"; exit 1; }

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
            -I $SRC_ROOT -I $PROJECT_DIR -I /opt/rocm/include'
        echo '  hipcc : '\$(hipcc --version 2>/dev/null | head -1)
        echo '  src   : $SRC_ROOT/resnet50.cu'
        echo '  compile component_loop.cu'
        hipcc \$HIPCC_FLAGS -c '$SRC_LOOP'              -o '$BUILD_DIR/component_loop.o'
        echo '  compile resnet50.cu'
        hipcc \$HIPCC_FLAGS -c '$SRC_ROOT/resnet50.cu' -o '$BUILD_DIR/resnet50.o'
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
        bnrelu)       DRV_MODE=bnrelu;   MODE_GPUS=1 ;;
        bnrelu-early) DRV_MODE=bnrelu;   MODE_ENV+=(RN50_BN_STAGE=early); MODE_GPUS=1 ;;
        bnrelu-late)  DRV_MODE=bnrelu;   MODE_ENV+=(RN50_BN_STAGE=late);  MODE_GPUS=1 ;;
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
