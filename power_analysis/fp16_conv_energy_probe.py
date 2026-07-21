#!/usr/bin/env python3
# ============================================================================
# fp16_conv_energy_probe.py -- PyTorch-only per-FLOP energy probe to test WHY
# the compositional model over-predicts ResNet-50 FP16 (+16.6%) vs BF16 (+5.1%).
#
# IDEA. The matrix FLOP count per fwd+bwd iteration is IDENTICAL across fp16 and
# bf16 (same convolutions, same shapes -- precision does not change FLOP count).
# So we do NOT need to count FLOPs: run the same resnet50 fwd+bwd on the same
# GPU-resident synthetic batch under each precision and compare DYNAMIC POWER per
# iteration. The ratio P_dyn/iters-per-s is the per-FLOP energy ratio directly.
#
# Synthetic GPU-resident data (no dataloader) keeps the matrix engine near 100%
# duty, which also REMOVES the duty-dilution confound present in the full runs.
#
# THREE phases, run in ABBA order across 2 rounds to cancel linear thermal drift:
#   bf16            : autocast bfloat16, no scaler   -> e_bf16 reference
#   fp16_noscaler   : autocast float16,  no scaler   -> e_fp16 (matches run 52861197)
#   fp16_scaler     : autocast float16,  GradScaler  -> data-dependence test
# If fp16_noscaler draws LESS dyn power per iter than bf16 -> FP16 MFMA is
# genuinely cheaper (coefficient bias). If fp16_scaler draws MORE than
# fp16_noscaler -> unscaled gradients were underflowing (data-dependent power).
#
# lr=0 so weights never move -> a reproducible steady regime for all phases.
# Emits [PHASE_START]/[PHASE_END]/[PHASE_RESULT] markers with unix timestamps;
# analyze_fp16_probe.py windows power_samples.csv by them.
#
# Env: PROBE_BATCH(256) PROBE_ITERS(400) PROBE_WARMUP(40) PROBE_ROUNDS(2) RN_ARCH(resnet50)
# ============================================================================
import os, time
import torch
import torchvision.models as tvm
from torch.amp import autocast, GradScaler

def main():
    assert torch.cuda.is_available(), "no CUDA/HIP device"
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True     # fixed shapes -> stable algo choice

    bs     = int(os.environ.get("PROBE_BATCH", "256"))
    iters  = int(os.environ.get("PROBE_ITERS", "400"))
    warmup = int(os.environ.get("PROBE_WARMUP", "40"))
    rounds = int(os.environ.get("PROBE_ROUNDS", "2"))
    arch   = os.environ.get("RN_ARCH", "resnet50")

    print(f"[PROBE] arch={arch} batch={bs} iters={iters} warmup={warmup} rounds={rounds} "
          f"torch={torch.__version__} dev={torch.cuda.get_device_name(0)}", flush=True)

    model = getattr(tvm, arch)(weights=None).to(dev)
    model.train()
    x   = torch.randn(bs, 3, 224, 224, device=dev)
    y   = torch.randint(0, 1000, (bs,), device=dev)
    crit = torch.nn.CrossEntropyLoss()
    opt  = torch.optim.SGD(model.parameters(), lr=0.0)   # lr=0: weights fixed, steady regime

    def run_phase(name, dtype, use_scaler):
        scaler = GradScaler("cuda", enabled=use_scaler)
        for _ in range(warmup):
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=dtype):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        torch.cuda.synchronize()
        t0 = time.time(); print(f"[PHASE_START] {name} {t0:.3f}", flush=True)
        for _ in range(iters):
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=dtype):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        torch.cuda.synchronize()
        t1 = time.time(); print(f"[PHASE_END] {name} {t1:.3f}", flush=True)
        ips = iters * bs / (t1 - t0)
        print(f"[PHASE_RESULT] {name} iters={iters} dur_s={t1-t0:.3f} "
              f"img_s={ips:.1f} loss={float(loss):.3f}", flush=True)

    # idle reference window (GPU quiescent) for the dynamic-power subtraction
    torch.cuda.synchronize()
    ti = time.time(); print(f"[PHASE_START] idle {ti:.3f}", flush=True)
    time.sleep(20.0)
    tj = time.time(); print(f"[PHASE_END] idle {tj:.3f}", flush=True)

    PHASES = [("bf16", torch.bfloat16, False),
              ("fp16_noscaler", torch.float16, False),
              ("fp16_scaler", torch.float16, True)]
    for r in range(rounds):
        order = PHASES if r % 2 == 0 else list(reversed(PHASES))   # ABBA
        for name, dtype, sc in order:
            run_phase(f"{name}#r{r}", dtype, sc)

    print("train.total_ips : 0.0", flush=True)   # keep measure_power.sh's parser happy

if __name__ == "__main__":
    main()
