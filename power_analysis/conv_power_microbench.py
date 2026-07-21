#!/usr/bin/env python3
# ============================================================================
# conv_power_microbench.py -- MATCHED-DUTY ResNet-50 fwd+bwd power microbench.
#
# PURPOSE. Discriminate the two surviving hypotheses for the RN-50 FP16
# no-loss-scaler energy over-prediction (+16.6%): (1) duty dilution (FP16 runs
# idler over wall time, so the busy-rate model over-charges) vs (2) coefficient
# bias (FP16 matrix ops genuinely cost less energy per FLOP than the shared
# e_mfma_bf16 = 3.426 pJ/FLOP they are priced at). The AMP training runs confound
# the two because their duty differs. This bench removes the confound: it hammers
# the SAME ResNet-50 forward+backward on a synthetic GPU-resident batch in a tight
# loop -- no dataloader, no optimizer step, no host sync -- so BOTH precisions run
# at ~identical, ~maximal GPU duty. Any remaining steady-power gap between float16
# and bfloat16 is then a per-FLOP energy effect, not a utilization artifact.
#
# The only difference between the two invocations is the autocast dtype (env DT).
# Emits measure_power.sh-compatible markers so the harness windows blocks 1..N-1
# (block 0 = warmup) and captures throughput.
#
# Env:  DT=float16|bfloat16   BATCH(256)   ITERS_PER_BLOCK(400)   BLOCKS(3)
# Run under measure_power.sh via measure_conv_microbench.slurm.
# ============================================================================
import os, time
import torch
import torchvision.models as tvm

DT = os.environ.get("DT", "float16")
BATCH = int(os.environ.get("BATCH", "256"))
IPB = int(os.environ.get("ITERS_PER_BLOCK", "400"))
BLOCKS = int(os.environ.get("BLOCKS", "3"))
dtype = torch.bfloat16 if DT == "bfloat16" else torch.float16

dev = torch.device("cuda")
torch.backends.cudnn.benchmark = True          # let MIOpen pick fastest algos (as in training)
print(f"[BENCH] dtype={DT} batch={BATCH} iters/block={IPB} blocks={BLOCKS} "
      f"dev={torch.cuda.get_device_name(0)}", flush=True)

model = tvm.resnet50(weights=None).to(dev, memory_format=torch.channels_last)
model.train()
crit = torch.nn.CrossEntropyLoss().to(dev)

# synthetic batch, resident on the GPU for the whole run (no H2D, no loader)
x = torch.randn(BATCH, 3, 224, 224, device=dev).to(memory_format=torch.channels_last)
y = torch.randint(0, 1000, (BATCH,), device=dev)


def run_block(n_iters):
    for _ in range(n_iters):
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype):
            out = model(x)
            loss = crit(out, y)
        loss.backward()                        # conv WrW+dgrad (matrix) + BN/ReLU (hbm/vec)
    torch.cuda.synchronize()


# warmup outside the timed blocks too (MIOpen autotune on first iters)
run_block(30)

for b in range(BLOCKS):
    print(f"[PWRMARK] epoch {b} {time.time():.3f}", flush=True)
    t0 = time.perf_counter()
    run_block(IPB)
    dt = time.perf_counter() - t0
    ips = BATCH * IPB / dt
    # 'epoch <n> train.compute_ips : <v>' -> measure_power.sh averages blocks >= warmup
    print(f"epoch {b} train.compute_ips : {ips:.2f}", flush=True)
    print(f"[BENCH] block {b}: {IPB} iters in {dt:.2f}s -> {ips:.1f} img/s", flush=True)

print("train.total_ips : %.2f" % ips, flush=True)
print("[BENCH] done", flush=True)
