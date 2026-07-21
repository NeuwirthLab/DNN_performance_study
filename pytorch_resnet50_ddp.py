"""
ResNet-50 ImageNet DDP training entrypoint.

Self-contained — stock torch + torchvision only. Runs on NVIDIA CUDA and AMD
ROCm PyTorch (HIP) without changes. Launched via `torch.distributed.run`,
which sets RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT.

The end-of-run throughput line is `train.total_ips : <value>` so the awk in
ngc_resnet_benchmark_amd.slurm picks it up unchanged.
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torchvision.transforms as T
import torchvision.models as tvm
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("data", help="ImageNet root (must contain train/)")
    p.add_argument("-b", "--batch-size", type=int, default=256,
                   help="per-GPU batch size")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=1,
                   help="Linear LR warmup epochs (matches train_multigpu.cu)")
    p.add_argument("--lr-schedule", choices=["constant", "cosine"],
                   default="cosine",
                   help="cosine matches train_multigpu.cu; "
                        "constant is a flat LR")
    p.add_argument("--amp-dtype", choices=["float16", "bfloat16"],
                   default="float16",
                   help="autocast dtype when --amp is set; bfloat16 needs no "
                        "GradScaler (FP32-range exponent) and matches the "
                        "native trainer's RN50_ENABLE_BF16 conv path")
    p.add_argument("--amp", action="store_true",
                   help="Mixed precision via torch.amp (FP32 by default to "
                        "match train_multigpu.cu)")
    p.add_argument("--memory-format", choices=["nchw", "channels_last"],
                   default="nchw",
                   help="nchw matches train_multigpu.cu (cuDNN/MIOpen default); "
                        "channels_last (NHWC) may be faster on Tensor Cores")
    p.add_argument("--print-freq", type=int, default=50)
    p.add_argument("--warmup-batches", type=int, default=10,
                   help="Batches to skip before starting the throughput timer")
    p.add_argument("--workspace", default=".",
                   help="Output dir (rank 0 creates if missing)")
    p.add_argument("--profile-steps", type=int, default=0,
                   help="If >0, emit coarse roctx ranges (TrainStep/H2D/"
                        "forward/backward/optim) and stop after this many timed "
                        "steps. Coarse ranges are cheap, so this is also the "
                        "mode to use for a clean OCCUPANCY trace. 0 = normal "
                        "full run (no markers).")
    p.add_argument("--profile-emit-ops", action="store_true",
                   help="Additionally wrap every aten op via emit_nvtx() so "
                        "rocprofv3 --marker-trace can map each GPU kernel to the "
                        "PyTorch op. NOTE: high CPU overhead — use for kernel "
                        "NAMING/attribution, NOT for occupancy measurement.")
    return p.parse_args()


def cosine_lr(base_lr: float, epoch: int, num_epochs: int,
              warmup_epochs: int) -> float:
    # Matches train_multigpu.cu cosine_lr(): linear warmup, then cosine to 0.
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / max(1, warmup_epochs)
    t = epoch - warmup_epochs
    T = num_epochs - warmup_epochs
    if T <= 0:
        return base_lr
    import math
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t / T))


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", init_method="env://")

    if rank == 0:
        os.makedirs(args.workspace, exist_ok=True)
        print(f"[rank0] world_size={world} per_gpu_batch={args.batch_size} "
              f"global_batch={args.batch_size * world} amp={args.amp} "
              f"amp_dtype={args.amp_dtype if args.amp else 'n/a'} "
              f"workers/rank={args.workers}", flush=True)

    train_tf = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    train_ds = ImageFolder(os.path.join(args.data, "train"), train_tf)
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank,
                                 shuffle=True, drop_last=True)
    loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=(args.workers > 0),
                        drop_last=True)

    mem_fmt = (torch.channels_last if args.memory_format == "channels_last"
               else torch.contiguous_format)
    # Model is selectable via RN_ARCH (default resnet50 keeps every existing
    # script/run identical). Any torchvision.models constructor name works, e.g.
    # RN_ARCH=resnet101 for the out-of-sample energy-model validation.
    arch = os.environ.get("RN_ARCH", "resnet50")
    if rank == 0:
        print(f"[ARCH] model = {arch}", flush=True)
    model = getattr(tvm, arch)(weights=None).to(device, memory_format=mem_fmt)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # Loss scaling is only needed for float16 (narrow exponent); bfloat16 has
    # FP32's range, so the scaler stays disabled and step() runs unscaled.
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    # RN50_NO_GRADSCALER=1 forces the loss scaler off even for float16. The
    # scaler's per-step inf/nan check syncs the GPU every iteration, which makes
    # an FP16 run idle/sync-bound; disabling it gives a compute-bound FP16 power
    # measurement (not for convergence -- power/energy characterization only).
    _no_scale = os.environ.get("RN50_NO_GRADSCALER", "0") == "1"
    scaler = GradScaler("cuda",
                        enabled=(args.amp and amp_dtype is torch.float16 and not _no_scale))
    if rank == 0 and args.amp and amp_dtype is torch.float16:
        print(f"[AMP] float16 GradScaler enabled={scaler.is_enabled()}", flush=True)

    # ── Profiling markers ───────────────────────────────────────────────────
    # When --profile-steps>0 we emit two layers of roctx/NVTX ranges so a
    # rocprofv3 --marker-trace run can attribute each otherwise-anonymous GPU
    # kernel to the PyTorch work that launched it:
    #   * coarse ranges  : TrainStep / H2D / forward / backward / optim
    #   * fine ranges    : emit_nvtx() wraps EVERY aten op (e.g.
    #                      "aten::convolution", "aten::batch_norm") in a range,
    #                      so the kernel->op mapping comes for free.
    # nvtx.range_push/pop map to roctxRangePush/Pop on ROCm PyTorch.
    import contextlib
    prof_on = args.profile_steps > 0
    nvtx = torch.cuda.nvtx
    def rng_push(name):
        if prof_on:
            nvtx.range_push(name)
    def rng_pop():
        if prof_on:
            nvtx.range_pop()
    op_marker = (torch.autograd.profiler.emit_nvtx(record_shapes=False)
                 if (prof_on and args.profile_emit_ops)
                 else contextlib.nullcontext())
    stop_at = (args.warmup_batches + args.profile_steps) if prof_on else None
    if prof_on and rank == 0:
        print(f"[profile] roctx markers ON, stop after step {stop_at}", flush=True)

    last_ips = float("nan")
    done = False
    with op_marker:
      for epoch in range(args.epochs):
        if rank == 0:
            # wall-clock epoch-boundary marker for epoch-based power measurement
            print(f"[PWRMARK] epoch {epoch} {time.time():.3f}", flush=True)
        if done:
            break
        if args.lr_schedule == "cosine":
            lr = cosine_lr(args.lr, epoch, args.epochs, args.warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            if rank == 0:
                print(f"epoch {epoch} lr {lr:.5f}", flush=True)
        sampler.set_epoch(epoch)
        model.train()
        t_start = None
        n_imgs_global = 0
        for i, (imgs, tgts) in enumerate(loader):
            rng_push(f"TrainStep{i}")
            rng_push("H2D")
            imgs = imgs.to(device, memory_format=mem_fmt, non_blocking=True)
            tgts = tgts.to(device, non_blocking=True)
            rng_pop()

            optimizer.zero_grad(set_to_none=True)
            rng_push("forward")
            with autocast("cuda", dtype=amp_dtype, enabled=args.amp):
                out = model(imgs)
                loss = criterion(out, tgts)
            rng_pop()
            rng_push("backward")
            scaler.scale(loss).backward()
            rng_pop()
            rng_push("optim")
            scaler.step(optimizer)
            scaler.update()
            rng_pop()
            rng_pop()  # TrainStep

            # Start the throughput timer after warmup to discount first-batch
            # MIOpen/cuDNN autotune cost.
            if i == args.warmup_batches:
                torch.cuda.synchronize()
                t_start = time.perf_counter()
                n_imgs_global = 0
                continue
            if t_start is not None:
                n_imgs_global += imgs.size(0) * world

            if rank == 0 and i % args.print_freq == 0:
                print(f"epoch {epoch} iter {i}/{len(loader)} "
                      f"loss {loss.item():.3f}", flush=True)

            if stop_at is not None and i >= stop_at:
                done = True
                break

        torch.cuda.synchronize()
        dist.barrier()
        if t_start is not None:
            elapsed = time.perf_counter() - t_start
            ips = n_imgs_global / elapsed if elapsed > 0 else float("nan")
            last_ips = ips
            if rank == 0:
                print(f"epoch {epoch} train.compute_ips : {ips:.2f}",
                      flush=True)

    if rank == 0:
        print(f"train.total_ips : {last_ips:.2f}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
