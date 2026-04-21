#!/usr/bin/env python3
"""
dali_benchmark.py — NVIDIA DALI data loading benchmark for ImageNet & Tiny ImageNet

Benchmarks GPU-accelerated JPEG decode + augmentation pipeline and compares
throughput against the C++ stb_image loader.

Supports both ImageNet-1K (224x224) and Tiny ImageNet (64x64).

Prerequisites on Helma (NHR@FAU):
    module load cuda/12.6.2 cudnn/9.8.0.87-12 nvhpc
    pip install --user nvidia-dali-cuda120

Usage:
    # Generate file lists from your dataset
    python3 dali_benchmark.py --generate-lists --data-dir /path/to/ILSVRC/Data/CLS-LOC

    # Benchmark train pipeline (ImageNet 224x224)
    python3 dali_benchmark.py --file-list file_list_train.txt --batch-size 128 --image-size 224

    # Benchmark Tiny ImageNet (64x64)
    python3 dali_benchmark.py --file-list file_list_train.txt --batch-size 256 --image-size 64 --tiny

    # Multi-GPU shard test
    python3 dali_benchmark.py --file-list file_list_train.txt --shard-id 0 --num-shards 2
"""

import argparse
import os
import sys
import time

# ── File list generation (run before benchmarking) ────────────────────────────

def generate_file_lists(data_dir, output_prefix="file_list"):
    """
    Scans data_dir/{train,val}/<wnid>/*.JPEG and writes:
      file_list_train.txt
      file_list_val.txt
    Each line: /abs/path/to/image.JPEG <int_label>

    The wnid->label mapping is sorted alphabetically (same as PyTorch ImageFolder).
    """
    for split in ["train", "val"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            print(f"WARNING: {split_dir} not found, skipping")
            continue

        # Collect wnid directories and sort for deterministic label assignment
        wnids = sorted([d for d in os.listdir(split_dir)
                        if os.path.isdir(os.path.join(split_dir, d))])
        wnid_to_label = {w: i for i, w in enumerate(wnids)}

        out_path = f"{output_prefix}_{split}.txt"
        count = 0
        with open(out_path, "w") as f:
            for wnid in wnids:
                cls_dir = os.path.join(split_dir, wnid)
                for fname in sorted(os.listdir(cls_dir)):
                    if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                        abs_path = os.path.join(cls_dir, fname)
                        f.write(f"{abs_path} {wnid_to_label[wnid]}\n")
                        count += 1

        print(f"  {out_path}: {count} images, {len(wnids)} classes")

    print("\nFile lists ready. Now run benchmark:")
    print(f"  python3 {sys.argv[0]} --file-list file_list_train.txt --batch-size 128")


# ── DALI Pipelines ────────────────────────────────────────────────────────────

def create_dali_pipeline(file_list, batch_size, num_threads, device_id,
                         shard_id, num_shards, image_size, is_train, is_tiny):
    """
    Build and return a DALI pipeline.

    Import is deferred so --generate-lists works without DALI installed.
    """
    try:
        from nvidia.dali import pipeline_def
        import nvidia.dali.fn as fn
        import nvidia.dali.types as types
    except ImportError:
        print("ERROR: nvidia-dali not installed.")
        print("  pip install --user nvidia-dali-cuda120")
        print("  Or on Helma: module load cuda/12.6.2 && pip install --user nvidia-dali-cuda120")
        sys.exit(1)

    if is_tiny:
        # Tiny ImageNet: normalize to [-1, 1]
        # (x / 127.5) - 1.0
        mean = [127.5, 127.5, 127.5]
        std  = [127.5, 127.5, 127.5]
    else:
        # ImageNet standard: (x/255 - mean) / std
        # DALI normalizes as: (x - mean*255) / (std*255) when input is uint8
        mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        std  = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    @pipeline_def
    def pipe():
        images, labels = fn.readers.file(
            file_list=file_list,
            random_shuffle=is_train,
            shard_id=shard_id,
            num_shards=num_shards,
            name="Reader",
        )

        # Decode JPEGs on GPU (mixed backend: CPU read + GPU decode)
        images = fn.decoders.image(images, device="mixed", output_type=types.RGB)

        if is_train and not is_tiny:
            # ImageNet training: RandomResizedCrop
            images = fn.random_resized_crop(
                images,
                size=(image_size, image_size),
                random_area=(0.08, 1.0),
                random_aspect_ratio=(3.0/4.0, 4.0/3.0),
                device="gpu",
            )
            mirror = fn.random.coin_flip(probability=0.5)
        elif is_train and is_tiny:
            # Tiny ImageNet training: just crop to 64x64 (already that size) + flip
            images = fn.resize(images, size=(image_size, image_size), device="gpu")
            mirror = fn.random.coin_flip(probability=0.5)
        else:
            if is_tiny:
                # Tiny val: just resize to 64x64
                images = fn.resize(images, size=(image_size, image_size), device="gpu")
            else:
                # ImageNet val: resize shortest side to 256, center crop 224
                images = fn.resize(images, resize_shorter=256, device="gpu")
            mirror = 0

        # Normalize + HWC->CHW + float32 (all in one fused op on GPU)
        images = fn.crop_mirror_normalize(
            images,
            dtype=types.FLOAT,
            output_layout="CHW",
            crop=(image_size, image_size),
            mean=mean,
            std=std,
            mirror=mirror,
        )

        labels = labels.gpu()
        return images, labels

    p = pipe(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        seed=42 + shard_id,
    )
    p.build()
    return p


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run_benchmark(args):
    pipe = create_dali_pipeline(
        file_list=args.file_list,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        device_id=args.device_id,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        image_size=args.image_size,
        is_train=not args.val,
        is_tiny=args.tiny,
    )

    mode = "val" if args.val else "train"
    dataset = "Tiny ImageNet" if args.tiny else "ImageNet"
    print(f"\n{'='*70}")
    print(f"DALI Benchmark: {dataset} {mode} pipeline")
    print(f"  Image size   : {args.image_size}x{args.image_size}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  CPU threads  : {args.num_threads}")
    print(f"  GPU device   : {args.device_id}")
    print(f"  Shard        : {args.shard_id}/{args.num_shards}")
    print(f"  Warmup iters : {args.warmup}")
    print(f"  Bench iters  : {args.iters}")
    print(f"{'='*70}\n")

    # Warmup
    print("Warming up...", end="", flush=True)
    for i in range(args.warmup):
        try:
            outputs = pipe.run()
        except StopIteration:
            pipe.reset()
            outputs = pipe.run()
    print(" done.\n")

    # Benchmark
    total_images = 0
    start = time.perf_counter()

    for i in range(args.iters):
        try:
            outputs = pipe.run()
        except StopIteration:
            pipe.reset()
            outputs = pipe.run()

        images = outputs[0]  # TensorListGPU
        labels = outputs[1]  # TensorListGPU

        batch_actual = images.as_tensor().shape()[0]
        total_images += batch_actual

        if i == 0:
            tensor = images.as_tensor()
            print(f"First batch:")
            print(f"  images shape : {tensor.shape()}")
            print(f"  images dtype : {tensor.dtype()}")
            print(f"  labels shape : {labels.as_tensor().shape()}")
            print()

    elapsed = time.perf_counter() - start
    ips = total_images / elapsed

    print(f"{'='*70}")
    print(f"Results:")
    print(f"  Total images : {total_images}")
    print(f"  Elapsed time : {elapsed:.3f}s")
    print(f"  Throughput   : {ips:.1f} images/sec")
    print(f"  Per batch    : {elapsed/args.iters*1000:.2f} ms")
    print(f"{'='*70}")
    print()
    print(f"Compare with your C++ stb_image loader (observed: ~28% GPU util on H100).")
    print(f"If DALI is 3-5x faster, GPU-accelerated loading will eliminate the bottleneck.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="NVIDIA DALI data loading benchmark for ResNet-50 training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument("--generate-lists", action="store_true",
                     help="Generate file_list_train.txt and file_list_val.txt from data dir")
    ap.add_argument("--data-dir", type=str, default=".",
                     help="Root data dir (containing train/ and val/ subdirs)")
    ap.add_argument("--file-list", type=str, default="file_list_train.txt",
                     help="Path to file list (filename label per line)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-threads", type=int, default=8,
                     help="DALI CPU worker threads for file I/O")
    ap.add_argument("--device-id", type=int, default=0, help="GPU device ID")
    ap.add_argument("--shard-id", type=int, default=0,
                     help="Shard ID for multi-GPU (matches DDP rank)")
    ap.add_argument("--num-shards", type=int, default=1,
                     help="Total shards (matches DDP world_size)")
    ap.add_argument("--image-size", type=int, default=224,
                     help="Target image size (224 for ImageNet, 64 for Tiny)")
    ap.add_argument("--tiny", action="store_true",
                     help="Use Tiny ImageNet normalization [-1, 1]")
    ap.add_argument("--val", action="store_true",
                     help="Use validation pipeline (no augmentation)")
    ap.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    ap.add_argument("--iters", type=int, default=200, help="Benchmark iterations")

    args = ap.parse_args()

    if args.generate_lists:
        print("Generating file lists from:", args.data_dir)
        generate_file_lists(args.data_dir)
        return

    if not os.path.isfile(args.file_list):
        print(f"ERROR: File list not found: {args.file_list}")
        print(f"  Generate it first: python3 {sys.argv[0]} --generate-lists --data-dir /path/to/data")
        sys.exit(1)

    run_benchmark(args)


if __name__ == "__main__":
    main()
