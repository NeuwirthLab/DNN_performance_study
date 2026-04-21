from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def build_pipelines(args):
    try:
        from nvidia.dali import pipeline_def
        import nvidia.dali.fn as fn
        import nvidia.dali.types as types
    except ImportError as exc:
        raise SystemExit(
            "DALI is not available in this Python environment. Install NVIDIA DALI or run this inside an NGC container."
        ) from exc

    @pipeline_def
    def train_pipeline(file_list: str, shard_id: int, num_shards: int, decoder_device: str):
        images, labels = fn.readers.file(
            file_list=file_list,
            random_shuffle=True,
            shard_id=shard_id,
            num_shards=num_shards,
            stick_to_shard=True,
            pad_last_batch=False,
            name="Reader",
        )
        images = fn.decoders.image(images, device=decoder_device, output_type=types.RGB)
        mirror = fn.random.coin_flip(probability=0.5)
        images = fn.crop_mirror_normalize(
            images.gpu(),
            dtype=types.FLOAT,
            output_layout="CHW",
            crop=(64, 64),
            mean=[127.5, 127.5, 127.5],
            std=[127.5, 127.5, 127.5],
            mirror=mirror,
        )
        labels = labels.gpu()
        return images, labels

    @pipeline_def
    def val_pipeline(file_list: str, shard_id: int, num_shards: int, decoder_device: str):
        images, labels = fn.readers.file(
            file_list=file_list,
            random_shuffle=False,
            shard_id=shard_id,
            num_shards=num_shards,
            stick_to_shard=True,
            pad_last_batch=False,
            name="Reader",
        )
        images = fn.decoders.image(images, device=decoder_device, output_type=types.RGB)
        images = fn.crop_mirror_normalize(
            images.gpu(),
            dtype=types.FLOAT,
            output_layout="CHW",
            crop=(64, 64),
            mean=[127.5, 127.5, 127.5],
            std=[127.5, 127.5, 127.5],
            mirror=0,
        )
        labels = labels.gpu()
        return images, labels

    pipe_fn = val_pipeline if args.val else train_pipeline
    return pipe_fn(
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        device_id=args.device_id,
        seed=args.seed,
        prefetch_queue_depth=args.prefetch_queue_depth,
        exec_pipelined=True,
        exec_async=True,
        file_list=str(args.file_list),
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        decoder_device=args.decoder,
    )


def tensor_list_shape(tensor_list) -> tuple[int, ...]:
    as_tensor = tensor_list.as_tensor()
    shape = as_tensor.shape()
    if hasattr(shape, "tolist"):
        return tuple(shape.tolist())
    return tuple(shape)


def sample_min_max(images_cpu) -> tuple[float, float]:
    sample = images_cpu.at(0)
    if hasattr(sample, "min") and hasattr(sample, "max"):
        return float(sample.min()), float(sample.max())
    flat = []
    for channel in sample:
        for row in channel:
            flat.extend(row)
    return float(min(flat)), float(max(flat))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a Tiny ImageNet DALI pipeline that matches tiny_dataloader.h semantics."
    )
    parser.add_argument("--file-list", type=Path, required=True, help="Path to file_list_train.txt or file_list_val.txt")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--prefetch-queue-depth", type=int, default=2)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decoder", choices=["cpu", "mixed"], default="mixed")
    parser.add_argument("--val", action="store_true", help="Use validation pipeline semantics")
    args = parser.parse_args()

    if not args.file_list.is_file():
        raise SystemExit(f"Missing file list: {args.file_list}")

    pipe = build_pipelines(args)
    pipe.build()

    last_images = None
    last_labels = None

    for _ in range(args.warmup_iters):
        last_images, last_labels = pipe.run()

    start = time.perf_counter()
    total_images = 0

    for iteration in range(args.iters):
        last_images, last_labels = pipe.run()
        total_images += args.batch_size

        if iteration == 0:
            images_cpu = last_images.as_cpu()
            labels_cpu = last_labels.as_cpu()
            label_sample = [int(labels_cpu.at(i)[0]) for i in range(min(args.batch_size, 8))]
            img_min, img_max = sample_min_max(images_cpu)
            print("First batch sanity check:")
            print(f"  images shape: {tensor_list_shape(last_images)}")
            print(f"  labels shape: {tensor_list_shape(last_labels)}")
            print(f"  labels[0:{len(label_sample)}]: {label_sample}")
            print(f"  sample min/max: {img_min:.3f}, {img_max:.3f}")

    # Force the last batch to finish before stopping the timer.
    assert last_images is not None and last_labels is not None
    _ = last_images.as_cpu()
    _ = last_labels.as_cpu()
    elapsed = time.perf_counter() - start

    images_per_second = total_images / elapsed if elapsed > 0 else 0.0
    print("\nBenchmark summary:")
    print(f"  mode: {'val' if args.val else 'train'}")
    print(f"  decoder: {args.decoder}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  num_threads: {args.num_threads}")
    print(f"  shard: {args.shard_id}/{args.num_shards}")
    print(f"  iterations: {args.iters}")
    print(f"  total_images: {total_images}")
    print(f"  elapsed_sec: {elapsed:.3f}")
    print(f"  images_per_sec: {images_per_second:.1f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)