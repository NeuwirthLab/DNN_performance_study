#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


def import_dali():
    try:
        from nvidia.dali import pipeline_def
        import nvidia.dali.fn as fn
        import nvidia.dali.types as types
    except ImportError as exc:
        raise SystemExit("DALI is not installed in this Python environment") from exc
    return pipeline_def, fn, types


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serialize per-rank DALI ImageNet pipelines that can be consumed by the native DALI loader path."
    )
    parser.add_argument("--filelist-dir", type=Path, required=True, help="Directory containing train_rank*.txt and val_rank*.txt")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write train_rank*.dali and val_rank*.dali")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--decoder", choices=["cpu", "mixed"], default="mixed")
    args = parser.parse_args()

    pipeline_def, fn, types = import_dali()
    resize_device = "gpu" if args.decoder == "mixed" else "cpu"

    @pipeline_def
    def train_pipeline(file_list: str, decoder_device: str, resize_device: str):
        images, labels = fn.readers.file(
            file_list=file_list,
            random_shuffle=True,
            stick_to_shard=True,
            pad_last_batch=False,
            name="Reader",
        )
        images = fn.decoders.image_random_crop(
            images,
            device=decoder_device,
            output_type=types.RGB,
            random_aspect_ratio=[0.75, 1.333333],
            random_area=[0.08, 1.0],
            num_attempts=10,
        )
        images = fn.resize(images, device=resize_device, resize_x=224, resize_y=224)
        if resize_device == "cpu":
            images = images.gpu()
        mirror = fn.random.coin_flip(probability=0.5)
        images = fn.crop_mirror_normalize(
            images,
            dtype=types.FLOAT,
            output_layout="CHW",
            crop=(224, 224),
            mean=[0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0],
            std=[0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0],
            mirror=mirror,
        )
        labels = fn.cast(labels, dtype=types.INT32).gpu()
        return images, labels

    @pipeline_def
    def val_pipeline(file_list: str, decoder_device: str, resize_device: str):
        images, labels = fn.readers.file(
            file_list=file_list,
            random_shuffle=False,
            stick_to_shard=True,
            pad_last_batch=False,
            name="Reader",
        )
        images = fn.decoders.image(images, device=decoder_device, output_type=types.RGB)
        images = fn.resize(images, device=resize_device, resize_shorter=256)
        if resize_device == "cpu":
            images = images.gpu()
        images = fn.crop_mirror_normalize(
            images,
            dtype=types.FLOAT,
            output_layout="CHW",
            crop=(224, 224),
            crop_pos_x=0.5,
            crop_pos_y=0.5,
            mean=[0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0],
            std=[0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0],
            mirror=0,
        )
        labels = fn.cast(labels, dtype=types.INT32).gpu()
        return images, labels

    train_lists = sorted(args.filelist_dir.glob("train_rank*.txt"))
    val_lists = sorted(args.filelist_dir.glob("val_rank*.txt"))
    if not train_lists or not val_lists:
        raise SystemExit("Missing train_rank*.txt or val_rank*.txt file lists")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for file_list in train_lists:
        rank = file_list.stem.replace("train_rank", "")
        pipe = train_pipeline(
            batch_size=args.batch_size,
            num_threads=args.num_threads,
            device_id=0,
            seed=42,
            exec_pipelined=True,
            exec_async=True,
            prefetch_queue_depth=2,
            file_list=str(file_list),
            decoder_device=args.decoder,
            resize_device=resize_device,
        )
        pipe.build()
        (args.output_dir / f"train_rank{rank}.dali").write_bytes(pipe.serialize())

    for file_list in val_lists:
        rank = file_list.stem.replace("val_rank", "")
        pipe = val_pipeline(
            batch_size=args.batch_size,
            num_threads=args.num_threads,
            device_id=0,
            seed=42,
            exec_pipelined=True,
            exec_async=True,
            prefetch_queue_depth=2,
            file_list=str(file_list),
            decoder_device=args.decoder,
            resize_device=resize_device,
        )
        pipe.build()
        (args.output_dir / f"val_rank{rank}.dali").write_bytes(pipe.serialize())

    print(f"Wrote serialized DALI pipelines to {args.output_dir}")


if __name__ == "__main__":
    main()