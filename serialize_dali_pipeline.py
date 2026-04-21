#!/usr/bin/env python3
"""
serialize_dali_pipeline.py — Serialize DALI pipelines for C++ consumption

Generates serialized pipeline files that can be loaded by the DALI C API
in the C++ training code (dali_loader.h).

Usage:
    # Generate train + val pipelines for ImageNet
    python3 serialize_dali_pipeline.py \
        --file-list-train file_list_train.txt \
        --file-list-val file_list_val.txt \
        --batch-size 128 \
        --image-size 224 \
        --num-threads 8 \
        --num-shards 2 \
        --output-dir /path/to/output

    # Each shard gets its own serialized pipeline file:
    #   dali_train_shard0.bin, dali_train_shard1.bin
    #   dali_val_shard0.bin, dali_val_shard1.bin

Prerequisites:
    pip install --user nvidia-dali-cuda120
"""

import argparse
import os
import sys


def build_train_pipeline(file_list, batch_size, num_threads, device_id,
                         shard_id, num_shards, image_size):
    from nvidia.dali import pipeline_def
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types

    # ImageNet normalization: (x/255 - mean) / std
    # DALI operates on uint8, so: (x - mean*255) / (std*255)
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std  = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    @pipeline_def
    def pipe():
        images, labels = fn.readers.file(
            file_list=file_list,
            random_shuffle=True,
            shard_id=shard_id,
            num_shards=num_shards,
            name="Reader",
            pad_last_batch=True,
        )
        images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
        images = fn.random_resized_crop(
            images,
            size=(image_size, image_size),
            random_area=(0.08, 1.0),
            random_aspect_ratio=(3.0/4.0, 4.0/3.0),
            device="gpu",
        )
        mirror = fn.random.coin_flip(probability=0.5)
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


def build_val_pipeline(file_list, batch_size, num_threads, device_id,
                       shard_id, num_shards, image_size):
    from nvidia.dali import pipeline_def
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types

    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std  = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    @pipeline_def
    def pipe():
        images, labels = fn.readers.file(
            file_list=file_list,
            random_shuffle=False,
            shard_id=shard_id,
            num_shards=num_shards,
            name="Reader",
            pad_last_batch=True,
        )
        images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
        images = fn.resize(images, resize_shorter=256, device="gpu")
        images = fn.crop_mirror_normalize(
            images,
            dtype=types.FLOAT,
            output_layout="CHW",
            crop=(image_size, image_size),
            mean=mean,
            std=std,
            mirror=0,
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


def main():
    ap = argparse.ArgumentParser(description="Serialize DALI pipelines for C++ loader")
    ap.add_argument("--file-list-train", required=True)
    ap.add_argument("--file-list-val", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--num-threads", type=int, default=8)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--output-dir", type=str, default=".")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for shard_id in range(args.num_shards):
        print(f"Building train pipeline for shard {shard_id}/{args.num_shards}...")
        train_pipe = build_train_pipeline(
            args.file_list_train, args.batch_size, args.num_threads,
            device_id=0, shard_id=shard_id, num_shards=args.num_shards,
            image_size=args.image_size,
        )
        train_ser = train_pipe.serialize()
        train_path = os.path.join(args.output_dir, f"dali_train_shard{shard_id}.bin")
        with open(train_path, "wb") as f:
            f.write(train_ser)
        print(f"  Saved: {train_path} ({len(train_ser)} bytes)")

        print(f"Building val pipeline for shard {shard_id}/{args.num_shards}...")
        val_pipe = build_val_pipeline(
            args.file_list_val, args.batch_size, args.num_threads,
            device_id=0, shard_id=shard_id, num_shards=args.num_shards,
            image_size=args.image_size,
        )
        val_ser = val_pipe.serialize()
        val_path = os.path.join(args.output_dir, f"dali_val_shard{shard_id}.bin")
        with open(val_path, "wb") as f:
            f.write(val_ser)
        print(f"  Saved: {val_path} ({len(val_ser)} bytes)")

    print(f"\nAll pipelines serialized to {args.output_dir}/")
    print("Use these files with the C++ DALILoader in dali_loader.h")


if __name__ == "__main__":
    main()
