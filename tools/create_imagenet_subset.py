#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def collect_classes(train_root: Path, max_classes: int) -> list[Path]:
    classes = sorted(entry for entry in train_root.iterdir() if entry.is_dir())
    if max_classes > 0:
        classes = classes[:max_classes]
    return classes


def copy_train_subset(source_root: Path, output_root: Path, max_classes: int, max_images: int) -> tuple[int, int]:
    train_src = source_root / "train"
    train_dst = output_root / "train"
    train_dst.mkdir(parents=True, exist_ok=True)

    classes = collect_classes(train_src, max_classes)
    copied = 0
    for class_dir in classes:
        dst_class = train_dst / class_dir.name
        dst_class.mkdir(parents=True, exist_ok=True)
        for image in sorted(class_dir.iterdir()):
            if not image.is_file() or image.suffix.lower() not in {".jpeg", ".jpg"}:
                continue
            shutil.copy2(image, dst_class / image.name)
            copied += 1
            if max_images > 0 and copied >= max_images:
                return len(classes), copied
    return len(classes), copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a small extracted ImageNet subset for smoke tests."
    )
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="Extracted ImageNet CLS-LOC directory containing train/")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for the reduced subset")
    parser.add_argument("--max-classes", type=int, default=2,
                        help="Maximum number of train classes to copy (0 = all)")
    parser.add_argument("--max-images", type=int, default=100,
                        help="Maximum number of train images to copy (0 = all)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete the output directory first if it already exists")
    args = parser.parse_args()

    if args.max_images < 0 or args.max_classes < 0:
        raise SystemExit("max-images and max-classes must be >= 0")

    if not (args.source_dir / "train").is_dir():
        raise SystemExit(f"Missing train directory: {args.source_dir / 'train'}")

    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    class_count, copied = copy_train_subset(
        args.source_dir,
        args.output_dir,
        args.max_classes,
        args.max_images,
    )

    print(f"source={args.source_dir}")
    print(f"output={args.output_dir}")
    print(f"classes={class_count}")
    print(f"images={copied}")


if __name__ == "__main__":
    main()