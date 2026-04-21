#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


def sorted_classes(split_root: Path) -> list[str]:
    return sorted(entry.name for entry in split_root.iterdir() if entry.is_dir())


def collect_records(split_root: Path, class_names: list[str]) -> list[tuple[Path, int]]:
    records: list[tuple[Path, int]] = []
    for label, class_name in enumerate(class_names):
        class_dir = split_root / class_name
        if not class_dir.is_dir():
            continue
        for image in sorted(class_dir.iterdir()):
            if image.is_file() and image.suffix.lower() in {".jpeg", ".jpg"}:
                records.append((image.resolve(), label))
    records.sort(key=lambda item: str(item[0]))
    return records


def shard_records(records: list[tuple[Path, int]], rank: int, world_size: int) -> list[tuple[Path, int]]:
    return [records[index] for index in range(rank, len(records), world_size)]


def write_file_list(path: Path, records: list[tuple[Path, int]]) -> None:
    lines = [f"{image_path} {label}" for image_path, label in records]
    path.write_text(("\n".join(lines) + "\n") if lines else "")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-rank ImageNet file lists for DALI using the same sorted-path round-robin sharding as train_multigpu.cu."
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to the extracted ImageNet CLS-LOC directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the generated per-rank file lists")
    parser.add_argument("--num-shards", type=int, required=True, help="Number of GPU ranks / shards")
    args = parser.parse_args()

    train_root = args.data_dir / "train"
    val_root = args.data_dir / "val"
    if not train_root.is_dir():
        raise SystemExit(f"Missing train directory: {train_root}")
    if not val_root.is_dir():
        raise SystemExit(f"Missing val directory: {val_root}")

    class_names = sorted_classes(train_root)
    train_records = collect_records(train_root, class_names)
    val_records = collect_records(val_root, class_names)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for rank in range(args.num_shards):
        write_file_list(args.output_dir / f"train_rank{rank}.txt", shard_records(train_records, rank, args.num_shards))
        write_file_list(args.output_dir / f"val_rank{rank}.txt", shard_records(val_records, rank, args.num_shards))

    print(f"Classes: {len(class_names)}")
    print(f"Train records: {len(train_records)}")
    print(f"Val records: {len(val_records)}")
    print(f"Wrote per-rank file lists to {args.output_dir}")


if __name__ == "__main__":
    main()