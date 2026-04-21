#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


def load_wnids(data_dir: Path) -> list[str]:
    wnids_path = data_dir / "wnids.txt"
    if not wnids_path.is_file():
        raise FileNotFoundError(f"Missing wnids.txt at {wnids_path}")
    return [line.strip() for line in wnids_path.read_text().splitlines() if line.strip()]


def build_train_entries(data_dir: Path, wnids: list[str]) -> list[tuple[Path, int]]:
    entries: list[tuple[Path, int]] = []
    for label, wnid in enumerate(wnids):
        image_dir = data_dir / "train" / wnid / "images"
        if not image_dir.is_dir():
            continue
        for image_path in sorted(image_dir.glob("*.JPEG")):
            entries.append((image_path, label))
    return entries


def build_val_entries(data_dir: Path, wnid_to_label: dict[str, int]) -> list[tuple[Path, int]]:
    ann_path = data_dir / "val" / "val_annotations.txt"
    if not ann_path.is_file():
        return []

    entries: list[tuple[Path, int]] = []
    for line in ann_path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        filename, wnid = fields[0], fields[1]
        if wnid not in wnid_to_label:
            raise KeyError(f"Validation wnid {wnid} is not present in wnids.txt")
        image_path = data_dir / "val" / "images" / filename
        if not image_path.is_file():
            continue
        entries.append((image_path, wnid_to_label[wnid]))
    return entries


def format_entry(path: Path, label: int, absolute_paths: bool) -> str:
    final_path = path.resolve() if absolute_paths else path
    return f"{final_path} {label}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Tiny ImageNet DALI file_list files with the same label mapping as tiny_dataloader.h."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to the extracted tiny-imagenet-200 directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where file_list_train.txt and file_list_val.txt will be written. Defaults to data dir.",
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Write relative image paths instead of absolute paths.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or data_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    wnids = load_wnids(data_dir)
    wnid_to_label = {wnid: index for index, wnid in enumerate(wnids)}

    train_entries = build_train_entries(data_dir, wnids)
    val_entries = build_val_entries(data_dir, wnid_to_label)

    use_absolute_paths = not args.relative_paths
    train_lines = [format_entry(path, label, use_absolute_paths) for path, label in train_entries]
    val_lines = [format_entry(path, label, use_absolute_paths) for path, label in val_entries]

    train_path = output_dir / "file_list_train.txt"
    val_path = output_dir / "file_list_val.txt"
    train_path.write_text("\n".join(train_lines) + "\n")
    val_path.write_text(("\n".join(val_lines) + "\n") if val_lines else "")

    print(f"Wrote {len(train_entries)} training entries to {train_path}")
    print(f"Wrote {len(val_entries)} validation entries to {val_path}")
    print(f"Classes: {len(wnids)}")


if __name__ == "__main__":
    main()