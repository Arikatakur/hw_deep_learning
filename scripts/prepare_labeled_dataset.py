"""Prepare a YOLO train/validation dataset from a CVAT YOLO export."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match CVAT YOLO labels to existing images and split for training."
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("with_labels/yolo_with_label/obj_Train_data"),
        help="Folder containing YOLO .txt labels from CVAT.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("dataset/train/images"),
        help="Folder containing the matching source images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset_labeled"),
        help="Output YOLO dataset folder.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of labeled images to use for training.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def find_image(images_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def reset_split_dirs(output_dir: Path) -> None:
    for split in ["train", "val"]:
        for kind in ["images", "labels"]:
            path = output_dir / split / kind
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)


def copy_pair(image_path: Path, label_path: Path, output_dir: Path, split: str) -> None:
    shutil.copy2(image_path, output_dir / split / "images" / image_path.name)
    shutil.copy2(label_path, output_dir / split / "labels" / label_path.name)


def main() -> None:
    args = parse_args()
    if not args.labels_dir.exists():
        raise FileNotFoundError(f"Labels folder not found: {args.labels_dir}")
    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {args.images_dir}")

    pairs: list[tuple[Path, Path]] = []
    missing_images: list[Path] = []

    for label_path in sorted(args.labels_dir.glob("*.txt")):
        image_path = find_image(args.images_dir, label_path.stem)
        if image_path is None:
            missing_images.append(label_path)
            continue
        pairs.append((image_path, label_path))

    if not pairs:
        raise RuntimeError("No matching image/label pairs found.")

    random.seed(args.seed)
    random.shuffle(pairs)
    train_count = int(len(pairs) * args.train_ratio)
    train_pairs = pairs[:train_count]
    val_pairs = pairs[train_count:]

    reset_split_dirs(args.output_dir)
    for image_path, label_path in train_pairs:
        copy_pair(image_path, label_path, args.output_dir, "train")
    for image_path, label_path in val_pairs:
        copy_pair(image_path, label_path, args.output_dir, "val")

    print("Prepared labeled YOLO dataset")
    print(f"matched pairs: {len(pairs)}")
    print(f"missing images: {len(missing_images)}")
    print(f"train pairs: {len(train_pairs)}")
    print(f"val pairs: {len(val_pairs)}")
    print(f"output: {args.output_dir}")


if __name__ == "__main__":
    main()
