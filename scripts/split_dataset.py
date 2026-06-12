from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split collected JPG images into train and test folders."
    )
    parser.add_argument(
        "--input",
        default="data/raw",
        help="Folder containing collected JPG/JPEG images.",
    )
    parser.add_argument(
        "--output",
        default="data",
        help="Dataset root where train/test folders will be created.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of images to copy into the training split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits.",
    )
    return parser.parse_args()


def validate_ratio(train_ratio: float) -> None:
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("--train-ratio must be between 0 and 1.")


def collect_images(input_dir: Path) -> list[Path]:
    images = [
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise FileNotFoundError(f"No JPG/JPEG images found in {input_dir}.")
    return images


def copy_split(images: list[Path], output_dir: Path, train_ratio: float, seed: int) -> None:
    rng = random.Random(seed)
    shuffled = images[:]
    rng.shuffle(shuffled)

    split_index = int(len(shuffled) * train_ratio)
    train_images = shuffled[:split_index]
    test_images = shuffled[split_index:]

    train_dir = output_dir / "train" / "images"
    test_dir = output_dir / "test" / "images"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for image in train_images:
        shutil.copy2(image, train_dir / image.name)

    for image in test_images:
        shutil.copy2(image, test_dir / image.name)

    print(f"Total images: {len(images)}")
    print(f"Train images: {len(train_images)} -> {train_dir}")
    print(f"Test images: {len(test_images)} -> {test_dir}")


def main() -> None:
    args = parse_args()
    validate_ratio(args.train_ratio)

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    images = collect_images(input_dir)
    copy_split(images, output_dir, args.train_ratio, args.seed)


if __name__ == "__main__":
    main()
