from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
RESULT_TEXT = {
    "yes": "positive test",
    "no": "negative test",
    "invalid": "invalid test",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root() / path


def find_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.lower() in IMAGE_EXTENSIONS else []
    if source.is_dir():
        return sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return []


def clamp(value: float, low: int, high: int) -> int:
    return max(low, min(high, int(round(value))))


def padded_box(xyxy: list[float], width: int, height: int, padding: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = box_width * padding
    pad_y = box_height * padding
    return (
        clamp(x1 - pad_x, 0, width - 1),
        clamp(y1 - pad_y, 0, height - 1),
        clamp(x2 + pad_x, 1, width),
        clamp(y2 + pad_y, 1, height),
    )


def draw_annotation(
    image_path: Path,
    output_path: Path,
    box: tuple[int, int, int, int] | None,
    label: str,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    if box is not None:
        draw.rectangle(box, outline="lime", width=4)

    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = 8
    y = 8
    draw.rectangle((x - 4, y - 4, x + text_width + 4, y + text_height + 4), fill="black")
    draw.text((x, y), label, fill="white", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def unique_output_path(directory: Path, image_path: Path, suffix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / f"{image_path.stem}{suffix}{image_path.suffix.lower()}"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{image_path.stem}{suffix}_{counter:03d}{image_path.suffix.lower()}"
        counter += 1
    return candidate


def classify_crop(classifier: YOLO, crop_path: Path, imgsz: int) -> tuple[str, float]:
    result = classifier.predict(source=str(crop_path), imgsz=imgsz, verbose=False)[0]
    names = result.names
    top_index = int(result.probs.top1)
    top_conf = float(result.probs.top1conf)
    return str(names[top_index]), top_conf


def run_demo(args: argparse.Namespace) -> int:
    detector_path = resolve_path(args.detector)
    classifier_path = resolve_path(args.classifier)
    source = resolve_path(args.source)
    output_dir = resolve_path(args.output)
    crops_dir = output_dir / "crops"
    annotated_dir = output_dir / "annotated"
    csv_path = output_dir / "predictions.csv"

    if not detector_path.exists():
        raise FileNotFoundError(f"Detector weights not found: {detector_path}")
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier weights not found: {classifier_path}")

    images = find_images(source)
    if not images:
        raise FileNotFoundError(f"No demo images found in: {source}")

    detector = YOLO(str(detector_path))
    classifier = YOLO(str(classifier_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image",
        "detection_confidence",
        "crop_path",
        "predicted_class",
        "class_confidence",
        "final_result",
        "annotated_path",
    ]
    rows: list[dict[str, str]] = []
    print(",".join(fieldnames))
    for image_path in images:
        detection = detector.predict(
            source=str(image_path),
            conf=args.det_conf,
            imgsz=args.det_imgsz,
            verbose=False,
        )[0]

        if detection.boxes is None or len(detection.boxes) == 0:
            label = "no detection"
            annotated_path = unique_output_path(annotated_dir, image_path, "_annotated")
            draw_annotation(image_path, annotated_path, None, label)
            row = {
                "image": str(image_path.relative_to(repo_root()) if image_path.is_relative_to(repo_root()) else image_path),
                "detection_confidence": "",
                "crop_path": "",
                "predicted_class": "",
                "class_confidence": "",
                "final_result": "no detection",
                "annotated_path": str(annotated_path.relative_to(repo_root())),
            }
            rows.append(row)
            print(",".join(row[key] for key in row))
            continue

        best_box_index = int(detection.boxes.conf.argmax())
        detection_conf = float(detection.boxes.conf[best_box_index])
        xyxy = detection.boxes.xyxy[best_box_index].tolist()

        with Image.open(image_path).convert("RGB") as image:
            box = padded_box(xyxy, image.width, image.height, args.crop_padding)
            crop = image.crop(box)
            crop_path = unique_output_path(crops_dir, image_path, "_crop")
            crop.save(crop_path)

        predicted_class, class_conf = classify_crop(classifier, crop_path, args.cls_imgsz)
        final_result = RESULT_TEXT.get(predicted_class, predicted_class)
        label = (
            f"{predicted_class} ({class_conf:.2f}) | "
            f"detection {detection_conf:.2f} | {final_result}"
        )
        annotated_path = unique_output_path(annotated_dir, image_path, "_annotated")
        draw_annotation(image_path, annotated_path, box, label)

        row = {
            "image": str(image_path.relative_to(repo_root()) if image_path.is_relative_to(repo_root()) else image_path),
            "detection_confidence": f"{detection_conf:.4f}",
            "crop_path": str(crop_path.relative_to(repo_root())),
            "predicted_class": predicted_class,
            "class_confidence": f"{class_conf:.4f}",
            "final_result": final_result,
            "annotated_path": str(annotated_path.relative_to(repo_root())),
        }
        rows.append(row)
        print(",".join(row[key] for key in row))

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Processed {len(rows)} image(s)")
    print(f"CSV summary: {csv_path.relative_to(repo_root())}")
    print(f"Crops: {crops_dir.relative_to(repo_root())}")
    print(f"Annotated images: {annotated_dir.relative_to(repo_root())}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the COVID rapid test detector and classifier demo.")
    parser.add_argument("--source", default="demo-run/input", help="Image file or folder of full demo images.")
    parser.add_argument("--output", default="demo-run/output", help="Folder for crops, annotations, and CSV output.")
    parser.add_argument(
        "--detector",
        default="runs/detect/runs/detect/covid_testkit_labeler/weights/best.pt",
        help="Phase 1 detector weights.",
    )
    parser.add_argument(
        "--classifier",
        default="runs/classify/covid_result_classifier/weights/best.pt",
        help="Phase 2 classifier weights.",
    )
    parser.add_argument("--det-conf", type=float, default=0.10, help="Detector confidence threshold.")
    parser.add_argument("--det-imgsz", type=int, default=640, help="Detector image size.")
    parser.add_argument("--cls-imgsz", type=int, default=224, help="Classifier image size.")
    parser.add_argument("--crop-padding", type=float, default=0.05, help="Extra padding around detected box.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_demo(parse_args()))
