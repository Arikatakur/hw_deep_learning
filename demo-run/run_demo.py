from __future__ import annotations

import argparse
import csv
import math
import tempfile
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


def rotate_box_to_original(
    xyxy: list[float],
    angle: float,
    original_width: int,
    original_height: int,
) -> list[float]:
    x1, y1, x2, y2 = xyxy
    box_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    if angle == 0:
        mapped = box_points
    else:
        theta = math.radians(angle)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        center_x = original_width / 2
        center_y = original_height / 2
        original_corners = [
            (0, 0),
            (original_width, 0),
            (original_width, original_height),
            (0, original_height),
        ]
        rotated_corners = []
        for x, y in original_corners:
            dx = x - center_x
            dy = y - center_y
            rotated_corners.append(
                (
                    center_x + cos_theta * dx - sin_theta * dy,
                    center_y + sin_theta * dx + cos_theta * dy,
                )
            )

        min_x = min(point[0] for point in rotated_corners)
        min_y = min(point[1] for point in rotated_corners)
        mapped = []
        for x, y in box_points:
            rotated_x = x + min_x
            rotated_y = y + min_y
            dx = rotated_x - center_x
            dy = rotated_y - center_y
            mapped.append(
                (
                    center_x + cos_theta * dx + sin_theta * dy,
                    center_y - sin_theta * dx + cos_theta * dy,
                )
            )

    xs = [point[0] for point in mapped]
    ys = [point[1] for point in mapped]
    return [
        clamp(min(xs), 0, original_width - 1),
        clamp(min(ys), 0, original_height - 1),
        clamp(max(xs), 1, original_width),
        clamp(max(ys), 1, original_height),
    ]


def detect_best_box(
    detector: YOLO,
    image_path: Path,
    conf: float,
    imgsz: int,
    try_rotations: bool,
) -> tuple[list[float], float, float] | None:
    rotations = [0]
    if try_rotations:
        rotations.extend([-30, -20, -15, -10, 10, 15, 20, 30, 90, 180, 270])

    best: tuple[list[float], float, float] | None = None

    with Image.open(image_path).convert("RGB") as original:
        original_width, original_height = original.size

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for angle in rotations:
                if angle == 0:
                    candidate_path = image_path
                else:
                    rotated = original.rotate(angle, expand=True)
                    candidate_path = temp_path / f"{image_path.stem}_rot{angle}{image_path.suffix}"
                    rotated.save(candidate_path)

                result = detector.predict(
                    source=str(candidate_path),
                    conf=conf,
                    imgsz=imgsz,
                    verbose=False,
                )[0]

                if result.boxes is None or len(result.boxes) == 0:
                    continue

                best_box_index = int(result.boxes.conf.argmax())
                detection_conf = float(result.boxes.conf[best_box_index])
                xyxy = result.boxes.xyxy[best_box_index].tolist()
                original_xyxy = rotate_box_to_original(
                    xyxy,
                    angle,
                    original_width,
                    original_height,
                )

                if best is None or detection_conf > best[1]:
                    best = (original_xyxy, detection_conf, angle)

                if angle == 0:
                    return best

    return best


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
        detection = detect_best_box(
            detector,
            image_path,
            args.det_conf,
            args.det_imgsz,
            args.try_rotations,
        )

        if detection is None:
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

        xyxy, detection_conf, rotation_angle = detection

        with Image.open(image_path).convert("RGB") as image:
            box = padded_box(xyxy, image.width, image.height, args.crop_padding)
            crop = image.crop(box)
            crop_path = unique_output_path(crops_dir, image_path, "_crop")
            crop.save(crop_path)

        predicted_class, class_conf = classify_crop(classifier, crop_path, args.cls_imgsz)
        final_result = RESULT_TEXT.get(predicted_class, predicted_class)
        label = (
            f"{predicted_class} ({class_conf:.2f}) | "
            f"detection {detection_conf:.2f} | rot {rotation_angle} | {final_result}"
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
    parser.add_argument(
        "--no-rotations",
        action="store_false",
        dest="try_rotations",
        help="Disable fallback detection on small angle and right-angle rotations.",
    )
    parser.set_defaults(try_rotations=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_demo(parse_args()))
