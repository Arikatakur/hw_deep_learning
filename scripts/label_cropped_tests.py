"""Crop detected COVID test devices and label the crops interactively."""

from __future__ import annotations

import argparse
import csv
import shutil
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageTk
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LABEL_KEYS = {
    "p": ("yes", "positive"),
    "n": ("no", "negative"),
    "i": ("invalid", "invalid"),
}


@dataclass
class Action:
    source: Path
    crop: Path | None
    labeled: Path | None
    label: str
    manifest_row: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Phase 1 detector, crop each detected test device, and show "
            "the crop for manual yes/no/invalid labeling."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("dataset/train/images"),
        help="Folder containing full-size source images.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Classification dataset split to label into.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/detect/runs/detect/covid_testkit_labeler/weights/best.pt"),
        help="Phase 1 YOLO detection model.",
    )
    parser.add_argument(
        "--crop-dir",
        type=Path,
        default=Path("cropped_tests/images"),
        help="Folder where cropped test images are saved.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("classification_dataset"),
        help="Output classification dataset folder.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("classification_dataset/labels_manifest.csv"),
        help="CSV file used to resume and undo labeling decisions.",
    )
    parser.add_argument("--conf", type=float, default=0.10, help="Detection confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO detection image size.")
    parser.add_argument(
        "--padding",
        type=float,
        default=0.06,
        help="Padding added around the detected box, as a fraction of box size.",
    )
    parser.add_argument(
        "--use-full-image",
        action="store_true",
        help="Skip YOLO detection and label the full source image. Useful for already-cropped images.",
    )
    parser.add_argument(
        "--relabel",
        action="store_true",
        help="Show images even if they already appear in the manifest.",
    )
    return parser.parse_args()


def iter_images(source: Path) -> list[Path]:
    return sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def ensure_output_dirs(output_dir: Path, split: str) -> None:
    for class_name in ["yes", "no", "invalid"]:
        (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)


def read_done_sources(manifest: Path, relabel: bool) -> set[str]:
    if relabel or not manifest.exists():
        return set()
    with manifest.open("r", newline="", encoding="utf-8") as csv_file:
        return {row["source"] for row in csv.DictReader(csv_file)}


def append_manifest(manifest: Path, row: dict[str, str]) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    exists = manifest.exists()
    with manifest.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "source",
                "crop",
                "split",
                "label",
                "detection_confidence",
                "labeled_output",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def remove_manifest_row(manifest: Path, row_to_remove: dict[str, str]) -> None:
    if not manifest.exists():
        return
    with manifest.open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    for index in range(len(rows) - 1, -1, -1):
        if rows[index] == row_to_remove:
            del rows[index]
            break
    with manifest.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "source",
                "crop",
                "split",
                "label",
                "detection_confidence",
                "labeled_output",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}_{index:04d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create a unique file name for {path}")


def detect_best_box(model: YOLO, image_path: Path, conf: float, imgsz: int) -> tuple[list[float], float] | None:
    results = model.predict(source=str(image_path), conf=conf, imgsz=imgsz, verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None

    boxes = results[0].boxes
    confidences = boxes.conf.cpu().tolist()
    best_index = max(range(len(confidences)), key=confidences.__getitem__)
    return boxes.xyxy[best_index].cpu().tolist(), float(confidences[best_index])


def crop_image(image_path: Path, crop_path: Path, box: list[float], padding: float) -> Path:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        x1, y1, x2, y2 = box
        pad_x = (x2 - x1) * padding
        pad_y = (y2 - y1) * padding
        left = max(0, int(x1 - pad_x))
        top = max(0, int(y1 - pad_y))
        right = min(width, int(x2 + pad_x))
        bottom = min(height, int(y2 + pad_y))
        crop = image.crop((left, top, right, bottom))
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(crop_path, quality=95)
    return crop_path


def save_full_image(image_path: Path, output_path: Path) -> Path:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, quality=95)
    return output_path


class LabelerApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model = YOLO(str(args.model))
        ensure_output_dirs(args.output_dir, args.split)

        done_sources = read_done_sources(args.manifest, args.relabel)
        self.images = [path for path in iter_images(args.source) if str(path) not in done_sources]
        self.index = 0
        self.actions: list[Action] = []
        self.current_source: Path | None = None
        self.current_crop: Path | None = None
        self.current_confidence = ""
        self.current_photo: ImageTk.PhotoImage | None = None

        self.root = tk.Tk()
        self.root.title("COVID Test Crop Labeler")
        self.root.geometry("980x760")
        self.root.bind("<Key>", self.on_key)

        self.status = tk.StringVar()
        self.help_text = tk.StringVar(
            value="p = positive   n = negative   i = invalid   s = skip   u = undo   q = quit"
        )
        self.image_label = tk.Label(self.root, bg="#111111")
        self.status_label = tk.Label(self.root, textvariable=self.status, anchor="w", padx=12, pady=8)
        self.help_label = tk.Label(self.root, textvariable=self.help_text, anchor="w", padx=12, pady=8)

        self.status_label.pack(fill="x")
        self.image_label.pack(fill="both", expand=True)
        self.help_label.pack(fill="x")

    def run(self) -> None:
        if not self.args.source.exists():
            raise FileNotFoundError(f"Source folder not found: {self.args.source}")
        if not self.args.model.exists():
            raise FileNotFoundError(f"Detection model not found: {self.args.model}")
        self.show_next_image()
        self.root.mainloop()

    def show_next_image(self) -> None:
        self.current_source = None
        self.current_crop = None
        self.current_confidence = ""

        while self.index < len(self.images):
            source = self.images[self.index]
            self.index += 1
            if self.args.use_full_image:
                crop_name = unique_path(self.args.crop_dir / f"{source.stem}_full.jpg")
                crop_path = save_full_image(source, crop_name)
                self.current_source = source
                self.current_crop = crop_path
                self.current_confidence = "full image"
                self.display_crop(crop_path)
                return

            detection = detect_best_box(self.model, source, self.args.conf, self.args.imgsz)
            if detection is None:
                self.record_action(source, None, None, "no_detection", "")
                continue

            box, confidence = detection
            crop_name = unique_path(self.args.crop_dir / f"{source.stem}_crop.jpg")
            crop_path = crop_image(source, crop_name, box, self.args.padding)

            self.current_source = source
            self.current_crop = crop_path
            self.current_confidence = f"{confidence:.4f}"
            self.display_crop(crop_path)
            return

        self.status.set("Done. No more images to label.")
        self.image_label.configure(image="", text="Done", fg="white", font=("Segoe UI", 36))

    def display_crop(self, crop_path: Path) -> None:
        with Image.open(crop_path) as image:
            image = image.convert("RGB")
            image.thumbnail((940, 650))
            self.current_photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.current_photo, text="")
        self.status.set(
            f"{self.index}/{len(self.images)}  {self.current_source}  "
            f"detection confidence: {self.current_confidence}"
        )

    def on_key(self, event: tk.Event) -> None:
        key = event.char.lower()
        if key in LABEL_KEYS:
            class_name, label = LABEL_KEYS[key]
            self.label_current(class_name, label)
        elif key == "s":
            self.skip_current()
        elif key == "u":
            self.undo_last()
        elif key == "q":
            self.root.destroy()

    def label_current(self, class_name: str, label: str) -> None:
        if self.current_source is None or self.current_crop is None:
            return
        output_path = unique_path(
            self.args.output_dir / self.args.split / class_name / self.current_crop.name
        )
        shutil.copy2(self.current_crop, output_path)
        self.record_action(self.current_source, self.current_crop, output_path, label, self.current_confidence)
        self.show_next_image()

    def skip_current(self) -> None:
        if self.current_source is None:
            return
        self.record_action(self.current_source, self.current_crop, None, "skipped", self.current_confidence)
        self.show_next_image()

    def record_action(
        self,
        source: Path,
        crop: Path | None,
        labeled: Path | None,
        label: str,
        confidence: str,
    ) -> None:
        row = {
            "source": str(source),
            "crop": str(crop or ""),
            "split": self.args.split,
            "label": label,
            "detection_confidence": confidence,
            "labeled_output": str(labeled or ""),
        }
        append_manifest(self.args.manifest, row)
        self.actions.append(Action(source, crop, labeled, label, row))

    def undo_last(self) -> None:
        if not self.actions:
            messagebox.showinfo("Undo", "Nothing to undo yet.")
            return
        action = self.actions.pop()
        if action.labeled and action.labeled.exists():
            action.labeled.unlink()
        if action.crop and action.crop.exists():
            action.crop.unlink()
        remove_manifest_row(self.args.manifest, action.manifest_row)

        if self.current_crop and self.current_crop.exists():
            self.current_crop.unlink()
        try:
            self.index = self.images.index(action.source)
        except ValueError:
            self.index = max(0, self.index - 1)
        self.status.set(f"Undid {action.label}: {action.source}")
        self.show_next_image()


def main() -> None:
    args = parse_args()
    app = LabelerApp(args)
    app.run()


if __name__ == "__main__":
    main()
