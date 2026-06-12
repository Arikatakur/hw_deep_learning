# COVID-19 Rapid Test Kit Object Detection

This project builds a YOLO object detection dataset and model for COVID-19 rapid antigen test kit images.

The current model has one class:

```text
object-detection
```

That means the model detects the test kit / result area. It does not classify positive vs negative because the current labels are not separate `positive` and `negative` classes.

## Phase 1: Data Collection, Labeling, Training, And Test Prediction

Everything in this README is part of the current Phase 1 work. Later project phases will be added after this phase is fully finished.

### 1. Data Collection

- Downloaded images from Roboflow Universe using the official Roboflow SDK/API, not website scraping.
- Used this project:

```text
https://universe.roboflow.com/new-workspace-4ssos/atk-detection-kit-dataset
```

- Used Roboflow workspace/project/version:

```text
workspace: new-workspace-4ssos
project: atk-detection-kit-dataset
version: 1
```

- Downloaded and processed 500 valid JPG images.
- Removed 26 duplicates.
- Split the images into:

```text
dataset/
  train/images/   400 images
  test/images/    100 images
```

### 2. Labeling

- Labeled the 400 training images in CVAT.
- Exported labels in YOLO format to:

```text
with_labels/yolo_with_label/
```

- The CVAT export contained 400 label files.
- Prepared a YOLO-ready labeled dataset:

```text
dataset_labeled/
  train/
    images/   320
    labels/   320
  val/
    images/   80
    labels/   80
```

- Config file:

```text
configs/covid_test_dataset.yaml
```

### 3. Training And Test Prediction

- Trained YOLOv8 for 50 epochs on CPU.
- Final model weights:

```text
runs/detect/runs/detect/covid_testkit_labeler/weights/best.pt
runs/detect/runs/detect/covid_testkit_labeler/weights/last.pt
```

- Final validation metrics were about:

```text
Precision: 0.718
Recall:    0.755
mAP50:     0.724
mAP50-95:  0.511
```

- Ran prediction on the 100 held-out test images.
- Final prediction outputs:

```text
runs/detect/runs/detect/test_predictions_final/
runs/detect/runs/detect/test_predictions_final/labels/
```

- The model created prediction label files for 67 of the 100 test images at the default confidence threshold.

## Setup On Windows

Open PowerShell in this folder.

Optional virtual environment:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install ultralytics
```

If `yolo` is not recognized, add this folder to your user PATH:

```text
C:\Users\salee\AppData\Local\Python\pythoncore-3.14-64\Scripts
```

Temporary PowerShell-only PATH fix:

```powershell
$env:Path += ";C:\Users\salee\AppData\Local\Python\pythoncore-3.14-64\Scripts"
```

Check YOLO:

```powershell
yolo version
```

## Download The Dataset Again

Put your Roboflow key in `.env`:

```env
ROBOFLOW_API_KEY=your_key_here
ROBOFLOW_VERSION=1
```

Then run:

```powershell
python download_dataset.py --provider roboflow --roboflow-workspace new-workspace-4ssos --roboflow-project atk-detection-kit-dataset --roboflow-version 1 --target 500
```

Expected summary:

```text
total downloaded: 500
duplicates removed: 26
train count: 400
test count: 100
```

## Prepare Labels After CVAT Export

CVAT exported the YOLO labels to:

```text
with_labels/yolo_with_label/obj_Train_data/
```

Prepare the labeled YOLO dataset:

```powershell
python scripts/prepare_labeled_dataset.py
```

Expected summary:

```text
matched pairs: 400
missing images: 0
train pairs: 320
val pairs: 80
```

## Train YOLOv8

Train from the pretrained YOLOv8 nano model:

```powershell
yolo detect train model=yolov8n.pt data=configs/covid_test_dataset.yaml epochs=50 imgsz=640 project=runs/detect name=covid_testkit_labeler
```

If training stops or times out, resume from `last.pt`:

```powershell
yolo detect train resume model=runs/detect/runs/detect/covid_testkit_labeler/weights/last.pt
```

Use:

```text
last.pt = resume training
best.pt = prediction/testing
```

## Predict Labels On The 100 Test Images

Run prediction with the final trained model:

```powershell
yolo detect predict model=runs/detect/runs/detect/covid_testkit_labeler/weights/best.pt source=dataset/test/images save=True save_txt=True save_conf=True project=runs/detect name=test_predictions_final
```

Outputs:

```text
runs/detect/runs/detect/test_predictions_final/
runs/detect/runs/detect/test_predictions_final/labels/
```

If too many images have no detection, try a lower confidence threshold:

```powershell
yolo detect predict model=runs/detect/runs/detect/covid_testkit_labeler/weights/best.pt source=dataset/test/images save=True save_txt=True save_conf=True conf=0.10 project=runs/detect name=test_predictions_low_conf
```

Lower confidence creates more labels, but it may also create more wrong boxes.

## Tilted Test Kits

If tilted test kits are missed, train with stronger rotation augmentation:

```powershell
yolo detect train model=runs/detect/runs/detect/covid_testkit_labeler/weights/best.pt data=configs/covid_test_dataset.yaml epochs=75 imgsz=640 degrees=45 translate=0.15 scale=0.6 fliplr=0.5 project=runs/detect name=covid_testkit_rotated_strong
```

Then predict with:

```powershell
yolo detect predict model=runs/detect/covid_testkit_rotated_strong/weights/best.pt source=dataset/test/images save=True save_txt=True save_conf=True project=runs/detect name=test_predictions_rotated
```

The best fix is to label more tilted examples and retrain.

## File Layout

```text
configs/
  covid_test_dataset.yaml
docs/
  first-three-phases.md
  firstThreePhases.pdf
  Project2.pdf
dataset/
  train/images/          # 400 original train images
  test/images/           # 100 held-out test images
dataset_labeled/
  train/images/          # 320 labeled training images
  train/labels/          # 320 YOLO labels
  val/images/            # 80 validation images
  val/labels/            # 80 YOLO labels
runs/detect/
  runs/detect/covid_testkit_labeler/weights/best.pt
  runs/detect/test_predictions_final/
scripts/
  prepare_labeled_dataset.py
with_labels/
  yolo_with_label/       # CVAT YOLO export
```
