# Phase 2 Instructions: COVID Test Result Classification

Phase 2 builds on the object detection model from Phase 1. The detector finds the COVID rapid test device in an image. Phase 2 then crops that detected device and trains a new classification model to decide the test result.

## Goal

Build a diagnostic classification model that predicts one of three results from a cropped COVID rapid antigen test image:

```text
Yes      = positive test, two visible lines: C and T
No       = negative test, one visible line: C only
Invalid  = invalid test, no visible valid control/result lines
```

## Pipeline

The intended full workflow is:

```text
Original image
  -> Phase 1 YOLO detector finds the test kit / result area
  -> Crop image using YOLO bounding box
  -> Phase 2 classifier predicts Yes / No / Invalid
```

## Inputs We Already Have

From Phase 1:

```text
runs/detect/runs/detect/covid_testkit_labeler/weights/best.pt
```

This is the trained YOLO object detection model.

We also have:

```text
dataset/
  train/images/
  test/images/

dataset_labeled/
  train/images/
  train/labels/
  val/images/
  val/labels/

runs/detect/runs/detect/test_predictions_final/
```

## What We Need To Create

For Phase 2, create a classification dataset like this:

```text
classification_dataset/
  train/
    yes/
    no/
    invalid/
  val/
    yes/
    no/
    invalid/
  test/
    yes/
    no/
    invalid/
```

Each image inside those folders should be a cropped rapid test device/result area, not the full original photo.

## Step 1: Crop The Test Devices

Use the Phase 1 YOLO model to detect the device/result area in each image.

For every image:

1. Run YOLO detection.
2. Take the best bounding box.
3. Crop the image around that box.
4. Save the cropped image.

Suggested output folder:

```text
cropped_tests/
  images/
```

If YOLO finds no box, either:

- skip the image, or
- manually crop it later.

For project quality, keep a small report count:

```text
input images:
cropped successfully:
no detection:
manual review needed:
```

## Step 2: Label Crops For Classification

Manually sort each cropped image into one of three classes:

```text
yes      positive: C line and T line
no       negative: C line only
invalid  no valid control/result line
```

Important rule:

- If there is no C line, the test should usually be considered invalid.
- A positive test needs both C and T.
- A negative test needs C only.

## Step 3: Add More Invalid Images

The current dataset may not contain enough invalid examples.

To improve the model, collect or create more invalid cases:

- blank test cassette
- unclear/no control line
- failed test with no visible lines
- images where line area is empty

Keep the same legal/ethical collection rule from Phase 1:

- use APIs or public datasets when possible
- do not scrape websites in a way that violates terms
- keep only valid image files

## Step 4: Train A Classification Model

Recommended simple model:

```text
YOLOv8 classification
```

Reason: we already use Ultralytics YOLO, so the workflow stays simple.

Example command:

```powershell
yolo classify train model=yolov8n-cls.pt data=classification_dataset epochs=50 imgsz=224 project=runs/classify name=covid_result_classifier
```

Expected output:

```text
runs/classify/covid_result_classifier/weights/best.pt
```

## Step 5: Test The Classifier

Run prediction on held-out cropped test images:

```powershell
yolo classify predict model=runs/classify/covid_result_classifier/weights/best.pt source=classification_dataset/test save=True project=runs/classify name=result_predictions
```

Check:

- accuracy
- confusion matrix
- examples of wrong predictions
- whether invalid is underrepresented

## Step 6: Combine Detector And Classifier

After the classifier works, create one script that performs the complete diagnostic flow:

```text
input full image
  -> detect device/result area with YOLO detection best.pt
  -> crop bounding box
  -> classify crop as Yes / No / Invalid
  -> print final result
```

Suggested script name:

```text
diagnose_covid_test.py
```

Suggested output:

```text
image: covid_test_000003.jpg
detection confidence: 0.82
classification: No
classification confidence: 0.91
final result: negative test
```

## Files To Add Later

Likely Phase 2 files:

```text
scripts/crop_detected_tests.py
scripts/prepare_classification_dataset.py
diagnose_covid_test.py
configs/classification_dataset.yaml   optional
classification_dataset/               generated, ignored by git
cropped_tests/                         generated, ignored by git
```

## README Updates Later

When Phase 2 is implemented, update `README.md` with:

- Phase 2 goal
- crop-generation instructions
- classification dataset structure
- training command
- prediction command
- final diagnostic pipeline command
- results/accuracy

## Main Risk

The biggest likely issue is class imbalance.

There may be many positive/negative images and too few invalid images. If invalid has too few examples, the classifier will not learn it well. Plan to collect or manually create more invalid samples before final training.
