# Demo Run

This folder is for the live demo. Put full COVID rapid test images in:

```text
demo-run/input/
```

Then run the full two-stage pipeline from the repository root:

```powershell
python demo-run/run_demo.py --source demo-run/input
```

The script automatically:

```text
1. loads the Phase 1 YOLO detector
2. detects the test kit/result area in each full image
3. crops the detected area
4. loads the Phase 2 YOLO classifier
5. predicts yes, no, or invalid
6. writes crops, annotated images, and a CSV summary
```

The detector also tries rotated copies of the image during the demo, so mildly tilted photos have a better chance of being detected.

Outputs are written to:

```text
demo-run/output/
  annotated/       images with the detected box and prediction
  crops/           cropped test/result areas sent to the classifier
  predictions.csv  one-line result per image
```

## Setup

Install dependencies first:

```powershell
python -m pip install -r requirements.txt
```

Check that the model files exist:

```powershell
Test-Path runs/detect/runs/detect/covid_testkit_labeler/weights/best.pt
Test-Path runs/classify/covid_result_classifier/weights/best.pt
```

Both commands should print `True`.

## Main Demo Command

```powershell
python demo-run/run_demo.py --source demo-run/input
```

This command runs both models:

```text
full image -> object detector -> crop -> classifier -> final result
```

## Demo With One Image

```powershell
python demo-run/run_demo.py --source path\to\covid_test_image.jpg
```

## Use A Lower Detection Threshold

If the detector misses an image, rerun with a lower threshold:

```powershell
python demo-run/run_demo.py --source demo-run/input --det-conf 0.05
```

## Disable Rotation Fallback

Rotation fallback is enabled by default. To test only the original orientation:

```powershell
python demo-run/run_demo.py --source demo-run/input --no-rotations
```

## Run Only The Classifier

The classifier expects cropped test/result images, not full camera photos. After running the full demo once, use the generated crops:

```powershell
yolo classify predict model=runs/classify/covid_result_classifier/weights/best.pt source=demo-run/output/crops save=True project=demo-run/output name=classifier_only
```

Run the classifier on one cropped image:

```powershell
yolo classify predict model=runs/classify/covid_result_classifier/weights/best.pt source=demo-run/output/crops/IMAGE_NAME_crop.jpg save=True project=demo-run/output name=classifier_only
```

Check classifier accuracy on the labeled validation set:

```powershell
yolo classify val model=runs/classify/covid_result_classifier/weights/best.pt data=classification_dataset
```

## Expected Terminal Output

```text
image,detection_confidence,crop_path,predicted_class,class_confidence,final_result,annotated_path
dataset\test\images\covid_test_000003.jpg,0.82,demo-run\output\crops\covid_test_000003_crop.jpg,no,0.91,negative test,demo-run\output\annotated\covid_test_000003_annotated.jpg
```

Class meanings:

```text
yes      positive test
no       negative test
invalid  invalid test
```
