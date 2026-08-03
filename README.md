# M²-IFE

**M²-IFE: A Two-Stage Multi-Backbone Ensemble Framework for Multi-Label
Classification of Immunofixation Electrophoresis Images**

M²-IFE supports three independent workflows:

1. train and test the patient-region detector;
2. train and test the multi-label classifier;
3. run the complete detector-to-classifier pipeline on full IFE images.

## Method

```text
Full IFE image
  -> YOLO patient-region detection
  -> 4 or 9 ordered patient crops
  -> VGG16 / ResNet18 / ConvNeXt-Large classifiers
  -> five-fold mean within each backbone
  -> weighted mean across the three backbones
  -> IgG, IgA, IgM, kappa and lambda probabilities for every patient
```

The detector preserves the historical crop order: rows are processed from
bottom to top and patients within each row are processed from left to right.
The default multi-label threshold is `0.30`.

## Repository layout

```text
M2IFE/
├── detector/
│   ├── train.py              # detector training entry
│   ├── test.py               # detector validation/test entry
│   ├── model.py              # detection and ordered cropping API
│   ├── box_order.py          # 4/9-box ordering logic
│   ├── prepare_data.py       # optional LabelMe-to-YOLO conversion
│   └── data.yaml             # relative detector dataset paths
├── classifier/
│   ├── train.py              # 3 backbones x 5 folds training entry
│   ├── test.py               # single-backbone or fused classifier test entry
│   ├── model.py              # VGG16, ResNet18 and ConvNeXt-Large models
│   ├── dataset.py            # multilabel crop dataset
│   ├── ensemble.py           # fold averaging and backbone fusion
│   ├── prepare_splits.py     # fixed test set and five-fold split creation
│   ├── expert_compare.py     # optional expert comparison
│   └── heatmap.py            # optional fused Grad-CAM
├── pipeline/
│   ├── engine.py             # detector-to-classifier Python API
│   └── infer.py              # complete-image command-line entry
├── scripts/                  # PowerShell wrappers for all primary workflows
├── tests/                    # dependency-light unit tests
├── weights/README.md         # expected checkpoint layout
├── config.yaml               # one shared relative-path configuration
└── requirements.txt
```

The five primary commands are:

```bash
python -m detector.train
python -m detector.test
python -m classifier.train
python -m classifier.test
python -m pipeline.infer --input path/to/full_image_or_directory
```

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

All dataset, checkpoint and output locations in `config.yaml` are relative to
the repository root. This repository contains no clinical data or model
checkpoint.

## 1. Detector training and testing

Expected YOLO dataset layout:

```text
data/detector/
├── images/train/
├── images/val/
├── images/test/
├── labels/train/
├── labels/val/
└── labels/test/
```

Train the detector:

```bash
python -m detector.train --config config.yaml --device cuda:0
```

Ultralytics training logs are written under `runs/detector/`. The best
checkpoint is copied automatically to the detector weight path configured in
`config.yaml`, which defaults to `weights/detector/best.pt`.

Test on the YOLO test split:

```bash
python -m detector.test --config config.yaml --device cuda:0 --split test
```

To test the validation split:

```bash
python -m detector.test --config config.yaml --device cuda:0 --split val
```

Detector test outputs include Ultralytics detection metrics, plots and
confusion matrices under `outputs/detector_test/`.

## 2. Classifier training and testing

Each patient crop has a five-bit label in this order:

```text
IgG, IgA, IgM, kappa, lambda
```

Example labels include `00000`, `10010`, `01001` and `00110`.

Prepare `data/classifier/manifest.csv` with two columns:

```text
image,label
crop_0001.png,00000
crop_0002.png,10010
```

Create the fixed test set and five folds:

```bash
python -m classifier.prepare_splits \
  --manifest data/classifier/manifest.csv \
  --output data/classifier/splits
```

Train all three backbones and all five folds:

```bash
python -m classifier.train --config config.yaml --device cuda:0 --backbone all --fold all
```

Train one backbone or one fold:

```bash
python -m classifier.train --config config.yaml --device cuda:0 --backbone vgg16 --fold 0
```

The best checkpoint from every training run is saved directly to the configured
`weights/classifier/<backbone>/foldN.ckpt` path. These same paths are used by
classifier testing and by the complete pipeline.

Test the full three-backbone, fifteen-model ensemble:

```bash
python -m classifier.test --config config.yaml --device cuda:0 --backbone all
```

Test a single five-fold backbone family:

```bash
python -m classifier.test --config config.yaml --device cuda:0 --backbone resnet18
```

Classifier test outputs include `predictions.csv`, `metrics.json`, per-label
ROC curves and the nine-class confusion matrix under
`outputs/classifier_test/`.

## 3. Complete full-image pipeline

Place the detector checkpoint and all fifteen classifier checkpoints according
to `weights/README.md`, then run:

```bash
python -m pipeline.infer \
  --config config.yaml \
  --device cuda:0 \
  --input path/to/full_ife_image.bmp
```

A directory can be processed in one command:

```bash
python -m pipeline.infer \
  --config config.yaml \
  --device cuda:0 \
  --input path/to/full_image_directory \
  --recursive
```

For every full image, the pipeline:

1. detects patient regions;
2. verifies that the detector returned 4 or 9 boxes;
3. orders and crops the patient regions;
4. classifies every crop with the three-backbone ensemble;
5. writes one result row per patient.

Outputs under `outputs/pipeline/`:

```text
outputs/pipeline/
├── crops/<full-image-name>/patient_01.png ...
├── predictions.csv
└── predictions.json
```

Each result contains the source image, patient order, bounding box, detector
confidence, five probabilities, five-bit multi-label prediction and optional
nine-class interpretation.

Python API:

```python
from config_utils import load_config
from pipeline import M2IFEPipeline

config = load_config("config.yaml")
model = M2IFEPipeline(config)
rows = model.run("path/to/full_image.bmp", "outputs/pipeline")
```

## PowerShell scripts

Equivalent Windows wrappers are available:

```powershell
.\scripts\train_detector.ps1
.\scripts\test_detector.ps1
.\scripts\train_classifiers.ps1
.\scripts\test_classifier.ps1
.\scripts\pipeline_infer.ps1 -InputPath path\to\full_image.bmp
```

## Optional analysis tools

Expert comparison:

```bash
python -m classifier.expert_compare \
  --config config.yaml \
  --predictions outputs/classifier_test/predictions.csv
```

Fused Grad-CAM:

```bash
python -m classifier.heatmap \
  --config config.yaml \
  --input path/to/deidentified_patient_crops \
  --output outputs/heatmaps
```

## Tests

```bash
python -m unittest discover -s tests -v
```

Full smoke tests require PyTorch, torchvision, Ultralytics and trained
checkpoints.

## Privacy and release

- Do not commit patient images, patient identifiers, workbooks, split CSVs,
  expert sheets, model checkpoints or generated results.
- Publish only institution-approved, irreversibly de-identified examples.
- Checkpoint files are ignored by Git and can be distributed separately.
- Review `THIRD_PARTY_NOTICES.md` before selecting the repository license.
