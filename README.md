# M²-IFE

**M²-IFE: A Two-Stage Multi-Backbone Ensemble Framework for Multi-Label
Classification of Immunofixation Electrophoresis Images**

M²-IFE first detects and orders patient subimages in a complete
immunofixation-electrophoresis image. It then predicts five labels with a
three-backbone ensemble.

## Method

```text
complete IFE image
  -> YOLO11x patient-region detector
  -> bottom-to-top, left-to-right crop ordering
  -> VGG16 / ResNet18 / ConvNeXt-Large classifiers
  -> five-fold mean inside each backbone
  -> weighted mean across the three backbones
  -> IgG, IgA, IgM, kappa and lambda probabilities
  -> five-bit multilabel prediction and optional nine-class interpretation
```

The default external ensemble weights are all `1.0`, so the current public
configuration uses an equal mean across the three backbones. The default
multilabel threshold is `0.30`. The historical post-processing rule converts a
prediction ending in `00` to `00000`; it can be disabled in `config.yaml`.

## Repository layout

```text
M2IFE/
├── detector/                 # stage 1: training, validation, detection and cropping
├── classifier/               # stage 2: model, training, ensemble, evaluation
├── infer.py                  # complete two-stage command-line entry point
├── pipeline.py               # Python API connecting both stages
├── config.yaml               # portable public configuration
├── scripts/                  # PowerShell run scripts
├── tests/                    # dependency-light unit tests
└── weights/README.md         # expected checkpoint layout
```

The original copied scripts and local `yolo/` directory may remain on the
development machine. `.gitignore` excludes them from the public repository.

## Installation

Use Python 3.10 or 3.11 for the widest compatibility with PyTorch and
Ultralytics:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The inspected YOLO code is based on Ultralytics 8.3.109, which is pinned in the
dependency file.

## Configuration

`config.yaml` uses relative placeholder paths for datasets, patient workbooks,
expert results, model weights and generated outputs. No dataset, patient record,
expert workbook, checkpoint or generated result is included in this repository.

For a private machine-specific setup, copy `config.yaml` to
`config.local.yaml`, update that copy locally, and pass it with `--config`.
`config.local.yaml` is excluded from Git.

Before public use, arrange downloadable weights as documented in
`weights/README.md`.

The existing Lightning checkpoints contain legacy Python metadata. Convert each
trusted checkpoint before public distribution:

```bash
python -m classifier.convert_checkpoint --input old.ckpt --output converted.ckpt
```

After every configured checkpoint has been converted, set
`classifier.trust_legacy_checkpoints` to `false`.

## Detector

Prepare LabelMe rectangle annotations:

```bash
python -m detector.prepare_data --input data/labelme --output data/detector/labels/train
```

Train:

```bash
python -m detector.train --config config.yaml --device cuda:0
```

Evaluate:

```bash
python -m detector.evaluate --config config.yaml --device cuda:0 --split test
```

The original detector inference accepted only four or nine boxes. That check is
preserved through `expected_box_counts` and `strict_box_count`. The inspected
training labels contain classes `0`, `1`, and `2`; these numeric class names are
retained in `detector/data.yaml`, while every detected class is passed to the
same patient-crop classifier.

## Classifier

Create a CSV manifest with `image` and `label` columns. `label` is a five-bit
string ordered as `IgG, IgA, IgM, kappa, lambda`.

Create the fixed test set and five folds:

```bash
python -m classifier.prepare_splits ^
  --manifest data/classifier/manifest.csv ^
  --output data/classifier/splits
```

Train every backbone and fold:

```bash
python -m classifier.train --config config.yaml --device cuda:0 --backbone all --fold all
```

Evaluate the 15-model ensemble:

```bash
python -m classifier.evaluate --config config.yaml --device cuda:0
```

The evaluation writes `predictions.csv`, `metrics.json`, per-label ROC figures,
and a nine-class confusion matrix.

## Complete inference

One image:

```bash
python infer.py --config config.yaml --device cuda:0 --input example.bmp
```

A directory:

```bash
python infer.py --config config.yaml --device cuda:0 --input input_images --recursive
```

Outputs include ordered patient crops, `predictions.csv`, and
`predictions.json`.

Python API:

```python
from config_utils import load_config
from pipeline import M2IFEPipeline

config = load_config("config.yaml")
pipeline = M2IFEPipeline(config)
rows = pipeline.run("example.bmp", "outputs/inference")
```

## Optional patient metadata

Set `patient_metadata.enabled: true` and configure `excel_path` to associate
ordered crops with sample IDs and ground-truth labels. The adapter retains the
original Chinese workbook column conventions. It accepts both the newer
`样本日期`/`样本号` layout and the earlier `图片名称`/`条码编号` layout; blank image
cells in the earlier layout are forward-filled within each patient-image group.

Patient workbooks and images belong under `data/private/`, which is excluded
from Git.

## Expert comparison

First create model predictions with `classifier.evaluate` or the complete
pipeline. Then run:

```bash
python -m classifier.expert_compare ^
  --config config.yaml ^
  --predictions outputs/evaluation/predictions.csv
```

The implementation aligns the historical `P1.xlsx` ... `P10.xlsx` files and
reports their unmodified predictions. It contains no error-count adjustment.

## Grad-CAM

```bash
python -m classifier.heatmap ^
  --config config.yaml ^
  --input data/deidentified_crops ^
  --output outputs/heatmaps
```

The script generates one fused heatmap from representative VGG16, ResNet18 and
ConvNeXt-Large fold models while the prediction probabilities still use the
full 15-model ensemble.

## Tests

The lightweight tests cover box ordering, multilabel post-processing and
patient-label conversion:

```bash
python -m unittest discover -s tests -v
```

Full detector and classifier smoke tests require PyTorch, torchvision,
Ultralytics, and the trained weights.

## Privacy and release checklist

- Keep patient images, sample IDs, workbooks, split CSVs, expert sheets and
  generated outputs outside Git.
- Publish only institution-approved, irreversibly de-identified examples.
- Publish model checkpoints through a release service and include SHA-256
  checksums.
- Validate `config.yaml` on a clean machine.
- Add a repository license after institutional and Ultralytics license review.
- State clearly that the software is for research use and requires independent
  clinical validation.

See `THIRD_PARTY_NOTICES.md` for dependency licensing notes.
