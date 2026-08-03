from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config_utils import load_config, resolve_path, with_device
from .ensemble import M2IFEEnsemble
from .labels import CLASS_ORDER_9, LABEL_NAMES, bit_array, bits_to_class


def load_test_rows(split_dir: Path, folds: int, deduplicate: bool = True) -> pd.DataFrame:
    rows = []
    for fold in range(folds):
        path = split_dir / f"fold{fold}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Classifier split file not found: {path}")
        frame = pd.read_csv(path, index_col=0, dtype=str)
        subset = frame[["test", "test_label"]].dropna()
        for image, label in subset.itertuples(index=False, name=None):
            if str(image).strip() and str(label).strip():
                rows.append({"image": str(image), "ground_truth": str(label), "fold": fold})
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError(f"No classifier test samples found in {split_dir}")
    if deduplicate:
        result = result.drop_duplicates("image", keep="first")
    return result.reset_index(drop=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    metrics = {
        "samples": int(len(y_true)),
        "exact_match_accuracy": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "hamming_accuracy": float(np.mean(y_true == y_pred)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    per_label = {}
    auc_values = []
    matrices = multilabel_confusion_matrix(y_true, y_pred)
    for index, name in enumerate(LABEL_NAMES):
        tn, fp, fn, tp = matrices[index].ravel()
        auc_value = None
        if len(np.unique(y_true[:, index])) == 2:
            auc_value = float(roc_auc_score(y_true[:, index], y_prob[:, index]))
            auc_values.append(auc_value)
        per_label[name] = {
            "accuracy": float((tp + tn) / max(1, tp + tn + fp + fn)),
            "sensitivity": float(tp / max(1, tp + fn)),
            "specificity": float(tn / max(1, tn + fp)),
            "precision": float(tp / max(1, tp + fp)),
            "f1": float(f1_score(y_true[:, index], y_pred[:, index], zero_division=0)),
            "auc": auc_value,
        }
    metrics["macro_auc"] = float(np.mean(auc_values)) if auc_values else None
    metrics["per_label"] = per_label
    return metrics


def save_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, output_dir: Path) -> None:
    roc_dir = output_dir / "roc_curves"
    roc_dir.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(LABEL_NAMES):
        if len(np.unique(y_true[:, index])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, index], y_prob[:, index])
        auc_value = roc_auc_score(y_true[:, index], y_prob[:, index])
        fig, axis = plt.subplots(figsize=(5, 5))
        axis.plot(fpr, tpr, linewidth=2, label=f"AUC={auc_value:.3f}")
        axis.plot([0, 1], [0, 1], linestyle="--", color="0.6")
        axis.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=f"ROC - {name}")
        axis.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(roc_dir / f"roc_{name}.png", dpi=200)
        plt.close(fig)


def save_9class_confusion(predictions: pd.DataFrame, output_dir: Path) -> dict:
    valid = predictions.dropna(subset=["ground_truth_class", "prediction_class"]).copy()
    if valid.empty:
        return {"valid_samples": 0}
    matrix = confusion_matrix(
        valid["ground_truth_class"],
        valid["prediction_class"],
        labels=CLASS_ORDER_9,
    )
    pd.DataFrame(matrix, index=CLASS_ORDER_9, columns=CLASS_ORDER_9).to_csv(
        output_dir / "confusion_matrix_9class.csv"
    )
    return {
        "valid_samples": int(len(valid)),
        "accuracy": float(accuracy_score(valid["ground_truth_class"], valid["prediction_class"])),
        "macro_precision": float(
            precision_score(
                valid["ground_truth_class"],
                valid["prediction_class"],
                labels=CLASS_ORDER_9,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                valid["ground_truth_class"],
                valid["prediction_class"],
                labels=CLASS_ORDER_9,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                valid["ground_truth_class"],
                valid["prediction_class"],
                labels=CLASS_ORDER_9,
                average="macro",
                zero_division=0,
            )
        ),
    }


def test_classifier(config: dict, output_dir: Path, backbones: tuple[str, ...]) -> None:
    classifier_cfg = config["classifier"]
    train_cfg = classifier_cfg["train"]
    image_root = resolve_path(config, train_cfg["image_root"])
    split_dir = resolve_path(config, train_cfg["split_dir"])
    test_rows = load_test_rows(split_dir, int(classifier_cfg.get("folds", 5)))
    ensemble = M2IFEEnsemble(config, backbones=backbones)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows = []
    for index, row in test_rows.iterrows():
        image_path = image_root / row["image"]
        if not image_path.is_file():
            print(f"[missing] {image_path}")
            continue
        result = ensemble.predict_pil(Image.open(image_path).convert("RGB"))
        prediction_rows.append(
            {
                **row.to_dict(),
                "prediction": result["multilabel"],
                "ground_truth_class": bits_to_class(row["ground_truth"]),
                "prediction_class": result["class_9"],
                **{
                    f"p_{name}": probability
                    for name, probability in zip(LABEL_NAMES, result["probabilities"])
                },
            }
        )
        print(f"[{index + 1}/{len(test_rows)}] {row['image']} -> {result['multilabel']}")

    if not prediction_rows:
        raise RuntimeError("No classifier images were tested; check image_root and split CSV paths")
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    y_true = bit_array(predictions["ground_truth"])
    y_pred = bit_array(predictions["prediction"])
    y_prob = predictions[[f"p_{name}" for name in LABEL_NAMES]].to_numpy(dtype=np.float64)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["backbones"] = list(backbones)
    metrics["nine_class"] = save_9class_confusion(predictions, output_dir)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_roc_curves(y_true, y_prob, output_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test one classifier family or the full three-backbone M2-IFE ensemble."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--backbone",
        default="all",
        choices=("all", "vgg16", "resnet18", "convnext_large"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = with_device(load_config(args.config), args.device)
    backbones = (
        M2IFEEnsemble.SUPPORTED_BACKBONES
        if args.backbone == "all"
        else (args.backbone,)
    )
    test_cfg = config["classifier"].get("test", {})
    output = args.output or test_cfg.get("output_dir", "outputs/classifier_test")
    test_classifier(config, resolve_path(config, output), backbones)


if __name__ == "__main__":
    main()
