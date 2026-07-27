from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from config_utils import load_config, resolve_path
from .labels import CLASS_ORDER_9, bits_to_class, validate_bits


def sample_stem(name: str) -> str:
    text = os.path.basename(str(name).strip())
    for extension in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        if text.lower().endswith(extension):
            return text[: -len(extension)]
    return text


def base_key(name: str) -> str:
    stem = sample_stem(name)
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and len(parts[1]) == 5 and not set(parts[1]) - {"0", "1"}:
        return parts[0]
    return stem


def relaxed_key(name: str) -> str | None:
    stem = base_key(name)
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        return None
    head, lane = parts
    date_token = head.split("-", 1)[0]
    date_parts = date_token.split(".")
    if all(part.isdigit() for part in date_parts):
        date_token = ".".join(str(int(part)) for part in date_parts)
    return f"{date_token}|{lane}"


def load_expert_workbook(path: Path) -> dict[str, str]:
    frame = pd.read_excel(path, sheet_name=0, header=None, dtype=str)
    predictions: dict[str, str] = {}
    for column in range(max(0, frame.shape[1] - 1)):
        for row_index in range(frame.shape[0]):
            image_value = frame.iat[row_index, column]
            prediction_value = frame.iat[row_index, column + 1]
            if pd.isna(image_value):
                continue
            stem = sample_stem(str(image_value))
            suffix = stem.rsplit("_", 1)[-1] if "_" in stem else ""
            if len(suffix) != 5 or set(suffix) - {"0", "1"}:
                continue
            prediction = "" if pd.isna(prediction_value) else str(prediction_value).strip()
            if len(prediction) != 5 or set(prediction) - {"0", "1"}:
                prediction = suffix
            predictions.setdefault(base_key(stem), prediction)
    return predictions


def prediction_columns(frame: pd.DataFrame) -> tuple[str, str, str]:
    image_candidates = ("image", "source_name", "source_image")
    ground_truth_candidates = ("ground_truth", "gt")
    prediction_candidates = ("prediction", "pred")
    image_column = next((name for name in image_candidates if name in frame), None)
    gt_column = next((name for name in ground_truth_candidates if name in frame), None)
    pred_column = next((name for name in prediction_candidates if name in frame), None)
    if not all((image_column, gt_column, pred_column)):
        raise ValueError(
            "Prediction CSV must contain image/source_name, ground_truth/gt and prediction/pred columns"
        )
    return image_column, gt_column, pred_column


def compute_expert_metrics(true_classes: list[str], predicted_classes: list[str]) -> dict:
    return {
        "aligned_samples": len(true_classes),
        "accuracy": float(accuracy_score(true_classes, predicted_classes)),
        "macro_precision": float(
            precision_score(
                true_classes,
                predicted_classes,
                labels=CLASS_ORDER_9,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                true_classes,
                predicted_classes,
                labels=CLASS_ORDER_9,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                true_classes,
                predicted_classes,
                labels=CLASS_ORDER_9,
                average="macro",
                zero_division=0,
            )
        ),
    }


def compare(prediction_csv: Path, expert_dir: Path, output_dir: Path) -> None:
    frame = pd.read_csv(prediction_csv, dtype=str)
    image_column, gt_column, model_pred_column = prediction_columns(frame)
    frame = frame.dropna(subset=[image_column, gt_column, model_pred_column]).copy()
    frame["base_key"] = frame[image_column].map(base_key)
    frame["relaxed_key"] = frame[image_column].map(relaxed_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    matched_rows = []
    for workbook in sorted(expert_dir.glob("*.xlsx")):
        expert_name = workbook.stem
        predictions = load_expert_workbook(workbook)
        relaxed_predictions: dict[str, str] = {}
        ambiguous = set()
        for key, value in predictions.items():
            key_relaxed = relaxed_key(key)
            if key_relaxed is None:
                continue
            if key_relaxed in relaxed_predictions and relaxed_predictions[key_relaxed] != value:
                relaxed_predictions.pop(key_relaxed, None)
                ambiguous.add(key_relaxed)
            elif key_relaxed not in ambiguous:
                relaxed_predictions[key_relaxed] = value

        true_classes = []
        expert_classes = []
        for _, row in frame.iterrows():
            expert_bits = predictions.get(row["base_key"])
            match_type = "exact"
            if expert_bits is None and row["relaxed_key"] not in ambiguous:
                expert_bits = relaxed_predictions.get(row["relaxed_key"])
                match_type = "relaxed"
            if expert_bits is None:
                continue
            true_class = bits_to_class(validate_bits(row[gt_column]))
            expert_class = bits_to_class(validate_bits(expert_bits))
            if true_class is None or expert_class is None:
                continue
            true_classes.append(true_class)
            expert_classes.append(expert_class)
            matched_rows.append(
                {
                    "expert": expert_name,
                    "image": row[image_column],
                    "ground_truth": row[gt_column],
                    "model_prediction": row[model_pred_column],
                    "expert_prediction": expert_bits,
                    "match_type": match_type,
                }
            )
        if true_classes:
            summaries.append({"expert": expert_name, **compute_expert_metrics(true_classes, expert_classes)})
        else:
            summaries.append({"expert": expert_name, "aligned_samples": 0})

    pd.DataFrame(summaries).to_csv(output_dir / "expert_metrics.csv", index=False)
    pd.DataFrame(matched_rows).to_csv(output_dir / "expert_aligned_predictions.csv", index=False)
    (output_dir / "expert_metrics.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare M2-IFE predictions with expert workbooks.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--expert-dir", default=None)
    parser.add_argument("--output", default="outputs/expert_comparison")
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    expert_value = args.expert_dir or config["expert_comparison"]["expert_dir"]
    compare(
        Path(args.predictions),
        resolve_path(config, expert_value),
        resolve_path(config, args.output),
    )


if __name__ == "__main__":
    main()

