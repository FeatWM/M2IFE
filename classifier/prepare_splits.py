from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from .labels import validate_bits


def create_splits(
    manifest: pd.DataFrame,
    output_dir: Path,
    image_column: str = "image",
    label_column: str = "label",
    folds: int = 5,
    test_ratio: float = 0.2,
    seed: int = 2025,
    stratify: bool = True,
) -> None:
    frame = manifest[[image_column, label_column]].dropna().copy()
    frame[image_column] = frame[image_column].astype(str).str.strip()
    frame[label_column] = frame[label_column].astype(str).map(validate_bits)
    frame = frame[frame[image_column].ne("")].sort_values(image_column).reset_index(drop=True)
    indices = np.arange(len(frame))
    labels = frame[label_column].to_numpy()

    stratify_labels = labels if stratify else None
    try:
        remaining_indices, test_indices = train_test_split(
            indices,
            test_size=test_ratio,
            random_state=seed,
            stratify=stratify_labels,
        )
    except ValueError:
        remaining_indices, test_indices = train_test_split(
            indices,
            test_size=test_ratio,
            random_state=seed,
            stratify=None,
        )

    remaining_labels = labels[remaining_indices]
    label_counts = pd.Series(remaining_labels).value_counts()
    use_stratified = stratify and not label_counts.empty and int(label_counts.min()) >= folds
    if use_stratified:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        split_iterator = splitter.split(remaining_indices, remaining_labels)
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        split_iterator = splitter.split(remaining_indices)

    output_dir.mkdir(parents=True, exist_ok=True)
    for fold, (train_position, val_position) in enumerate(split_iterator):
        train_indices = remaining_indices[train_position]
        val_indices = remaining_indices[val_position]
        columns = {
            "train": frame.loc[train_indices, image_column].tolist(),
            "train_label": frame.loc[train_indices, label_column].tolist(),
            "val": frame.loc[val_indices, image_column].tolist(),
            "val_label": frame.loc[val_indices, label_column].tolist(),
            "test": frame.loc[test_indices, image_column].tolist(),
            "test_label": frame.loc[test_indices, label_column].tolist(),
        }
        max_length = max(len(values) for values in columns.values())
        padded = {
            key: values + [""] * (max_length - len(values))
            for key, values in columns.items()
        }
        pd.DataFrame(padded).to_csv(output_dir / f"fold{fold}.csv", index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create fixed-test, five-fold classifier splits.")
    parser.add_argument("--manifest", required=True, help="CSV with image and five-bit label columns.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--no-stratify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_splits(
        pd.read_csv(args.manifest, dtype=str),
        Path(args.output),
        image_column=args.image_column,
        label_column=args.label_column,
        folds=args.folds,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratify=not args.no_stratify,
    )


if __name__ == "__main__":
    main()

