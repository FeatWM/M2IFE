from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from config_utils import load_config, resolve_path, with_device
from .checkpoints import save_compatible_checkpoint
from .dataset import IFEFoldDataset
from .model import MultiLabelClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_macro_auc(targets: np.ndarray, scores: np.ndarray) -> float:
    values = []
    for index in range(targets.shape[1]):
        if len(np.unique(targets[:, index])) < 2:
            continue
        values.append(roc_auc_score(targets[:, index], scores[:, index]))
    return float(np.mean(values)) if values else float("nan")


@torch.no_grad()
def validate(
    model: MultiLabelClassifier,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    losses = []
    targets = []
    scores = []
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        losses.append(float(criterion(output["logits"], labels).item()))
        targets.append(labels.detach().cpu().numpy())
        scores.append(output["Y_prob"].detach().cpu().numpy())
    target_array = np.concatenate(targets, axis=0)
    score_array = np.concatenate(scores, axis=0)
    return float(np.mean(losses)), safe_macro_auc(target_array, score_array)


def train_one(
    config: dict,
    backbone: str,
    fold: int,
) -> Path:
    classifier_cfg = config["classifier"]
    train_cfg = classifier_cfg["train"]
    device = torch.device(config.get("project", {}).get("device", "cpu"))
    image_root = resolve_path(config, train_cfg["image_root"])
    split_dir = resolve_path(config, train_cfg["split_dir"])
    fold_csv = split_dir / f"fold{fold}.csv"
    run_dir = resolve_path(config, train_cfg.get("output_dir", "runs/classifier")) / backbone / f"fold{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = classifier_cfg.get("checkpoints", {}).get(backbone, [])
    if fold >= len(checkpoint_paths):
        raise ValueError(f"No output checkpoint path configured for {backbone} fold {fold}")
    best_path = resolve_path(config, checkpoint_paths[fold])
    best_path.parent.mkdir(parents=True, exist_ok=True)

    image_size = int(classifier_cfg.get("image_size", 224))
    train_dataset = IFEFoldDataset(image_root, fold_csv, "train", image_size)
    val_dataset = IFEFoldDataset(image_root, fold_csv, "val", image_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get("batch_size", 16)),
        shuffle=True,
        num_workers=int(train_cfg.get("workers", 8)),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg.get("batch_size", 16)),
        shuffle=False,
        num_workers=int(train_cfg.get("workers", 8)),
        pin_memory=device.type == "cuda",
    )

    model = MultiLabelClassifier(backbone=backbone, num_labels=5).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )
    epochs = int(train_cfg.get("epochs", 200))
    patience = int(train_cfg.get("patience", 20))
    best_auc = -float("inf")
    patience_counter = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)["logits"]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_loss, val_auc = validate(model, val_loader, criterion, device)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            "val_macro_auc": val_auc,
        }
        history.append(record)
        print(
            f"[{backbone} fold{fold}] epoch={epoch:03d} "
            f"train_loss={record['train_loss']:.5f} val_loss={val_loss:.5f} val_auc={val_auc:.5f}"
        )

        score = val_auc if np.isfinite(val_auc) else -val_loss
        if score > best_auc:
            best_auc = score
            patience_counter = 0
            save_compatible_checkpoint(
                best_path,
                model,
                backbone=backbone,
                fold=fold,
                epoch=epoch,
                val_macro_auc=val_auc,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[early stop] no improvement for {patience} epochs")
                break

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    (run_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "backbone": backbone,
                "fold": fold,
                "best_score": best_auc,
                "checkpoint": str(best_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return best_path


def parse_fold_values(value: str, folds: int) -> Iterable[int]:
    if value.lower() == "all":
        return range(folds)
    return [int(part.strip()) for part in value.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train M2-IFE multi-label classifiers.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--backbone",
        default="all",
        choices=("all", "vgg16", "resnet18", "convnext_large"),
    )
    parser.add_argument("--fold", default="all", help="all, one fold, or comma-separated folds.")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = with_device(load_config(args.config), args.device)
    seed = int(config.get("project", {}).get("seed", 2025))
    set_seed(seed)
    backbones = (
        MultiLabelClassifier.SUPPORTED_BACKBONES
        if args.backbone == "all"
        else (args.backbone,)
    )
    folds = int(config["classifier"].get("folds", 5))
    for backbone in backbones:
        for fold in parse_fold_values(args.fold, folds):
            train_one(config, backbone, fold)


if __name__ == "__main__":
    main()
