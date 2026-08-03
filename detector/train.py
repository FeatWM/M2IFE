from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from config_utils import load_config, resolve_path, with_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the M2-IFE patient-region detector.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default=None, help="Optional YOLO pretrained model override.")
    parser.add_argument("--data", default=None, help="Optional detector data YAML override.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = with_device(load_config(args.config), args.device)
    detector_cfg = config["detector"]
    train_cfg = detector_cfg.get("train", {})
    model_value = args.model or detector_cfg.get("pretrained_model", "yolo11x.pt")
    model_path = Path(model_value)
    if model_path.suffix and not model_path.is_absolute() and (resolve_path(config, model_path)).exists():
        model_value = str(resolve_path(config, model_path))
    data_path = resolve_path(config, args.data or detector_cfg["data"])

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required for detector training") from exc

    model = YOLO(str(model_value))
    kwargs = {
        "data": str(data_path),
        "epochs": args.epochs or int(train_cfg.get("epochs", 100)),
        "batch": int(train_cfg.get("batch", 16)),
        "workers": int(train_cfg.get("workers", 8)),
        "imgsz": int(detector_cfg.get("image_size", 640)),
        "device": config.get("project", {}).get("device", "cpu"),
        "project": str(resolve_path(config, train_cfg.get("project", "runs/detector"))),
        "name": args.name or train_cfg.get("name", "train"),
        "exist_ok": True,
    }
    if train_cfg.get("disable_augmentation", True):
        kwargs.update(
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.0,
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
        )
    model.train(**kwargs)

    best_source = Path(str(getattr(model.trainer, "best", "")))
    best_target = resolve_path(config, detector_cfg["weights"])
    if best_source.is_file():
        best_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_source, best_target)
        print(f"[detector] best checkpoint copied to {best_target}")
    else:
        print(f"[detector] training finished; best checkpoint reported as {best_source}")


if __name__ == "__main__":
    main()
