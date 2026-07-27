from __future__ import annotations

import argparse

from config_utils import load_config, resolve_path, with_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained M2-IFE detector.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = with_device(load_config(args.config), args.device)
    detector_cfg = config["detector"]
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required for detector evaluation") from exc

    model = YOLO(str(resolve_path(config, detector_cfg["weights"])))
    metrics = model.val(
        data=str(resolve_path(config, detector_cfg["data"])),
        split=args.split,
        imgsz=int(detector_cfg.get("image_size", 640)),
        conf=float(detector_cfg.get("confidence", 0.25)),
        iou=float(detector_cfg.get("iou", 0.7)),
        device=config.get("project", {}).get("device", "cpu"),
        project=str(resolve_path(config, "runs/detector")),
        name=f"evaluate_{args.split}",
    )
    print(metrics)


if __name__ == "__main__":
    main()

