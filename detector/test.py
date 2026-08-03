from __future__ import annotations

import argparse

from config_utils import load_config, resolve_path, with_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the trained M2-IFE patient-region detector on a YOLO split."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default=None, help="cpu, cuda:0, cuda:1, ...")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--weights", default=None, help="Optional detector checkpoint override.")
    parser.add_argument("--output", default=None, help="Optional Ultralytics run directory override.")
    return parser.parse_args()


def test_detector(config: dict, split: str, weights: str | None, output: str | None) -> None:
    detector_cfg = config["detector"]
    checkpoint = resolve_path(config, weights or detector_cfg["weights"])
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Detector checkpoint not found: {checkpoint}. Train the detector first or pass --weights."
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required for detector testing") from exc

    test_cfg = detector_cfg.get("test", {})
    project_dir = resolve_path(config, output or test_cfg.get("output_dir", "outputs/detector_test"))
    model = YOLO(str(checkpoint))
    metrics = model.val(
        data=str(resolve_path(config, detector_cfg["data"])),
        split=split,
        imgsz=int(detector_cfg.get("image_size", 640)),
        conf=float(detector_cfg.get("confidence", 0.25)),
        iou=float(detector_cfg.get("iou", 0.70)),
        device=config.get("project", {}).get("device", "cpu"),
        project=str(project_dir),
        name=split,
        exist_ok=True,
        plots=True,
    )
    print(metrics)


def main() -> None:
    args = parse_args()
    config = with_device(load_config(args.config), args.device)
    test_detector(config, args.split, args.weights, args.output)


if __name__ == "__main__":
    main()
