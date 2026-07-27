from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from config_utils import resolve_path
from .box_order import order_box_indices, xywh_to_xyxy


@dataclass
class DetectionCrop:
    patient_index: int
    source_box_index: int
    box_xyxy: tuple[int, int, int, int]
    confidence: float
    detector_class: int
    image: Image.Image


class IFEDetector:
    def __init__(self, config: dict[str, Any]):
        detector_cfg = config["detector"]
        self.config = config
        self.device = config.get("project", {}).get("device", "cpu")
        self.image_size = int(detector_cfg.get("image_size", 640))
        self.confidence = float(detector_cfg.get("confidence", 0.25))
        self.iou = float(detector_cfg.get("iou", 0.7))
        self.expected_counts = tuple(int(value) for value in detector_cfg.get("expected_box_counts", [4, 9]))
        self.strict_count = bool(detector_cfg.get("strict_box_count", True))
        self.order = detector_cfg.get("order", "bottom_to_top_left_to_right")
        self.weight_path = resolve_path(config, detector_cfg["weights"])

        if not self.weight_path.is_file():
            raise FileNotFoundError(f"Detector weight not found: {self.weight_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is required for detector inference") from exc
        self.model = YOLO(str(self.weight_path))

    def detect_and_crop(self, source: str | Path | Image.Image | np.ndarray) -> list[DetectionCrop]:
        pil_image, yolo_source = self._prepare_source(source)
        results = self.model.predict(
            source=yolo_source,
            imgsz=self.image_size,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
            save=False,
        )
        if not results:
            return []

        boxes = results[0].boxes
        count = len(boxes)
        if count == 0:
            return []
        if self.strict_count and self.expected_counts and count not in self.expected_counts:
            raise RuntimeError(
                f"Unexpected detector box count {count}; expected one of {self.expected_counts}"
            )

        xywh = boxes.xywh.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        ordered_indices = order_box_indices(xywh, self.order)

        width, height = pil_image.size
        crops: list[DetectionCrop] = []
        for patient_index, source_index in enumerate(ordered_indices, start=1):
            box_xyxy = xywh_to_xyxy(xywh[source_index], width, height)
            crop = pil_image.crop(box_xyxy).convert("RGB")
            crops.append(
                DetectionCrop(
                    patient_index=patient_index,
                    source_box_index=source_index,
                    box_xyxy=box_xyxy,
                    confidence=float(confidences[source_index]),
                    detector_class=int(classes[source_index]),
                    image=crop,
                )
            )
        return crops

    @staticmethod
    def _prepare_source(
        source: str | Path | Image.Image | np.ndarray,
    ) -> tuple[Image.Image, str | np.ndarray]:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"Input image not found: {path}")
            return Image.open(path).convert("RGB"), str(path)
        if isinstance(source, Image.Image):
            image = source.convert("RGB")
            return image, np.asarray(image)
        if isinstance(source, np.ndarray):
            array = source
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError("NumPy input must have shape H x W x 3")
            return Image.fromarray(array.astype(np.uint8)).convert("RGB"), array
        raise TypeError(f"Unsupported detector input type: {type(source)!r}")

