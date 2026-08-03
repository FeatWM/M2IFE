from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from classifier.labels import LABEL_NAMES
from config_utils import resolve_path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_images(path: str | Path, recursive: bool = False) -> list[Path]:
    source = Path(path)
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Input path not found: {source}")
    iterator = source.rglob("*") if recursive else source.glob("*")
    return sorted(
        item for item in iterator if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


class M2IFEPipeline:
    """Complete-image inference: detector -> ordered crops -> classifier ensemble."""

    def __init__(self, config: dict[str, Any]):
        from classifier import M2IFEEnsemble
        from detector import IFEDetector

        self.config = config
        self.detector = IFEDetector(config)
        self.classifier = M2IFEEnsemble(config)
        metadata_cfg = config.get("patient_metadata", {})
        self.metadata_resolver = None
        if metadata_cfg.get("enabled", False):
            from detector.patient_metadata import PatientMetadataResolver

            self.metadata_resolver = PatientMetadataResolver(
                resolve_path(config, metadata_cfg["excel_path"]),
                negative_text=metadata_cfg.get("negative_text", "阴性(-)"),
            )

    def predict_image(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        save_crops: bool = True,
    ) -> list[dict[str, Any]]:
        path = Path(image_path)
        output_path = Path(output_dir)

        # Stage 1: detect the four or nine patient regions and preserve the
        # historical bottom-to-top, left-to-right ordering.
        crops = self.detector.detect_and_crop(path)
        if not crops:
            raise RuntimeError(f"No patient regions detected in {path.name}")

        metadata = (
            self.metadata_resolver.resolve(path.name, len(crops))
            if self.metadata_resolver
            else [{"sample_id": None, "ground_truth": None} for _ in crops]
        )
        crop_dir = output_path / "crops" / path.stem
        if save_crops:
            crop_dir.mkdir(parents=True, exist_ok=True)

        # Stage 2: send every ordered crop through the 15-model ensemble
        # (three backbones x five folds) and record one row per patient.
        rows = []
        for detection, patient_record in zip(crops, metadata):
            prediction = self.classifier.predict_pil(detection.image)
            crop_path = None
            if save_crops:
                crop_path = crop_dir / f"patient_{detection.patient_index:02d}.png"
                detection.image.save(crop_path)
            rows.append(
                {
                    "source_image": str(path),
                    "source_name": path.name,
                    "patient_count": len(crops),
                    "patient_index": detection.patient_index,
                    "sample_id": patient_record.get("sample_id"),
                    "ground_truth": patient_record.get("ground_truth"),
                    "detector_confidence": detection.confidence,
                    "detector_class": detection.detector_class,
                    "box_x1": detection.box_xyxy[0],
                    "box_y1": detection.box_xyxy[1],
                    "box_x2": detection.box_xyxy[2],
                    "box_y2": detection.box_xyxy[3],
                    "crop_path": str(crop_path) if crop_path else None,
                    "prediction": prediction["multilabel"],
                    "prediction_class": prediction["class_9"],
                    "active_labels": prediction["active_labels"],
                    **{
                        f"p_{name}": probability
                        for name, probability in zip(LABEL_NAMES, prediction["probabilities"])
                    },
                }
            )
        return rows

    def run(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        recursive: bool = False,
        save_crops: bool = True,
    ) -> list[dict[str, Any]]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        images = collect_images(input_path, recursive=recursive)
        if not images:
            raise RuntimeError(f"No supported images found in {input_path}")

        all_rows: list[dict[str, Any]] = []
        for index, image_path in enumerate(images, start=1):
            print(f"[pipeline] {index}/{len(images)} {image_path.name}")
            try:
                image_rows = self.predict_image(image_path, output_path, save_crops=save_crops)
                all_rows.extend(image_rows)
                print(f"[pipeline] {image_path.name}: {len(image_rows)} patients")
            except Exception as exc:
                print(f"[pipeline error] {image_path}: {exc}")
                all_rows.append(
                    {
                        "source_image": str(image_path),
                        "source_name": image_path.name,
                        "error": str(exc),
                    }
                )
        self.save_rows(all_rows, output_path)
        return all_rows

    @staticmethod
    def save_rows(rows: Iterable[dict[str, Any]], output_dir: Path) -> None:
        rows_list = list(rows)
        (output_dir / "predictions.json").write_text(
            json.dumps(rows_list, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if not rows_list:
            return
        fieldnames = sorted({key for row in rows_list for key in row})
        with (output_dir / "predictions.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows_list:
                serializable = {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
                writer.writerow(serializable)
