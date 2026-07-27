from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from config_utils import resolve_path
from .checkpoints import extract_model_state, load_checkpoint_file
from .dataset import build_transform
from .labels import active_labels, bits_to_class, probabilities_to_bits
from .model import MultiLabelClassifier


class M2IFEEnsemble:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        classifier_cfg = config["classifier"]
        self.device = torch.device(config.get("project", {}).get("device", "cpu"))
        self.threshold = float(classifier_cfg.get("threshold", 0.3))
        self.apply_rule = bool(classifier_cfg.get("apply_endswith_00_rule", True))
        self.trust_legacy_checkpoints = bool(classifier_cfg.get("trust_legacy_checkpoints", True))
        self.transform = build_transform(int(classifier_cfg.get("image_size", 224)))
        self.family_weights = {
            str(key): float(value)
            for key, value in classifier_cfg.get("ensemble_weights", {}).items()
        }
        self.families: dict[str, list[MultiLabelClassifier]] = {}

        checkpoint_config = classifier_cfg.get("checkpoints", {})
        for backbone in ("vgg16", "resnet18", "convnext_large"):
            paths = checkpoint_config.get(backbone, [])
            if not paths:
                raise ValueError(f"No checkpoints configured for {backbone}")
            models = [self._load_model(backbone, resolve_path(config, path)) for path in paths]
            self.families[backbone] = models

        missing_weights = set(self.families) - set(self.family_weights)
        if missing_weights:
            raise ValueError(f"Missing ensemble weights for: {sorted(missing_weights)}")
        if sum(self.family_weights.values()) <= 0:
            raise ValueError("Ensemble weights must sum to a positive value")

    def _load_model(self, backbone: str, checkpoint_path: Path) -> MultiLabelClassifier:
        model = MultiLabelClassifier(backbone=backbone, num_labels=5)
        checkpoint = load_checkpoint_file(
            checkpoint_path,
            trust_legacy=self.trust_legacy_checkpoints,
        )
        model.load_state_dict(extract_model_state(checkpoint), strict=True)
        model.eval().to(self.device)
        return model

    @torch.no_grad()
    def predict_tensor(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.to(self.device)
        weighted_probabilities = []
        total_weight = 0.0
        for backbone, models in self.families.items():
            fold_probabilities = [model(batch)["Y_prob"] for model in models]
            family_probability = torch.stack(fold_probabilities, dim=0).mean(dim=0)
            weight = self.family_weights[backbone]
            weighted_probabilities.append(weight * family_probability)
            total_weight += weight
        return sum(weighted_probabilities) / total_weight

    def predict_pil(self, image: Image.Image) -> dict[str, Any]:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0)
        probabilities = self.predict_tensor(tensor)[0].detach().cpu().numpy().astype(np.float64)
        bits = probabilities_to_bits(probabilities, self.threshold, self.apply_rule)
        return {
            "probabilities": [float(value) for value in probabilities],
            "multilabel": bits,
            "active_labels": active_labels(bits),
            "class_9": bits_to_class(bits),
        }
