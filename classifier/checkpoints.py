from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch


def load_checkpoint_file(
    path: str | Path,
    map_location: str | torch.device = "cpu",
    trust_legacy: bool = True,
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    except pickle.UnpicklingError:
        if not trust_legacy:
            raise RuntimeError(
                f"{checkpoint_path} contains legacy pickled metadata. "
                "Convert it with python -m classifier.convert_checkpoint before loading."
            )
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint object in {checkpoint_path}")
    return checkpoint


def extract_model_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint state_dict must be a mapping")

    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = str(key)
        if new_key.startswith("model."):
            new_key = new_key[6:]
        cleaned[new_key] = value
    return cleaned


def save_compatible_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    **metadata: Any,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state = {f"model.{key}": value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({"state_dict": state, **metadata}, checkpoint_path)
