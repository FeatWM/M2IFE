from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as functional
from PIL import Image

from config_utils import load_config, resolve_path, with_device
from pipeline import collect_images
from .ensemble import M2IFEEnsemble
from .labels import LABEL_NAMES


def find_last_conv2d(module: nn.Module) -> nn.Module:
    layer = None
    for candidate in module.modules():
        if isinstance(candidate, nn.Conv2d):
            layer = candidate
    if layer is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM")
    return layer


def disable_inplace_relu(module: nn.Module) -> None:
    for candidate in module.modules():
        if isinstance(candidate, nn.ReLU):
            candidate.inplace = False


def normalize_cam(cam: torch.Tensor) -> torch.Tensor:
    cam = cam - cam.min()
    return cam / (cam.max() + 1e-6)


def grad_cam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    label_index: int,
) -> torch.Tensor:
    target_layer = find_last_conv2d(model.model)
    activations = None
    gradients = None

    def forward_hook(_module, _inputs, output):
        nonlocal activations
        activations = output

    def backward_hook(_module, _grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    disable_inplace_relu(model)
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        output = model(input_tensor)
        output["logits"][0, label_index].backward()
        if activations is None or gradients is None:
            raise RuntimeError("Grad-CAM hooks did not receive activations and gradients")
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = functional.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return normalize_cam(cam.detach())
    finally:
        forward_handle.remove()
        backward_handle.remove()


def save_overlay(image: Image.Image, cam: torch.Tensor, output_path: Path, alpha: float = 0.4) -> None:
    resized = cam.cpu().numpy()
    color = matplotlib.colormaps["jet"](resized)[..., :3]
    heatmap = Image.fromarray((color * 255).astype(np.uint8)).resize(image.size)
    Image.blend(image.convert("RGB"), heatmap, alpha=alpha).save(output_path)


def run_heatmaps(config: dict[str, Any], input_path: Path, output_dir: Path, recursive: bool) -> None:
    ensemble = M2IFEEnsemble(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = collect_images(input_path, recursive=recursive)
    device = ensemble.device
    rows = []

    for index, image_path in enumerate(image_paths, start=1):
        print(f"[heatmap] {index}/{len(image_paths)} {image_path.name}")
        image = Image.open(image_path).convert("RGB")
        tensor = ensemble.transform(image).unsqueeze(0).to(device)
        probability_tensor = ensemble.predict_tensor(tensor)[0].detach().cpu()
        label_index = int(torch.argmax(probability_tensor).item())
        cams = []
        for backbone in ("vgg16", "resnet18", "convnext_large"):
            model = ensemble.families[backbone][0]
            cams.append(grad_cam(model, tensor, label_index).cpu())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        fused_cam = normalize_cam(torch.stack(cams, dim=0).mean(dim=0))
        output_path = output_dir / f"{image_path.stem}_{LABEL_NAMES[label_index]}_cam.png"
        save_overlay(image, fused_cam, output_path)
        rows.append(
            {
                "image": str(image_path),
                "cam_label": LABEL_NAMES[label_index],
                "output": str(output_path),
                **{
                    f"p_{name}": float(value)
                    for name, value in zip(LABEL_NAMES, probability_tensor.tolist())
                },
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "heatmap_predictions.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fused M2-IFE Grad-CAM overlays.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/heatmaps")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = with_device(load_config(args.config), args.device)
    run_heatmaps(
        config,
        Path(args.input),
        resolve_path(config, args.output),
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()

