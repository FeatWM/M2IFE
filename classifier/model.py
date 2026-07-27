from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class MultiLabelClassifier(nn.Module):
    """Checkpoint-compatible classifier used by M2-IFE.

    The historical implementation keeps the torchvision ImageNet classifier
    and adds a second 1000-to-5 linear layer. This layout is retained so the
    existing VGG16, ResNet18 and ConvNeXt-Large checkpoints load unchanged.
    """

    SUPPORTED_BACKBONES = ("vgg16", "resnet18", "convnext_large")

    def __init__(self, backbone: str = "vgg16", num_labels: int = 5):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "vgg16":
            self.model = torchvision.models.vgg16(weights=None)
        elif backbone == "resnet18":
            self.model = torchvision.models.resnet18(weights=None)
        elif backbone == "convnext_large":
            self.model = torchvision.models.convnext_large(weights=None)
        else:
            raise ValueError(
                f"Unsupported backbone {backbone!r}; choose from {self.SUPPORTED_BACKBONES}"
            )
        self.fc = nn.Linear(1000, num_labels)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.model(images)
        logits = self.fc(features)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()
        return {
            "logits": logits,
            "Y_prob": probabilities,
            "Y_hat": predictions,
        }

