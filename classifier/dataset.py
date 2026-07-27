from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .labels import validate_bits


def build_transform(image_size: int = 224) -> Callable:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class IFEFoldDataset(Dataset):
    def __init__(
        self,
        image_root: str | Path,
        fold_csv: str | Path,
        split: str,
        image_size: int = 224,
        transform: Callable | None = None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val or test")
        self.image_root = Path(image_root)
        self.fold_csv = Path(fold_csv)
        self.split = split
        self.transform = transform or build_transform(image_size)

        frame = pd.read_csv(self.fold_csv, index_col=0, dtype=str)
        image_column = split
        label_column = f"{split}_label"
        if image_column not in frame or label_column not in frame:
            raise ValueError(f"{self.fold_csv} must contain {image_column} and {label_column}")
        subset = frame[[image_column, label_column]].dropna()
        subset = subset[
            subset[image_column].astype(str).str.strip().ne("")
            & subset[label_column].astype(str).str.strip().ne("")
        ]
        self.images = subset[image_column].astype(str).tolist()
        self.labels = [validate_bits(value) for value in subset[label_column].astype(str)]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        image_name = self.images[index]
        image_path = self.image_root / image_name
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        label = torch.from_numpy(
            np.asarray([int(bit) for bit in self.labels[index]], dtype=np.float32)
        )
        return image_tensor, label, image_name

