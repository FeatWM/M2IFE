from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def order_box_indices(
    boxes_xywh: Sequence[Sequence[float]] | np.ndarray,
    direction: str = "bottom_to_top_left_to_right",
) -> list[int]:
    """Return stable patient-box order used by the original M2-IFE pipeline.

    Boxes are grouped into equally sized rows using their y centers. Rows are
    visited from bottom to top and boxes inside each row from left to right.
    The trained detector expects a square grid (four or nine boxes).
    """

    boxes = np.asarray(boxes_xywh, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] < 4:
        raise ValueError(f"Expected an N x 4 xywh array, got shape {boxes.shape}")
    count = len(boxes)
    if count == 0:
        return []

    row_size = int(round(math.sqrt(count)))
    if row_size * row_size != count:
        raise ValueError(f"Patient boxes must form a square grid; received {count} boxes")
    if direction != "bottom_to_top_left_to_right":
        raise ValueError(f"Unsupported box order: {direction}")

    indices = np.arange(count)
    by_y_desc = indices[np.argsort(boxes[:, 1], kind="stable")[::-1]]
    ordered: list[int] = []
    for row in np.array_split(by_y_desc, row_size):
        row_sorted = row[np.argsort(boxes[row, 0], kind="stable")]
        ordered.extend(int(index) for index in row_sorted)
    return ordered


def xywh_to_xyxy(box_xywh: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x_center, y_center, box_width, box_height = (float(value) for value in box_xywh[:4])
    x1 = max(0, int(round(x_center - box_width / 2)))
    y1 = max(0, int(round(y_center - box_height / 2)))
    x2 = min(width, int(round(x_center + box_width / 2)))
    y2 = min(height, int(round(y_center + box_height / 2)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop after clipping: {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2

