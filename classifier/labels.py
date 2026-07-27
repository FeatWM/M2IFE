from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


LABEL_NAMES = ("IgG", "IgA", "IgM", "kappa", "lambda")
CLASS_ORDER_9 = (
    "negative",
    "IgG-kappa",
    "IgG-lambda",
    "IgA-kappa",
    "IgA-lambda",
    "IgM-kappa",
    "IgM-lambda",
    "free-kappa",
    "free-lambda",
)
BITS_TO_CLASS_9 = {
    "00000": "negative",
    "10010": "IgG-kappa",
    "10001": "IgG-lambda",
    "01010": "IgA-kappa",
    "01001": "IgA-lambda",
    "00110": "IgM-kappa",
    "00101": "IgM-lambda",
    "00010": "free-kappa",
    "00001": "free-lambda",
}


def validate_bits(bits: str) -> str:
    value = str(bits).strip()
    if len(value) != 5 or set(value) - {"0", "1"}:
        raise ValueError(f"Expected a five-bit multilabel string, received {bits!r}")
    return value


def apply_post_rule(bits: str, enabled: bool = True) -> str:
    value = validate_bits(bits)
    if enabled and value.endswith("00"):
        return "00000"
    return value


def probabilities_to_bits(
    probabilities: Sequence[float] | np.ndarray,
    threshold: float = 0.3,
    apply_endswith_00_rule: bool = True,
) -> str:
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(values) != 5:
        raise ValueError(f"Expected five probabilities, received {len(values)}")
    bits = "".join("1" if value >= threshold else "0" for value in values)
    return apply_post_rule(bits, apply_endswith_00_rule)


def bits_to_class(bits: str) -> str | None:
    return BITS_TO_CLASS_9.get(validate_bits(bits))


def active_labels(bits: str) -> list[str]:
    value = validate_bits(bits)
    return [name for name, bit in zip(LABEL_NAMES, value) if bit == "1"]


def bit_array(bits_values: Iterable[str]) -> np.ndarray:
    return np.asarray([[int(bit) for bit in validate_bits(value)] for value in bits_values], dtype=np.int64)


def probabilities_to_9class_scores(probabilities: Sequence[float]) -> np.ndarray:
    probabilities_array = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    scores = []
    inverse = {class_name: bits for bits, class_name in BITS_TO_CLASS_9.items()}
    for class_name in CLASS_ORDER_9:
        bits = inverse[class_name]
        score = 1.0
        for bit, probability in zip(bits, probabilities_array):
            score *= probability if bit == "1" else (1.0 - probability)
        scores.append(score)
    scores_array = np.asarray(scores, dtype=np.float64)
    total = scores_array.sum()
    return scores_array / total if total > 0 else scores_array

