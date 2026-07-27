from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .checkpoints import extract_model_state, load_checkpoint_file


def convert_checkpoint(input_path: Path, output_path: Path) -> None:
    checkpoint = load_checkpoint_file(input_path, map_location="cpu", trust_legacy=True)
    state = extract_model_state(checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state}, output_path)
    print(f"[converted] {input_path} -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one trusted historical checkpoint to a tensor-only state_dict."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_checkpoint(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
