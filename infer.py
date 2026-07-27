from __future__ import annotations

import argparse

from config_utils import load_config, resolve_path, with_device
from pipeline import M2IFEPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete M2-IFE detector-to-classifier pipeline."
    )
    parser.add_argument("--input", required=True, help="An IFE image or a directory of IFE images.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default=None, help="cpu, cuda:0, cuda:1, ...")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--no-save-crops", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = with_device(load_config(args.config), args.device)
    inference_cfg = config.get("inference", {})
    output_dir = args.output or inference_cfg.get("output_dir", "outputs/inference")
    pipeline = M2IFEPipeline(config)
    rows = pipeline.run(
        args.input,
        resolve_path(config, output_dir),
        recursive=args.recursive or bool(inference_cfg.get("recursive", False)),
        save_crops=not args.no_save_crops and bool(inference_cfg.get("save_crops", True)),
    )
    successful = sum(1 for row in rows if "error" not in row)
    failed = sum(1 for row in rows if "error" in row)
    print(f"[done] predictions={successful} failed_images={failed}")


if __name__ == "__main__":
    main()

