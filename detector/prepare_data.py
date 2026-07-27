from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def convert_labelme_file(
    json_path: Path,
    output_dir: Path,
    class_map: dict[str, int] | None = None,
) -> Path:
    with json_path.open("r", encoding="utf-8") as handle:
        annotation = json.load(handle)
    image_path = (json_path.parent / annotation["imagePath"]).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image referenced by {json_path} does not exist: {image_path}")
    with Image.open(image_path) as image:
        image_width, image_height = image.size

    labels = []
    for shape in annotation.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue
        points = shape.get("points", [])
        if len(points) != 2:
            continue
        label_name = str(shape.get("label", "0"))
        if class_map is None:
            class_id = int(label_name) if label_name.isdigit() else 0
        else:
            class_id = class_map[label_name]
        (x1, y1), (x2, y2) = points
        x_center = (float(x1) + float(x2)) / 2.0 / image_width
        y_center = (float(y1) + float(y2)) / 2.0 / image_height
        width = abs(float(x2) - float(x1)) / image_width
        height = abs(float(y2) - float(y1)) / image_height
        labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{json_path.stem}.txt"
    output_path.write_text("\n".join(labels), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LabelMe rectangles to YOLO labels.")
    parser.add_argument("--input", required=True, help="Directory containing LabelMe JSON files.")
    parser.add_argument("--output", required=True, help="Directory for YOLO TXT labels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    json_paths = sorted(input_dir.glob("*.json"))
    if not json_paths:
        raise RuntimeError(f"No LabelMe JSON files found in {input_dir}")
    for json_path in json_paths:
        output_path = convert_labelme_file(json_path, output_dir)
        print(f"[converted] {json_path.name} -> {output_path}")


if __name__ == "__main__":
    main()
