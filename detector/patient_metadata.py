from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


LABEL_COLUMNS = {
    "IgG_lambda": "IgGλ型单克隆免疫球蛋白",
    "IgG_kappa": "IgGκ型单克隆免疫球蛋白",
    "IgA_lambda": "IgAλ型单克隆免疫球蛋白",
    "IgA_kappa": "IgAκ型单克隆免疫球蛋白",
    "IgM_lambda": "IgMλ型单克隆免疫球蛋白",
    "IgM_kappa": "IgMκ型单克隆免疫球蛋白",
    "free_kappa": "轻链κ型单克隆免疫球蛋白",
    "free_lambda": "轻链λ型单克隆免疫球蛋白",
}

FILENAME_PATTERN = re.compile(
    r"(?P<year>\d{4})\.(?P<month>\d{1,2})\.(?P<day>\d{1,2})\."
    r"(?P<start>\d+)\.(?P<end>\d+)\.bmp$",
    re.IGNORECASE,
)


def compute_multilabel(row: dict[str, Any], negative_text: str = "阴性(-)") -> str:
    def positive(key: str) -> bool:
        value = row.get(LABEL_COLUMNS[key], negative_text)
        if pd.isna(value):
            return False
        return str(value).strip() not in {"", "-", negative_text}

    bits = [0, 0, 0, 0, 0]
    pairs = (
        (0, "IgG_kappa", "IgG_lambda"),
        (1, "IgA_kappa", "IgA_lambda"),
        (2, "IgM_kappa", "IgM_lambda"),
    )
    for heavy_index, kappa_key, lambda_key in pairs:
        if positive(kappa_key) or positive(lambda_key):
            bits[heavy_index] = 1
        if positive(kappa_key):
            bits[3] = 1
        if positive(lambda_key):
            bits[4] = 1
    if positive("free_kappa"):
        bits[3] = 1
    if positive("free_lambda"):
        bits[4] = 1
    return "".join(str(bit) for bit in bits)


class PatientMetadataResolver:
    def __init__(
        self,
        excel_path: str | Path,
        negative_text: str = "阴性(-)",
        date_column: str = "样本日期",
        sample_column: str = "样本号",
        image_column: str = "图片名称",
        barcode_column: str = "条码编号",
    ):
        self.path = Path(excel_path)
        self.negative_text = negative_text
        self.date_column = date_column
        self.sample_column = sample_column
        self.image_column = image_column
        self.barcode_column = barcode_column
        if not self.path.is_file():
            raise FileNotFoundError(f"Patient metadata workbook not found: {self.path}")
        frame = pd.read_excel(self.path, sheet_name=0)

        if {date_column, sample_column}.issubset(frame.columns):
            self.mode = "date_sample"
            frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce").dt.date
            frame[sample_column] = frame[sample_column].map(self._sample_id)
        elif {image_column, barcode_column}.issubset(frame.columns):
            self.mode = "image_barcode"
            frame[image_column] = frame[image_column].ffill().map(self._filename_key)
            frame[barcode_column] = frame[barcode_column].map(self._sample_id)
        else:
            expected = (
                f"({date_column!r}, {sample_column!r}) or "
                f"({image_column!r}, {barcode_column!r})"
            )
            raise ValueError(
                f"Unsupported patient workbook columns in {self.path}; expected {expected}"
            )
        self.frame = frame

    @staticmethod
    def _filename_key(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).strip().replace("\\", "/").rsplit("/", 1)[-1].casefold()

    @staticmethod
    def _sample_id(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value).strip()

    @staticmethod
    def parse_filename(filename: str) -> tuple[date, list[str]] | None:
        match = FILENAME_PATTERN.search(Path(filename).name)
        if not match:
            return None
        day = date(int(match["year"]), int(match["month"]), int(match["day"]))
        sample_ids = [str(value) for value in range(int(match["start"]), int(match["end"]) + 1)]
        return day, sample_ids

    def resolve(self, filename: str, crop_count: int) -> list[dict[str, str | None]]:
        if self.mode == "image_barcode":
            key = self._filename_key(filename)
            rows = self.frame[self.frame[self.image_column] == key].head(crop_count)
            records = [
                {
                    "sample_id": self._sample_id(row[self.barcode_column]) or None,
                    "ground_truth": compute_multilabel(row.to_dict(), self.negative_text),
                }
                for _, row in rows.iterrows()
            ]
            while len(records) < crop_count:
                records.append({"sample_id": None, "ground_truth": None})
            return records

        parsed = self.parse_filename(filename)
        if parsed is None:
            return [{"sample_id": None, "ground_truth": None} for _ in range(crop_count)]
        day, sample_ids = parsed
        day_frame = self.frame[self.frame[self.date_column] == day].set_index(self.sample_column)
        records: list[dict[str, str | None]] = []
        for sample_id in sample_ids[:crop_count]:
            if sample_id not in day_frame.index:
                records.append({"sample_id": sample_id, "ground_truth": None})
                continue
            row = day_frame.loc[sample_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            records.append(
                {
                    "sample_id": sample_id,
                    "ground_truth": compute_multilabel(row.to_dict(), self.negative_text),
                }
            )
        while len(records) < crop_count:
            records.append({"sample_id": None, "ground_truth": None})
        return records
