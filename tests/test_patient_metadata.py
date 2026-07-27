import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from detector.patient_metadata import PatientMetadataResolver, compute_multilabel


class PatientMetadataTests(unittest.TestCase):
    def test_igg_kappa(self):
        row = {
            "IgGκ型单克隆免疫球蛋白": "阳性",
            "IgGλ型单克隆免疫球蛋白": "阴性(-)",
        }
        self.assertEqual(compute_multilabel(row), "10010")

    def test_negative(self):
        self.assertEqual(compute_multilabel({}), "00000")

    def test_legacy_image_barcode_workbook(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.xlsx"
            pd.DataFrame(
                {
                    "图片名称": ["2023.6.21-61-62.bmp", None],
                    "条码编号": [1761, 1762],
                    "IgGκ型单克隆免疫球蛋白": ["+", "-"],
                    "IgGλ型单克隆免疫球蛋白": ["-", "-"],
                }
            ).to_excel(path, index=False)

            resolver = PatientMetadataResolver(path)
            records = resolver.resolve("2023.6.21-61-62.bmp", crop_count=2)

        self.assertEqual(records[0], {"sample_id": "1761", "ground_truth": "10010"})
        self.assertEqual(records[1], {"sample_id": "1762", "ground_truth": "00000"})


if __name__ == "__main__":
    unittest.main()
