import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from pipeline.engine import M2IFEPipeline


class FakeDetector:
    def __init__(self, count: int):
        self.count = count

    def detect_and_crop(self, source):
        return [
            SimpleNamespace(
                patient_index=index,
                image=Image.new("RGB", (16, 16), color=(index, index, index)),
                confidence=0.9,
                detector_class=0,
                box_xyxy=(0, 0, 16, 16),
            )
            for index in range(1, self.count + 1)
        ]


class FakeClassifier:
    def predict_pil(self, image):
        return {
            "multilabel": "10010",
            "class_9": "IgG-kappa",
            "active_labels": ["IgG", "kappa"],
            "probabilities": [0.8, 0.1, 0.1, 0.7, 0.1],
        }


class PipelineTests(unittest.TestCase):
    def test_four_patient_full_image_flow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "full.bmp"
            Image.new("RGB", (64, 64), color="white").save(source)

            pipeline = M2IFEPipeline.__new__(M2IFEPipeline)
            pipeline.detector = FakeDetector(4)
            pipeline.classifier = FakeClassifier()
            pipeline.metadata_resolver = None
            rows = pipeline.predict_image(source, root / "output", save_crops=True)

            self.assertEqual(len(rows), 4)
            self.assertEqual([row["patient_index"] for row in rows], [1, 2, 3, 4])
            self.assertTrue(all(row["patient_count"] == 4 for row in rows))
            self.assertTrue(all(row["prediction"] == "10010" for row in rows))
            self.assertTrue((root / "output" / "crops" / "full" / "patient_04.png").is_file())


if __name__ == "__main__":
    unittest.main()
