import unittest

import numpy as np

from detector.box_order import order_box_indices, xywh_to_xyxy


class BoxOrderTests(unittest.TestCase):
    def test_nine_box_order(self):
        boxes = np.asarray(
            [
                [100, 100, 20, 20],
                [200, 100, 20, 20],
                [300, 100, 20, 20],
                [100, 200, 20, 20],
                [200, 200, 20, 20],
                [300, 200, 20, 20],
                [100, 300, 20, 20],
                [200, 300, 20, 20],
                [300, 300, 20, 20],
            ],
            dtype=float,
        )
        self.assertEqual(order_box_indices(boxes), [6, 7, 8, 3, 4, 5, 0, 1, 2])

    def test_non_square_count_rejected(self):
        with self.assertRaises(ValueError):
            order_box_indices(np.zeros((5, 4)))

    def test_crop_is_clipped(self):
        self.assertEqual(xywh_to_xyxy((-5, -5, 30, 30), 100, 100), (0, 0, 10, 10))


if __name__ == "__main__":
    unittest.main()

