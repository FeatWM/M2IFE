import unittest

import numpy as np

from classifier.labels import (
    active_labels,
    bits_to_class,
    probabilities_to_bits,
    probabilities_to_9class_scores,
)


class LabelTests(unittest.TestCase):
    def test_threshold_and_post_rule(self):
        probabilities = [0.8, 0.1, 0.1, 0.2, 0.2]
        self.assertEqual(probabilities_to_bits(probabilities, threshold=0.3), "00000")

    def test_valid_nine_class(self):
        bits = probabilities_to_bits([0.8, 0.1, 0.1, 0.8, 0.1], threshold=0.3)
        self.assertEqual(bits, "10010")
        self.assertEqual(bits_to_class(bits), "IgG-kappa")
        self.assertEqual(active_labels(bits), ["IgG", "kappa"])

    def test_nine_class_scores_normalized(self):
        scores = probabilities_to_9class_scores([0.8, 0.1, 0.1, 0.8, 0.1])
        self.assertEqual(scores.shape, (9,))
        self.assertAlmostEqual(float(np.sum(scores)), 1.0, places=7)


if __name__ == "__main__":
    unittest.main()

