import unittest

from experiments.duration_outlier_analysis import get_ranks, get_spearman


class DurationOutlierAnalysisTests(unittest.TestCase):
    def test_ranks_average_ties(self):
        self.assertEqual(get_ranks([20, 10, 20, 40]), [1.5, 0, 1.5, 3])

    def test_spearman_detects_monotonic_directions(self):
        self.assertAlmostEqual(get_spearman([1, 2, 3], [10, 20, 30]), 1)
        self.assertAlmostEqual(get_spearman([1, 2, 3], [30, 20, 10]), -1)

if __name__ == "__main__":
    unittest.main()
