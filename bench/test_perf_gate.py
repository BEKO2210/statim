#!/usr/bin/env python3
"""Standard-library-only tests for the performance gate statistics."""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_gate import timing_decision


class TimingDecisionTest(unittest.TestCase):
    def test_clear_regression_is_slower(self):
        result = timing_decision([100.0, 101.0, 99.0], [106.0, 107.0, 105.0], .02)
        self.assertEqual(result["verdict"], "slower")

    def test_noise_masks_small_difference(self):
        result = timing_decision([95.0, 105.0, 100.0], [104.0, 106.0, 105.0], .02)
        self.assertEqual(result["verdict"], "same")

    def test_inconsistent_difference_is_same(self):
        result = timing_decision([100.0, 100.0, 100.0], [110.0, 90.0, 110.0], .02)
        self.assertEqual(result["verdict"], "same")

    def test_clear_improvement_is_faster(self):
        result = timing_decision([100.0, 101.0, 99.0], [94.0, 95.0, 93.0], .02)
        self.assertEqual(result["verdict"], "faster")

    def test_cell_noise_floor_masks_lucky_state(self):
        result = timing_decision([100.0, 100.0], [110.0, 111.0], .02, noise_floor=.04)
        self.assertEqual(result["verdict"], "same")


if __name__ == "__main__":
    unittest.main()
