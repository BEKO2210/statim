#!/usr/bin/env python3
"""Tests for parity_summary.py on ctest JUnit output."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parity_summary  # noqa: E402

JUNIT = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="statim" tests="3">
 <testcase name="model_parity_multilingual" status="run"><system-out>
items 240 | argmax agree 240/240 | max |dlogit| 5.94e-04 | max |dact| 0.00e+00 | 535.1 ms/state avg
PASS (tol 1.0e-03)</system-out></testcase>
 <testcase name="engine_parity_english" status="run"><system-out>
token ids identical: 240/240 | choice identical: 90/90 | answer fields over tol: 0 | worst |diff| 1.00e-04
PASS (tol 1.5e-04)</system-out></testcase>
 <testcase name="security" status="run"><system-out>all good</system-out></testcase>
</testsuite>
"""


class ParitySummary(unittest.TestCase):
    def test_extracts_values_and_skips_other_tests(self):
        rows = parity_summary.summarize(JUNIT)
        self.assertEqual([r["test"] for r in rows], ["model_parity_multilingual", "engine_parity_english"])
        self.assertEqual(rows[0]["max_dlogit"], "5.94e-04")
        self.assertEqual(rows[0]["argmax"], "240/240")
        self.assertEqual(rows[1]["worst_answer_diff"], "1.00e-04")
        self.assertEqual(rows[1]["tolerance"], "1.5e-04")

    def test_parity_test_without_value_fails(self):
        broken = JUNIT.replace("max |dlogit| 5.94e-04", "output cut")
        with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
            f.write(broken)
        try:
            self.assertEqual(parity_summary.main([f.name]), 1)
        finally:
            os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()
