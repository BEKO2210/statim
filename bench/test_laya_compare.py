#!/usr/bin/env python3
"""Standard-library tests for bench/laya_compare.py."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import laya_compare  # noqa: E402


class SummaryTests(unittest.TestCase):
    def test_short_mean_excludes_long_state_by_token_count(self):
        timings = [float(i) for i in range(1, 31)]
        tokens = [100] * 30
        tokens[7] = 770

        result = laya_compare.summarize(timings, tokens)

        expected_short = sum(value for i, value in enumerate(timings) if i != 7) / 29
        self.assertEqual(result["short"]["states"], 29)
        self.assertEqual(result["short"]["max_tokens"], 100)
        self.assertAlmostEqual(result["short"]["mean_ms"], expected_short)
        self.assertEqual(result["long"], {
            "states": 1, "state_index": 7, "tokens": 770, "median_ms": 8.0,
        })
        self.assertEqual(result["per_state_ms"], timings)
        self.assertEqual(result["state_token_counts"], tokens)
        self.assertEqual(result["p50_ms"], 15.5)
        self.assertEqual(result["p95_ms"], 29.0)

    def test_summary_requires_one_longest_state(self):
        with self.assertRaises(ValueError):
            laya_compare.summarize([1.0, 2.0], [770, 770])

    def test_summary_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            laya_compare.summarize([1.0], [10, 20])


class ArgumentParsingTests(unittest.TestCase):
    def test_parity_arguments(self):
        args = laya_compare.parse_args([
            "parity", "--model", "model", "--rows", "rows.jsonl", "--out", "out.json",
        ])
        self.assertEqual(args.subcommand, "parity")
        self.assertEqual(args.model, Path("model"))
        self.assertEqual(args.rows, Path("rows.jsonl"))
        self.assertEqual(args.out, Path("out.json"))

    def test_raw_defaults(self):
        args = laya_compare.parse_args([
            "raw", "--model", "model", "--rows", "rows.jsonl", "--out", "out.json",
        ])
        self.assertEqual(args.threads, [1, 4, 8])
        self.assertEqual(args.repeats, 3)
        self.assertIsNone(args._worker_thread)

    def test_raw_explicit_arguments(self):
        args = laya_compare.parse_args([
            "raw", "--model", "m", "--rows", "r", "--threads", "2", "6",
            "--repeats", "5", "--out", "o",
        ])
        self.assertEqual(args.threads, [2, 6])
        self.assertEqual(args.repeats, 5)

    def test_startup_arguments(self):
        args = laya_compare.parse_args(["startup", "--model", "m", "--out", "o"])
        self.assertEqual(args.subcommand, "startup")
        self.assertFalse(args._worker)


if __name__ == "__main__":
    unittest.main()
