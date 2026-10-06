#!/usr/bin/env python3
"""Offline tests for the S1Bench section of hf_publish.py (standard library only)."""
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_publish import s1bench_md, s1bench_svg  # noqa: E402

DOC = "docs/reproductions/s1bench-0.10.0-2026-10-07.md"


def summary(holm_p=1.0):
    m = {"a": 0.6573, "a_without_overlap": 0.6562, "b": 0.6378, "lev": 0.6893, "jev": 0.761}
    return {"subsets": 13, "items": 3880,
            "unseen_subsets": ["vitaminc-dev", "boolq", "squad2", "paws", "summeval-relevance",
                               "summeval-consistency", "pubmedqa"],
            "board_subsets": ["vitaminc-dev", "massive-en-US", "boolq", "helpsteer2", "aegis2", "paws"],
            "macro": {"all": m, "unseen": dict(m, a=0.5837), "board": dict(m, a=0.6746)},
            "mean_ece": {"a": 0.126, "b": 0.138},
            "per_subset": {"paws": {"a": 0.676, "b": 0.656, "holm_p": holm_p}}}


class S1BenchSection(unittest.TestCase):
    def test_numbers_come_from_the_summary(self):
        md = s1bench_md(summary(), "0.10.0", "0.7.0", "Beko2210/x", DOC)
        self.assertIn("scores **0.657** macro accuracy, up from 0.638 for 0.7.0", md)
        self.assertIn("| all 13 subsets | **0.657** | 0.638 | 0.689 | 0.761 |", md)
        self.assertIn("| 7 subsets from sources Statim never trained on | **0.584** |", md)
        self.assertIn("6 subsets are in-domain", md)
        self.assertIn(f"https://github.com/BEKO2210/statim/blob/main/{DOC}", md)
        self.assertIn("https://huggingface.co/Beko2210/x/resolve/main/media/s1bench.svg", md)

    def test_significance_is_stated_either_way(self):
        self.assertIn("No single subset changed significantly", s1bench_md(summary(), "0.10.0", "0.7.0", "r", DOC))
        self.assertIn("paws up", s1bench_md(summary(holm_p=0.01), "0.10.0", "0.7.0", "r", DOC))

    def test_speed_line_only_with_two_devices(self):
        s1 = summary()
        self.assertNotIn("server time", s1bench_md(s1, "0.10.0", "0.7.0", "r", DOC))
        s1["latency"] = {"CPU (4 threads)": {"n": 3880, "median_ms": 167.2, "p95_ms": 567.6},
                         "GPU (RTX 3070)": {"n": 3880, "median_ms": 22.4, "p95_ms": 50.1}}
        md = s1bench_md(s1, "0.10.0", "0.7.0", "r", DOC)
        self.assertIn("167 ms on CPU (4 threads), 22 ms on GPU (RTX 3070) (p95 568 ms, 50 ms)", md)
        self.assertNotIn("same answer", md)
        s1["answers_differing_from_same_as"] = 0
        self.assertIn("same answer to every item", s1bench_md(s1, "0.10.0", "0.7.0", "r", DOC))

    def test_chart_is_valid_svg_with_every_bar(self):
        svg = s1bench_svg(summary(), "0.10.0", "0.7.0")
        root = ET.fromstring(svg)
        texts = [t.text for t in root.iter("{http://www.w3.org/2000/svg}text")]
        for value in ("0.761", "0.689", "0.657", "0.638"):
            self.assertIn(value, texts)
        self.assertIn("Statim Decide 0.10.0", texts)


if __name__ == "__main__":
    unittest.main()
