#!/usr/bin/env python3
"""Tests for fail_on_skip.py on real ctest output lines."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fail_on_skip  # noqa: E402

PASSED = """\
 7/16 Test  #7: gguf_preflight ...................   Passed    0.13 sec
 8/16 Test  #8: security_http ....................   Passed   32.53 sec
100% tests passed, 0 tests failed out of 16
"""
SKIPPED = """\
 8/16 Test  #8: security_http ....................***Skipped   0.05 sec
 9/16 Test  #9: api_contract .....................***Skipped   0.05 sec
10/16 Test #10: server_microbatch ................   Passed    4.59 sec
100% tests passed, 0 tests failed out of 16
"""


class FailOnSkip(unittest.TestCase):
    def run_on(self, text):
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
            f.write(text)
        try:
            return fail_on_skip.main([f.name])
        finally:
            os.unlink(f.name)

    def test_all_passed(self):
        self.assertEqual(self.run_on(PASSED), 0)

    def test_skipped_security_http_fails(self):
        self.assertEqual(fail_on_skip.skipped_tests(SKIPPED), ["security_http", "api_contract"])
        self.assertEqual(self.run_on(SKIPPED), 1)

    def test_usage(self):
        self.assertEqual(fail_on_skip.main([]), 2)


if __name__ == "__main__":
    unittest.main()
