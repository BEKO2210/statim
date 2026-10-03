#!/usr/bin/env python3
"""Offline tests for check_vendored.py."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import check_vendored  # noqa: E402


class TestCheckVendored(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.temp_dir.name)
        third_party = self.repo_root / "third_party"
        third_party.mkdir()
        self.httplib = third_party / "httplib.h"
        self.httplib.write_bytes(b"// cpp-httplib fixture\n")
        (third_party / "httplib.version").write_text("cpp-httplib v0.58.0\n", encoding="utf-8")
        self.json_header = third_party / "json.hpp"
        self.json_header.write_text(
            "#define NLOHMANN_JSON_VERSION_MAJOR 3\n"
            "#define NLOHMANN_JSON_VERSION_MINOR 12\n"
            "#define NLOHMANN_JSON_VERSION_PATCH 0\n",
            encoding="utf-8",
        )
        manifest = {
            "httplib.h": {
                "version": "0.58.0",
                "sha256": hashlib.sha256(self.httplib.read_bytes()).hexdigest(),
                "source": "https://example.invalid/httplib.h",
            },
            "json.hpp": {
                "version": "3.12.0",
                "sha256": hashlib.sha256(self.json_header.read_bytes()).hexdigest(),
                "source": "https://example.invalid/json.hpp",
            },
        }
        (third_party / "VENDORED.json").write_text(json.dumps(manifest), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def check(self) -> list[str]:
        with mock.patch("vendored_cves.subprocess.check_output", return_value=b"160000 commit deadbeef\tthird_party/ggml\n"):
            return check_vendored.check_vendored(self.repo_root)

    def test_matching_headers_pass(self) -> None:
        self.assertEqual(self.check(), [])

    def test_tampered_byte_fails(self) -> None:
        self.httplib.write_bytes(self.httplib.read_bytes() + b"x")
        self.assertTrue(any("SHA-256 mismatch for httplib.h" in error for error in self.check()))

    def test_httplib_version_mismatch_fails(self) -> None:
        (self.repo_root / "third_party" / "httplib.version").write_text(
            "cpp-httplib v0.57.0\n", encoding="utf-8"
        )
        self.assertTrue(any("version mismatch for httplib.h" in error for error in self.check()))

    def test_json_version_mismatch_fails(self) -> None:
        content = self.json_header.read_text(encoding="utf-8").replace("PATCH 0", "PATCH 1")
        self.json_header.write_text(content, encoding="utf-8")
        self.assertTrue(any("version mismatch for json.hpp" in error for error in self.check()))


if __name__ == "__main__":
    unittest.main()
