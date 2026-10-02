#!/usr/bin/env python3
"""test_vendored_cves.py - Offline unit tests for vendored_cves.py."""

from __future__ import annotations

import datetime
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock
import urllib.error

# Ensure tools/security is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import vendored_cves  # noqa: E402


class TestRangeParser(unittest.TestCase):
    """Test range parsing logic against the required six forms and edge cases."""

    def test_form_1_less_than_or_equal(self) -> None:
        expr = "<= 0.43.4"
        self.assertTrue(vendored_cves.is_version_in_range("0.43.0", expr))
        self.assertTrue(vendored_cves.is_version_in_range("0.43.4", expr))
        self.assertFalse(vendored_cves.is_version_in_range("0.44.0", expr))
        self.assertFalse(vendored_cves.is_version_in_range("0.58.0", expr))

    def test_form_2_strictly_less(self) -> None:
        expr = "< 0.22.0"
        self.assertTrue(vendored_cves.is_version_in_range("0.21.0", expr))
        self.assertFalse(vendored_cves.is_version_in_range("0.22.0", expr))
        self.assertFalse(vendored_cves.is_version_in_range("0.58.0", expr))

    def test_form_3_compound_range(self) -> None:
        expr = ">= 0.31.0, < 0.47.0"
        self.assertFalse(vendored_cves.is_version_in_range("0.30.0", expr))
        self.assertTrue(vendored_cves.is_version_in_range("0.31.0", expr))
        self.assertTrue(vendored_cves.is_version_in_range("0.40.0", expr))
        self.assertFalse(vendored_cves.is_version_in_range("0.47.0", expr))
        self.assertFalse(vendored_cves.is_version_in_range("0.58.0", expr))

    def test_form_4_bare_version(self) -> None:
        expr = "0.21.0"
        self.assertTrue(vendored_cves.is_version_in_range("0.21.0", expr))
        self.assertFalse(vendored_cves.is_version_in_range("0.22.0", expr))
        self.assertFalse(vendored_cves.is_version_in_range("0.58.0", expr))

    def test_form_5_dotted_v_prefix(self) -> None:
        expr = "v.0.50.0"
        self.assertTrue(vendored_cves.is_version_in_range("0.50.0", expr))
        self.assertFalse(vendored_cves.is_version_in_range("0.51.0", expr))
        self.assertFalse(vendored_cves.is_version_in_range("0.58.0", expr))

    def test_form_6_commit_annotation(self) -> None:
        expr = "v0.49.0 (commit 0fa4912)"
        self.assertTrue(vendored_cves.is_version_in_range("0.49.0", expr))
        self.assertFalse(vendored_cves.is_version_in_range("0.50.0", expr))
        self.assertFalse(vendored_cves.is_version_in_range("0.58.0", expr))

    def test_malformed_range_fails_closed(self) -> None:
        # Malformed ranges must be treated as affected (fail closed)
        with mock.patch("sys.stderr", new_callable=io.StringIO) as fake_err:
            res = vendored_cves.is_version_in_range("0.58.0", "!!invalid-range-syntax!!")
            self.assertTrue(res)
            self.assertIn("treating as affected", fake_err.getvalue())


class TestVendoredCvesTool(unittest.TestCase):
    """Offline tests for vendored_cves tool execution and exit codes."""

    def setUp(self) -> None:
        self.repo_root = SCRIPT_DIR.parent.parent
        self.triage_path = self.repo_root / "tools" / "security" / "cve-triage.json"
        self.today = datetime.date(2026, 10, 1)

    def _default_clean_fetcher(self, url: str, data: bytes | None = None, headers: dict | None = None, timeout: int = 15) -> bytes:
        if "api.osv.dev/v1/query" in url:
            req_data = json.loads(data.decode("utf-8")) if data else {}
            if req_data.get("version") == "v0.43.0":
                # Control query
                return json.dumps({
                    "vulns": [{"id": "CVE-2026-45352", "severity": [{"type": "CVSS_V3", "score": "5.3"}]}]
                }).encode("utf-8")
            # Query for v0.58.0
            return json.dumps({"vulns": []}).encode("utf-8")
        elif "repos/yhirose/cpp-httplib/security-advisories" in url:
            return json.dumps([]).encode("utf-8")
        elif "repos/nlohmann/json/security-advisories" in url:
            return json.dumps([]).encode("utf-8")
        elif "repos/ggml-org/ggml/security-advisories" in url:
            return json.dumps([]).encode("utf-8")
        elif "repos/ggml-org/llama.cpp/security-advisories" in url:
            return json.dumps([]).encode("utf-8")
        raise vendored_cves.NetworkError(f"Unexpected URL: {url}")

    def test_clean_run_returns_zero(self) -> None:
        with mock.patch("sys.stdout", new_callable=io.StringIO):
            rc = vendored_cves.run_checks(
                repo_root=self.repo_root,
                fetch_fn=self._default_clean_fetcher,
                triage_path=self.triage_path,
                today=self.today,
                run_control=True,
            )
        self.assertEqual(rc, 0)

    def test_main_cli_control_success(self) -> None:
        """Test main CLI entrypoint with --control."""
        with mock.patch("vendored_cves.default_http_fetch", side_effect=self._default_clean_fetcher):
            with mock.patch("sys.stdout", new_callable=io.StringIO):
                rc = vendored_cves.main(["--control"])
                self.assertEqual(rc, 0)

    def test_main_cli_control_failure(self) -> None:
        """Test main CLI entrypoint with --control when OSV returns 0 vulns."""
        def empty_fetch(url: str, data: bytes | None = None, headers: dict | None = None, timeout: int = 15) -> bytes:
            return json.dumps({"vulns": []}).encode("utf-8")

        with mock.patch("vendored_cves.default_http_fetch", side_effect=empty_fetch):
            with mock.patch("sys.stderr", new_callable=io.StringIO):
                rc = vendored_cves.main(["--control"])
                self.assertEqual(rc, 2)

    def test_planted_advisory_matching_version_exits_one(self) -> None:
        """Planted advisory matching our pinned version makes the tool exit 1."""
        def mock_fetch(url: str, data: bytes | None = None, headers: dict | None = None, timeout: int = 15) -> bytes:
            if "repos/yhirose/cpp-httplib/security-advisories" in url:
                return json.dumps([
                    {
                        "ghsa_id": "GHSA-planted-httplib-001",
                        "summary": "Planted critical RCE in httplib",
                        "severity": "critical",
                        "vulnerabilities": [
                            {"vulnerable_version_range": "<= 0.58.0"}
                        ],
                    }
                ]).encode("utf-8")
            return self._default_clean_fetcher(url, data, headers, timeout)

        with mock.patch("sys.stdout", new_callable=io.StringIO), mock.patch("sys.stderr", new_callable=io.StringIO):
            rc = vendored_cves.run_checks(
                repo_root=self.repo_root,
                fetch_fn=mock_fetch,
                triage_path=self.triage_path,
                today=self.today,
                run_control=True,
            )
        self.assertEqual(rc, 1)

    def test_planted_untriaged_ggml_advisory_exits_one(self) -> None:
        """Planted untriaged ggml/GGUF high/critical advisory exits 1."""
        def mock_fetch(url: str, data: bytes | None = None, headers: dict | None = None, timeout: int = 15) -> bytes:
            if "repos/ggml-org/llama.cpp/security-advisories" in url:
                return json.dumps([
                    {
                        "ghsa_id": "GHSA-planted-ggml-untriaged",
                        "cve_id": "CVE-2026-99999",
                        "summary": "Untriaged heap overflow in GGUF tensor parser",
                        "severity": "high",
                        "description": "Critical flaw parsing tensor data in ggml",
                    }
                ]).encode("utf-8")
            return self._default_clean_fetcher(url, data, headers, timeout)

        with mock.patch("sys.stdout", new_callable=io.StringIO), mock.patch("sys.stderr", new_callable=io.StringIO):
            rc = vendored_cves.run_checks(
                repo_root=self.repo_root,
                fetch_fn=mock_fetch,
                triage_path=self.triage_path,
                today=self.today,
                run_control=True,
            )
        self.assertEqual(rc, 1)

    def _run(self, fetch) -> int:
        with mock.patch("sys.stdout", new_callable=io.StringIO), mock.patch("sys.stderr", new_callable=io.StringIO):
            return vendored_cves.run_checks(repo_root=self.repo_root, fetch_fn=fetch,
                                            triage_path=self.triage_path, today=self.today,
                                            run_control=True)

    def test_untriaged_advisory_without_keywords_exits_one(self) -> None:
        """A high llama.cpp advisory that names neither ggml nor GGUF still needs triage."""
        def fetch(url, data=None, headers=None, timeout=15):
            if "repos/ggml-org/llama.cpp/security-advisories" in url:
                return json.dumps([{"ghsa_id": "GHSA-planted-nokeyword", "summary": "Heap overflow in model loading",
                                    "severity": "high", "description": "A crafted file corrupts memory."}]).encode()
            return self._default_clean_fetcher(url, data, headers, timeout)
        self.assertEqual(self._run(fetch), 1)

    def test_second_page_advisory_is_read(self) -> None:
        """A planted advisory on page 2 is found: a full first page must not end the listing."""
        filler = [{"ghsa_id": f"GHSA-filler-{i}", "summary": "old", "severity": "low",
                   "vulnerabilities": [{"vulnerable_version_range": "< 0.1.0"}]} for i in range(vendored_cves.PAGE_SIZE)]
        def fetch(url, data=None, headers=None, timeout=15):
            if "repos/yhirose/cpp-httplib/security-advisories" in url:
                if url.endswith("&page=1"):
                    return json.dumps(filler).encode()
                if url.endswith("&page=2"):
                    return json.dumps([{"ghsa_id": "GHSA-planted-page2", "summary": "planted", "severity": "critical",
                                        "vulnerabilities": [{"vulnerable_version_range": "<= 0.58.0"}]}]).encode()
            return self._default_clean_fetcher(url, data, headers, timeout)
        self.assertEqual(self._run(fetch), 1)

    def test_endless_pages_exit_two(self) -> None:
        """More pages than MAX_PAGES is a database error, never a silent cap."""
        full = [{"ghsa_id": "GHSA-x", "summary": "old", "severity": "low",
                 "vulnerabilities": [{"vulnerable_version_range": "< 0.1.0"}]}] * vendored_cves.PAGE_SIZE
        def fetch(url, data=None, headers=None, timeout=15):
            if "repos/yhirose/cpp-httplib/security-advisories" in url:
                return json.dumps(full).encode()
            return self._default_clean_fetcher(url, data, headers, timeout)
        self.assertEqual(self._run(fetch), 2)

    def test_expired_triage_entry_exits_one(self) -> None:
        """Expired triage entry exits 1."""
        # Simulated run with today set to 2028-01-01 (past the 2027-10-01 review_by date)
        future_date = datetime.date(2028, 1, 1)
        with mock.patch("sys.stdout", new_callable=io.StringIO), mock.patch("sys.stderr", new_callable=io.StringIO):
            rc = vendored_cves.run_checks(
                repo_root=self.repo_root,
                fetch_fn=self._default_clean_fetcher,
                triage_path=self.triage_path,
                today=future_date,
                run_control=True,
            )
        self.assertEqual(rc, 1)

    def test_moderate_advisory_exits_zero_with_warning(self) -> None:
        """Moderate or low advisory exits 0 with a warning."""
        def mock_fetch(url: str, data: bytes | None = None, headers: dict | None = None, timeout: int = 15) -> bytes:
            if "repos/ggml-org/llama.cpp/security-advisories" in url:
                return json.dumps([
                    {
                        "ghsa_id": "GHSA-planted-medium-advisory",
                        "summary": "Moderate GGUF parsing notice",
                        "severity": "medium",
                        "description": "Informational issue in gguf tensor metadata",
                    }
                ]).encode("utf-8")
            return self._default_clean_fetcher(url, data, headers, timeout)

        with mock.patch("sys.stdout", new_callable=io.StringIO) as fake_out:
            rc = vendored_cves.run_checks(
                repo_root=self.repo_root,
                fetch_fn=mock_fetch,
                triage_path=self.triage_path,
                today=self.today,
                run_control=True,
            )
            self.assertEqual(rc, 0)
            self.assertIn("WARN:", fake_out.getvalue())

    def test_network_failure_mocked_urlopen_exits_two(self) -> None:
        """Network failure (a mocked urlopen raising) exits 2."""
        with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            with mock.patch("sys.stderr", new_callable=io.StringIO):
                rc = vendored_cves.run_checks(
                    repo_root=self.repo_root,
                    fetch_fn=vendored_cves.default_http_fetch,
                    triage_path=self.triage_path,
                    today=self.today,
                    run_control=True,
                )
        self.assertEqual(rc, 2)

    def test_control_step_empty_osv_coverage_exits_two(self) -> None:
        """When OSV coverage is lost (0 vulns for v0.43.0), control check exits 2."""
        def empty_control_fetch(url: str, data: bytes | None = None, headers: dict | None = None, timeout: int = 15) -> bytes:
            return json.dumps({"vulns": []}).encode("utf-8")

        with mock.patch("sys.stderr", new_callable=io.StringIO):
            rc = vendored_cves.check_osv_control(empty_control_fetch)
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
