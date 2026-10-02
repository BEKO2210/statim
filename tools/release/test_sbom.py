#!/usr/bin/env python3
"""Offline unit tests for the release SPDX generator."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SBOM = ROOT / "tools/release/sbom.py"
sys.path.insert(0, str(ROOT / "tools" / "security"))
from vendored_cves import get_pinned_versions  # noqa: E402


class TestSbom(unittest.TestCase):
    def setUp(self) -> None:
        if not shutil.which("readelf"):
            raise unittest.SkipTest("readelf is absent")
        true_binary = Path("/bin/true")
        if not true_binary.is_file():
            raise unittest.SkipTest("/bin/true is absent")
        self.temporary = tempfile.TemporaryDirectory()
        self.work = Path(self.temporary.name)
        self.root_name = "statim-9.8.7-linux-x86_64-cpu"
        self.staging = self.work / self.root_name
        self.staging.mkdir()
        shutil.copyfile(true_binary, self.staging / "statim")
        shutil.copyfile(true_binary, self.staging / "statim-quantize")
        (self.staging / "LICENSE").write_text("test license\n", encoding="utf-8")
        (self.staging / "README.md").write_text("test readme\n", encoding="utf-8")
        self.archive = self.work / f"{self.root_name}.tar.gz"
        self._make_archive(self.archive)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _make_archive(self, destination: Path) -> None:
        with tarfile.open(destination, "w:gz") as bundle:
            bundle.add(self.staging, arcname=self.root_name)

    def _run(self, archive: Path, output: Path) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["SOURCE_DATE_EPOCH"] = "1700000000"
        return subprocess.run(
            [sys.executable, str(SBOM), "--archive", str(archive), "--source", str(ROOT), "--out", str(output)],
            capture_output=True, text=True, env=environment,
        )

    def test_complete_deterministic_sbom(self) -> None:
        first = self.work / "first.json"
        second = self.work / "second.json"
        for output in (first, second):
            result = self._run(self.archive, output)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(first.read_bytes(), second.read_bytes())

        document = json.loads(first.read_text(encoding="utf-8"))
        self.assertTrue({
            "spdxVersion", "dataLicense", "SPDXID", "name", "documentNamespace",
            "creationInfo", "packages", "files", "relationships",
        }.issubset(document))
        self.assertEqual(document["spdxVersion"], "SPDX-2.3")
        self.assertEqual(document["dataLicense"], "CC0-1.0")
        self.assertEqual(document["SPDXID"], "SPDXRef-DOCUMENT")
        self.assertTrue(document["name"])
        self.assertTrue(document["documentNamespace"].startswith("https://github.com/BEKO2210/statim/releases/v9.8.7/"))
        self.assertNotIn("#", document["documentNamespace"])  # SPDX 2.3: a URI without a fragment
        self.assertEqual(document["creationInfo"]["created"], "2023-11-14T22:13:20Z")
        self.assertEqual(
            document["creationInfo"]["creators"],
            ["Tool: statim-sbom.py", "Organization: BEKO2210"],
        )

        packages = {item["name"]: item for item in document["packages"]}
        pinned = get_pinned_versions(ROOT)
        self.assertEqual(packages["ggml"]["versionInfo"], pinned["ggml_commit"])
        self.assertEqual(packages["cpp-httplib"]["versionInfo"], pinned["httplib"])
        self.assertEqual(packages["nlohmann/json"]["versionInfo"], pinned["json"])
        for item in packages.values():
            self.assertTrue(item["SPDXID"])
            self.assertTrue(item["name"])
            self.assertTrue(item["downloadLocation"])
        self.assertIn("libc.so.6", packages)
        self.assertEqual(packages["libc.so.6"]["versionInfo"], "NOASSERTION")
        self.assertIn("provided by the operating system, not shipped", packages["libc.so.6"]["comment"])

        files = {item["fileName"]: item for item in document["files"]}
        for relative, item in files.items():
            expected = (self.staging / relative).read_bytes()
            algorithms = {value["algorithm"]: value["checksumValue"] for value in item["checksums"]}
            self.assertEqual(algorithms["SHA1"], hashlib.sha1(expected).hexdigest())
            self.assertEqual(algorithms["SHA256"], hashlib.sha256(expected).hexdigest())

        identifiers = {document["SPDXID"]}
        identifiers.update(item["SPDXID"] for item in document["packages"])
        identifiers.update(item["SPDXID"] for item in document["files"])
        for relationship in document["relationships"]:
            self.assertIn(relationship["spdxElementId"], identifiers)
            self.assertIn(relationship["relatedSpdxElement"], identifiers)

    def test_missing_statim_fails_clearly(self) -> None:
        (self.staging / "statim").unlink()
        broken = self.work / f"{self.root_name}.tar.gz"
        self._make_archive(broken)
        result = self._run(broken, self.work / "broken.json")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("archive has no regular 'statim' binary", result.stderr)


if __name__ == "__main__":
    unittest.main()
