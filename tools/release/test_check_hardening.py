#!/usr/bin/env python3
"""Unit tests for check_hardening.py.

Compiles a tiny C program twice:
- once hardened (must pass)
- once with -no-pie -z norelro -fno-stack-protector (must fail)
Skips if cc or readelf is absent.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "tools/release/check_hardening.py"

TINY_C_SOURCE = """#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    char buf[16];
    memcpy(buf, argv[0], argc);
    printf("%s\\n", buf);
    return buf[0];
}
"""


class TestCheckHardening(unittest.TestCase):
    def setUp(self) -> None:
        if not shutil.which("cc"):
            raise unittest.SkipTest("cc compiler is absent")
        if not shutil.which("readelf"):
            raise unittest.SkipTest("readelf is absent")
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.src_file = Path(self.tmp_dir.name) / "test.c"
        self.src_file.write_text(TINY_C_SOURCE, encoding="utf-8")
        self.hardened_bin = Path(self.tmp_dir.name) / "test_hardened"
        self.unhardened_bin = Path(self.tmp_dir.name) / "test_unhardened"

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_hardened_and_unhardened_binaries(self) -> None:
        # Compile hardened binary
        cmd_hardened = [
            "cc",
            "-O2",
            "-fPIE",
            "-pie",
            "-fstack-protector-strong",
            "-Wl,-z,relro,-z,now",
            "-Wl,-z,noexecstack",
            "-U_FORTIFY_SOURCE",
            "-D_FORTIFY_SOURCE=2",
            str(self.src_file),
            "-o",
            str(self.hardened_bin),
        ]
        res_h = subprocess.run(cmd_hardened, capture_output=True, text=True)
        self.assertEqual(res_h.returncode, 0, f"Failed to compile hardened binary: {res_h.stderr}")

        # Compile unhardened binary
        cmd_unhardened = [
            "cc",
            "-O2",
            "-no-pie",
            "-z",
            "norelro",
            "-fno-stack-protector",
            str(self.src_file),
            "-o",
            str(self.unhardened_bin),
        ]
        res_u = subprocess.run(cmd_unhardened, capture_output=True, text=True)
        self.assertEqual(res_u.returncode, 0, f"Failed to compile unhardened binary: {res_u.stderr}")

        # Check hardened binary with check_hardening.py -> must pass (exit 0)
        proc_pass = subprocess.run(
            [sys.executable, str(CHECKER), str(self.hardened_bin)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            proc_pass.returncode,
            0,
            f"Expected check_hardening.py to pass on hardened binary:\nSTDOUT: {proc_pass.stdout}\nSTDERR: {proc_pass.stderr}",
        )
        self.assertIn("PASS", proc_pass.stdout)
        self.assertIn("__stack_chk_fail", proc_pass.stdout)

        # Check unhardened binary with check_hardening.py -> must fail (exit 1)
        proc_fail = subprocess.run(
            [sys.executable, str(CHECKER), str(self.unhardened_bin)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            proc_fail.returncode,
            1,
            f"Expected check_hardening.py to fail on unhardened binary:\nSTDOUT: {proc_fail.stdout}\nSTDERR: {proc_fail.stderr}",
        )
        self.assertIn("FAIL", proc_fail.stderr)
        self.assertIn("missing symbol: __stack_chk_fail", proc_fail.stderr)


class TestGlibcBaseline(unittest.TestCase):
    """--max-glibc reads the highest GLIBC_x.y that an imported symbol needs (READINESS P1 #55)."""

    def test_highest_version_wins(self) -> None:
        sys.path.insert(0, str(CHECKER.parent))
        import check_hardening as ch
        out = ("     5: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND memcpy@GLIBC_2.14 (3)\n"
               "     6: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND __isoc23_strtol@GLIBC_2.38 (5)\n"
               "     7: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND pthread_create@GLIBC_2.2.5 (2)\n")
        self.assertEqual(ch.glibc_needed(out), (2, 38))
        self.assertIsNone(ch.glibc_needed("no versioned symbols"))

    def test_system_binary_against_a_tight_and_a_loose_limit(self) -> None:
        if not shutil.which("readelf"):
            raise unittest.SkipTest("readelf is absent")
        target = Path("/bin/true")
        loose = subprocess.run([sys.executable, str(CHECKER), "--max-glibc", "99.0", str(target)],
                               capture_output=True, text=True)
        tight = subprocess.run([sys.executable, str(CHECKER), "--max-glibc", "2.0", str(target)],
                               capture_output=True, text=True)
        self.assertNotIn("needs glibc", loose.stdout + loose.stderr)
        self.assertIn("needs glibc", tight.stderr)


if __name__ == "__main__":
    unittest.main()
