#!/usr/bin/env python3
"""Verify defense-in-depth hardening flags on ELF binaries.

Checks:
- Position-Independent Executable (Type: DYN)
- Full RELRO (GNU_RELRO segment and BIND_NOW or FLAGS_1 NOW)
- Non-executable stack (GNU_STACK segment without E flag)
- Stack protector (imports __stack_chk_fail)
- Fortify source (imports at least one __*_chk symbol)

Standard library only.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def is_elf_file(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


def inspect_elf(path: Path, readelf: str) -> tuple[list[str], list[str]]:
    """Run readelf on an ELF binary and check hardening properties.

    Returns (errors, fortify_symbols).
    """
    if not path.is_file():
        return [f"not a file: {path}"], []
    if not is_elf_file(path):
        return [f"not an ELF binary: {path}"], []

    cmd = [readelf, "-h", "-l", "-d", "-W", "--dyn-syms", str(path)]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env={**os.environ, "LC_ALL": "C", "LANG": "C"},
        )
    except FileNotFoundError:
        return [f"readelf executable not found: {readelf}"], []

    if proc.returncode != 0:
        err_msg = proc.stderr.strip() or f"readelf exited with code {proc.returncode}"
        return [f"readelf error: {err_msg}"], []

    out = proc.stdout
    errors: list[str] = []

    # 1. PIE check: Type: DYN
    m_type = re.search(r"^\s*Typ(?:e)?:\s+(\S+)", out, re.MULTILINE)
    elf_type = m_type.group(1) if m_type else "UNKNOWN"
    if elf_type != "DYN":
        errors.append(f"PIE missing (ELF type is {elf_type}, expected DYN)")

    # 2. RELRO check: GNU_RELRO segment + BIND_NOW or FLAGS_1 with NOW
    has_relro = bool(re.search(r"^\s*GNU_RELRO\b", out, re.MULTILINE))
    has_bind_now = bool(re.search(r"\(FLAGS\)\s+.*?\bBIND_NOW\b", out)) or bool(
        re.search(r"\(FLAGS_1\)\s+.*?\bNOW\b", out)
    )
    if not has_relro:
        errors.append("GNU_RELRO segment missing")
    if not has_bind_now:
        errors.append("BIND_NOW / FLAGS_1 NOW missing (full RELRO required)")

    # 3. Non-executable stack: GNU_STACK without E flag
    m_stack = re.search(r"^\s*GNU_STACK\s+(.*)$", out, re.MULTILINE)
    if not m_stack:
        errors.append("GNU_STACK segment missing")
    else:
        tokens = m_stack.group(1).split()
        flags = tokens[-2] if len(tokens) >= 2 and tokens[-1].startswith("0x") else tokens[-1]
        if "E" in flags:
            errors.append(f"GNU_STACK is executable (flags: {flags})")

    # 4 & 5. Imported symbols: __stack_chk_fail and fortify __*_chk
    imported_symbols: list[str] = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[6] in ("UND", "*UND*"):
            sym = parts[7].split("@")[0]
            imported_symbols.append(sym)

    if "__stack_chk_fail" not in imported_symbols:
        errors.append("stack protector missing: missing symbol: __stack_chk_fail")

    chk_symbols = sorted({s for s in imported_symbols if s.startswith("__") and s.endswith("_chk")})
    if not chk_symbols:
        errors.append("fortify missing: missing symbol: __*_chk")

    return errors, chk_symbols


def check_target(target_path: Path, readelf: str) -> tuple[int, int]:
    """Check a binary file or tar.gz archive.

    Returns (passed_count, failed_count).
    """
    if not target_path.exists():
        print(f"FAIL: {target_path}: file does not exist", file=sys.stderr)
        return 0, 1

    if target_path.name.endswith(".tar.gz") or target_path.name.endswith(".tgz"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                with tarfile.open(target_path, "r:*") as tar:
                    tar.extractall(path=tmp_dir)
            except Exception as e:
                print(f"FAIL: {target_path}: failed to extract archive: {e}", file=sys.stderr)
                return 0, 1

            extracted_root = Path(tmp_dir)
            elf_files: list[Path] = []
            for root, _, files in os.walk(extracted_root):
                for f in files:
                    fp = Path(root) / f
                    if is_elf_file(fp):
                        elf_files.append(fp)

            if not elf_files:
                print(f"FAIL: {target_path}: no ELF binaries found in archive", file=sys.stderr)
                return 0, 1

            passed = 0
            failed = 0
            for elf in elf_files:
                rel = elf.relative_to(extracted_root)
                label = f"{target_path}:{rel}"
                errs, chks = inspect_elf(elf, readelf)
                if errs:
                    failed += 1
                    print(f"FAIL: {label}:", file=sys.stderr)
                    for err in errs:
                        print(f"  - {err}", file=sys.stderr)
                else:
                    passed += 1
                    chk_str = ", ".join(chks)
                    print(f"PASS: {label} (PIE, full RELRO, noexecstack, __stack_chk_fail, {chk_str})")
            return passed, failed

    errs, chks = inspect_elf(target_path, readelf)
    if errs:
        print(f"FAIL: {target_path}:", file=sys.stderr)
        for err in errs:
            print(f"  - {err}", file=sys.stderr)
        return 0, 1

    chk_str = ", ".join(chks)
    print(f"PASS: {target_path} (PIE, full RELRO, noexecstack, __stack_chk_fail, {chk_str})")
    return 1, 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binaries", nargs="+", type=Path, help="ELF binaries or archives to check")
    parser.add_argument("--readelf", default="readelf", help="path to readelf executable (default: readelf)")
    args = parser.parse_args()

    total_passed = 0
    total_failed = 0
    for target in args.binaries:
        p, f = check_target(target, args.readelf)
        total_passed += p
        total_failed += f

    if total_failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
