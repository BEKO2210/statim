#!/usr/bin/env python3
"""Verify vendored headers against their pinned versions and SHA-256 digests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from vendored_cves import DatabaseError, get_pinned_versions


HEADER_VERSION_KEYS = {
    "httplib.h": "httplib",
    "json.hpp": "json",
}


def check_vendored(repo_root: Path) -> list[str]:
    """Return validation errors for the vendored-header manifest and files."""
    manifest_path = repo_root / "third_party" / "VENDORED.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read {manifest_path}: {exc}"]

    errors: list[str] = []
    if not isinstance(manifest, dict):
        return [f"{manifest_path} must contain a JSON object"]

    try:
        pinned_versions = get_pinned_versions(repo_root)
    except DatabaseError as exc:
        return [str(exc)]

    for filename, version_key in HEADER_VERSION_KEYS.items():
        entry = manifest.get(filename)
        if not isinstance(entry, dict):
            errors.append(f"missing manifest entry for {filename}")
            continue

        expected_hash = entry.get("sha256")
        expected_version = entry.get("version")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            errors.append(f"invalid SHA-256 in manifest for {filename}")
        else:
            path = repo_root / "third_party" / filename
            try:
                actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                errors.append(f"cannot read {path}: {exc}")
            else:
                if actual_hash != expected_hash:
                    errors.append(
                        f"SHA-256 mismatch for {filename}: expected {expected_hash}, got {actual_hash}"
                    )

        actual_version = pinned_versions[version_key]
        if actual_version != expected_version:
            errors.append(
                f"version mismatch for {filename}: manifest has {expected_version}, "
                f"vendored file reports {actual_version}"
            )

    return errors


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    errors = check_vendored(repo_root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Vendored header hashes and versions match third_party/VENDORED.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
