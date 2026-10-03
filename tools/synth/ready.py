"""Mixture-facing readiness gate for grounded synthetic output."""
import json
from pathlib import Path


def assert_ready(directory):
    """Return the manifest, or reject output whose local leakage check is incomplete."""
    path = Path(directory) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError("synthesis manifest not found: %s" % path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("leakage", {}).get("checked") is not True:
        raise ValueError("synthesis output is not ready: leakage.checked must be true")
    return manifest
