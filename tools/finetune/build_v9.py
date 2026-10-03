#!/usr/bin/env python3
"""Compose the policy-checked v9 mixture from pinned, reproducible parts.

The strict policy check and clean-registry check run before any part or output
is touched. Existing parts are verified; absent parts are built with the fixed
arguments below. Do not run this script merely to test it: use the unit tests.
"""

from __future__ import annotations

import collections
import gzip
import hashlib
import io
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from .source_policy import (INPUT_RELATIVE_PATHS, exclusion_matches, input_hashes,
                                row_is_excluded, tasksource_families, matching_tasksource_families)
except ImportError:  # direct script execution
    from source_policy import (INPUT_RELATIVE_PATHS, exclusion_matches, input_hashes,
                               row_is_excluded, tasksource_families, matching_tasksource_families)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data"
POLICY = HERE / "sources" / "policy.json"
V6_REGISTRY = HERE / "sources" / "v6-keep.json"
AUDIT = HERE / "licence_audit.json"
SEED = 20260927
# Immutable snapshot already present in the local cache and selected as v9's
# fixed v5 input. build_mixture.py itself still requires --revision because the
# legacy manifest did not record one.
V5_REVISION = "071f0cf2201f06b2dea3e55323aa3e264284e139"
PARTS = {
    "v6": DATA / "mixture-v6.jsonl.gz",
    "v5": DATA / "mixture-v5.jsonl.gz",
    "extra": DATA / "extra-v1.jsonl.gz",
}
PART_MANIFESTS = {
    "v6": DATA / "mixture-v6.manifest.json",
    "v5": DATA / "mixture-v5.manifest.json",
    "extra": DATA / "extra-v1.jsonl.gz.manifest.json",
}
OUTPUT = DATA / "mixture-v9.jsonl.gz"
MANIFEST = ROOT / "docs" / "reproductions" / "mixture-v9.manifest.json"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path):
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(Path(path))


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, text=True,
                          stdout=subprocess.PIPE).stdout.strip()


def preflight():
    subprocess.run([sys.executable, str(HERE / "check_sources.py")], cwd=ROOT, check=True)
    watched = list(INPUT_RELATIVE_PATHS)
    dirty = git("status", "--porcelain", "--", *watched)
    if dirty:
        raise RuntimeError("policy/registry files are dirty; commit them before building v9:\n" + dirty)
    return git("rev-parse", "HEAD")


def fixed_commands():
    return {
        "v6": [sys.executable, str(HERE / "mixture_v6" / "build.py"), "--out", str(PARTS["v6"]),
               "--per-source", "6000", "--dev-per-source", "200", "--seed", str(SEED)],
        "v5": [sys.executable, str(HERE / "build_mixture.py"), "--out", str(PARTS["v5"]),
               "--per-source", "120", "--audit", str(AUDIT), "--policy", str(POLICY),
               "--revision", V5_REVISION],
        "extra": [sys.executable, str(HERE / "build_extra.py"), "--out", str(PARTS["extra"]),
                  "--per-dataset", "3000"],
    }


def ensure_parts(commands):
    expected_inputs = input_hashes(ROOT)
    for name in ("v6", "v5", "extra"):
        part, manifest = PARTS[name], PART_MANIFESTS[name]
        if part.exists() != manifest.exists():
            raise RuntimeError(f"{name}: part and manifest must either both exist or both be absent")
        reusable = False
        if part.exists():
            try:
                reusable = json.loads(manifest.read_text(encoding="utf-8")).get("inputs") == expected_inputs
            except (OSError, ValueError):
                reusable = False
        if not reusable:
            backups = []
            targets = (part, manifest, Path(str(part) + ".provenance.jsonl.gz"))
            for old in targets:
                if old.exists():
                    backup = old.with_name(old.name + ".v9-stale")
                    if backup.exists():
                        raise RuntimeError(f"refusing to replace existing stale-part backup: {backup}")
                    os.replace(old, backup)
                    backups.append((old, backup))
            try:
                subprocess.run(commands[name], cwd=ROOT, check=True)
            except BaseException:
                for target in targets:
                    target.unlink(missing_ok=True)
                for old, backup in reversed(backups):
                    os.replace(backup, old)
                raise
            else:
                for _, backup in backups:
                    backup.unlink()
        if not part.is_file() or not manifest.is_file():
            raise RuntimeError(f"{name}: builder did not create its part and manifest")
    v6 = json.loads(PART_MANIFESTS["v6"].read_text(encoding="utf-8"))
    v5 = json.loads(PART_MANIFESTS["v5"].read_text(encoding="utf-8"))
    extra = json.loads(PART_MANIFESTS["extra"].read_text(encoding="utf-8"))
    for name, manifest in (("v6", v6), ("v5", v5), ("extra", extra)):
        if manifest.get("inputs") != expected_inputs:
            raise RuntimeError(f"{name} part manifest input hashes do not match the current builders and policy")
    if (v6.get("seed"), v6.get("per_source_cap"), v6.get("dev_per_source")) != (SEED, 6200, 200):
        raise RuntimeError("v6 part manifest does not match fixed v9 arguments")
    if (v5.get("seed"), v5.get("per_source_cap"), v5.get("audit"), v5.get("revision")) != \
            (SEED, 120, AUDIT.name, V5_REVISION):
        raise RuntimeError("v5 part manifest does not match fixed v9 arguments")
    if (extra.get("seed"), extra.get("per_dataset_cap")) != (SEED, 3000):
        raise RuntimeError("extra part manifest does not match fixed v9 arguments")
    return {"v6": v6, "v5": v5, "extra": extra}


class Resolver:
    def __init__(self, policy_path=None, v6_path=None, audit_path=None):
        policy_path = POLICY if policy_path is None else policy_path
        v6_path = V6_REGISTRY if v6_path is None else v6_path
        audit_path = AUDIT if audit_path is None else audit_path
        self.policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
        self.v6 = json.loads(Path(v6_path).read_text(encoding="utf-8"))
        self.audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
        self.v6_names = {f"v6/{e['id']}/{e.get('config') or 'default'}": e for e in self.v6
                         if e.get("use") is True}
        self.v5 = tasksource_families(self.audit)
        self.extra = {name: rec for name, rec in self.audit["direct_sources"].items()
                      if rec.get("use") is True}
        self.excluded = [x for x in self.policy["exclusions"] if x["scope"] == "source"]

    def resolve(self, src):
        if not isinstance(src, str) or not src:
            raise ValueError("row has no src")
        if src.startswith("v6/"):
            entry = self.v6_names.get(src)
            if entry is None:
                raise ValueError(f"unresolved or excluded src: {src}")
            return "v6", entry["id"], entry.get("config", "default"), entry
        for exclusion in self.excluded:
            if exclusion.get("registry") in {"v5", "extra"} and exclusion_matches(
                    exclusion["registry"], src, None, exclusion):
                raise ValueError(f"excluded src: {src}")
        extra = [name for name in self.extra if src == name or src.startswith(name + "/")]
        if len(extra) == 1:
            name = extra[0]
            return "extra", name, None, self.extra[name]
        families = matching_tasksource_families(src, self.v5)
        if len(families) == 1:
            name = families[0]
            return "v5", name, None, self.v5[name]
        if len(families) > 1:
            raise ValueError(f"ambiguous admitted src: {src}")
        raise ValueError(f"unresolved or excluded src: {src}")


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield line, json.loads(line)


def write_reproducible(path, lines):
    with open(path, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="\n") as out:
                for line in lines:
                    out.write(line.rstrip("\n") + "\n")


def compose(head, manifests, out=None, manifest_path=None):
    out = OUTPUT if out is None else out
    manifest_path = MANIFEST if manifest_path is None else manifest_path
    resolver = Resolver()
    dev_count = manifests["v6"].get("dev_items")
    if not isinstance(dev_count, int) or dev_count < 0:
        raise RuntimeError("v6 manifest has no valid dev_items prefix length")
    prefix, rest, counts, categories = [], [], collections.Counter(), collections.Counter()
    rows_excluded = collections.Counter()
    source_meta = {}
    for part_name in ("v6", "v5", "extra"):
        for index, (line, row) in enumerate(read_rows(PARTS[part_name])):
            registry, sid, config, record = resolver.resolve(row.get("src"))
            if row_is_excluded(registry, sid, config, row, resolver.policy):
                rows_excluded[f"{registry}:{sid}:{config or 'default'}"] += 1
                continue
            if registry == "v6" and record.get("pinned_commit") in (None, "TODO"):
                raise ValueError(f"unpinned v6 source: {row.get('src')}")
            key = (registry, sid, config)
            counts[key] += 1
            category = record.get("category") or ("9-intent-dialogue-act" if registry == "extra" else "v5-broad")
            categories[category] += 1
            source_meta[key] = record
            if part_name == "v6" and index < dev_count:
                prefix.append(line)
            else:
                rest.append(line)
    random.Random(SEED).shuffle(rest)
    out, manifest_path = Path(out), Path(manifest_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    output_fd, output_name = tempfile.mkstemp(prefix=out.name + ".", suffix=".tmp", dir=out.parent)
    os.close(output_fd)
    tmp = Path(output_name)
    try:
        write_reproducible(tmp, prefix + rest)
        sources = []
        for key in sorted(counts, key=lambda x: tuple("" if v is None else v for v in x)):
            registry, sid, config = key
            record = source_meta[key]
            if registry == "v5":
                revision = V5_REVISION
            elif registry == "extra":
                revision = record["revision"]
            else:
                revision = record["pinned_commit"]
            sources.append({"registry": registry, "id": sid, "config": config,
                            "rows": counts[key], "licence_spdx": record["licence_spdx"],
                            "generator": record["generator"], "revision": revision})
        result = {
            "version": 9, "git_head": head, "seed": SEED, "dev_prefix_rows": len(prefix),
            "inputs": input_hashes(ROOT),
            "parts": {name: {"path": relative(PARTS[name]), "sha256": sha256(PARTS[name]),
                              "manifest": relative(PART_MANIFESTS[name]),
                              "manifest_sha256": sha256(PART_MANIFESTS[name])} for name in PARTS},
            "sources": sources, "categories": dict(sorted(categories.items())),
            "rows_scoped_exclusions": [
                {k: x[k] for k in ("registry", "id", "scope", "row_filter")}
                for x in resolver.policy["exclusions"] if x.get("scope") == "rows"],
            "excluded_rows": dict(sorted(rows_excluded.items())),
            "rows": len(prefix) + len(rest), "output": relative(out), "output_sha256": sha256(tmp),
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_fd, manifest_name = tempfile.mkstemp(prefix=manifest_path.name + ".", suffix=".tmp",
                                                       dir=manifest_path.parent)
        os.close(manifest_fd)
        manifest_tmp = Path(manifest_name)
        try:
            manifest_tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            os.replace(tmp, out)
            os.replace(manifest_tmp, manifest_path)
        finally:
            manifest_tmp.unlink(missing_ok=True)
    finally:
        tmp.unlink(missing_ok=True)
    return result


def read_manifest_output(manifest_path=MANIFEST):
    """Read a completed pair, refusing a missing or mismatched output."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = ROOT / manifest["output"]
    if not output.is_file() or sha256(output) != manifest.get("output_sha256"):
        raise ValueError("v9 output sha256 does not match its manifest")
    return manifest


def main():
    head = preflight()
    commands = fixed_commands()
    manifests = ensure_parts(commands)
    result = compose(head, manifests)
    print(json.dumps({"output": result["output"], "rows": result["rows"],
                      "sha256": result["output_sha256"]}))


if __name__ == "__main__":
    main()
