#!/usr/bin/env python3
"""Offline, stdlib-only licence-policy validation for every training registry."""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_POLICY = HERE / "sources" / "policy.json"
DEFAULT_V6 = HERE / "sources" / "v6-keep.json"
DEFAULT_AUDIT = HERE / "licence_audit.json"
DEFAULT_MANIFEST = ROOT / "data" / "mixture-v9.manifest.json"
REQUIRED = ("licence_spdx", "licence_evidence", "provenance", "text_licence_spdx",
            "generator", "checked")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def records(v6, audit):
    out = []
    for entry in v6:
        out.append(("v6", entry["id"], entry.get("config", "default"), entry,
                    entry.get("use") is True))
    for name, entry in audit.get("keep_families", {}).items():
        out.append(("v5", name, None, entry, entry.get("use", True) is True))
    for name, entry in audit.get("direct_sources", {}).items():
        out.append(("extra", name, None, entry, entry.get("use") is True))
    return out


def licence_allowed(value, allowed, text=False):
    if isinstance(value, str) and value in allowed:
        return True
    return isinstance(value, str) and bool(re.fullmatch(r"CC-BY-\d+(?:\.\d+)*", value))


def generator_error(value, policy):
    if value in ("human", "template"):
        return None
    if not isinstance(value, list) or not value:
        return "generator must be human, template, or a non-empty model/role list"
    for item in value:
        if not isinstance(item, dict) or set(item) != {"model", "role"} or not all(
                isinstance(item[k], str) and item[k].strip() for k in ("model", "role")):
            return "generator list entries must contain non-empty model and role"
        model = item["model"]
        if any(re.search(rule["model_pattern"], model, re.I) for rule in policy["denied_generators"]):
            return "denied generator: " + model
        if not any(re.search(rule["model_pattern"], model, re.I) and rule.get("allows_training") is True
                   for rule in policy["generators"]):
            return "unknown generator: " + model
    return None


def manifest_errors(path, all_records):
    path = Path(path)
    if not path.exists():
        return []
    manifest = load(path)
    if manifest.get("version") != 9:
        return [f"{path}: manifest version is not 9"]
    admitted = {(reg, sid, config) for reg, sid, config, _, use in all_records if use}
    errors = []
    for source in manifest.get("sources", []):
        key = (source.get("registry"), source.get("id"), source.get("config"))
        if key not in admitted and (key[0], key[1], None) not in admitted:
            errors.append(f"{path}: non-admitted manifest source {key}")
    return errors


def check(policy_path=DEFAULT_POLICY, v6_path=DEFAULT_V6, audit_path=DEFAULT_AUDIT,
          manifest_path=DEFAULT_MANIFEST, allow_todo=False):
    policy, v6, audit = load(policy_path), load(v6_path), load(audit_path)
    all_records = records(v6, audit)
    errors, todos = [], Counter()
    data_allowed = set(policy["licences"]["data"])
    text_allowed = set(policy["licences"]["text"])

    for registry, sid, config, record, admitted in all_records:
        if not admitted:
            continue
        label = f"{registry}:{sid}" + (f"::{config}" if config else "")
        for field in REQUIRED:
            if field not in record:
                errors.append(f"{label}: missing {field}")
                continue
            if record[field] == "TODO":
                todos[registry] += 1
                if not allow_todo:
                    errors.append(f"{label}: TODO {field}")
        if record.get("licence_spdx") != "TODO" and not licence_allowed(
                record.get("licence_spdx"), data_allowed):
            errors.append(f"{label}: disallowed licence_spdx {record.get('licence_spdx')!r}")
        if record.get("text_licence_spdx") != "TODO" and not licence_allowed(
                record.get("text_licence_spdx"), text_allowed, text=True):
            errors.append(f"{label}: disallowed text_licence_spdx {record.get('text_licence_spdx')!r}")
        evidence = record.get("licence_evidence")
        if evidence != "TODO" and (not isinstance(evidence, list) or not evidence or
                                   any(not isinstance(url, str) or not url.startswith("https://") for url in evidence)):
            errors.append(f"{label}: licence_evidence must be a non-empty list of https URLs")
        provenance = record.get("provenance")
        if provenance != "TODO" and (not isinstance(provenance, str) or not provenance.strip()):
            errors.append(f"{label}: provenance must be non-empty text")
        generator = record.get("generator")
        if generator != "TODO":
            problem = generator_error(generator, policy)
            if problem:
                errors.append(f"{label}: {problem}")
        checked = record.get("checked")
        if checked != "TODO":
            try:
                datetime.date.fromisoformat(checked)
            except (TypeError, ValueError):
                errors.append(f"{label}: checked is not an ISO date")

    for exclusion in policy["exclusions"]:
        matches = [(reg, sid, config, rec, use) for reg, sid, config, rec, use in all_records
                   if reg == exclusion["registry"] and sid == exclusion["id"] and
                   ("config" not in exclusion or config == exclusion["config"])]
        label = f"{exclusion['registry']}:{exclusion['id']}"
        if len(matches) != 1:
            errors.append(f"{label}: exclusion resolves to {len(matches)} records, expected exactly one")
        elif exclusion["scope"] == "source" and matches[0][4]:
            errors.append(f"{label}: excluded source is admitted")
        if exclusion["scope"] == "rows" and not exclusion.get("row_filter"):
            errors.append(f"{label}: rows-scoped exclusion has no row_filter")

    errors.extend(manifest_errors(manifest_path, all_records))
    return errors, todos


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-todo", action="store_true")
    ap.add_argument("--policy", default=DEFAULT_POLICY)
    ap.add_argument("--v6", default=DEFAULT_V6)
    ap.add_argument("--audit", default=DEFAULT_AUDIT)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = ap.parse_args(argv)
    errors, todos = check(args.policy, args.v6, args.audit, args.manifest, args.allow_todo)
    for registry in ("v5", "v6", "extra"):
        print(f"TODO fields [{registry}]: {todos[registry]}")
    if errors:
        for error in errors:
            print("ERROR: " + error, file=sys.stderr)
        return 1
    print("source policy check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
