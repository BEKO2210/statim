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
from urllib.parse import urlsplit

try:
    from .source_policy import (config_tokens, exclusion_matches, tasksource_families,
                                tasksource_family_matches, file_sha256)
except ImportError:  # direct script execution
    from source_policy import (config_tokens, exclusion_matches, tasksource_families,
                               tasksource_family_matches, file_sha256)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_POLICY = HERE / "sources" / "policy.json"
DEFAULT_V6 = HERE / "sources" / "v6-keep.json"
DEFAULT_AUDIT = HERE / "licence_audit.json"
DEFAULT_MANIFEST = ROOT / "docs" / "reproductions" / "mixture-v9.manifest.json"
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
    return isinstance(value, str) and value in allowed


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


def manifest_errors(path, all_records, policy):
    path = Path(path)
    if not path.exists():
        return []
    manifest = load(path)
    if manifest.get("version") != 9:
        return [f"{path}: manifest version is not 9"]
    errors = []
    if not isinstance(manifest.get("sources"), list):
        return [f"{path}: manifest lacks sources"]
    admitted = {(reg, sid, config) for reg, sid, config, _, use in all_records if use}
    for source in manifest["sources"]:
        key = (source.get("registry"), source.get("id"), source.get("config"))
        if key not in admitted and (key[0], key[1], None) not in admitted:
            errors.append(f"{path}: non-admitted manifest source {key}")
    expected_rows = [{k: x[k] for k in ("registry", "id", "scope", "row_filter")}
                     for x in policy["exclusions"] if x.get("scope") == "rows"]
    if manifest.get("rows_scoped_exclusions") != expected_rows:
        errors.append(f"{path}: rows-scoped exclusions do not match policy.json")
    output = manifest.get("output")
    if isinstance(output, str):
        output_path = ROOT / output
        if output_path.exists() and manifest.get("output_sha256") != file_sha256(output_path):
            errors.append(f"{path}: output sha256 does not match manifest")
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
                                   any(not isinstance(url, str) or
                                       urlsplit(url).scheme != "https" or not urlsplit(url).netloc
                                       for url in evidence)):
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
                date = datetime.date.fromisoformat(checked)
                if date > datetime.date.today():
                    errors.append(f"{label}: checked is in the future")
            except (TypeError, ValueError):
                errors.append(f"{label}: checked is not an ISO date")

        if registry == "v6":
            pin = record.get("pinned_commit")
            if pin is None:
                errors.append(f"{label}: missing pinned_commit")
            elif pin == "TODO":
                todos[registry] += 1
                if not allow_todo:
                    errors.append(f"{label}: TODO pinned_commit")

    registries = {"v5", "v6", "extra"}
    scopes = {"source", "rows"}
    reasons = {"breaks-policy", "unverified", "model-output"}
    families = tasksource_families(audit)
    source_names = audit.get("source_names", {})
    if set(source_names) != set(audit.get("keep_families", {})):
        errors.append("v5 source_names must list every keep_families key exactly once")
    owners = Counter(name for family, names in source_names.items()
                     if families.get(family, {}).get("use") is True
                     for name in names
                     if tasksource_family_matches(name, family, families[family]))
    for family, names in source_names.items():
        for name in names:
            if not tasksource_family_matches(name, family, families[family]):
                errors.append(f"v5:{family}: pinned source is outside family: {name}")
    for name, count in owners.items():
        if count != 1:
            errors.append(f"v5:{name}: admitted by {count} families, expected exactly one")

    for exclusion in policy["exclusions"]:
        if exclusion.get("registry") not in registries:
            errors.append(f"exclusion has invalid registry {exclusion.get('registry')!r}")
        if exclusion.get("scope") not in scopes:
            errors.append(f"exclusion has invalid scope {exclusion.get('scope')!r}")
        if exclusion.get("reason_class") not in reasons:
            errors.append(f"exclusion has invalid reason_class {exclusion.get('reason_class')!r}")
        if exclusion.get("scope") == "rows" and exclusion.get("registry") != "v6":
            errors.append("rows-scoped exclusions are only valid for v6")
        matches = [(reg, sid, config, rec, use) for reg, sid, config, rec, use in all_records
                   if reg == exclusion.get("registry") and sid == exclusion.get("id") and
                   ("config" not in exclusion or exclusion["config"] in config_tokens(config))]
        label = f"{exclusion['registry']}:{exclusion['id']}"
        if len(matches) != 1:
            errors.append(f"{label}: exclusion resolves to {len(matches)} records, expected exactly one")
        if exclusion.get("scope") == "source":
            covered = [(reg, sid, use) for reg, sid, config, _, use in all_records
                       if exclusion_matches(reg, sid, config, exclusion)]
            if any(use for _, _, use in covered):
                errors.append(f"{label}: excluded source is admitted")
        if exclusion["scope"] == "rows" and not exclusion.get("row_filter"):
            errors.append(f"{label}: rows-scoped exclusion has no row_filter")

    for family, record in families.items():
        if record.get("use") is False and not any(
                x.get("scope") == "source" and exclusion_matches("v5", family, None, x)
                for x in policy["exclusions"]):
            errors.append(f"v5:{family}: use:false family has no policy exclusion")
        if record.get("use") is True:
            for name in record.get("sources") or []:
                if any(x.get("scope") == "source" and exclusion_matches("v5", name, None, x)
                       for x in policy["exclusions"]):
                    errors.append(f"v5:{family}: admitted family covers excluded source {name}")

    errors.extend(manifest_errors(manifest_path, all_records, policy))
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
