"""Offline tests for licence policy, row filtering, and the v9 resolver."""

import copy
import json
from pathlib import Path

import pytest

from tools.finetune import build_v9, check_sources
from tools.finetune.mixture_v6 import registry


ROOT = Path(__file__).resolve().parents[2]
POLICY = json.loads((ROOT / "tools/finetune/sources/policy.json").read_text())
V6 = json.loads((ROOT / "tools/finetune/sources/v6-keep.json").read_text())
AUDIT = json.loads((ROOT / "tools/finetune/licence_audit.json").read_text())


def run(tmp_path, *, policy=None, v6=None, audit=None, allow_todo=True):
    paths = []
    for name, value in (("policy", policy or POLICY), ("v6", v6 or V6), ("audit", audit or AUDIT)):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(value), encoding="utf-8")
        paths.append(path)
    return check_sources.check(*paths, manifest_path=tmp_path / "absent.json", allow_todo=allow_todo)[0]


def admitted(v6):
    return next(entry for entry in v6 if entry.get("use") is True)


def test_real_repo_passes_with_todos_allowed():
    errors, todos = check_sources.check(allow_todo=True)
    assert errors == []
    assert sum(todos.values()) > 0


def test_missing_required_field_fails(tmp_path):
    v6 = copy.deepcopy(V6)
    admitted(v6).pop("provenance")
    assert any("missing provenance" in error for error in run(tmp_path, v6=v6))


def test_noncommercial_licence_fails(tmp_path):
    v6 = copy.deepcopy(V6)
    admitted(v6)["licence_spdx"] = "CC-BY-NC-4.0"
    assert any("disallowed licence_spdx" in error for error in run(tmp_path, v6=v6))


def test_denied_generator_fails(tmp_path):
    v6 = copy.deepcopy(V6)
    admitted(v6)["generator"] = [{"model": "ChatGPT-4", "role": "writer"}]
    assert any("denied generator" in error for error in run(tmp_path, v6=v6))


def test_unknown_exclusion_id_fails(tmp_path):
    policy = copy.deepcopy(POLICY)
    policy["exclusions"].append({"registry": "v6", "id": "missing/source", "scope": "source",
                                  "reason_class": "unverified", "reason": "fixture", "audit": "2026-10-03"})
    assert any("resolves to 0 records" in error for error in run(tmp_path, policy=policy))


def test_excluded_id_reenabled_fails(tmp_path):
    v6 = copy.deepcopy(V6)
    target = next(entry for entry in v6 if entry["id"] == "ankitkupadhyay/XNLI")
    target["use"] = True
    assert any("excluded source is admitted" in error for error in run(tmp_path, v6=v6))


def test_rows_exclusion_without_filter_fails(tmp_path):
    policy = copy.deepcopy(POLICY)
    target = next(x for x in policy["exclusions"] if x["scope"] == "rows")
    target.pop("row_filter")
    assert any("has no row_filter" in error for error in run(tmp_path, policy=policy))


def test_todo_fails_by_default(tmp_path):
    assert any("TODO" in error for error in run(tmp_path, allow_todo=False))


def test_wikinews_adapter_drops_arabic_and_persian_rows():
    entry = next(e for e in V6 if e["id"] == "Fumika/Wikinews-multilingual")
    rows = [
        {"title": "خبر", "text": "هذا خبر باللغة العربية", "lang": "ar", "categories": ["Politics and conflicts"]},
        {"title": "خبر", "text": "این یک خبر فارسی است", "lang": "fa", "categories": ["Politics and conflicts"]},
        {"title": "Vote", "text": "The parliament held a vote today.", "lang": "en", "categories": ["Politics and conflicts"]},
        {"title": "Final", "text": "The cup final was played today.", "lang": "en", "categories": ["Sports"]},
    ]
    items = list(registry.adapt(entry, rows, 7))
    assert items
    assert {item["lang"] for item in items} == {"en"}


def test_v9_resolver_refuses_excluded_src():
    resolver = build_v9.Resolver()
    with pytest.raises(ValueError, match="excluded"):
        resolver.resolve("WANLI")
    with pytest.raises(ValueError, match="excluded"):
        resolver.resolve("v6/allenai/prosocial-dialog/default")
