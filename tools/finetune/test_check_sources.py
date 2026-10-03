"""Offline tests for licence policy, row filtering, and the v9 resolver."""

import copy
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest

from tools.finetune import build_extra, build_mixture, build_v9, check_sources, data_licenses
from tools.finetune.source_policy import (input_hashes, tasksource_families,
                                          validate_mixture_dev)
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
        resolver.resolve("HelpSteer3/future-subset")
    with pytest.raises(ValueError, match="excluded"):
        resolver.resolve("v6/allenai/prosocial-dialog/default")


def test_v5_matcher_uses_pinned_exact_source_list():
    families = tasksource_families(AUDIT)
    assert build_mixture.audited("babi_nli/basic-coreference", AUDIT, POLICY)
    assert build_mixture.audited("commonsense_qa_2.0", AUDIT, POLICY)
    assert build_mixture.audited("spartqa-yn", AUDIT, POLICY)
    assert not build_mixture.audited("babi_nli/new-subset", AUDIT, POLICY)
    disabled = copy.deepcopy(AUDIT)
    disabled["keep_families"]["babi_nli/"]["use"] = False
    assert not build_mixture.audited("babi_nli/basic-coreference", disabled, POLICY)
    with pytest.raises(ValueError, match="required"):
        build_mixture.audited("FOL-nli", None, POLICY)
    assert families["babi_nli/"]["sources"]


def test_build_mixture_cli_requires_audit_and_policy(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["build_mixture.py", "--out", str(tmp_path / "x.gz"),
                                      "--revision", "abc"])
    with pytest.raises(SystemExit):
        build_mixture.main()


def test_model_output_families_are_prefix_excluded_and_gpt2_is_allowed():
    excluded = {x["id"]: x for x in POLICY["exclusions"] if x["registry"] == "v5"}
    for family in ("HelpSteer", "prm800k_dpo/", "hh-rlhf/"):
        assert excluded[family]["reason_class"] == "model-output"
        assert AUDIT["keep_families"][family]["use"] is False
    assert not build_mixture.audited("HelpSteer3/future-subset", AUDIT, POLICY)
    assert check_sources.generator_error(
        [{"model": "openai-community/gpt2", "role": "generator"}], POLICY) is None


@pytest.mark.parametrize("model", [
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-Large-Instruct-2407",
    "mistralai/Mistral-Small-Instruct-2409",
    "mistralai/Codestral-22B-v0.1",
    "Qwen/Qwen3-Max",
])
def test_restricted_generators_are_denied(model):
    assert check_sources.generator_error([{"model": model, "role": "writer"}], POLICY).startswith("denied")


def test_generator_allowlist_is_exact():
    allowed = ["Qwen/Qwen3-8B", "Qwen/Qwen3-32B", "mistralai/Mistral-7B-Instruct-v0.3",
               "mistralai/Mixtral-8x7B-Instruct-v0.1", "deepseek-ai/DeepSeek-R1",
               "deepseek-ai/DeepSeek-V3", "microsoft/phi-4", "microsoft/Phi-4-mini-instruct",
               "openai/gpt-oss-20b", "openai/gpt-oss-120b", "Google Cloud Translation API"]
    assert all(check_sources.generator_error([{"model": x, "role": "writer"}], POLICY) is None
               for x in allowed)
    assert "unknown" in check_sources.generator_error(
        [{"model": "Qwen/Qwen3-8B-extra", "role": "writer"}], POLICY)


def test_combined_config_cannot_bypass_exclusion(tmp_path):
    v6 = copy.deepcopy(V6)
    rows = [e for e in v6 if e["id"] == "brighter-dataset/BRIGHTER-emotion-categories"]
    rows[0]["config"] = "hin|mar"
    rows[0]["use"] = True
    v6.remove(rows[1])
    errors = run(tmp_path, v6=v6)
    assert any("excluded source is admitted" in e for e in errors)


def test_invalid_policy_enums_and_unexplained_disabled_family_fail(tmp_path):
    policy = copy.deepcopy(POLICY)
    policy["exclusions"][0]["reason_class"] = "made-up"
    audit = copy.deepcopy(AUDIT)
    audit["keep_families"]["FOL-nli"]["use"] = False
    errors = run(tmp_path, policy=policy, audit=audit)
    assert any("invalid reason_class" in e for e in errors)
    assert any("use:false family has no policy exclusion" in e for e in errors)


def test_manifest_requires_sources_rows_policy_and_matching_output(tmp_path):
    manifest = tmp_path / "mixture-v9.manifest.json"
    manifest.write_text(json.dumps({"version": 9}), encoding="utf-8")
    assert any("lacks sources" in e for e in check_sources.manifest_errors(
        manifest, check_sources.records(V6, AUDIT), POLICY))
    manifest.write_text(json.dumps({"version": 9, "sources": [], "rows_scoped_exclusions": []}),
                        encoding="utf-8")
    assert any("rows-scoped" in e for e in check_sources.manifest_errors(
        manifest, check_sources.records(V6, AUDIT), POLICY))
    assert "docs/reproductions" in str(check_sources.DEFAULT_MANIFEST)


def test_evidence_future_date_and_unlisted_spdx_fail(tmp_path):
    v6 = copy.deepcopy(V6)
    target = admitted(v6)
    target["licence_evidence"] = ["https://"]
    target["checked"] = "2999-01-01"
    target["licence_spdx"] = "CC-BY-99"
    errors = run(tmp_path, v6=v6)
    assert any("https URLs" in e for e in errors)
    assert any("future" in e for e in errors)
    assert any("disallowed licence_spdx" in e for e in errors)
    target["licence_evidence"] = ["not a URL"]
    assert any("https URLs" in e for e in run(tmp_path, v6=v6))


def test_direct_builder_consults_policy(tmp_path):
    audit = copy.deepcopy(AUDIT)
    policy = copy.deepcopy(POLICY)
    policy["exclusions"].append({"registry": "extra", "id": "PolyAI/minds14", "scope": "source",
                                  "reason_class": "unverified", "reason": "fixture", "audit": "2026-10-03"})
    audit_path, policy_path = tmp_path / "audit.json", tmp_path / "policy.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    assert "PolyAI/minds14" not in build_extra.admitted_direct_sources(audit_path, policy_path)


def test_v6_missing_pin_is_recorded_as_todo_and_loader_refuses():
    admitted_rows = [e for e in V6 if e.get("use") is True]
    assert all("pinned_commit" in e for e in admitted_rows)
    todo = next(e for e in admitted_rows if e["pinned_commit"] == "TODO")
    from tools.finetune.mixture_v6 import loaders
    with pytest.raises(RuntimeError, match="pinned_commit"):
        loaders.load_rows(todo, 1)
    assert "tools/finetune/mixture_v6/loaders.py" in build_v9.INPUT_RELATIVE_PATHS


def test_registry_row_filter_is_loaded_from_policy(monkeypatch, tmp_path):
    policy = copy.deepcopy(POLICY)
    target = next(x for x in policy["exclusions"] if x["scope"] == "rows")
    target["row_filter"] = {"lang": ["fa"]}
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    monkeypatch.setattr(registry, "POLICY_PATH", path)
    entry = next(e for e in V6 if e["id"] == "Fumika/Wikinews-multilingual")
    rows = [{"title": "خبر", "text": "هذا خبر باللغة العربية", "lang": "ar",
             "categories": ["Politics and conflicts"]},
            {"title": "Final", "text": "The cup final was played today.", "lang": "en",
             "categories": ["Sports"]}]
    assert {item["lang"] for item in registry.adapt(entry, rows, 7)} == {"ar", "en"}


def test_stale_part_hashes_force_rebuild(monkeypatch, tmp_path):
    expected = input_hashes(build_v9.ROOT)
    parts = {name: tmp_path / f"{name}.jsonl.gz" for name in ("v6", "v5", "extra")}
    manifests = {name: tmp_path / f"{name}.manifest.json" for name in parts}
    valid = {
        "v6": {"seed": build_v9.SEED, "per_source_cap": 6200, "dev_per_source": 200},
        "v5": {"seed": build_v9.SEED, "per_source_cap": 120,
               "audit": build_v9.AUDIT.name, "revision": build_v9.V5_REVISION},
        "extra": {"seed": build_v9.SEED, "per_dataset_cap": 3000},
    }
    for name in parts:
        parts[name].write_bytes(name.encode())
        manifests[name].write_text(json.dumps({**valid[name], "inputs": expected}), encoding="utf-8")
    manifests["v5"].write_text(json.dumps({**valid["v5"], "inputs": {"stale": "hash"}}), encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        name = command[0]
        calls.append(name)
        parts[name].write_bytes((name + "-rebuilt").encode())
        manifests[name].write_text(json.dumps({**valid[name], "inputs": expected}), encoding="utf-8")

    monkeypatch.setattr(build_v9, "PARTS", parts)
    monkeypatch.setattr(build_v9, "PART_MANIFESTS", manifests)
    monkeypatch.setattr(build_v9.subprocess, "run", fake_run)
    build_v9.ensure_parts({name: [name] for name in parts})
    assert calls == ["v5"]
    assert not list(tmp_path.glob("*.v9-stale"))


def _write_gzip(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def _tiny_build(monkeypatch, tmp_path, bad_src=None, wikinews_lang="en"):
    policy = {"exclusions": [{"registry": "v5", "id": "blocked", "scope": "source",
                               "reason_class": "unverified", "reason": "fixture", "audit": "2026-10-03"},
                              {"registry": "v6", "id": "wiki", "scope": "rows",
                               "row_filter": {"lang": ["ar", "fa"]},
                               "reason_class": "breaks-policy", "reason": "fixture", "audit": "2026-10-03"}]}
    complete = {"licence_spdx": "MIT", "generator": "human"}
    v6 = [{"id": "wiki", "config": "default", "use": True, "pinned_commit": "abc",
           "category": "8-topic", **complete}]
    audit = {"source_names": {"family/": ["family/sub"]},
             "keep_families": {"family/": {"use": True, **complete}},
             "direct_sources": {"extra/id": {"use": True, "revision": "def", **complete}}}
    for name, value in (("policy.json", policy), ("v6.json", v6), ("audit.json", audit)):
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    parts = {name: tmp_path / f"{name}.jsonl.gz" for name in ("v6", "v5", "extra")}
    rows = {"v6": [{"src": "v6/wiki/default", "lang": wikinews_lang, "state": "v6"}],
            "v5": [{"src": bad_src or "family/sub", "state": "v5"}],
            "extra": [{"src": "extra/id", "state": "extra"}]}
    manifests = {"v6": {"dev_items": 1}, "v5": {}, "extra": {}}
    part_manifests = {}
    for name in parts:
        _write_gzip(parts[name], rows[name])
        part_manifests[name] = tmp_path / f"{name}.manifest.json"
        part_manifests[name].write_text("{}", encoding="utf-8")
    monkeypatch.setattr(build_v9, "POLICY", tmp_path / "policy.json")
    monkeypatch.setattr(build_v9, "V6_REGISTRY", tmp_path / "v6.json")
    monkeypatch.setattr(build_v9, "AUDIT", tmp_path / "audit.json")
    monkeypatch.setattr(build_v9, "PARTS", parts)
    monkeypatch.setattr(build_v9, "PART_MANIFESTS", part_manifests)
    monkeypatch.setattr(build_v9, "OUTPUT", tmp_path / "mixture-v9.jsonl.gz")
    monkeypatch.setattr(build_v9, "MANIFEST", tmp_path / "mixture-v9.manifest.json")
    monkeypatch.setattr(build_v9, "preflight", lambda: "head")
    monkeypatch.setattr(build_v9, "fixed_commands", lambda: {})
    monkeypatch.setattr(build_v9, "ensure_parts", lambda commands: manifests)
    return manifests


def test_build_v9_end_to_end_tiny_parts(monkeypatch, tmp_path):
    _tiny_build(monkeypatch, tmp_path, wikinews_lang="ar")
    build_v9.main()
    manifest = build_v9.read_manifest_output(tmp_path / "mixture-v9.manifest.json")
    assert manifest["rows"] == 2
    assert manifest["dev_prefix_rows"] == 0
    assert manifest["excluded_rows"] == {"v6:wiki:default": 1}
    assert (tmp_path / "mixture-v9.jsonl.gz").is_file()


def test_resolver_accepts_every_explicitly_admitted_v5_name():
    resolver = build_v9.Resolver()
    for family, record in tasksource_families(AUDIT).items():
        if record.get("use") is True:
            for source in record["sources"]:
                assert resolver.resolve(source)[:2] == ("v5", family)


def test_build_v9_refuses_excluded_row_without_output(monkeypatch, tmp_path):
    manifests = _tiny_build(monkeypatch, tmp_path, bad_src="blocked/sub")
    with pytest.raises(ValueError, match="excluded"):
        build_v9.compose("head", manifests)
    assert not (tmp_path / "mixture-v9.jsonl.gz").exists()
    assert not (tmp_path / "mixture-v9.manifest.json").exists()


def test_build_v9_refuses_unpinned_v6_row(monkeypatch, tmp_path):
    manifests = _tiny_build(monkeypatch, tmp_path)
    v6_path = tmp_path / "v6.json"
    v6 = json.loads(v6_path.read_text(encoding="utf-8"))
    v6[0]["pinned_commit"] = "TODO"
    v6_path.write_text(json.dumps(v6), encoding="utf-8")
    with pytest.raises(ValueError, match="unpinned"):
        build_v9.compose("head", manifests)


def test_output_reader_refuses_mismatched_pair(monkeypatch, tmp_path):
    monkeypatch.setattr(build_v9, "ROOT", tmp_path)
    output = tmp_path / "mix.gz"
    output.write_bytes(b"new")
    manifest = tmp_path / "mix.manifest.json"
    manifest.write_text(json.dumps({"output": "mix.gz", "output_sha256": hashlib.sha256(b"old").hexdigest()}),
                        encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        build_v9.read_manifest_output(manifest)


def test_train_multitask_enforces_v9_dev_prefix(tmp_path):
    mixture = tmp_path / "mixture-v9.jsonl.gz"
    mixture.write_bytes(b"")
    (tmp_path / "mixture-v9.manifest.json").write_text(
        json.dumps({"version": 9, "dev_prefix_rows": 23}), encoding="utf-8")
    validate_mixture_dev(mixture, 23)
    with pytest.raises(ValueError, match="dev_prefix_rows"):
        validate_mixture_dev(mixture, 22)


def test_data_licence_generator_keeps_policy_warning():
    assert "weights listed under [Licence findings]" in data_licenses.POLICY


def test_changelog_describes_todo_enforcement_without_overclaim():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    unreleased = changelog.split("## [Unreleased]", 1)[1].split("\n## [", 1)[0]
    assert "no generator evidence was filled in by this change" in unreleased
    assert "builders fail closed" not in unreleased
