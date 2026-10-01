"""Tests for tools/finetune/train_lora.py and tools/finetune/lora_experiment.py.

    python -m pytest -q tools/finetune/test_train_lora.py

The first part needs only the standard library (CI's offline job installs pytest and pyarrow). The last
test trains and converts a real adapter; it runs only when STATIM_LORA_BASE_DIR (a Laya checkpoint
directory) and STATIM_LORA_BASE_GGUF (that checkpoint's Statim GGUF) are set. Training uses
STATIM_LORA_TRAIN_PYTHON (torch, peft and laya) and conversion uses STATIM_LORA_TOOLS_PYTHON (gguf),
defaulting to the sibling statim checkout's two venvs.
"""
import gzip
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "bench"))

import lora_experiment  # noqa: E402
import train_lora  # noqa: E402

REGISTRY = HERE / "sources" / "v6-keep.json"


def _entries():
    return [e for e in json.loads(REGISTRY.read_text(encoding="utf-8")) if e.get("use") is True]


def test_converter_categories_match():
    import ast
    # convert_lora imports numpy and gguf, which the offline job lacks: read its CATEGORIES from the source
    tree = ast.parse((ROOT / "tools" / "convert_lora.py").read_text(encoding="utf-8"))
    value = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                 and any(getattr(t, "id", None) == "CATEGORIES" for t in n.targets))
    assert tuple(ast.literal_eval(value)) == train_lora.CATEGORIES


def test_categories_match_the_held_out_suites():
    import eval_categories
    from mixture_v6 import registry
    assert train_lora.CATEGORIES == tuple(eval_categories.HELD_OUT)
    assert set(train_lora.CATEGORY_TESTS) == set(train_lora.CATEGORIES)
    for entry in _entries():
        task = registry.task_for(entry)
        if task in train_lora.CATEGORY_TESTS:
            assert train_lora.CATEGORY_TESTS[task](entry.get("category", "")), (task, entry["id"])


def test_category_sources_use_exact_src_names():
    chosen, enabled = train_lora.category_sources(REGISTRY, "fact_check")
    assert "v6/zenodo:3609356 (ClaimBuster)/crowdsourced/groundtruth" in chosen  # config with "/"
    assert all(src in enabled for src in chosen)
    by_src = {train_lora.source_name(e): e["category"] for e in _entries()}
    assert chosen == {s for s, c in by_src.items() if "10-fact-check" in c or "10-claim" in c}
    sentiment, _ = train_lora.category_sources(REGISTRY, "sentiment")
    complaint, _ = train_lora.category_sources(REGISTRY, "complaint")
    assert sentiment & complaint  # "3-complaint; 1-sentiment" serves both suites
    pii, _ = train_lora.category_sources(REGISTRY, "pii")
    assert pii and not pii & chosen  # 10-pii and 10-fact-check are separate families


def _row(src, state, gold=0):
    return {"state": state, "q": {"type": "choice", "instructions": "Which emotion?",
                                  "criteria": {"joy": "joy", "anger": "anger"}},
            "target": [1.0, 0.0] if gold == 0 else [0.0, 1.0], "src": src, "lang": "en"}


def test_select_rows_streams_one_category(tmp_path):
    chosen, enabled = train_lora.category_sources(REGISTRY, "emotion")
    emo = sorted(chosen)[0]
    other = next(s for s in enabled if s not in chosen)
    path = tmp_path / "mix.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i, src in enumerate([emo, other, "v6/unknown/x", emo, "synth-v1/emotion/en"]):
            f.write(json.dumps(_row(src, "text %d" % i)) + "\n")
    rows, stats = train_lora.select_rows(path, chosen, enabled)
    assert [r["state"] for r in rows] == ["text 0", "text 3"]
    assert stats["per_source"] == {emo: 2} and stats["other_category_rows"] == 1
    assert stats["unknown_src_rows"] == 2
    rows, stats = train_lora.select_rows(path, chosen, enabled, limit=1)
    assert len(rows) == 1 and stats["limited"]


def test_dev_split_is_deterministic_and_disjoint_by_state():
    rows = [_row("v6/a/default", "state %d" % (i % 60)) for i in range(120)]  # every state twice
    rows += [_row("v6/b/default", "other %d" % i) for i in range(40)]
    train, dev = train_lora.split_dev(rows, 30)
    assert set(train).isdisjoint(dev) and sorted(train + dev) == list(range(len(rows)))
    dev_states = {rows[i]["state"] for i in dev}
    assert not dev_states & {rows[i]["state"] for i in train}
    assert {rows[i]["src"] for i in dev} == {"v6/a/default", "v6/b/default"}  # quota per source
    assert (train, dev) == train_lora.split_dev(list(rows), 30)
    shuffled = rows[::-1]
    assert {shuffled[i]["state"] for i in train_lora.split_dev(shuffled, 30)[1]} == dev_states
    assert train_lora.split_dev(rows[:4], 100)[1].__len__() <= 2  # at most half goes to dev


def _records(path, accs, pool="p1"):
    item_path = str(path).replace(".jsonl", "-items.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for lang, acc in accs.items():
            actual = round(round(acc * 150) / 150, 4)
            f.write(json.dumps({"family": "categories", "suite": "emotion", "lang": lang, "model": "m",
                                "n": 150, "accuracy": actual, "pool": pool}) + "\n")
    with open(item_path, "w", encoding="utf-8") as f:
        for lang, acc in accs.items():
            right = round(acc * 150)
            for i in range(150):
                f.write(json.dumps({"suite": "emotion", "lang": lang, "i": i,
                                    "item": "%064x" % i, "gold": 0,
                                    "pred": 0 if i < right else 1}) + "\n")
    return path


def test_decision_promotes_a_clear_gain_and_blocks_a_regression(tmp_path):
    langs = ["de", "en", "es", "fr", "hi", "pt", "ru", "zh"]
    base = _records(tmp_path / "base.jsonl", {l: 0.55 for l in langs})
    better = _records(tmp_path / "better.jsonl", {l: 0.70 for l in langs})
    res = lora_experiment.decide(base, better, log=lambda *a: None)
    assert res["promote"] and len(res["cells"]) == 8 and not res["harms"]
    assert all(c["verdict"] == "gain (Holm)" for c in res["cells"])
    mixed = _records(tmp_path / "mixed.jsonl", {**{l: 0.72 for l in langs}, "de": 0.30})
    res = lora_experiment.decide(base, mixed, log=lambda *a: None)
    assert not res["promote"] and [h.rsplit("/", 1)[1] for h in res["harms"]] == ["de"]
    same = _records(tmp_path / "same.jsonl", {l: 0.55 for l in langs})
    assert not lora_experiment.decide(base, same, log=lambda *a: None)["promote"]


def test_decision_compares_only_the_same_pool(tmp_path):
    base = _records(tmp_path / "base.jsonl", {"de": 0.5, "en": 0.5})
    other = _records(tmp_path / "other.jsonl", {"de": 0.9, "en": 0.9}, pool="p2")
    res = lora_experiment.decide(base, other, log=lambda *a: None)
    assert not res["promote"] and not res["cells"]


def test_decision_refuses_different_realized_items(tmp_path):
    base = _records(tmp_path / "base.jsonl", {"en": 0.5})
    adapter = _records(tmp_path / "adapter.jsonl", {"en": 0.9})
    rows = [json.loads(line) for line in Path(base).read_text().splitlines()]
    rows[0]["pool_items_sha256"] = "base-items"
    Path(base).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    rows = [json.loads(line) for line in Path(adapter).read_text().splitlines()]
    rows[0]["pool_items_sha256"] = "adapter-items"
    Path(adapter).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(ValueError, match="different realized pool items"):
        lora_experiment.decide(base, adapter, log=lambda *a: None)


def test_skip_train_recovers_and_verifies_mixture(tmp_path):
    mixture = tmp_path / "mix.jsonl"
    mixture.write_text('{}\n', encoding="utf-8")
    adapter = tmp_path / "work" / "emotion"
    adapter.mkdir(parents=True)
    (adapter / "train_lora.json").write_text(json.dumps({
        "mixture": str(mixture), "mixture_sha256": train_lora.sha256_file(mixture),
    }), encoding="utf-8")
    args = lora_experiment.parse_args([
        "--base-checkpoint", "base", "--base-gguf", "base.gguf", "--skip-train", "--work", str(tmp_path / "work"),
        "--categories", "emotion", "--train-python", "train-python", "--tools-python", "tools-python",
        "--eval-skip", "40",
    ])
    plan = lora_experiment.plan(args, "emotion")
    assert plan["exclude_mixture"] == str(mixture)
    assert plan["train"][0] == "train-python" and plan["convert"][0] == plan["eval_base"][0] == "tools-python"
    assert "--strict" in plan["eval_base"] and "--strict" in plan["eval_adapter"]
    for command in (plan["eval_base"], plan["eval_adapter"]):
        assert command[command.index("--skip") + 1] == "40"
        assert "--predictions" in command
    mixture.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="SHA-256 differs"):
        lora_experiment.plan(args, "emotion")
    mixture.unlink()
    with pytest.raises(SystemExit, match="recorded mixture is missing"):
        lora_experiment.plan(args, "emotion")


def test_explicit_no_exclusion_warns_and_needs_no_training_record(tmp_path, capsys):
    args = lora_experiment.parse_args([
        "--base-checkpoint", "base", "--base-gguf", "base.gguf", "--skip-train", "--work", str(tmp_path),
        "--categories", "emotion", "--exclude-mixture", "none",
    ])
    assert lora_experiment.plan(args, "emotion")["exclude_mixture"] is None
    assert "not held out" in capsys.readouterr().err


def test_python_alias_sets_both_interpreters():
    args = lora_experiment.parse_args([
        "--base-checkpoint", "base", "--base-gguf", "base.gguf", "--mixture", "mix", "--python", "old-python",
    ])
    assert args.train_python == args.tools_python == "old-python"


def test_default_interpreters_use_repository_venvs(monkeypatch, tmp_path):
    train = tmp_path / ".venv-train" / "bin" / "python"
    tools = tmp_path / ".venv" / "bin" / "python"
    train.parent.mkdir(parents=True)
    tools.parent.mkdir(parents=True)
    train.touch()
    tools.touch()
    monkeypatch.setattr(lora_experiment, "ROOT", str(tmp_path))
    assert lora_experiment.default_python(".venv-train") == str(train)
    assert lora_experiment.default_python(".venv") == str(tools)


def test_tied_dev_keeps_the_initial_adapter(monkeypatch):
    torch = pytest.importorskip("torch")

    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([0.0]))

        def forward(self, *unused):
            return self.weight.reshape(1, 1), self.weight

    def collate(*unused):
        return tuple(torch.zeros((1, 1)) for _ in range(6))

    monkeypatch.setitem(sys.modules, "train_banking77", types.SimpleNamespace(
        batches=lambda items, *unused: [[0]], collate=collate))
    monkeypatch.setattr(train_lora, "predict", lambda *unused: [[1.0, 0.0]])
    monkeypatch.setattr(train_lora, "rlcd_loss", lambda logits, *unused: (logits.sum(), logits.detach().abs().mean()))
    pm = Adapter()
    args = types.SimpleNamespace(seed=1, lr=0.1, weight_decay=0.0, max_tokens=8, max_rows=1, accum=1,
                                 epochs=1, max_steps=1, warmup=0.0, eval_every=0)
    item = {"target": [1.0, 0.0], "markers": [0, 1], "lang": "en"}
    result = train_lora.train(pm, types.SimpleNamespace(head=None), [item], [item], 0, torch.device("cpu"),
                              None, args, log=lambda *a, **k: None)
    assert result["best"] == {"dev_acc": 1.0, "epoch": 0, "update": 0, "saved_initial": True}
    assert pm.weight.item() == 0.0


def test_summary_calls_an_initial_adapter_no_gain():
    category = {"status": "ok", "promote": False, "reason": "(saved adapter is the initial zero-delta adapter: no gain)",
                "family": {}, "cells": [], "reported": {}, "skipped": [], "convert": "ok", "commands": {},
                "train": {"items": {}, "updates": 1, "dev_before": {"dev_acc": 0.5},
                          "best": {"dev_acc": 0.5, "epoch": 0, "update": 0, "saved_initial": True}}}
    summary = {"base_checkpoint": "base", "base_gguf": "base.gguf", "mixture": "mix", "exclude_mixture": "mix",
               "n": 2, "eval_skip": 1, "seed": 1, "z": 2.0, "alpha": 0.05, "created": "now",
               "categories": {"emotion": category}}
    text = lora_experiment.summary_md(summary)
    assert "sample: n=2 draw per language cell, skip 1, seed 1" in text
    assert "initial zero-delta adapter; this category has no gain" in text


# --------------------------------------------------------------------------- opt-in: real base model

BASE_DIR, BASE_GGUF = os.environ.get("STATIM_LORA_BASE_DIR"), os.environ.get("STATIM_LORA_BASE_GGUF")


@pytest.mark.skipif(not (BASE_DIR and BASE_GGUF), reason="set STATIM_LORA_BASE_DIR and STATIM_LORA_BASE_GGUF")
def test_real_base_trains_and_converts(tmp_path):
    train_python = os.environ.get("STATIM_LORA_TRAIN_PYTHON", str(ROOT.parent / "statim" / ".venv-train" / "bin" / "python"))
    tools_python = os.environ.get("STATIM_LORA_TOOLS_PYTHON", str(ROOT.parent / "statim" / ".venv" / "bin" / "python"))
    if not all(Path(p).is_file() for p in (train_python, tools_python)):
        pytest.skip("set STATIM_LORA_TRAIN_PYTHON and STATIM_LORA_TOOLS_PYTHON")
    chosen, _ = train_lora.category_sources(REGISTRY, "emotion")
    src = sorted(chosen)[0]
    path = tmp_path / "mix.jsonl.gz"
    texts = ["I finally passed the exam, what a day!", "They cancelled my train again.",
             "We got the keys to our new flat today.", "Nobody answers the hotline, this is outrageous."]
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i in range(24):
            f.write(json.dumps(_row(src, "%s (%d)" % (texts[i % 4], i), gold=i % 2)) + "\n")
    out = tmp_path / "emotion"
    subprocess.run([train_python, str(HERE / "train_lora.py"), BASE_DIR, "--mixture", str(path),
                    "--category", "emotion", "--out", str(out), "--device", "cpu", "--dev-items", "4",
                    "--max-steps", "2", "--max-tokens", "1024", "--accum", "1"], check=True)
    cfg = json.loads((out / "adapter_config.json").read_text())
    assert cfg["bias"] == "none" and not cfg.get("modules_to_save") and not cfg.get("use_dora")
    assert (out / "adapter_model.safetensors").exists()
    res = subprocess.run([tools_python, str(ROOT / "tools" / "convert_lora.py"), str(out), "-o",
                          str(tmp_path / "emotion.gguf"), "--base", BASE_GGUF, "--category", "emotion"],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "88 LoRA pairs" in res.stdout
