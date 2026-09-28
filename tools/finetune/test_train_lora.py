"""Tests for tools/finetune/train_lora.py and tools/finetune/lora_experiment.py.

    python -m pytest -q tools/finetune/test_train_lora.py

The first part needs only the standard library (CI's offline job installs pytest and pyarrow). The last
test trains and converts a real adapter; it runs only when STATIM_LORA_BASE_DIR (a Laya checkpoint
directory) and STATIM_LORA_BASE_GGUF (that checkpoint's Statim GGUF) are set, and needs torch, peft and laya.
"""
import gzip
import json
import os
import subprocess
import sys
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


def test_categories_match_the_held_out_suites():
    import eval_categories
    assert train_lora.CATEGORIES == tuple(eval_categories.HELD_OUT)
    assert set(train_lora.CATEGORY_TESTS) == set(train_lora.CATEGORIES)


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
    with open(path, "w", encoding="utf-8") as f:
        for lang, acc in accs.items():
            f.write(json.dumps({"family": "categories", "suite": "emotion", "lang": lang, "model": "m",
                                "n": 150, "accuracy": acc, "pool": pool}) + "\n")
    return path


def test_decision_promotes_a_clear_gain_and_blocks_a_regression(tmp_path):
    langs = ["de", "en", "es", "fr", "hi", "pt", "ru", "zh"]
    base = _records(tmp_path / "base.jsonl", {l: 0.55 for l in langs})
    better = _records(tmp_path / "better.jsonl", {l: 0.70 for l in langs})
    res = lora_experiment.decide(base, better, log=lambda *a: None)
    assert res["promote"] and len(res["cells"]) == 8 and not res["harms"]
    assert all(c["verdict"] == "gain (2 SE)" for c in res["cells"])
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


# --------------------------------------------------------------------------- opt-in: real base model

BASE_DIR, BASE_GGUF = os.environ.get("STATIM_LORA_BASE_DIR"), os.environ.get("STATIM_LORA_BASE_GGUF")


@pytest.mark.skipif(not (BASE_DIR and BASE_GGUF), reason="set STATIM_LORA_BASE_DIR and STATIM_LORA_BASE_GGUF")
def test_real_base_trains_and_converts(tmp_path):
    for mod in ("torch", "peft", "laya", "gguf"):
        pytest.importorskip(mod)
    chosen, _ = train_lora.category_sources(REGISTRY, "emotion")
    src = sorted(chosen)[0]
    path = tmp_path / "mix.jsonl.gz"
    texts = ["I finally passed the exam, what a day!", "They cancelled my train again.",
             "We got the keys to our new flat today.", "Nobody answers the hotline, this is outrageous."]
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i in range(24):
            f.write(json.dumps(_row(src, "%s (%d)" % (texts[i % 4], i), gold=i % 2)) + "\n")
    out = tmp_path / "emotion"
    subprocess.run([sys.executable, str(HERE / "train_lora.py"), BASE_DIR, "--mixture", str(path),
                    "--category", "emotion", "--out", str(out), "--device", "cpu", "--dev-items", "4",
                    "--max-steps", "2", "--max-tokens", "1024", "--accum", "1"], check=True)
    cfg = json.loads((out / "adapter_config.json").read_text())
    assert cfg["bias"] == "none" and not cfg.get("modules_to_save") and not cfg.get("use_dora")
    assert (out / "adapter_model.safetensors").exists()
    res = subprocess.run([sys.executable, str(ROOT / "tools" / "convert_lora.py"), str(out), "-o",
                          str(tmp_path / "emotion.gguf"), "--base", BASE_GGUF, "--category", "emotion"],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "88 LoRA pairs" in res.stdout
