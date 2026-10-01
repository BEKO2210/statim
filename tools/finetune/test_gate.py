#!/usr/bin/env python3
"""Offline tests for the paired promotion gate (no models or server)."""
import ast
import gzip
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import gate  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _load_tool(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(suite, a_right, b_right, n=100, hashes=None):
    hashes = hashes or ["%064x" % i for i in range(n)]
    a, b = [], []
    for i in range(n):
        common = {"suite": suite, "lang": "en", "i": i, "item": hashes[i], "gold": 0}
        a.append(dict(common, pred=0 if i in a_right else 1))
        b.append(dict(common, pred=0 if i in b_right else 1))
    return a, b


def _write_pair(root, cells, validation=None):
    """cells maps key -> (champ-right indices, challenger-right indices, n)."""
    validation = validation or {"v": (set(range(80)), set(range(80)), 100)}
    dirs = [root / "champ", root / "chall"]
    all_rows = [[], []]
    heldout = [{}, {}]
    vals = [{}, {}]
    for key, (ar, br, n) in cells.items():
        pair = _rows(key, ar, br, n)
        for side in range(2):
            all_rows[side] += pair[side]
            heldout[side][key] = {"acc": sum(r["pred"] == r["gold"] for r in pair[side]) / n, "n": n}
    for name, (ar, br, n) in validation.items():
        pair = _rows("validation:" + name, ar, br, n)
        for side in range(2):
            all_rows[side] += pair[side]
            vals[side][name] = {"acc": sum(r["pred"] == r["gold"] for r in pair[side]) / n, "n": n}
    for side, directory in enumerate(dirs):
        directory.mkdir(parents=True)
        item_path = directory / gate.ITEMS_NAME
        with gzip.open(item_path, "wt", encoding="utf-8") as f:
            for row in all_rows[side]:
                f.write(json.dumps(row) + "\n")
        with open(directory / "eval.json", "w", encoding="utf-8") as f:
            json.dump({"validation": vals[side], "heldout": heldout[side],
                       "eval_items_sha256": gate.sha256_file(item_path)}, f)
    return dirs


@pytest.mark.parametrize("b,c,expected", [
    (0, 0, 1.0), (0, 5, 1.0), (5, 0, 1 / 32), (1, 0, 0.5),
    (4, 1, 6 / 32),
    (80, 20, sum(math.comb(100, k) for k in range(21)) / 2 ** 100),
])
def test_mcnemar_drop_exact_hand_computed(b, c, expected):
    arows, brows = _rows("x", set(range(b)), set(range(b, b + c)), b + c or 1)
    assert gate.paired_cell("x", arows, brows)["p_drop"] == pytest.approx(expected)


def test_holm_on_89_null_cells_controls_gain_familywise_error():
    rng = random.Random(20261001)
    runs, rejected = 240, 0
    for _ in range(runs):
        tests = []
        for s in range(89):
            # Under the null, discordant directions are exchangeable within every paired cell.
            b = sum(rng.random() < 0.5 for _ in range(24))
            c = 24 - b
            tests.append({"name": f"s{s}", "d": (c - b) / 150,
                          "p_gain": gate.binomial_upper_half(24, c),
                          "p_drop": gate.binomial_upper_half(24, b)})
        rejected += bool(gate.holm_gains(tests))
    assert rejected / runs <= gate.ALPHA


def test_correlated_regression_is_missed_unpaired_but_gate_rejects(tmp_path, capsys):
    # 90% -> 82%, with only eight discordances. Unpaired z=1.64; exact paired p=1/256.
    ar, br = set(range(90)), set(range(82))
    unpaired = math.sqrt(.9 * .1 / 100 + .82 * .18 / 100)
    assert .08 < 2 * unpaired
    champ, chall = _write_pair(tmp_path, {"categories:x/en": (ar, br, 100)})
    assert gate.compare(str(champ), str(chall)) is False
    out = capsys.readouterr().out
    assert "b=8, c=0" in out and "held-out regression" in out


def test_publishing_tools_report_paired_only_drop_as_loss(tmp_path):
    # The old independent test misses this 90% -> 82% drop; paired McNemar is significant.
    champ, chall = _write_pair(tmp_path, {"categories:x/en": (set(range(90)), set(range(82)), 100)})
    tools = [
        _load_tool("test_hf_publish", "tools/release/hf_publish.py"),
        _load_tool("test_gate_grid", "tools/site/gate_grid.py"),
        _load_tool("test_gate_chart", "tools/diagrams/gate_chart.py"),
    ]
    for tool in tools:
        result = tool.gate_summary(str(champ), str(chall))
        assert result["counts"] == {"gain": 0, "noise": 0, "loss": 1}
        assert result["cells"]["categories:x/en"]["verdict"] == "loss"
        assert result["decision"]["families"]["categories"]["rows"]["flag"] == "REGRESSION"


def test_publishing_tools_fail_closed_without_items_with_compare_message(tmp_path):
    champ, chall = _write_pair(tmp_path, {"categories:x/en": (set(range(90)), set(range(82)), 100)})
    (chall / gate.ITEMS_NAME).unlink()
    tools = [
        _load_tool("missing_hf_publish", "tools/release/hf_publish.py"),
        _load_tool("missing_gate_grid", "tools/site/gate_grid.py"),
        _load_tool("missing_gate_chart", "tools/diagrams/gate_chart.py"),
    ]
    expected = re.escape(f"{chall / gate.ITEMS_NAME}: missing per-item outcomes; "
                         f"rerun tools/finetune/gate.py eval {chall}")
    for tool in tools:
        with pytest.raises(SystemExit, match=expected):
            tool.gate_summary(str(champ), str(chall))


def test_no_unpaired_two_proportion_standard_error_outside_allowlist():
    patterns = [
        re.compile(
            r'''(?P<x>\w+)\s*\[\s*["']acc["']\s*\]\s*\*\s*\(\s*1\s*-\s*(?P=x)\s*\[\s*["']acc["']\s*\]\s*\)\s*/\s*(?P=x)\s*\[\s*["']n["']\s*\]\s*\+\s*(?P<y>\w+)\s*\[\s*["']acc["']\s*\]\s*\*\s*\(\s*1\s*-\s*(?P=y)\s*\[\s*["']acc["']\s*\]\s*\)\s*/\s*(?P=y)\s*\[\s*["']n["']\s*\]''',
            re.X | re.S,
        ),
        re.compile(
            r'''(?P<pa>\b\w+)\s*\*\s*\(\s*1\s*-\s*(?P=pa)\s*\)\s*/\s*(?P<na>\w+)\s*\+\s*(?P<pb>\w+)\s*\*\s*\(\s*1\s*-\s*(?P=pb)\s*\)\s*/\s*(?P<nb>\w+)''',
            re.X | re.S,
        ),
    ]
    # Historical report-only output, never used for a promotion or published comparison.
    allowlist = {("tools/finetune/gate.py", "_legacy_unpaired"):
                 "explicit legacy report for historical inspection; it can never promote"}
    found, violations = set(), []
    for top in (ROOT / "tools", ROOT / "bench"):
        for path in top.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            functions = [node for node in ast.walk(ast.parse(source))
                         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            for pattern in patterns:
                for match in pattern.finditer(source):
                    line = source.count("\n", 0, match.start()) + 1
                    owners = [node for node in functions
                              if node.lineno <= line <= getattr(node, "end_lineno", node.lineno)]
                    owner = min(owners, key=lambda node: node.end_lineno - node.lineno).name if owners else "<module>"
                    key = (path.relative_to(ROOT).as_posix(), owner)
                    if key in allowlist:
                        found.add(key)
                    else:
                        violations.append(f"{key[0]}:{line} ({owner})")
    assert not violations, "unpaired two-proportion SE found at " + ", ".join(violations)
    assert found == set(allowlist), "stale unpaired-SE allowlist: " + repr(set(allowlist) - found)


def test_unpaired_gain_can_be_noise_when_correctness_is_negatively_correlated():
    # Disjoint correct sets: the old independent formula crosses 2 SE, exact pairing does not.
    ar, br = set(range(40)), set(range(40, 95))
    a, b = _rows("categories:x/en", ar, br)
    t = gate.paired_cell("categories:x/en", a, b)
    unpaired = math.sqrt(.4 * .6 / 100 + .55 * .45 / 100)
    assert t["d"] > 2 * unpaired
    assert t["p_gain"] > gate.ALPHA


def test_clear_paired_family_gain_promotes(tmp_path, capsys):
    cells = {f"categories:x/{i}": (set(range(50)), set(range(70)), 100) for i in range(4)}
    champ, chall = _write_pair(tmp_path, cells)
    assert gate.compare(str(champ), str(chall)) is True
    assert "VERDICT: PROMOTE" in capsys.readouterr().out


def test_heldout_items_must_match_published_count_and_accuracy():
    arows, brows = _rows("categories:x/en", set(range(50)), set(range(60)))
    metrics = {"categories:x/en": {"acc": .5, "n": 100}}
    with pytest.raises(SystemExit, match=r"categories:x/en: challenger item accuracy 0.6000.*0.5000"):
        gate.heldout_decision(metrics, metrics, {"categories:x/en": arows},
                              {"categories:x/en": brows}, log=lambda *_: None)
    bad_n = {"categories:x/en": {"acc": .5, "n": 99}}
    with pytest.raises(SystemExit, match=r"categories:x/en: champion item count 100.*eval.json n 99"):
        gate.heldout_decision(bad_n, metrics, {"categories:x/en": arows},
                              {"categories:x/en": arows}, log=lambda *_: None)


def test_validation_items_must_match_published_accuracy():
    arows, brows = _rows("validation:v", set(range(50)), set(range(60)))
    with pytest.raises(SystemExit, match=r"validation:v: challenger item accuracy 0.6000.*0.5000"):
        gate.validation_stats({"v": {"acc": .5, "n": 100}}, {"v": {"acc": .5, "n": 100}},
                              {"validation:v": arows},
                              {"validation:v": brows}, ["v"])
    with pytest.raises(SystemExit, match=r"validation:v: champion item count 100.*eval.json n 99"):
        gate.validation_stats({"v": {"acc": .5, "n": 99}}, {"v": {"acc": .5, "n": 100}},
                              {"validation:v": arows}, {"validation:v": arows}, ["v"])


def test_family_gain_holm_controls_correlated_null_over_real_layout():
    rng = random.Random(20261001)
    langs = ["de", "en", "fr", "es", "it", "tr", "pl", "ru", "ja", "zh-CN", "ar", "hi"]
    keys = ([f"amazon_massive_intent/{lang}" for lang in langs]
            + [f"{suite}/en" for suite in sorted(gate.ZERO_SHOT)]
            + [f"multilingual_sentiments/{lang}" for lang in langs]
            + [f"categories:category_{i}/en" for i in range(14)])
    runs, any_gain, n = 300, 0, 120
    for run in range(runs):
        a, b, ai, bi = {}, {}, {}, {}
        for cell_no, key in enumerate(keys):
            p = .35 + .5 * ((cell_no % 9) / 8)
            ar, br = set(), set()
            for i in range(n):
                shared = rng.random() < p
                if rng.random() < .85:  # strongly correlated models with identical null marginals
                    ac = bc = shared
                else:
                    ac, bc = rng.random() < p, rng.random() < p
                if ac:
                    ar.add(i)
                if bc:
                    br.add(i)
            arows, brows = _rows(key, ar, br, n)
            ai[key], bi[key] = arows, brows
            a[key] = {"acc": round(len(ar) / n, 4), "n": n}
            b[key] = {"acc": round(len(br) / n, 4), "n": n}
        result = gate.heldout_decision(a, b, ai, bi, log=lambda *_: None)
        any_gain += bool(result["family_gains"])
    rate = any_gain / runs
    margin = 3 * math.sqrt(gate.ALPHA * (1 - gate.ALPHA) / runs)
    assert rate <= gate.ALPHA + margin, (any_gain, runs, rate, margin)


def test_eval_dev_refuses_one_predictions_file_for_multiple_models(tmp_path):
    script = os.path.join(os.path.dirname(__file__), "eval_dev.py")
    run = subprocess.run([sys.executable, script, "model-a", "model-b", "--predictions",
                          str(tmp_path / "items.jsonl")], capture_output=True, text=True)
    assert run.returncode == 2
    assert "--predictions requires exactly one model" in run.stderr


def test_eval_laya_refuses_one_predictions_file_for_multiple_head_lengths(tmp_path):
    script = os.path.join(os.path.dirname(__file__), "eval_laya.py")
    run = subprocess.run([sys.executable, script, "model", "--head-max-len", "192", "512",
                          "--predictions", str(tmp_path / "items.jsonl")],
                         capture_output=True, text=True)
    assert run.returncode == 2
    assert "--predictions requires exactly one --head-max-len value" in run.stderr


def test_validation_margin_stays_decisive_and_logs_paired_interval(tmp_path, capsys):
    cells = {"categories:x/en": (set(range(50)), set(range(75)), 100)}
    champ, chall = _write_pair(tmp_path, cells,
                               validation={"v": (set(range(80)), set(range(78)), 100)})
    assert gate.compare(str(champ), str(chall)) is False
    out = capsys.readouterr().out
    assert "validation fell" in out and "paired 95% CI" in out


def test_missing_items_file_refuses_old_eval(tmp_path):
    for name in ("champ", "chall"):
        d = tmp_path / name
        d.mkdir()
        (d / "eval.json").write_text(json.dumps({"validation": {"v": .5}, "heldout": {}}))
    with pytest.raises(SystemExit, match=r"old eval.json.*gate.py eval"):
        gate.compare(str(tmp_path / "champ"), str(tmp_path / "chall"))


def test_missing_hashed_items_file_refuses_comparison(tmp_path):
    champ, chall = _write_pair(tmp_path, {"categories:x/en": (set(range(50)), set(range(60)), 100)})
    (chall / gate.ITEMS_NAME).unlink()
    with pytest.raises(SystemExit, match=r"missing per-item outcomes.*gate.py eval"):
        gate.compare(str(champ), str(chall))


def test_adapter_decision_refuses_outcomes_from_old_callers():
    with pytest.raises(SystemExit, match=r"per-item outcomes.*gate.py eval"):
        gate.adapter_decision({}, {})


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows[0].update(item="f" * 64), "hashes/order differ"),
    (lambda rows: rows.reverse(), "hashes/order differ"),
])
def test_mismatched_hash_or_order_refuses_comparison(tmp_path, mutation, match):
    champ, chall = _write_pair(tmp_path, {"categories:x/en": (set(range(50)), set(range(60)), 100)})
    path = chall / gate.ITEMS_NAME
    with gzip.open(path, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    mutation(rows)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    data = json.loads((chall / "eval.json").read_text())
    data["eval_items_sha256"] = gate.sha256_file(path)
    (chall / "eval.json").write_text(json.dumps(data))
    with pytest.raises(SystemExit, match=match):
        gate.compare(str(champ), str(chall))


def test_legacy_unpaired_is_report_only_and_never_promotes(tmp_path, capsys):
    champ, chall = _write_pair(tmp_path, {"categories:x/en": (set(range(20)), set(range(90)), 100)})
    assert gate.compare(str(champ), str(chall), legacy_unpaired=True) is False
    out = capsys.readouterr().out
    assert "UNPAIRED (legacy, not a decision)" in out and "VERDICT: REJECT" in out


def test_pools_differ_realized_hash_only_when_both_have_it():
    old = {"acc": 0.5, "n": 150, "pool": "p1"}
    new = {"acc": 0.5, "n": 150, "pool": "p1", "pool_items_sha256": "aa"}
    assert not gate.pools_differ(old, new)
    assert not gate.pools_differ(new, dict(new))
    assert gate.pools_differ(new, dict(new, pool_items_sha256="bb"))
    assert gate.pools_differ(old, dict(old, pool="p2"))
