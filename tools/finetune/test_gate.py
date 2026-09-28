"""Offline tests for the promotion gate's decision rule (no models, no server).

    .venv-train/bin/python -m pytest -q tools/finetune/test_gate.py
"""
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
import gate  # noqa: E402


def _write(dirpath, validation, heldout):
    os.makedirs(dirpath, exist_ok=True)
    with open(os.path.join(dirpath, "eval.json"), "w") as f:
        json.dump({"validation": validation, "heldout": heldout}, f)


def _suites(acc, n=150, count=10, prefix="categories:x"):
    return {f"{prefix}/{i}": {"acc": acc, "n": n} for i in range(count)}


def test_holm_ignores_one_borderline_drop_among_many_suites():
    tests = [("s%d" % i, 0.7, 0.7, 0.0, 0.05) for i in range(87)]
    tests.append(("belebele/de", 0.34, 0.2133, -0.1267, 0.0512))  # z = -2.5, nominally "significant"
    assert gate.holm_regressions(tests) == []


def test_holm_keeps_a_real_regression():
    tests = [("s%d" % i, 0.7, 0.7, 0.0, 0.05) for i in range(87)]
    tests.append(("collapsed", 0.8, 0.5, -0.30, 0.05))  # z = -6
    assert [t[0] for t in gate.holm_regressions(tests)] == ["collapsed"]


def test_null_comparisons_rarely_reject_with_holm_but_mostly_with_plain_2se():
    """Two equally good models on 88 suites of 150 items: how often does a per-suite rule call a drop?"""
    rng = random.Random(20260928)
    runs, plain, holm = 200, 0, 0
    for _ in range(runs):
        tests = []
        for s in range(88):
            p = 0.4 + 0.5 * (s % 10) / 10
            x = sum(rng.random() < p for _ in range(150)) / 150
            y = sum(rng.random() < p for _ in range(150)) / 150
            se = math.sqrt(x * (1 - x) / 150 + y * (1 - y) / 150) or 1e-9
            tests.append(("s%d" % s, x, y, y - x, se))
        plain += any(t[3] < -2 * t[4] for t in tests)
        holm += bool(gate.holm_regressions(tests))
    assert plain / runs > 0.6          # the old rule rejects most equally good models
    assert holm / runs <= 0.10         # Holm keeps the family-wise error near 5 %


def test_compare_promotes_small_validation_loss_with_a_family_gain(tmp_path, capsys):
    val = {"a": 0.80, "b": 0.60}
    _write(tmp_path / "champ", val, _suites(0.60))
    _write(tmp_path / "chall", {"a": 0.795, "b": 0.595}, _suites(0.78))  # -0.5 points, categories +18
    assert gate.compare(str(tmp_path / "champ"), str(tmp_path / "chall")) is True
    assert "VERDICT: PROMOTE" in capsys.readouterr().out


def test_compare_rejects_validation_loss_beyond_one_point(tmp_path, capsys):
    _write(tmp_path / "champ", {"a": 0.80}, _suites(0.60))
    _write(tmp_path / "chall", {"a": 0.785}, _suites(0.78))
    assert gate.compare(str(tmp_path / "champ"), str(tmp_path / "chall")) is False
    assert "validation fell" in capsys.readouterr().out


def test_compare_rejects_a_family_regression(tmp_path, capsys):
    held_a = {**_suites(0.60), **_suites(0.70, prefix="belebele")}
    held_b = {**_suites(0.78), **_suites(0.60, prefix="belebele")}  # zero-shot family drops 10 points
    _write(tmp_path / "champ", {"a": 0.80}, held_a)
    _write(tmp_path / "chall", {"a": 0.80}, held_b)
    assert gate.compare(str(tmp_path / "champ"), str(tmp_path / "chall")) is False
    assert "held-out regression" in capsys.readouterr().out


def test_compare_rejects_when_nothing_improves(tmp_path, capsys):
    _write(tmp_path / "champ", {"a": 0.80}, _suites(0.60))
    _write(tmp_path / "chall", {"a": 0.81}, _suites(0.60))
    assert gate.compare(str(tmp_path / "champ"), str(tmp_path / "chall")) is False
    assert "no family improved" in capsys.readouterr().out


def test_pools_differ_realized_hash_only_when_both_have_it():
    old = {"acc": 0.5, "n": 150, "pool": "p1"}
    new = {"acc": 0.5, "n": 150, "pool": "p1", "pool_items_sha256": "aa"}
    assert not gate.pools_differ(old, new)          # an eval.json from before the realized hash
    assert not gate.pools_differ(new, dict(new))
    assert gate.pools_differ(new, dict(new, pool_items_sha256="bb"))
    assert gate.pools_differ(old, dict(old, pool="p2"))
