#!/usr/bin/env python3
"""No-harm promotion gate: evaluate checkpoints, then make a paired promotion decision.

    # writes <model>/eval.json and <model>/eval-items.jsonl.gz
    .venv-train/bin/python tools/finetune/gate.py eval models/laya-multilingual-clean
    .venv-train/bin/python tools/finetune/gate.py compare models/champion models/challenger

Validation uses held-out development suites. The lower bound of a paired 95% interval for the
unweighted mean of suite deltas may not fall below -VAL_MARGIN (one point); each suite's half-width
is at least the zero-event Clopper-Pearson bound. Two held-out regression tests block, and either one is
enough: per-cell one-sided exact McNemar tests Holm-corrected across cells (one collapsed language
must not hide behind gains in the others), and capability-level exact McNemar tests that pool
paired items across cells and are Holm-corrected across capabilities (a loss spread thinly over
many small cells must not hide either); a capability also blocks on a drop beyond its tolerance. Family row- and suite-weighted means use the observed per-item
differences d_i = challenger_correct - champion_correct. Their standard errors combine the sample
variance of each cell's mean with the same cell/group weights as the reported pooled mean. This
analytic paired estimator was chosen over a bootstrap because it is deterministic, fast, and
directly estimates the uncertainty of these stratified weighted means. The up-to-eight one-sided
family gain tests are Holm-corrected together; family regressions retain the cautious uncorrected
two-standard-error screen.

Why paired: both models answer the same items, so treating their accuracies as independent throws
away the usually strong positive correlation. No existing model artifact in this worktree has
per-item outcomes, so this is a synthetic illustration: on 150 items, accuracies 0.80 and 0.76 with
9 champion-right/challenger-wrong and 3 reverse discordances have unpaired SE 0.0478, but the
per-item paired SE is 0.0229 (2.1x smaller).

A challenger is promoted only when validation holds, no held-out regression remains, and at
least one paired family mean has a Holm-significant gain. Evaluation is strict by default. Every
artifact records suite/registry definition hashes and suite completion; missing or failed suites,
suite-set/item-pool/registry mismatches, and unverifiable old artifacts produce ``VERDICT: BLOCKED``.
``eval --no-strict`` artifacts and ``compare --no-strict`` are report-only and can never promote.
``compare --legacy-unpaired`` remains historical report-only. LoRA adapters use the same paired
held-out path through adapter_decision.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "bench"))
from paired_stats import binomial_upper_half  # noqa: E402

PY = os.path.join(ROOT, ".venv-train", "bin", "python")
PORT = int(os.environ.get("STATIM_GATE_PORT", "8097"))
BIN = os.environ.get("STATIM_BIN", os.path.join(ROOT, "build-vk", "statim"))
DEVICE = os.environ.get("STATIM_GATE_DEVICE", "vulkan")
CONVERT_PY = os.environ.get("STATIM_CONVERT_PY", os.path.join(ROOT, ".venv", "bin", "python"))
ITEMS_NAME = "eval-items.jsonl.gz"
ALPHA = 0.05
VAL_MARGIN = 0.01
DEFAULT_CAPABILITY_TOLERANCE = 0.02
UNDERPOWERED_N = 600
EVAL_SCRIPTS = ("tools/finetune/eval_dev.py", "tools/finetune/eval_laya.py",
                "bench/eval_multilingual.py", "bench/eval_zeroshot.py", "bench/eval_categories.py")

# One mapping owns the semantic grouping. Exact names precede prefixes; unknown suites get a stable
# family of their own, so adding a benchmark never silently drops it from capability testing.
CAPABILITY_RULES = {
    "reading": {"belebele", "categories:reading"},
    "nli": {"indonli", "farstail", "categories:nli"},
    "paraphrase/similarity": {"semrel", "paws", "paws-x", "categories:similarity"},
    "intent": {"test/banking77", "hwu64", "categories:intent"},
    "multilingual-intent": {"amazon_massive_intent"},
    "sentiment": {"multilingual_sentiments", "categories:sentiment"},
    "emotion": {"go_emotions", "test/emotion", "categories:emotion"},
    "safety": {"multi_hatecheck", "categories:safety"},
    "pii": {"categories:pii"},
    "fact_check": {"categories:fact_check"},
    "topic": {"sib200", "test/ag_news", "categories:topic"},
    "stance": {"categories:stance"},
    "formality": {"categories:formality"},
    "urgency": {"categories:urgency"},
    "complaint": {"categories:complaint"},
}


def run(cmd):
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if out.returncode:
        sys.stderr.write(out.stdout[-2000:] + out.stderr[-4000:])
        raise SystemExit(f"failed: {' '.join(cmd)}")
    return out.stdout


def jsonl(text):
    return [json.loads(line) for line in text.splitlines() if line.startswith("{")]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def definition_hash(paths):
    """Hash named definition files, including names to prevent concatenation ambiguity."""
    h = hashlib.sha256()
    for rel in sorted(paths):
        h.update(rel.encode("utf-8") + b"\0")
        with open(os.path.join(ROOT, rel), "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


def current_definitions():
    registry = "tools/finetune/mixture_v6/registry.py"
    return {
        "registry_sha256": definition_hash([registry]),
        "suite_sha256": {script: definition_hash([script]) for script in EVAL_SCRIPTS
                         if os.path.isfile(os.path.join(ROOT, script))},
    }


def capability_for(cell):
    stem = cell if cell.startswith("test/") else cell.rsplit("/", 1)[0]
    for capability, names in CAPABILITY_RULES.items():
        if stem in names or any(stem.startswith(name + "/") for name in names):
            return capability
    if stem.startswith("categories:"):
        return stem.split(":", 1)[1]
    if stem.startswith("test/"):
        return stem.split("/", 1)[1]
    return stem


def gguf_for(model_dir):
    path = model_dir.rstrip("/") + "-f32.gguf"
    if not os.path.exists(path):
        run([CONVERT_PY if os.path.exists(CONVERT_PY) else PY, "tools/convert_laya.py", model_dir,
             "-o", path, "--type", "f32", "--embd-type", "f16",
             "--name", os.path.basename(model_dir.rstrip("/"))])
    return path


def _prediction_rows(path, cell_key):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            row["suite"] = cell_key(row)
            rows.append({k: row[k] for k in ("suite", "lang", "i", "item", "gold", "pred")})
    return rows


def http_suites(model_dir, tmp, mixture=None, strict=True):
    srv = subprocess.Popen([BIN, "serve", "--device", DEVICE, "-m", f"m={gguf_for(model_dir)}",
                            "--port", str(PORT), "--no-access-log"], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/ready", timeout=2)
                break
            except Exception:
                time.sleep(1)
        rows, items, status = [], [], {}
        categories = ["--n", "150"] + (["--exclude-mixture", mixture] if mixture else [])
        for script, extra in (("bench/eval_multilingual.py", ["--n", "150"]),
                              ("bench/eval_zeroshot.py", ["--n", "150"]),
                              ("bench/eval_categories.py", categories)):
            if not os.path.exists(os.path.join(ROOT, script)):
                status[script] = {"status": "missing", "detail": "script does not exist"}
                continue
            stem = os.path.basename(script)
            out = os.path.join(tmp, stem + ".jsonl")
            pred = os.path.join(tmp, stem + ".items.jsonl")
            command = [PY, script, "--url", f"http://127.0.0.1:{PORT}", "--model", "multilingual",
                       "--out", out, "--predictions", pred, *extra]
            if stem == "eval_categories.py":
                command.append("--strict" if strict else "--no-strict")
            try:
                run(command)
            except SystemExit as exc:
                status[script] = {"status": "failed", "detail": str(exc)}
                continue
            script_rows = [json.loads(line) for line in open(out, encoding="utf-8")]
            scored = [r for r in script_rows if r.get("lang") != "macro" and "accuracy" in r]
            if not scored:
                status[script] = {"status": "failed", "detail": "suite produced no scored cells"}
                continue
            status[script] = {"status": "ok", "cells": len(scored)}
            rows += script_rows
            if stem == "eval_categories.py":
                key = lambda r: f"categories:{r['suite']}/{r['lang']}"
            else:
                key = lambda r: f"{r['suite']}/{r['lang']}"
            items += _prediction_rows(pred, key)
        return rows, items, status
    finally:
        srv.send_signal(signal.SIGINT)
        srv.wait(timeout=60)


def evaluate(model_dir, mixture=None, strict=True):
    tmp = os.path.join(model_dir, "eval-tmp")
    os.makedirs(tmp, exist_ok=True)
    res = {"model": model_dir, "validation": {}, "heldout": {}, "reported": {}, "mixture": mixture,
           "gate_schema": 2, "strict": strict, "definitions": current_definitions(), "suite_status": {}}
    dev_items = os.path.join(tmp, "eval_dev.items.jsonl")
    items = []
    try:
        dev = jsonl(run([PY, "tools/finetune/eval_dev.py", model_dir, "--predictions", dev_items]))[0]
        res["validation"] = {k: {"acc": dev[k], "n": dev["n"][k]}
                             for k in dev if k not in ("model", "n", "seconds", "mean",
                                                       "suite_definition_sha256")}
        items += _prediction_rows(dev_items, lambda r: "validation:" + r["suite"])
        res["suite_status"]["tools/finetune/eval_dev.py"] = {
            "status": "ok" if res["validation"] else "failed",
            "detail": "" if res["validation"] else "suite produced no scored cells",
            "cells": len(res["validation"])}
    except (SystemExit, IndexError, OSError, KeyError) as exc:
        res["suite_status"]["tools/finetune/eval_dev.py"] = {"status": "failed", "detail": str(exc)}

    laya_items = os.path.join(tmp, "eval_laya.items.jsonl")
    command = [PY, "tools/finetune/eval_laya.py", model_dir, "--n", "2000", "--head-max-len", "512",
               "--predictions", laya_items]
    try:
        for r in jsonl(run(command)):
            res["heldout"][f"test/{r['suite']}"] = {"acc": r["accuracy"], "n": r.get("n", 2000),
                                                      "ece": r.get("ece")}
        items += _prediction_rows(laya_items, lambda r: "test/" + r["suite"])
        laya_cells = sum(k.startswith("test/") for k in res["heldout"])
        res["suite_status"]["tools/finetune/eval_laya.py"] = {
            "status": "ok" if laya_cells else "failed",
            "detail": "" if laya_cells else "suite produced no scored cells", "cells": laya_cells}
    except (SystemExit, OSError, KeyError) as exc:
        res["suite_status"]["tools/finetune/eval_laya.py"] = {"status": "failed", "detail": str(exc)}

    http_records, http_items, http_status = http_suites(model_dir, tmp, mixture, strict)
    res["suite_status"].update(http_status)
    res["skipped"] = skipped_cells(http_records)
    heldout, reported = heldout_cells(http_records)
    res["heldout"].update(heldout)
    res["reported"].update(reported)
    items += http_items

    items_path = os.path.join(model_dir, ITEMS_NAME)
    with gzip.open(items_path, "wt", encoding="utf-8") as f:
        for row in items:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    res["eval_items_sha256"] = sha256_file(items_path)
    with open(os.path.join(model_dir, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    validation_mean = (round(sum(v["acc"] for v in res["validation"].values()) /
                             len(res["validation"]), 4) if res["validation"] else None)
    print(json.dumps({"validation_mean": validation_mean,
                      "heldout_suites": len(res["heldout"]),
                      "eval_items_sha256": res["eval_items_sha256"],
                      "suite_status": res["suite_status"]}))


def skipped_cells(records):
    """Category cells the pool cannot fill (too few items, one class dominates). The set is a
    function of the pool and the seed, so champion and challenger must skip the same cells; it is
    recorded so that missing coverage is visible instead of silently absent."""
    return {f"categories:{r['suite']}/{r['lang']}": r["skipped"]
            for r in records if r.get("family") == "categories" and r.get("skipped")}


def suite_key(record):
    return ("categories:" + record["suite"]) if record.get("family") == "categories" else record["suite"]


def heldout_cells(records):
    heldout, reported = {}, {}
    for r in records:
        if r.get("lang") == "macro" or "accuracy" not in r:
            continue
        row = {"acc": r["accuracy"], "n": r["n"], "ece": r.get("ece")}
        if r.get("family") == "categories":
            row["pool"] = r.get("pool")
            if "pool_items_sha256" in r:
                row["pool_items_sha256"] = r["pool_items_sha256"]
            if r.get("gate") is False:
                reported[f"{suite_key(r)}/{r['lang']}"] = dict(row, note=r.get("gate_note"))
                continue
        heldout[f"{suite_key(r)}/{r['lang']}"] = row
    return heldout, reported


ZERO_SHOT = {"go_emotions", "multi_hatecheck", "sib200", "indonli", "farstail", "belebele", "semrel"}
FAMILIES = {
    "trained": lambda k: k.split("/")[0] == "amazon_massive_intent" or k in ("test/banking77", "test/typed_decisions"),
    "zero-shot": lambda k: k.split("/")[0] in ZERO_SHOT or k in ("test/ag_news", "test/emotion"),
    "sentiment": lambda k: k.split("/")[0] == "multilingual_sentiments",
    "categories": lambda k: k.startswith("categories:"),
}


def pools_differ(x, y):
    if x.get("pool") != y.get("pool"):
        return True
    hx, hy = x.get("pool_items_sha256"), y.get("pool_items_sha256")
    return not hx or not hy or hx != hy


def artifact_causes(label, data):
    causes = []
    if data.get("gate_schema") != 2:
        causes.append(f"{label}: missing gate schema/strict completeness metadata")
    definitions = data.get("definitions") or {}
    if not definitions.get("registry_sha256"):
        causes.append(f"{label}: missing registry SHA-256")
    suite_hashes = definitions.get("suite_sha256") or {}
    for script in EVAL_SCRIPTS:
        if script not in suite_hashes:
            causes.append(f"{label}: missing suite definition SHA-256 for {script}")
    if not isinstance(data.get("skipped"), dict):
        causes.append(f"{label}: missing record of skipped category cells")
    status = data.get("suite_status") or {}
    for script in EVAL_SCRIPTS:
        state = status.get(script)
        if not state:
            causes.append(f"{label}: missing suite status for {script}")
        elif state.get("status") != "ok":
            causes.append(f"{label}: {state.get('status', 'failed')} suite {script}: "
                          f"{state.get('detail', 'no detail')}")
    return causes


def compatibility_causes(a, b):
    causes = artifact_causes("champion", a) + artifact_causes("challenger", b)
    ad, bd = a.get("definitions") or {}, b.get("definitions") or {}
    if ad.get("registry_sha256") != bd.get("registry_sha256"):
        causes.append("registry hash mismatch between champion and challenger")
    ah, bh = ad.get("suite_sha256") or {}, bd.get("suite_sha256") or {}
    for script in sorted(set(ah) | set(bh)):
        if ah.get(script) != bh.get(script):
            causes.append(f"suite definition hash mismatch: {script}")
    for section in ("validation", "heldout", "reported", "skipped"):
        ak, bk = set(a.get(section) or {}), set(b.get(section) or {})
        for name in sorted(ak - bk):
            causes.append(f"challenger missing {section} suite {name}")
        for name in sorted(bk - ak):
            causes.append(f"champion missing {section} suite {name}")
    for key in sorted(set(a.get("heldout", {})) & set(b.get("heldout", {}))):
        if key.startswith("categories:") and pools_differ(a["heldout"][key], b["heldout"][key]):
            causes.append(f"item-pool mismatch: {key}")
    return causes


def load_items(path):
    grouped = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                grouped.setdefault(row["suite"], []).append(row)
    return grouped


def load_evaluation(model_dir):
    eval_path = os.path.join(model_dir, "eval.json")
    with open(eval_path, encoding="utf-8") as f:
        data = json.load(f)
    expected = data.get("eval_items_sha256")
    item_path = os.path.join(model_dir, ITEMS_NAME)
    fix = f"rerun tools/finetune/gate.py eval {model_dir}"
    if not expected:
        raise SystemExit(f"{eval_path}: old eval.json has no per-item outcomes; {fix}")
    if not os.path.isfile(item_path):
        raise SystemExit(f"{item_path}: missing per-item outcomes; {fix}")
    actual = sha256_file(item_path)
    if actual != expected:
        raise SystemExit(f"{item_path}: SHA-256 differs from eval.json; {fix}")
    return data, load_items(item_path)


def paired_cell(name, a_rows, b_rows):
    if not a_rows or not b_rows:
        raise SystemExit(f"{name}: missing per-item outcomes; rerun gate.py eval for both models")
    if len(a_rows) != len(b_rows):
        raise SystemExit(f"{name}: per-item outcome counts differ ({len(a_rows)} != {len(b_rows)}); rerun gate.py eval")
    diffs, b_disc, c_disc = [], 0, 0
    for pos, (x, y) in enumerate(zip(a_rows, b_rows)):
        if x.get("i") != y.get("i") or x.get("item") != y.get("item"):
            raise SystemExit(f"{name}: item hashes/order differ at position {pos}; rerun gate.py eval with the same suites")
        if x.get("gold") != y.get("gold"):
            raise SystemExit(f"{name}: gold label differs at item {pos}; rerun gate.py eval with the same suites")
        xr, yr = int(x["pred"] == x["gold"]), int(y["pred"] == y["gold"])
        diffs.append(yr - xr)
        b_disc += xr == 1 and yr == 0
        c_disc += xr == 0 and yr == 1
    n = len(diffs)
    d = sum(diffs) / n
    variance = sum((v - d) ** 2 for v in diffs) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(variance / n)
    xa = sum(int(r["pred"] == r["gold"]) for r in a_rows) / n
    ya = sum(int(r["pred"] == r["gold"]) for r in b_rows) / n
    discord = b_disc + c_disc
    return {"name": name, "a": xa, "b": ya, "d": d, "se": se, "diffs": diffs,
            "b_disc": b_disc, "c_disc": c_disc,
            "p_drop": binomial_upper_half(discord, b_disc),
            "p_gain": binomial_upper_half(discord, c_disc)}


def _check_published_cell(name, side, published, rows):
    """Fail closed unless paired items reproduce the evaluator's rounded accuracy and count."""
    item_n = len(rows) if rows is not None else 0
    eval_n = published.get("n")
    if eval_n is not None and item_n != eval_n:
        raise SystemExit(f"{name}: {side} item count {item_n} != eval.json n {eval_n}")
    if rows is None:
        raise SystemExit(f"{name}: {side} has no item outcomes for eval.json accuracy {published['acc']}")
    item_acc = round(sum(r["pred"] == r["gold"] for r in rows) / item_n, 4) if item_n else 0.0
    eval_acc = round(float(published["acc"]), 4)
    if item_acc != eval_acc:
        raise SystemExit(f"{name}: {side} item accuracy {item_acc:.4f} != eval.json accuracy {eval_acc:.4f}")


def holm_adjusted(tests, direction="drop"):
    field = "p_" + direction
    ranked = sorted(tests, key=lambda t: t[field])
    out, running, m = {}, 0.0, len(ranked)
    for i, t in enumerate(ranked):
        running = max(running, min(1.0, (m - i) * t[field]))
        out[t["name"]] = running
    return out


def holm_regressions(tests, alpha=ALPHA):
    adjusted = holm_adjusted(tests, "drop")
    return [t for t in tests if t["d"] < 0 and adjusted[t["name"]] <= alpha]


def holm_gains(tests, alpha=ALPHA):
    adjusted = holm_adjusted(tests, "gain")
    return [t for t in tests if t["d"] > 0 and adjusted[t["name"]] <= alpha]


def _binomial_upper_tail(n, x, p):
    if x <= 0:
        return 1.0
    if x > n or p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    logs = [(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
             + k * math.log(p) + (n - k) * math.log1p(-p)) for k in range(x, n + 1)]
    peak = max(logs)
    return min(1.0, math.exp(peak) * sum(math.exp(v - peak) for v in logs))


def zero_event_half_width(n, confidence=0.95):
    """Clopper-Pearson bound on a proportion when none of n items differ: with no discordant pair
    the difference is not known to be zero, only to be below 1 - (alpha/2)^(1/n)."""
    return 1.0 - ((1.0 - confidence) / 2.0) ** (1.0 / n) if n else 1.0


def minimal_detectable_drop(n, discord, alpha):
    """Smallest observed net loss/n significant at alpha, conditional on discordant count."""
    if not n or not discord:
        return None
    for losses in range((discord + 1) // 2, discord + 1):
        if binomial_upper_half(discord, losses) <= alpha:
            return (2 * losses - discord) / n
    return None


def capability_tests(tests, tolerance=DEFAULT_CAPABILITY_TOLERANCE, tolerances=None):
    tolerances = tolerances or {}
    grouped = {}
    for test in tests:
        grouped.setdefault(capability_for(test["name"]), []).append(test)
    out = []
    for capability, members in sorted(grouped.items()):
        diffs = [v for member in members for v in member["diffs"]]
        losses = sum(v < 0 for v in diffs)
        gains = sum(v > 0 for v in diffs)
        n = len(diffs)
        discord = losses + gains
        out.append({"name": capability, "cells": len(members), "n": n,
                    "d": sum(diffs) / n, "diffs": diffs,
                    "b_disc": losses, "c_disc": gains,
                    "p_drop": binomial_upper_half(discord, losses),
                    "p_gain": binomial_upper_half(discord, gains),
                    "tolerance": tolerances.get(capability, tolerance)})
    p_drop = holm_adjusted(out, "drop")
    p_gain = holm_adjusted(out, "gain")
    threshold = ALPHA / len(out) if out else ALPHA
    regressions, gains = [], []
    for test in out:
        test["p_drop_holm"] = p_drop[test["name"]]
        test["p_gain_holm"] = p_gain[test["name"]]
        test["mdd"] = minimal_detectable_drop(test["n"], test["b_disc"] + test["c_disc"], threshold)
        test["underpowered"] = test["n"] < UNDERPOWERED_N
        test["regression_significant"] = test["d"] < 0 and test["p_drop_holm"] <= ALPHA
        test["regression_tolerance"] = test["d"] < -test["tolerance"]
        if test["regression_significant"] or test["regression_tolerance"]:
            regressions.append(test)
        if test["d"] > 0 and test["p_gain_holm"] <= ALPHA:
            gains.append(test)
    return out, regressions, gains


def _pooled(groups, tests_by_name):
    d, variance = 0.0, 0.0
    for group in groups:
        group_weight = 1.0 / len(groups)
        for key in group:
            t = tests_by_name[key]
            weight = group_weight / len(group)
            d += weight * t["d"]
            variance += weight * weight * t["se"] ** 2
    return d, math.sqrt(variance)


def heldout_decision(a, b, a_items, b_items, z=2.0, log=print,
                     capability_tolerance=DEFAULT_CAPABILITY_TOLERANCE, capability_tolerances=None):
    if set(a) != set(b):
        missing = sorted(set(a) ^ set(b))
        raise SystemExit("held-out suite sets differ: " + ", ".join(missing))
    shared = sorted(a)
    different = [k for k in shared if k.startswith("categories:") and pools_differ(a[k], b[k])]
    if different:
        raise SystemExit("item-pool mismatch: " + ", ".join(different))
    for k in shared:
        _check_published_cell(k, "champion", a[k], a_items.get(k))
        _check_published_cell(k, "challenger", b[k], b_items.get(k))
    tests = [paired_cell(k, a_items.get(k), b_items.get(k)) for k in shared]
    by_name = {t["name"]: t for t in tests}
    p_drop_holm = holm_adjusted(tests, "drop")
    p_gain_holm = holm_adjusted(tests, "gain")
    cell_harms, gains = holm_regressions(tests), holm_gains(tests)
    capabilities, harms, capability_gains = capability_tests(
        tests, capability_tolerance, capability_tolerances)
    families, family_tests = {}, []
    for fam, member in FAMILIES.items():
        ks = [k for k in shared if member(k)]
        if not ks:
            continue
        bysuite = {}
        for k in ks:
            bysuite.setdefault(k if k.startswith("test/") else k.split("/")[0], []).append(k)
        for weighting, groups in (("rows", [[k] for k in ks]), ("suites", list(bysuite.values()))):
            d, se = _pooled(groups, by_name)
            p_gain = 1.0 if se == 0 and d <= 0 else (0.0 if se == 0 else
                                                     0.5 * math.erfc(d / se / math.sqrt(2.0)))
            test = {"name": f"family:{fam}/{weighting}", "family": fam, "weighting": weighting,
                    "d": d, "se": se, "p_gain": p_gain}
            family_tests.append(test)
            families.setdefault(fam, {})[weighting] = {
                "d": d, "se": se, "groups": len(groups), "p_gain": p_gain}
    family_p_holm = holm_adjusted(family_tests, "gain")
    family_gains, family_harms = [], []
    for t in family_tests:
        # A false harm merely rejects this challenger; a false gain can ship it. Therefore gains
        # get family-wise correction, while harms retain the more cautious uncorrected 2-SE screen.
        regression = t["d"] < -z * t["se"]
        gain = t["d"] > 0 and family_p_holm[t["name"]] <= ALPHA
        flag = "REGRESSION" if regression else ("gain (Holm)" if gain else "within noise")
        row = families[t["family"]][t["weighting"]]
        row.update(flag=flag, p_gain_holm=family_p_holm[t["name"]])
        log(f"family {t['family']:10s} ({t['weighting']:6s}, {row['groups']:2d}): "
            f"{t['d'] * 100:+.2f} pts, 2se {2 * t['se'] * 100:.2f}, "
            f"Holm p(gain)={family_p_holm[t['name']]:.3g} -> {flag}")
        if regression:
            family_harms.append(t)
        if gain:
            if t["family"] not in family_gains:
                family_gains.append(t["family"])
    # Every test blocks on its own: a pooled capability hides one collapsed language behind gains
    # elsewhere, a single cell is too small to see a loss spread thinly over many cells, and the
    # family screen keeps the pre-capability rule so no regression it caught can pass now.
    return {"compared": shared, "not_compared": [], "tests": tests, "gains": gains,
            "harms": harms + cell_harms + family_harms, "capability_harms": harms,
            "family_harms": family_harms,
            "families": families, "family_gains": family_gains,
            "cell_harms": cell_harms, "capabilities": capabilities,
            "capability_gains": capability_gains,
            "family_p_gain_holm": family_p_holm,
            "p_drop_holm": p_drop_holm, "p_gain_holm": p_gain_holm}


def summary(champ_dir, chall_dir, z=2.0, log=lambda *_: None):
    """Return verified evaluations, the paired decision, and per-cell publishing verdicts."""
    champion, champion_items = load_evaluation(champ_dir)
    challenger, challenger_items = load_evaluation(chall_dir)
    decision = heldout_decision(champion["heldout"], challenger["heldout"],
                                champion_items, challenger_items, z=z, log=log)
    gains = {test["name"] for test in decision["gains"]}
    losses = {test["name"] for test in decision["cell_harms"]}
    cells = {}
    counts = {"gain": 0, "noise": 0, "loss": 0}
    for test in decision["tests"]:
        verdict = "gain" if test["name"] in gains else "loss" if test["name"] in losses else "noise"
        cells[test["name"]] = dict(test, verdict=verdict,
                                   p_drop_holm=decision["p_drop_holm"][test["name"]],
                                   p_gain_holm=decision["p_gain_holm"][test["name"]])
        counts[verdict] += 1
    return {"champion": champion, "challenger": challenger, "decision": decision,
            "cells": cells, "counts": counts}


def report(res, log=print):
    log(f"per-suite paired tests (Holm FWER {ALPHA:.0%}; adjusted p for each direction): {len(res['tests'])}")
    for t in res["tests"]:
        if t["d"]:
            log(f"  {t['name']:40s} {t['a']:.4f} -> {t['b']:.4f} ({t['d']:+.4f}, "
                f"paired 2se {2 * t['se']:.4f}, b={t['b_disc']}, c={t['c_disc']}, "
                f"Holm p(drop)={res['p_drop_holm'][t['name']]:.3g}, "
                f"p(gain)={res['p_gain_holm'][t['name']]:.3g})")
    log(f"significant gains (paired exact McNemar + Holm): {len(res['gains'])}")
    log(f"significant per-cell regressions (Holm, blocking): {len(res['cell_harms'])}")
    log("capability tests (pooled paired items; exact McNemar + Holm):")
    for t in res["capabilities"]:
        flags = []
        if t["underpowered"]:
            flags.append("UNDERPOWERED")
        if t in res["capability_harms"]:
            flags.append("REGRESSION")
        elif t in res["capability_gains"]:
            flags.append("gain (Holm)")
        else:
            flags.append("within noise")
        mdd = "n/a" if t["mdd"] is None else f"{t['mdd'] * 100:.2f} pts"
        log(f"  {t['name']:24s} n={t['n']:5d} discord={t['b_disc'] + t['c_disc']:4d} "
            f"(loss={t['b_disc']}, gain={t['c_disc']}), drop={t['d'] * 100:+.2f} pts, "
            f"Holm p(drop)={t['p_drop_holm']:.3g}, MDD={mdd} -> {', '.join(flags)}")
    log(f"capability regressions: {len(res['capability_harms'])}")
    log(f"family regressions (2-SE screen): {len(res['family_harms'])}")


def validation_stats(a, b, a_items, b_items, keys, z=1.96):
    """Paired 95% interval for the unweighted mean of suite deltas, the number the gate prints and
    the margin refers to. Each suite contributes its paired standard error, floored at the
    zero-event bound so a small or all-tie suite cannot claim certainty; suites are independent
    samples, so the half-widths add in quadrature with weight 1/len(keys)."""
    for k in keys:
        cell = "validation:" + k
        a_rows, b_rows = a_items.get(cell), b_items.get(cell)
        _check_published_cell(cell, "champion", a[k], a_rows)
        _check_published_cell(cell, "challenger", b[k], b_rows)
    tests = [paired_cell("validation:" + k, a_items.get("validation:" + k),
                         b_items.get("validation:" + k)) for k in keys]
    d = sum(t["d"] for t in tests) / len(tests)
    half = math.sqrt(sum(max(z * t["se"], zero_event_half_width(len(t["diffs"]))) ** 2
                         for t in tests)) / len(tests)
    return d, d - half, d + half


def _legacy_unpaired(a, b, z=2.0, log=print):
    log("UNPAIRED (legacy, not a decision)")
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        se = math.sqrt(x["acc"] * (1 - x["acc"]) / x["n"] + y["acc"] * (1 - y["acc"]) / y["n"])
        d = y["acc"] - x["acc"]
        flag = "drop" if d < -z * se else ("gain" if d > z * se else "within noise")
        log(f"  {k:40s} {x['acc']:.4f} -> {y['acc']:.4f} ({d:+.4f}, unpaired 2se {2 * se:.4f}) {flag}")


def compare(champ_dir, chall_dir, z=2.0, legacy_unpaired=False, strict=True,
            capability_tolerance=DEFAULT_CAPABILITY_TOLERANCE, capability_tolerances=None):
    if legacy_unpaired:
        with open(os.path.join(champ_dir, "eval.json"), encoding="utf-8") as f:
            a = json.load(f)
        with open(os.path.join(chall_dir, "eval.json"), encoding="utf-8") as f:
            b = json.load(f)
        _legacy_unpaired(a["heldout"], b["heldout"], z)
        print("VERDICT: REPORT (legacy report-only mode can never promote)")
        return False
    try:
        a, ai = load_evaluation(champ_dir)
        b, bi = load_evaluation(chall_dir)
    except (SystemExit, OSError, ValueError, KeyError) as exc:
        print(f"BLOCKED: {exc}")
        print("VERDICT: BLOCKED")
        return False
    causes = compatibility_causes(a, b)
    if causes:
        for cause in causes:
            print("BLOCKED:", cause)
        print("VERDICT: BLOCKED")
        return False
    keys = sorted(a["validation"])
    if not keys:
        print("BLOCKED: no validation suites")
        print("VERDICT: BLOCKED")
        return False
    va = sum(a["validation"][k]["acc"] for k in keys) / len(keys)
    vb = sum(b["validation"][k]["acc"] for k in keys) / len(keys)
    try:
        vd, vlo, vhi = validation_stats(a["validation"], b["validation"], ai, bi, keys)
        res = heldout_decision(a["heldout"], b["heldout"], ai, bi, z,
                               capability_tolerance=capability_tolerance,
                               capability_tolerances=capability_tolerances)
    except (SystemExit, OSError, ValueError, KeyError, ZeroDivisionError) as exc:
        print(f"BLOCKED: {exc}")
        print("VERDICT: BLOCKED")
        return False
    print(f"validation mean: champion {va:.4f} -> challenger {vb:.4f} ({vb - va:+.4f}); "
          f"paired 95% CI [{vlo:+.4f}, {vhi:+.4f}]")
    report(res)
    for cell, why in sorted((a.get("skipped") or {}).items()):
        print(f"not covered (skipped for both models): {cell}: {why}")
    holds, better = vlo >= -VAL_MARGIN, bool(res["family_gains"])
    ok = holds and not res["harms"] and better
    report_only = not strict or not a.get("strict", False) or not b.get("strict", False)
    if report_only:
        print("VERDICT: REPORT (--no-strict evaluation/comparison can never promote)")
        return False
    why = ("" if ok else f"(validation 95% lower bound below -{VAL_MARGIN * 100:.0f} point)" if not holds
           else "(held-out regression)" if res["harms"] else "(no family improved significantly)")
    print("VERDICT:", "PROMOTE" if ok else "BLOCKED", why)
    return ok


def adapter_decision(base, adapter, base_items=None, adapter_items=None, z=2.0, log=print):
    """Paired category-only decision for a LoRA adapter against the base on identical items."""
    if base_items is None or adapter_items is None:
        raise SystemExit("adapter_decision: per-item outcomes are required; rerun gate.py eval "
                         "(or rerun tools/finetune/lora_experiment.py for an adapter)")
    cats = lambda h: {k: v for k, v in h.items() if k.startswith("categories:")}
    res = heldout_decision(cats(base), cats(adapter), base_items, adapter_items, z, log)
    report(res, log)
    if not res["compared"]:
        ok, why = False, "(no category cell compared)"
    else:
        ok = not res["harms"] and bool(res["family_gains"])
        why = "" if ok else "(held-out regression)" if res["harms"] else "(no family improved significantly)"
    log("VERDICT: " + ("PROMOTE" if ok else "REJECT") + (" " + why if why else ""))
    res.update(promote=ok, reason=why, p_holm=res["p_drop_holm"])
    return res


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("model")
    e.add_argument("--mixture", default=os.environ.get("STATIM_GATE_MIXTURE"))
    e.add_argument("--no-strict", action="store_true",
                   help="allow an incomplete category pool; artifact is report-only")
    c = sub.add_parser("compare")
    c.add_argument("champion")
    c.add_argument("challenger")
    c.add_argument("--z", type=float, default=2.0)
    c.add_argument("--legacy-unpaired", action="store_true",
                   help="report the old unpaired approximation; never returns PROMOTE")
    c.add_argument("--no-strict", action="store_true", help="report only; never returns PROMOTE")
    c.add_argument("--capability-tolerance", type=float, default=2.0, metavar="POINTS",
                   help="maximum capability drop in percentage points (default: 2.0)")
    a = ap.parse_args()
    if a.cmd == "eval":
        evaluate(a.model, a.mixture, strict=not a.no_strict)
    else:
        if a.capability_tolerance < 0:
            c.error("--capability-tolerance must be >= 0")
        sys.exit(0 if compare(a.champion, a.challenger, a.z, a.legacy_unpaired,
                              strict=not a.no_strict,
                              capability_tolerance=a.capability_tolerance / 100.0) else 1)


if __name__ == "__main__":
    main()
