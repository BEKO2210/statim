#!/usr/bin/env python3
"""No-harm promotion gate: evaluate checkpoints, then make a paired promotion decision.

    # writes <model>/eval.json and <model>/eval-items.jsonl.gz
    .venv-train/bin/python tools/finetune/gate.py eval models/laya-multilingual-clean
    .venv-train/bin/python tools/finetune/gate.py compare models/champion models/challenger

Validation uses held-out development suites. Its equally weighted mean may not fall by more than
VAL_MARGIN (one point); the log also gives a paired 95% interval for the mean difference. Held-out
test cells use one-sided exact McNemar tests on discordant answers, with separate Holm-Bonferroni
correction for drops and gains. Family row- and suite-weighted means use the observed per-item
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

A challenger is promoted only when validation holds, no paired held-out regression remains, and at
least one paired family mean has a Holm-significant gain. Old eval.json files cannot
prove pairing and are refused. ``compare --legacy-unpaired`` can print the former approximation for
historical inspection, but is labelled report-only and can never promote. LoRA adapters use the same
paired held-out path through adapter_decision.
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


def http_suites(model_dir, tmp, mixture=None):
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
        rows, items = [], []
        categories = ["--n", "150"] + (["--exclude-mixture", mixture] if mixture else [])
        for script, extra in (("bench/eval_multilingual.py", ["--n", "150"]),
                              ("bench/eval_zeroshot.py", ["--n", "150"]),
                              ("bench/eval_categories.py", categories)):
            if not os.path.exists(os.path.join(ROOT, script)):
                continue
            stem = os.path.basename(script)
            out = os.path.join(tmp, stem + ".jsonl")
            pred = os.path.join(tmp, stem + ".items.jsonl")
            run([PY, script, "--url", f"http://127.0.0.1:{PORT}", "--model", "multilingual",
                 "--out", out, "--predictions", pred, *extra])
            rows += [json.loads(line) for line in open(out, encoding="utf-8")]
            if stem == "eval_categories.py":
                key = lambda r: f"categories:{r['suite']}/{r['lang']}"
            else:
                key = lambda r: f"{r['suite']}/{r['lang']}"
            items += _prediction_rows(pred, key)
        return rows, items
    finally:
        srv.send_signal(signal.SIGINT)
        srv.wait(timeout=60)


def evaluate(model_dir, mixture=None):
    tmp = os.path.join(model_dir, "eval-tmp")
    os.makedirs(tmp, exist_ok=True)
    res = {"model": model_dir, "validation": {}, "heldout": {}, "reported": {}, "mixture": mixture}
    dev_items = os.path.join(tmp, "eval_dev.items.jsonl")
    dev = jsonl(run([PY, "tools/finetune/eval_dev.py", model_dir, "--predictions", dev_items]))[0]
    res["validation"] = {k: {"acc": dev[k], "n": dev["n"][k]}
                         for k in dev if k not in ("model", "n", "seconds", "mean")}
    items = _prediction_rows(dev_items, lambda r: "validation:" + r["suite"])

    laya_items = os.path.join(tmp, "eval_laya.items.jsonl")
    command = [PY, "tools/finetune/eval_laya.py", model_dir, "--n", "2000", "--head-max-len", "512",
               "--predictions", laya_items]
    for r in jsonl(run(command)):
        res["heldout"][f"test/{r['suite']}"] = {"acc": r["accuracy"], "n": r.get("n", 2000),
                                                  "ece": r.get("ece")}
    items += _prediction_rows(laya_items, lambda r: "test/" + r["suite"])

    http_records, http_items = http_suites(model_dir, tmp, mixture)
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
    print(json.dumps({"validation_mean": round(sum(v["acc"] for v in res["validation"].values()) /
                                                len(res["validation"]), 4),
                      "heldout_suites": len(res["heldout"]),
                      "eval_items_sha256": res["eval_items_sha256"]}))


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
    return hx is not None and hy is not None and hx != hy


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


def heldout_decision(a, b, a_items, b_items, z=2.0, log=print):
    shared = sorted(set(a) & set(b))
    different = [k for k in shared if k.startswith("categories:") and pools_differ(a[k], b[k])]
    if different:
        log(f"categories: {len(different)} cells not compared (their pools differ)")
    shared = [k for k in shared if k not in different]
    for k in shared:
        _check_published_cell(k, "champion", a[k], a_items.get(k))
        _check_published_cell(k, "challenger", b[k], b_items.get(k))
    tests = [paired_cell(k, a_items.get(k), b_items.get(k)) for k in shared]
    by_name = {t["name"]: t for t in tests}
    p_drop_holm = holm_adjusted(tests, "drop")
    p_gain_holm = holm_adjusted(tests, "gain")
    harms, gains = holm_regressions(tests), holm_gains(tests)
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
    family_gains = []
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
            harms.append({"name": t["name"], "a": 0.0, "b": t["d"],
                          "d": t["d"], "se": t["se"], "p_drop": None, "p_gain": t["p_gain"]})
        elif gain:
            if t["family"] not in family_gains:
                family_gains.append(t["family"])
    return {"compared": shared, "not_compared": different, "tests": tests, "gains": gains,
            "harms": harms, "families": families, "family_gains": family_gains,
            "family_p_gain_holm": family_p_holm,
            "p_drop_holm": p_drop_holm, "p_gain_holm": p_gain_holm}


def summary(champ_dir, chall_dir, z=2.0, log=lambda *_: None):
    """Return verified evaluations, the paired decision, and per-cell publishing verdicts."""
    champion, champion_items = load_evaluation(champ_dir)
    challenger, challenger_items = load_evaluation(chall_dir)
    decision = heldout_decision(champion["heldout"], challenger["heldout"],
                                champion_items, challenger_items, z=z, log=log)
    gains = {test["name"] for test in decision["gains"]}
    losses = {test["name"] for test in decision["harms"] if not test["name"].startswith("family:")}
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
    log(f"significant regressions (paired exact McNemar + Holm, or paired family): {len(res['harms'])}")


def validation_stats(a, b, a_items, b_items, keys):
    for k in keys:
        cell = "validation:" + k
        a_rows, b_rows = a_items.get(cell), b_items.get(cell)
        _check_published_cell(cell, "champion", a[k], a_rows)
        _check_published_cell(cell, "challenger", b[k], b_rows)
    tests = [paired_cell("validation:" + k, a_items.get("validation:" + k),
                         b_items.get("validation:" + k)) for k in keys]
    return _pooled([[t["name"]] for t in tests], {t["name"]: t for t in tests})


def _legacy_unpaired(a, b, z=2.0, log=print):
    log("UNPAIRED (legacy, not a decision)")
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        se = math.sqrt(x["acc"] * (1 - x["acc"]) / x["n"] + y["acc"] * (1 - y["acc"]) / y["n"])
        d = y["acc"] - x["acc"]
        flag = "drop" if d < -z * se else ("gain" if d > z * se else "within noise")
        log(f"  {k:40s} {x['acc']:.4f} -> {y['acc']:.4f} ({d:+.4f}, unpaired 2se {2 * se:.4f}) {flag}")


def compare(champ_dir, chall_dir, z=2.0, legacy_unpaired=False):
    if legacy_unpaired:
        with open(os.path.join(champ_dir, "eval.json"), encoding="utf-8") as f:
            a = json.load(f)
        with open(os.path.join(chall_dir, "eval.json"), encoding="utf-8") as f:
            b = json.load(f)
        _legacy_unpaired(a["heldout"], b["heldout"], z)
        print("VERDICT: REJECT (legacy report-only mode can never promote)")
        return False
    a, ai = load_evaluation(champ_dir)
    b, bi = load_evaluation(chall_dir)
    keys = sorted(set(a["validation"]) & set(b["validation"]))
    if not keys:
        raise SystemExit("compare: the two eval.json files share no validation suite")
    va = sum(a["validation"][k]["acc"] for k in keys) / len(keys)
    vb = sum(b["validation"][k]["acc"] for k in keys) / len(keys)
    vd, vse = validation_stats(a["validation"], b["validation"], ai, bi, keys)
    res = heldout_decision(a["heldout"], b["heldout"], ai, bi, z)
    print(f"validation mean: champion {va:.4f} -> challenger {vb:.4f} ({vb - va:+.4f}); "
          f"paired 95% CI [{vd - 1.96 * vse:+.4f}, {vd + 1.96 * vse:+.4f}]")
    report(res)
    holds, better = vb >= va - VAL_MARGIN, bool(res["family_gains"])
    ok = holds and not res["harms"] and better
    why = ("" if ok else f"(validation fell by more than {VAL_MARGIN * 100:.0f} point)" if not holds
           else "(held-out regression)" if res["harms"] else "(no family improved significantly)")
    print("VERDICT:", "PROMOTE" if ok else "REJECT", why)
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
    c = sub.add_parser("compare")
    c.add_argument("champion")
    c.add_argument("challenger")
    c.add_argument("--z", type=float, default=2.0)
    c.add_argument("--legacy-unpaired", action="store_true",
                   help="report the old unpaired approximation; never returns PROMOTE")
    a = ap.parse_args()
    if a.cmd == "eval":
        evaluate(a.model, a.mixture)
    else:
        sys.exit(0 if compare(a.champion, a.challenger, a.z, a.legacy_unpaired) else 1)


if __name__ == "__main__":
    main()
