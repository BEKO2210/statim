#!/usr/bin/env python3
"""No-harm promotion gate: evaluate checkpoints on every suite, then decide whether a challenger may
replace the champion.

    # evaluate (writes <model>/eval.json; needs the GGUF next to the model dir and a Vulkan build,
    # or STATIM_BIN=build/statim STATIM_GATE_DEVICE=cpu for a CPU-only machine)
    .venv-train/bin/python tools/finetune/gate.py eval models/laya-multilingual-clean
    # compare
    .venv-train/bin/python tools/finetune/gate.py compare models/champion models/challenger

Suites and their role:
- validation (eval_dev.py: Banking77 held-out train rows, MASSIVE validation, sentiment valid,
  Emotion validation, AG News train rows): the challenger's mean may not fall by more than
  VAL_MARGIN (one point). Model selection never looks at test splits.
- held-out tests (eval_laya.py on the first 2,000 rows; bench/eval_multilingual.py per language;
  bench/eval_zeroshot.py suites never trained on; bench/eval_categories.py, one suite per decision
  category from the unused splits of the mixture v6 sources):
  * per suite, a drop is a regression when a one-sided z-test (binomial standard errors of both
    models, combined) stays significant after Holm-Bonferroni over all compared suites, at a
    family-wise error rate of ALPHA = 5 %;
  * per family (trained, zero-shot, sentiment, categories; pooled by rows and by suites), a pooled
    drop of more than 2 standard errors is a regression;
  * at least one family must improve by more than 2 standard errors.
A challenger is promoted only when validation holds, no held-out regression remains and it is
measurably better somewhere: no overfitting, no forgetting, and a real gain.

Why Holm-Bonferroni: with 88 suites and a plain 2-SE rule per suite, an unchanged model shows at
least one "significant" drop about 87 % of the time (1 - 0.977**88), and a point comparison of the
validation mean rejects half of all equally good models. Both rules together rejected an equally
good challenger about 94 % of the time.

LoRA adapters (tools/finetune/lora_experiment.py) are judged by adapter_decision: the held-out part
of compare() on the adapter's category cells alone, without the validation step (see its docstring).
"""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY = os.path.join(ROOT, ".venv-train", "bin", "python")
PORT = int(os.environ.get("STATIM_GATE_PORT", "8097"))
# Third parties reproduce on whatever they have: STATIM_BIN and STATIM_GATE_DEVICE override the
# Vulkan build this project measures with (exact f32 on GPU and CPU give the same answers).
BIN = os.environ.get("STATIM_BIN", os.path.join(ROOT, "build-vk", "statim"))
DEVICE = os.environ.get("STATIM_GATE_DEVICE", "vulkan")
CONVERT_PY = os.environ.get("STATIM_CONVERT_PY", os.path.join(ROOT, ".venv", "bin", "python"))


def run(cmd):
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if out.returncode:
        sys.stderr.write(out.stdout[-2000:] + out.stderr[-4000:])
        raise SystemExit(f"failed: {' '.join(cmd)}")
    return out.stdout


def jsonl(text):
    return [json.loads(line) for line in text.splitlines() if line.startswith("{")]


def gguf_for(model_dir):
    path = model_dir.rstrip("/") + "-f32.gguf"
    if not os.path.exists(path):
        run([CONVERT_PY if os.path.exists(CONVERT_PY) else PY, "tools/convert_laya.py", model_dir, "-o", path,
             "--type", "f32", "--embd-type", "f16", "--name", os.path.basename(model_dir.rstrip("/"))])
    return path


def http_suites(model_dir, tmp, mixture=None):
    srv = subprocess.Popen([BIN, "serve", "--device", DEVICE, "-m",
                            f"m={gguf_for(model_dir)}", "--port", str(PORT), "--no-access-log"],
                           cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/ready", timeout=2)
                break
            except Exception:
                time.sleep(1)
        rows = []
        # the benchmark scripts only accept known model names; the server has a single model, and an
        # unknown name falls back to it, so pass a valid name
        categories = ["--n", "150"] + (["--exclude-mixture", mixture] if mixture else [])
        for script, extra in (("bench/eval_multilingual.py", ["--n", "150"]),
                              ("bench/eval_zeroshot.py", ["--n", "150"]),
                              ("bench/eval_categories.py", categories)):
            if not os.path.exists(os.path.join(ROOT, script)):
                continue
            out = os.path.join(tmp, os.path.basename(script) + ".jsonl")
            run([PY, script, "--url", f"http://127.0.0.1:{PORT}", "--model", "multilingual", "--out", out, *extra])
            rows += [json.loads(l) for l in open(out)]
        return rows
    finally:
        srv.send_signal(signal.SIGINT)
        srv.wait(timeout=60)


def evaluate(model_dir, mixture=None):
    tmp = os.path.join(model_dir, "eval-tmp")
    os.makedirs(tmp, exist_ok=True)
    res = {"model": model_dir, "validation": {}, "heldout": {}, "reported": {}, "mixture": mixture}
    dev = jsonl(run([PY, "tools/finetune/eval_dev.py", model_dir]))[0]
    res["validation"] = {k: v for k, v in dev.items() if k not in ("model", "seconds", "mean")}
    for r in jsonl(run([PY, "tools/finetune/eval_laya.py", model_dir, "--n", "2000", "--head-max-len", "512"])):
        n = r.get("n", 2000)
        res["heldout"][f"test/{r['suite']}"] = {"acc": r["accuracy"], "n": n, "ece": r.get("ece")}
    heldout, reported = heldout_cells(http_suites(model_dir, tmp, mixture))
    res["heldout"].update(heldout)
    res["reported"].update(reported)
    json.dump(res, open(os.path.join(model_dir, "eval.json"), "w"), indent=1)
    print(json.dumps({"validation_mean": round(sum(res["validation"].values()) / len(res["validation"]), 4),
                      "heldout_suites": len(res["heldout"])}))


def suite_key(record):
    """eval_categories.py names its suites after the category ("sentiment", "nli"); prefix them so they
    cannot collide with a suite of another script."""
    return ("categories:" + record["suite"]) if record.get("family") == "categories" else record["suite"]


def heldout_cells(records):
    """(heldout, reported) from the per-language records of the HTTP benchmark scripts
    (eval_multilingual.py, eval_zeroshot.py, eval_categories.py): {"<suite key>/<lang>": {"acc", "n",
    "ece"[, "pool", "pool_items_sha256"]}}. Macro lines and skipped languages are left out; category
    cells with "gate": false (zero-shot or biased) go to `reported` with their note and are never gated."""
    heldout, reported = {}, {}
    for r in records:
        if r.get("lang") == "macro" or "accuracy" not in r:  # macro line, or a skipped language
            continue
        row = {"acc": r["accuracy"], "n": r["n"], "ece": r.get("ece")}
        if r.get("family") == "categories":
            row["pool"] = r.get("pool")  # compare() only compares category cells built from the same pool
            if "pool_items_sha256" in r:
                row["pool_items_sha256"] = r["pool_items_sha256"]
            if r.get("gate") is False:  # zero-shot or biased cell: reported, never gated
                reported[f"{suite_key(r)}/{r['lang']}"] = dict(row, note=r.get("gate_note"))
                continue
        heldout[f"{suite_key(r)}/{r['lang']}"] = row
    return heldout, reported


ALPHA = 0.05       # family-wise error rate of the per-suite regression tests
VAL_MARGIN = 0.01  # the validation mean may fall by at most one point


def p_drop(d, se):
    """One-sided p-value of the z-test that a change d (challenger - champion) with standard error se
    is a drop: Phi(d / se)."""
    return 0.5 * math.erfc(-(d / se) / math.sqrt(2)) if se > 0 else (0.0 if d < 0 else 1.0)


def holm_regressions(tests, alpha=ALPHA):
    """tests: (name, champion_acc, challenger_acc, d, se) per suite. Returns the suites whose drop is
    significant after Holm-Bonferroni: sort by one-sided p-value, compare the i-th smallest with
    alpha / (m - i), stop at the first one that is not significant."""
    ranked = sorted(tests, key=lambda t: p_drop(t[3], t[4]))
    out, m = [], len(ranked)
    for i, t in enumerate(ranked):
        if t[3] >= 0 or p_drop(t[3], t[4]) > alpha / (m - i):
            break
        out.append(t)
    return out


def holm_adjusted(tests):
    """{name: Holm-adjusted p-value of a drop} for the same tests: the running maximum of (m - i) * p
    over the suites in p-value order, capped at 1. A suite is in holm_regressions(tests, alpha) exactly
    when its adjusted p-value is at most alpha (drops only; a gain has p >= 0.5). For reports; the
    decision is holm_regressions."""
    ranked = sorted(tests, key=lambda t: p_drop(t[3], t[4]))
    out, running, m = {}, 0.0, len(ranked)
    for i, t in enumerate(ranked):
        running = max(running, min(1.0, (m - i) * p_drop(t[3], t[4])))
        out[t[0]] = running
    return out


ZERO_SHOT = {"go_emotions", "multi_hatecheck", "sib200", "indonli", "farstail", "belebele", "semrel"}


def _var(x):
    return x["acc"] * (1 - x["acc"]) / x["n"]


# Families pooled by heldout_decision (suite keys as in eval.json).
FAMILIES = {
    "trained": lambda k: k.split("/")[0] == "amazon_massive_intent" or k in ("test/banking77", "test/typed_decisions"),
    "zero-shot": lambda k: k.split("/")[0] in ZERO_SHOT or k in ("test/ag_news", "test/emotion"),
    "sentiment": lambda k: k.split("/")[0] == "multilingual_sentiments",
    "categories": lambda k: k.startswith("categories:"),
}


def pools_differ(x, y):
    """Two category cells differ in their pool when the suite specifications do, or when both record
    the realized sample (pool_items_sha256) and those differ. eval.json files written before the
    realized hash existed carry only "pool"; they stay comparable on it."""
    if x.get("pool") != y.get("pool"):
        return True
    hx, hy = x.get("pool_items_sha256"), y.get("pool_items_sha256")
    return hx is not None and hy is not None and hx != hy


def heldout_decision(a, b, z=2.0, log=print):
    """The held-out part of the gate for champion `a` and challenger `b`, two {suite key: {"acc", "n"
    [, "pool", "pool_items_sha256"]}} dicts (eval.json "heldout", or heldout_cells). Per suite a one-sided z-test with
    Holm-Bonferroni over every compared suite; per family (FAMILIES, pooled by rows and by suites) a
    pooled drop beyond z SE is a regression and a pooled gain beyond z SE a family gain. Category cells
    whose static or realized pools differ are not compared. Prints the family lines and returns {"compared",
    "not_compared", "tests", "gains", "harms", "nominal", "families", "family_gains"}; tests are
    (key, a_acc, b_acc, d, se)."""
    harms, gains = [], []
    # Category cells are comparable only when both models drew them from the same pool (same suite
    # spec, sources, adapters and, with --mixture, the same training mixture excluded).
    shared = sorted(set(a) & set(b))
    different = [k for k in shared if k.startswith("categories:") and pools_differ(a[k], b[k])]
    if different:
        log(f"categories: {len(different)} cells not compared (their pools differ)")
    shared = [k for k in shared if k not in different]
    tests = []
    for k in shared:
        x, y = a[k], b[k]
        se = math.sqrt(x["acc"] * (1 - x["acc"]) / x["n"] + y["acc"] * (1 - y["acc"]) / y["n"])
        d = y["acc"] - x["acc"]
        tests.append((k, x["acc"], y["acc"], d, se))
        if d > z * se:
            gains.append((k, x["acc"], y["acc"], d, se))
    harms += holm_regressions(tests)
    nominal = [t for t in tests if t[3] < -z * t[4] and t not in harms]  # reported, not decisive
    families, family_gains = {}, []
    # Family-level check: many small suites can each stay inside their noise band while all drifting
    # the same way. Pool each family (row-weighted and suite-weighted) and treat a significant pooled
    # drop as a regression too.
    for fam, member in FAMILIES.items():
        ks = [k for k in shared if member(k)]
        if not ks:
            continue
        bysuite = {}
        for k in ks:
            bysuite.setdefault(k if k.startswith("test/") else k.split("/")[0], []).append(k)

        def pooled(groups):
            ds, var = [], 0.0
            for g in groups:
                ds.append(sum(b[k]["acc"] - a[k]["acc"] for k in g) / len(g))
                var += sum(_var(a[k]) + _var(b[k]) for k in g) / len(g) ** 2
            return sum(ds) / len(ds), math.sqrt(var) / len(ds)
        for weighting, groups in (("rows", [[k] for k in ks]), ("suites", list(bysuite.values()))):
            d, se = pooled(groups)
            flag = "REGRESSION" if d < -z * se else ("gain" if d > z * se else "within noise")
            log(f"family {fam:10s} ({weighting:6s}, {len(groups):2d}): {d * 100:+.2f} pts, 2se {2 * se * 100:.2f} -> {flag}")
            families.setdefault(fam, {})[weighting] = {"d": d, "se": se, "groups": len(groups), "flag": flag}
            if d < -z * se:
                harms.append((f"family:{fam}/{weighting}", 0.0, d, d, se))
            elif d > z * se:
                family_gains.append(fam)
    return {"compared": shared, "not_compared": different, "tests": tests, "gains": gains, "harms": harms,
            "nominal": nominal, "families": families, "family_gains": family_gains}


def report(res, log=print):
    """Print the gains, regressions and nominal drops of a heldout_decision result."""
    for name, rows in (("significant gains (2 SE, per suite)", res["gains"]),
                       (f"significant regressions (Holm, FWER {ALPHA:.0%}, or family)", res["harms"]),
                       ("nominal drops beyond 2 SE, not significant after Holm", res["nominal"])):
        log(f"{name}: {len(rows)}")
        for k, x, y, d, se in rows:
            log(f"  {k:40s} {x:.4f} -> {y:.4f} ({d:+.4f}, 2se {2 * se:.4f})")


def compare(champ_dir, chall_dir, z=2.0):
    a = json.load(open(os.path.join(champ_dir, "eval.json")))
    b = json.load(open(os.path.join(chall_dir, "eval.json")))
    keys = sorted(set(a["validation"]) & set(b["validation"]))
    if not keys:
        raise SystemExit("compare: the two eval.json files share no validation suite")
    va = sum(a["validation"][k] for k in keys) / len(keys)
    vb = sum(b["validation"][k] for k in keys) / len(keys)
    res = heldout_decision(a["heldout"], b["heldout"], z)
    print(f"validation mean: champion {va:.4f} -> challenger {vb:.4f} ({vb - va:+.4f})")
    report(res)
    holds, better = vb >= va - VAL_MARGIN, bool(res["family_gains"])
    ok = holds and not res["harms"] and better
    why = ("" if ok else "(validation fell by more than %.0f point)" % (VAL_MARGIN * 100) if not holds
           else "(held-out regression)" if res["harms"] else "(no family improved significantly)")
    print("VERDICT:", "PROMOTE" if ok else "REJECT", why)
    return ok


def adapter_decision(base, adapter, z=2.0, log=print):
    """The "categories" family decision alone, for a LoRA adapter against its own base model: the same
    GGUF served with and without "adapter" in the request (lora_experiment.py). `base` and `adapter`
    are heldout dicts (heldout_cells of bench/eval_categories.py records); only "categories:" cells
    count. The rule is compare()'s held-out rule: promote when the pooled category family gains more
    than z SE and no cell (language) regresses after Holm-Bonferroni over the compared cells, nor the
    family. There is no validation step: the adapter exists only as a GGUF delta, which eval_dev.py
    cannot load, and it only answers requests that name it, so the base's other suites cannot change.
    Returns heldout_decision's dict plus "promote", "reason" and "p_holm" ({cell: Holm-adjusted
    p-value of a drop})."""
    def cats(h):
        return {k: v for k, v in h.items() if k.startswith("categories:")}
    res = heldout_decision(cats(base), cats(adapter), z, log)
    report(res, log)
    if not res["compared"]:
        ok, why = False, "(no category cell compared)"
    else:
        ok = not res["harms"] and bool(res["family_gains"])
        why = "" if ok else "(held-out regression)" if res["harms"] else "(no family improved significantly)"
    log("VERDICT: " + ("PROMOTE" if ok else "REJECT") + (" " + why if why else ""))
    res.update(promote=ok, reason=why, p_holm=holm_adjusted(res["tests"]))
    return res


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("model")
    e.add_argument("--mixture", default=os.environ.get("STATIM_GATE_MIXTURE"),
                   help="built mixture the model was trained on (jsonl.gz); category suite items that share "
                        "a text with it are dropped (env STATIM_GATE_MIXTURE)")
    c = sub.add_parser("compare")
    c.add_argument("champion")
    c.add_argument("challenger")
    c.add_argument("--z", type=float, default=2.0)
    a = ap.parse_args()
    if a.cmd == "eval":
        evaluate(a.model, a.mixture)
    else:
        sys.exit(0 if compare(a.champion, a.challenger, a.z) else 1)


if __name__ == "__main__":
    main()
