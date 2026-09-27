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
  Emotion validation, AG News train rows): the challenger's mean must be higher. Model selection
  never looks at test splits.
- held-out tests (eval_laya.py on the first 2,000 rows; bench/eval_multilingual.py per language;
  bench/eval_zeroshot.py suites never trained on; bench/eval_categories.py, one suite per decision
  category from the unused splits of the mixture v6 sources): no suite may drop by more than 2 standard errors
  (binomial, sqrt(p(1-p)/n) for each model, combined). That margin separates real regressions
  from sampling noise.
A challenger that improves validation but harms any held-out suite is rejected: it has started to
overfit or to forget, which is exactly the sweet spot this gate protects.
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


def http_suites(model_dir, tmp):
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
        for script, extra in (("bench/eval_multilingual.py", ["--n", "150"]),
                              ("bench/eval_zeroshot.py", ["--n", "150"]),
                              ("bench/eval_categories.py", ["--n", "150"])):
            if not os.path.exists(os.path.join(ROOT, script)):
                continue
            out = os.path.join(tmp, os.path.basename(script) + ".jsonl")
            run([PY, script, "--url", f"http://127.0.0.1:{PORT}", "--model", "multilingual", "--out", out, *extra])
            rows += [json.loads(l) for l in open(out)]
        return rows
    finally:
        srv.send_signal(signal.SIGINT)
        srv.wait(timeout=60)


def evaluate(model_dir):
    tmp = os.path.join(model_dir, "eval-tmp")
    os.makedirs(tmp, exist_ok=True)
    res = {"model": model_dir, "validation": {}, "heldout": {}}
    dev = jsonl(run([PY, "tools/finetune/eval_dev.py", model_dir]))[0]
    res["validation"] = {k: v for k, v in dev.items() if k not in ("model", "seconds", "mean")}
    for r in jsonl(run([PY, "tools/finetune/eval_laya.py", model_dir, "--n", "2000", "--head-max-len", "512"])):
        n = r.get("n", 2000)
        res["heldout"][f"test/{r['suite']}"] = {"acc": r["accuracy"], "n": n, "ece": r.get("ece")}
    for r in http_suites(model_dir, tmp):
        if r.get("lang") == "macro" or "accuracy" not in r:  # macro line, or a skipped language
            continue
        res["heldout"][f"{suite_key(r)}/{r['lang']}"] = {"acc": r["accuracy"], "n": r["n"], "ece": r.get("ece")}
    json.dump(res, open(os.path.join(model_dir, "eval.json"), "w"), indent=1)
    print(json.dumps({"validation_mean": round(sum(res["validation"].values()) / len(res["validation"]), 4),
                      "heldout_suites": len(res["heldout"])}))


def suite_key(record):
    """eval_categories.py names its suites after the category ("sentiment", "nli"); prefix them so they
    cannot collide with a suite of another script."""
    return ("categories:" + record["suite"]) if record.get("family") == "categories" else record["suite"]


ZERO_SHOT = {"go_emotions", "multi_hatecheck", "sib200", "indonli", "farstail", "belebele", "semrel"}


def _var(x):
    return x["acc"] * (1 - x["acc"]) / x["n"]


def compare(champ_dir, chall_dir, z=2.0):
    a = json.load(open(os.path.join(champ_dir, "eval.json")))
    b = json.load(open(os.path.join(chall_dir, "eval.json")))
    keys = sorted(set(a["validation"]) & set(b["validation"]))
    va = sum(a["validation"][k] for k in keys) / len(keys)
    vb = sum(b["validation"][k] for k in keys) / len(keys)
    harms, gains = [], []
    for k in sorted(set(a["heldout"]) & set(b["heldout"])):
        x, y = a["heldout"][k], b["heldout"][k]
        se = math.sqrt(x["acc"] * (1 - x["acc"]) / x["n"] + y["acc"] * (1 - y["acc"]) / y["n"])
        d = y["acc"] - x["acc"]
        if d < -z * se:
            harms.append((k, x["acc"], y["acc"], d, se))
        elif d > z * se:
            gains.append((k, x["acc"], y["acc"], d, se))
    # Family-level check: many small suites can each stay inside their noise band while all drifting
    # the same way. Pool each family (row-weighted and suite-weighted) and treat a significant pooled
    # drop as a regression too.
    families = {
        "trained": lambda k: k.split("/")[0] == "amazon_massive_intent" or k in ("test/banking77", "test/typed_decisions"),
        "zero-shot": lambda k: k.split("/")[0] in ZERO_SHOT or k in ("test/ag_news", "test/emotion"),
        "sentiment": lambda k: k.split("/")[0] == "multilingual_sentiments",
        "categories": lambda k: k.startswith("categories:"),
    }
    common = sorted(set(a["heldout"]) & set(b["heldout"]))
    for fam, member in families.items():
        ks = [k for k in common if member(k)]
        if not ks:
            continue
        bysuite = {}
        for k in ks:
            bysuite.setdefault(k if k.startswith("test/") else k.split("/")[0], []).append(k)
        def pooled(groups):
            ds, var = [], 0.0
            for g in groups:
                ds.append(sum(b["heldout"][k]["acc"] - a["heldout"][k]["acc"] for k in g) / len(g))
                var += sum(_var(a["heldout"][k]) + _var(b["heldout"][k]) for k in g) / len(g) ** 2
            return sum(ds) / len(ds), math.sqrt(var) / len(ds)
        for weighting, groups in (("rows", [[k] for k in ks]), ("suites", list(bysuite.values()))):
            d, se = pooled(groups)
            flag = "REGRESSION" if d < -z * se else ("gain" if d > z * se else "within noise")
            print(f"family {fam:10s} ({weighting:6s}, {len(groups):2d}): {d * 100:+.2f} pts, 2se {2 * se * 100:.2f} -> {flag}")
            if d < -z * se:
                harms.append((f"family:{fam}/{weighting}", 0.0, d, d, se))
    print(f"validation mean: champion {va:.4f} -> challenger {vb:.4f} ({vb - va:+.4f})")
    for name, rows in (("significant gains", gains), ("significant regressions", harms)):
        print(f"{name}: {len(rows)}")
        for k, x, y, d, se in rows:
            print(f"  {k:40s} {x:.4f} -> {y:.4f} ({d:+.4f}, 2se {2 * se:.4f})")
    ok = vb > va and not harms
    print("VERDICT:", "PROMOTE" if ok else "REJECT", "" if ok else
          ("(validation did not improve)" if vb <= va else "(held-out regression)"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("model")
    c = sub.add_parser("compare")
    c.add_argument("champion")
    c.add_argument("challenger")
    c.add_argument("--z", type=float, default=2.0)
    a = ap.parse_args()
    if a.cmd == "eval":
        evaluate(a.model)
    else:
        sys.exit(0 if compare(a.champion, a.challenger, a.z) else 1)


if __name__ == "__main__":
    main()
