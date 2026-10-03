#!/usr/bin/env python3
"""LoRA specialist experiment: for each category, train an adapter (train_lora.py), convert it
(tools/convert_lora.py), evaluate the base model and the adapter on that category's held-out suites
(bench/eval_categories.py, --adapter for the second run) and decide with the promotion gate's Holm
rule (gate.adapter_decision). Writes <work>/summary.json and <work>/summary.md.

    python tools/finetune/lora_experiment.py \\
        --base-checkpoint dist/statim-decide-multilingual-base/checkpoint \\
        --base-gguf dist/statim-decide-multilingual-base/statim-decide-multilingual-base-f32.gguf \\
        --mixture data/mixture-v8.jsonl.gz --statim build/statim --work models/lora
    # print the commands only
    python tools/finetune/lora_experiment.py ... --dry-run
    # CPU smoke: a few training steps, small samples
    python tools/finetune/lora_experiment.py ... --categories emotion --device cpu --n 20 \\
        --train-args "--limit-items 64 --dev-items 16 --max-steps 4 --max-tokens 1024"

Per category (default: emotion fact_check sentiment safety pii), in order:
  train    .venv-train/bin/python tools/finetune/train_lora.py <checkpoint> --mixture ... --category C --out <work>/C
           (a subprocess, so a crash in one category leaves the others running; --train-args are passed
           through; --skip-train reuses an existing <work>/C)
  convert  .venv/bin/python tools/convert_lora.py <work>/C -o <work>/C.lora.gguf --base <gguf> --category C --name C
  serve    statim serve -m multilingual=<gguf> --adapter multilingual:C=<work>/C.lora.gguf
  eval     .venv/bin/python bench/eval_categories.py --strict --suites C [--exclude-mixture X] --n N --seed S,
           once without and once
           with --adapter C, into <work>/C.base.jsonl and <work>/C.adapter.jsonl
           (--eval-skip K passes --skip K to both evaluations for a fresh replication suffix)
  decide   gate.adapter_decision on the two files' cells (gate.heldout_cells)

One server per category, not one for all: the server loads every --adapter at startup and refuses to
start if one is unusable, so a single broken adapter would block every category; merge mode (the
default on f32 weights) keeps a merged copy of the adapted matrices per adapter (438 MB for the
multilingual model), so five adapters in one process would need about 2.2 GB more; and the base run
and the adapter run of a category still come from one process, so they differ only in the "adapter"
field. The price is one model load per category (seconds).

--exclude-mixture defaults to --mixture: the adapter is trained on rows of that file, so held-out
items that share a text with it are dropped for both runs (same file, same pool fingerprint). If the
base was trained on another mixture, pass one file that holds both (eval_categories takes one path),
or "none" to exclude nothing. With --skip-train, the default is recovered from each adapter's
train_lora.json and its recorded SHA-256 is verified before evaluation.

The decision per category: PROMOTE when the paired pooled category gain survives Holm correction
over all available family weighting tests and no language cell regresses after exact McNemar tests
with Holm-Bonferroni (family-wise 5 %); per-cell gains use Holm too. See gate.adapter_decision. Adapter
answers use the base model's temperatures (an adapter GGUF carries none), so NLL and ECE of the
adapter run are uncalibrated; accuracy, which the decision uses, is not affected.
"""
import argparse
import datetime
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import gate  # noqa: E402
import train_lora  # noqa: E402  (standard library only at import time)

DEFAULT_CATEGORIES = ["emotion", "fact_check", "sentiment", "safety", "pii"]
MODEL = "multilingual"  # the -m name; eval_categories --model accepts it, so requests name the model exactly
DEFAULT_N = 150          # bench/eval_categories.DEFAULT_N
DEFAULT_SEED = 20260927  # bench/eval_categories.DEFAULT_SEED


def default_python(venv):
    path = os.path.join(ROOT, venv, "bin", "python")
    return path if os.path.exists(path) else sys.executable


def adapter_mixture(adapter_dir, supplied=None):
    """Verified training mixture recorded beside an adapter reused with --skip-train."""
    path = os.path.join(adapter_dir, "train_lora.json")
    if not os.path.exists(path):
        raise SystemExit("--skip-train: no training record in %s" % adapter_dir)
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    expected = record.get("mixture_sha256")
    mixture = supplied or record.get("mixture")
    if not mixture or not expected:
        raise SystemExit("--skip-train: %s does not record the mixture path and SHA-256" % path)
    mixture = os.path.abspath(mixture)
    if not os.path.isfile(mixture):
        raise SystemExit("--skip-train: recorded mixture is missing: %s" % mixture)
    actual = train_lora.sha256_file(mixture)
    if actual != expected:
        raise SystemExit("--skip-train: mixture SHA-256 differs for %s (recorded %s, got %s)" %
                         (mixture, expected, actual))
    return mixture


def plan(a, category):
    """Paths and commands for one category (lists for subprocess)."""
    work = a.work
    adapter_dir = os.path.join(work, category)
    gguf = os.path.join(work, category + ".lora.gguf")
    exclude = (None if (a.exclude_mixture or "").lower() == "none" else a.exclude_mixture) if a.exclude_mixture \
        else (adapter_mixture(adapter_dir, a.mixture) if a.skip_train else a.mixture)
    evaluate = [a.tools_python, os.path.join(ROOT, "bench", "eval_categories.py"), "--strict", "--url",
                "http://127.0.0.1:%d" % a.port,
                "--model", MODEL, "--suites", category, "--n", str(a.n), "--skip", str(a.eval_skip),
                "--seed", str(a.seed)]
    if exclude:
        evaluate += ["--exclude-mixture", exclude]
    p = {
        "category": category, "adapter_dir": adapter_dir, "gguf": gguf,
        "base_jsonl": os.path.join(work, category + ".base.jsonl"),
        "adapter_jsonl": os.path.join(work, category + ".adapter.jsonl"),
        "base_items": os.path.join(work, category + ".base-items.jsonl"),
        "adapter_items": os.path.join(work, category + ".adapter-items.jsonl"),
        "train_log": os.path.join(work, category + ".train.log"),
        "server_log": os.path.join(work, category + ".server.log"),
        "exclude_mixture": exclude,
        "train": [a.train_python, os.path.join(HERE, "train_lora.py"), a.base_checkpoint, "--mixture", a.mixture or "",
                  "--category", category, "--registry", a.registry, "--out", adapter_dir, "--device", a.device]
        + shlex.split(a.train_args or ""),
        "convert": [a.tools_python, os.path.join(ROOT, "tools", "convert_lora.py"), adapter_dir, "-o", gguf,
                    "--base", a.base_gguf, "--category", category, "--name", category],
        "serve": [a.statim, "serve", "-m", "%s=%s" % (MODEL, a.base_gguf), "--adapter",
                  "%s:%s=%s" % (MODEL, category, gguf), "--device", a.server_device, "--threads", str(a.threads),
                  "--port", str(a.port), "--no-access-log", "--inference-timeout", str(a.inference_timeout)],
    }
    p["eval_base"] = evaluate + ["--out", p["base_jsonl"], "--predictions", p["base_items"]]
    p["eval_adapter"] = evaluate + ["--adapter", category, "--out", p["adapter_jsonl"],
                                    "--predictions", p["adapter_items"]]
    return p


def records(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _category_items(path):
    grouped = {}
    for row in records(path):
        grouped.setdefault("categories:%s/%s" % (row["suite"], row["lang"]), []).append(row)
    return grouped


def decide(base_jsonl, adapter_jsonl, base_items=None, adapter_items=None, z=2.0, log=print):
    """gate.adapter_decision on two eval_categories.py outputs; a JSON-ready result with one row per
    compared cell (language)."""
    base_records, adapter_records = records(base_jsonl), records(adapter_jsonl)
    def realized(rows):
        return {(r.get("suite"), r.get("lang")): r.get("pool_items_sha256")
                for r in rows if "accuracy" in r and r.get("lang") != "macro"}

    if realized(base_records) != realized(adapter_records):
        raise ValueError("base and adapter evaluations used different realized pool items")
    scored = [r for r in base_records + adapter_records if "accuracy" in r and r.get("lang") != "macro"]
    if any(r.get("strict") is not True for r in scored):
        raise ValueError("a non-strict category evaluation is report-only and cannot promote an adapter")
    for field in ("registry_sha256", "suite_definition_sha256"):
        values = {r.get(field) for r in scored}
        if None in values or len(values) != 1:
            raise ValueError(f"base and adapter evaluations differ or lack {field}")
    if gate.skipped_cells(base_records) != gate.skipped_cells(adapter_records):
        raise ValueError("base and adapter evaluations skipped different category cells")
    base, base_reported = gate.heldout_cells(base_records)
    adapter, adapter_reported = gate.heldout_cells(adapter_records)
    base_items = base_items or str(base_jsonl).replace(".jsonl", "-items.jsonl")
    adapter_items = adapter_items or str(adapter_jsonl).replace(".jsonl", "-items.jsonl")
    res = gate.adapter_decision(base, adapter, _category_items(base_items), _category_items(adapter_items),
                                z=z, log=log)
    regress = {t["name"] for t in res["harms"]}
    cells = []
    for t in res["tests"]:
        k, x, y, d, se = t["name"], t["a"], t["b"], t["d"], t["se"]
        verdict = "regression (Holm)" if k in regress else (
            "gain (Holm)" if k in {g["name"] for g in res["gains"]} else "within noise")
        cells.append({"cell": k, "lang": k.rsplit("/", 1)[1], "n_base": base[k]["n"], "n_adapter": adapter[k]["n"],
                      "base_acc": x, "adapter_acc": y, "delta": round(d, 4), "se": round(se, 4),
                      "z": round(d / se, 2) if se > 0 else None, "p_drop": t["p_drop"],
                      "p_drop_holm": res["p_holm"][k], "p_gain_holm": res["p_gain_holm"][k],
                      "verdict": verdict})
    fam = res["families"].get("categories", {})
    skipped = [r for r in base_records if r.get("skipped")]
    return {"promote": res["promote"], "reason": res["reason"] or "", "cells": cells,
            "family": {w: {"delta": round(v["d"], 4), "se": round(v["se"], 4), "groups": v["groups"], "flag": v["flag"]}
                       for w, v in fam.items()},
            "not_compared": res["not_compared"], "harms": [t["name"] for t in res["harms"]],
            "reported": {"base": base_reported, "adapter": adapter_reported},
            "skipped": [{"suite": r["suite"], "lang": r["lang"], "why": r["skipped"]} for r in skipped],
            "pool": sorted({v.get("pool") for v in base.values()} | {v.get("pool") for v in adapter.values()} - {None})}


def run(cmd, log_path=None):
    """Run a command; its output goes to log_path (appended) or to this process's stdout."""
    print("$ " + shlex.join(cmd), flush=True)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as log:
            log.write("$ " + shlex.join(cmd) + "\n")
            log.flush()
            return subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    return subprocess.run(cmd, cwd=ROOT).returncode


class Server:
    """statim serve for one category; waits for /health, stops on exit."""

    def __init__(self, cmd, port, log_path, timeout=600):
        self.cmd, self.port, self.log_path, self.timeout = cmd, port, log_path, timeout

    def __enter__(self):
        print("$ " + shlex.join(self.cmd), flush=True)
        self.log = open(self.log_path, "w", encoding="utf-8")
        env = dict(os.environ)
        env.pop("STATIM_API_KEY", None)  # eval_categories would send it; the local server runs without keys
        self.proc = subprocess.Popen(self.cmd, cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT, env=env)
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            if self.proc.poll() is not None:
                raise RuntimeError("statim exited with %s before listening (see %s)" % (self.proc.returncode, self.log_path))
            try:
                urllib.request.urlopen("http://127.0.0.1:%d/health" % self.port, timeout=2)
                self.startup = round(time.time() - t0, 1)
                return self
            except OSError:
                time.sleep(0.5)
        raise RuntimeError("statim did not answer /health within %d s (see %s)" % (self.timeout, self.log_path))

    def __exit__(self, *exc):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.log.close()
        return False


def train_record(adapter_dir):
    path = os.path.join(adapter_dir, "train_lora.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        r = json.load(f)
    return {k: r.get(k) for k in ("items", "best", "dev_before", "updates", "seconds", "peak_memory_mb", "device",
                                   "base_sha256", "mixture", "mixture_sha256", "lora")}


def run_category(a, p):
    """Train, convert, serve, evaluate and decide one category. Returns its summary entry."""
    cat = p["category"]
    out = {"status": "failed", "stage": None, "commands": {}}
    for stage in ("train", "convert", "serve", "eval_base", "eval_adapter"):
        out["commands"][stage] = shlex.join(p[stage])
    if a.skip_train:
        if not os.path.exists(os.path.join(p["adapter_dir"], "adapter_config.json")):
            out.update(stage="train", error="--skip-train: no adapter in %s" % p["adapter_dir"])
            return out
    else:
        code = run(p["train"], p["train_log"])
        if code:
            out.update(stage="train", error="train_lora.py exited with %d (see %s)" % (code, p["train_log"]))
            return out
    out["train"] = train_record(p["adapter_dir"])
    conv = subprocess.run(p["convert"], cwd=ROOT, capture_output=True, text=True)
    print("$ " + shlex.join(p["convert"]) + "\n" + conv.stdout + conv.stderr, end="", flush=True)
    if conv.returncode:
        out.update(stage="convert", error=(conv.stderr or conv.stdout).strip()[-500:])
        return out
    out["convert"] = conv.stdout.strip()
    try:
        with Server(p["serve"], a.port, p["server_log"], a.server_timeout) as srv:
            out["server_startup_seconds"] = srv.startup
            for stage in ("eval_base", "eval_adapter"):
                t0 = time.time()
                code = run(p[stage])
                out[stage + "_seconds"] = round(time.time() - t0, 1)
                if code:
                    out.update(stage=stage, error="eval_categories.py exited with %d" % code)
                    return out
    except RuntimeError as exc:
        out.update(stage="serve", error=str(exc))
        return out
    print("--- decision for %s" % cat, flush=True)
    out.update(decide(p["base_jsonl"], p["adapter_jsonl"], p["base_items"], p["adapter_items"], a.z))
    if ((out.get("train") or {}).get("best") or {}).get("saved_initial"):
        out.update(promote=False, reason="(saved adapter is the initial zero-delta adapter: no gain)")
    out.update(status="ok", stage="done")
    return out


def pct(x):
    return "%.1f" % (100 * x)


def summary_md(summary):
    """Markdown report of a summary dict (summary.json)."""
    s = summary
    lines = ["# LoRA specialists: base vs adapter", "",
             "- base checkpoint: `%s`" % s["base_checkpoint"], "- base GGUF: `%s`" % s["base_gguf"],
             "- training mixture: `%s`" % s["mixture"], "- excluded from the suites: `%s`" % s["exclude_mixture"],
             "- sample: n=%d draw per language cell, skip %d, seed %d; decision: gate.adapter_decision (paired "
             "pooled category gain with Holm correction, no paired exact Holm-significant cell regression, "
             "family-wise %g)" % (
                 s["n"], s.get("eval_skip", 0), s["seed"], s["alpha"]),
             "- created %s" % s["created"], ""]
    lines += ["| category | promote | cells | pooled delta (pts) | 2 SE (pts) | reason |", "|---|---|---|---|---|---|"]
    for cat, r in s["categories"].items():
        if r.get("status") != "ok":
            lines.append("| %s | failed (%s) | | | | %s |" % (cat, r.get("stage"), (r.get("error") or "").replace("|", "/")[:160]))
            continue
        fam = r["family"].get("rows") or {}
        lines.append("| %s | %s | %d | %+.2f | %.2f | %s |" % (
            cat, "yes" if r["promote"] else "no", len(r["cells"]), 100 * fam.get("delta", 0.0),
            200 * fam.get("se", 0.0), r["reason"] or "gain, no regression"))
    for cat, r in s["categories"].items():
        lines += ["", "## %s" % cat, ""]
        if r.get("status") != "ok":
            lines += ["Failed at %s: %s" % (r.get("stage"), r.get("error")), ""]
        else:
            t = r.get("train") or {}
            if t:
                best, before = t.get("best") or {}, t.get("dev_before") or {}
                lines += ["Training: %s train / %s dev items, %s updates, dev accuracy %s before -> %s (best at epoch %s, "
                          "update %s), %s s on %s, peak %s MB." % (
                              (t.get("items") or {}).get("train"), (t.get("items") or {}).get("dev"), t.get("updates"),
                              before.get("dev_acc"), best.get("dev_acc"), best.get("epoch"), best.get("update"),
                              t.get("seconds"), t.get("device"), t.get("peak_memory_mb")), ""]
                if best.get("saved_initial"):
                    lines += ["The saved adapter is the initial zero-delta adapter; this category has no gain.", ""]
            lines += ["Converted: %s" % r.get("convert"), "",
                      "| lang | n | base acc | adapter acc | delta (pts) | 2 SE (pts) | p (drop) | Holm p (drop) | Holm p (gain) | verdict |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
            for c in r["cells"]:
                lines.append("| %s | %d | %.4f | %.4f | %+.1f | %.1f | %.3g | %.3g | %.3g | %s |" % (
                    c["lang"], c["n_base"], c["base_acc"], c["adapter_acc"], 100 * c["delta"], 200 * c["se"],
                    c["p_drop"], c["p_drop_holm"], c["p_gain_holm"], c["verdict"]))
            for w, f in r["family"].items():
                lines.append("")
                lines.append("Family (%s, %d groups): %+.2f pts, 2 SE %.2f pts -> %s." % (
                    w, f["groups"], 100 * f["delta"], 200 * f["se"], f["flag"]))
            if r.get("not_compared"):
                lines += ["", "Not compared (pools differ): %s" % ", ".join(r["not_compared"])]
            if r.get("skipped"):
                lines += ["", "Skipped cells: %s" % "; ".join("%s/%s (%s)" % (x.get("suite", "?"), x["lang"], x["why"])
                                                          for x in r["skipped"])]
            lines += ["", "**Verdict: %s** %s" % ("PROMOTE" if r["promote"] else "REJECT", r["reason"])]
        lines += ["", "Commands:", "", "```"] + [r["commands"][k] for k in r.get("commands", {})] + ["```"]
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-checkpoint", required=True, help="Laya checkpoint directory of the base model")
    ap.add_argument("--base-gguf", required=True, help="that checkpoint's Statim GGUF (f32 or quantized)")
    ap.add_argument("--mixture", help="built mixture the adapters are trained on (jsonl.gz)")
    ap.add_argument("--registry", default=os.path.join(HERE, "sources", "v6-keep.json"))
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES,
                    choices=train_lora.CATEGORIES, metavar="CATEGORY")
    ap.add_argument("--work", default=os.path.join("models", "lora"),
                    help="adapters, GGUFs, eval records, logs and the summary (default models/lora)")
    ap.add_argument("--statim", default=os.path.join(ROOT, "build", "statim"), help="statim binary with LoRA support")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="training device (train_lora.py)")
    ap.add_argument("--server-device", default="cpu", help="statim serve --device (cpu, gpu, vulkan)")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4, help="statim serve --threads")
    ap.add_argument("--port", type=int, default=8098)
    ap.add_argument("--inference-timeout", type=int, default=600,
                    help="statim serve --inference-timeout (s); a 16-state batch of long texts can exceed the "
                         "default 120 s on a CPU")
    ap.add_argument("--server-timeout", type=int, default=600, help="seconds to wait for the server to listen")
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="eval_categories --n (items per language cell)")
    ap.add_argument("--eval-skip", type=int, default=0,
                    help="eval_categories --skip (drop this previously scored prefix from each draw)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="eval_categories --seed")
    ap.add_argument("--exclude-mixture", default=None,
                    help='eval_categories --exclude-mixture (default: --mixture; "none" to exclude nothing)')
    ap.add_argument("--z", type=float, default=2.0, help="standard errors for a gain (gate default)")
    ap.add_argument("--skip-train", action="store_true", help="reuse the adapters in <work>/<category>")
    ap.add_argument("--train-args", default="", help="extra train_lora.py arguments, one shell-quoted string")
    ap.add_argument("--train-python", default=None, help="training interpreter (default: .venv-train/bin/python)")
    ap.add_argument("--tools-python", default=None, help="conversion and evaluation interpreter (default: .venv/bin/python)")
    ap.add_argument("--python", default=None, help="deprecated alias that sets --train-python and --tools-python")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    a = ap.parse_args(argv)
    if a.python and (a.train_python or a.tools_python):
        ap.error("--python cannot be combined with --train-python or --tools-python")
    if a.python:
        print("WARNING: --python is deprecated; use --train-python and --tools-python.", file=sys.stderr)
        a.train_python = a.tools_python = a.python
    else:
        a.train_python = a.train_python or default_python(".venv-train")
        a.tools_python = a.tools_python or default_python(".venv")
    if not a.skip_train and not a.mixture:
        ap.error("--mixture is required unless --skip-train")
    if a.eval_skip < 0 or a.eval_skip >= a.n:
        ap.error("--eval-skip must be >= 0 and less than --n")
    if (a.exclude_mixture or "").lower() == "none":
        print("WARNING: --exclude-mixture none disables training-overlap filtering; results are not held out.",
              file=sys.stderr)
    # the subprocesses run in the repository root: make the user's paths absolute
    for key in ("base_checkpoint", "base_gguf", "mixture", "registry", "work", "statim", "exclude_mixture"):
        value = getattr(a, key)
        if value and not (key == "exclude_mixture" and value.lower() == "none"):
            setattr(a, key, os.path.abspath(value))
    return a


def main(argv=None):
    a = parse_args(argv)
    plans = [plan(a, c) for c in a.categories]
    if a.dry_run:
        for p in plans:
            print("# %s" % p["category"])
            for stage in ("train", "convert", "serve", "eval_base", "eval_adapter"):
                if stage == "train" and a.skip_train:
                    continue
                print(("  (background) " if stage == "serve" else "  ") + shlex.join(p[stage]))
        return 0
    os.makedirs(a.work, exist_ok=True)
    summary = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
               "base_checkpoint": os.path.abspath(a.base_checkpoint), "base_gguf": os.path.abspath(a.base_gguf),
               "mixture": a.mixture and os.path.abspath(a.mixture), "exclude_mixture": plans[0]["exclude_mixture"],
               "n": a.n, "eval_skip": a.eval_skip, "seed": a.seed, "z": a.z, "alpha": gate.ALPHA,
               "argv": sys.argv, "categories": {}}
    t0 = time.time()
    for p in plans:
        print("=== %s" % p["category"], flush=True)
        t = time.time()
        try:
            entry = run_category(a, p)
        except Exception as exc:  # one category must not end the experiment
            entry = {"status": "failed", "stage": "runner", "error": "%s: %s" % (type(exc).__name__, exc)}
        entry["seconds"] = round(time.time() - t, 1)
        summary["categories"][p["category"]] = entry
        summary["seconds"] = round(time.time() - t0, 1)
        with open(os.path.join(a.work, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1, ensure_ascii=False)
        with open(os.path.join(a.work, "summary.md"), "w", encoding="utf-8") as f:
            f.write(summary_md(summary))
    print("wrote %s and %s" % (os.path.join(a.work, "summary.json"), os.path.join(a.work, "summary.md")), flush=True)
    for cat, r in summary["categories"].items():
        print("%-10s %s" % (cat, ("PROMOTE" if r.get("promote") else "REJECT " + (r.get("reason") or ""))
                             if r.get("status") == "ok" else "FAILED at %s: %s" % (r.get("stage"), r.get("error"))))
    return 0 if all(r.get("status") == "ok" for r in summary["categories"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
