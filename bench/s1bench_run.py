#!/usr/bin/env python3
"""Run S1Bench through levbench against a Statim server and keep every item's record.

    cd <lev checkout at 745535b> && uv run python <statim>/bench/s1bench_run.py \
        --tasks data/s1bench --base-url http://127.0.0.1:8080 --out s1bench-statim.json

levbench's own `eval` command prints a report but stores no per-item results; this wrapper calls
the same `levbench.runner.run_eval` (backend "lev", concurrency 1, timeout 120 s, as Lev's authors
ran Lev and Jev) and writes, per subset, levbench's accuracy and ECE plus every item's predicted
label, truth and top probability, so two runs can be paired item by item.
Protocol: docs/reproductions/s1bench-protocol.md.
"""
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

from levbench import runner
from levbench.tasks import dataset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, type=Path, help="directory written by `lev s1bench export`")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--limit", type=int, default=0, help="items per subset (0: all); for smoke runs only")
    ap.add_argument("--model", default=None,
                    help="request field model (default: levbench's \"local\"); \"consensus\" asks Statim for both checkpoints")
    a = ap.parse_args()

    client, model = runner.build_client("lev", a.model, a.base_url, a.timeout)
    index = json.loads((a.tasks / "index.json").read_text())
    out = {"harness": "levbench run_eval, backend lev, concurrency 1", "base_url": a.base_url,
           "timeout_s": a.timeout, "model_field": a.model or "local", "python": platform.python_version(), "subsets": {}}
    try:
        out["lev_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                           check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        out["lev_commit"] = None
    for name in index["subsets"]:
        items, questions = dataset(a.tasks / f"{name}.json")
        if a.limit:
            items = items[: a.limit]
        report = runner.run_eval(client, "lev", model, items, questions, concurrency=1)
        (qname, cal), = report.per_question.items()
        recs = report.records[qname]
        out["subsets"][name] = {
            "n": cal.n, "accuracy": cal.accuracy, "ece": cal.ece, "served_by": report.model,
            "records": [{"pred": p, "truth": t, "top_p": round(tp, 6)} for _, p, t, tp in recs],
        }
        print(f"{name:22s} n={cal.n:4d} acc={cal.accuracy:.4f} ece={cal.ece:.4f}", flush=True)
        a.out.write_text(json.dumps(out, indent=1, default=str))  # partial results survive an abort
    accs = [s["accuracy"] for s in out["subsets"].values()]
    out["macro_all"] = sum(accs) / len(accs)
    a.out.write_text(json.dumps(out, indent=1, default=str))
    print(f"macro over {len(accs)} subsets: {out['macro_all']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
