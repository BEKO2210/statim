#!/usr/bin/env python3
"""Archive the measured parity of a green CI run.

    ctest --test-dir build --output-junit ctest.xml ...
    python3 tools/ci/parity_summary.py ctest.xml --json parity.json --markdown "$GITHUB_STEP_SUMMARY"

Reads ctest's JUnit file (it keeps the output of passing tests) and extracts, per parity test, the
worst |Δlogit| (`model_parity_*`, `lora_*`), the worst answer difference (`engine_parity_*`) and the
tolerance the test enforced. The JSON goes into the run's artifacts, so the margin to the tolerance
is on record for every green run, not only when a test fails. Exit 1 when a parity test in the file
reports no value. Standard library only.
"""
import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET

PATTERNS = {
    "max_dlogit": re.compile(r"(?:max \|dlogit\||logits vs PyTorch, max \|diff\|) ([0-9.eE+-]+)"),
    "worst_answer_diff": re.compile(r"worst \|diff\| ([0-9.eE+-]+)"),
    "argmax": re.compile(r"argmax agree (\d+/\d+)"),
    "tolerance": re.compile(r"PASS \(tol ([0-9.eE+-]+)\)"),
}


def summarize(junit_text):
    rows = []
    for case in ET.fromstring(junit_text).iter("testcase"):
        name = case.get("name", "")
        if "parity" not in name:
            continue
        out = "".join(e.text or "" for e in case.iter("system-out"))
        row = {"test": name, "status": "failed" if case.find("failure") is not None else "passed"}
        for key, pattern in PATTERNS.items():
            found = pattern.findall(out)
            if found:
                row[key] = max(found, key=lambda v: float(v)) if key in ("max_dlogit", "worst_answer_diff") else found[-1]
        rows.append(row)
    return rows


def markdown(rows):
    lines = ["### Parity of this run", "", "| Test | Status | max \\|Δlogit\\| | worst answer diff | argmax | tolerance |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['test']}` | {r['status']} | {r.get('max_dlogit', '')} | {r.get('worst_answer_diff', '')} "
                     f"| {r.get('argmax', '')} | {r.get('tolerance', '')} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("junit")
    ap.add_argument("--json")
    ap.add_argument("--markdown", help="file to append the table to (e.g. $GITHUB_STEP_SUMMARY)")
    a = ap.parse_args(argv)
    rows = summarize(open(a.junit, encoding="utf-8").read())
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
    if a.markdown:
        open(a.markdown, "a", encoding="utf-8").write(markdown(rows))
    print(markdown(rows))
    empty = [r["test"] for r in rows if not ("max_dlogit" in r or "worst_answer_diff" in r)]
    if empty:
        print("parity tests without a measured value: " + ", ".join(empty), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
