#!/usr/bin/env python3
"""Reproducible CPU measurements for the Laya PyTorch reference implementation."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from typing import Any

# The protocol is CPU-only.  Set this before any worker imports torch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
WORKER_RESULT = "__LAYA_COMPARE_RESULT__="
STARTUP_READY = "__LAYA_COMPARE_ANSWER_READY__"
PARITY_ITEMS = 240
PARITY_TOLERANCE = 1e-3

QUICK_START_STATE = {
    "subject": "Duplicate charge on invoice #4411",
    "body": "We were billed twice for March. Please refund the duplicate today or we will cancel.",
}
QUICK_START_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages",
            "sales": "pricing, new contracts",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "refund": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?",
    },
}


def json_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def state_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["state_index"]), []).append(row)
    return [grouped[index] for index in sorted(grouped)]


def state_token_counts(groups: list[list[dict[str, Any]]]) -> list[int]:
    """The ORT document calls a state by the longest sequence in its 8-row batch."""
    return [max(len(row["ids"]) for row in group) for group in groups]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]


def summarize(per_state_ms: list[float], token_counts: list[int]) -> dict[str, Any]:
    """Summarize the 29 short states separately from the single longest state."""
    if not per_state_ms or len(per_state_ms) != len(token_counts):
        raise ValueError("timings and token counts must be non-empty and have equal lengths")
    long_tokens = max(token_counts)
    long_indexes = [i for i, count in enumerate(token_counts) if count == long_tokens]
    if len(long_indexes) != 1:
        raise ValueError("expected exactly one longest state")
    long_index = long_indexes[0]
    short_values = [value for i, value in enumerate(per_state_ms) if i != long_index]
    short_tokens = [count for i, count in enumerate(token_counts) if i != long_index]
    if not short_values:
        raise ValueError("expected at least one short state")
    return {
        "mean_ms": statistics.mean(per_state_ms),
        "p50_ms": statistics.median(per_state_ms),
        "p95_ms": percentile(per_state_ms, 0.95),
        "per_state_ms": per_state_ms,
        "state_token_counts": token_counts,
        "short": {
            "states": len(short_values),
            "max_tokens": max(short_tokens),
            "mean_ms": statistics.mean(short_values),
        },
        "long": {
            "states": 1,
            "state_index": long_index,
            "tokens": long_tokens,
            "median_ms": per_state_ms[long_index],
        },
    }


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="subcommand", required=True)

    parity = commands.add_parser("parity", help="compare Laya logits with stored golden logits")
    parity.add_argument("--model", type=Path, required=True)
    parity.add_argument("--rows", type=Path, required=True)
    parity.add_argument("--out", type=Path, required=True)

    raw = commands.add_parser("raw", help="measure forward-pass-only scoring")
    raw.add_argument("--model", type=Path, required=True)
    raw.add_argument("--rows", type=Path, required=True)
    raw.add_argument("--threads", type=positive_int, nargs="+", default=[1, 4, 8])
    raw.add_argument("--repeats", type=positive_int, default=3)
    raw.add_argument("--out", type=Path, required=True)
    raw.add_argument("--_worker-thread", type=positive_int, help=argparse.SUPPRESS)

    startup = commands.add_parser("startup", help="measure import, load, and first answer")
    startup.add_argument("--model", type=Path, required=True)
    startup.add_argument("--out", type=Path, required=True)
    startup.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def cpu_model() -> str:
    path = Path("/proc/cpuinfo")
    if path.exists():
        match = re.search(r"^model name\s*:\s*(.+)$", path.read_text(errors="replace"), re.MULTILINE)
        if match:
            return match.group(1).strip()
    return platform.processor() or "unknown"


def git_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def metadata(argv: list[str] | None = None) -> dict[str, Any]:
    command_args = sys.argv[1:] if argv is None else argv
    return {
        "command": [sys.executable, str(Path(__file__).resolve()), *command_args],
        "versions": {
            "laya": package_version("laya"),
            "torch": package_version("torch"),
            "numpy": package_version("numpy"),
            "python": platform.python_version(),
        },
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_model": cpu_model(),
        "git_commit": git_commit(),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def load_agent(model: Path):
    import torch
    import laya

    agent = laya.load(str(model.resolve()), device="cpu")
    assert agent.dtype == torch.float32, f"expected float32, got {agent.dtype}"
    assert not agent.amp_enabled, "AMP must be disabled"
    agent.model.eval()
    return agent


def batch_for_group(group: list[dict[str, Any]], pad_id: int):
    from laya.common import collate_items

    items = [{"ids": row["ids"], "markers": row["markers"], "qtype": row["qtype"]}
             for row in group]
    return collate_items([items], pad_id)


def forward(agent, batch):
    return agent.model(
        batch["input_ids"], batch["attention_mask"], batch["marker_pos"],
        batch["marker_mask"], batch["qtype"],
    )


def run_parity(args: argparse.Namespace, argv: list[str] | None) -> int:
    import torch

    rows = json_rows(args.rows)
    groups = state_groups(rows)
    agent = load_agent(args.model)
    agreed = 0
    maximum = 0.0
    with torch.inference_mode():
        for group in groups:
            batch = batch_for_group(group, agent.tok.pad_token_id)
            logits, _ = forward(agent, batch)
            for index, row in enumerate(group):
                expected = torch.tensor(row["logits"], dtype=torch.float32)
                actual = logits[index, :len(expected)].float().cpu()
                maximum = max(maximum, float(torch.max(torch.abs(actual - expected)).item()))
                agreed += int(int(torch.argmax(actual)) == int(torch.argmax(expected)))
    passed = len(rows) == PARITY_ITEMS and agreed == PARITY_ITEMS and maximum <= PARITY_TOLERANCE
    result = {
        **metadata(argv),
        "model": str(args.model),
        "rows": str(args.rows),
        "parity": {
            "items": len(rows),
            "argmax_agree": agreed,
            "required_argmax_agree": PARITY_ITEMS,
            "max_abs_logit": maximum,
            "tolerance": PARITY_TOLERANCE,
            "pass": passed,
        },
    }
    write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if passed else 1


def raw_worker(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    threads = args._worker_thread
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    groups = state_groups(json_rows(args.rows))
    counts = state_token_counts(groups)
    load_started = time.perf_counter()
    agent = load_agent(args.model)
    load_ms = (time.perf_counter() - load_started) * 1000.0
    medians: list[float] = []
    with torch.inference_mode():
        for group in groups:
            batch = batch_for_group(group, agent.tok.pad_token_id)
            forward(agent, batch)  # one untimed warm-up per state
            samples = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                forward(agent, batch)
                samples.append((time.perf_counter() - started) * 1000.0)
            medians.append(statistics.median(samples))
    return {
        "engine": "laya-pytorch",
        "threads": threads,
        "states": len(groups),
        "repeats": args.repeats,
        "warmup_passes_per_state": 1,
        "model_load_ms": load_ms,
        **summarize(medians, counts),
    }


def worker_payload(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(WORKER_RESULT):
            return json.loads(line[len(WORKER_RESULT):])
    raise RuntimeError("worker did not emit a result")


def run_raw(args: argparse.Namespace, argv: list[str] | None) -> int:
    if args._worker_thread is not None:
        print(WORKER_RESULT + json.dumps(raw_worker(args), separators=(",", ":")), flush=True)
        return 0

    results = []
    for threads in args.threads:
        command = [
            sys.executable, str(Path(__file__).resolve()), "raw",
            "--model", str(args.model), "--rows", str(args.rows),
            "--threads", str(threads), "--repeats", str(args.repeats),
            "--out", str(args.out), "--_worker-thread", str(threads),
        ]
        proc = subprocess.run(
            command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        if proc.returncode:
            raise RuntimeError(f"raw worker for {threads} threads failed:\n{proc.stderr[-3000:]}")
        row = worker_payload(proc.stdout)
        row["worker_command"] = command
        results.append(row)
    result = {
        **metadata(argv),
        "model": str(args.model),
        "rows": str(args.rows),
        "raw": results,
    }
    write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def peak_rss_bytes() -> int | None:
    path = Path("/proc/self/status")
    if not path.exists():
        return None
    match = re.search(r"^VmHWM:\s+(\d+)\s+kB$", path.read_text(), re.MULTILINE)
    return int(match.group(1)) * 1024 if match else None


def startup_worker(args: argparse.Namespace) -> int:
    import torch

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    agent = load_agent(args.model)
    with torch.inference_mode():
        answer = agent.system_one(QUICK_START_STATE, QUICK_START_QUESTIONS)
    peak = peak_rss_bytes()
    print(STARTUP_READY, flush=True)
    payload = {"answer": answer, "peak_rss_bytes": peak}
    print(WORKER_RESULT + json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)
    return 0


def directory_size(path: Path) -> int:
    total = 0
    for root, directories, files in os.walk(path):
        for name in directories:
            total += (Path(root) / name).lstat().st_size
        for name in files:
            total += (Path(root) / name).lstat().st_size
    return total


def run_startup(args: argparse.Namespace, argv: list[str] | None) -> int:
    if args._worker:
        return startup_worker(args)

    site_packages = Path(sysconfig.get_paths()["purelib"])
    site_packages_bytes = directory_size(site_packages)
    command = [
        sys.executable, str(Path(__file__).resolve()), "startup",
        "--model", str(args.model), "--out", str(args.out), "--_worker",
    ]
    started = time.perf_counter()
    process = subprocess.Popen(
        command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    before_marker = []
    for line in process.stdout:
        if line.rstrip("\n") == STARTUP_READY:
            answer_ms = (time.perf_counter() - started) * 1000.0
            break
        before_marker.append(line)
    else:
        answer_ms = None
    remaining_stdout, stderr = process.communicate()
    stdout = "".join(before_marker) + remaining_stdout
    if process.returncode or answer_ms is None:
        raise RuntimeError(f"startup worker failed:\n{stderr[-3000:]}\n{stdout[-1000:]}")
    payload = worker_payload(stdout)
    result = {
        **metadata(argv),
        "model": str(args.model),
        "startup": {
            "threads": 8,
            "process_start_to_answer_ms": answer_ms,
            "peak_rss_bytes": payload["peak_rss_bytes"],
            "site_packages": str(site_packages),
            "site_packages_bytes": site_packages_bytes,
            "worker_command": command,
            "answer": payload["answer"],
        },
    }
    write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.subcommand == "parity":
        return run_parity(args, argv)
    if args.subcommand == "raw":
        return run_raw(args, argv)
    if args.subcommand == "startup":
        return run_startup(args, argv)
    raise AssertionError(args.subcommand)


if __name__ == "__main__":
    raise SystemExit(main())
