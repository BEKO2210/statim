#!/usr/bin/env python3
"""Paired performance and fidelity gate for two Statim builds.

The portable bundle needed by ``--ssh`` is deliberately small: this script and
``bench/ort_compare.py``; ``statim`` and ``test_model_parity`` from each build;
the two GGUF files; ``build-ort/golden_v9.jsonl``;
``tests/data/golden_inputs.json``; ``models/laya-multilingual-v9/``'s
``rl_agent_config.json`` and ``tokenizer/tokenizer.json``; and a Linux x86-64
venv containing tokenizers, numpy and psutil.  The venv defaults to
``build-ort/venv-gate`` (``python3 -m venv build-ort/venv-gate && build-ort/venv-gate/bin/pip
install tokenizers==0.23.2 numpy==2.5.3 psutil==7.2.2``) and may be selected with ``--venv``.
The remote host needs the same Python minor version, because the venv is copied.

Reference tolerances are 1e-3 for f32 and 0.35 for the shipped q8_0 model.

Run on an otherwise idle CPU-only x86-64 Linux host.  Baseline and candidate
measurements are interleaved within every cell.  ``test_model_parity`` must have
the optional STATIM_PARITY_DUMP support in this checkout.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = (ROOT / "models/laya-multilingual-v9-f32.gguf",
                  ROOT / "build-ort/laya-multilingual-v9-q8_0.gguf")
GOLDEN = ROOT / "build-ort/golden_v9.jsonl"
INPUTS = ROOT / "tests/data/golden_inputs.json"
MODEL_DIR = ROOT / "models/laya-multilingual-v9"
VARIANT_TOLERANCE = {"f32": 1e-3, "q8_0": 0.35}


def timing_decision(baseline: list[float], candidate: list[float], tolerance: float,
                    noise_multiplier: float = 3.0, noise_floor: float = 0.0) -> dict:
    """Classify paired lower-is-better samples; exported for the unit tests."""
    if not baseline or len(baseline) != len(candidate) or any(v <= 0 for v in baseline):
        raise ValueError("equal, non-empty timing samples with positive baseline values required")
    base = statistics.median(baseline)
    cand = statistics.median(candidate)
    noise = max(noise_floor, max(abs(v - base) for v in baseline) / base)
    delta = (cand - base) / base
    limit = max(tolerance, noise_multiplier * noise)
    paired = [(c - b) / b for b, c in zip(baseline, candidate)]
    slower_consistent = all(value > limit for value in paired)
    faster_consistent = all(value < -limit for value in paired)
    if delta > limit and slower_consistent:
        verdict = "slower"
    elif -delta > limit and faster_consistent:
        verdict = "faster"
    else:
        verdict = "same"
    return {"baseline": base, "candidate": cand, "delta_fraction": delta,
            "noise_fraction": noise, "limit_fraction": limit,
            "consistent": slower_consistent if delta >= 0 else faster_consistent,
            "verdict": verdict}


def fixed_decision(baseline: list[float], candidate: list[float], tolerance: float) -> dict:
    result = timing_decision(baseline, candidate, tolerance, 0.0)
    if result["delta_fraction"] > tolerance:
        result["verdict"] = "slower"
    elif result["delta_fraction"] < -tolerance:
        result["verdict"] = "faster"
    else:
        result["verdict"] = "same"
    result["noise_fraction"] = max(abs(v - statistics.median(baseline))
                                   for v in baseline) / statistics.median(baseline)
    return result


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def groups(data: list[dict]) -> list[list[dict]]:
    by_state: dict[int, list[dict]] = {}
    for row in data:
        by_state.setdefault(int(row["state_index"]), []).append(row)
    return [by_state[key] for key in sorted(by_state)]


def environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    return {**os.environ, "CUDA_VISIBLE_DEVICES": "", "STATIM_DEVICE": "cpu",
            "STATIM_VERBOSE": "1", **(extra or {})}


def parse_env(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value or not value.split("=", 1)[0]:
            raise ValueError(f"environment entry must be KEY=VALUE: {value!r}")
        key, val = value.split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"invalid environment key: {key!r}")
        result[key] = val
    return result


def write_repeated(data: list[dict], scratch: Path) -> None:
    state_groups = groups(data)
    output = []
    for pass_index in range(2):                 # one warm-up, one measured pass
        for state_index, group in enumerate(state_groups):
            for source in group:
                output.append({**source, "state_index": pass_index * len(state_groups) + state_index})
    scratch.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in output))


def parity_run(binary: Path, model: Path, data: list[dict], threads: int,
               scratch_dir: Path, tag: str, extra_env: dict[str, str]) -> dict:
    source = scratch_dir / f"{tag}.jsonl"
    dump = scratch_dir / f"{tag}-dump.jsonl"
    write_repeated(data, source)
    command = [str(binary), str(model), str(source), "1e9", str(threads)]
    proc = subprocess.run(command, cwd=ROOT, env=environment({**extra_env, "STATIM_PARITY_DUMP": str(dump)}),
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    state_count = len(groups(data))
    times = [float(v) for v in re.findall(r"state\s+\d+:.*?([0-9.]+) ms,", proc.stdout)]
    summary = re.search(r"items (\d+) \| argmax agree (\d+)/(\d+) \| max \|dlogit\| ([0-9.eE+-]+) "
                        r"\| max \|dact\| ([0-9.eE+-]+)", proc.stdout)
    if len(times) != state_count * 2 or not summary or not dump.exists():
        raise RuntimeError(f"could not parse {binary} output:\n{proc.stdout[-3000:]}")
    dumped = rows(dump)
    if len(dumped) != len(data) * 2:
        raise RuntimeError(f"{binary} dumped {len(dumped)} items, expected {len(data) * 2}; rebuild it")
    return {"per_state_ms": times[state_count:], "outputs": dumped[len(data):],
            "parity": {"items": int(summary.group(1)) // 2,
                       "argmax_agree": int(summary.group(2)) // 2,
                       "max_abs_logit": float(summary.group(4)),
                       "max_abs_act": float(summary.group(5))},
            "command": command}


def direct_fidelity(reference_rows: list[dict], baseline: dict, candidate: dict,
                    tolerance: float) -> dict:
    max_direct = 0.0
    losses = 0
    base_agree = cand_agree = 0
    for ref, base, cand in zip(reference_rows, baseline["outputs"], candidate["outputs"]):
        rlog, blog, clog = ref["logits"], base["logits"], cand["logits"]
        if len(blog) != len(clog):
            return {"pass": False, "error": "baseline/candidate logit shape differs"}
        max_direct = max(max_direct, *(abs(float(a) - float(b)) for a, b in zip(blog, clog)))
        ri = max(range(len(rlog)), key=lambda i: rlog[i])
        bi = max(range(len(blog)), key=lambda i: blog[i])
        ci = max(range(len(clog)), key=lambda i: clog[i])
        base_agree += bi == ri
        cand_agree += ci == ri
        losses += bi == ri and ci != ri
    cp = candidate["parity"]
    passed = losses == 0 and cp["max_abs_logit"] <= tolerance
    return {"items": len(reference_rows), "baseline_argmax_agree": base_agree,
            "candidate_argmax_agree": cand_agree, "candidate_argmax_losses": losses,
            "candidate_max_abs_logit_vs_reference": cp["max_abs_logit"],
            "candidate_max_abs_logit_vs_baseline": max_direct,
            "tolerance": tolerance, "pass": passed}


def add_baseline_reference(data: list[dict], output: list[dict]) -> None:
    for row, actual in zip(data, output):
        row["logits"] = actual["logits"]
        row["act"] = actual["act"]


def benchmark_cell(base_binary: Path, cand_binary: Path, model: Path, data: list[dict],
                   threads: int, rounds: int, tolerance: float, candidate_env: dict[str, str],
                   scratch: Path, tag: str, seed_reference: bool = False) -> tuple[list[list[float]], list[list[float]], list[dict]]:
    base_rounds, cand_rounds, fidelity = [], [], []
    for round_index in range(rounds):
        base = parity_run(base_binary, model, data, threads, scratch,
                          f"{tag}-r{round_index}-a", {})
        if seed_reference and round_index == 0:
            add_baseline_reference(data, base["outputs"])
        cand = parity_run(cand_binary, model, data, threads, scratch,
                          f"{tag}-r{round_index}-b", candidate_env)
        base_rounds.append(base["per_state_ms"])
        cand_rounds.append(cand["per_state_ms"])
        fidelity.append(direct_fidelity(data, base, cand, tolerance))
    return base_rounds, cand_rounds, fidelity


def metric_row(section: str, name: str, baseline: list[float], candidate: list[float],
               tolerance: float, unit: str = "ms", noisy: bool = True,
               noise_floor: float = 0.0, key: tuple | None = None) -> dict:
    decision = timing_decision(baseline, candidate, tolerance, noise_floor=noise_floor) if noisy else fixed_decision(baseline, candidate, tolerance)
    row = {"section": section, "cell": name, "unit": unit,
           "baseline_rounds": baseline, "candidate_rounds": candidate, **decision}
    if key is not None:
        row["cell_key"] = list(key)
    return row


def raw_rows(args, variant: str, threads: int, golden_rows: list[dict], br: list[list[float]],
             cr: list[list[float]]) -> list[dict]:
    state_groups = groups(golden_rows)
    lengths = [max(len(r["ids"]) for r in group) for group in state_groups]
    output = []
    key = ("raw", variant, threads)
    # A state can land on a different high-thread scheduling wave even in identical
    # binaries.  The cell's baseline A-vs-A spread is therefore a shared noise floor
    # for its per-state results; otherwise a lucky pair of identical samples creates
    # a false zero-noise claim for that state.
    cell_noise = max(
        max(abs(values[i] - statistics.median(r[i] for r in br))
            for values in br) / statistics.median(r[i] for r in br)
        for i in range(len(state_groups)))
    for index in range(len(state_groups)):
        output.append(metric_row("raw", f"{variant} t{threads} state {index}",
                                 [r[index] for r in br], [r[index] for r in cr], args.tolerance,
                                 noise_floor=cell_noise, key=key))
    for label, predicate in (("short", lambda n: n <= 256), ("long", lambda n: n > 256)):
        indexes = [i for i, length in enumerate(lengths) if predicate(length)]
        if indexes:
            output.append(metric_row("raw", f"{variant} t{threads} {label} mean",
                [statistics.mean(r[i] for i in indexes) for r in br],
                [statistics.mean(r[i] for i in indexes) for r in cr], args.tolerance, key=key))
    return output


def measure_raw(args, variant: str, model: Path, threads: int, golden_rows: list[dict],
                scratch: Path, rounds: int, tag: str) -> tuple[list[list[float]], list[list[float]], list[dict]]:
    return benchmark_cell(args.baseline / "test_model_parity", args.candidate / "test_model_parity", model,
                          golden_rows, threads, rounds, VARIANT_TOLERANCE[variant], args.candidate_env,
                          scratch, tag)


def raw_section(args, models: dict[str, Path], golden_rows: list[dict], scratch: Path) -> tuple[list[dict], list[dict], dict]:
    output, fidelity, samples = [], [], {}
    for variant, model in models.items():
        for threads in args.threads:
            br, cr, checks = measure_raw(args, variant, model, threads, golden_rows, scratch,
                                         args.rounds, f"raw-{variant}-t{threads}")
            samples[("raw", variant, threads)] = (br, cr)
            output += raw_rows(args, variant, threads, golden_rows, br, cr)
            fidelity.append({"cell": f"raw {variant} t{threads}",
                             "rounds": checks, "pass": all(c["pass"] for c in checks)})
    return output, fidelity, samples


def make_length_batches(lengths: list[int]) -> tuple[dict[int, list[dict]], list[dict]]:
    # Use the exact builders from ort_compare, but no ONNX Runtime/model is needed: references
    # are populated from the baseline's first pass (the gate analogue of add_reference).
    from tokenizers import Tokenizer
    from ort_compare import ORTEngine, length_sweep_rows
    engine = ORTEngine.__new__(ORTEngine)
    engine.cfg = json.loads((MODEL_DIR / "rl_agent_config.json").read_text())
    engine.tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer/tokenizer.json"))
    engine.ids = {name: engine.tok.token_to_id(token) for name, token in
                  {"pad": "<pad>", "cls": "<bos>", "sep": "<eos>", "mask": "<mask>"}.items()}
    batches, skipped = length_sweep_rows(engine, INPUTS, lengths)
    for batch in batches.values():
        for row in batch:
            row["logits"] = [0.0] * len(row["markers"])
            row["act"] = [0.0, 0.0]
    return batches, skipped


def sweep_section(args, models: dict[str, Path], scratch: Path) -> tuple[list[dict], list[dict], list[dict], dict]:
    batches, skipped = make_length_batches(args.lengths)
    output, fidelity, samples = [], [], {}
    for variant, model in models.items():
        for threads in args.sweep_threads:
            for length, batch in batches.items():
                data = [dict(row) for row in batch]
                br, cr, checks = benchmark_cell(args.baseline / "test_model_parity",
                    args.candidate / "test_model_parity", model, data, threads, args.rounds,
                    VARIANT_TOLERANCE[variant], args.candidate_env, scratch,
                    f"sweep-{variant}-l{length}-t{threads}", seed_reference=True)
                key = ("sweep", variant, length, threads)
                samples[key] = (br, cr, data)
                output.append(metric_row("length sweep", f"{variant} L{length} t{threads}",
                                         [r[0] for r in br], [r[0] for r in cr], args.tolerance, key=key))
                fidelity.append({"cell": f"length sweep {variant} L{length} t{threads}",
                                 "rounds": checks, "pass": all(c["pass"] for c in checks)})
    return output, fidelity, skipped, samples


def confirm_slower(args, models: dict[str, Path], golden_rows: list[dict], scratch: Path,
                   measurements: list[dict], raw_samples: dict, sweep_samples: dict) -> tuple[list[dict], list[dict]]:
    """Re-measure every timing cell with a "slower" row for --confirm-rounds more interleaved
    rounds, and decide again on all rounds together. At high thread counts a single round can land
    on an unlucky scheduling wave even for identical binaries; only a slowdown that persists over
    the added rounds is a regression."""
    flagged = {tuple(m["cell_key"]) for m in measurements
               if m["verdict"] == "slower" and m.get("cell_key")}
    if not flagged or args.confirm_rounds <= 0:
        return measurements, []
    notes, replaced = [], {}
    for key in sorted(flagged, key=str):
        print(f"confirming {key} with {args.confirm_rounds} more rounds", flush=True)
        if key[0] == "raw":
            _, variant, threads = key
            br, cr = raw_samples[key]
            br2, cr2, _ = measure_raw(args, variant, models[variant], threads, golden_rows, scratch,
                                      args.confirm_rounds, f"confirm-raw-{variant}-t{threads}")
            replaced[key] = raw_rows(args, variant, threads, golden_rows, br + br2, cr + cr2)
        else:
            _, variant, length, threads = key
            br, cr, data = sweep_samples[key]
            br2, cr2, _ = benchmark_cell(args.baseline / "test_model_parity", args.candidate / "test_model_parity",
                                         models[variant], data, threads, args.confirm_rounds,
                                         VARIANT_TOLERANCE[variant], args.candidate_env, scratch,
                                         f"confirm-sweep-{variant}-l{length}-t{threads}")
            br, cr = br + br2, cr + cr2
            replaced[key] = [metric_row("length sweep", f"{variant} L{length} t{threads}",
                                        [r[0] for r in br], [r[0] for r in cr], args.tolerance, key=key)]
        still = [r["cell"] for r in replaced[key] if r["verdict"] == "slower"]
        notes.append({"cell_key": list(key), "confirm_rounds": args.confirm_rounds,
                      "still_slower": still})
    out = [m for m in measurements if not (m.get("cell_key") and tuple(m["cell_key"]) in replaced)]
    for rows_ in replaced.values():
        out += rows_
    return out, notes


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def post(port: int, body: bytes) -> None:
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/systemone", data=body,
                                     headers={"Content-Type": "application/json"})
    urllib.request.urlopen(request, timeout=120).read()


def startup_once(binary: Path, model: Path, bodies: list[bytes], extra_env: dict[str, str]) -> tuple[float, float]:
    import psutil
    port = free_port()
    command = [str(binary), "serve", "--device", "cpu", "--threads", "4", "--host", "127.0.0.1",
               "--port", str(port), "-m", "multilingual=" + str(model)]
    started = time.perf_counter()
    proc = subprocess.Popen(command, cwd=ROOT, env=environment(extra_env),
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    process = psutil.Process(proc.pid)
    try:
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited with {proc.returncode}: {proc.stderr.read().decode()[-2000:]}")
            try:
                post(port, bodies[0])
                first_ms = (time.perf_counter() - started) * 1000
                break
            except (urllib.error.URLError, ConnectionError):
                if time.perf_counter() - started > 300:
                    raise TimeoutError("server did not answer in 300 seconds")
                time.sleep(0.02)
        for index in range(1, 4):
            post(port, bodies[index % len(bodies)])
        return first_ms, float(process.memory_info().rss)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def startup_section(args, models: dict[str, Path]) -> list[dict]:
    data = json.loads(INPUTS.read_text())
    bodies = [json.dumps({"state": state, "questions": data["questions"],
                          "model": "multilingual"}).encode() for state in data["states"]]
    output = []
    for variant, model in models.items():
        bt, ct, bm, cm = [], [], [], []
        for _ in range(3):
            t, m = startup_once(args.baseline / "statim", model, bodies, {})
            bt.append(t); bm.append(m)
            t, m = startup_once(args.candidate / "statim", model, bodies, args.candidate_env)
            ct.append(t); cm.append(m)
        output.append(metric_row("start-up", f"{variant} first answer", bt, ct, .05))
        output.append(metric_row("memory", f"{variant} RSS after 4 requests", bm, cm, .03,
                                 unit="bytes", noisy=False))
    return output


def markdown_table(headers: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def render_report(result: dict) -> str:
    lines = ["# Statim performance regression gate", ""]
    for title, section in (("Raw scoring", "raw"), ("Length sweep", "length sweep"),
                           ("Start-up", "start-up"), ("Memory", "memory")):
        subset = [r for r in result["measurements"] if r["section"] == section]
        lines += [f"## {title}", ""]
        if subset:
            lines.append(markdown_table(["Cell", "Baseline", "Candidate", "Delta", "Noise", "Verdict"],
                [[r["cell"], f"{r['baseline']:.3f} {r['unit']}", f"{r['candidate']:.3f} {r['unit']}",
                  f"{r['delta_fraction']:+.2%}", f"{r['noise_fraction']:.2%}", r["verdict"]]
                 for r in subset]))
        else:
            lines.append("Not run in this environment.")
        lines.append("")
    lines += ["## Fidelity", "", markdown_table(
        ["Cell", "Candidate argmax", "Candidate/reference", "Candidate/baseline", "Argmax losses", "Verdict"],
        [[f["cell"], f"{min(r['candidate_argmax_agree'] for r in f['rounds'])}/{f['rounds'][0]['items']}",
          f"{max(r['candidate_max_abs_logit_vs_reference'] for r in f['rounds']):.6g}",
          f"{max(r['candidate_max_abs_logit_vs_baseline'] for r in f['rounds']):.6g}",
          str(sum(r['candidate_argmax_losses'] for r in f['rounds'])), "pass" if f["pass"] else "FAIL"]
         for f in result["fidelity"]]), "", f"**Overall: {'PASS' if result['pass'] else 'FAIL'}**", ""]
    return "\n".join(lines)


def relative(path: Path) -> str:
    # Preserve in-tree symlink spellings (notably models/ and build-ort/): their targets may be
    # in a sibling worktree, while rsync -L deliberately copies their contents into this layout.
    resolved = Path(os.path.abspath(path))
    try:
        return str(resolved.relative_to(Path(os.path.abspath(ROOT))))
    except ValueError as exc:
        raise ValueError(f"--ssh requires paths inside {ROOT}: {path}") from exc


def remote_run(args) -> int:
    if not args.remote_dir:
        raise ValueError("--ssh requires --remote-dir")
    required = [Path("bench/perf_gate.py"), Path("bench/ort_compare.py"), Path("tests/data/golden_inputs.json"),
                Path("models/laya-multilingual-v9/rl_agent_config.json"),
                Path("models/laya-multilingual-v9/tokenizer/tokenizer.json"), Path(relative(args.golden)),
                *(Path(relative(p)) for p in args.models),
                Path(relative(args.baseline / "statim")), Path(relative(args.baseline / "test_model_parity")),
                Path(relative(args.candidate / "statim")), Path(relative(args.candidate / "test_model_parity")),
                Path(relative(args.venv))]
    subprocess.run(["ssh", args.ssh, f"mkdir -p -- {shlex.quote(args.remote_dir)}"], check=True)
    destination = f"{args.ssh}:{args.remote_dir.rstrip('/')}/"
    subprocess.run(["rsync", "-aL", "--relative", *map(str, required), destination], cwd=ROOT, check=True)
    remote_args = ["bench/perf_gate.py", "--baseline", relative(args.baseline),
                   "--candidate", relative(args.candidate), "--models",
                   *(relative(p) for p in args.models), "--golden", relative(args.golden),
                   "--out", relative(args.out), "--report", relative(args.report),
                   "--rounds", str(args.rounds), "--tolerance", str(args.tolerance),
                   "--confirm-rounds", str(args.confirm_rounds)]
    for key, value in args.candidate_env.items():
        remote_args += ["--candidate-env", f"{key}={value}"]
    if args.quick:
        remote_args.append("--quick")
    if args.skip_startup:
        remote_args.append("--skip-startup")
    executable = str(Path(relative(args.venv)) / "bin/python")
    shell_command = (f"cd {shlex.quote(args.remote_dir)} && exec " +
                     " ".join(shlex.quote(v) for v in [executable, *remote_args]))
    code = subprocess.run(["ssh", args.ssh, shell_command]).returncode
    # bring the results home next to where a local run would have written them
    for path in (args.out, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", f"{destination}{relative(path)}", str(path)], check=False)
    return code


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Interleaved Statim performance regression gate")
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--models", type=Path, nargs=2, metavar=("F32", "Q8_0"), default=DEFAULT_MODELS)
    ap.add_argument("--golden", type=Path, default=GOLDEN)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tolerance", type=float, default=.02)
    ap.add_argument("--candidate-env", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--confirm-rounds", type=int, default=5,
                    help="extra interleaved rounds for every timing cell first judged slower (0: off)")
    ap.add_argument("--skip-startup", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--ssh")
    ap.add_argument("--remote-dir")
    ap.add_argument("--venv", type=Path, default=ROOT / "build-ort/venv-gate")
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = parser()
    args = ap.parse_args(argv)
    if args.rounds < 2 or args.tolerance < 0:
        ap.error("--rounds must be at least 2 and --tolerance non-negative")
    try:
        args.candidate_env = parse_env(args.candidate_env)
    except ValueError as exc:
        ap.error(str(exc))
    if args.quick:
        args.threads, args.sweep_threads, args.lengths, args.rounds = [1, 8], [1, 8], [128, 512], 2
    else:
        args.threads, args.sweep_threads, args.lengths = [1, 4, 8, 16], [1, 4, 8], [128, 256, 512, 1024]
    if args.ssh:
        return remote_run(args)
    for directory in (args.baseline, args.candidate):
        for binary in ("statim", "test_model_parity"):
            if not (directory / binary).is_file():
                ap.error(f"missing {directory / binary}")
    args.golden = args.golden.resolve()
    models = {"f32": args.models[0].resolve(), "q8_0": args.models[1].resolve()}
    estimate = 25 if args.quick else 90
    print(f"CPU-only gate: about {estimate} minutes; {args.rounds} interleaved rounds per timing cell.", flush=True)
    with tempfile.TemporaryDirectory(prefix="statim-perf-gate-") as tmp:
        scratch = Path(tmp)
        golden_rows = rows(args.golden)
        measurements, fidelity, raw_samples = raw_section(args, models, golden_rows, scratch)
        sweep, sweep_fidelity, skipped, sweep_samples = sweep_section(args, models, scratch)
        measurements += sweep
        fidelity += sweep_fidelity
        measurements, confirmations = confirm_slower(args, models, golden_rows, scratch, measurements,
                                                     raw_samples, sweep_samples)
        if not args.skip_startup:
            measurements += startup_section(args, models)
    failed = any(r["verdict"] == "slower" for r in measurements) or any(not f["pass"] for f in fidelity)
    result = {"schema_version": 1, "cpu_only": True, "baseline": str(args.baseline),
              "candidate": str(args.candidate), "candidate_env": args.candidate_env,
              "rounds": args.rounds, "tolerance_fraction": args.tolerance,
              "quick": args.quick, "startup_skipped": args.skip_startup,
              "measurements": measurements, "fidelity": fidelity,
              "length_sweep_skipped": skipped, "confirmations": confirmations, "pass": not failed}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    report = render_report(result)
    args.report.write_text(report)
    print(report, end="")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
