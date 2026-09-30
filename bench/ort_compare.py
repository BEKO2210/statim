#!/usr/bin/env python3
"""Reproducible CPU comparison of Statim 0.9.0 and ONNX Runtime.

Raw scoring is the default. ``--socket-benchmark`` runs the HTTP, cold-start
and RSS measurements. ``--serve-ort`` runs the
faithful comparison server. ``--render-md FILE`` renders every result table.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import sysconfig
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# CPU is mandatory unless the reviewer explicitly names a GPU EP. This keeps
# every default and every subprocess in the published CPU protocol GPU-blind.
_provider_arg = sys.argv[sys.argv.index("--ort-provider") + 1] if "--ort-provider" in sys.argv else "cpu"
if _provider_arg == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ORT_TELEMETRY_DISABLED"] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build-ort"
TEMPLATE_GOLDEN = ROOT / "tests/data/golden_laya-multilingual.jsonl"
V9_GOLDEN = OUT / "golden_v9.jsonl"
INPUTS = ROOT / "tests/data/golden_inputs.json"
MODEL_DIR = ROOT / "models/laya-multilingual-v9"
GGUF_F32 = ROOT / "models/laya-multilingual-v9-f32.gguf"
GGUF_Q8 = OUT / "laya-multilingual-v9-q8_0.gguf"
ONNX_F32 = OUT / "laya-multilingual-v9-f32.onnx"
ONNX_INT8 = OUT / "laya-multilingual-v9-8bit-blockwise.onnx"  # MatMulNBits 8-bit, block 32, f32 compute (accuracy_level 0)
ONNX_INT8_DYNAMIC = OUT / "laya-multilingual-v9-int8-dynamic.onnx"  # quantize_dynamic, per channel, reduce_range
RELEASE = OUT / "release-0.9.0/statim-0.9.0-linux-x86_64-cpu/statim"
INPUT_NAMES = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
QTYPES = {"choice": 0, "score": 1, "noul": 2}


def json_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def state_groups(rows: list[dict]) -> list[list[dict]]:
    out: dict[int, list[dict]] = {}
    for row in rows:
        out.setdefault(int(row["state_index"]), []).append(row)
    return [out[key] for key in sorted(out)]


def collate(rows: list[dict], pad_id: int = 0) -> dict[str, np.ndarray]:
    batch, seq = len(rows), max(len(row["ids"]) for row in rows)
    options = max(len(row["markers"]) for row in rows)
    ids = np.full((batch, seq), pad_id, dtype=np.int64)
    attention = np.zeros((batch, seq), dtype=np.int64)
    marker_pos = np.zeros((batch, options), dtype=np.int64)
    marker_mask = np.zeros((batch, options), dtype=np.bool_)
    qtype = np.empty(batch, dtype=np.int64)
    for i, row in enumerate(rows):
        n, k = len(row["ids"]), len(row["markers"])
        ids[i, :n], attention[i, :n] = row["ids"], 1
        marker_pos[i, :k], marker_mask[i, :k] = row["markers"], True
        qtype[i] = row["qtype"]
    return dict(zip(INPUT_NAMES, (ids, attention, marker_pos, marker_mask, qtype)))


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]


def summarize(per_state: list[float]) -> dict:
    return {"mean_ms": statistics.mean(per_state), "p50_ms": statistics.median(per_state),
            "p95_ms": percentile(per_state, 0.95), "per_state_ms": per_state}


def session(path: Path, threads: int, provider: str = "cpu", profile: bool = False, spinning: bool = True):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if profile:
        opts.enable_profiling = True
        opts.profile_file_prefix = str(OUT / "ort-profile")
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.add_session_config_entry("session.set_denormal_as_zero", "1")
    # ORT's default; the HTTP benchmark also measures it off as a sensitivity check
    opts.add_session_config_entry("session.intra_op.allow_spinning", "1" if spinning else "0")
    providers = {
        "cpu": ["CPUExecutionProvider"],
        "cuda": ["CUDAExecutionProvider"],
        "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
    }[provider]
    available = set(ort.get_available_providers())
    if providers[0] not in available:
        raise RuntimeError(f"requested {providers[0]}, available providers are {sorted(available)}")
    return ort.InferenceSession(str(path), sess_options=opts, providers=providers)


def softmax(values: np.ndarray) -> np.ndarray:
    z = values - values.max(axis=-1, keepdims=True)
    out = np.exp(z)
    return out / out.sum(axis=-1, keepdims=True)


# ---- Laya rendering/packing and answer semantics, without PyTorch. ----


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(q: dict) -> list[str]:
    kind, criteria = q["t"], q.get("crit")
    if kind == "choice":
        return [str(key) if value is None or value == "" else
                f"{key}: {value if isinstance(value, str) else compact_json(value)}"
                for key, value in criteria.items()]
    if kind == "score":
        return [f"level {i}: {value if isinstance(value, str) else compact_json(value)}"
                for i, value in enumerate(criteria)]
    labels, criteria = q.get("labels") or {"false": "false", "true": "true"}, criteria or {}
    return [
        labels["false"] + ": " + (str(criteria.get("false")) if criteria.get("false") not in (None, "")
                                    else "no, the statement does not hold"),
        labels["true"] + ": " + (str(criteria.get("true")) if criteria.get("true") not in (None, "")
                                   else "yes, the statement holds"),
    ]


def internal_question(qdef: dict) -> dict:
    kind, criteria = qdef["type"], qdef.get("criteria")
    if kind == "choice" and isinstance(criteria, list):
        criteria = OrderedDict((value, None) for value in criteria)
    elif kind == "noul" and isinstance(criteria, dict):
        criteria = {str(key).lower(): value for key, value in criteria.items()}
    instructions = qdef["instructions"]
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions, ensure_ascii=False)
    out = {"t": kind, "ins": instructions, "crit": criteria}
    if "labels" in qdef:
        out["labels"] = qdef["labels"]
    return out


class ORTEngine:
    def __init__(self, model_dir: Path, model: Path, threads: int, provider: str = "cpu", spinning: bool = True):
        from tokenizers import Tokenizer

        self.cfg = json.loads((model_dir / "rl_agent_config.json").read_text())
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer/tokenizer.json"))
        self.ids = {name: self.tok.token_to_id(token) for name, token in
                    {"pad": "<pad>", "cls": "<bos>", "sep": "<eos>", "mask": "<mask>"}.items()}
        self.sess = session(model, threads, provider, spinning=spinning)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text.replace("<mask>", " "), add_special_tokens=False).ids

    def build(self, state: Any, q: dict, state_ids: list[int], max_len: int | None = None,
              head_max_len: int | None = None) -> tuple[list[int], list[int]]:
        max_len = max_len or self.cfg.get("max_len", 512)
        head_max_len = head_max_len or self.cfg.get("head_max_len", 192)
        option_ids = [[self.ids["mask"], *self.encode(" " + text)[:48]] for text in render_options(q)]
        budget = head_max_len - sum(map(len, option_ids))
        if budget < 16:
            per = max(4, (head_max_len - 16) // max(1, len(option_ids)))
            option_ids = [values[:per] for values in option_ids]
            budget = head_max_len - sum(map(len, option_ids))
        head = self.encode(f"{q['t']} question: {str(q['ins']).replace('<mask>', ' ')}")
        ids = [self.ids["cls"], *head[:max(8, budget)], self.ids["sep"]]
        markers = []
        for option in option_ids:
            markers.append(len(ids))
            ids.extend(option)
        ids.append(self.ids["sep"])
        room = max(0, max_len - len(ids) - 1)
        selected = state_ids[max(0, len(state_ids) - room):] if isinstance(state, list) else state_ids[:room]
        ids = [*ids, *selected, self.ids["sep"]][:max_len]
        markers = [marker for marker in markers if marker < max_len]
        if len(markers) != len(option_ids):
            raise ValueError("question options exceed head_max_len")
        return ids, markers

    @staticmethod
    def confidence(p: np.ndarray) -> float:
        if len(p) <= 1:
            return 1.0
        entropy = -float(np.sum(p * np.log(np.maximum(p, 1e-12)))) / math.log(len(p))
        return min(1.0, max(0.0, 1.0 - entropy))

    def _decode(self, q: dict, logits: np.ndarray, act_logits: np.ndarray, lang: str | None) -> dict:
        k, qt = len(render_options(q)), QTYPES[q["t"]]
        bucket = f"{q['t']}:{'2' if k <= 2 else '3-5' if k <= 5 else '6-10' if k <= 10 else '11+'}"
        temps = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        by_options = self.cfg.get("temperature_by_options", {})
        scale = by_options.get(bucket, temps[qt])
        if lang:
            lang_cfg = self.cfg.get("lang_temperatures", {}).get(lang.split("-")[0].lower())
            if lang_cfg:
                scale = lang_cfg.get("temperature_by_options", {}).get(
                    bucket, lang_cfg.get("temperature", temps)[qt])
        scale = min(5.0, max(0.5, float(scale))) if math.isfinite(float(scale)) else 1.0
        p, act = softmax(logits[:k] / scale), softmax(act_logits)
        answer_conf = round(float(p.max()), 4)
        action = {"act_probability": round(float(act[0]), 4)}
        if q["t"] == "choice":
            keys = list(q["crit"])
            return {"type": "choice", "choice": keys[int(p.argmax())],
                    "probabilities": {str(key): round(float(value), 4) for key, value in zip(keys, p)},
                    "confidence": round(self.confidence(p), 4), "answer_confidence": answer_conf,
                    "action": action}
        if q["t"] == "score":
            return {"type": "score", "score": round(float(np.arange(k) @ p), 4),
                    "legend": {str(i): value for i, value in enumerate(q["crit"])},
                    "probabilities": {str(i): round(float(value), 4) for i, value in enumerate(p)},
                    "confidence": round(self.confidence(p), 4), "answer_confidence": answer_conf,
                    "action": action}
        return {"type": "noul", "noul": round(float(p[1]), 4), "confidence": answer_conf,
                "answer_confidence": answer_conf, "action": action}

    def decide_many(self, states: list[Any], questions: dict, lang: str | None = None,
                    max_len: int | None = None, head_max_len: int | None = None) -> list[dict]:
        names, internal = list(questions), {}
        for name, qdef in questions.items():
            if not isinstance(qdef, dict) or qdef.get("type") not in QTYPES or "instructions" not in qdef:
                raise ValueError(f"invalid question {name!r}")
            internal[name] = internal_question(qdef)
        if not names:
            return [self._result({}, 0) for _ in states]
        rows, offsets, token_counts = [], [], []
        for state in states:
            text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
            state_ids, offset, count = self.encode(text), len(rows), 0
            for name in names:
                q = internal[name]
                ids, markers = self.build(state, q, state_ids, max_len, head_max_len)
                rows.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
                count += len(ids)
            offsets.append(offset)
            token_counts.append(count)
        logits, act_logits = self.sess.run(["logits", "act_logits"], collate(rows, self.ids["pad"]))
        results = []
        for offset, token_count in zip(offsets, token_counts):
            answers = {name: self._decode(internal[name], logits[offset + i], act_logits[offset + i], lang)
                       for i, name in enumerate(names)}
            results.append(self._result(answers, token_count))
        return results

    def _result(self, answers: dict, input_tokens: int) -> dict:
        return {"model": self.cfg.get("model_name", "laya-multilingual"), "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                "routing": {"model": "multilingual", "reason": "requested", "engine": "onnxruntime",
                            "weights": "int8" if "int8" in str(self.sess._model_path) else "f32"}}

    def decide(self, request: dict) -> dict:
        return self.decide_many([request["state"]], request["questions"], request.get("lang"),
                                request.get("max_len"), request.get("head_max_len"))[0]


def serve_ort(args) -> None:
    engine = ORTEngine(args.model_dir, args.onnx, args.threads, args.ort_provider, spinning=bool(args.ort_spinning))
    if INPUTS.exists():
        try:
            warm_data = json.loads(INPUTS.read_text())
            engine.decide({"state": warm_data["states"][0], "questions": warm_data["questions"]})
        except Exception:
            pass

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/health", "/ready"):
                payload, status = b'{"status":"ok"}', 200
            else:
                payload, status = b'{"error":"not found"}', 404
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            try:
                request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                if self.path == "/v1/systemone":
                    result = engine.decide(request)
                elif self.path == "/v1/systemone/batch":
                    result = {"results": engine.decide_many(request["states"], request["questions"],
                                                             request.get("lang"), request.get("max_len"),
                                                             request.get("head_max_len"))}
                else:
                    raise KeyError("not found")
                payload, status = json.dumps(result, ensure_ascii=False).encode(), 200
            except KeyError as exc:
                payload, status = json.dumps({"error": str(exc)}).encode(), 404
            except Exception as exc:
                payload, status = json.dumps({"error": str(exc)}).encode(), 400
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format, *_args):
            pass

    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


def require_parity(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError("missing build-ort/export-report.json; run bench/ort_export.py first")
    report = json.loads(path.read_text())
    if not report.get("f32_gates_pass"):
        raise RuntimeError("not every f32 parity gate passed; refusing to time")
    return report


def packing_check(engine: ORTEngine, inputs_path: Path, golden_path: Path) -> dict:
    data, rows = json.loads(inputs_path.read_text()), json_rows(golden_path)
    expected = {(int(row["state_index"]), row["question"]): row for row in rows}
    mismatches, total = [], 0
    for state_index, state in enumerate(data["states"]):
        text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        state_ids = engine.encode(text)
        for name, qdef in data["questions"].items():
            ids, markers = engine.build(state, internal_question(qdef), state_ids)
            ref = expected[(state_index, name)]
            total += 1
            if ids != ref["ids"] or markers != ref["markers"]:
                mismatches.append({"state_index": state_index, "question": name})
    result = {"items": total, "matched": total - len(mismatches), "mismatches": mismatches[:20],
              "pass": not mismatches}
    if mismatches:
        raise RuntimeError(f"packing check failed for {len(mismatches)}/{total} items")
    return result


def answer_differences(expected: dict, actual: dict, tolerance: float) -> list[str]:
    problems = []
    for name, ref in expected["answers"].items():
        got = actual.get("answers", {}).get(name)
        if not got:
            problems.append(f"{name}: missing")
            continue
        if ref["type"] != got.get("type"):
            problems.append(f"{name}: type")
        if "choice" in ref and ref["choice"] != got.get("choice"):
            problems.append(f"{name}: choice {ref['choice']!r} != {got.get('choice')!r}")
        for field in ("probabilities",):
            for key, value in ref.get(field, {}).items():
                if abs(float(value) - float(got.get(field, {}).get(key, float("inf")))) > tolerance:
                    problems.append(f"{name}.{field}.{key}")
        for field in ("score", "noul", "confidence", "answer_confidence"):
            if field in ref and abs(float(ref[field]) - float(got.get(field, float("inf")))) > tolerance:
                problems.append(f"{name}.{field}")
        if abs(float(ref.get("action", {}).get("act_probability", 0)) -
               float(got.get("action", {}).get("act_probability", float("inf")))) > tolerance:
            problems.append(f"{name}.action.act_probability")
    return problems


def answer_check(engine: ORTEngine, statim: Path, gguf: Path, inputs_path: Path,
                 tolerance: float = 1e-3) -> dict:
    data, mismatches = json.loads(inputs_path.read_text()), []
    command = [str(statim), "decide", "-m", str(gguf), "--device", "cpu", "--threads", "4"]
    for state_index, state in enumerate(data["states"]):
        request = {"state": state, "questions": data["questions"]}
        proc = subprocess.run(command, input=json.dumps(request), text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, cwd=ROOT,
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, check=False)
        if proc.returncode:
            raise RuntimeError(f"Statim answer check failed at state {state_index}: {proc.stderr[-1000:]}")
        ref, got = json.loads(proc.stdout), engine.decide(request)
        problems = answer_differences(ref, got, tolerance)
        if problems:
            mismatches.append({"state_index": state_index, "differences": problems[:20]})
    result = {"states": len(data["states"]), "questions_per_state": len(data["questions"]),
              "matched_states": len(data["states"]) - len(mismatches), "mismatches": mismatches,
              "tolerance": tolerance, "command": command, "pass": not mismatches}
    if mismatches:
        raise RuntimeError(f"answer check failed for {len(mismatches)}/{len(data['states'])} states")
    return result


def repeated_golden(rows: list[dict], repeats: int, path: Path) -> None:
    groups = state_groups(rows)
    output = []
    for pass_index in range(repeats + 1):
        for state_index, group in enumerate(groups):
            for source in group:
                output.append({**source, "state_index": pass_index * len(groups) + state_index})
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output))


def statim_raw(binary: Path, model: Path, rows: list[dict], threads: int, repeats: int,
               scratch: Path) -> tuple[dict, dict]:
    repeated_golden(rows, repeats, scratch)
    command = [str(binary), str(model), str(scratch), "1e9", str(threads)]
    proc = subprocess.run(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "",
                          "STATIM_DEVICE": "cpu", "STATIM_VERBOSE": "1"}, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    times = [float(value) for value in re.findall(r"state\s+\d+:.*?([0-9.]+) ms,", proc.stdout)]
    summary = re.search(r"items (\d+) \| argmax agree (\d+)/(\d+) \| max \|dlogit\| ([0-9.eE+-]+) "
                        r"\| max \|dact\| ([0-9.eE+-]+)", proc.stdout)
    states = len(state_groups(rows))
    if len(times) != (repeats + 1) * states or not summary:
        raise RuntimeError("could not parse repeated test_model_parity output:\n" + proc.stdout[-3000:])
    measured = times[states:]
    medians = [statistics.median(measured[p * states + state] for p in range(repeats))
               for state in range(states)]
    timing = {"engine": "statim", "artifact": model.name, "threads": threads, "states": states,
              "repeats": repeats, "warmup_passes": 1, "command": command, **summarize(medians)}
    parity = {"items": int(summary.group(1)), "argmax_agree": int(summary.group(2)),
              "max_abs_logit": float(summary.group(4)), "max_abs_act": float(summary.group(5)),
              "exit_code": proc.returncode}
    return timing, parity


def ort_raw(path: Path, rows: list[dict], threads: int, repeats: int, invocation: list[str],
            provider: str = "cpu") -> tuple[dict, dict]:
    started = time.perf_counter()
    sess = session(path, threads, provider)
    load_ms = (time.perf_counter() - started) * 1000
    groups, passes = state_groups(rows), []
    max_diff = max_act = 0.0
    agreed = total = 0
    for pass_index in range(repeats + 1):
        pass_times = []
        for group in groups:
            feed = collate(group)
            start = time.perf_counter()
            logits, act_logits = sess.run(["logits", "act_logits"], feed)
            pass_times.append((time.perf_counter() - start) * 1000)
            if pass_index == 1:
                acts = softmax(act_logits)
                for i, row in enumerate(group):
                    ref = np.asarray(row["logits"], dtype=np.float32)
                    got = logits[i, :len(ref)]
                    max_diff = max(max_diff, float(np.max(np.abs(got - ref))))
                    max_act = max(max_act, float(np.max(np.abs(acts[i] - np.asarray(row["act"])))))
                    agreed += int(np.argmax(got) == np.argmax(ref))
                    total += 1
        if pass_index:
            passes.append(pass_times)
    medians = [statistics.median(values[state] for values in passes) for state in range(len(groups))]
    timing = {"engine": "ort", "provider": provider, "artifact": path.name, "threads": threads,
              "states": len(groups),
              "repeats": repeats, "warmup_passes": 1, "session_load_ms": load_ms,
              "command": invocation, **summarize(medians)}
    return timing, {"items": total, "argmax_agree": agreed, "max_abs_logit": max_diff,
                    "max_abs_act": max_act}


def binding_overhead(path: Path, rows: list[dict], threads: int, repeats: int) -> dict:
    """Cost of calling ORT from Python: wall time around session.run minus ORT's own profiled
    model_run time for the same call. Profiling bookkeeping outside model_run counts as overhead
    too, so this is an upper bound for the binding cost."""
    sess = session(path, threads, profile=True)
    groups, walls = state_groups(rows), []
    for pass_index in range(repeats + 1):
        for group in groups:
            feed = collate(group)
            start = time.perf_counter()
            sess.run(["logits", "act_logits"], feed)
            if pass_index:
                walls.append((time.perf_counter() - start) * 1e6)
    trace_path = Path(sess.end_profiling())
    runs = [event["dur"] for event in json.loads(trace_path.read_text())
            if event.get("cat") == "Session" and event.get("name") == "model_run"]
    trace_path.unlink()
    runs = runs[len(groups):]  # drop the warm-up pass
    if len(runs) != len(walls):
        raise RuntimeError(f"profile has {len(runs)} measured model_run events, expected {len(walls)}")
    overhead = [wall - run for wall, run in zip(walls, runs)]
    return {"threads": threads, "calls": len(overhead), "repeats": repeats, "warmup_passes": 1,
            "median_wall_us": statistics.median(walls), "median_model_run_us": statistics.median(runs),
            "median_overhead_us": statistics.median(overhead), "max_overhead_us": max(overhead),
            "median_overhead_share": statistics.median(o / w for o, w in zip(overhead, walls))}


def parity_pass(parity: dict, tolerance: float = 1e-3) -> bool:
    return parity["argmax_agree"] == parity["items"] and parity["max_abs_logit"] <= tolerance


def length_sweep_rows(engine: ORTEngine, inputs_path: Path,
                      lengths: list[int]) -> tuple[dict[int, list[dict]], list[dict]]:
    data = json.loads(inputs_path.read_text())
    questions = [(name, internal_question(qdef)) for name, qdef in data["questions"].items()]
    max_len = int(engine.cfg.get("max_len", 512))
    text_parts = [state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
                  for state in data["states"]]
    long_text = "\n".join(text_parts)
    while len(engine.encode(long_text)) < max(lengths, default=0):
        long_text += "\n" + "\n".join(text_parts)
    state_ids = engine.encode(long_text)
    batches, skipped = {}, []
    for length in lengths:
        if length > max_len:
            skipped.append({"length": length, "reason": f"exceeds model max_len {max_len}"})
            continue
        rows, too_long = [], []
        for name, question in questions:
            head_ids, _ = engine.build(long_text, question, [], max_len=max_len)
            if len(head_ids) > length:
                too_long.append(name)
                continue
            ids, markers = engine.build(long_text, question, state_ids, max_len=length)
            if len(ids) != length:
                raise RuntimeError(f"could not build {name!r} at exactly {length} tokens (got {len(ids)})")
            rows.append({"state_index": 0, "question": name, "ids": ids, "markers": markers,
                         "qtype": QTYPES[question["t"]]})
        if too_long:
            skipped.append({"length": length, "reason": "question head exceeds target",
                            "questions": too_long})
        else:
            batches[length] = rows
    return batches, skipped


def add_reference(rows: list[dict], engine: ORTEngine) -> None:
    logits, act_logits = engine.sess.run(["logits", "act_logits"], collate(rows, engine.ids["pad"]))
    acts = softmax(act_logits)
    for row, row_logits, row_act in zip(rows, logits, acts):
        row["logits"] = row_logits[:len(row["markers"])].tolist()
        row["act"] = row_act.tolist()


def run_length_sweep(args) -> dict:
    reference = ORTEngine(args.model_dir, args.onnx_f32, min(args.thread_counts), args.ort_provider)
    batches, skipped = length_sweep_rows(reference, args.inputs, args.lengths)
    for rows in batches.values():
        add_reference(rows, reference)
    del reference
    output_rows = []
    for threads in args.thread_counts:
        for length, rows in batches.items():
            result = {"length": length, "threads": threads,
                      "sequence_lengths": [len(row["ids"]) for row in rows], "timings": {}, "parity": {}}
            pairs = (("f32", args.gguf_f32, args.onnx_f32),
                     ("quantized", args.gguf_q8, args.onnx_int8))
            for pair, gguf, onnx in pairs:
                scratch = OUT / f"length-sweep-l{length}-{pair}-t{threads}.jsonl"
                statim_timing, statim_parity = statim_raw(
                    args.statim_parity, gguf, rows, threads, args.repeats, scratch)
                invocation = [str(args.ort_python), str(Path(__file__).resolve()), "--length-sweep",
                              "--lengths", *map(str, args.lengths), "--thread-counts", str(threads),
                              "--repeats", str(args.repeats), "--ort-provider", args.ort_provider,
                              "--out", str(args.out)]
                ort_timing, ort_parity = ort_raw(
                    onnx, rows, threads, args.repeats, invocation, args.ort_provider)
                if pair == "f32":
                    statim_parity["pass"] = parity_pass(statim_parity)
                    ort_parity["pass"] = parity_pass(ort_parity)
                    if not statim_parity["pass"]:
                        raise RuntimeError(f"Statim f32 parity failed at length {length}, threads {threads}")
                result["timings"][pair] = {"statim": statim_timing, "ort": ort_timing}
                result["parity"][pair] = {"statim": statim_parity, "ort": ort_parity}
            output_rows.append(result)
    return {"lengths": args.lengths, "questions": 8, "repeats": args.repeats,
            "warmup_passes": 1, "rows": output_rows, "skipped": skipped}


def file_set(path: Path) -> list[Path]:
    return [item for item in (path, path.with_suffix(path.suffix + ".data")) if item.exists()]


def disk_bytes(paths: list[Path]) -> int:
    return sum(path.stat().st_size for path in paths)


def du_bytes(path: Path) -> int:
    proc = subprocess.run(["du", "-sb", str(path)], text=True, stdout=subprocess.PIPE, check=True)
    return int(proc.stdout.split()[0])


def inventory() -> dict:
    loaded = [MODEL_DIR / "tokenizer/tokenizer.json", MODEL_DIR / "rl_agent_config.json"]
    artifacts = {"onnx_f32": file_set(ONNX_F32) + loaded, "onnx_int8": file_set(ONNX_INT8) + loaded,
                 "gguf_f32": [GGUF_F32], "gguf_q8_0": [GGUF_Q8]}
    tarball = OUT / "release-0.9.0/statim-0.9.0-linux-x86_64-cpu.tar.gz"
    venv = OUT / "venv-server"
    stdlib = Path(sysconfig.get_path("stdlib"))
    executable_bytes = Path(sys.executable).stat().st_size
    stdlib_bytes = du_bytes(stdlib)
    return {
        "artifact_bytes": {name: disk_bytes(files) for name, files in artifacts.items()},
        "artifact_files": {name: [str(path.relative_to(ROOT)) for path in files] for name, files in artifacts.items()},
        "install_bytes": {"statim_release_tarball": tarball.stat().st_size,
                          "statim_extracted_binary": RELEASE.stat().st_size,
                          "ort_server_venv_without_cpython": du_bytes(venv)},
        "cpython": {"version": platform.python_version(), "executable": sys.executable,
                    "executable_bytes": executable_bytes, "stdlib": str(stdlib),
                    "stdlib_bytes": stdlib_bytes, "runtime_bytes": executable_bytes + stdlib_bytes},
    }


def machine_info() -> dict:
    cpu_model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        match = re.search(r"^model name\s*:\s*(.+)$", cpuinfo.read_text(), re.MULTILINE)
        if match:
            cpu_model = match.group(1)
    physical = set()
    if cpuinfo.exists():
        for block in cpuinfo.read_text().split("\n\n"):
            package = re.search(r"physical id\s*:\s*(\d+)", block)
            core = re.search(r"core id\s*:\s*(\d+)", block)
            if package and core:
                physical.add((package.group(1), core.group(1)))
    memory = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        match = re.search(r"^MemTotal:\s+(\d+) kB", meminfo.read_text(), re.MULTILINE)
        memory = int(match.group(1)) * 1024 if match else 0
    governors = sorted({path.read_text().strip() for path in
                        Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor")})
    return {"cpu_model": cpu_model, "physical_cores": len(physical) or None,
            "logical_threads": os.cpu_count(), "ram_bytes": memory, "kernel": platform.release(),
            "platform": platform.platform(), "cpu_governor": governors or "unreadable"}


def versions(export: dict, statim: Path) -> dict:
    statim_version = subprocess.run([str(statim), "--version"], text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, check=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, check=True).stdout.strip()
    out = dict(export.get("versions", {}))
    out.update({"benchmark_python": platform.python_version(), "statim": statim_version,
                "git_commit": commit})
    return out


def post(url: str, body: bytes, endpoint: str = "/v1/systemone", timeout: float = 600) -> bytes:
    request = urllib.request.Request(url + endpoint, data=body, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=timeout).read()


def wait_first(url: str, body: bytes, process: subprocess.Popen, started: float) -> float:
    while time.perf_counter() - started < 600:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with {process.returncode}")
        try:
            post(url, body, timeout=120)
            return (time.perf_counter() - started) * 1000
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.02)
    raise TimeoutError("server did not answer within 600 seconds")


def load_test(url: str, bodies: list[bytes], concurrency: int, requests: int, server=None) -> dict:
    """Closed-loop load. With the server's psutil.Process, also report the CPU time it used, as
    average busy cores: thread settings are not comparable across engines under concurrency
    (Statim divides --threads among its workers; each concurrent ORT run adds its calling thread
    to the intra-op pool)."""
    times = []
    cpu_before = server.cpu_times() if server else None
    ctx_before = server.num_ctx_switches() if server else None
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = []
        for i in range(requests):
            body = bodies[i % len(bodies)]
            futures.append(pool.submit(lambda value=body: (lambda t: (post(url, value),
                                         (time.perf_counter() - t) * 1000))(time.perf_counter())))
        for future in futures:
            _, elapsed = future.result()
            times.append(elapsed)
    wall = time.perf_counter() - start
    result = {"concurrency": concurrency, "requests": requests, "throughput_rps": requests / wall,
              "p50_ms": statistics.median(times), "p95_ms": percentile(times, 0.95),
              "p99_ms": percentile(times, 0.99)}
    if server:
        cpu_after, ctx_after = server.cpu_times(), server.num_ctx_switches()
        cpu = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
        switches = (ctx_after.voluntary - ctx_before.voluntary) + (ctx_after.involuntary - ctx_before.involuntary)
        result.update({"server_cpu_seconds": cpu, "avg_busy_cores": cpu / wall,
                       "context_switches_per_request": switches / requests})
    return result


@contextmanager
def running_server(command: list[str], port: int, body: bytes):
    import psutil

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "STATIM_DEVICE": "cpu"}
    started = time.perf_counter()
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    peak, stop = [0], threading.Event()

    def sample_rss():
        process = psutil.Process(proc.pid)
        while not stop.wait(0.01):
            try:
                peak[0] = max(peak[0], process.memory_info().rss)
            except psutil.Error:
                return

    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()
    url = f"http://127.0.0.1:{port}"
    try:
        cold = wait_first(url, body, proc, started)
        yield url, cold, peak, psutil.Process(proc.pid)
    finally:
        stop.set()
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        sampler.join(timeout=1)


def cpu_layout(threads: int, available: set[int]) -> tuple[list[int], list[int]]:
    """Server CPUs: one logical CPU on each of the first `threads` physical cores, so every server
    thread has a core to itself. Client CPUs: the logical CPUs of the remaining cores, or, when the
    server takes every core, the SMT siblings of the server's CPUs."""
    cores: dict[tuple[int, int], list[int]] = {}
    for cpu in sorted(available):
        topo = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = (int((topo / "physical_package_id").read_text()), int((topo / "core_id").read_text()))
        cores.setdefault(key, []).append(cpu)
    ordered = [cpus for _, cpus in sorted(cores.items())]
    if threads > len(ordered):
        raise RuntimeError(f"--pin needs {threads} physical cores, this machine has {len(ordered)}")
    server = [cpus[0] for cpus in ordered[:threads]]
    client = [cpu for cpus in ordered[threads:] for cpu in cpus] or \
             [cpu for cpus in ordered[:threads] for cpu in cpus[1:]]
    if not client:
        raise RuntimeError("--pin found no CPU left for the load generator")
    return server, client


def socket_benchmark(args) -> dict:
    data = json.loads(args.inputs.read_text())
    states = data["states"][:args.max_states or None]
    bodies = [json.dumps({"state": state, "questions": data["questions"], "model": "multilingual"}).encode()
              for state in states]
    variants = {"f32": (args.gguf_f32, args.onnx_f32), "quantized": (args.gguf_q8, args.onnx_int8)}
    rows, cold = [], []
    own_affinity = os.sched_getaffinity(0)
    for variant, (gguf, onnx) in variants.items():
        for threads in args.thread_counts:
            prefix, placement = [], {}
            if args.pin:
                server_cpus, client_cpus = cpu_layout(threads, own_affinity)  # not the narrowed client set
                prefix = ["taskset", "-c", ",".join(map(str, server_cpus))]
                os.sched_setaffinity(0, client_cpus)
                placement = {"server_cpus": server_cpus, "client_cpus": client_cpus}
            configs = [
                ("statim", {}, prefix + [str(args.release_statim), "serve", "--device", "cpu", "--threads", str(threads),
                                         "--host", "127.0.0.1", "--port", str(args.statim_port),
                                         "-m", "multilingual=" + str(gguf)], args.statim_port),
            ]
            for spinning in args.ort_spinning_variants:
                configs.append(("ort", {"ort_spinning": spinning},
                                prefix + [str(args.ort_python), str(Path(__file__).resolve()), "--serve-ort",
                                          "--threads", str(threads), "--host", "127.0.0.1", "--port", str(args.ort_port),
                                          "--onnx", str(onnx), "--model-dir", str(args.model_dir),
                                          "--ort-provider", args.ort_provider, "--ort-spinning", str(spinning)],
                                args.ort_port))
            for engine, extra, command, port in configs:
                with running_server(command, port, bodies[0]) as (url, cold_ms, peak, server):
                    for i in range(args.warmup_requests):
                        post(url, bodies[i % len(bodies)])
                    cold.append({"variant": variant, "engine": engine, "threads": threads, **extra, **placement,
                                 "cold_start_to_first_answer_ms": cold_ms, "peak_rss_bytes": peak[0],
                                 "command": command})
                    for concurrency in (1, 4):
                        rows.append({"variant": variant, "engine": engine, "threads": threads, **extra, **placement,
                                     "microbatch": False, "command": command,
                                     **load_test(url, bodies, concurrency, args.requests, server)})
            for workers in args.statim_workers:
                command = prefix + [str(args.release_statim), "serve", "--device", "cpu", "--threads", str(threads),
                           "--workers", str(workers), "--host", "127.0.0.1",
                           "--port", str(args.statim_port), "-m", "multilingual=" + str(gguf)]
                with running_server(command, args.statim_port, bodies[0]) as (url, _cold_ms, _peak, server):
                    for i in range(args.warmup_requests):
                        post(url, bodies[i % len(bodies)])
                    rows.append({"variant": variant, "engine": "statim", "threads": threads, **placement,
                                 "workers": workers, "microbatch": False, "command": command,
                                 **load_test(url, bodies, 4, args.requests, server)})
            command = prefix + [str(args.release_statim), "serve", "--device", "cpu", "--threads", str(threads),
                       "--host", "127.0.0.1", "--port", str(args.statim_port), "--batch-window-ms", "2",
                       "--max-batch", "16", "-m", "multilingual=" + str(gguf)]
            with running_server(command, args.statim_port, bodies[0]) as (url, _cold_ms, _peak, server):
                for i in range(args.warmup_requests):
                    post(url, bodies[i % len(bodies)])
                rows.append({"variant": variant, "engine": "statim", "threads": threads, **placement,
                             "microbatch": True, "command": command,
                             **load_test(url, bodies, 4, args.requests, server)})
    os.sched_setaffinity(0, own_affinity)
    return {"warmup_requests": args.warmup_requests, "pinned": args.pin, "rows": rows, "cold_start_memory": cold}


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    cell = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(cell, headers)) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)


def raw_state_lengths(golden_path: Path) -> list[int]:
    return [max(len(row["ids"]) for row in group) for group in state_groups(json_rows(golden_path))]


def category_mean(row: dict, state_lengths: list[int], predicate) -> str:
    values = [value for value, length in zip(row["per_state_ms"], state_lengths) if predicate(length)]
    return f"{statistics.mean(values):.2f}" if values else "—"


def render_md(path: Path, golden_path: Path = V9_GOLDEN, long_threshold: int = 256) -> str:
    result = json.loads(path.read_text())
    lines = ["## Generated comparison tables", "", "### Parity", ""]
    parity = result.get("export", {}).get("parity", {})
    lines.append(markdown_table(["Variant", "Argmax", "max |Δlogit|", "max |Δact|"],
        [[name, f"{value['argmax_agree']}/{value['items']}", f"{value['max_abs_logit']:.6g}",
          f"{value['max_abs_act']:.6g}"] for name, value in parity.items()]))
    lines += ["", "### Raw scoring", ""]
    raw = result.get("raw")
    if raw:
        state_lengths = raw_state_lengths(golden_path)
        state_count = max((len(row.get("per_state_ms", [])) for row in raw), default=0)
        used_lengths = state_lengths[:state_count]
        short_count = sum(length <= long_threshold for length in used_lengths)
        long_count = sum(length > long_threshold for length in used_lengths)
        lines.append(markdown_table(["Pair", "Engine", "Threads", "Mean ms", "p50 ms", "p95 ms",
                                     f"L≤{long_threshold} mean ms ({short_count} states)",
                                     f"L>{long_threshold} mean ms ({long_count} states)"],
            [[row["pair"], row["engine"], row["threads"], f"{row['mean_ms']:.2f}",
              f"{row['p50_ms']:.2f}", f"{row['p95_ms']:.2f}",
              category_mean(row, used_lengths, lambda length: length <= long_threshold),
              category_mean(row, used_lengths, lambda length: length > long_threshold)] for row in raw]))
    else:
        lines.append("pending: reviewer run")
    lines += ["", "### Sequence-length sweep", ""]
    sweep = result.get("length_sweep")
    if isinstance(sweep, dict) and sweep.get("rows"):
        sweep_rows = []
        for row in sweep["rows"]:
            f32, quantized = row["timings"]["f32"], row["timings"]["quantized"]
            sf, of = f32["statim"]["p50_ms"], f32["ort"]["p50_ms"]
            sq, oq = quantized["statim"]["p50_ms"], quantized["ort"]["p50_ms"]
            sweep_rows.append([row["length"], row["threads"], f"{sf:.2f}", f"{of:.2f}",
                               f"{sf / of:.2f}", f"{sq:.2f}", f"{oq:.2f}", f"{sq / oq:.2f}"])
        lines.append(markdown_table(["Length", "Threads", "Statim f32 ms", "ORT f32 ms", "Ratio",
                                     "Statim q8_0 ms", "ORT int8 ms", "Ratio"], sweep_rows))
        for skipped in sweep.get("skipped", []):
            lines.append(f"\nSkipped length {skipped['length']}: {skipped['reason']}.")
    else:
        lines.append("pending: reviewer run")
    overhead = result.get("binding_overhead")
    if overhead:
        lines += ["", "### ORT Python-binding overhead", ""]
        lines.append(markdown_table(["Threads", "Calls", "Median session.run ms", "Median model_run ms",
                                     "Median overhead µs", "Max overhead µs", "Median share"],
            [[row["threads"], row["calls"], f"{row['median_wall_us'] / 1000:.2f}",
              f"{row['median_model_run_us'] / 1000:.2f}", f"{row['median_overhead_us']:.0f}",
              f"{row['max_overhead_us']:.0f}", f"{row['median_overhead_share']:.4%}"] for row in overhead]))
    lines += ["", "### HTTP end to end", ""]
    socket = result.get("socket")
    if isinstance(socket, dict) and socket.get("rows"):
        if socket.get("pinned"):
            first = socket["rows"][0]
            lines.append(f"Pinned: server CPUs {first.get('server_cpus')}, load generator CPUs {first.get('client_cpus')} "
                         "(first row; per row in the JSON).\n")
        lines.append(markdown_table(["Variant", "Engine", "Threads", "Workers", "ORT spin", "Clients", "Microbatch",
                                     "p50 ms", "p95 ms", "p99 ms", "req/s", "Busy cores", "Ctx switches/req"],
            [[row["variant"], row["engine"], row["threads"], row.get("workers", 1) if row["engine"] == "statim" else "—",
              row.get("ort_spinning", 1) if row["engine"] == "ort" else "—", row["concurrency"], row["microbatch"],
              f"{row['p50_ms']:.2f}", f"{row['p95_ms']:.2f}",
              f"{row['p99_ms']:.2f}" if "p99_ms" in row else "—", f"{row['throughput_rps']:.2f}",
              f"{row['avg_busy_cores']:.2f}" if "avg_busy_cores" in row else "—",
              f"{row['context_switches_per_request']:.0f}" if "context_switches_per_request" in row else "—"]
             for row in socket["rows"]]))
    else:
        lines.append("pending: reviewer run")
    lines += ["", "### Cold start and peak RSS", ""]
    cold = socket.get("cold_start_memory") if isinstance(socket, dict) else None
    if cold:
        lines.append(markdown_table(["Variant", "Engine", "Threads", "ORT spin", "First answer ms", "Peak RSS MiB"],
            [[row["variant"], row["engine"], row["threads"], row.get("ort_spinning", 1) if row["engine"] == "ort" else "—",
              f"{row['cold_start_to_first_answer_ms']:.2f}", f"{row['peak_rss_bytes'] / 2**20:.1f}"] for row in cold]))
    else:
        lines.append("pending: reviewer run")
    inv = result.get("inventory", {})
    lines += ["", "### Artifact sizes", ""]
    lines.append(markdown_table(["Artifact", "MiB"],
        [[name, f"{value / 2**20:.1f}"] for name, value in inv.get("artifact_bytes", {}).items()]))
    lines += ["", "### Install footprint", ""]
    lines.append(markdown_table(["Install", "MiB"],
        [[name, f"{value / 2**20:.1f}"] for name, value in inv.get("install_bytes", {}).items()]))
    lines += ["", "### Held-out accuracy", "", "pending: reviewer run"]
    return "\n".join(lines) + "\n"


def subset_rows(rows: list[dict], maximum: int) -> list[dict]:
    if not maximum:
        return rows
    keep = {int(row["state_index"]) for row in rows}
    selected = set(sorted(keep)[:maximum])
    return [row for row in rows if int(row["state_index"]) in selected]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serve-ort", action="store_true")
    ap.add_argument("--socket-benchmark", action="store_true")
    ap.add_argument("--length-sweep", action="store_true")
    ap.add_argument("--binding-overhead", action="store_true",
                    help="measure the Python-binding cost of ORT session.run against its profiled model_run time")
    ap.add_argument("--ort-only", action="store_true", help="omit Statim rows (for pending GPU EP runs)")
    ap.add_argument("--ort-provider", choices=("cpu", "cuda", "tensorrt"), default="cpu")
    ap.add_argument("--checks-only", action="store_true")
    ap.add_argument("--inventory-only", action="store_true")
    ap.add_argument("--render-md", type=Path, metavar="RESULTS")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--statim-port", type=int, default=8091)
    ap.add_argument("--ort-port", type=int, default=8092)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--thread-counts", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 512, 1024])
    ap.add_argument("--long-threshold", type=int, default=256)
    ap.add_argument("--statim-workers", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--pin", action="store_true",
                    help="HTTP benchmark: pin each server to one logical CPU per physical core (taskset) and the load generator to other CPUs")
    ap.add_argument("--ort-spinning", type=int, choices=(0, 1), default=1, help="--serve-ort: intra-op thread spinning")
    ap.add_argument("--ort-spinning-variants", type=int, nargs="+", choices=(0, 1), default=[1],
                    help="HTTP benchmark: ORT server rows per spinning setting")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-states", type=int, default=0, help="smoke-test subset; 0 means all 30")
    ap.add_argument("--requests", type=int, default=60)
    ap.add_argument("--warmup-requests", type=int, default=4)
    ap.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    ap.add_argument("--onnx", type=Path, default=ONNX_F32, help="model used by --serve-ort")
    ap.add_argument("--onnx-f32", type=Path, default=ONNX_F32)
    ap.add_argument("--onnx-int8", type=Path, default=ONNX_INT8)
    ap.add_argument("--onnx-int8-dynamic", type=Path, default=ONNX_INT8_DYNAMIC)
    ap.add_argument("--gguf-f32", type=Path, default=GGUF_F32)
    ap.add_argument("--gguf-q8", type=Path, default=GGUF_Q8)
    ap.add_argument("--golden", type=Path, default=V9_GOLDEN)
    ap.add_argument("--template-golden", type=Path, default=TEMPLATE_GOLDEN)
    ap.add_argument("--inputs", type=Path, default=INPUTS)
    ap.add_argument("--report", type=Path, default=OUT / "export-report.json")
    ap.add_argument("--statim", type=Path, default=ROOT / "build-rel/statim")
    ap.add_argument("--statim-parity", type=Path, default=ROOT / "build-rel/test_model_parity")
    ap.add_argument("--release-statim", type=Path, default=RELEASE)
    ap.add_argument("--ort-python", type=Path, default=OUT / "venv-server/bin/python")
    ap.add_argument("--out", type=Path, default=OUT / "compare-results.json")
    args = ap.parse_args()
    if args.render_md:
        print(render_md(args.render_md, args.golden, args.long_threshold), end="")
        return 0
    if args.serve_ort:
        serve_ort(args)
        return 0
    if args.repeats < 1 or args.max_states < 0:
        ap.error("--repeats must be positive and --max-states non-negative")
    if any(value < 1 or value > 16 for value in args.thread_counts):
        ap.error("--thread-counts values must be in 1..16")
    if any(value < 1 for value in args.lengths):
        ap.error("--lengths values must be positive")
    if args.long_threshold < 1:
        ap.error("--long-threshold must be positive")
    if any(value < 1 or value > 64 for value in args.statim_workers):
        ap.error("--statim-workers values must be in 1..64")

    export = require_parity(args.report)
    base = {"cpu_only": True, "machine": machine_info(), "versions": versions(export, args.statim),
            "inventory": inventory(), "export": export}
    if args.inventory_only:
        print(json.dumps(base, indent=2, sort_keys=True))
        return 0

    check_engine = ORTEngine(args.model_dir, args.onnx_f32, min(4, args.threads), args.ort_provider)
    base["preflight"] = {
        "packing": packing_check(check_engine, args.inputs, args.template_golden),
        "answers": answer_check(check_engine, args.statim, args.gguf_f32, args.inputs),
    }
    del check_engine
    if args.checks_only:
        base["raw"] = None
        base["socket"] = "pending: reviewer run"
    elif args.socket_benchmark:
        base["raw"] = None
        base["socket"] = socket_benchmark(args)
    elif args.binding_overhead:
        base["raw"] = None
        rows = subset_rows(json_rows(args.golden), args.max_states)
        base["binding_overhead"] = [binding_overhead(args.onnx_f32, rows, threads, args.repeats)
                                    for threads in args.thread_counts]
        base["socket"] = "pending: reviewer run"
    elif args.length_sweep:
        base["raw"] = None
        base["length_sweep"] = run_length_sweep(args)
        base["socket"] = "pending: reviewer run"
    else:
        rows, raw, parity_results = subset_rows(json_rows(args.golden), args.max_states), [], {}
        pairs = [("f32", args.gguf_f32, args.onnx_f32), ("quantized", args.gguf_q8, args.onnx_int8)]
        for threads in args.thread_counts:
            for pair, gguf, onnx in pairs:
                scratch = OUT / f"raw-{pair}-t{threads}-repeated.jsonl"
                invocation = [str(args.ort_python), str(Path(__file__).resolve()), "--thread-counts", str(threads),
                              "--repeats", str(args.repeats), "--ort-provider", args.ort_provider,
                              "--out", str(args.out)]
                if args.ort_only:
                    invocation.insert(-2, "--ort-only")
                else:
                    statim_timing, statim_p = statim_raw(args.statim_parity, gguf, rows, threads,
                                                         args.repeats, scratch)
                    raw.append({"pair": pair, **statim_timing})
                    parity_results[f"{pair}_statim_t{threads}"] = statim_p
                ort_timing, ort_p = ort_raw(onnx, rows, threads, args.repeats, invocation, args.ort_provider)
                raw.append({"pair": pair, **ort_timing})
                parity_results[f"{pair}_ort_t{threads}"] = ort_p
            if args.onnx_int8_dynamic and args.onnx_int8_dynamic.exists():
                invocation = [str(args.ort_python), str(Path(__file__).resolve()), "--thread-counts", str(threads),
                              "--repeats", str(args.repeats), "--ort-provider", args.ort_provider,
                              "--out", str(args.out)]
                if args.ort_only:
                    invocation.insert(-2, "--ort-only")
                ort_timing, ort_p = ort_raw(args.onnx_int8_dynamic, rows, threads, args.repeats, invocation, args.ort_provider)
                raw.append({"pair": "dynamic_int8", **ort_timing})
                parity_results[f"dynamic_int8_ort_t{threads}"] = ort_p
        base["raw"], base["raw_parity"] = raw, parity_results
        base["limited_states"] = len(state_groups(rows))
        base["socket"] = "pending: reviewer run"
    base["accuracy"] = "pending: reviewer run; held-out splits may require network"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(base, indent=2, sort_keys=True) + "\n")
    print(json.dumps(base, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
