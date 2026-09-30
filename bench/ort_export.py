#!/usr/bin/env python3
"""Export and prove the CPU graphs used by the Statim/ORT comparison.

Run with Laya's training interpreter and the isolated ONNX packages::

  CUDA_VISIBLE_DEVICES="" PYTHONPATH=build-ort/venv/lib/python3.12/site-packages \
    ../statim/.venv-train/bin/python bench/ort_export.py

The original checkpoint is checked against the independent repository golden.
The shipped v9 checkpoint gets a fresh PyTorch golden, which is then checked by
both ORT and Statim. Quantization is attempted only after every f32 gate passes.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ORT_TELEMETRY_DISABLED"] = "1"

import numpy as np
import torch

torch.set_num_threads(4)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "models/laya-multilingual"
SHIPPED = ROOT / "models/laya-multilingual-v9"
GOLDEN = ROOT / "tests/data/golden_laya-multilingual.jsonl"
OUT = ROOT / "build-ort"
INPUT_NAMES = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
OUTPUT_NAMES = ["logits", "act_logits"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def groups(rows: list[dict]) -> list[list[dict]]:
    by_state: dict[int, list[dict]] = {}
    for row in rows:
        by_state.setdefault(int(row["state_index"]), []).append(row)
    return [by_state[key] for key in sorted(by_state)]


def collate(rows: list[dict], pad_id: int) -> tuple[torch.Tensor, ...]:
    batch, seq = len(rows), max(len(row["ids"]) for row in rows)
    options = max(len(row["markers"]) for row in rows)
    ids = torch.full((batch, seq), pad_id, dtype=torch.int64)
    attention = torch.zeros((batch, seq), dtype=torch.int64)
    marker_pos = torch.zeros((batch, options), dtype=torch.int64)
    marker_mask = torch.zeros((batch, options), dtype=torch.bool)
    qtype = torch.empty(batch, dtype=torch.int64)
    for i, row in enumerate(rows):
        n, k = len(row["ids"]), len(row["markers"])
        ids[i, :n] = torch.tensor(row["ids"], dtype=torch.int64)
        attention[i, :n] = 1
        marker_pos[i, :k] = torch.tensor(row["markers"], dtype=torch.int64)
        marker_mask[i, :k] = True
        qtype[i] = row["qtype"]
    return ids, attention, marker_pos, marker_mask, qtype


def make_session(path: Path, threads: int):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def softmax(values: np.ndarray) -> np.ndarray:
    z = values - values.max(axis=-1, keepdims=True)
    out = np.exp(z)
    return out / out.sum(axis=-1, keepdims=True)


def parity(path: Path, rows: list[dict], pad_id: int, threads: int) -> dict:
    sess = make_session(path, threads)
    max_logit = max_act = 0.0
    agreed = total = 0
    for state_rows in groups(rows):
        tensors = collate(state_rows, pad_id)
        feed = {name: value.numpy() for name, value in zip(INPUT_NAMES, tensors)}
        logits, act_logits = sess.run(OUTPUT_NAMES, feed)
        act = softmax(act_logits)
        for i, row in enumerate(state_rows):
            ref = np.asarray(row["logits"], dtype=np.float32)
            got = logits[i, : len(ref)]
            max_logit = max(max_logit, float(np.max(np.abs(got - ref))))
            max_act = max(max_act, float(np.max(np.abs(act[i] - np.asarray(row["act"], dtype=np.float32)))))
            agreed += int(int(np.argmax(got)) == int(np.argmax(ref)))
            total += 1
    return {"items": total, "argmax_agree": agreed, "max_abs_logit": max_logit,
            "max_abs_act": max_act}


def gate(values: dict, tolerance: float) -> dict:
    return {**values, "tolerance": tolerance,
            "pass": values["items"] == values["argmax_agree"] and values["max_abs_logit"] <= tolerance}


def drop_value_info(path: Path) -> None:
    """Remove exporter hints which contradict Gemm's inferred weight orientation."""
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    if model.graph.value_info:
        del model.graph.value_info[:]
        onnx.save(model, str(path))


def externalize(path: Path) -> list[Path]:
    """Store tensors in one named sidecar, making size accounting deterministic."""
    import onnx

    header = onnx.load(str(path), load_external_data=False)
    locations = {entry.value for tensor in header.graph.initializer for entry in tensor.external_data
                 if entry.key == "location"}
    if locations and all((path.parent / name).is_file() for name in locations):
        return [path, *(path.parent / name for name in sorted(locations))]
    model = onnx.load(str(path), load_external_data=True)
    sidecar = path.with_suffix(path.suffix + ".data")
    sidecar.unlink(missing_ok=True)
    onnx.save_model(model, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=sidecar.name, size_threshold=1024, convert_attribute=False)
    return [path, sidecar] if sidecar.exists() else [path]


def export(agent, sample: tuple[torch.Tensor, ...], path: Path, opset: int) -> None:
    path.unlink(missing_ok=True)
    path.with_suffix(path.suffix + ".data").unlink(missing_ok=True)
    torch.onnx.export(
        agent.model, sample, str(path), input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"}, "attention_mask": {0: "batch", 1: "sequence"},
            "marker_pos": {0: "batch", 1: "options"}, "marker_mask": {0: "batch", 1: "options"},
            "qtype": {0: "batch"}, "logits": {0: "batch", 1: "options"},
            "act_logits": {0: "batch"},
        },
        opset_version=opset, do_constant_folding=True, dynamo=True,
    )
    externalize(path)
    drop_value_info(path)


def pytorch_golden(agent, template: list[dict], path: Path) -> list[dict]:
    rows: list[dict] = []
    with torch.no_grad():
        for state_rows in groups(template):
            tensors = collate(state_rows, agent.tok.pad_token_id)
            logits, act_logits = agent.model(*tensors)
            act = torch.softmax(act_logits.float(), -1)
            for i, source in enumerate(state_rows):
                k = len(source["markers"])
                rows.append({
                    "state_index": int(source["state_index"]), "question": source["question"],
                    "ids": source["ids"], "markers": source["markers"], "qtype": int(source["qtype"]),
                    "logits": logits[i, :k].float().cpu().tolist(), "act": act[i].cpu().tolist(),
                })
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(paths: list[Path]) -> dict:
    existing = [path for path in paths if path.exists()]
    return {"bytes": sum(path.stat().st_size for path in existing),
            "files": [str(path.relative_to(ROOT)) for path in existing]}


def versions() -> dict[str, str]:
    import laya
    import onnx
    import onnxruntime

    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "laya": getattr(laya, "__version__", importlib.metadata.version("laya")),
            "onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__,
            "numpy": np.__version__}


def statim_parity(binary: Path, model: Path, golden: Path, tolerance: float, threads: int) -> dict:
    command = [str(binary), str(model), str(golden), str(tolerance), str(threads)]
    proc = subprocess.run(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    match = re.search(r"items (\d+) \| argmax agree (\d+)/(\d+) \| max \|dlogit\| ([0-9.eE+-]+) "
                      r"\| max \|dact\| ([0-9.eE+-]+)", proc.stdout)
    if not match:
        raise RuntimeError("could not parse test_model_parity output:\n" + proc.stdout[-3000:])
    result = {"items": int(match.group(1)), "argmax_agree": int(match.group(2)),
              "max_abs_logit": float(match.group(4)), "max_abs_act": float(match.group(5)),
              "tolerance": tolerance, "exit_code": proc.returncode, "command": command}
    result["pass"] = result["items"] == result["argmax_agree"] and result["max_abs_logit"] <= tolerance
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--original-model", type=Path, default=ORIGINAL)
    ap.add_argument("--shipped-model", type=Path, default=SHIPPED)
    ap.add_argument("--golden", type=Path, default=GOLDEN)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--statim-parity", type=Path, default=ROOT / "build-rel/test_model_parity")
    ap.add_argument("--statim-quantize", type=Path, default=ROOT / "build-rel/statim-quantize")
    ap.add_argument("--shipped-gguf", type=Path, default=ROOT / "models/laya-multilingual-v9-f32.gguf")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--tolerance", type=float, default=1e-3)
    ap.add_argument("--skip-export", action="store_true", help="reuse existing f32 ONNX files")
    args = ap.parse_args()
    if not 1 <= args.threads <= 4:
        ap.error("--threads must be in 1..4 while exporting")
    try:
        import laya
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError as exc:
        ap.error(f"missing {exc.name}; use the interpreter/PYTHONPATH shown in the docstring")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    template = read_jsonl(args.golden)
    original_onnx = args.out_dir / "laya-multilingual-f32.onnx"
    shipped_onnx = args.out_dir / "laya-multilingual-v9-f32.onnx"
    int8_onnx = args.out_dir / "laya-multilingual-v9-int8.onnx"
    dynamic_onnx = args.out_dir / "laya-multilingual-v9-int8-dynamic.onnx"
    blockwise_onnx = args.out_dir / "laya-multilingual-v9-8bit-blockwise.onnx"
    v9_golden_path = args.out_dir / "golden_v9.jsonl"
    q8_path = args.out_dir / "laya-multilingual-v9-q8_0.gguf"
    report_path = args.out_dir / "export-report.json"

    original_agent = laya.load(str(args.original_model.resolve()), device="cpu")
    original_agent.model.eval()
    if not args.skip_export:
        export(original_agent, collate(groups(template)[0], original_agent.tok.pad_token_id), original_onnx, args.opset)
    else:
        drop_value_info(original_onnx)
    original = gate(parity(original_onnx, template, original_agent.tok.pad_token_id, args.threads), args.tolerance)
    print(json.dumps({"variant": "original_ort_f32", **original}, sort_keys=True), flush=True)
    del original_agent
    gc.collect()

    shipped_agent = laya.load(str(args.shipped_model.resolve()), device="cpu")
    shipped_agent.model.eval()
    v9_rows = pytorch_golden(shipped_agent, template, v9_golden_path)
    if not args.skip_export:
        export(shipped_agent, collate(groups(v9_rows)[0], shipped_agent.tok.pad_token_id), shipped_onnx, args.opset)
    else:
        drop_value_info(shipped_onnx)
    shipped_ort = gate(parity(shipped_onnx, v9_rows, shipped_agent.tok.pad_token_id, args.threads), args.tolerance)
    print(json.dumps({"variant": "shipped_ort_f32", **shipped_ort}, sort_keys=True), flush=True)
    del shipped_agent
    gc.collect()

    shipped_statim = statim_parity(args.statim_parity, args.shipped_gguf, v9_golden_path,
                                   args.tolerance, args.threads)
    print(json.dumps({"variant": "shipped_statim_f32", **shipped_statim}, sort_keys=True), flush=True)
    f32_pass = original["pass"] and shipped_ort["pass"] and shipped_statim["pass"]
    if not f32_pass:
        report = {"f32_gates_pass": False, "parity": {"original_ort_f32": original,
                  "shipped_ort_f32": shipped_ort, "shipped_statim_f32": shipped_statim},
                  "versions": versions()}
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("an f32 parity gate failed; refusing to quantize", file=sys.stderr)
        return 1

    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_onnx.unlink(missing_ok=True)
    int8_onnx.with_suffix(int8_onnx.suffix + ".data").unlink(missing_ok=True)
    quantize_dynamic(str(shipped_onnx), str(int8_onnx), weight_type=QuantType.QInt8,
                     per_channel=False, reduce_range=False, use_external_data_format=True)
    externalize(int8_onnx)
    shipped_int8 = parity(int8_onnx, v9_rows, 0, args.threads)
    print(json.dumps({"variant": "shipped_ort_int8", **shipped_int8}, sort_keys=True), flush=True)

    # The ORT advocate's two alternatives (docs/ORT.md): the most accurate dynamic int8 setting on
    # AVX2 without VNNI, and 8-bit blockwise weights (block 32, f32 compute), ORT's closest
    # counterpart of q8_0. bench/ort_compare.py uses these two by default.
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    for path in (dynamic_onnx, blockwise_onnx):
        path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".data").unlink(missing_ok=True)
    quantize_dynamic(str(shipped_onnx), str(dynamic_onnx), weight_type=QuantType.QInt8,
                     per_channel=True, reduce_range=True, use_external_data_format=True)
    externalize(dynamic_onnx)
    shipped_dynamic = parity(dynamic_onnx, v9_rows, 0, args.threads)
    print(json.dumps({"variant": "shipped_ort_int8_dynamic", **shipped_dynamic}, sort_keys=True), flush=True)
    quantizer = MatMulNBitsQuantizer(model=str(shipped_onnx), bits=8, block_size=32, is_symmetric=True,
                                     accuracy_level=0)
    quantizer.process()
    quantizer.model.save_model_to_file(str(blockwise_onnx), use_external_data_format=True)
    externalize(blockwise_onnx)
    shipped_blockwise = parity(blockwise_onnx, v9_rows, 0, args.threads)
    print(json.dumps({"variant": "shipped_ort_8bit_blockwise", **shipped_blockwise}, sort_keys=True), flush=True)

    subprocess.run([str(args.statim_quantize), str(args.shipped_gguf), str(q8_path), "q8_0"], cwd=ROOT,
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, check=True)
    shipped_q8 = statim_parity(args.statim_parity, q8_path, v9_golden_path, 1e9, args.threads)
    print(json.dumps({"variant": "shipped_statim_q8_0", **shipped_q8}, sort_keys=True), flush=True)

    report = {
        "f32_gates_pass": True, "opset": args.opset, "versions": versions(),
        "checkpoints": {
            "original": {"path": str(args.original_model.relative_to(ROOT)),
                         "model_safetensors_sha256": sha256(args.original_model / "model.safetensors")},
            "shipped_v9": {"path": str(args.shipped_model.relative_to(ROOT)),
                           "model_safetensors_sha256": sha256(args.shipped_model / "model.safetensors")},
        },
        "goldens": {"original": str(args.golden.relative_to(ROOT)),
                    "shipped_v9": str(v9_golden_path.relative_to(ROOT))},
        "parity": {"original_ort_f32": original, "shipped_ort_f32": shipped_ort,
                   "shipped_statim_f32": shipped_statim, "shipped_ort_int8": shipped_int8,
                   "shipped_ort_int8_dynamic": shipped_dynamic, "shipped_ort_8bit_blockwise": shipped_blockwise,
                   "shipped_statim_q8_0": shipped_q8},
        "artifacts": {
            "original_onnx_f32": artifact([original_onnx, original_onnx.with_suffix(".onnx.data")]),
            "shipped_onnx_f32": artifact([shipped_onnx, shipped_onnx.with_suffix(".onnx.data")]),
            "shipped_onnx_int8": artifact([int8_onnx, int8_onnx.with_suffix(".onnx.data")]),
            "shipped_onnx_int8_dynamic": artifact([dynamic_onnx, dynamic_onnx.with_suffix(".onnx.data")]),
            "shipped_onnx_8bit_blockwise": artifact([blockwise_onnx, blockwise_onnx.with_suffix(".onnx.data")]),
            "shipped_gguf_f32": artifact([args.shipped_gguf]), "shipped_gguf_q8_0": artifact([q8_path]),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
