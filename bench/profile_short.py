#!/usr/bin/env python3
"""Profile Statim and ONNX Runtime on the short-input length-sweep batches."""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ORT_TELEMETRY_DISABLED"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.ort_compare import (INPUTS, MODEL_DIR, ONNX_F32, add_reference, collate, length_sweep_rows,
                               repeated_golden, session, ORTEngine)

CATEGORIES = [
    "token embedding", "QKV projection", "rotary",
    "attention scores, softmax and weighted sum (and masks)",
    "attention output projection", "MLP up (Wi) and the GeGLU/GELU activation",
    "MLP down (Wo)", "LayerNorm and residual adds", "decision head",
    "copies, reshapes and transposes", "other",
]
COPY_OPS = {"Reshape", "Transpose", "Slice", "Squeeze", "Unsqueeze", "Expand", "Concat",
            "Split", "Shape", "GatherND", "GatherElements", "Cast"}
ATTN_OPS = {"Softmax", "Where", "IsNaN", "GreaterOrEqual", "LessOrEqual", "And", "Not"}


def statim_category(node: dict) -> str:
    name, op = node["name"], node["op"]
    if name.startswith("embedding.token"):
        return "token embedding"
    if ".attn.qkv." in name or ".attn.kv." in name or re.search(r"\.attn\.q\.matmul$", name):
        return "QKV projection"
    if ".rotary" in name:
        return "rotary"
    if any(x in name for x in (".attn.scores", ".attn.softmax", ".attn.weighted_sum", ".attn.flash")):
        return "attention scores, softmax and weighted sum (and masks)"
    if ".attn.output" in name:
        return "attention output projection"
    if ".mlp.up" in name or ".mlp.geglu" in name or ".mlp.relu" in name:
        return "MLP up (Wi) and the GeGLU/GELU activation"
    if ".mlp.down" in name:
        return "MLP down (Wo)"
    if "layernorm" in name or ".residual" in name:
        return "LayerNorm and residual adds"
    if name.startswith("decision.") or name.startswith("head.type_embedding"):
        return "decision head"
    if name.startswith("head."):
        # Decision-transformer attention/MLP nodes retain their functional categories above.
        if op in {"NORM", "ADD", "MUL"}:
            return "LayerNorm and residual adds"
    if op in {"RESHAPE", "VIEW", "PERMUTE", "CONT", "CPY", "GET_ROWS"} or "copy" in name or "transpose" in name:
        return "copies, reshapes and transposes"
    return "other"


def linear_category(index: int) -> str:
    if index < 88:
        return ("QKV projection", "attention output projection",
                "MLP up (Wi) and the GeGLU/GELU activation", "MLP down (Wo)")[index % 4]
    if index < 96:
        return ("QKV projection", "attention output projection",
                "MLP up (Wi) and the GeGLU/GELU activation", "MLP down (Wo)")[(index - 88) % 4]
    return "decision head"


def ort_category(event: dict) -> str:
    name = event.get("name", "")
    op = event.get("args", {}).get("op_name", "")
    m = re.search(r"(?:node_)?linear_(\d+)(?:_|$)", name)
    if m:
        return linear_category(int(m.group(1)))
    if name == "node_linear_kernel_time":
        return "QKV projection"
    if "embedding_kernel" in name and "embedding_1" not in name:
        return "token embedding"
    if "embedding_1" in name:
        return "decision head"
    if "layer_norm" in name or op == "LayerNormalization" or op == "Add":
        return "LayerNorm and residual adds"
    if op == "MatMul":
        outputs = event.get("args", {}).get("output_type_shape", [])
        inputs = event.get("args", {}).get("input_type_shape", [])
        out_shape = next(iter(outputs[0].values()), []) if outputs else []
        in_shape = next(iter(inputs[0].values()), []) if inputs else []
        if out_shape and out_shape[-1] == 2304:
            return "QKV projection"
        if out_shape and out_shape[-1] == 3072:
            return "MLP up (Wi) and the GeGLU/GELU activation"
        if in_shape and in_shape[-1] == 3072 and out_shape and out_shape[-1] == 768:
            return "MLP down (Wo)"
        if out_shape and out_shape[-1] in {1, 2, 256, 768}:
            return "decision head"
    if "MatMulScaleFusion" in name or "scaled_dot_product_attention" in name or op in ATTN_OPS:
        return "attention scores, softmax and weighted sum (and masks)"
    if op in {"Sin", "Cos", "Neg", "Range"}:
        return "rotary"
    if op in {"Erf", "Relu"} or "gelu" in name:
        return "MLP up (Wi) and the GeGLU/GELU activation"
    if op in COPY_OPS:
        return "copies, reshapes and transposes"
    if op in {"Gemm", "TopK", "ReduceSum", "Log", "Clip"}:
        return "decision head"
    return "other"


def aggregate(nodes: list[dict], category, duration_key: str, scale: float) -> dict[str, float]:
    out = defaultdict(float)
    for node in nodes:
        out[category(node)] += float(node[duration_key]) * scale
    return {name: out[name] for name in CATEGORIES}


def aggregate_ort(nodes: list[dict]) -> dict[str, float]:
    """Use the export's repeated block order to classify generic rotary/activation arithmetic."""
    out = defaultdict(float)
    state = ""
    for node in nodes:
        cat = ort_category(node)
        op = node.get("args", {}).get("op_name", "")
        name = node.get("name", "")
        if cat == "QKV projection":
            state = "rotary"
        elif "MatMulScaleFusion" in name:
            cat, state = "attention scores, softmax and weighted sum (and masks)", "attention"
        elif cat == "attention output projection":
            state = ""
        elif cat == "MLP up (Wi) and the GeGLU/GELU activation" and ("linear_" in name or op == "MatMul"):
            state = "activation"
        elif cat == "MLP down (Wo)":
            state = ""
        elif state == "rotary":
            cat = "copies, reshapes and transposes" if op in COPY_OPS else "rotary"
        elif state == "attention":
            cat = ("copies, reshapes and transposes" if op in COPY_OPS else
                   "attention scores, softmax and weighted sum (and masks)")
        elif state == "activation":
            cat = "MLP up (Wi) and the GeGLU/GELU activation"
        out[cat] += float(node["dur"]) / 1000
    return {name: out[name] for name in CATEGORIES}


def median_dict(runs: list[dict[str, float]]) -> dict[str, float]:
    return {key: statistics.median(run[key] for run in runs) for key in CATEGORIES}


def profile_statim(binary: Path, gguf: Path, rows: list[dict], length: int,
                   threads: int, repeats: int, work: Path) -> dict:
    golden = work / f"profile-short-l{length}-t{threads}.jsonl"
    trace = work / f"statim-profile-l{length}-t{threads}.jsonl"
    repeated_golden(rows, repeats, golden)
    trace.unlink(missing_ok=True)
    command = [str(binary), str(gguf), str(golden), "1e9", str(threads)]
    proc = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False,
                          env={**os.environ, "STATIM_DEVICE": "cpu", "STATIM_PROFILE": str(trace)})
    if proc.returncode:
        raise RuntimeError("Statim profiling failed:\n" + proc.stdout[-4000:])
    raw = [json.loads(line) for line in trace.read_text().splitlines() if line]
    trace.unlink()
    golden.unlink()
    if len(raw) != repeats + 1:
        raise RuntimeError(f"Statim produced {len(raw)} runs, expected {repeats + 1}")
    measured = raw[1:]
    runs = [aggregate(run["nodes"], statim_category, "ns", 1e-6) for run in measured]
    phases = {key.removesuffix("_ns") + "_ms": statistics.median(r["phases"][key] for r in measured) / 1e6
              for key in measured[0]["phases"]}
    return {"command": command, "categories_ms": median_dict(runs), "category_runs_ms": runs,
            "phases_ms": phases, "raw_runs": raw}


def split_ort_runs(events: list[dict], count: int) -> tuple[list[list[dict]], list[float]]:
    model_runs = [e for e in events if e.get("cat") == "Session" and e.get("name") == "model_run"]
    nodes = [e for e in events if e.get("cat") == "Node" and e.get("name", "").endswith("kernel_time")]
    if len(model_runs) != count:
        raise RuntimeError(f"ORT profile has {len(model_runs)} model runs, expected {count}")
    grouped = []
    for run in model_runs:
        lo, hi = run["ts"], run["ts"] + run["dur"]
        grouped.append([e for e in nodes if lo <= e["ts"] < hi])
    return grouped, [float(e["dur"]) / 1000 for e in model_runs]


def profile_ort(onnx: Path, rows: list[dict], threads: int, repeats: int) -> dict:
    sess = session(onnx, threads, profile=True)
    feed = collate(rows)
    walls = []
    for _ in range(repeats + 1):
        start = time.perf_counter_ns()
        sess.run(["logits", "act_logits"], feed)
        walls.append((time.perf_counter_ns() - start) / 1e6)
    trace = Path(sess.end_profiling())
    events = json.loads(trace.read_text())
    grouped, model_ms = split_ort_runs(events, repeats + 1)
    runs = [aggregate_ort(nodes) for nodes in grouped[1:]]
    result = {"categories_ms": median_dict(runs), "category_runs_ms": runs,
              "wall_runs_ms": walls[1:], "total_ms": statistics.median(walls[1:]),
              "model_run_ms": statistics.median(model_ms[1:]),
              "raw_runs": grouped, "profile_path": str(trace)}
    trace.unlink(missing_ok=True)
    return result


def make_table(result: dict) -> str:
    s, o = result["statim"], result["ort"]
    st, ot = s["phases_ms"]["total_ms"], o["total_ms"]
    lines = ["| Category | Statim ms | ORT ms | Difference | Statim share | ORT share |",
             "|---|---:|---:|---:|---:|---:|"]
    for cat in CATEGORIES:
        sv, ov = s["categories_ms"][cat], o["categories_ms"][cat]
        lines.append(f"| {cat} | {sv:.3f} | {ov:.3f} | {sv-ov:+.3f} | {sv/st:.1%} | {ov/ot:.1%} |")
    su, ou = st - sum(s["categories_ms"].values()), ot - sum(o["categories_ms"].values())
    lines.append(f"| unaccounted | {su:.3f} | {ou:.3f} | {su-ou:+.3f} | {su/st:.1%} | {ou/ot:.1%} |")
    lines.append(f"| **total run** | **{st:.3f}** | **{ot:.3f}** | **{st-ot:+.3f}** | **100.0%** | **100.0%** |")
    return "\n".join(lines)


def render_report(results: list[dict]) -> str:
    repeats = results[0]["repeats"] if results else 0
    out = ["# Short-input CPU profile", "",
           f"CPU-only f32 runs; each result is the median of {repeats} runs after one warm-up. "
           "Statim uses one-node `ggml_graph_view` execution; ORT uses its kernel profiler. "
           "Times therefore include profiler dispatch overhead and are diagnostic, not replacement benchmark numbers.", ""]
    for r in results:
        out += [f"## L={r['length']}, {r['threads']} thread{'s' if r['threads'] != 1 else ''}", "", make_table(r), ""]
    gaps = []
    for r in results:
        for cat in CATEGORIES:
            gaps.append((r["statim"]["categories_ms"][cat] - r["ort"]["categories_ms"][cat],
                         r["length"], r["threads"], cat))
    out += ["## Largest category gaps", ""]
    for i, (gap, length, threads, cat) in enumerate(sorted(gaps, reverse=True)[:10], 1):
        out.append(f"{i}. L={length}, t={threads}: **{cat}**, {gap:+.3f} ms (Statim minus ORT).")
    out += ["", "The causes behind the leading positive gaps are:", "",
            "- **QKV and the other projections:** ggml broadcasts the 2-D weight over the batch dimension and calls the f32 matmul path per batch slice. With B=8, the source loop presents eight `(m,n,k)` SGEMMs such as `(2304,L,768)` instead of one projection over `L*B` rows. ORT hands the corresponding f32 MatMul to MLAS, whose profiled output is `[8,L,out]` and whose implementation treats the outer rows as one large GEMM.",
            "- **Wi plus GeGLU:** the same projection issue applies to the wide Wi matrix. Statim then writes `[2*ff,L,B]` and rereads it in the custom `geglu_op`; ORT uses MLAS for MatMul and vectorized Split/Div/Erf/Add/Mul kernels.",
            "- **Attention output and MLP down:** these are again f32 projection SGEMMs, so their smaller but consistent gaps have the same batching, packing and AVX2 microkernel cause.",
            "- **Attention at L=128/t=1:** Statim separately dispatches the head-batched score matmul, masked softmax, and weighted-sum matmul. ORT reports MLAS-backed MatMul, `MatMulScaleFusion`, and optimized softmax/mask kernels. At four threads ORT's attention is slower, so this is not the first target.",
            "- **LayerNorm/residual:** ggml schedules norm, scale and residual as separate full-tensor passes; ORT uses its vectorized LayerNormalization kernel and optimized elementwise scheduling.",
            "", "## Kernel interpretation", "",
            "For f32 projection nodes, ggml's CPU `MUL_MAT` first calls `llamafile_sgemm` when the activation is contiguous. "
            "On this AVX2 build that selects llamafile tinyBLAS' 8-float SIMD tile; if the operand/layout is rejected, ggml falls back to its `vec_dot` chunk loop. "
            "ORT's f32 MatMul/Gemm kernels use MLAS SGEMM, while its optimized graph also fuses attention-score scaling (`MatMulScaleFusion`) and some bias operations.", "",
            "The explicit Statim attention path separately runs score matmul, masked softmax and value weighted-sum kernels. "
            "The encoder GeGLU is Statim's custom row kernel; ORT expresses Split/Div/Erf/Add/Mul and schedules optimized elementwise kernels. "
            "Layer normalization and residual adds are separate ggml nodes, increasing thread-pool dispatch and full-tensor memory traffic at short lengths.", "",
            "## Ranked optimization candidates (not implemented)", "",
            "1. **Flatten `L*B` for every dense projection:** reshape the contiguous `[d,L,B]` activation to `[d,L*B]` around QKV/Wi/Wo and reshape the result back. This changes eight broadcast SGEMMs into one larger `llamafile_sgemm`, improving weight reuse and worker scheduling without changing dot-product arithmetic. It directly targets every top projection gap.",
            "2. **Improve/replace the f32 projection microkernel and packing:** benchmark MLAS or another proven AVX2 SGEMM against llamafile tinyBLAS for `(out,L*B,in)`; alternatively add persistent packed weights and tune tinyBLAS tiles. Do this after flattening so the comparison uses the intended large GEMM shapes.",
            "3. **Fuse the encoder Wi + GeGLU path:** consume the two Wi halves directly into gated output, avoiding an intermediate 2×FF tensor and separate dispatch/read. Preserve the exact erf arithmetic/order required by parity.",
            "4. **Fuse attention score scaling + mask + softmax:** eliminate separate launches and intermediate score passes for the explicit f32 path. The one-thread L=128 result supports it, but the four-thread result makes it lower priority than projections.",
            "5. **Fuse LayerNorm scale/bias and residual passes:** reduce dispatch and memory traffic; use size-aware thread thresholds for these short elementwise kernels.",
            "6. **Remove Q/K/V materialization where profitable:** make attention consume strided QKV views or write the projection directly in attention layout. ORT currently spends more in the broad copy/reshape category, so this ranks below the positive Statim gaps.", ""]
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--statim", type=Path, default=ROOT / "build-rel/test_model_parity")
    p.add_argument("--gguf", type=Path, default=ROOT / "models/laya-multilingual-v9-f32.gguf")
    p.add_argument("--onnx", type=Path, default=ONNX_F32)
    p.add_argument("--lengths", type=int, nargs="+", default=[128, 256])
    p.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--out", type=Path, default=ROOT / "build-ort/profile-short.json")
    p.add_argument("--report", type=Path, default=ROOT / "build-ort/profile-short.md")
    args = p.parse_args()
    # the Statim subprocess runs in ROOT, so every path must be absolute before it is passed on
    for name in ("statim", "gguf", "onnx", "out", "report"):
        setattr(args, name, getattr(args, name).resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    reference = ORTEngine(MODEL_DIR, args.onnx, min(args.threads))
    batches, skipped = length_sweep_rows(reference, INPUTS, args.lengths)
    if skipped:
        raise RuntimeError(f"could not construct requested batches: {skipped}")
    for rows in batches.values():
        add_reference(rows, reference)
    del reference
    results = []
    for length in args.lengths:
        for threads in args.threads:
            print(f"profiling L={length}, threads={threads}: Statim", flush=True)
            statim = profile_statim(args.statim, args.gguf, batches[length], length,
                                    threads, args.repeats, args.out.parent)
            print(f"profiling L={length}, threads={threads}: ORT", flush=True)
            ort = profile_ort(args.onnx, batches[length], threads, args.repeats)
            result = {"length": length, "threads": threads, "batch": len(batches[length]),
                      "warmup": 1, "repeats": args.repeats, "statim": statim, "ort": ort}
            results.append(result)
            print(make_table(result), flush=True)
    payload = {"protocol": {"cpu_only": True, "lengths": args.lengths, "threads": args.threads,
                             "warmup": 1, "repeats": args.repeats, "batch": 8}, "results": results}
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    args.report.write_text(render_report(results) + "\n")
    print(f"wrote {args.out} and {args.report}")


if __name__ == "__main__":
    main()
