# Statim

**A native C++20 engine for System-1 decision models.** Typed decisions — `choice`, `score`, `noul` —
over any text or JSON in a single forward pass, served from one static binary. No Python, no PyTorch,
no GPU required.

Statim runs the open [Laya](https://github.com/NandhaKishorM/laya) checkpoints (Apache-2.0) and speaks
the Jev/Laya `POST /v1/systemone` protocol, so existing clients switch by changing the base URL.

> *statim* (Latin): immediately, at once.

```bash
statim serve -m multilingual=laya-multilingual-f32.gguf --port 8080
curl -s localhost:8080/v1/systemone -d '{
  "state": {"subject": "Duplicate charge on invoice #4411",
            "body": "We were billed twice for March. Please refund the duplicate today or we will cancel."},
  "questions": {
    "department": {"type": "choice", "instructions": "Which department should handle this request?",
                   "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages",
                                "sales": "pricing, new contracts", "other": "everything else"}},
    "urgency":    {"type": "score", "instructions": "How urgent is this request?",
                   "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
    "refund":     {"type": "noul", "instructions": "Does the user explicitly request a refund?"}}}'
```

## Why Statim

| | Laya (Python) | **Statim** |
|---|---|---|
| Runtime | Python 3.10+, PyTorch, transformers | one binary (~3 MB) + one `.gguf` file |
| Answers | reference | **identical** — 240/240 token sequences, answers within 1e-4 (CI-gated) |
| Tokenizer | HF `tokenizers` (Rust) | native C++, 100% identical on 3,906 cases + 140k fuzz strings, ~10× faster |
| Cold start | see [benchmarks](#benchmarks) | see [benchmarks](#benchmarks) |
| Memory | see [benchmarks](#benchmarks) | mmap'd weights, shared between processes |
| Accuracy mode | — | `ensemble: K` — option-order ensembling, see [accuracy](#accuracy) |
| Server | FastAPI, one inference at a time | worker pool, admission control (503), bearer auth, Prometheus `/metrics`, JSON logs, `/health` + `/ready`, request IDs, graceful shutdown |
| Deploy | pip / Docker | static binary, distroless Docker, hardened systemd unit |

## How it works

```
state + questions ──► prompt builder (byte-exact with laya.common.build_sequence)
                          │  [CLS] <type> question: … [SEP] [MASK] opt₀ [MASK] opt₁ … [SEP] state [SEP]
                          ▼
                 native BPE tokenizer (Metaspace/byte-fallback + GPT-2 ByteLevel)
                          ▼
   ggml graph: ModernBERT / mmBERT encoder (RoPE, alternating global / sliding-window attention,
               GeGLU, fused flash attention) ─► type embedding ─► 2-layer decision head
               ─► scorer on each [MASK] ─► temperature-calibrated softmax
                          ▼
     {"answers": {"department": {"choice": "billing", "probabilities": …, "confidence": …}, …}}
```

All three Laya checkpoints share this graph; the converter stores architecture, calibration and the
tokenizer in the GGUF file, so a model is a single self-describing artifact.

## Quick start

```bash
git clone --recursive https://github.com/BEKO2210/statim && cd statim
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build
pip install numpy safetensors gguf            # converter only; not needed at runtime
tools/fetch_models.sh multilingual english    # download from Hugging Face + convert to GGUF
ctest --test-dir build                        # parity gates against the official package
./build/statim serve -m models/laya-multilingual-f32.gguf
```

Quantized variants: `build/statim-quantize models/laya-multilingual-f32.gguf out.gguf q8_0`.
See [accuracy](#accuracy) before choosing 4-bit.

## API

| endpoint | |
|---|---|
| `POST /v1/systemone` | `{state, questions, model?, lang?, ensemble?}` → `{model, answers, usage, routing}` (Jev/Laya shape) |
| `POST /v1/systemone/batch` | `{states: [...], questions, ...}` → `{results: [...]}` packed into shared forward passes |
| `GET /v1/models` | loaded models |
| `GET /health`, `GET /ready` | liveness / readiness |
| `GET /metrics` | Prometheus: request counts by status, latency histogram, tokens, in-flight, busy workers |

Errors use FastAPI's `{"detail": "..."}` shape with the same status codes as `laya-serve`
(400 malformed, 401 auth, 413 limits, 422 invalid question, 503 saturated). Limits match
`laya.serve`: 64 questions, 50,000 state characters, 2 MB body, 100 choice options.

Auth: `STATIM_API_KEY=key1,key2` or `--api-key-file`. Compared in constant time.

## Benchmarks

BENCHMARKS_PLACEHOLDER

## Accuracy

ACCURACY_PLACEHOLDER

## Status

v0.1 — CPU backend (x86-64 AVX2, ARM NEON via ggml). The ggml graph is backend-agnostic; CUDA,
Vulkan and Metal builds are on the roadmap. Language auto-routing between checkpoints (Laya's
`Router`) is not implemented yet: pass `"model"` explicitly.

## License

Apache-2.0. Statim is independent and not affiliated with the Laya authors or TypeSafe; see `NOTICE`.
