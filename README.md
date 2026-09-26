<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/logo-dark.svg">
    <img alt="Statim" src="assets/brand/logo-light.svg" height="64">
  </picture>
</h1>

**A native C++20 engine for System-1 decision models.** Typed decisions — `choice`, `score`, `noul` —
over any text or JSON in a single forward pass, served from one static binary. No Python, no PyTorch,
no GPU required — and ~8× faster when there is one ([GPU](#gpu-vulkan)).

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
| Runtime | Python 3.10+, PyTorch, transformers | one 3.3 MB binary + one `.gguf` file |
| Answers | reference | **identical** — 240/240 token sequences, answers within 1e-4 (CI-gated) |
| Tokenizer | HF `tokenizers` (Rust) | native C++, 100% identical on 3,906 cases + 140k fuzz strings, ~10× faster |
| Cold start → first answer | 11.35 s | **0.76 s** |
| Server memory | 3,915 MB | **650 MB** (mmap'd weights, shared between processes) |
| HTTP throughput | 0.43 req/s | **1.07 req/s** (2.5×) |
| Accuracy | one checkpoint per request | **consensus** of both checkpoints: equal on AG News, better on Emotion (+0.5 pt, NLL −8 %) and Banking77 (+6.25 pt), see [accuracy](#accuracy) |
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
./build/statim serve -m english=models/laya-english-f32.gguf -m multilingual=models/laya-multilingual-f32.gguf --consensus
# open http://127.0.0.1:8080/ for the playground
```

Quantized variants: `build/statim-quantize models/laya-multilingual-f32.gguf out.gguf q8_0`.
See [accuracy](#accuracy) before choosing 4-bit.

## API

| endpoint | |
|---|---|
| `POST /v1/systemone` | `{state, questions, model?, lang?}` → `{model, answers, usage, routing}` (Jev/Laya shape). `model`: `english`, `multilingual`, `consensus`, or omitted (auto-routing by language). Extras: `return_logits`, `calibrate`, `ensemble` |
| `POST /v1/systemone/batch` | `{states: [...], questions, ...}` → `{results: [...]}` packed into shared forward passes |
| `GET /v1/models` | loaded models |
| `GET /health`, `GET /ready` | liveness / readiness |
| `GET /metrics` | Prometheus: request counts by status, latency histogram, tokens, in-flight, busy workers |

Errors use FastAPI's `{"detail": "..."}` shape with the same status codes as `laya-serve`
(400 malformed, 401 auth, 413 limits, 422 invalid question, 503 saturated). Limits match
`laya.serve`: 64 questions, 50,000 state characters, 2 MB body, 100 choice options.

Auth: `STATIM_API_KEY=key1,key2` or `--api-key-file`. Compared in constant time.

## Benchmarks

Measured on a 2015-class laptop CPU (Intel Xeon E3-1505M v5, 4 cores / 8 threads, AVX2, turbo off,
31 GB RAM), `laya-multilingual` checkpoint, f32, Laya 0.3.20 on PyTorch 2.14. Workload: the 30 states
× 8 questions in `tests/data/golden_inputs.json` (choice, score and noul, up to 20 options, up to 770
tokens). Each engine at its best thread count. Scripts in `bench/`.

| | Laya (Python) | **Statim** | |
|---|---|---|---|
| Latency in-process, mean / p50 per state | 1,186 / 815 ms | **1,139 / 823 ms** | on par (−4 % mean) |
| HTTP, 1 client: throughput | 0.43 req/s (`laya-serve`, 4 threads) | **1.07 req/s** (6 threads) · 0.99 req/s (4 threads) | **2.3–2.5×** |
| HTTP, 1 client: p50 / p95 | 1,718 / 4,994 ms | **889 / 1,280 ms** | p95 **−74 %** |
| HTTP, 4 clients: throughput / p95 | 0.53 req/s / 11,958 ms | **1.12 req/s / 4,425 ms** | **2.1×** |
| Server resident memory | 3,915 MB | **650 MB** | **6× less** |
| Cold start → first answer | 11.35 s | **0.76 s** | **15× faster** |
| Peak memory, one-shot | 2,667 MB | **585 MB** (f32) · **234 MB** (q8_0) | **4.6–11× less** |
| Runtime footprint | PyTorch alone ≥ 1.2 GB | **3.3 MB** binary | |

Raw FLOPs are the same model either way, and PyTorch's fp32 GEMM (MKL/oneDNN) is already close to this
CPU's peak, so single-request latency is at parity. The gains come from everything around the forward
pass: no interpreter, no framework start-up, mmap'd weights (untouched embedding rows never enter
RAM), rows of many states packed into shared graphs, a worker pool instead of one global inference
lock, flash attention, a fused GeGLU kernel and an exact pruning of the last head layer to the rows
the scorer reads.

`q8_0` halves the file and cuts memory 2.5×, but on AVX2 without VNNI it is slower than f32 and moves
logits by up to ~0.4 (2 of 240 parity answers change); ARM (dotprod/i8mm) and AVX-512-VNNI CPUs are
where int8 pays off. 4-bit is not recommended for this model family (see below).

## GPU (Vulkan)

Optional, off by default. Needs the Vulkan headers, `glslc` and SPIR-V headers at build time
(Debian/Ubuntu: `libvulkan-dev glslc spirv-headers`) and a Vulkan driver at run time.

```bash
cmake -S . -B build-vk -DSTATIM_VULKAN=ON && cmake --build build-vk
ctest --test-dir build-vk                      # CPU gates + the same gates on the GPU (*_vulkan)
./build-vk/statim serve --device vulkan -m english=models/laya-english-f32.gguf -m multilingual=models/laya-multilingual-f32.gguf
```

`--device` takes `cpu` (default), `gpu`, `vulkan` or a device name such as `Vulkan0`
(`STATIM_DEVICE` sets the default). Weights are copied to VRAM once; both f32 checkpoints use ~3.3 GB.

Measured on an RTX 3070 (8 GB) vs. the same machine's Ryzen 7 5800X (16 threads), f32, the 30 × 8
golden workload:

| | CPU | **RTX 3070** | |
|---|---|---|---|
| multilingual, in-process per state | 535 ms | **54 ms** | **~10×** |
| english, in-process per state | 1,683 ms | **137 ms** | **~12×** |
| multilingual, HTTP 1 client | 2.68 req/s · p50 353 ms | **20.5 req/s · p50 45 ms** | **7.7×** |
| english, HTTP 1 client | 0.91 req/s · p50 1,039 ms | **8.1 req/s · p50 119 ms** | **8.9×** |
| parity vs. Laya, max \|Δlogit\| (multilingual / english) | 5.0e-4 / 2.0e-4 | **8.8e-5 / 1.6e-4** | 240/240 argmax |

**Exact by default.** ggml's Vulkan backend normally feeds f32 matmuls through f16 (cooperative
matrices / fp16 shaders), which moves logits by up to ~0.12 on long inputs. Statim disables those paths
unless asked, and always uses the flash-attention kernel on GPUs (the explicit softmax path converts
non-contiguous operands to f16). `--gpu-fast` (or `STATIM_GPU_FAST=1`) turns f16 back on: 24.5 ms per
state instead of 54, argmax still 240/240, but logits within ~1e-1 rather than 1e-4.

## Accuracy

Same construction as Laya's own Jev comparison (`research/scripts/bench_apps.py`): first 400 test
rows, identical prompts, CPU, fp32. Reproduce with `bench/eval_accuracy.py`.

| suite (400 cases) | Jev (published)¹ | Laya `laya` (English)² | Laya `laya-multilingual`² | **Statim consensus** |
|---|---|---|---|---|
| AG News (4 labels) | 0.910 | 0.950 | 0.935 | **0.950** |
| DAIR Emotion (6 labels) | 0.480 | 0.5925 | 0.5375 | **0.600** |
| Banking77 (all 77 labels at once) | 0.870 (72 labels) | 0.425 | 0.470 | **0.4875** |

| Emotion calibration | NLL | ECE | Brier |
|---|---|---|---|
| Laya `laya` | 2.019 | 0.306 | 0.696 |
| **Statim consensus** | **1.865** | **0.286** | **0.686** |

¹ Third-party published numbers, as quoted by Laya; different samples and prompts, indicative only.
² Measured through Statim's exact mode, which is bit-identical to the Laya package (CI-gated), so these
  are Laya's numbers. Laya reports 0.953 / 0.600 for the English checkpoint on its own run.

**Consensus** (`"model": "consensus"`, or `statim serve --consensus`) runs the English (ModernBERT-large)
and multilingual (mmBERT) checkpoints on the same request and averages their option log-probabilities.
Laya's router always answers with one checkpoint. Measured cost: 1.25–1.4× the English checkpoint alone.

What did **not** help, measured and kept out of the defaults:
- **Option-order ensembling** (cyclic rotations of the options): Emotion 0.5375 → 0.525. The checkpoint
  was trained on a fixed option order; permutations are out of distribution. Still available as
  `ensemble: K` (choice questions only; score levels are ordinal and never rotated).
- **Contextual calibration** (`calibrate: true`, Zhao et al. 2021): +2.0 points on Emotion for the
  multilingual checkpoint, neutral to slightly negative elsewhere; improves ECE on Banking77. Opt-in.
- **4-bit weights** (`q4_0`, `q4_K`): flip 1–2 of 16 parity answers. f32 is the reference; `q8_0`
  halves memory with small logit drift (see benchmarks).

## Many options and fine-tuning (Banking77)

**Option budget.** Laya fits all options into `head_max_len` tokens (192 English, 256 multilingual)
and, when they do not fit, cuts every option to `(head_max_len - 16) / k` tokens. With Banking77's 77
intents that is one or two subwords per intent: the model never reads the label names. Raising the
budget per request (`"head_max_len": 512`, same parameter as Laya's `predict_batch`; server default
`--head-max-len`) needs no training. It changes nothing for questions whose options already fit, and
`max_len` grows with it so the state keeps at least 128 tokens.

**Fine-tuning.** `tools/finetune/train_banking77.py` applies Laya's own RLCD recipe (from its
fine-tuning notebook) to the multilingual checkpoint: Banking77 *train* (option order shuffled per
item), replay of `LocalLLaMA/typed-decisions` train, token embeddings frozen (90 % of mmBERT's
parameters; weight decay would otherwise erode every language Banking77 never touches), best epoch by
held-out dev, temperatures refit on held-out items. `--distill N` adds learning-without-forgetting
replay: generic zero-shot questions over tweets and news (never the evaluation data) whose targets
are the *base* model's own answer distributions, so it anchors old behaviour without teaching any
label. 5 epochs, 35 minutes on an RTX 3070.

```bash
python -m venv .venv-train && .venv-train/bin/pip install torch laya==0.3.20 datasets
.venv-train/bin/python tools/finetune/train_banking77.py models/laya-multilingual models/laya-multilingual-banking77 --distill 6000 --epochs 5
.venv/bin/python tools/convert_laya.py models/laya-multilingual-banking77 -o models/laya-multilingual-banking77-f32.gguf --type f32 --embd-type f16
.venv-train/bin/python tools/finetune/eval_laya.py models/laya-multilingual-banking77 --n 2000 --head-max-len 512
```

First 2,000 test rows per suite (never trained on; ±1.1 pt standard error around 0.5), plus the
2,000 decisions of the `typed-decisions` test split:

| | multilingual | + `head_max_len` 512 | v1: fine-tuned, 3 ep. | **v3: + distillation, 5 ep.** |
|---|---|---|---|---|
| Banking77 accuracy | 0.4885 | 0.5175 | 0.8435 | **0.8655** |
| Banking77 ECE | 0.372 | 0.352 | 0.052 | **0.043** |
| AG News (held out) | 0.938 | 0.938 | 0.941 | **0.9385** |
| Emotion (held out) | 0.532 | 0.532 | 0.502 | **0.528** |
| Emotion ECE | 0.336 | 0.336 | 0.210 | **0.155** |
| typed-decisions test | 0.351 | 0.351 | 0.6665¹ | **0.7015**¹ |

¹ In-domain: its train split is replay data. AG News and Emotion were never trained on.

Without distillation, Emotion lost 3 points (v1); with it, every held-out suite stays within noise of
the base model while Banking77 gains 35 points and all calibration errors shrink. Statim and the Laya
package agree on the fine-tuned checkpoint (Banking77, first 400: 0.8675 vs. 0.870, bf16 vs. f32).
The weights are not in this repository; the script reproduces them.

## Status

v0.2 — CPU backend (x86-64 AVX2, ARM NEON via ggml) and an optional Vulkan GPU backend. See
[CHANGELOG.md](CHANGELOG.md) for releases and [docs/ROADMAP.md](docs/ROADMAP.md) for the path to 1.0. CUDA and
Metal builds are on the roadmap. Language routing is a light heuristic (English text →
English checkpoint, everything else → multilingual); Laya's full `Router` language detection and the
`typed-decisions` checkpoint are next.

## License

Apache-2.0. Statim is independent and not affiliated with the Laya authors or TypeSafe; see `NOTICE`.
