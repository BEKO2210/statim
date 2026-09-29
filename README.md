<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/logo-dark.svg">
    <img alt="Statim" src="assets/brand/logo-light.svg" height="64">
  </picture>
</h1>

<p align="center">
  <a href="https://github.com/BEKO2210/statim/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/BEKO2210/statim/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/BEKO2210/statim/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/BEKO2210/statim?color=0F9F6E"></a>
  <a href="LICENSE"><img alt="Code licence: Apache-2.0" src="https://img.shields.io/badge/code-Apache--2.0-161B22"></a>
  <a href="LICENSE-MODEL.md"><img alt="Model weights: PolyForm Noncommercial, Small Business, Free Trial or commercial" src="https://img.shields.io/badge/weights-PolyForm%20NC%20%C2%B7%20Small%20Business%20%C2%B7%20Trial%20%2B%20commercial-161B22"></a>
</p>

<p align="center">
  <a href="https://beko2210.github.io/statim/#film">
    <picture>
      <source media="(max-width: 640px)" srcset="assets/readme/statim-intro-9x16.webp">
      <img src="assets/readme/statim-intro-16x9.webp" width="840" alt="Statim in ten seconds: a support ticket in JSON becomes three typed decisions, department, urgency and refund, in one forward pass">
    </picture>
  </a>
  <br>
  <a href="https://beko2210.github.io/statim/#film"><b>Watch the 60-second film</b></a>
</p>

Statim answers typed questions about text or JSON: pick one of several labels (`choice`), rate on a
scale (`score`), or give a yes/no probability (`noul`). It is a native C++20 engine for System-1
decision models and computes all answers in one encoder forward pass, served from one static binary.
It needs no Python, PyTorch, or GPU at runtime. Statim runs Laya checkpoints and implements
the Jev/Laya `POST /v1/systemone` protocol, so existing clients can switch by changing the base URL.

> *statim* (Latin): immediately, at once.

**Try it:** [live demo](https://huggingface.co/spaces/Beko2210/statim) (no install, no key) ·
**Models:** [English](https://huggingface.co/Beko2210/statim-decide-en-large) and
[multilingual](https://huggingface.co/Beko2210/statim-decide-multilingual-base) on Hugging Face ·
**Example:** [ticket triage in ten minutes](examples/ticket-triage) ·
**Reproduce the results:** [REPRODUCE.md](REPRODUCE.md)

## At a glance

| Evidence | Statim | Comparison | Protocol and source |
|---|---:|---:|---|
| 14 decision categories, macro accuracy | **0.748** | Qwen3-8B 0.706; mDeBERTa-v3 XNLI 0.488 | Same 11,550 held-out items, questions, and options. Statim was trained on these categories; both baselines were zero-shot. [Full results](docs/BASELINES.md) |
| Banking77, 77 intents | **0.913** | Qwen3-8B 0.650; mDeBERTa-v3 XNLI 0.224 | Same trained-versus-zero-shot comparison. [Full results](docs/BASELINES.md) |
| One RTX 3070 | **68 decisions/s**, 11.6 ms/decision | Qwen3-8B ≈6; mDeBERTa-v3 XNLI 4 decisions/s | Statim Vulkan f32 in batches of 16; Qwen3-8B Ollama Q4_K_M with 2 parallel requests; mDeBERTa CUDA f32. [Measurements](docs/BASELINES.md#speed-on-the-same-machine) |
| CPU latency | **535 ms/state** multilingual; **1,683 ms/state** English | RTX 3070: 54 ms and 137 ms | Ryzen 7 5800X, 16 threads, f32, 30 states × 8 questions, up to 770 tokens. [Performance](#gpu-performance) |
| Reference parity | **240/240** token sequences | Answers within 1e-4 of Laya | CI-gated against the official Python package. [Reproduce](REPRODUCE.md#2-the-engine-matches-the-python-reference) |

These rows use different workloads. They are separate evidence for accuracy, throughput, latency,
and parity, not one combined benchmark.

## Quick start

Linux x86-64 with AVX2 (Haswell or newer), CPU. Three steps: download, start, ask.

```bash
# 1. Download the engine (3 MB) and the multilingual model (357 MB), and verify both
curl -fLO https://github.com/BEKO2210/statim/releases/download/v0.8.6/statim-0.8.6-linux-x86_64-cpu.tar.gz
curl -fLO https://github.com/BEKO2210/statim/releases/download/v0.8.6/SHA256SUMS
sha256sum -c --ignore-missing SHA256SUMS && tar -xzf statim-0.8.6-linux-x86_64-cpu.tar.gz
curl -fLO https://huggingface.co/Beko2210/statim-decide-multilingual-base/resolve/main/statim-decide-multilingual-base-q8_0.gguf
curl -fL -o SHA256SUMS.model https://huggingface.co/Beko2210/statim-decide-multilingual-base/resolve/main/SHA256SUMS
sha256sum -c --ignore-missing SHA256SUMS.model

# 2. Start the server (it keeps running; it is ready when it logs "listening")
./statim-0.8.6-linux-x86_64-cpu/statim serve -m multilingual=statim-decide-multilingual-base-q8_0.gguf --port 8080

# 3. In a second terminal: one ticket, three typed questions
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

The answer (statim-decide-multilingual-base, q8_0; excerpt):

```json
{"answers": {
  "department": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9948, "technical": 0.0011, "sales": 0.0014, "other": 0.0026}},
  "urgency": {"type": "score", "score": 1.585, "legend": {"0": "not urgent", "1": "soon", "2": "critical deadline or blocking issue"}, "probabilities": {"0": 0.0617, "1": 0.2916, "2": 0.6467}},
  "refund": {"type": "noul", "noul": 0.9224}}}
```

`choice` returns the winning label, `score` the expected level on the `legend` scale, and `noul` the
probability that the answer is yes.

The playground is at `http://127.0.0.1:8080/`. To build from source (any other OS or CPU), use a GPU,
or quantize a model, see [docs/BUILD.md](docs/BUILD.md). A complete application is in the
[ticket-triage example](examples/ticket-triage/README.md).

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/diagrams/architecture-dark.svg">
  <img alt="State and questions become one token sequence with a [MASK] per option; one forward pass of the encoder and decision head scores every option and returns calibrated answers." src="assets/diagrams/architecture-light.svg" width="100%">
</picture>

The state and all questions become one token sequence with a mask per option. The encoder and
decision head score every option in one pass. The converter stores architecture, calibration, and
the tokenizer in GGUF, making each model a self-describing artifact.

## Why Statim

| | Laya (Python) | Statim |
|---|---|---|
| Runtime | Python 3.10+, PyTorch, transformers | One 5.5 MB binary and one `.gguf` file |
| Answers | Reference | 240/240 token sequences; answers within 1e-4 |
| Tokenizer | HF `tokenizers` (Rust) | Native C++; identical on 3,906 cases plus 140k fuzz strings, about 10× faster |
| Cold start to first answer | 11.35 s | **0.76 s** |
| Server memory | 3,915 MB | **650 MB**, with mmap'd weights shared between processes |
| HTTP throughput | 0.43 req/s | **1.07 req/s** with 6 threads; 0.99 req/s with 4 threads |
| Server | FastAPI, one inference at a time | Worker pool, admission control, bearer auth, Prometheus metrics, JSON logs, health/readiness, request IDs, graceful shutdown |
| Deployment | pip or Docker | Static binary, distroless Docker, hardened systemd unit |

The comparison uses the laptop CPU protocol under [CPU performance](#cpu-performance). Consensus
across the two base checkpoints is reported separately in [the research results](docs/RESULTS.md#base-checkpoint-consensus).

## Models

Statim runs any Laya checkpoint. Published Statim Decide models use licence-audited fine-tuning data
and must pass a no-harm promotion gate.

| Model | Encoder | Languages | typed-decisions | Banking77 | MASSIVE | Files |
|---|---|---:|---:|---:|---:|---|
| [statim-decide-en-large 0.5.0](https://huggingface.co/Beko2210/statim-decide-en-large) | ModernBERT-large, 395M | English | **0.768** | **0.928** | 0.867 (en) | f32 1.58 GB; q8_0 0.45 GB |
| [statim-decide-multilingual-base 0.7.0](https://huggingface.co/Beko2210/statim-decide-multilingual-base) | mmBERT-base | 12 evaluated | 0.763 | 0.914 | 0.800 (12 languages) | f32 0.91 GB; q8_0 0.36 GB |

These are trained-suite promotion-gate results. Under the same published protocol,
typed-decisions scores are meraGPT 0.768, laya-typed-decisions 0.766, and Jev 0.727; supervised
MPNet scores 0.941 on Banking77. See [REPRODUCE.md](REPRODUCE.md#3-a-published-models-evaluation).

```bash
hf download Beko2210/statim-decide-en-large statim-decide-en-large-q8_0.gguf --local-dir models   # pip install huggingface_hub
./statim serve -m english=models/statim-decide-en-large-q8_0.gguf
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /v1/systemone` | Decide for one state; returns `model`, `answers`, `usage`, and `routing`. |
| `POST /v1/systemone/batch` | Decide for multiple states with shared questions and packed forward passes. |
| `GET /v1/models` | List loaded models and adapters. |
| `GET /health`, `GET /ready` | Liveness and readiness. |
| `GET /metrics` | Prometheus request, latency, token, concurrency, worker, batch, model, and adapter metrics. |
| `GET /` | Browser playground, enabled by default. |

Requests contain `state` and `questions`, with optional `model`, `adapter`, `lang`,
`min_confidence`, `return_logits`, `calibrate`, and `ensemble`. `model` may select a loaded model or
`consensus`; omission enables routing. A positive `min_confidence` adds `escalate` to answers below
the threshold. The complete schema, question types, responses, errors, routing, batching, limits,
and operations guide is [docs/API.md](docs/API.md).

### LoRA adapters

One base model can serve small per-category LoRA adapters selected by name or question family.

```sh
python tools/convert_lora.py runs/emotion-lora -o models/emotion.lora.gguf \
    --base models/laya-multilingual-q8_0.gguf --category emotion      # PEFT adapter -> GGUF
statim serve -m multilingual=models/laya-multilingual-q8_0.gguf \
    --adapter multilingual:emotion=models/emotion.lora.gguf
curl -s localhost:8080/v1/systemone -d '{"state": "...", "adapter": "emotion", "questions": {...}}'
```

The converter accepts plain LoRA on encoder attention and MLP projections, rejects unsupported
variants and modules, and binds the adapter to its base by a fingerprint of the checkpoint's norms
and biases, plus, since 0.8.1, a checkpoint SHA-256 over every source tensor; both are shared by a
checkpoint's f32 and quantized files, and the engine refuses an adapter whose base does not match.

- f32, f16, and bf16 default to merging `W + B·A` at load. Requests retain base latency; each
  adapter owns a copy of the adapted weights, 438 MB for the multilingual f32 model.
- Quantized weights default to runtime LoRA. A rank-4 adapter uses 3.3 MB and adds 17–28 % CPU
  latency. Merging the test adapter into q4_0 loses 91 % of its effect.

`--adapter-mode merge|runtime` overrides this choice. `"adapter": "auto"` matches the 14 category
keywords in every question ID, then its instructions; mixed or unmatched families use the base.
See [docs/API.md#lora-adapters](docs/API.md#lora-adapters) for constraints, routing, measurements,
responses, metrics, and tests. Trained adapters and the evidence for each are in
[docs/ADAPTERS.md](docs/ADAPTERS.md). On 0.7.0, the PII, emotion and safety adapters pass the gate
(safety in a replication on fresh items) and are published: [PII](https://huggingface.co/Beko2210/statim-decide-multilingual-base-pii),
[emotion](https://huggingface.co/Beko2210/statim-decide-multilingual-base-emotion), [safety](https://huggingface.co/Beko2210/statim-decide-multilingual-base-safety).

### Authentication, limits, and operations

Set `STATIM_API_KEY=key1,key2` or pass `--api-key-file FILE`. Authentication covers inference,
`/metrics`, and `/v1/models`; health, readiness, and the playground remain public. Key sources fail
closed and comparisons use constant-time code. A non-loopback server without keys emits
`auth_off_on_network`.

Requests are bounded before inference by body, JSON structure, state, question, option, token,
attention, response, concurrency, queue, and deadline limits. See [API limits](docs/API.md#limits-and-server-controls),
the [security report](docs/SECURITY.md), and the [production deployment guide](docs/DEPLOY.md).

### Client SDKs

The dependency-free Python and TypeScript clients retry `503` responses with backoff.

| Package | Path |
|---|---|
| Python `statim` | [`clients/python`](clients/python) (`pip install ./clients/python`) |
| TypeScript `@statim/client` | [`clients/js`](clients/js) (`npx tsc`, then import the package) |

```python
from statim import Client

client = Client("http://127.0.0.1:8080", timeout=120)
decision = client.decide(
    {"subject": "Duplicate charge on invoice #4411"},
    {"refund": {"type": "noul", "instructions": "Does the user explicitly request a refund?"}},
    model="multilingual",
)
print(decision.answers["refund"].noul, decision.answers["refund"].confidence)
```

Both clients provide `decide`, `decide_batch`, `models`, `health`, and `ready`; both decision methods
accept an `adapter` option. Yes/no questions use
wire type `noul` and return `YesNoAnswer`. See the [Python](clients/python/README.md) and
[TypeScript](clients/js/README.md) guides.

## Performance

### CPU performance

Measured on an Intel Xeon E3-1505M v5 laptop CPU (4 cores, 8 threads, AVX2, turbo off, 31 GB RAM),
using the f32 multilingual checkpoint and Laya 0.3.20 on PyTorch 2.14. The workload is 30 states × 8
questions from `tests/data/golden_inputs.json`, up to 20 options and 770 tokens. Each engine uses its
best thread count. Scripts are in `bench/`.

| Measurement | Laya | Statim | Difference |
|---|---:|---:|---:|
| In-process mean / p50 per state | 1,186 / 815 ms | **1,139 / 823 ms** | −4 % mean |
| HTTP, 1 client: throughput | 0.43 req/s, 4 threads | **1.07 req/s**, 6 threads; 0.99 req/s, 4 threads | 2.3–2.5× |
| HTTP, 1 client: p50 / p95 | 1,718 / 4,994 ms | **889 / 1,280 ms** | p95 −74 % |
| HTTP, 4 clients: throughput / p95 | 0.53 req/s / 11,958 ms | **1.12 req/s / 4,425 ms** | 2.1× |
| Server resident memory | 3,915 MB | **650 MB** | 6× less |
| Cold start to first answer | 11.35 s | **0.76 s** | 15× faster |
| Peak memory, one shot | 2,667 MB | **585 MB** f32; **234 MB** q8_0 | 4.6–11× less |
| Runtime footprint | PyTorch alone ≥ 1.2 GB | **5.5 MB** binary | |

On AVX2 without VNNI, q8_0 halves the file and cuts memory 2.5× but is slower than f32. Its logits
move by up to ~0.4 and 2 of 240 parity answers change. ARM dotprod/i8mm and AVX-512-VNNI are the
intended int8 CPU targets. Four-bit weights are not recommended for this model family.

### GPU performance

GPU build and launch instructions are in [docs/BUILD.md](docs/BUILD.md#gpu-backends).

`--device` accepts `cpu`, `gpu`, `vulkan`, `cuda`, or a name such as `Vulkan0`; `STATIM_DEVICE` sets
the default. Both f32 checkpoints use about 3.3 GB of VRAM. On an RTX 3070 and Ryzen 7 5800X, f32,
using the 30 × 8 golden workload:

| Model and path | CPU | RTX 3070 | Speed-up |
|---|---:|---:|---:|
| Multilingual, in-process per state | 535 ms | **54 ms** | ~10× |
| English, in-process per state | 1,683 ms | **137 ms** | ~12× |
| Multilingual, HTTP, 1 client | 2.68 req/s; p50 353 ms | **20.5 req/s; p50 45 ms** | 7.7× |
| English, HTTP, 1 client | 0.91 req/s; p50 1,039 ms | **8.1 req/s; p50 119 ms** | 8.9× |
| Parity max \|Δlogit\|, multilingual / English | 5.0e-4 / 2.0e-4 | **8.8e-5 / 1.6e-4** | 240/240 argmax |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/diagrams/gpu-dark.svg">
  <img alt="RTX 3070 vs. Ryzen 7 5800X: 7.7x (multilingual) and 8.9x (English) HTTP throughput, median latency 45 vs 353 ms and 119 vs 1,039 ms, same answers." src="assets/diagrams/gpu-light.svg" width="100%">
</picture>

Exact f32 is the default. `--gpu-fast` or `STATIM_GPU_FAST=1` enables f16 GPU math: 24.5 ms per
multilingual state instead of 54 ms, still 240/240 argmax, with logits within ~1e-1 rather than
1e-4. English-model backend results, one HTTP client and 60 requests:

| Mode | CPU | Vulkan | CUDA |
|---|---:|---:|---:|
| f32, exact | 0.84 req/s | 8.40 req/s | 8.35 req/s |
| f32, `--gpu-fast` | | **16.4 req/s** | 11.7 req/s |
| q8_0 | | 9.0 req/s | **15.7 req/s** |

The multilingual exact-f32 result is Vulkan 21.9 req/s and CUDA 16.2 req/s. Vulkan remains the
default; on NVIDIA, q8_0 with CUDA is the fastest measured large-model path with f32 activations.

```bash
./build-vk/statim serve --device vulkan -m english=models/laya-english-f32.gguf \
  --batch-window-ms 2 --max-batch 16
```

With this 2 ms micro-batch window, RTX 3070 English-model throughput changed from 8.47 to 8.30 req/s
with one client, 8.50 to 9.64 with 8 clients, and 8.48 to 10.78 with 16 clients. Compatible requests
must share model, adapter, questions, and inference options. Answers match standalone requests within
1e-4. The default window is 0; leave it disabled on CPU.

## Results

### Trained Statim versus zero-shot general models

Every system receives the same 11,550 held-out items, questions, and options: 37 language cells with
150 items each across 14 categories, plus the first 2,000 test rows of AG News, DAIR Emotion, and
Banking77. Training-mixture overlaps are removed. Statim 0.7.0 was trained on these categories;
Qwen3-8B and mDeBERTa were zero-shot. This compares out-of-the-box systems, not learning methods.

| | Statim 0.7.0 | Qwen3-8B, zero-shot | mDeBERTa XNLI, zero-shot |
|---|---:|---:|---:|
| Parameters | 307M | 8.2B | 279M |
| 14-category macro accuracy | **0.748** | 0.706 | 0.488 |
| Categories won | **9** | 5 | 0 |
| Banking77 | **0.913** | 0.650 | 0.224 |
| AG News | **0.929** | 0.847 | 0.581 |
| Decisions/s, same RTX 3070 | **68** | ≈6 | 4 |

Qwen3-8B leads on emotion, fact-check, sentiment, safety, and PII. See
[docs/BASELINES.md](docs/BASELINES.md) for all cells, limitations, speed, and reproduction commands.

### Published model gates

Published models must pass validation, held-out-suite, and pooled-family promotion gates before
replacing their base. The full method and charts are in [docs/RESULTS.md](docs/RESULTS.md#published-model-gates).

| Model | Trained suites | Never-trained suites | Gate versus base |
|---|---|---|---|
| English 0.5.0 | typed-decisions 0.768; Banking77 0.928; MASSIVE English 0.867; HWU64 0.833 | AG News 0.939; Emotion 0.588 | 54 suites: 11 significant gains, 0 regressions; zero-shot family within noise |
| Multilingual 0.7.0 | typed-decisions 0.763; Banking77 0.914; MASSIVE 0.800 over 12 languages | AG News 0.9295; Emotion 0.504 | 89 suites: 23 significant gains, 66 within noise, 0 regressions; 14-category macro 0.748 (0.4.0: 0.559) |

Category-adapter results and promotion evidence are in [docs/ADAPTERS.md](docs/ADAPTERS.md).

## Security and robustness

Statim applies bearer authentication, bounded parsing and inference budgets, admission control,
deadlines, fixed error responses, and load-time model validation. Malformed GGUF metadata, tensors,
hyperparameters, token IDs, vocabularies, calibration tables, fingerprints, and UTF-8 names are
rejected before serving.

libFuzzer harnesses exercise complete request handling, both tokenizers, and GGUF loading under ASan
and UBSan. CI runs each for 60 seconds per push from a cached corpus. Committed seeds and crash inputs
replay under `ctest`, and model validation adds 20 malformed-model cases. The HTTP security suite runs
against a live server:

```sh
python3 tests/security/test_http.py --binary build/statim --model models/laya-multilingual-f32.gguf
```

[docs/SECURITY.md](docs/SECURITY.md) has the findings, fixes, fuzzing campaign and coverage.

## Status and roadmap

The current release is v0.8.6: x86-64 AVX2 and ARM NEON CPU support through ggml, optional Vulkan
and CUDA, two published models, client SDKs, and a public demo. 0.8.6 gives the README a three-step
quick start; 0.8.5 publishes the safety adapter after a pre-registered replication; 0.8.4 publishes the PII and emotion adapters; 0.8.3 brings LoRA adapters to both client SDKs; 0.8.2 checks every document against
the code in CI and records the first category-adapter experiment; 0.8.1 binds each LoRA adapter to the
exact checkpoint it was trained on; 0.8.0 added per-category LoRA adapters,
the comparison with a local LLM and an NLI classifier, fuzzing in CI, load-time model and adapter
validation, and a warning when a non-loopback server starts without authentication. Three
category adapters are published: PII, emotion and safety ([docs/ADAPTERS.md](docs/ADAPTERS.md)).
Next, sentiment and fact-check need more data or larger held-out samples.

Before 1.0, the HTTP API may change between minor versions. See the [changelog](CHANGELOG.md) and
[roadmap](docs/ROADMAP.md).

## Licence

| Component or use | Licence |
|---|---|
| Source code: engine, server, tools | [Apache-2.0](LICENSE), free for any use |
| Statim weights, noncommercial | [PolyForm Noncommercial 1.0.0](LICENSE-MODEL.md): personal use, research, experiments, and noncommercial organisations |
| Statim weights, small companies | [PolyForm Small Business 1.0.0](LICENSE-MODEL.md): commercial use below 100 people and 1 M USD revenue |
| Statim weights, evaluation | [PolyForm Free Trial 1.0.0](LICENSE-MODEL.md): fewer than 32 consecutive days |
| Other commercial use | Paid licence; see [COMMERCIAL.md](COMMERCIAL.md) |

Released weights use commercially usable, non-ShareAlike data; sources are in
[DATA_LICENSES.md](DATA_LICENSES.md). Original Laya checkpoints are Apache-2.0. Statim is independent
and not affiliated with the Laya authors or TypeSafe; see [NOTICE](NOTICE).
