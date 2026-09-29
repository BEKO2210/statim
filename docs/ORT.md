# Statim and ONNX Runtime on CPU

This comparison asks whether exporting the shipped Laya model to ONNX and serving it with ONNX
Runtime (ORT) is enough. The answer depends on what is being compared. ORT is an excellent general
graph runtime and may win individual latency or throughput cells. Statim is the complete decision
runtime: model, tokenizer, calibration metadata and decision semantics in one GGUF, plus the bounded
HTTP API.

No timing result is accepted until the graph, request packing and decoded answers pass their parity
gates. All measurements use the shipped `models/laya-multilingual-v9` checkpoint. The earlier
`models/laya-multilingual` checkpoint is used only for an independent graph proof against
`tests/data/golden_laya-multilingual.jsonl`.

## Parity triangle

`bench/ort_export.py` exports the complete encoder, typed decision transformer, option scorer and
action scorer. It writes the v9 PyTorch reference to `build-ort/golden_v9.jsonl` and the machine-
readable report to `build-ort/export-report.json`.

| Comparison with its f32 reference | Argmax | max \|Δlogit\| | max \|Δact\| | Gate |
|---|---:|---:|---:|---:|
| Original checkpoint, ORT f32 | 240/240 | 9.44138e-05 | 0 | pass |
| Shipped v9, ORT f32 | 240/240 | 1.00732e-05 | 0 | pass |
| Shipped v9, Statim f32 | 240/240 | 1.38e-05 | 0 | pass |
| Shipped v9, ORT dynamic int8 | 196/240 | 7.95753 | 0 | descriptive |
| Shipped v9, Statim q8_0 | 235/240 | 0.309 | 0 | descriptive |

The f32 limit is 1e-3 and requires 240/240 argmax agreement. Quantized rows are descriptive: both
are compared with the v9 PyTorch f32 reference, without pretending that quantization is exact. On
these 240 items ORT dynamic int8 loses substantially more decisions than Statim q8_0.

The serving preflight independently reproduced all 240 golden token-ID and marker-position pairs.
For v9 f32 it then compared ORT with `build-rel/statim decide` on all 30 states and eight questions
per state. All choices matched, and every probability, score, confidence and action probability was
within 1e-3.

## Protocol

The raw test uses the 240 already-tokenized items, batching the eight questions belonging to one
state. Statim receives a generated JSONL containing one warm-up plus `R` copies of all 30 states,
with distinct state indices. ORT runs the same batches for one warm-up plus `R` measured passes.
For each engine the statistic for a state is the median across measured passes; mean, p50 and p95
are then computed over the 30 state medians. Session/model load is outside raw timing.

The HTTP test sends each `tests/data/golden_inputs.json` state with all eight questions. It compares
the verified 0.9.0 CPU release with the small ORT server at one and four clients. Four-client Statim
also gets a separate documented micro-batching row (`--batch-window-ms 2 --max-batch 16`). The first
answer is the cold-start observation; additional warm-up requests are excluded from latency and
throughput. Peak RSS is sampled from the serving process.

Every benchmark process sets `CUDA_VISIBLE_DEVICES=""`. CPU thread counts are 1, 4, 8 and 16. Each
pair sees identical inputs.

## Machine and software

| Item | Value |
|---|---|
| CPU | AMD Ryzen 7 5800X, 8 cores / 16 threads |
| RAM | 16,685,232,128 bytes |
| Kernel | 6.18.7-76061807-generic |
| Governor | performance |
| Python | 3.12.3 |
| ONNX Runtime | 1.30.0 |
| ONNX | 1.23.0 |
| PyTorch | 2.14.0+cu130, CPU execution for this comparison |
| Laya | 0.3.20 |
| Statim | 0.9.0 |
| Git commit | `bd9e5d502fbf3795655a5a028f7a0d709f3fd588` |

`bench/ort_compare.py --render-md RESULTS.json` renders every result table from the JSON. The
reviewer should replace the timing placeholders below with that output; numbers are not maintained
by hand.

## Raw scoring

pending: reviewer run

## HTTP end to end

pending: reviewer run

## Cold start and peak RSS

pending: reviewer run

## Sizes

Artifacts count everything loaded by that runtime. Thus each ONNX row includes its external data,
`tokenizer/tokenizer.json` and `rl_agent_config.json`; GGUF already contains its tokenizer and
configuration.

| Artifact | Bytes |
|---|---:|
| ONNX f32 server artifacts | 1,324,804,952 |
| ONNX dynamic-int8 server artifacts | 359,178,254 |
| GGUF f32 | 908,307,008 |
| GGUF q8_0 | 356,674,176 |

| Install | Bytes |
|---|---:|
| ORT server venv, excluding CPython | 195,401,377 |
| Statim 0.9.0 extracted executable | 5,460,960 |
| Statim 0.9.0 release tarball | 2,986,871 |

CPython is reported separately: Python 3.12.3 uses 61,276,800 bytes for the interpreter executable
and standard library on this machine. It is not added to the ORT venv row.

## Held-out accuracy

pending: reviewer run

The local attempt stopped before inference: the held-out cache is incomplete and every registered
source needed network access. The training interpreter has `pyarrow` and `datasets`; nothing was
installed or substituted. The reviewer can run one small `sentiment` suite once the registered data
is available.

## GPU providers

| ORT execution provider | Raw | HTTP | Cold start / RSS | Accuracy |
|---|---:|---:|---:|---:|
| CUDA EP | pending: GPU run | pending: GPU run | pending: GPU run | pending: GPU run |
| TensorRT EP | pending: GPU run | pending: GPU run | pending: GPU run | pending: GPU run |

Use the same exported graphs and 240 golden inputs. With an ORT GPU environment, these commands run
the same warm-up, repeat and per-state aggregation as the CPU protocol:

```bash
CUDA_VISIBLE_DEVICES=0 build-ort/venv/bin/python bench/ort_compare.py --ort-only \
  --ort-provider cuda --thread-counts 1 --repeats 5 --out build-ort/gpu-cuda-results.json
CUDA_VISIBLE_DEVICES=0 build-ort/venv/bin/python bench/ort_compare.py --ort-only \
  --ort-provider tensorrt --thread-counts 1 --repeats 5 --out build-ort/gpu-tensorrt-results.json
```

The command refuses to run when the requested provider is unavailable and records the command in
the results JSON. A GPU ORT installation belongs in a separate environment; it is not part of the
CPU install footprint.

## Reproduction

Export, build the quantized artifacts and run every parity gate:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=build-ort/venv/lib/python3.12/site-packages \
  ../statim/.venv-train/bin/python bench/ort_export.py --threads 1
```

Full raw CPU run (30 states, one warm-up plus five measured passes):

```bash
CUDA_VISIBLE_DEVICES="" build-ort/venv/bin/python bench/ort_compare.py \
  --thread-counts 1 4 8 16 --repeats 5 --out build-ort/cpu-raw-results.json
```

Full HTTP, cold-start and RSS run. It runs f32 and quantized servers, one and four clients, and the
separate Statim micro-batching rows:

```bash
CUDA_VISIBLE_DEVICES="" build-ort/venv/bin/python bench/ort_compare.py \
  --socket-benchmark --thread-counts 1 4 8 16 --requests 120 --warmup-requests 4 \
  --out build-ort/cpu-http-results.json
```

Render the Markdown from either result file:

```bash
build-ort/venv/bin/python bench/ort_compare.py --render-md build-ort/cpu-raw-results.json
build-ort/venv/bin/python bench/ort_compare.py --render-md build-ort/cpu-http-results.json
```

For each of the four accuracy servers (Statim f32, Statim q8_0, ORT f32 and ORT int8), point this
same command at its base URL and use a distinct output file:

```bash
../statim/.venv-train/bin/python bench/eval_categories.py \
  --url http://127.0.0.1:8090 --model multilingual --suites sentiment --n 25 \
  --out build-ort/accuracy-ENGINE-VARIANT.jsonl
```

Start Statim f32 or q8_0 with the corresponding model path:

```bash
CUDA_VISIBLE_DEVICES="" build-ort/release-0.9.0/statim-0.9.0-linux-x86_64-cpu/statim serve \
  --device cpu --threads 4 --port 8090 -m multilingual=models/laya-multilingual-v9-f32.gguf
CUDA_VISIBLE_DEVICES="" build-ort/release-0.9.0/statim-0.9.0-linux-x86_64-cpu/statim serve \
  --device cpu --threads 4 --port 8090 -m multilingual=build-ort/laya-multilingual-v9-q8_0.gguf
```

Start ORT f32 or int8 with the corresponding graph path:

```bash
CUDA_VISIBLE_DEVICES="" build-ort/venv-server/bin/python bench/ort_compare.py --serve-ort \
  --threads 4 --port 8090 --onnx build-ort/laya-multilingual-v9-f32.onnx
CUDA_VISIBLE_DEVICES="" build-ort/venv-server/bin/python bench/ort_compare.py --serve-ort \
  --threads 4 --port 8090 --onnx build-ort/laya-multilingual-v9-int8.onnx
```

## What this comparison does and does not show

ORT does some things better. It accepts a standard interchange graph, has a broad provider and
tooling ecosystem, and lets an existing ONNX deployment stack reuse familiar observability and
scheduling. A measured ORT cell that is faster belongs in the report as an ORT win.

Statim adds the parts that an ONNX graph does not define: the tokenizer and decision metadata in one
file; calibrated `choice`, `score` and `noul` answers; action probabilities; bounded parsing,
attention, concurrency, queue and deadline controls; explicit and micro-batching; and the frozen
HTTP API. The release is one small native executable rather than a Python environment plus graph,
external weights, tokenizer and application code.

This is not a comparison with Triton, the OpenVINO EP, ORT's transformer optimizer (it was not used),
or any GPU provider. Those could change latency and throughput and should not be inferred from the
CPU rows.
