# Statim and ONNX Runtime on CPU

A common question about Statim: why not export the model to ONNX and serve it with ONNX Runtime
(ORT)? This page answers it for the shipped multilingual model on one CPU, with the same inputs for
both engines and parity gates before any timing. The target is that Statim should beat or match
general-purpose runtimes on its own decision workload without sacrificing fidelity. Where ORT is
faster, this page says so.

**In short** (AMD Ryzen 7 5800X, 8 cores; details and every number's source below):

- **Same answers.** On the 240 golden items, ORT f32 and Statim f32 both reproduce the PyTorch
  reference within 1.4e-5. On sequences up to 1,024 tokens they agree with each other within
  1.4e-5, and on 2,850 held-out items they make the same decision on every item.
- **Latency depends on sequence length and threads.** With 128-token sequences at 1–4 threads, ORT
  is 6–11 % faster. With 256 tokens or more at 4 or more threads, Statim is faster: at 1,024 tokens
  on 8 threads, ORT takes 2.8× as long.
- **Serving, with both servers pinned to the same cores.**
  - With its default single worker, Statim answers 10–34 % more requests per second than ORT in
    every configuration measured. On 8 cores with one client: 2.58 against 2.00 req/s.
  - ORT has the lower median latency with one client: 257 against 286 ms.
  - With four clients, Statim needs `--workers 2` to match ORT's p95 latency. Its default single
    worker queues requests behind long ones.
- **8-bit.**
  - Statim's q8_0 keeps the f32 decisions: 30 of 2,850 held-out decisions change, with no
    accuracy change (exact McNemar p = 0.52).
  - ORT's standard dynamic int8 is the fastest variant on short inputs, but changes 443 decisions
    and costs 1.7 accuracy points (p = 0.009).
  - ORT's accurate 8-bit option (blockwise weights) is slower than Statim's q8_0 from 4 threads up.
- **Start and footprint.**

  | Measure | Statim | ORT |
  |---|---:|---:|
  | First answer after start, 8 cores | 0.45 s | 1.95 s |
  | Peak memory, f32 | 628 MiB | 729 MiB |
  | Peak memory, 8-bit | 281 MiB | 582 MiB |
  | Install | 5.2 MiB binary | 186 MiB of Python packages, without the interpreter |

## Parity

`bench/ort_export.py` exports the complete scoring graph to ONNX (opset 18): the encoder, the typed
decision transformer, the option scorer and the action scorer. Before anything is timed, it gates
the export on the logits (`bench/results/ort-cpu/export-report.json`).

| Comparison against the f32 PyTorch reference, 240 items | Argmax | max \|Δlogit\| |
|---|---:|---:|
| Original Laya checkpoint, ORT f32 (independent golden file) | 240/240 | 9.4e-5 |
| Shipped model (0.7.0 checkpoint), ORT f32 | 240/240 | 1.0e-5 |
| Shipped model, Statim f32 | 240/240 | 1.4e-5 |
| Shipped model, Statim q8_0 | 235/240 | 0.31 |
| Shipped model, ORT 8-bit blockwise (MatMulNBits, block 32) | 236/240 | 0.19 |
| Shipped model, ORT dynamic int8, per channel, reduce_range | 195/240 | 4.31 |
| Shipped model, ORT dynamic int8, per tensor | 196/240 | 7.96 |

- **The golden file.** `tests/data/golden_laya-multilingual.jsonl` belongs to the original
  checkpoint. For the shipped checkpoint, the export script computes the PyTorch reference on the
  same 240 items, and Statim's `test_model_parity` checks against it too.
- **Long sequences.** The length sweep below compares Statim f32 with ORT f32 on sequences of up
  to 1,024 tokens: every argmax matches, with max |Δlogit| 1.4e-5.
- **The ORT server.** It re-implements the request rendering and packing in Python. It reproduces
  all 240 golden token sequences exactly. On all 30 golden states it gives Statim's choices, with
  every probability within 1e-3.

## Held-out accuracy

Both engines served the same held-out items: the PII (11 languages) and emotion (8 languages)
suites of `bench/eval_categories.py`, 2,850 items. Per-item predictions were written with
`--predictions`, and `bench/compare_predictions.py` compared them pairwise against Statim f32.

| Against Statim f32 | Changed decisions | f32 right, other wrong | f32 wrong, other right | Accuracy | Exact McNemar p |
|---|---:|---:|---:|---|---:|
| ORT f32 | 0 | 0 | 0 | 0.7425 → 0.7425 | 1 |
| Statim q8_0 | 30 | 9 | 13 | 0.7425 → 0.7439 | 0.52 |
| ORT 8-bit blockwise | 17 | 7 | 7 | 0.7425 → 0.7425 | 1 |
| ORT dynamic int8 (per channel, reduce_range) | 443 | 193 | 144 | 0.7425 → 0.7253 | 0.009 |

By suite, ORT's dynamic int8 lowers PII from 0.855 to 0.838 (p = 0.018) and emotion from 0.588 to
0.571 (p = 0.18). The first attempt at the ORT f32 run was discarded: the previous server still held
the port, so the evaluation reached no server. All five runs were then repeated in full, each on its
own port.

## What ORT was given

A separate agent tuned the ORT side with one instruction: find ORT's fastest legitimate setup. Its
measurements are in `bench/results/ort-cpu/ort-tuning-*.json`, from 30 states, 3 passes and 8
threads on the same machine.

- **Graph optimization.** `ORT_ENABLE_ALL` fuses 24 MatMuls with their scaling and one
  Add+LayerNorm. It fuses no attention: this ModernBERT-style encoder uses rotary embeddings and
  alternates global and sliding-window attention.
- **The transformer optimizer** (`onnxruntime.transformers.optimizer`, model type `bert`) also
  fuses GELU (23 Gelu, 1 BiasGelu), but no attention, SkipLayerNorm or rotary embedding. It was
  slower than the plain export: 523 against 486 ms mean per state. So was the offline-optimized
  graph, at 535 ms.
- **Session settings.** `dynamic_block_base` 4 and denormals-as-zero changed the mean by under
  2 %. Turning spinning off was 13 % slower. The comparison therefore uses the plain export with
  `ORT_ENABLE_ALL`, denormals-as-zero, sequential execution and ORT's default spinning. The HTTP
  benchmark also runs ORT with spinning off, as a sensitivity check.
- **8-bit.** Several variants were measured for speed and parity:
  - dynamic int8, per tensor or per channel, with and without `reduce_range`;
  - MatMulNBits 8-bit blockwise weights at accuracy levels 0 and 4, the latter with int8 compute.

  Blockwise with f32 compute (accuracy level 0) was the fastest variant that keeps the decisions,
  so it is the 8-bit row. The best dynamic int8 variant is shown alongside it, because it is ORT's
  standard quantization path.
- **What the dynamic int8 rows measure.** On AVX2 without VNNI, ORT's documentation recommends
  `reduce_range` against saturation in its u8s8 kernels. Even so, dynamic int8 loses decisions on
  this model: 195/240.
- **The embedding.** ORT 1.30's weight-only quantizer quantizes the token embedding only at 4
  bits, so the 8-bit blockwise file keeps its 256,000 × 768 embedding in f32.

## Protocol

| | |
|---|---|
| Machine | AMD Ryzen 7 5800X: 8 cores, 16 threads, AVX2 and FMA, no AVX-512, no VNNI. 16 GB RAM, Linux 6.18, `performance` governor, otherwise idle |
| Statim | 0.9.0. Raw scoring with `test_model_parity` built with the release flags (`-DSTATIM_NATIVE=OFF`); HTTP with the released `statim-0.9.0-linux-x86_64-cpu` binary |
| ORT | onnxruntime 1.30.0, CPU execution provider, from Python 3.12 |
| Model | The shipped statim-decide-multilingual-base checkpoint. GGUF f32 (token embedding in f16, as the converter writes it) and q8_0 (`statim-quantize`); ONNX f32, 8-bit blockwise and dynamic int8 |
| Inputs | The 30 states × 8 questions of `tests/data/golden_inputs.json`, already tokenized in `build-ort/golden_v9.jsonl`. 29 states are at most 144 tokens long; one is 770 |
| Raw scoring | One batch of 8 sequences per state, with identical token ids for both engines. One warm-up pass, then 3 passes; the median per state across passes is reported |
| Length sweep | One batch of the 8 questions over one long state, cut so that every sequence is exactly the target length. One warm-up pass, then 3 passes, median |
| HTTP | Each state with its 8 questions as one `POST /v1/systemone`. 4 warm-up requests, then 60 requests per row, from 1 or 4 closed-loop clients. Each server is pinned with `taskset` to one logical CPU on each of 4 or 8 physical cores. The load generator runs on the other cores (for 4), or on the SMT siblings (for 8) |

Every process runs with `CUDA_VISIBLE_DEVICES=""`. The raw scoring calls ORT through its Python
bindings. The overhead was measured by comparing the wall time of `session.run` with ORT's own
profiled `model_run` time. It is about 50 µs per call, at most 0.02 % of the call. So it is
negligible for this workload, though that is not a measurement of a native C++ ORT integration
(`bench/results/ort-cpu/binding-overhead.json`).

| Threads | Median `session.run` | Median ORT `model_run` | Median overhead | Share |
|---:|---:|---:|---:|---:|
| 1 | 1,379.0 ms | 1,378.9 ms | 49 µs | 0.004 % |
| 8 | 267.9 ms | 267.8 ms | 53 µs | 0.020 % |

## Raw scoring

These are the 30 golden states, one batch of 8 questions per state. The table gives the mean
milliseconds per state: "short" is the 29 states of at most 144 tokens, and "770 tok" is the
remaining state (`bench/results/ort-cpu/raw.json`).

| Threads | Statim f32, short | ORT f32, short | Statim f32, 770 tok | ORT f32, 770 tok | Statim q8_0, short | ORT 8-bit, short | ORT dyn. int8, short | Statim q8_0, 770 tok | ORT 8-bit, 770 tok | ORT dyn. int8, 770 tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,651 | 1,424 | 16,862 | 19,169 | 1,837 | 1,714 | 812 | 18,886 | 19,928 | 13,260 |
| 4 | 470 | 431 | 4,836 | 7,944 | 520 | 524 | 253 | 5,421 | 8,273 | 6,356 |
| 8 | 272 | 268 | 2,907 | 6,961 | 307 | 328 | 169 | 3,428 | 7,306 | 6,077 |
| 16 | 442 | 372 | 4,240 | 8,027 | 381 | 435 | 239 | 3,611 | 8,411 | 7,060 |

- **Short states, f32.** Statim takes 16 % longer than ORT at 1 thread, 9 % at 4 threads and 1 %
  at 8 threads.
- **The 770-token state, f32.** Statim is faster by 1.1× at 1 thread and 2.4× at 8 threads.
- **16 threads.** These are 8 cores with SMT. Both engines are slower than at 8 threads.

## Sequence length

`--length-sweep` uses no padding: every sequence in a batch has the target length. Each cell is
milliseconds per batch of 8 sequences, and "ORT / Statim" above 1 means Statim is faster
(`bench/results/ort-cpu/length-sweep.json`).

| Tokens per sequence | Threads | Statim f32 ms | ORT f32 ms | ORT / Statim | Statim q8_0 ms | ORT 8-bit ms | ORT / Statim |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 2,418 | 2,163 | 0.89× | 2,671 | 2,446 | 0.92× |
| 256 | 1 | 4,919 | 4,714 | 0.96× | 5,539 | 5,131 | 0.93× |
| 512 | 1 | 10,605 | 10,757 | 1.01× | 11,928 | 11,498 | 0.96× |
| 1024 | 1 | 23,330 | 28,067 | 1.20× | 26,545 | 29,450 | 1.11× |
| 128 | 4 | 676 | 638 | 0.94× | 736 | 774 | 1.05× |
| 256 | 4 | 1,404 | 1,602 | 1.14× | 1,571 | 1,765 | 1.12× |
| 512 | 4 | 3,028 | 4,301 | 1.42× | 3,376 | 4,475 | 1.33× |
| 1024 | 4 | 6,805 | 12,426 | 1.83× | 7,576 | 12,997 | 1.72× |
| 128 | 8 | 380 | 405 | 1.06× | 433 | 509 | 1.18× |
| 256 | 8 | 807 | 1,153 | 1.43× | 918 | 1,326 | 1.45× |
| 512 | 8 | 1,780 | 3,659 | 2.06× | 2,143 | 3,961 | 1.85× |
| 1024 | 8 | 4,038 | 11,310 | 2.80× | 4,757 | 11,699 | 2.46× |

- **Crossover.** ORT leads on short sequences at low thread counts. Statim leads once sequences
  are long or several threads share the work.
- **A plausible cause, not isolated.** On the CPU, Statim computes attention with ggml's fused
  flash-attention kernel (`flash_attn`, on by default). ORT's graph for this model has no attention
  fusion, so it materializes the attention scores. No ablation of Statim with flash attention off
  was run.
- **Padding does not explain ORT's lead on short inputs.** The sweep has no padding, and ORT still
  leads at 128 tokens on 1–4 threads. That lead is in the per-operation work at small sizes; where
  exactly is open.

## HTTP end to end

Both servers are pinned to the same physical cores (`bench/results/ort-cpu/http-pinned.json`).
"Busy cores" is the server's CPU time divided by wall time. p99 over 60 requests is close to the
slowest request, which is the 770-token state.

| Variant | Cores | Server | Clients | p50 ms | p95 ms | p99 ms | req/s | Busy cores | Context switches / request |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| f32 | 4 | Statim, 1 worker | 1 | 460 | 774 | 4,864 | 1.60 | 3.88 | 1 |
| f32 | 4 | ORT | 1 | 420 | 740 | 8,072 | 1.46 | 3.83 | 5 |
| f32 | 4 | Statim, 1 worker | 4 | 1,825 | 6,315 | 6,529 | 1.60 | 3.89 | 1 |
| f32 | 4 | Statim, 2 workers | 4 | 1,812 | 3,068 | 10,347 | 1.47 | 3.52 | 1 |
| f32 | 4 | Statim, 4 workers | 4 | 1,829 | 3,084 | 18,973 | 1.24 | 2.98 | 1 |
| f32 | 4 | ORT | 4 | 1,884 | 3,245 | 21,392 | 1.45 | 3.93 | 5 |
| f32 | 8 | Statim, 1 worker | 1 | 286 | 481 | 3,009 | 2.58 | 7.29 | 1 |
| f32 | 8 | ORT | 1 | 257 | 471 | 7,188 | 2.00 | 7.42 | 4 |
| f32 | 8 | Statim, 1 worker | 4 | 1,137 | 3,883 | 4,000 | 2.61 | 7.41 | 1 |
| f32 | 8 | Statim, 2 workers | 4 | 1,076 | 1,816 | 5,950 | 2.54 | 6.82 | 1 |
| f32 | 8 | Statim, 4 workers | 4 | 1,045 | 1,775 | 10,751 | 2.21 | 5.98 | 1 |
| f32 | 8 | ORT | 4 | 1,186 | 2,010 | 15,084 | 2.07 | 7.74 | 4 |
| 8-bit | 4 | Statim q8_0, 1 worker | 1 | 509 | 875 | 5,471 | 1.44 | 3.85 | 1 |
| 8-bit | 4 | ORT blockwise | 1 | 503 | 877 | 8,402 | 1.28 | 3.85 | 5 |
| 8-bit | 4 | Statim q8_0, 2 workers | 4 | 1,967 | 3,376 | 11,640 | 1.34 | 3.50 | 1 |
| 8-bit | 4 | ORT blockwise | 4 | 2,359 | 3,572 | 22,835 | 1.23 | 3.94 | 5 |
| 8-bit | 8 | Statim q8_0, 1 worker | 1 | 310 | 486 | 3,375 | 2.36 | 6.82 | 1 |
| 8-bit | 8 | ORT blockwise | 1 | 319 | 576 | 7,445 | 1.76 | 7.50 | 5 |
| 8-bit | 8 | Statim q8_0, 2 workers | 4 | 1,172 | 2,018 | 6,630 | 2.32 | 6.59 | 1 |
| 8-bit | 8 | ORT blockwise | 4 | 1,556 | 2,419 | 15,989 | 1.76 | 7.79 | 4 |

- **Throughput.** With its default single worker, Statim's throughput is higher than ORT's in every
  pinned configuration, by 10 % (f32, 4 cores) to 34 % (8-bit, 8 cores). More workers trade some
  of it for lower latency.
- **Median latency with one client.** ORT is faster: 420 against 460 ms on 4 cores and 257 against
  286 ms on 8 cores. Most requests are short states.
- **Tail latency.** ORT's slowest requests, the long state, take far longer: p99 7,188 against
  3,009 ms on 8 cores with one client.
- **Four clients.** With its default single worker, Statim serializes requests. Short requests
  then wait behind the long one, and p95 rises to 3,883 ms on 8 cores. With `--workers 2`,
  Statim's p95 is below ORT's, 1,816 against 2,010 ms, at 23 % higher throughput. More workers
  lower the median further, but cost throughput.
- **Sensitivity checks** (all rows in the JSON).
  - ORT with spinning off: p95 and throughput within 8 % of spinning on.
  - Statim's micro-batching (`--batch-window-ms 2`) does not help on the CPU, as the README says.
- **Pinning matters for ORT.** An earlier run without pinning
  (`bench/results/ort-cpu/http-unpinned.json`) gave ORT at "4 threads" a p95 of 1,881 ms and 2.06
  req/s with 4 clients. Each concurrent `Run` call adds its calling thread to ORT's intra-op pool,
  so ORT can use more cores than its thread setting; pinned to 4 cores, it drops to 3,245 ms and
  1.45 req/s. Statim's `--threads` is a hard total that its workers share. The pinned rows are the
  comparison.

## Start, memory and size

| Variant | Cores | Server | First answer after start | Peak RSS |
|---|---:|---|---:|---:|
| f32 | 4 | Statim | 654 ms | 627 MiB |
| f32 | 4 | ORT | 2,300 ms | 729 MiB |
| f32 | 8 | Statim | 449 ms | 628 MiB |
| f32 | 8 | ORT | 1,949 ms | 729 MiB |
| 8-bit | 4 | Statim q8_0 | 668 ms | 281 MiB |
| 8-bit | 4 | ORT blockwise | 2,157 ms | 567 MiB |
| 8-bit | 8 | Statim q8_0 | 445 ms | 281 MiB |
| 8-bit | 8 | ORT blockwise | 1,827 ms | 582 MiB |

"First answer" runs from process start to the first complete response. For ORT this includes
starting Python and importing onnxruntime.

**Model files.** Each ONNX size includes the tokenizer and configuration files the ORT server
loads; a GGUF file contains them already.

| Model files | Statim (GGUF) | ORT (ONNX) |
|---|---:|---:|
| f32 | 866 MiB | 1,263 MiB |
| 8-bit (q8_0 against blockwise) | 340 MiB | 924 MiB |
| ORT dynamic int8 | — | 343 MiB |

The GGUF f32 file keeps the token embedding in f16, and the ONNX f32 file in f32. The ONNX 8-bit
file keeps it in f32, because ORT 1.30 quantizes embeddings only at 4 bits.

**Install.** Statim's release is one 5.2 MiB executable, in a 2.8 MiB archive. The ORT server's
Python packages take 186 MiB (onnxruntime, tokenizers, numpy and their dependencies), not counting
the CPython interpreter and standard library, which take 58 MiB.

## What this does not show

- **One machine.** Zen 3 is AVX2 without VNNI. ARM (dotprod, i8mm) and AVX-512-VNNI, the intended
  int8 targets, were not measured. Nor were Intel CPUs, where the OpenVINO execution provider is
  ORT's strongest CPU path.
- **No GPU.** ORT's CUDA and TensorRT execution providers were not compared with Statim's CUDA and
  Vulkan backends.
- **No native ORT server.** The HTTP rows compare Statim's server with a small Python server around
  ORT; that is a product comparison, not a runtime comparison. The raw and length-sweep rows are
  the runtime comparison, and the Python binding costs them at most 0.02 %. A C++ ORT server, and
  a serving stack such as Triton with dynamic batching, were not built or measured.
- **Sample sizes.**
  - Raw scoring uses 3 passes per state.
  - HTTP rows use 60 requests, so p99 is effectively the slowest request.
  - Accuracy covers two suites with 2,850 items.
- **One model.** Other architectures may behave differently.

An earlier pinned HTTP run stopped on a harness bug in the CPU placement after its first block, and
was discarded. The run above was repeated in full.

## Reproduce

Prerequisites:

- Statim built with tests (`docs/BUILD.md`), with the release flags; the default binary paths are
  `build-rel/statim`, `build-rel/test_model_parity` and `build-rel/statim-quantize`.
- The shipped checkpoint in `models/laya-multilingual-v9` and its GGUF
  `models/laya-multilingual-v9-f32.gguf`, and the original checkpoint `models/laya-multilingual`
  (`tools/fetch_models.sh multilingual`) for the independent graph proof.
- The 0.9.0 CPU release, extracted under `build-ort/release-0.9.0/`.
- The training environment `.venv-train` from REPRODUCE.md, for the export's PyTorch reference.

```bash
python3 -m venv build-ort/venv
build-ort/venv/bin/pip install onnx==1.23.0 onnxruntime==1.30.0 onnxscript tokenizers numpy psutil
python3 -m venv build-ort/venv-server
build-ort/venv-server/bin/pip install onnxruntime==1.30.0 tokenizers numpy

# Export, quantize, and run every parity gate
CUDA_VISIBLE_DEVICES="" PYTHONPATH=build-ort/venv/lib/python3.12/site-packages \
  .venv-train/bin/python bench/ort_export.py --threads 4

# Raw scoring, length sweep, Python-binding overhead, and pinned HTTP
CUDA_VISIBLE_DEVICES="" build-ort/venv/bin/python bench/ort_compare.py \
  --thread-counts 1 4 8 16 --repeats 3 --out build-ort/cpu-raw-results.json
CUDA_VISIBLE_DEVICES="" build-ort/venv/bin/python bench/ort_compare.py --length-sweep \
  --lengths 128 256 512 1024 --thread-counts 1 4 8 --repeats 3 --out build-ort/cpu-length-sweep-results.json
CUDA_VISIBLE_DEVICES="" build-ort/venv/bin/python bench/ort_compare.py --binding-overhead \
  --thread-counts 1 8 --repeats 3 --out build-ort/cpu-binding-results.json
CUDA_VISIBLE_DEVICES="" build-ort/venv/bin/python bench/ort_compare.py --socket-benchmark --pin \
  --thread-counts 4 8 --requests 60 --warmup-requests 4 --ort-spinning-variants 1 0 \
  --out build-ort/cpu-http-pinned-results.json

# Tables from any result file
build-ort/venv/bin/python bench/ort_compare.py --render-md build-ort/cpu-raw-results.json
```

**Held-out accuracy.** Start one server per variant, each on its own port. Statim:
`statim serve --device cpu --threads 8 --port 8201 -m multilingual=MODEL.gguf`. ORT:
`build-ort/venv-server/bin/python bench/ort_compare.py --serve-ort --threads 8 --port 8203 --onnx
MODEL.onnx`. Then evaluate each server and compare the predictions:

```bash
.venv-train/bin/python bench/eval_categories.py --url http://127.0.0.1:8201 --model multilingual \
  --suites pii emotion --out build-ort/accuracy-statim-f32.jsonl --predictions build-ort/predictions-statim-f32.jsonl
python3 bench/compare_predictions.py build-ort/predictions-statim-f32.jsonl build-ort/predictions-ort-f32.jsonl
```

**GPU providers.** They run with the same harness, from an environment with a GPU build of ORT:
`bench/ort_compare.py --ort-only --ort-provider cuda` (or `tensorrt`), with `CUDA_VISIBLE_DEVICES`
set.
