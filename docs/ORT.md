# Statim and ONNX Runtime on CPU

A common question about Statim: why not export the model to ONNX and serve it with ONNX Runtime
(ORT)? This page answers it for the shipped multilingual model on one CPU. Both engines get the
same inputs, and parity gates run before any timing. The target is that Statim should beat or
match general-purpose runtimes on its own decision workload without sacrificing fidelity. Where ORT
is faster, this page says so.

The measurements are for Statim 0.9.2. The 0.9.0/0.9.1 measurements this page first published are
in the repository at tag `v0.9.1`, and the section [What changed in 0.9.2](#what-changed-in-092)
compares the two.

## In short

The machine is an AMD Ryzen 7 5800X with 8 cores. The sections below give the details and the
source file of every number.

- **Same answers.**
  - On the 240 golden items, ORT f32 and Statim f32 both reproduce the PyTorch reference within
    2.4e-5.
  - On sequences of up to 1,024 tokens they agree with each other within 2.2e-5.
  - On a 2,850-item held-out sample they make the same decision on every item.
- **Raw scoring, f32.**
  - At 1 thread, Statim and ORT are level on short inputs: within 1.3 % on the golden states,
    0.4 % at 128 tokens.
  - At 4 and 8 threads, Statim is faster at every length measured: by 3–5 % on short inputs at 4
    threads, and 2.6× at 1,024 tokens on 8 threads.
  - At 16 threads, where SMT is in play, Statim takes 25 % longer than ORT on the short golden
    states.
- **Serving.** Statim's server against a small Python server around ORT, both pinned to the same
  cores. This is a product comparison, not a runtime comparison.
  - With its default single worker, Statim answers 11–32 % more requests per second in every
    configuration measured.
  - With one client on 8 cores, ORT's f32 median latency is lower (267 against 283 ms). On 4 cores
    Statim's is lower (409 against 416 ms).
  - With four clients, Statim with `--workers 2` has the lower p95 in every configuration.
- **8-bit.**
  - **Statim's q8_0** changes 30 of the 2,850 held-out decisions. Accuracy goes from 0.7425 to
    0.7439, which is not distinguishable from no change (exact McNemar p = 0.523).
  - **ORT's per-channel dynamic int8** is the fastest variant on the short golden states. It
    changes 443 decisions and costs 1.7 accuracy points (p = 0.00883).
  - **ORT's blockwise 8-bit weights** (f32 compute) keep the accuracy. They are 10 % faster than
    q8_0 on short inputs at 1 thread, and level or slower from 4 threads up.
- **Start and footprint.**

  | Measure | Statim | ORT |
  |---|---:|---:|
  | First answer after process start, f32, 8 cores | 0.45 s | 2.06 s (includes one warm-up decision) |
  | Resident memory in the start-up window, f32 | 637 MiB | 729 MiB |
  | Resident memory in the start-up window, 8-bit | 276 MiB | 579–581 MiB |
  | Install | 5.2 MiB executable | 186 MiB of Python packages, without the interpreter |

## And the PyTorch reference

The `laya` Python package on PyTorch, the reference Statim reproduces, was measured on the same
machine with the same token ids under a protocol fixed beforehand
([results](reproductions/laya-speed-2026-10-03.md)). Mean ms per state on the 29 short states:

| Threads | Statim | ORT | Laya (PyTorch) |
|---:|---:|---:|---:|
| 1 | 1,447 | 1,419 | 1,451 |
| 4 | 420 | 433 | 545 |
| 8 | 271 | 266 | 299 |

The Statim column is a control run on the day of the Laya measurement; the ORT column is this page's
0.9.2 measurement. Laya starts in 5.0 s and peaks at 2.65 GiB resident memory.

## What changed in 0.9.2

In 0.9.1, ORT was faster than Statim on short f32 inputs at 1 to 4 threads, by up to 16 %.

- **Profiling found the cause.** About 90 % of that time goes into the encoder's four projections
  (QKV, attention output, MLP up and down). There, ggml's f32 GEMM (llamafile's tinyBLAS) reached
  about 110 GFLOPS on one core, and ORT's MLAS about 127 GFLOPS. The profiler is `STATIM_PROFILE`
  with `bench/profile_short.py`.
- **A new f32 GEMM.** Statim 0.9.2 computes these projections with its own packed AVX2/FMA SGEMM
  (`src/kernels.cpp`), used on x86-64 CPUs with AVX2 and FMA.
  - It packs the activations once per call and each thread's weight block into a buffer that stays
    in L2 (GotoBLAS order).
  - Its 6×16 microkernel keeps 12 accumulators in registers.
  - Quantized, GPU and LoRA paths are unchanged, and `STATIM_SGEMM=0` restores the previous path.
  - Every f32 decision is unchanged: 0 of 2,850 held-out decisions differ from 0.9.1 or from ORT.
    The parity gates hold, with max |Δlogit| 2.4e-5 against the PyTorch reference.

| Mean ms per state, 29 short golden states, f32 | 0.9.1 | 0.9.2 | ORT |
|---|---:|---:|---:|
| 1 thread | 1,651 | 1,438 | 1,419 |
| 4 threads | 470 | 419 | 433 |
| 8 threads | 272 | 257 | 266 |
| 16 threads | 442 | 457 | 366 |

The ORT column is from the 0.9.2 run; ORT's own numbers moved by at most 1.6 % between the runs.
0.9.2 also removes a start-up regression that 0.9.1 introduced. The GGUF preflight skipped each of
the tokenizer's ~256,000 strings with a stream seek, which cost about 0.8 s per model load.

## Parity

`bench/ort_export.py` exports the complete scoring graph to ONNX (opset 18): the encoder, the typed
decision transformer, the option scorer and the action scorer. Before anything is timed, the export
is gated on the logits (`bench/results/ort-cpu/export-report.json`).

| Comparison against the f32 PyTorch reference, 240 items | Argmax | max \|Δlogit\| |
|---|---:|---:|
| Original Laya checkpoint, ORT f32 (independent golden file) | 240/240 | 9.4e-5 |
| Shipped model (0.7.0 checkpoint), ORT f32 | 240/240 | 1.0e-5 |
| Shipped model, Statim 0.9.2 f32 | 240/240 | 2.4e-5 |
| Shipped model, Statim q8_0 | 235/240 | 0.31 |
| Shipped model, ORT 8-bit blockwise (MatMulNBits, block 32, f32 compute) | 236/240 | 0.19 |
| Shipped model, ORT dynamic int8, per channel, `reduce_range` | 195/240 | 4.31 |
| Shipped model, ORT dynamic int8, per tensor (`quantize_dynamic` defaults) | 196/240 | 7.96 |

- **Where the Statim f32 figure comes from.** The export report records Statim 0.9.1 at 1.4e-5.
  2.4e-5 is `test_model_parity` with the 0.9.2 GEMM, which sums in a different order.
- **The golden file.** `tests/data/golden_laya-multilingual.jsonl` belongs to the original
  checkpoint. For the shipped checkpoint, the export script computes the PyTorch reference on the
  same 240 items, and Statim's `test_model_parity` is checked against it too.
- **Long sequences.** The length sweep below compares Statim f32 with ORT f32 on sequences of up to
  1,024 tokens: every argmax matches, with max |Δlogit| 2.2e-5.
- **The ORT server.** It re-implements the request rendering and packing in Python. It reproduces
  all 240 golden token sequences exactly, and on all 30 golden states gives Statim's choices with
  every probability within 1e-3.

## Held-out accuracy

- **The sample.** The PII suite (11 languages) and the emotion suite (8 languages) of
  `bench/eval_categories.py`: 150 items per language, seed 20260927, 2,850 items in total.
- **The runs.** Each engine and variant answered the same sample and wrote per-item predictions
  with `--predictions`. `bench/compare_predictions.py` then compared each run pairwise with Statim
  f32 (0.9.2; its predictions are identical to 0.9.1's).

| Against Statim f32 | Changed decisions | f32 right, other wrong | f32 wrong, other right | Accuracy | Exact McNemar p |
|---|---:|---:|---:|---|---:|
| ORT f32 | 0 | 0 | 0 | 0.7425 → 0.7425 | 1 |
| Statim q8_0 | 30 | 9 | 13 | 0.7425 → 0.7439 | 0.523 |
| ORT 8-bit blockwise | 17 | 7 | 7 | 0.7425 → 0.7425 | 1 |
| ORT dynamic int8, per channel, `reduce_range` | 443 | 193 | 144 | 0.7425 → 0.7253 | 0.00883 |

- **Size of the dynamic int8 drop.** The paired standard error of the 1.7-point drop is 0.64
  points.
- **By suite.** Dynamic int8 lowers PII from 0.855 to 0.838 (p = 0.018). On emotion the point
  estimate falls from 0.588 to 0.571, but the paired test does not resolve it (p = 0.175).
- **Not tested:** the per-tensor dynamic variant was not run on the held-out sample.
- **A discarded run.** The first attempt at the ORT f32 run reached no server, because the previous
  server still held the port. All runs were then repeated in full, each on its own port.

## What ORT was given

A separate agent tuned the ORT side with one instruction: find ORT's fastest legitimate setup.

- **Where the results are.** Timings are in `bench/results/ort-cpu/ort-tuning-*.json` (30 states,
  3 passes, 8 threads, on the same machine). The op counts of the graphs are in
  `ort-tuning-graph-ops.json`.
- **Graph optimization.** `ORT_ENABLE_ALL` fuses 24 MatMuls with their scaling and one
  Add+LayerNorm. It fuses no attention: this ModernBERT-style encoder uses rotary embeddings, and
  alternates global and sliding-window attention.
- **The transformer optimizer** (`onnxruntime.transformers.optimizer`, model type `bert`) also fuses
  GELU (23 Gelu, 1 BiasGelu). It fuses no attention, SkipLayerNorm or rotary embedding.
  - It was slower than the plain export: 523 against 486 ms mean per state.
  - So was the offline-optimized graph, at 535 ms.
- **Session settings.** `dynamic_block_base` 4 and denormals-as-zero changed the mean by under 2 %.
  Turning spinning off was 13 % slower.
- **The f32 setup chosen.** The plain export with `ORT_ENABLE_ALL`, denormals-as-zero, sequential
  execution and ORT's default spinning. The HTTP benchmark also runs ORT with spinning off, as a
  sensitivity check.
- **8-bit variants measured:**
  - dynamic int8, per tensor or per channel, with and without `reduce_range`;
  - MatMulNBits 8-bit blockwise weights at accuracy level 0 (f32 compute) and at accuracy level 4
    (int8 compute, closer to how q8_0 computes).

  Both blockwise levels change 4 of the 240 golden argmaxes. Level 0 was faster (546 against
  646 ms mean at 8 threads), so it is the 8-bit row.
- **Dynamic int8.** The per-channel `reduce_range` variant is shown alongside, as ORT's dynamic
  quantization path. On AVX2 without VNNI, ORT's documentation recommends `reduce_range` against
  saturation in its u8s8 kernels. Even so, dynamic int8 loses decisions on this model (195/240).
- **The token embedding.** ORT 1.30's weight-only quantizer quantizes Gather only at 4 bits, so the
  8-bit blockwise file keeps its 256,000 × 768 token embedding in f32.
- **Where the 8-bit files come from.** The tuning agent built the files that were timed.
  `bench/ort_export.py` now builds them with the same settings, and the weight files are
  byte-identical (SHA-256 of the `.data` files: `3ae660c1…` blockwise, `6eb3033f…` dynamic).

## Protocol

| | |
|---|---|
| Machine | AMD Ryzen 7 5800X: 8 cores, 16 threads, AVX2 and FMA, no AVX-512, no VNNI. 16 GB RAM, Linux 6.18.7, `performance` governor, otherwise idle |
| Statim | 0.9.2, built with the release flags (`-DCMAKE_BUILD_TYPE=Release -DSTATIM_NATIVE=OFF`): `test_model_parity` for raw scoring and the length sweep, `statim serve` for HTTP |
| ORT | onnxruntime 1.30.0, CPU execution provider, called from Python 3.12 (tokenizers 0.23.2, numpy 2.5.3) |
| Model | The shipped statim-decide-multilingual-base checkpoint. GGUF f32 (token embedding in f16, as the converter writes it) and q8_0 (`statim-quantize`); ONNX f32 (embedding in f32), 8-bit blockwise and dynamic int8 |
| Inputs | The 30 states × 8 questions of `tests/data/golden_inputs.json`, already tokenized in `build-ort/golden_v9.jsonl`. 29 states are at most 144 tokens long; one is 770 |
| Raw scoring | One batch of 8 sequences per state, with identical token ids for both engines. One warm-up pass, then 3 passes; only the median per state is stored |
| Length sweep | One batch of the 8 questions over one long state, cut so that every sequence is exactly the target length. One warm-up pass, then 3 passes, median |
| HTTP | Each state with its 8 questions as one `POST /v1/systemone`. 4 warm-up requests, then 60 requests per row, from 1 or 4 closed-loop clients. Each server is pinned with `taskset` to one logical CPU on each of 4 or 8 physical cores. The load generator runs on the other cores for the 4-core rows, and on the SMT siblings of the server's cores for the 8-core rows |

**Python-binding overhead.** Every process runs with `CUDA_VISIBLE_DEVICES=""`. The raw scoring
calls ORT through its Python bindings. Comparing the wall time of `session.run` with ORT's own
profiled `model_run` time puts the overhead at about 50 µs per call. The median share is at most
0.020 %, and the largest overhead recorded is 63 µs. So it is negligible for this workload. It is
not a measurement of a native C++ ORT integration (`bench/results/ort-cpu/binding-overhead.json`).

| Threads | Median `session.run` | Median ORT `model_run` | Median overhead | Max overhead | Median share |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,379.0 ms | 1,378.9 ms | 49 µs | 55 µs | 0.004 % |
| 8 | 267.9 ms | 267.8 ms | 53 µs | 63 µs | 0.020 % |

## Raw scoring

These are the 30 golden states, one batch of 8 questions per state (`bench/results/ort-cpu/raw.json`).
Each cell is the mean milliseconds per state.

- "Short" is the 29 states of at most 144 tokens; "770 tok" is the remaining state.
- Statim f32 keeps the token embedding in f16, and ORT f32 in f32. This difference was not isolated.

| Threads | Statim f32, short | ORT f32, short | Statim f32, 770 tok | ORT f32, 770 tok | Statim q8_0, short | ORT 8-bit, short | ORT dyn. int8, short | Statim q8_0, 770 tok | ORT 8-bit, 770 tok | ORT dyn. int8, 770 tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,438 | 1,419 | 15,170 | 18,947 | 1,858 | 1,688 | 810 | 18,751 | 19,901 | 13,001 |
| 4 | 419 | 433 | 4,573 | 8,016 | 533 | 525 | 255 | 5,467 | 8,191 | 6,367 |
| 8 | 257 | 266 | 3,065 | 7,006 | 319 | 327 | 168 | 3,503 | 7,227 | 6,217 |
| 16 | 457 | 366 | 3,455 | 8,128 | 417 | 438 | 250 | 3,752 | 8,491 | 7,189 |

- **Short states, f32.**
  - At 1 thread, Statim takes 1.3 % longer than ORT, a tie at this precision.
  - At 4 and 8 threads, Statim is 3 % faster.
  - At 16 threads, Statim takes 25 % longer.
- **The 770-token state, f32.** ORT takes 1.25× as long as Statim at 1 thread, 1.75× at 4, 2.29×
  at 8 and 2.35× at 16.
- **8-bit, short states.**
  - At 1 thread, Statim's q8_0 takes 10 % longer than ORT's blockwise 8-bit.
  - At 4 threads they are within 2 %.
  - At 8 and 16 threads, ORT's blockwise 8-bit takes 2 % and 5 % longer.
  - ORT's dynamic int8 is the fastest column on the short states at every thread count. On the
    770-token state it is slower than q8_0 from 4 threads up.
- **16 threads.** These are 8 cores with SMT. Every engine and variant is slower than at 8
  threads.

## Sequence length

`--length-sweep` uses no padding: every sequence in a batch has exactly the target length
(`bench/results/ort-cpu/length-sweep.json`).

- Each cell is milliseconds per batch of 8 sequences.
- "ORT / Statim" above 1 means Statim is faster.
- The embedding note above applies here too.

| Tokens per sequence | Threads | Statim f32 ms | ORT f32 ms | ORT / Statim | Statim q8_0 ms | ORT 8-bit ms | ORT / Statim |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 2,102 | 2,111 | 1.00× | 2,694 | 2,450 | 0.91× |
| 256 | 1 | 4,382 | 4,686 | 1.07× | 5,644 | 5,119 | 0.91× |
| 512 | 1 | 9,425 | 10,827 | 1.15× | 11,868 | 11,536 | 0.97× |
| 1024 | 1 | 21,771 | 28,200 | 1.30× | 26,191 | 29,621 | 1.13× |
| 128 | 4 | 606 | 638 | 1.05× | 752 | 761 | 1.01× |
| 256 | 4 | 1,279 | 1,590 | 1.24× | 1,586 | 1,776 | 1.12× |
| 512 | 4 | 2,831 | 4,311 | 1.52× | 3,404 | 4,559 | 1.34× |
| 1024 | 4 | 6,448 | 12,497 | 1.94× | 7,620 | 13,025 | 1.71× |
| 128 | 8 | 371 | 416 | 1.12× | 441 | 497 | 1.13× |
| 256 | 8 | 812 | 1,192 | 1.47× | 953 | 1,315 | 1.38× |
| 512 | 8 | 1,854 | 3,741 | 2.02× | 2,153 | 3,978 | 1.85× |
| 1024 | 8 | 4,399 | 11,421 | 2.60× | 4,820 | 11,659 | 2.42× |

- **f32.** Statim is level with ORT at 128 tokens on 1 thread, and faster in every other cell. The
  gap grows with length and threads. The sweep has no 16-thread rows.
- **8-bit.** ORT's blockwise 8-bit leads on 1 thread up to 512 tokens. Statim's q8_0 leads
  everywhere else.
- **Two plausible causes of the long-sequence gap, neither isolated.**
  - On the CPU, Statim computes attention with ggml's fused flash-attention kernel (`flash_attn`,
    on by default). ORT's graph for this model has no attention fusion, so it materializes the
    attention scores.
  - In the last layer of the decision head, Statim computes queries, the output projection and the
    feed-forward only for the option-marker rows (`src/model.cpp`); the exported graph computes
    them for every row.

  No ablation of either was run.

## HTTP end to end

This compares Statim's server with a Python `ThreadingHTTPServer` around ORT, which also tokenizes
and renders in Python. Both are pinned to the same physical cores
(`bench/results/ort-cpu/http-pinned.json`).

- **Busy cores** is the server's CPU time divided by wall time.
- **p99** over 60 requests is the slowest request.
- **Omitted rows.** The table leaves out ORT with spinning off and Statim with micro-batching.
  They are in the JSON.

| Variant | Cores | Server | Clients | p50 ms | p95 ms | p99 (max) ms | req/s | Busy cores |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| f32 | 4 | Statim, 1 worker | 1 | 409 | 688 | 4,631 | 1.77 | 3.83 |
| f32 | 4 | ORT | 1 | 416 | 754 | 8,152 | 1.46 | 3.83 |
| f32 | 4 | Statim, 1 worker | 4 | 1,648 | 5,819 | 6,023 | 1.78 | 3.84 |
| f32 | 4 | Statim, 2 workers | 4 | 1,626 | 2,743 | 9,721 | 1.63 | 3.46 |
| f32 | 4 | Statim, 4 workers | 4 | 1,551 | 2,690 | 18,178 | 1.41 | 2.99 |
| f32 | 4 | ORT | 4 | 1,919 | 2,970 | 20,470 | 1.43 | 3.93 |
| f32 | 8 | Statim, 1 worker | 1 | 283 | 464 | 3,173 | 2.58 | 6.68 |
| f32 | 8 | ORT | 1 | 267 | 467 | 7,236 | 1.97 | 7.43 |
| f32 | 8 | Statim, 1 worker | 4 | 1,110 | 3,945 | 4,057 | 2.62 | 6.80 |
| f32 | 8 | Statim, 2 workers | 4 | 1,004 | 1,735 | 5,987 | 2.69 | 6.50 |
| f32 | 8 | Statim, 4 workers | 4 | 944 | 1,639 | 11,092 | 2.40 | 5.90 |
| f32 | 8 | ORT | 4 | 1,188 | 2,045 | 15,536 | 2.03 | 7.74 |
| 8-bit | 4 | Statim q8_0, 1 worker | 1 | 525 | 889 | 5,484 | 1.41 | 3.80 |
| 8-bit | 4 | ORT blockwise | 1 | 507 | 878 | 8,397 | 1.27 | 3.85 |
| 8-bit | 4 | Statim q8_0, 1 worker | 4 | 2,072 | 7,039 | 7,286 | 1.42 | 3.82 |
| 8-bit | 4 | Statim q8_0, 2 workers | 4 | 2,067 | 3,447 | 11,475 | 1.32 | 3.47 |
| 8-bit | 4 | Statim q8_0, 4 workers | 4 | 1,938 | 3,368 | 20,533 | 1.14 | 2.96 |
| 8-bit | 4 | ORT blockwise | 4 | 2,377 | 3,637 | 22,276 | 1.21 | 3.92 |
| 8-bit | 8 | Statim q8_0, 1 worker | 1 | 323 | 554 | 3,509 | 2.28 | 6.69 |
| 8-bit | 8 | ORT blockwise | 1 | 329 | 592 | 7,361 | 1.73 | 7.50 |
| 8-bit | 8 | Statim q8_0, 1 worker | 4 | 1,329 | 4,505 | 4,621 | 2.28 | 6.71 |
| 8-bit | 8 | Statim q8_0, 2 workers | 4 | 1,195 | 2,184 | 6,734 | 2.26 | 6.54 |
| 8-bit | 8 | Statim q8_0, 4 workers | 4 | 1,124 | 1,930 | 11,791 | 2.02 | 5.80 |
| 8-bit | 8 | ORT blockwise | 4 | 1,483 | 2,600 | 16,142 | 1.74 | 7.77 |

- **Throughput.** With its default single worker, Statim's throughput is higher than ORT's in all
  eight pinned one-worker comparisons: by 11 % (8-bit, 4 cores, 1 client) to 32 % (8-bit, 8 cores,
  1 client). Most of the difference comes from the long state.
- **Median with one client.**
  - f32: Statim is faster on 4 cores (409 against 416 ms), ORT on 8 cores (267 against 283 ms).
  - 8-bit: ORT is 18 ms faster on 4 cores, Statim 6 ms faster on 8 cores.
- **The slowest request with one client.** This is the 770-token state: ORT takes 7,236 ms and
  Statim 3,173 ms (f32, 8 cores).
- **Four clients.**
  - With its default single worker, Statim serializes requests. Its throughput does not rise from
    1 to 4 clients, and short requests wait behind the long one.
  - With `--workers 2`, Statim's p95 is below ORT's in every four-client comparison. For f32 on 8
    cores: 1,735 against 2,045 ms, at 33 % higher throughput.
  - More workers lower the median further, but cost throughput.
  - With four clients, p99 includes queueing, not only one long inference.
- **Sensitivity checks** (every row is in the JSON).
  - **ORT spinning.** Turning it off keeps throughput within 6 %, and p95 within 12 %, except in
    the one-client 8-core rows, where p95 is 14–27 % slower.
  - **Micro-batching.** Statim's micro-batching (`--batch-window-ms 2`) was slower than one worker
    without it in every measured row, as the README says for the CPU.
- **Why the servers are pinned.** In an earlier unpinned run with Statim 0.9.1
  (`bench/results/ort-cpu/http-unpinned.json`), ORT at `--threads 4` with 4 clients reached a p95
  of 1,881 ms and 2.06 req/s. Pinned to 4 cores, it drops to about 3,000 ms and 1.45 req/s.
  - The likely cause: each concurrent ORT `Run` call adds its calling thread to the intra-op pool,
    so ORT used more cores than its thread setting. This is inferred from the collapse under
    pinning; that run did not record CPU time.
  - Statim divides `--threads` among its workers by integer division, which leaves no remainder in
    the measured rows.

## Start, memory and size

| Variant | Cores | Server | First answer after start | RSS high-water in the start-up window |
|---|---:|---|---:|---:|
| f32 | 4 | Statim | 557 ms | 633 MiB |
| f32 | 4 | ORT | 2,260 ms | 724 MiB |
| f32 | 8 | Statim | 447 ms | 637 MiB |
| f32 | 8 | ORT | 2,060 ms | 729 MiB |
| 8-bit | 4 | Statim q8_0 | 685 ms | 276 MiB |
| 8-bit | 4 | ORT blockwise | 2,155 ms | 581 MiB |
| 8-bit | 8 | Statim q8_0 | 431 ms | 276 MiB |
| 8-bit | 8 | ORT blockwise | 1,804 ms | 579 MiB |

- **First answer** runs from process start to the first complete response.
  - For ORT it includes starting Python and importing onnxruntime.
  - It also includes one warm-up decision, which the ORT server runs before it listens.
- **Memory.** The resident set of the server process is sampled every 10 ms from start through the
  four warm-up requests. Those are short states, with one worker.
  - The 770-token request and the multi-worker servers are not in this window.
  - ORT's figure includes the Python interpreter's heap.

**Model files.** Each ONNX size includes the tokenizer and configuration files the ORT server
loads; a GGUF file already contains them.

| Model files | Statim (GGUF) | ORT (ONNX) |
|---|---:|---:|
| f32 | 866 MiB | 1,263 MiB |
| 8-bit (q8_0 against blockwise) | 340 MiB | 924 MiB |
| ORT dynamic int8 | — | 343 MiB |

- The GGUF f32 file keeps the token embedding in f16; the ONNX f32 file keeps it in f32.
- The ONNX 8-bit blockwise file keeps the embedding in f32, because ORT 1.30 quantizes embeddings
  only at 4 bits.

**Install.**

- Statim's release is one 5.2 MiB executable, in a 2.8 MiB archive.
- The ORT server's Python packages take 186 MiB: onnxruntime, tokenizers, numpy and their
  dependencies.
- That excludes the CPython interpreter and standard library, which take 58 MiB.

## What this does not show

- **One machine.** Zen 3 has AVX2 but no VNNI.
  - ARM (dotprod, i8mm) and AVX-512-VNNI, the intended int8 targets, were not measured. Statim's
    new f32 GEMM is x86-only; other CPUs keep ggml's.
  - Nor were Intel CPUs, where ORT also offers the OpenVINO execution provider.
- **No GPU.** ORT's CUDA and TensorRT execution providers were not compared with Statim's CUDA and
  Vulkan backends.
- **No native ORT server.** The HTTP rows are a product comparison; the raw and length-sweep rows
  are the runtime comparison. A C++ ORT server was not built. Nor was a serving stack with dynamic
  batching, such as Triton.
- **Sample sizes.**
  - Raw scoring stores only the median of 3 passes per state, so gaps of a few percent are ties.
  - HTTP rows use 60 requests.
  - Accuracy covers two suites, with 150 items per language.
- **One model.** Other architectures may behave differently.
- **Discarded runs.** An earlier pinned HTTP run stopped on a harness bug in the CPU placement
  after its first block, and was discarded. The runs above were repeated in full.

## Reproduce

**Prerequisites.**

- Statim built with tests and the release flags (`docs/BUILD.md`). The default binary paths are
  `build-rel/statim`, `build-rel/test_model_parity` and `build-rel/statim-quantize`.
- The shipped checkpoint in `models/laya-multilingual-v9`, and its GGUF
  `models/laya-multilingual-v9-f32.gguf`.
- The original checkpoint `models/laya-multilingual` (`tools/fetch_models.sh multilingual`), for
  the independent graph proof.
- The training environment `.venv-train` from REPRODUCE.md, for the export's PyTorch reference.

`--repeats 3` must be passed, because the flag defaults to 5. The HTTP rows above ran
`build-rel/statim`, passed with `--release-statim`.

```bash
python3 -m venv build-ort/venv
build-ort/venv/bin/pip install onnx==1.23.0 onnxruntime==1.30.0 onnxscript==0.7.2 tokenizers==0.23.2 numpy==2.5.3 psutil==7.2.2
python3 -m venv build-ort/venv-server
build-ort/venv-server/bin/pip install onnxruntime==1.30.0 tokenizers==0.23.2 numpy==2.5.3

# Export, quantize, and run every parity gate (PYTHONPATH names this machine's Python 3.12 venv)
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
  --statim-workers 2 4 --release-statim build-rel/statim --out build-ort/cpu-http-pinned-results.json

# Tables from any result file; per-operation profiles of both engines
build-ort/venv/bin/python bench/ort_compare.py --render-md build-ort/cpu-raw-results.json
CUDA_VISIBLE_DEVICES="" build-ort/venv/bin/python bench/profile_short.py --lengths 128 256 --threads 1 4
```

**Held-out accuracy.** Start one server per variant, each on its own port.

- **Statim:** `statim serve --device cpu --threads 8 --port 8201 -m multilingual=FILE`. FILE is
  `models/laya-multilingual-v9-f32.gguf` or `build-ort/laya-multilingual-v9-q8_0.gguf`.
- **ORT:** `build-ort/venv-server/bin/python bench/ort_compare.py --serve-ort --threads 8 --port
  8203 --onnx FILE`. FILE is `build-ort/laya-multilingual-v9-f32.onnx`,
  `build-ort/laya-multilingual-v9-8bit-blockwise.onnx` or
  `build-ort/laya-multilingual-v9-int8-dynamic.onnx`.

Then evaluate each server and compare its predictions with Statim f32:

```bash
.venv-train/bin/python bench/eval_categories.py --url http://127.0.0.1:8201 --model multilingual \
  --suites pii emotion --n 150 --seed 20260927 \
  --out build-ort/accuracy-statim-f32.jsonl --predictions build-ort/predictions-statim-f32.jsonl
python3 bench/compare_predictions.py build-ort/predictions-statim-f32.jsonl build-ort/predictions-ort-f32.jsonl
```

**GPU providers.** They run through the same harness, from an environment with a GPU build of ORT:
`bench/ort_compare.py --ort-only --ort-provider cuda` (or `tensorrt`), with `CUDA_VISIBLE_DEVICES`
set.
