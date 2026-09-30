# Statim and ONNX Runtime on CPU

A common question about Statim: why not export the model to ONNX and serve it with ONNX Runtime
(ORT)? This page answers it for the shipped multilingual model on one CPU, with the same inputs for
both engines, and with parity gates run before any timing. The target is for Statim to beat or
match general-purpose runtimes on its own decision workload without sacrificing fidelity. Where ORT
is faster, this page says so.

## In short

The measurements are from an AMD Ryzen 7 5800X with 8 cores. The sections below give the details
and the source file of every number.

- **Same answers.**
  - On the 240 golden items, ORT f32 and Statim f32 both reproduce the PyTorch reference within
    1.4e-5.
  - On sequences of up to 1,024 tokens they agree with each other within 1.4e-5.
  - On a 2,850-item held-out sample they make the same decision on every item.
- **Raw scoring depends on sequence length and threads (f32).**
  - At 128 tokens per sequence, Statim takes 12 % longer than ORT on 1 thread and 6 % longer on 4
    threads.
  - In this sweep, Statim is faster at 256 tokens and more on 4 and 8 threads. At 1,024 tokens on 8
    threads, ORT takes 2.8× as long.
- **Serving.** This compares Statim's server with a small Python server around ORT, both pinned to
  the same cores. It is a product comparison, not a runtime comparison.
  - With its default single worker, Statim answers 10–34 % more requests per second in every
    configuration measured.
  - With one client, ORT's median latency is lower for f32 (257 against 286 ms on 8 cores). It is
    higher for 8-bit on 8 cores (319 against 310 ms).
  - With four clients, Statim needs `--workers 2` to match or beat ORT's p95 latency. Its default
    single worker queues short requests behind long ones.
- **8-bit.**
  - **Statim's q8_0** changes 30 of the 2,850 held-out decisions. Accuracy goes from 0.7425 to
    0.7439, which is not distinguishable from no change (exact McNemar p = 0.523).
  - **ORT's per-channel dynamic int8** is the fastest variant on the short golden states. It
    changes 443 decisions and costs 1.7 accuracy points (p = 0.00883).
  - **ORT's blockwise 8-bit weights** (f32 compute) keep the accuracy, and are slower than q8_0 at 8
    threads and above.
- **Start and footprint.**

  | Measure | Statim | ORT |
  |---|---:|---:|
  | First answer after process start, f32, 8 cores | 0.45 s | 1.95 s (includes one warm-up decision) |
  | Resident memory in the start-up window, f32 | 628 MiB | 729 MiB |
  | Resident memory in the start-up window, 8-bit | 281 MiB | 567–582 MiB |
  | Install | 5.2 MiB executable | 186 MiB of Python packages, without the interpreter |

## Parity

`bench/ort_export.py` exports the complete scoring graph to ONNX (opset 18): the encoder, the typed
decision transformer, the option scorer and the action scorer. Before anything is timed, the export
is gated on the logits (`bench/results/ort-cpu/export-report.json`).

| Comparison against the f32 PyTorch reference, 240 items | Argmax | max \|Δlogit\| |
|---|---:|---:|
| Original Laya checkpoint, ORT f32 (independent golden file) | 240/240 | 9.4e-5 |
| Shipped model (0.7.0 checkpoint), ORT f32 | 240/240 | 1.0e-5 |
| Shipped model, Statim f32 | 240/240 | 1.4e-5 |
| Shipped model, Statim q8_0 | 235/240 | 0.31 |
| Shipped model, ORT 8-bit blockwise (MatMulNBits, block 32, f32 compute) | 236/240 | 0.19 |
| Shipped model, ORT dynamic int8, per channel, `reduce_range` | 195/240 | 4.31 |
| Shipped model, ORT dynamic int8, per tensor (`quantize_dynamic` defaults) | 196/240 | 7.96 |

- **The golden file.** `tests/data/golden_laya-multilingual.jsonl` belongs to the original
  checkpoint. For the shipped checkpoint, the export script computes the PyTorch reference on the
  same 240 items. Statim's `test_model_parity` is checked against that reference too.
- **Long sequences.** The length sweep below compares Statim f32 with ORT f32 on sequences of up
  to 1,024 tokens. Every argmax matches, with max |Δlogit| 1.4e-5.
- **The ORT server.** It re-implements the request rendering and the packing in Python. It
  reproduces all 240 golden token sequences exactly. On all 30 golden states it gives Statim's
  choices, with every probability within 1e-3.

## Held-out accuracy

- **The sample.** The PII suite (11 languages) and the emotion suite (8 languages) of
  `bench/eval_categories.py`, 150 items per language (seed 20260927), 2,850 items in total.
- **The runs.** Each engine and variant answered the same sample, and wrote per-item predictions
  with `--predictions`. `bench/compare_predictions.py` then compared each run with Statim f32,
  pairwise.

| Against Statim f32 | Changed decisions | f32 right, other wrong | f32 wrong, other right | Accuracy | Exact McNemar p |
|---|---:|---:|---:|---|---:|
| ORT f32 | 0 | 0 | 0 | 0.7425 → 0.7425 | 1 |
| Statim q8_0 | 30 | 9 | 13 | 0.7425 → 0.7439 | 0.523 |
| ORT 8-bit blockwise | 17 | 7 | 7 | 0.7425 → 0.7425 | 1 |
| ORT dynamic int8, per channel, `reduce_range` | 443 | 193 | 144 | 0.7425 → 0.7253 | 0.00883 |

- **Size of the dynamic int8 drop.** The paired standard error of the 1.7-point drop is 0.64
  points.
- **By suite.** Dynamic int8 lowers PII from 0.855 to 0.838 (p = 0.018). On emotion, the point
  estimate falls from 0.588 to 0.571, but the paired test does not resolve it (p = 0.175).
- **Not tested:** the per-tensor dynamic variant was not run on the held-out sample.
- **A discarded attempt.** The first attempt at the ORT f32 run reached no server: the previous
  server still held the port. All five runs were then repeated in full, each on its own port.

## What ORT was given

A separate agent tuned the ORT side with one instruction: find ORT's fastest legitimate setup.

- **Where the results are.** Timings are in `bench/results/ort-cpu/ort-tuning-*.json`: 30 states,
  3 passes, 8 threads, on the same machine. The op counts of the graphs are in
  `ort-tuning-graph-ops.json`.
- **Graph optimization.** `ORT_ENABLE_ALL` fuses 24 MatMuls with their scaling and one
  Add+LayerNorm. It fuses no attention: this ModernBERT-style encoder uses rotary embeddings, and
  alternates global and sliding-window attention.
- **The transformer optimizer** (`onnxruntime.transformers.optimizer`, model type `bert`) also
  fuses GELU: 23 Gelu and 1 BiasGelu. It fuses no attention, SkipLayerNorm or rotary embedding.
  It was slower than the plain export, at 523 against 486 ms mean per state. So was the
  offline-optimized graph, at 535 ms.
- **Session settings.** `dynamic_block_base` 4 and denormals-as-zero changed the mean by under 2 %.
  Turning spinning off was 13 % slower.
- **The f32 setup chosen.** The plain export with `ORT_ENABLE_ALL`, denormals-as-zero, sequential
  execution and ORT's default spinning. The HTTP benchmark also runs ORT with spinning off, as a
  sensitivity check.
- **8-bit variants measured:**
  - dynamic int8, per tensor or per channel, with and without `reduce_range`;
  - MatMulNBits 8-bit blockwise weights, at accuracy level 0 (f32 compute) and at accuracy level 4
    (int8 compute, closer to how q8_0 computes).
- **The 8-bit row.** Both blockwise levels change 4 of the 240 golden argmaxes. Level 0 was faster
  (546 against 646 ms mean at 8 threads), so it is the 8-bit row.
- **The dynamic int8 row.** The per-channel `reduce_range` variant is shown alongside, as ORT's
  dynamic quantization path. On AVX2 without VNNI, ORT's documentation recommends `reduce_range`
  against saturation in its u8s8 kernels. Even so, dynamic int8 loses decisions on this model
  (195/240).
- **The token embedding.** ORT 1.30's weight-only quantizer quantizes Gather only at 4 bits. The
  8-bit blockwise file therefore keeps its 256,000 × 768 token embedding in f32.
- **Where the 8-bit files come from.** The tuning agent built the files that were timed.
  `bench/ort_export.py` now builds them with the same settings, and the weight files are
  byte-identical. The SHA-256 of the `.data` files starts with `3ae660c1…` for the blockwise
  weights and `6eb3033f…` for the dynamic ones.

## Protocol

| | |
|---|---|
| Machine | AMD Ryzen 7 5800X: 8 cores, 16 threads, AVX2 and FMA, no AVX-512, no VNNI. 16 GB RAM, Linux 6.18.7, `performance` governor, otherwise idle |
| Statim | 0.9.0. Raw scoring runs through `test_model_parity`, built with the release flags (`-DSTATIM_NATIVE=OFF`). HTTP uses the released `statim-0.9.0-linux-x86_64-cpu` binary |
| ORT | onnxruntime 1.30.0, CPU execution provider, called from Python 3.12 (tokenizers 0.23.2, numpy 2.5.3) |
| Model | The shipped statim-decide-multilingual-base checkpoint. GGUF files: f32, which keeps the token embedding in f16 as the converter writes it, and q8_0 from `statim-quantize`. ONNX files: f32 (embedding in f32), 8-bit blockwise and dynamic int8 |
| Inputs | The 30 states × 8 questions of `tests/data/golden_inputs.json`, already tokenized in `build-ort/golden_v9.jsonl`. 29 states are at most 144 tokens long; one is 770 |
| Raw scoring | One batch of 8 sequences per state, with identical token ids for both engines. One warm-up pass, then 3 passes. Only the median per state is stored |
| Length sweep | One batch of the 8 questions over one long state, cut so that every sequence is exactly the target length. One warm-up pass, then 3 passes, median |
| HTTP | Each state with its 8 questions as one `POST /v1/systemone`. 4 warm-up requests, then 60 requests per row, from 1 or 4 closed-loop clients. Each server is pinned with `taskset` to one logical CPU on each of 4 or 8 physical cores. The load generator runs on the other cores for the 4-core rows, and on the SMT siblings of the server's cores for the 8-core rows |

**Python-binding overhead.** Every process runs with `CUDA_VISIBLE_DEVICES=""`. Raw scoring calls
ORT through its Python bindings. Comparing the wall time of `session.run` with ORT's own profiled
`model_run` time puts the overhead at about 50 µs per call. The median share is at most 0.020 %,
and the largest overhead recorded is 63 µs. That is negligible for this workload. It is not a
measurement of a native C++ ORT integration (`bench/results/ort-cpu/binding-overhead.json`).

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
| 1 | 1,651 | 1,424 | 16,862 | 19,169 | 1,837 | 1,714 | 812 | 18,886 | 19,928 | 13,260 |
| 4 | 470 | 431 | 4,836 | 7,944 | 520 | 524 | 253 | 5,421 | 8,273 | 6,356 |
| 8 | 272 | 268 | 2,907 | 6,961 | 307 | 328 | 169 | 3,428 | 7,306 | 6,077 |
| 16 | 442 | 372 | 4,240 | 8,027 | 381 | 435 | 239 | 3,611 | 8,411 | 7,060 |

- **Short states, f32.** Statim takes 16 % longer than ORT at 1 thread, 9 % at 4 threads and 19 %
  at 16 threads. At 8 threads the two are within 2 %, a tie at this precision.
- **The 770-token state, f32.** ORT takes 1.14× as long as Statim at 1 thread, 1.64× at 4, 2.39× at
  8 and 1.89× at 16.
- **8-bit, short states.**
  - At 1 thread, Statim's q8_0 takes 7 % longer than ORT's blockwise 8-bit.
  - At 4 threads they tie.
  - At 8 and 16 threads, ORT's blockwise 8-bit takes 7 % and 14 % longer.
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

- **Where each engine leads.**
  - For f32, ORT leads at 128 tokens on 1 and 4 threads, and at 256 tokens on 1 thread.
  - For 8-bit, ORT leads on 1 thread up to 512 tokens.
  - Every other cell favours Statim, and the gap grows with length and threads. The sweep has no
    16-thread rows.
- **Two plausible causes, neither isolated.**
  - On the CPU, Statim computes attention with ggml's fused flash-attention kernel (`flash_attn`,
    on by default). ORT's graph for this model has no attention fusion, so it materializes the
    attention scores.
  - In the last layer of the decision head, Statim computes queries, the output projection and the
    feed-forward only for the option-marker rows (`src/model.cpp`). The exported graph computes
    them for every row.

  No ablation of either was run.
- **Padding does not explain ORT's lead on short f32 inputs.** The sweep has no padding, and ORT
  still leads at 128 tokens on 1–4 threads. Where exactly the time goes is open.

## HTTP end to end

This compares Statim's server with a Python `ThreadingHTTPServer` around ORT, which also tokenizes
and renders in Python. Both are pinned to the same physical cores
(`bench/results/ort-cpu/http-pinned.json`).

- **Busy cores** is the server's CPU time divided by wall time. The context-switch counts in the JSON
  cover only each server's main thread, so they are not compared here.
- **p99** over 60 requests is the slowest request.
- **Omitted rows.** The table leaves out Statim's 8-bit one-worker four-client rows: p95 7,078 ms
  on 4 cores and 4,323 ms on 8 cores. They are in the JSON with every other row.

| Variant | Cores | Server | Clients | p50 ms | p95 ms | p99 (max) ms | req/s | Busy cores |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| f32 | 4 | Statim, 1 worker | 1 | 460 | 774 | 4,864 | 1.60 | 3.88 |
| f32 | 4 | ORT | 1 | 420 | 740 | 8,072 | 1.46 | 3.83 |
| f32 | 4 | Statim, 1 worker | 4 | 1,825 | 6,315 | 6,529 | 1.60 | 3.89 |
| f32 | 4 | Statim, 2 workers | 4 | 1,812 | 3,068 | 10,347 | 1.47 | 3.52 |
| f32 | 4 | Statim, 4 workers | 4 | 1,829 | 3,084 | 18,973 | 1.24 | 2.98 |
| f32 | 4 | ORT | 4 | 1,884 | 3,245 | 21,392 | 1.45 | 3.93 |
| f32 | 8 | Statim, 1 worker | 1 | 286 | 481 | 3,009 | 2.58 | 7.29 |
| f32 | 8 | ORT | 1 | 257 | 471 | 7,188 | 2.00 | 7.42 |
| f32 | 8 | Statim, 1 worker | 4 | 1,137 | 3,883 | 4,000 | 2.61 | 7.41 |
| f32 | 8 | Statim, 2 workers | 4 | 1,076 | 1,816 | 5,950 | 2.54 | 6.82 |
| f32 | 8 | Statim, 4 workers | 4 | 1,045 | 1,775 | 10,751 | 2.21 | 5.98 |
| f32 | 8 | ORT | 4 | 1,186 | 2,010 | 15,084 | 2.07 | 7.74 |
| 8-bit | 4 | Statim q8_0, 1 worker | 1 | 509 | 875 | 5,471 | 1.44 | 3.85 |
| 8-bit | 4 | ORT blockwise | 1 | 503 | 877 | 8,402 | 1.28 | 3.85 |
| 8-bit | 4 | Statim q8_0, 2 workers | 4 | 1,967 | 3,376 | 11,640 | 1.34 | 3.50 |
| 8-bit | 4 | ORT blockwise | 4 | 2,359 | 3,572 | 22,835 | 1.23 | 3.94 |
| 8-bit | 8 | Statim q8_0, 1 worker | 1 | 310 | 486 | 3,375 | 2.36 | 6.82 |
| 8-bit | 8 | ORT blockwise | 1 | 319 | 576 | 7,445 | 1.76 | 7.50 |
| 8-bit | 8 | Statim q8_0, 2 workers | 4 | 1,172 | 2,018 | 6,630 | 2.32 | 6.59 |
| 8-bit | 8 | ORT blockwise | 4 | 1,556 | 2,419 | 15,989 | 1.76 | 7.79 |

- **Throughput.** With its default single worker, Statim's throughput is higher than ORT's in all
  eight pinned one-worker comparisons, by 10 % (f32, 4 cores, 1 client) to 34 % (8-bit, 8 cores,
  1 client). ORT still has the lower f32 median, so the difference comes mostly from the long
  state.
- **Median with one client.**
  - f32: ORT is faster, 420 against 460 ms on 4 cores and 257 against 286 ms on 8 cores. Most
    requests are short states.
  - 8-bit: ORT is 6 ms faster on 4 cores; Statim is 9 ms faster on 8 cores.
- **The slowest request with one client.** This is the 770-token state: ORT takes 7,188 ms, Statim
  3,009 ms (f32, 8 cores). The ratio, 2.39, is the same as in raw scoring.
- **Four clients.**
  - With its default single worker, Statim serializes requests. Its throughput does not rise from
    1 to 4 clients, and short requests wait behind the long one. p95 rises to 3,883 ms (f32, 8
    cores).
  - With `--workers 2`, Statim's p95 is below ORT's in every four-client comparison. For f32 on 8
    cores: 1,816 against 2,010 ms, at 22 % higher throughput. For f32 on 4 cores: 3,068 against
    3,245 ms, at equal throughput.
  - More workers lower the median further, but cost throughput.
  - With four clients, p99 includes queueing, not just one long inference.
- **Sensitivity checks** (every row is in the JSON).
  - **ORT spinning.** Turning it off keeps throughput within 4 %, and p95 within 8 %, except in the
    two one-client 8-core rows, where p95 is about 14 % slower.
  - **Micro-batching.** Statim's micro-batching (`--batch-window-ms 2`) was slower than one worker
    without it in every measured row. The README advises leaving it off on the CPU.
- **Why the servers are pinned.** An earlier unpinned run (`bench/results/ort-cpu/http-unpinned.json`)
  gave ORT at `--threads 4` with 4 clients a p95 of 1,881 ms and 2.06 req/s. Pinned to 4 cores, it
  drops to 3,245 ms and 1.45 req/s; Statim at the same setting barely moves.
  - The likely cause: each concurrent ORT `Run` call adds its calling thread to the intra-op pool,
    so ORT used more cores than its thread setting. This is inferred from the collapse under
    pinning; that run did not record CPU time.
  - Statim divides `--threads` among its workers by integer division, which leaves no remainder in
    the measured rows.
  - The pinned rows are the comparison.

## Start, memory and size

| Variant | Cores | Server | First answer after start | RSS high-water in the start-up window |
|---|---:|---|---:|---:|
| f32 | 4 | Statim | 654 ms | 627 MiB |
| f32 | 4 | ORT | 2,300 ms | 729 MiB |
| f32 | 8 | Statim | 449 ms | 628 MiB |
| f32 | 8 | ORT | 1,949 ms | 729 MiB |
| 8-bit | 4 | Statim q8_0 | 668 ms | 281 MiB |
| 8-bit | 4 | ORT blockwise | 2,157 ms | 567 MiB |
| 8-bit | 8 | Statim q8_0 | 445 ms | 281 MiB |
| 8-bit | 8 | ORT blockwise | 1,827 ms | 582 MiB |

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
  - ARM (dotprod, i8mm) and AVX-512-VNNI, the intended int8 targets, were not measured.
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
- **A discarded run.** An earlier pinned HTTP run stopped on a harness bug in the CPU placement
  after its first block. It was discarded, and the run above was repeated in full.

## Reproduce

**Prerequisites.**

- Statim built with tests and the release flags (`docs/BUILD.md`). The default binary paths are
  `build-rel/statim`, `build-rel/test_model_parity` and `build-rel/statim-quantize`.
- The shipped checkpoint in `models/laya-multilingual-v9`, and its GGUF
  `models/laya-multilingual-v9-f32.gguf`.
- The original checkpoint `models/laya-multilingual` (`tools/fetch_models.sh multilingual`), for
  the independent graph proof.
- The 0.9.0 CPU release, extracted under `build-ort/release-0.9.0/`.
- The training environment `.venv-train` from REPRODUCE.md, for the export's PyTorch reference.

`--repeats 3` must be passed, because the flag defaults to 5.

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
  --out build-ort/cpu-http-pinned-results.json

# Tables from any result file
build-ort/venv/bin/python bench/ort_compare.py --render-md build-ort/cpu-raw-results.json
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
