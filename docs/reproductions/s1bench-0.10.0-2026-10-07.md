# Statim 0.10.0 on S1Bench (2026-10-07)

The same pre-registered protocol as the 0.7.0 run ([s1bench-protocol.md](s1bench-protocol.md),
[s1bench-stage1-protocol.md](s1bench-stage1-protocol.md)), unchanged: same harness, tasks, scoring
and overlap check. Only the engine and model versions differ. Raw records are in
[`bench/results/s1bench/`](../../bench/results/s1bench/) (`s1-statim0100-f32`, `s1-statim0100-gpu-f32`,
`s1-consensus0100-f32`). Every table below is the output of
`bench/s1bench_compare.py` on them.

## Short version
- **Statim Decide Multilingual 0.10.0 scores 0.657 macro over all 13 subsets**, up from 0.638 for
  0.7.0 and 0.579 for the untuned Laya base. Lev (Qwen3.5-4B + LoRA, GPU) scores 0.689 and Jev
  0.761. **Statim is still behind both.**
- Over the 7 subsets from sources Statim never trained on: 0.584 (0.7.0: 0.565, Lev 0.635, Jev 0.766).
- No single subset changes significantly against 0.7.0 (paired exact McNemar, Holm). The gain is
  spread across many subsets, and the largest moves are summeval-consistency (+11.8 points), aegis2
  (+5.2), squad2 (+3.7) and vitaminc (+3.3). pubmedqa drops 6.0 points.
- **The consensus of en-large 0.5.0 and multilingual 0.10.0 scores 0.673**, 1.7 points below Lev
  (the 0.7.0 consensus was 3.8 below), and 0.595 over the never-trained 7.
- Calibration: mean ECE 0.126 for the single model (0.7.0: 0.138), 0.116 for the consensus
  (Lev 0.115, Jev 0.091).
- **On a GPU the same file gives the same answer to all 3,880 items** (RTX 3070, Vulkan release binary,
  exact f32 math): median server time per request 22 ms against 167 ms on 4 CPU threads (p95 50 ms
  against 568 ms).
- On the 6 subsets of the third-party board snapshot the single model scores 0.675 and the consensus
  0.695. Both rank 7 of 21, behind decider-2b (0.703).

## Setup
| | |
|---|---|
| Engine | Statim 0.10.0 release binary `statim-0.10.0-linux-x86_64-cpu` (archive SHA-256 checked against the release `SHA256SUMS`), CPU (Ryzen 7 5800X), default serve flags; 4 threads for the single model, 8 for the consensus (as in stage 1) |
| Models | statim-decide-multilingual-base 0.10.0 f32 (file SHA-256 `12da2d1f…`, equal to the Hub file); en-large 0.5.0 f32 (Hub commit `41055a7c`) for the consensus, loaded as `english`, with `--consensus` and requests sending `"model": "consensus"` |
| Harness | `levbench` (lev @ `745535b`), backend `lev`, concurrency 1, timeout 120 s; 3,880 items from `lev s1bench export`; `bench/s1bench_run.py` keeps every record |
| GPU run | `statim-0.10.0-linux-x86_64-vulkan`, `--device gpu` (Vulkan0, NVIDIA GeForce RTX 3070), 4 threads, no `--gpu-fast`; same model file and harness |
| Latency | server-side `ms` of every `/v1/systemone` request in the serve log (`bench/results/s1bench/serve-0100-{cpu,gpu}.log.gz`, local paths removed); the harness sends one request at a time |
| Overlap | 0.10.0 was trained on the same sources as 0.7.0 (mixture v8, Banking77, MASSIVE), so the 0.7.0 overlap list (`overlap-v8.json`) applies unchanged |

## Results

| Subset | 0.10.0 | consensus (0.10.0 + en-large) | 0.7.0 | consensus (0.7.0 + en-large) | Laya base | Lev | Jev |
|---|---:|---:|---:|---:|---:|---:|---:|
| vitaminc-dev | 0.756 | 0.778 | 0.723 | 0.775 | 0.765 | 0.668 | 0.801 |
| massive-en-US | 0.814 | 0.849 | 0.791 | 0.843 | 0.617 | 0.857 | 0.874 |
| massive-de-DE | 0.797 | 0.800 | 0.783 | 0.774 | 0.443 | 0.823 | 0.871 |
| boolq | 0.683 | 0.757 | 0.687 | 0.750 | 0.753 | 0.827 | 0.893 |
| squad2 | 0.592 | 0.585 | 0.555 | 0.559 | 0.595 | 0.813 | 0.836 |
| paws | 0.676 | 0.660 | 0.656 | 0.612 | 0.764 | 0.776 | 0.900 |
| multinli | 0.809 | 0.873 | 0.796 | 0.866 | 0.846 | 0.890 | 0.836 |
| civil_comments | 0.920 | 0.933 | 0.930 | 0.927 | 0.920 | 0.760 | 0.803 |
| aegis2 | 0.752 | 0.724 | 0.700 | 0.588 | 0.604 | 0.800 | 0.804 |
| helpsteer2 | 0.365 | 0.402 | 0.333 | 0.394 | 0.277 | 0.386 | 0.341 |
| summeval-relevance | 0.271 | 0.287 | 0.287 | 0.292 | 0.125 | 0.358 | 0.358 |
| summeval-consistency | 0.660 | 0.611 | 0.542 | 0.549 | 0.292 | 0.271 | 0.812 |
| pubmedqa | 0.448 | 0.488 | 0.508 | 0.540 | 0.520 | 0.732 | 0.764 |
| **macro, all 13** | **0.657** | **0.673** | 0.638 | 0.651 | 0.579 | 0.689 | 0.761 |
| **macro, 7 never-trained sources** | **0.584** | **0.595** | 0.565 | 0.582 | 0.545 | 0.635 | 0.766 |
| **macro, 6 board subsets** | **0.675** | **0.695** | 0.648 | 0.660 | 0.630 | 0.719 | 0.769 |
| mean ECE | 0.126 | 0.116 | 0.138 | 0.105 | 0.179 | 0.115 | 0.091 |

Removing the 16 overlap items changes no macro by more than 0.003.

Paired against 0.7.0, item by item (exact McNemar, Holm over 13 subsets):

| Subset | 0.10.0 right, 0.7.0 wrong | 0.10.0 wrong, 0.7.0 right | Holm p |
|---|---:|---:|---:|
| vitaminc-dev | 51 | 31 | 0.416 |
| massive-en-US | 14 | 6 | 1 |
| massive-de-DE | 13 | 8 | 1 |
| boolq | 17 | 18 | 1 |
| squad2 | 17 | 6 | 0.416 |
| paws | 22 | 17 | 1 |
| multinli | 18 | 14 | 1 |
| civil_comments | 9 | 12 | 1 |
| aegis2 | 41 | 28 | 1 |
| helpsteer2 | 30 | 22 | 1 |
| summeval-relevance | 33 | 37 | 1 |
| summeval-consistency | 29 | 12 | 0.15 |
| pubmedqa | 21 | 36 | 0.627 |

## CPU and GPU

| | CPU (Ryzen 7 5800X, 4 threads) | GPU (RTX 3070, Vulkan) |
|---|---:|---:|
| macro, all 13 | 0.657 | 0.657 |
| answers that differ from the CPU run | — | 0 of 3,880 |
| median ms per request | 167 | 22 |
| p95 ms per request | 568 | 50 |

The largest difference in a top probability between the two runs is 0.0001, the rounding step of the
records. Latency is the server's own time per request on an otherwise idle machine, not end-to-end
client time. `bench/results/s1bench/summary-0.10.0.json` holds these numbers and the macros above.

## What it shows
- The A100 retrain that lifted the 14 held-out decision categories (0.748 to 0.826) also lifts the
  public suite, but by less: 1.9 points macro, none of it significant per subset. Reading
  comprehension (boolq, squad2, pubmedqa) stays far below Lev and Jev; it is the gap the next mixture
  has to close.
- The two checkpoints remain complementary: the consensus gains 7.4 points on boolq and 6.4 on
  MultiNLI over the single model, and loses on aegis2 and summeval-consistency.

## Reproduce
```sh
# lev @ 745535b, tasks exported as in s1bench-2026-10-03.md
statim serve -m multilingual=statim-decide-multilingual-base-f32.gguf --threads 4 --port 8080 &
uv run python <statim>/bench/s1bench_run.py --tasks data/s1bench --base-url http://127.0.0.1:8080 --out s1-statim0100-f32.json
statim serve -m english=en-large-f32.gguf -m multilingual=statim-decide-multilingual-base-f32.gguf --consensus --threads 8 --port 8081 &
uv run python <statim>/bench/s1bench_run.py --tasks data/s1bench --base-url http://127.0.0.1:8081 --model consensus --out s1-cons0100.json
statim serve -m multilingual=statim-decide-multilingual-base-f32.gguf --device gpu --threads 4 --port 8082 > gpu.log &
uv run python <statim>/bench/s1bench_run.py --tasks data/s1bench --base-url http://127.0.0.1:8082 --out s1-statim0100-gpu-f32.json
B=<statim>/bench/results/s1bench
python3 <statim>/bench/s1bench_compare.py $B/s1-statim0100-f32.json.gz $B/s1-statim070-f32.json.gz \
    --overlap $B/overlap-v8.json --same-as $B/s1-statim0100-gpu-f32.json.gz \
    --latency "CPU (Ryzen 7 5800X, 4 threads)=$B/serve-0100-cpu.log.gz" \
    --latency "GPU (RTX 3070, Vulkan)=$B/serve-0100-gpu.log.gz" --json summary-0.10.0.json
```
