# S1Bench, stage 1: deployable configurations (2026-10-03)

Protocol, fixed before these runs: [s1bench-stage1-protocol.md](s1bench-stage1-protocol.md). Same
harness, tasks, engine (0.9.4 release binary, 8 threads) and scoring as the first run
([s1bench-2026-10-03.md](s1bench-2026-10-03.md)). All three pre-registered configurations are reported.

## Result
**B, the consensus of en-large 0.5.0 and multilingual 0.7.0, is the best deployable configuration:**
0.651 macro over all 13 subsets and 0.600 over the 9 subsets from sources Statim never trained on,
both above the single multilingual model (0.638, 0.580), which meets the headline rule. Its mean ECE,
0.105, is the lowest of the four and below Lev's 0.115. It is still 3.8 points below Lev over all 13
subsets (2.1 over the 9) and 11.0 below Jev.

To deploy it: load both checkpoints as `english` and `multilingual` and send `"model": "consensus"`
(or start the server with `--consensus` and omit `model`). Both models answer every question, so it
costs about the two models' time together.

| Subset | Statim 0.7.0 (first run) | A: en-large 0.5.0 | B: consensus | C: 0.7.0 + calibrate | Lev | Jev |
|---|---:|---:|---:|---:|---:|---:|
| vitaminc-dev | 0.723 | 0.743 | 0.775 | 0.638 | 0.668 | 0.801 |
| massive-en-US | 0.791 | 0.846 | 0.843 | 0.800 | 0.857 | 0.874 |
| massive-de-DE | 0.783 | 0.551 | 0.774 | 0.774 | 0.823 | 0.871 |
| boolq | 0.687 | 0.790 | 0.750 | 0.687 | 0.827 | 0.893 |
| squad2 | 0.555 | 0.649 | 0.559 | 0.555 | 0.813 | 0.836 |
| paws | 0.656 | 0.456 | 0.612 | 0.656 | 0.776 | 0.900 |
| multinli | 0.796 | 0.843 | 0.866 | 0.753 | 0.890 | 0.836 |
| civil_comments | 0.930 | 0.910 | 0.927 | 0.930 | 0.760 | 0.803 |
| aegis2 | 0.700 | 0.516 | 0.588 | 0.700 | 0.800 | 0.804 |
| helpsteer2 | 0.333 | 0.406 | 0.394 | 0.333 | 0.386 | 0.341 |
| summeval-relevance | 0.287 | 0.242 | 0.292 | 0.287 | 0.358 | 0.358 |
| summeval-consistency | 0.542 | 0.458 | 0.549 | 0.542 | 0.271 | 0.812 |
| pubmedqa | 0.508 | 0.492 | 0.540 | 0.336 | 0.732 | 0.764 |
| **macro, all 13** | **0.638** | **0.608** | **0.651** | **0.615** | 0.689 | 0.761 |
| **macro, 9 never-trained sources** | **0.580** | **0.572** | **0.600** | **0.552** | 0.621 | 0.723 |
| mean ECE | 0.138 | 0.139 | 0.105 | 0.141 | 0.115 | 0.091 |

## What it shows
- en-large reads better: boolq 0.790, squad2 0.649, MultiNLI 0.843, against 0.687, 0.555 and 0.796
  for the multilingual model. It is far worse on PAWS (0.456, below the 0.516 of always answering
  the majority label) and on aegis2 and German MASSIVE.
- The two models are complementary, which is why the consensus wins.
- Contextual calibration (C) lowers the macro; it costs most on pubmedqa (0.508 to 0.336).
- The post-hoc best-of-base-and-fine-tuned bound (D in the protocol, chosen with the test labels,
  never a result) is 0.662.

Records: `bench/results/s1bench/s1-{A-enlarge,B-consensus,C-calibrate}.json.gz`; pairwise tables with
`bench/s1bench_compare.py <run> bench/results/s1bench/s1-statim070-f32.json.gz --overlap bench/results/s1bench/overlap-v8.json`.
