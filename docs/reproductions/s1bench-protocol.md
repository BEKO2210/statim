# Statim on S1Bench: protocol, fixed before running

This protocol is committed before any S1Bench item is sent to Statim. The results are published
next to it whatever they show, including subsets where Statim loses.

## What S1Bench is
Thirteen public subsets, 3,880 items, pinned by the manifests of
[bespokelabsai/nimble](https://github.com/bespokelabsai/nimble) (`docs/PUBLIC_BENCHMARKS.md`). Each
item asks one typed question (`choice`, `score` or `noul`) about a state, over the Jev-compatible
`POST /v1/systemone` API that Statim also serves. The executable harness is `levbench` from
[Abhinavexists/lev](https://github.com/Abhinavexists/lev) (the tool Lev's authors used for their
Lev and Jev numbers). There is no permanent public leaderboard; a snapshot of a third-party board is
in `lev/data/s1bench-snapshot.json`.

## Fixed setup
| | |
|---|---|
| Harness | `lev` repository at commit `745535b`; tasks exported with `lev s1bench export` (3,880 items, ids and label counts checked by the exporter); `levbench eval --backend lev --base-url <statim> --concurrency 1 --timeout 120`, plus a wrapper around `levbench.runner.run_eval` that also writes every item's record, so results can be paired |
| Engine | Statim 0.9.4 release binary (`statim-0.9.4-linux-x86_64-cpu`), CPU, default serve flags except `--threads`; no `calibrate`, no `ensemble`, no prompt changes |
| Models | (1) statim-decide-multilingual-base **0.7.0**, f32 GGUF from the Hub; (2) the original **laya-multilingual** checkpoint converted to f32 GGUF (`tools/fetch_models.sh multilingual`), as the untuned baseline on the same engine |
| Scoring | levbench's own: accuracy per subset (choice: returned `choice`; score: argmax of `probabilities`; noul: `noul >= 0.5`), macro = unweighted mean over subsets, ECE with 10 equal-width bins on the top probability |

## Contamination, decided in advance
- Statim Decide 0.7.0 was trained on the **train** splits of four S1Bench sources: MASSIVE
  (massive-en-US, massive-de-DE), MultiNLI (multinli), and Aegis 2.0 (aegis2; Aegis 1.0 too). S1Bench
  uses their test or dev splits. Those four subsets are reported as **in-domain**, not zero-shot.
  Lev excluded all 13 sources from its training, so on these four the comparison favours Statim.
- Exact-overlap check: every string in each item's state (at least 30 characters, lower-cased,
  whitespace collapsed) is looked up in the states of the 0.7.0 training mixture
  (`data/mixture-v8.jsonl.gz`) and the MASSIVE train split. Items with a hit are listed; every macro
  is reported with and without them.
- Reported macros: all 13 subsets; the **9 subsets whose sources Statim never trained on**
  (vitaminc-dev, boolq, squad2, paws, civil_comments, helpsteer2, summeval-relevance,
  summeval-consistency, pubmedqa); and the 6 subsets of the third-party board snapshot.

## Comparison and its limits
- Lev and Jev numbers are taken from Lev's `docs/FINDINGS.md` (levbench, same tasks); they are not
  re-measured here. Jev's "published" values in the S1Bench files equal Nimble's own measurement of
  Jev; no TypeSafe publication of them was found, and the report says so.
- Paired statistics (exact McNemar with Holm correction) only for Statim 0.7.0 against the
  untuned Laya checkpoint, which both run here item by item. Against Lev and Jev only aggregate
  numbers exist; per-subset differences under about 5-9 points are within noise at these sizes.
- Latency is not compared (different hardware: our CPU vs their H100 and a hosted API).
