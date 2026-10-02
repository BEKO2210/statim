# Paired gate recomputation (2026-10-01)

The published model and adapter decisions were first made with unpaired standard errors. Both
models answer the same items, so their outcomes are correlated, and an unpaired test misjudges the
noise. The gate now uses paired tests: an exact McNemar test per cell and the standard error of the
per-item differences for pools. Gains and drops are both Holm-corrected. This record documents the
recomputation of every published decision with the paired gate (READINESS P0 #9).

## Setup

- Code: the `eval/paired-gate` branch, with `tools/finetune/gate.py` writing `eval-items.jsonl.gz`
  next to `eval.json`.
- Engine: Statim built from `main` with Vulkan, on an RTX 3070. The f32 GPU answers equal the CPU
  answers.
  - Environment: `STATIM_BIN=…/build-vk/statim STATIM_GATE_DEVICE=vulkan`.
- Each model was evaluated once. `--mixture` is the mixture its comparison partner was trained on,
  so both sides draw the decision-category cells from the same pool.

| Model directory | Published as | `--mixture` | Held-out cells | `eval_items_sha256` |
|---|---|---|---:|---|
| `models/laya` | base of English 0.5.0 | `mixture-v5` | 91 | `a27533bd1bd842e01b44a2aa001a74c3e0a1614bfb4627cb1bf9007bcc078ca7` |
| `models/laya-english-big1` | statim-decide-en-large 0.5.0 | `mixture-v5` | 91 | `b4ad041debd6c968615e6c737a1f687e6cf4e8b0bd06cc4f5410340e7201f9ee` |
| `models/laya-multilingual` | base of multilingual | `mixture-v8` | 91 | `244524727101909483a2f09a90e88c0ee273dc4a7c60a18052795d4bc454597d` |
| `models/laya-multilingual-big1` | statim-decide-multilingual-base 0.4.0 | `mixture-v8` | 91 | `88c51bfe657fc2d45e3c3b18a159bd14753f2e4b79b6348287b835af68b99bae` |
| `models/laya-multilingual-v9` | statim-decide-multilingual-base 0.7.0 | `mixture-v8` | 91 | `313dcbebf4c94ef31198fb1b04697331d10ee6e03c13acbbc2d763d8555dc2ca` |

The earlier `eval.json` files are kept beside the new ones as `eval.json.pre-paired-2026-10-01`.

## Model gates

```sh
.venv-train/bin/python tools/finetune/gate.py compare models/laya models/laya-english-big1
.venv-train/bin/python tools/finetune/gate.py compare models/laya-multilingual-big1 models/laya-multilingual-v9
.venv-train/bin/python tools/finetune/gate.py compare models/laya-multilingual models/laya-multilingual-v9
```

| Comparison | Validation mean (paired 95 % CI of the change) | Gains | Within noise | Regressions | Verdict |
|---|---|---:|---:|---:|---|
| English 0.5.0 against `laya` | 0.5595 → 0.6534 (+0.0787 to +0.1091) | 10 | 81 | 0 | PROMOTE |
| Multilingual 0.7.0 against 0.4.0 | 0.7404 → 0.7340 (−0.0173 to +0.0045) | 21 | 70 | 0 | PROMOTE |
| Multilingual 0.7.0 against `laya-multilingual` | 0.5511 → 0.7340 (+0.1664 to +0.1994) | 37 | 54 | 0 | PROMOTE |

Before the recomputation, the published numbers were:

- **English:** 54 suites, 11 gains, 0 regressions.
- **Multilingual 0.7.0 against 0.4.0:** 89 suites, 23 gains, 66 within noise, 0 regressions.

The cell count grew to 91 because the decision-category suites are now evaluated for every model.

The paired gate confirms **0 regressions** for both published models. It counts slightly fewer
gains than before because gains are now Holm-corrected as well.

The smallest adjusted drop p-value among the 0.7.0 cells against 0.4.0 is Belebele German, at
0.0855. It is not significant, and Belebele is near chance for both models.

## Adapter decisions

The published adapter files were reused unchanged (`--skip-train`). Each conversion is
SHA-256-identical to the published file: PII `2991a33b…`, emotion `c832dc6a…`, safety `9620969c…`.

```sh
.venv/bin/python tools/finetune/lora_experiment.py --base-checkpoint models/laya-multilingual-v9 \
  --base-gguf models/laya-multilingual-v9-f32.gguf --mixture data/mixture-v8.jsonl.gz \
  --statim build-vk/statim --device cuda --server-device vulkan --categories pii emotion safety \
  --skip-train --work models/lora-paired-2026-10-01
# the safety replication: add --n 1500 --eval-skip 150 --seed 20260927, --categories safety
```

| Adapter | Items | Pooled change | Unpaired 2 SE (published) | Paired 2 SE | Paired decision |
|---|---|---:|---:|---:|---|
| PII | 11 × 150 | +5.45 | 2.22 | 1.40 | PROMOTE |
| emotion | 8 × 150 | +4.75 | 3.88 | 2.12 | PROMOTE |
| safety, first run | 150 | +8.00 | 9.72 | 7.45 | PROMOTE (Holm p = 0.032) |
| safety, replication | 1,350 fresh | +9.78 | 3.28 | 2.39 | PROMOTE (Holm p = 2.5e-16) |

No adapter cell regresses. All three published adapters keep their decision.

Under the unpaired rule, safety needed a replication to be promoted. The paired rule detects the
gain on the first 150 items already.
