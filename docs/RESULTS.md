# Research results

This page contains the method, charts, and experiments behind the summary in the
[README](../README.md#results).

### Public benchmark: S1Bench

On S1Bench (13 public subsets, 3,880 items, run with Lev's `levbench` harness under a protocol fixed
before the run), Statim Decide Multilingual 0.7.0 scores **0.638** macro: above its untuned Laya base
(0.579) and below Lev (0.689) and Jev (0.761). Six subsets come from sources Statim trained on (their
train splits: MASSIVE twice, MultiNLI, Aegis 2.0, Civil Comments, HelpSteer2); over the seven others it
scores 0.565 (Lev 0.635, Jev 0.766). It is behind on reading comprehension and paraphrase. Every number, the per-item records and the reproduction
commands: [s1bench-2026-10-03.md](reproductions/s1bench-2026-10-03.md).

### Published model gates

A new model replaces the one it was trained from only through the promotion gate
(`tools/finetune/gate.py`). It fails closed: evaluation is strict by default, and a missing or failed
suite, different suite set or item pool, or different registry/suite-definition SHA-256 produces
`BLOCKED`. `--no-strict` is report-only and can never promote; old artifacts without these hashes
and completeness records are also blocked.

Both models answer the same items. Two regression tests block promotion, and either is enough. Each
cell gets a paired exact McNemar test, Holm-corrected across cells, so one collapsed language cannot
hide behind gains in the others. The gate also pools paired outcomes by capability (reading, NLI,
paraphrase/similarity, intent, sentiment, emotion, safety, PII, fact checking, topic, stance,
formality, urgency, complaint, multilingual intent, and stable new families). Exact McNemar tests
are Holm-corrected across capabilities. Either a significant loss or a drop over the configurable
per-capability tolerance (2.0 points by default) blocks promotion. The report gives each
capability's item count, discordant pairs and minimal detectable drop at the Holm alpha; fewer than
600 items is labelled `UNDERPOWERED` without blocking by itself. Validation passes only when the
lower bound of a paired 95% interval for the mean of suite deltas is at least -1 point; a suite
without any differing item still contributes its zero-event bound, so a small validation set cannot
claim certainty. The family screen of earlier versions still blocks a family mean that falls more
than two standard errors. Category cells that the pool cannot fill are recorded as skipped, must be
the same for both models, and are listed in the report. At least one family must still improve
significantly after Holm correction.

The first four suites use 2,000 deterministic test rows; MASSIVE and HWU64 cells use 150 seeded
stratified rows. See [REPRODUCE.md](../REPRODUCE.md#3-a-published-models-evaluation).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/diagrams/results-0.5.0-dark.svg">
  <img alt="Statim 0.5.0 English model vs. the English base checkpoint: Banking77 0.550 to 0.928, MASSIVE English 0.533 to 0.867, typed decisions 0.361 to 0.768, HWU64 0.607 to 0.833; zero-shot suites within noise." src="../assets/diagrams/results-0.5.0-light.svg" width="100%">
</picture>

```bash
.venv-train/bin/python tools/finetune/train_multitask.py models/laya models/laya-english-big1 --clean \
    --mixture data/mixture-v5.jsonl.gz --massive-langs en --massive-per-lang 11000 --max-len 1024 \
    --epochs 12 --patience 3 --distill 12000 --budget banking77=12000,massive=8000,mixture=20000,typed=4000,distill=6000 \
    --warmup 0.06 --ema 0 --optim adamw8bit --max-tokens 3072 --accum 6
.venv-train/bin/python tools/finetune/gate.py eval models/laya-english-big1
.venv-train/bin/python tools/finetune/gate.py compare models/laya models/laya-english-big1
```

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/diagrams/results-0.4.0-dark.svg">
  <img alt="Statim 0.4.0 vs. the base checkpoint: MASSIVE 0.340 to 0.772, Banking77 0.517 to 0.903, typed decisions 0.351 to 0.758, zero-shot suites within noise." src="../assets/diagrams/results-0.4.0-light.svg" width="100%">
</picture>

```bash
.venv-train/bin/python tools/finetune/build_mixture.py --out data/mixture-v4.jsonl.gz --per-source 2000 --audit tools/finetune/licence_audit.json
.venv-train/bin/python tools/finetune/build_extra.py --out data/extra-v1.jsonl.gz   # then merge v4 + extra into data/mixture-v5.jsonl.gz
.venv-train/bin/python tools/finetune/train_multitask.py models/laya-multilingual models/laya-multilingual-big1 --clean \
    --mixture data/mixture-v5.jsonl.gz --massive-per-lang 2000 --epochs 20 --patience 3 \
    --budget banking77=12000,massive=16000,mixture=20000,typed=4000,distill=3000 --warmup 0.06 --ema 0.999
.venv-train/bin/python tools/finetune/gate.py eval models/laya-multilingual-big1
.venv-train/bin/python tools/finetune/gate.py compare models/laya-multilingual-clean models/laya-multilingual-big1
```

### Base-checkpoint consensus

This separate experiment uses the first 400 test rows, identical prompts, CPU, and fp32. It uses
the original Laya checkpoints, not the fine-tuned Statim Decide models. Reproduce it with
`bench/eval_accuracy.py`.

| Suite, 400 cases | Jev, published¹ | Laya English² | Laya multilingual² | Consensus |
|---|---:|---:|---:|---:|
| AG News, 4 labels | 0.910 | 0.950 | 0.935 | **0.950** |
| DAIR Emotion, 6 labels | 0.480 | 0.5925 | 0.5375 | **0.600** |
| Banking77, all 77 labels | 0.870, 72 labels | 0.425 | 0.470 | **0.4875** |

| Emotion calibration | NLL | ECE | Brier |
|---|---:|---:|---:|
| Laya English | 2.019 | 0.306 | 0.696 |
| Consensus | **1.865** | **0.286** | **0.686** |

¹ Third-party published values quoted by Laya use different samples and prompts. ² Measured through
Statim exact mode; Laya reports 0.953 / 0.600 for English in its own run. Consensus averages option
log-probabilities and costs 1.25–1.4× the English checkpoint alone.

### Consensus, calibration, and quantization

- Option-order ensembling changed multilingual Emotion from 0.5375 to 0.525. It remains available
  as `ensemble: K` for choice questions; ordinal score levels are never rotated.
- Contextual calibration added 2.0 points on multilingual Emotion, was neutral to slightly negative
  elsewhere, and improved Banking77 ECE. Enable it with `calibrate: true`.
- q4_0 and q4_K changed 1–2 of 16 parity answers. f32 is the reference; q8_0 halves memory with
  smaller logit drift.

### Many-option tasks and Banking77 fine-tuning

When options exceed `head_max_len` (192 English, 256 multilingual), Laya cuts each option to
`(head_max_len - 16) / k` tokens. With 77 intents, that is one or two subwords per intent. Setting
`"head_max_len": 512` or `--head-max-len 512` needs no training and preserves at least 128 state
tokens.

The recipe trains on Banking77 train with shuffled option order, replays typed-decisions train,
freezes token embeddings, selects the best epoch on held-out development data, and refits
temperatures. Distillation adds generic tweets and news labeled by the base model's distributions.
Five epochs take 35 minutes on an RTX 3070.

```bash
python -m venv .venv-train && .venv-train/bin/pip install torch laya==0.3.20 datasets
python -m venv .venv && .venv/bin/pip install numpy safetensors gguf    # the converter's environment
.venv-train/bin/python tools/finetune/train_banking77.py models/laya-multilingual models/laya-multilingual-banking77 --distill 6000 --epochs 5
.venv/bin/python tools/convert_laya.py models/laya-multilingual-banking77 -o models/laya-multilingual-banking77-f32.gguf --type f32 --embd-type f16
.venv-train/bin/python tools/finetune/eval_laya.py models/laya-multilingual-banking77 --n 2000 --head-max-len 512
```

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/diagrams/finetune-dark.svg">
  <img alt="Banking77 accuracy rises from 0.4885 to 0.8655 while held-out AG News and Emotion stay flat; calibration error falls from 0.372 to 0.043." src="../assets/diagrams/finetune-light.svg" width="100%">
</picture>

Protocol: first 2,000 test rows per suite, never trained on, with ±1.1 points standard error around
0.5, plus 2,000 typed-decisions test decisions.

| | Base | + budget 512 | 3 epochs | + distillation, 5 epochs |
|---|---:|---:|---:|---:|
| Banking77 accuracy | 0.4885 | 0.5175 | 0.8435 | **0.8655** |
| Banking77 ECE | 0.372 | 0.352 | 0.052 | **0.043** |
| AG News, held out | 0.938 | 0.938 | 0.941 | **0.9385** |
| Emotion, held out | 0.532 | 0.532 | 0.502 | **0.528** |
| Emotion ECE | 0.336 | 0.336 | 0.210 | **0.155** |
| typed-decisions test | 0.351 | 0.351 | 0.6665¹ | **0.7015¹** |

¹ In-domain because its train split is replay data. AG News and Emotion were never trained on.
Without distillation, Emotion lost 3 points; with it, each held-out suite stays within noise while
Banking77 gains 35 points. Statim and Laya score 0.8675 versus 0.870 on the first 400 Banking77 rows,
bf16 versus f32. The weights are not committed; the script reproduces them.
