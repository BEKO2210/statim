# How Statim compares

Statim Decide Multilingual 0.7.0 against two widely used ways to make the same decisions without
training a model: a general LLM prompted zero-shot, and a zero-shot NLI classifier. Every system
answers the **same held-out items with the same question and the same options**; accuracy is the
share of gold answers.

**In short:**
- Over 14 decision categories, Statim scores **0.748** macro accuracy. Qwen3-8B, an LLM 26 times its
  size, scores 0.706. mDeBERTa-XNLI scores 0.488.
- Statim wins 9 of the 14 categories. The LLM is ahead on emotion, fact-check, sentiment, safety and
  PII.
- On the same GPU, Statim answers about 68 decisions per second, the LLM about 6.

## Protocol

- **Items:** the promotion gate's held-out items for 0.7.0. There are 37 language cells with 150
  items each, over 14 decision categories. They come from `bench/eval_categories.py`: splits of the
  training sources that the mixture never loads, with every text that also occurs in the training
  mixture removed. Added to these are the first 2,000 test rows of AG News, DAIR Emotion and
  Banking77, with the questions and options of `bench/eval_accuracy.py`. That makes 11,550 items.
  Statim's run reproduces the gate's numbers on all 35 gate cells exactly, so the items are the
  gate's.
- **Statim Decide Multilingual 0.7.0:** 307M parameters, f32 on Vulkan. It is queried over HTTP in
  batches of 16, with `head_max_len` 512 as in the gate.
- **Qwen3-8B:** Q4_K_M through Ollama 0.32.5, zero-shot, thinking off, temperature 0. The answer is
  constrained by a JSON schema to one of the option labels, with 2 parallel requests. The prompt
  shows the text, the question and the options with their descriptions. It is in
  `bench/baselines.py` (`_prompt`).
- **mDeBERTa-v3-base-mnli-xnli:** the Hugging Face zero-shot-classification pipeline. The
  hypothesis is the question followed by "The answer is {option}."
- **Hardware:** RTX 3070 8 GB, Ryzen 7 5800X.
- **Fact-check, answered again on 2026-09-29:** in the first run, two of the three fact-check
  options were described by their bare labels (a lookup bug, fixed in #35). All three systems
  answered the 150 fact-check items again with the intended descriptions. Only Qwen3-8B changed,
  from 0.493 to 0.513. The timings are from the first run.

## What this does and does not show

- Statim was fine-tuned on the training splits of these categories, while both baselines are
  zero-shot. The comparison is between what each option gives you out of the box on your own GPU,
  not between learning methods. A fine-tuned specialist baseline (SetFit or XLM-R per category) is
  the next comparison.
- Qwen3-8B runs 4-bit quantised with a single fixed prompt. Larger or hosted LLMs would likely score
  higher, at far higher cost and latency. Hosted models were not measured.
- 150 items per cell means about ±4 points of standard error per cell. The category means over
  languages, and the mean over 14 categories, are much more stable.
- Speed is per decision. Statim is measured batched, the LLM with 2 parallel requests; both run on
  the same GPU.

## Reproduce

```bash
python3 bench/baselines.py export --mixture data/mixture-v8.jsonl.gz --out data/baselines/items.jsonl
python3 bench/baselines.py run statim --url http://127.0.0.1:8098 --items data/baselines/items.jsonl \
    --out data/baselines/statim.jsonl --name "Statim Decide 0.7.0"
python3 bench/baselines.py run ollama --model qwen3:8b --workers 2 --items data/baselines/items.jsonl \
    --out data/baselines/qwen3-8b.jsonl --name "Qwen3-8B (zero-shot)"   # resumable
python3 bench/baselines.py run nli --model MoritzLaurer/mDeBERTa-v3-base-mnli-xnli \
    --items data/baselines/items.jsonl --out data/baselines/mdeberta-xnli.jsonl
python3 bench/baselines.py report --items data/baselines/items.jsonl \
    --preds data/baselines/statim.jsonl data/baselines/qwen3-8b.jsonl data/baselines/mdeberta-xnli.jsonl \
    --out data/baselines/report.md   # the tables below
```

The export needs the training mixture the model was trained on (`data/mixture-v8.jsonl.gz`). The
remove-overlap step uses it; without it, suite items that share a text with training would stay in.

## Decision categories

| Category (macro over languages) | Statim Decide 0.7.0 | Qwen3-8B (zero-shot) | mDeBERTa-v3-base XNLI (zero-shot) |
|---|---|---|---|
| sentiment | 0.800 | **0.873** | 0.737 |
| emotion | 0.586 | **0.726** | 0.327 |
| complaint | **0.767** | 0.673 | 0.553 |
| nli | **0.747** | 0.744 | 0.376 |
| safety | 0.727 | **0.753** | 0.507 |
| reading | **0.927** | 0.880 | 0.540 |
| similarity | **0.833** | 0.727 | 0.593 |
| topic | **0.607** | 0.360 | 0.160 |
| intent | **0.753** | 0.689 | 0.356 |
| stance | **0.893** | 0.733 | 0.873 |
| formality | **0.773** | 0.497 | 0.460 |
| urgency | **0.893** | 0.833 | 0.500 |
| fact_check | 0.313 | **0.513** | 0.347 |
| pii | 0.856 | **0.878** | 0.507 |
| **mean of categories** | **0.748** | **0.706** | **0.488** |

## Every suite

| Suite | Statim Decide 0.7.0 | Qwen3-8B (zero-shot) | mDeBERTa-v3-base XNLI (zero-shot) |
|---|---|---|---|
| sentiment (en) | **0.800** | **0.800** | 0.667 |
| sentiment (zh) | 0.800 | **0.947** | 0.807 |
| emotion (de) | 0.453 | **0.587** | 0.227 |
| emotion (en) | 0.573 | **0.693** | 0.273 |
| emotion (es) | 0.573 | **0.740** | 0.360 |
| emotion (fr) | 0.660 | **0.800** | 0.353 |
| emotion (hi) | 0.713 | **0.880** | 0.440 |
| emotion (pt) | 0.507 | **0.587** | 0.307 |
| emotion (ru) | 0.693 | **0.887** | 0.520 |
| emotion (zh) | 0.540 | **0.653** | 0.307 |
| complaint (en) | **0.767** | 0.673 | 0.553 |
| nli (en) | **0.807** | 0.747 | 0.440 |
| nli (ja) | 0.633 | **0.673** | 0.333 |
| nli (tr) | 0.800 | **0.813** | 0.353 |
| safety (en) | 0.727 | **0.753** | 0.507 |
| reading (en) | **0.927** | 0.880 | 0.540 |
| similarity (pt) | **0.833** | 0.727 | 0.593 |
| topic (en) | **0.607** | 0.360 | 0.160 |
| intent (en) | **0.820** | 0.800 | 0.553 |
| intent (nl) | **0.440** | **0.440** | 0.107 |
| intent (tr) | **1.000** | 0.827 | 0.407 |
| stance (en) | **0.893** | 0.733 | 0.873 |
| formality (ja) | **0.547** | 0.493 | 0.480 |
| formality (tr) | **1.000** | 0.500 | 0.440 |
| urgency (en) | **0.893** | 0.833 | 0.500 |
| fact_check (en) | 0.313 | **0.513** | 0.347 |
| pii (ar) | 0.847 | **0.893** | 0.480 |
| pii (de) | 0.840 | **0.893** | 0.540 |
| pii (en) | **0.893** | 0.887 | 0.500 |
| pii (es) | 0.853 | **0.867** | 0.493 |
| pii (fr) | **0.893** | 0.880 | 0.533 |
| pii (it) | 0.840 | **0.873** | 0.533 |
| pii (ja) | 0.847 | **0.893** | 0.533 |
| pii (nl) | 0.760 | **0.800** | 0.533 |
| pii (ru) | 0.907 | **0.947** | 0.487 |
| pii (sv) | **0.833** | 0.800 | 0.533 |
| pii (zh) | 0.900 | **0.920** | 0.413 |
| test/ag_news | **0.929** | 0.847 | 0.581 |
| test/emotion | 0.502 | **0.566** | 0.129 |
| test/banking77 | **0.913** | 0.650 | 0.224 |

## Speed on the same machine

| System | Parameters | Hardware | ms per decision | decisions per second |
|---|---|---|---:|---:|
| Statim Decide 0.7.0 | 307M | RTX 3070, Vulkan f32 | 11.6 | 67.9 |
| Qwen3-8B (zero-shot) | 8.2B | RTX 3070, Ollama Q4_K_M | 316.0 | ≈6.3 |
| mDeBERTa-v3-base XNLI (zero-shot) | 279M | RTX 3070, CUDA f32 | 75.0 | 4.1 |
