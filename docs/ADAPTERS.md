# Category adapters

Statim serves LoRA adapters next to a base model and routes a request to one by name or by question
category ([docs/API.md](API.md#lora-adapters)). This page records the adapters trained for Statim
Decide and the evidence behind each decision. An adapter is published only when the gate promotes
it.

## Published adapters

| Adapter | Repository | File (13.5 MB) | SHA-256 |
|---|---|---|---|
| PII | [Beko2210/statim-decide-multilingual-base-pii](https://huggingface.co/Beko2210/statim-decide-multilingual-base-pii) | `statim-decide-multilingual-base-pii.lora.gguf` | `2991a33b5d9db4f5b679081885faab1f84289d6eae7660830447e790bd168b29` |
| Emotion | [Beko2210/statim-decide-multilingual-base-emotion](https://huggingface.co/Beko2210/statim-decide-multilingual-base-emotion) | `statim-decide-multilingual-base-emotion.lora.gguf` | `c832dc6aada9bf0b07e7f481b554d51ddff2fe0c1365ba49145de13f3952bdc8` |
| Safety | [Beko2210/statim-decide-multilingual-base-safety](https://huggingface.co/Beko2210/statim-decide-multilingual-base-safety) | `statim-decide-multilingual-base-safety.lora.gguf` | `9620969cef129a9ede45816db575a6c5b9ecf27574443b0110259bec1b703202` |

All three adapt statim-decide-multilingual-base 0.7.0 and are bound to it by its fingerprint. They
were checked on the published base files. On f32, merged at load, they reproduce the measurements
below cell for cell: the experiment for PII and emotion, and the replication for safety. On q8_0, as
runtime LoRA, the mean change over the language cells is +5.52 points for PII, +5.33 for emotion and
+10.30 for safety. Each repository carries its evaluation files, training record, PEFT
source and checksums. `tools/release/hf_publish_adapter.py` builds them from the experiment's
outputs and refuses an adapter the gate did not promote.

```sh
statim serve -m multilingual=statim-decide-multilingual-base-q8_0.gguf \
    --adapter multilingual:pii=statim-decide-multilingual-base-pii.lora.gguf \
    --adapter multilingual:emotion=statim-decide-multilingual-base-emotion.lora.gguf \
    --adapter multilingual:safety=statim-decide-multilingual-base-safety.lora.gguf
```

With `"adapter": "auto"`, PII, emotion and safety questions use their adapter, and every other
question uses the base weights.

## Specialists for the weak categories (2026-09-29)

**Question.** A zero-shot Qwen3-8B leads statim-decide-multilingual-base 0.7.0 in five of the 14
decision categories ([BASELINES.md](BASELINES.md)). Does a LoRA adapter per category, trained on
top of 0.7.0, close that gap?

### Protocol

- **Base:** statim-decide-multilingual-base 0.7.0 in f32 (the checkpoint in the model repository's
  `checkpoint/` folder), served on an RTX 3070 with Vulkan.
- **Training** (`tools/finetune/train_lora.py`): LoRA rank 16 on `attn.Wqkv`, `attn.Wo`, `mlp.Wi`
  and `mlp.Wo` of every encoder layer (88 modules). Everything else stays frozen.
  - Data: the category's rows of the 0.7.0 training mixture.
  - Schedule: 2 epochs at learning rate 2e-4.
  - Kept: the adapter with the best accuracy on 400 held-back training items.
- **Evaluation** (`bench/eval_categories.py`): the held-out category suites.
  - 150 items per language cell, seed 20260927.
  - Items whose text occurs in the training mixture are removed.
  - The same server answers every item twice: once without the adapter and once with it.
- **Decision** (`gate.adapter_decision`): promote when the category family gains significantly,
  pooled by rows or by suites, and nothing regresses. A regression is a pooled drop beyond 2
  standard errors under either pooling, or a drop in one language cell that stays significant after
  Holm-Bonferroni (family-wise 5 %).
  - Base and adapter answer the same items, so every test is paired: an exact McNemar test per cell
    and the standard error of the per-item differences for the pools. Family gains are
    Holm-corrected.
  - These decisions were first made with an unpaired rule, which treated the two runs as
    independent samples. On 2026-10-01 they were recomputed with the paired rule: the same
    adapter files (SHA-256 identical), the same items, the same accuracies. The table shows
    both.

### Results

| Category | Language cells | Pooled change (points) | 2 SE unpaired | Decision then | 2 SE paired | Paired decision |
|---|---:|---:|---:|---|---:|---|
| PII | 11 | +5.46 | 2.22 | **promote** | 1.40 | **promote** |
| emotion | 8 | +4.75 | 3.88 | **promote** | 2.12 | **promote** |
| safety | 1 | +8.00 | 9.72 | reject, within noise | 7.45 | promote (Holm p = 0.03) |
| sentiment | 2 | +2.33 | 6.38 | reject, within noise | not recomputed | — |
| fact-check | 1 | +0.67 | 10.74 | reject, within noise | not recomputed | — |

The pooled PII change is +5.45 points when recomputed from the items; the table keeps the published
+5.46, a rounding difference.

No language cell regressed in any category. Accuracy per cell for the two promoted adapters:

| PII | ar | de | en | es | fr | it | ja | nl | ru | sv | zh |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0.847 | 0.840 | 0.893 | 0.853 | 0.893 | 0.840 | 0.847 | 0.760 | 0.907 | 0.833 | 0.900 |
| adapter | 0.907 | 0.867 | 0.927 | 0.893 | 0.940 | 0.920 | 0.927 | 0.867 | 0.947 | 0.873 | 0.947 |

| Emotion | de | en | es | fr | hi | pt | ru | zh |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0.453 | 0.573 | 0.573 | 0.660 | 0.713 | 0.507 | 0.693 | 0.540 |
| adapter | 0.460 | 0.620 | 0.580 | 0.787 | 0.827 | 0.540 | 0.720 | 0.560 |

### What this shows

- **PII and emotion gain significantly.** Each adapter is a 13.5 MB file (rank 16). The base
  accuracies equal the Statim column of [BASELINES.md](BASELINES.md) cell for cell, so both runs
  used the same items.
  - PII: the mean over the 11 cells rises from 0.856 to 0.910, above Qwen3-8B's 0.878.
  - Emotion: the mean over its six gate cells rises from 0.586 to 0.639. Qwen3-8B's 0.726 stays
    ahead.
- **The emotion result does not rest on untrained languages.** The pt and ru cells had no emotion
  training data in the 0.7.0 mixture: their sources were registered later. Without them the gain is
  +5.34 points (2 SE 4.47), still a promotion.
- **Safety and fact-check need larger samples.** Each has one language cell of 150 items, so a gain
  must exceed about 10 points to count. Safety's +8.0 was the candidate for a larger held-out
  sample. The replication below confirms it.
- **Fact-check needs a new run.** The fact-check adapter was trained and measured while two of its
  three options had no description, a lookup bug fixed after this run. It needs a rebuilt mixture.

The promoted adapters are published (see above), and the client SDKs select adapters since 0.8.3.
New runs for the rejected categories are next ([ROADMAP](ROADMAP.md)).

### Replication: safety on fresh items

Registered on 2026-09-29, before any fresh item was scored. The safety adapter gained +8.0 points on
its single cell of 150 items, inside the noise band (2 SE 9.7). This replication tests the same file
on items the first run never scored.

- **Adapter:** the file measured above, unchanged: `models/lora-exp1/safety.lora.gguf`, SHA-256
  `9620969cef129a9ede45816db575a6c5b9ecf27574443b0110259bec1b703202`.
- **Base:** statim-decide-multilingual-base 0.7.0, f32 on Vulkan, as above.
- **Items:** the safety suite in English, the only language with at least 150 pooled items.
  - The draw uses seed 20260927 and 1,500 items with the first 150 dropped (`--n 1500 --skip 150`).
  - The draw is prefix-stable, so these 1,350 items are disjoint from the 150 above.
  - The whole draw has 752 and 748 items per class (listed without a model).
  - Items that share a text with the 0.7.0 training mixture are removed.
- **Power:** at the first run's accuracies, 2 SE is about 3.2 points.
- **Decision:** `gate.adapter_decision` with z = 2, on the fresh items alone. A promotion means the
  adapter is published as the third one; a rejection means it is not. Both runs are reported here.

**Result: promote.**

- On the 1,350 fresh items, accuracy rises from 0.708 to 0.806: +9.78 points, 2 SE 3.28, and no
  regression. Recomputed with the paired rule: 2 SE 2.39, Holm p = 2.5e-16, promote.
- The converted file is byte-identical to the registered one (the SHA-256 above).
- The first run's +8.0 on 150 items was therefore an underpowered true effect, not noise. The
  paired rule detects it on those 150 items too (Holm p = 0.03). The unpaired rule could not.
- The adapter is published (see above).
- A zero-shot Qwen3-8B was measured only on the first 150 items (0.753), so no comparison on the
  fresh items is claimed.

### Reproduce

```sh
.venv/bin/python tools/finetune/lora_experiment.py --base-checkpoint models/laya-multilingual-v9 \
    --base-gguf models/laya-multilingual-v9-f32.gguf --mixture data/mixture-v8.jsonl.gz \
    --statim build-vk/statim --device cuda --server-device vulkan --work models/lora-exp1
```

The safety replication reuses the trained adapter (`--skip-train`) in a work dir of its own and
scores the fresh items:

```sh
mkdir -p models/lora-exp1-safety-rep && cp -r models/lora-exp1/safety models/lora-exp1-safety-rep/
.venv-train/bin/python tools/finetune/lora_experiment.py --base-checkpoint models/laya-multilingual-v9 \
    --base-gguf models/laya-multilingual-v9-f32.gguf --mixture data/mixture-v8.jsonl.gz \
    --statim build-vk/statim --device cuda --server-device vulkan --categories safety --skip-train \
    --work models/lora-exp1-safety-rep --n 1500 --eval-skip 150 --seed 20260927
```

- `models/laya-multilingual-v9` is the 0.7.0 checkpoint, and `models/laya-multilingual-v9-f32.gguf`
  is its conversion (`tools/convert_laya.py`).
- `data/mixture-v8.jsonl.gz` is the 0.7.0 training mixture
  ([REPRODUCE.md](../REPRODUCE.md#6-training-the-model-again)).
- Training runs in `.venv-train` (torch, peft, laya). Conversion and evaluation run in `.venv`
  (gguf, numpy). Evaluation also needs datasets, huggingface_hub and pyarrow, unless the category
  pool cache `data/category-suites.jsonl.gz` is current.
- The run writes `summary.md` and `summary.json` to `--work`, with every command. It takes about
  two hours on an RTX 3070.
