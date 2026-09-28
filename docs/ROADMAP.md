# Roadmap to 1.0

Goal: a decision engine that is measurably at the top of its class on public benchmarks, with every
claim reproducible from this repository, shipped as a polished product.

"Top" is judged against published numbers under the same protocol (see `docs/` benchmark notes):
zero-shot results against zero-shot systems (Jev, GLiClass, NLI classifiers, LLMs), fine-tuned
results against supervised baselines (MASSIVE paper, Banking77 literature).

## Where we stand (0.8.0)

| Field | Statim | Best published, same protocol |
|---|---|---|
| typed-decisions test | **0.768** statim-decide-en-large; 0.763 statim-decide-multilingual-base | 0.768 meraGPT, 0.766 laya-typed-decisions, 0.727 Jev |
| Banking77, trained on train split | **0.928** en-large; 0.914 multilingual-base | 94.1 MPNet (supervised) |
| MASSIVE, trained | 0.867 en-large (English); 0.800 multilingual-base (12 languages) | 85.7 XLM-R base (12 languages, supervised, full data) |
| AG News, zero-shot (never trained) | 0.939 en-large; 0.9295 multilingual-base | 0.950 Laya, 0.926 GPT-3 (CARP), 0.881 Jev |
| 14 decision categories, held out | **0.748** multilingual-base (0.7.0) | 0.704 Qwen3-8B, 0.488 mDeBERTa-v3 XNLI (both zero-shot, measured on the same items: docs/BASELINES.md) |
| DAIR Emotion (this dataset never trained on) | 0.588 en-large; 0.504 multilingual-base (first 2,000 rows); 0.600 consensus of the Laya checkpoints (first 400) | 0.59–0.60 zero-shot field (Jev 0.590 on 2,000) |

Gaps we measure and work on next: a zero-shot Qwen3-8B still leads in five of the 14 categories,
emotion (0.586 vs 0.726), fact-check (0.313 vs 0.493), sentiment (0.800 vs 0.873), safety (0.727 vs
0.753) and PII (0.856 vs 0.878) (docs/BASELINES.md), and Belebele reading is near chance. Next: more
licence-clean data for these categories and per-category LoRA adapters (engine support since 0.8.0),
compared with the generalist on the same held-out suites.

## Milestones

**0.3 — one generalist model, no regressions**
- A generalist checkpoint trained with the 0.2.0 tooling (task budgets, warmup, EMA; research: T5, UniMax).
- Close the supervised gap on MASSIVE / Banking77: full training data, longer schedules, LR sweep.
- Broader training mixture (15–30 datasets across intent, topic, sentiment, emotion, NLI,
  moderation, support routing; label descriptions, instruction paraphrases, option shuffling),
  with 3–5 datasets held out to measure true zero-shot generalisation.

**0.4 — experts and routing** (engine side done in 0.8.0: per-category LoRA adapters, chosen by name or
by question family; the specialist-versus-generalist experiment is next)
- Router over domain experts with the generalist as fallback; evaluated against the single model on
  the same validation data.
- Option-isolated attention and per-option positions (UniMC) in the engine, if the ablation pays.

**0.5 — performance (done)**
- CUDA backend with exact f32 parity gates; Vulkan vs CUDA measured (f32, f16, q8_0). Batching tuned for GPU remains open.

**0.9 — product**
- Brand applied everywhere (the 0.2.0 logo, icons and social preview), documentation site, OpenAPI spec, client SDKs, Docker images (CPU + GPU), packaged releases,
  security review, model cards and licence notices for published weights.

**1.0 — release**
- Stable API, benchmark report against the published state of the art with reproducible scripts,
  presentation material.
