# Roadmap to 1.0

Goal: a decision engine that is measurably at the top of its class on public benchmarks, with every
claim reproducible from this repository, shipped as a polished product.

"Top" is judged against published numbers under the same protocol (see `docs/` benchmark notes):
zero-shot results against zero-shot systems (Jev, GLiClass, NLI classifiers, LLMs), fine-tuned
results against supervised baselines (MASSIVE paper, Banking77 literature).

## Where we stand (0.8.2)

| Field | Statim | Best published, same protocol |
|---|---|---|
| typed-decisions test | **0.775** statim-decide-multilingual-base; 0.768 statim-decide-en-large | 0.768 meraGPT, 0.766 laya-typed-decisions, 0.727 Jev |
| Banking77, trained on train split | **0.928** en-large; 0.918 multilingual-base | 94.1 MPNet (supervised) |
| MASSIVE, trained | 0.867 en-large (English); 0.816 multilingual-base (12 languages) | 85.7 XLM-R base (12 languages, supervised, full data) |
| AG News, zero-shot (never trained) | 0.939 en-large; 0.921 multilingual-base | 0.950 Laya, 0.926 GPT-3 (CARP), 0.881 Jev |
| 14 decision categories, held out | **0.826** multilingual-base (0.10.0; 0.748 for 0.7.0) | 0.706 Qwen3-8B, 0.488 mDeBERTa-v3 XNLI (both zero-shot, measured on the same items: docs/BASELINES.md) |
| DAIR Emotion (this dataset never trained on) | 0.588 en-large; 0.530 multilingual-base (first 2,000 rows); 0.600 consensus of the Laya checkpoints (first 400) | 0.59–0.60 zero-shot field (Jev 0.590 on 2,000) |

Gaps we measure and work on next: a zero-shot Qwen3-8B still leads in two of the 14 categories,
emotion (0.664 vs 0.726) and fact-check (0.467 vs 0.513) (docs/BASELINES.md); with 0.7.0 it led in
five. Belebele reading is near chance. The 0.10.0 weights carry the same noncommercial licence as
0.7.0; a version trained only on cleared data is next. The first per-category LoRA adapters
([ADAPTERS.md](ADAPTERS.md)), trained on 0.7.0 and bound to it, pass the gate for PII (0.856 to 0.910, above
Qwen3-8B) and emotion (0.586 to 0.639). In a pre-registered replication on 1,350 fresh items,
safety passes too (0.708 to 0.806). All three are published on Hugging Face, and the SDKs select
adapters. Sentiment and fact-check stay within noise. Next:
- a larger held-out sample for fact-check, the other category with one language cell, after its
  retraining;
- more licence-clean data for the rest, generated locally where no human-labelled data exists.
- reading comprehension and minimal pairs: on the public S1Bench suite the 0.10.0 model scores 0.657
  macro (0.7.0: 0.638; Lev 0.689, Jev 0.761) and is still below its own base checkpoint on PAWS,
  boolq, pubmedqa, MultiNLI, squad2 and VitaminC ([s1bench-0.10.0-2026-10-07.md](reproductions/s1bench-0.10.0-2026-10-07.md)); the next mixture adds
  licence-clean reading-comprehension, paraphrase and yes/no QA sources, with S1Bench held out.

## Milestones

**0.3 — one generalist model, no regressions**
- A generalist checkpoint trained with the 0.2.0 tooling (task budgets, warmup, EMA; research: T5, UniMax).
- Close the supervised gap on MASSIVE / Banking77: full training data, longer schedules, LR sweep.
- Broader training mixture (15–30 datasets across intent, topic, sentiment, emotion, NLI,
  moderation, support routing; label descriptions, instruction paraphrases, option shuffling),
  with 3–5 datasets held out to measure true zero-shot generalisation.

**0.4 — experts and routing** (engine side done in 0.8.0: per-category LoRA adapters, chosen by name or
by question family; the first specialist experiment is in docs/ADAPTERS.md)
- Router over domain experts with the generalist as fallback; evaluated against the single model on
  the same validation data.
- Option-isolated attention and per-option positions (UniMC) in the engine, if the ablation pays.

**0.5 — performance (done)**
- CUDA backend with exact f32 parity gates; Vulkan vs CUDA measured (f32, f16, q8_0). Batching tuned for GPU remains open.

**0.9 — product**
- OpenAPI spec, client SDKs, Docker images (CPU + GPU), packaged releases, a security review, and
  model cards and licence notices for published weights shipped between 0.1.0 and 0.5.0. Still
  open: a documentation site and full brand consistency (the 0.2.0 logo, icons and social preview)
  across every surface.

**1.0 — release**
- Every P0 item in [READINESS.md](READINESS.md) closed with its proof (all nine closed as of 0.9.3).
  The server-reliability items P1 #23 to #27 are release criteria too, and all are closed: the 72 h
  soak passed ([soak-2026-10-02.md](reproductions/soak-2026-10-02.md)).
- API v1 frozen (0.9.0): contract tests against the real server and a breaking-change check in CI;
  benchmark report against the published state of the art with reproducible scripts,
  presentation material.
