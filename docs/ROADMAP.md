# Roadmap to 1.0

Goal: a decision engine that is measurably at the top of its class on public benchmarks, with every
claim reproducible from this repository, shipped as a polished product.

"Top" is judged against published numbers under the same protocol (see `docs/` benchmark notes):
zero-shot results against zero-shot systems (Jev, GLiClass, NLI classifiers, LLMs), fine-tuned
results against supervised baselines (MASSIVE paper, Banking77 literature).

## Where we stand (0.4.0)

| Field | Statim | Best published, same protocol |
|---|---|---|
| AG News, zero-shot (never trained) | 0.9315 | 0.950 Laya, 0.926 GPT-3 (CARP), 0.881 Jev |
| typed-decisions test | 0.7585 (licence-clean model) | 0.768 meraGPT, 0.766 laya-typed-decisions, 0.727 Jev |
| Banking77, trained on train split | 0.903 (licence-clean model) | 94.1 MPNet (supervised) |
| MASSIVE, 12 languages, trained | 0.772 (licence-clean model, 2,000 rows per language) | 85.7 XLM-R base (supervised, MASSIVE paper, full data) |
| Emotion, zero-shot (never trained) | 0.600 consensus, first 400 rows (`bench/results/acc_consensus.json`); 0.528 fine-tuned v3, first 2,000 | 0.59–0.60 zero-shot field (Jev 0.590 on 2,000) |

## Milestones

**0.3 — one generalist model, no regressions**
- A generalist checkpoint trained with the 0.2.0 tooling (task budgets, warmup, EMA; research: T5, UniMax).
- Close the supervised gap on MASSIVE / Banking77: full training data, longer schedules, LR sweep.
- Broader training mixture (15–30 datasets across intent, topic, sentiment, emotion, NLI,
  moderation, support routing; label descriptions, instruction paraphrases, option shuffling),
  with 3–5 datasets held out to measure true zero-shot generalisation.

**0.4 — experts and routing**
- Router over domain experts with the generalist as fallback; evaluated against the single model on
  the same validation data.
- Option-isolated attention and per-option positions (UniMC) in the engine, if the ablation pays.

**0.5 — performance**
- CUDA backend, f16/q8 on GPU with parity gates, batching tuned for GPU.

**0.9 — product**
- Brand applied everywhere (the 0.2.0 logo, icons and social preview), documentation site, OpenAPI spec, client SDKs, Docker images (CPU + GPU), packaged releases,
  security review, model cards and licence notices for published weights.

**1.0 — release**
- Stable API, benchmark report against the published state of the art with reproducible scripts,
  presentation material.
