# Changelog

All notable changes to Statim are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0 the HTTP API may still change
between minor versions; every change is listed here.

## [Unreleased]

## [0.2.0] - 2026-09-27

GPU inference, a production-grade playground, and a reproducible fine-tuning and evaluation toolkit.

### Added
- **Vulkan GPU backend.** `--device cpu|gpu|vulkan|<name>` (or `STATIM_DEVICE`) selects the ggml
  backend; weights are copied to VRAM once. Exact f32 by default: ggml-vulkan's f16 matmul paths
  moved logits by up to 0.12, so they are disabled unless `--gpu-fast` / `STATIM_GPU_FAST=1`.
  Build with `-DSTATIM_VULKAN=ON`, which also registers the parity gates on the GPU. RTX 3070 vs.
  Ryzen 7 5800X: 7.7–8.9× HTTP throughput, parity 240/240 at max |Δlogit| 8.8e-5 / 1.6e-4.
- **Option budget per request.** `max_len` / `head_max_len` request fields (as in Laya's
  `predict_batch`) and server defaults `--max-len` / `--head-max-len`. Many-option questions
  (e.g. 77 intents) otherwise see one subword per option.
- **Playground redesign.** Two-pane layout, live server status, model details from `/v1/models`,
  API key dialog, light/dark theme, examples (including a German support ticket with a separate
  revenue-impact question), question templates, JSON linting, per-type result views with
  confidence meters, JSON and cURL tabs, session history, keyboard shortcut.
- **Fine-tuning toolkit** (`tools/finetune/`):
  - `train_banking77.py`: Laya's RLCD recipe on one task, frozen token embeddings, typed-decisions
    replay and `--distill` learning-without-forgetting replay.
  - `train_multitask.py`: Banking77 + MASSIVE (51 languages) + multilingual sentiment (12
    languages); train rows that also occur in a test split are dropped (14,121 MASSIVE and 452
    sentiment rows, per language in `bench/results/multitask_train_test_overlap.json`); per-epoch
    task budgets, LR warmup and EMA weights.
  - `merge.py`: model soup, task arithmetic and TIES merging of checkpoints from one base.
  - `eval_laya.py` (test suites on GPU) and `eval_dev.py` (model selection on validation data only).
- **Multilingual benchmark** `bench/eval_multilingual.py`: MASSIVE intents (59 labels) and
  multilingual sentiment, 12 languages each, seeded stratified samples with identical MASSIVE rows
  across languages; results for the base checkpoints in `bench/results/`.
- `CHANGELOG.md` and `docs/ROADMAP.md`.

### Changed
- `/health` and the startup log report the actual compute device instead of a fixed `"cpu"`.
- The engine raises `max_len` with a raised `head_max_len` so the state keeps at least 128 tokens.
  This never binds at the checkpoint defaults; CPU and GPU parity are unchanged.
- The model parity test prints per-state deviations (`STATIM_VERBOSE`) and can run one graph per
  item (`STATIM_BATCH1`).

### Fixed
- Playground: the mode selector silently overwrote the chosen model. `consensus` is now a model
  option, offered only when both checkpoints are loaded.

### Results (weights are reproducible with the scripts; not shipped)
- Banking77 fine-tune v3 (`--distill 6000 --epochs 5`), first 2,000 test rows: 0.4885 → 0.8655,
  ECE 0.37 → 0.04; held-out Emotion 0.532 → 0.528 and AG News 0.938 → 0.9385 (within noise).
- Multi-task checkpoint (experimental): MASSIVE macro over 12 languages 0.340 → 0.689, multilingual
  sentiment 0.559 → 0.654, but Banking77 0.8275 (−3.8 vs. v3). Not yet recommended.

## [0.1.0] - 2026-09-26

First release: C++20 runtime for Laya decision checkpoints on ggml, native tokenizer (identical to
Hugging Face on 3,906 cases), parity gates against the official package, HTTP server with the
Jev/Laya `POST /v1/systemone` protocol, batching, consensus mode, contextual calibration, worker
pool, auth, Prometheus metrics, playground, Docker and systemd packaging.

[Unreleased]: https://github.com/BEKO2210/statim/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/BEKO2210/statim/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/BEKO2210/statim/releases/tag/v0.1.0
