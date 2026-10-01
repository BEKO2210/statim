# CLAUDE.md

Guidance for Claude Code and other coding agents working in this repository. The user-facing
documentation starts at [README.md](README.md); this file covers how to change the code safely.

## What Statim is

Statim is a C++20 inference engine and HTTP server for Laya decision models. One binary loads a
GGUF file that holds the encoder, the decision head and the tokenizer, and answers typed questions
(`choice`, `score`, `noul`) with calibrated probabilities. Nothing runs Python at inference time.
The Statim Decide models are fine-tuned Laya checkpoints, published on Hugging Face as
`Beko2210/statim-decide-en-large` and `Beko2210/statim-decide-multilingual-base`.

## Repository map

| Path | Contents |
|---|---|
| `src/`, `include/statim/` | the engine: GGUF loading and validation (`model.cpp`), tokenizer, scoring (`engine.cpp`), HTTP server, admission and micro-batching (`server.cpp`), request limits (`security.cpp`, `security.h`), CLI (`main.cpp`) |
| `tests/` | CTest suites: parity with the reference Laya package (golden files in `tests/data/`), security, LoRA, micro-batching, quantization |
| `fuzz/` | libFuzzer harnesses, seeds, and every crash input as a regression test |
| `tools/` | conversion (`convert_laya.py`, `convert_lora.py`, `quantize.cpp`), training and data (`tools/finetune/`, `tools/synth/`), release (`tools/release/`), documentation checks (`tools/docs/`) |
| `bench/` | evaluation and benchmarks (`bench/eval_categories.py`, `bench/baselines.py`, `bench/bench_server.py`) |
| `clients/` | the official Python and TypeScript SDKs |
| `docs/` | API reference (`docs/API.md`, `docs/openapi.yaml`), baselines, deployment, security review, roadmap |
| `site/` | the GitHub Pages site, with its Playwright QA in `site/tests/` |
| `third_party/` | vendored ggml (submodule) and cpp-httplib. Do not edit. |

`build*/`, `models/`, `data/`, `dist/` and `logs/` are local and gitignored. Never commit them.

## Build and test

```sh
git submodule update --init --recursive
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release   # add -DSTATIM_VULKAN=ON or -DSTATIM_CUDA=ON for a GPU backend
cmake --build build
tools/fetch_models.sh multilingual english                 # reference checkpoints for the parity tests
ctest --test-dir build --output-on-failure
```

- Run CTest serially and as a regular user. Several tests use every core, and the security suites
  expect a file with mode 000 to be unreadable, which root can read ([REPRODUCE.md](REPRODUCE.md)).
- `-DSTATIM_FUZZ=ON` builds the fuzzers with clang. `fuzz/run.sh` runs one harness ([fuzz/README.md](fuzz/README.md)).
- `-DSTATIM_SANITIZE=ON` builds everything with ASan and UBSan (clang).

Checks that need no build. CI runs the first two on every push, and the others when their files change:

```sh
python3 tools/docs/test_check_docs.py && python3 tools/docs/check_docs.py   # documentation agrees with the code
python3 tools/release/check_versions.py                                      # every copy of the engine version
python3 -m pytest -q tools/finetune/mixture_v6/test_adapters.py tools/finetune/mixture_v6/test_label_fixes.py \
    tools/finetune/test_gate.py tools/finetune/test_train_lora.py bench/test_eval_categories.py   # needs pytest, pyarrow
python3 site/tests/check.py                                                  # after a site change (needs Playwright)
```

## Invariants

- **Parity.** The engine reproduces the reference Laya package. Token ids must be identical, and
  logits and answers must stay within the tolerances the parity tests set. A change that moves
  numbers needs a reason in the PR. Only a deliberate change to the model or the reference
  regenerates the golden files (`tools/gen_golden.py`).
- **Limits before work.** Every request is bounded before inference (`src/security.cpp`). Each
  limit is a serve flag with its default in `include/statim/security.h` or
  `include/statim/server.h`. A new input path needs a limit and a test. A parser also needs a fuzz
  seed.
- **The API is a contract.** Errors keep the `{"detail": ...}` shape. A change to routes, fields,
  errors, metrics or serve flags updates `docs/API.md`, `docs/openapi.yaml` and both SDKs in the
  same PR. API v1 is frozen in `docs/api-v1.contract.json`; `python3
  tools/docs/api_contract.py --check` rejects breaking changes. `check_docs.py` compares the
  documents with the C++ source.
- **Adapters are bound to their base.** A LoRA adapter records the fingerprint and checkpoint
  SHA-256 of the model it was trained on, and the engine refuses a mismatch. By default adapters
  merge on f32, f16 and bf16 weights and run as runtime LoRA on quantized weights.

## Documentation

- `tools/docs/check_docs.py` runs first in CI. It checks links, paths and command flags, the API
  documents against the server source, the `/metrics` example, release downloads, and the published
  numbers repeated across README, ROADMAP, BASELINES and the site (`FACTS`). Change a number at
  its source (the README model table or `docs/BASELINES.md`). The check lists every copy that
  still has the old value.
- Every result names its protocol and source. Zero-shot and trained results are never mixed.
- Records stay as written: released CHANGELOG sections, `docs/reproductions/`, and
  `tools/finetune/sources/v6-research.md`.
- Every user-visible change gets an entry under `[Unreleased]` in CHANGELOG.md (Keep a Changelog).

## Models, data and evaluation

- **Training data licences.** Only licences that allow commercial use and put no ShareAlike or
  copyleft on the model (Apache-2.0, MIT, BSD, CC0, CC-BY, ODC-By). The licence is verified at the
  source and recorded in `tools/finetune/sources/v6-keep.json`, with the evidence in
  `tools/finetune/sources/v6-research.md`.
- **Never train on:**
  - non-commercial, ShareAlike or copyleft data;
  - data under custom or unknown terms, or gated data;
  - outputs of models whose terms forbid it;
  - public benchmarks;
  - any source that the evaluation holds out.
- **Environments.** Training, evaluation and the gate run in a separate virtual environment with
  torch and the reference `laya` package; [REPRODUCE.md](REPRODUCE.md) creates it as `.venv-train`.
  Measure timings only on an otherwise idle machine, because one GPU is often shared with training.
- **Audit before training.** Build a mixture, then audit its content (`tools/finetune/mixture_v6/audit.py`).
- **The gate decides.** A model ships only when `tools/finetune/gate.py` promotes it: Holm-Bonferroni
  over the held-out suites, validation non-inferiority, and at least one family gain. An adapter
  ships only when `tools/finetune/lora_experiment.py` promotes it (`gate.adapter_decision`).
  Published numbers come from these outputs and are never typed from memory.

## Changes and releases

- Every PR that can affect speed or memory attaches the `bench/perf_gate.py` summary against the latest release.
- One branch and one PR per change, green CI, squash merge. Never push to main directly. The PR
  description lists the commands that verified the change and their results.
- A release: `python3 tools/release/check_versions.py --set X.Y.Z` bumps every copy of the engine
  version (never a model version). Move `[Unreleased]` into a dated section, and update
  `docs/ROADMAP.md` when a milestone moves. After the merge,
  `gh release create vX.Y.Z --target main --notes-file NOTES.md` publishes it: `release.yml`
  attaches the binaries and `SHA256SUMS`. Then upload `deploy/hf-space/Dockerfile`, which
  `--set` already moved to the new version, to the demo Space.
- Semantic Versioning covers the HTTP API, the CLI and the GGUF metadata that the engine reads.
  Since 0.9.0, HTTP API v1 permits only additive changes; a breaking change requires a new major
  version. Every change is in the CHANGELOG.

## Conventions

- C++20. Match the surrounding code, and write comments that explain why, not what.
- The Python tools that CI runs before installing anything (`tools/docs/`, `tools/release/`) use
  the standard library only.
- Back every claim with evidence: a command's output, a test result, or a measurement with its
  protocol.
