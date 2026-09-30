# Reproducing Statim's numbers

Every number Statim publishes comes from a script in this repository, run on data you can download.
This page lists the commands and the values you should see. Differences beyond the stated noise
are worth an issue: https://github.com/BEKO2210/statim/issues.

Hardware: a CPU with 8 GB of free RAM is enough for everything except the training recipe; a GPU
(Vulkan or CUDA) makes the evaluations about ten times faster and gives the same answers, because
Statim's GPU modes are exact f32 by default.

## 1. Ten minutes: the engine answers real tickets

Needs a checkout of this repository (for `examples/ticket-triage/triage.py`, Python 3.10+, standard
library only), the release binary and one published model. The downloads go to `dist/`, which is
gitignored.

```sh
git clone https://github.com/BEKO2210/statim && cd statim
mkdir -p dist && cd dist
# binary (Linux x86-64; or build from source, see docs/BUILD.md)
curl -fLO https://github.com/BEKO2210/statim/releases/download/v0.9.1/statim-0.9.1-linux-x86_64-cpu.tar.gz
curl -fLO https://github.com/BEKO2210/statim/releases/download/v0.9.1/SHA256SUMS
sha256sum -c --ignore-missing SHA256SUMS && tar -xzf statim-0.9.1-linux-x86_64-cpu.tar.gz
# model (q8_0 for CPU, 357 MB) and its checksum list from the model repository
curl -fLO https://huggingface.co/Beko2210/statim-decide-multilingual-base/resolve/main/statim-decide-multilingual-base-q8_0.gguf
curl -fL -o SHA256SUMS.model https://huggingface.co/Beko2210/statim-decide-multilingual-base/resolve/main/SHA256SUMS
sha256sum -c --ignore-missing SHA256SUMS.model
cd ..
dist/statim-0.9.1-linux-x86_64-cpu/statim serve --device cpu -m multilingual=dist/statim-decide-multilingual-base-q8_0.gguf --port 8080 &
until curl -sf localhost:8080/health; do sleep 1; done
python3 examples/ticket-triage/triage.py eval --limit 500 --concurrency 2 --seed 0
```

Expected, measured with the v0.7.0 release and the multilingual 0.7.0 model (Banking77 test split,
seeded stratified sample of 500 tickets, 77 intents in one question): intent accuracy **0.908**
(454/500); with `min_confidence` 0.9 the server escalates 2 % of the tickets and accuracy on the rest
is **0.920** (0.9204, 451/490). Exact values and the coverage curve are in
[examples/ticket-triage/README.md](examples/ticket-triage/README.md). The first run
downloads the Banking77 test split (about 100 KB) into `examples/ticket-triage/.cache/`. On a 4-core
cloud VM the 500 requests take about 6 minutes.

The command rewrites the committed `examples/ticket-triage/results.json` and `results.md` with your
machine's CPU and latency; the accuracy rows should be unchanged. `git checkout examples/ticket-triage`
restores the published files.

## 2. The engine matches the Python reference

The parity gates compare Statim's logits with the Laya reference implementation on 240 recorded
sequences (tests/data/golden_*.jsonl) and require agreement within 1e-4.

```sh
git submodule update --init --recursive         # third_party/ggml; skip if you cloned with --recursive
pip install numpy safetensors gguf              # converter only; use a venv where the system Python refuses pip (PEP 668)
tools/fetch_models.sh multilingual english     # Laya base checkpoints -> models/*-f32.gguf (2.5 GB)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build
ctest --test-dir build                          # add -DSTATIM_VULKAN=ON / -DSTATIM_CUDA=ON for the GPU gates
```

Expected: `100% tests passed`; `-DSTATIM_VULKAN=ON` and `-DSTATIM_CUDA=ON` add their GPU parity tests. Without the
submodule, CMake stops with `third_party/ggml does not contain a CMakeLists.txt file`.

Run `ctest` as a regular user. As root (the default in Docker and many cloud sandboxes) `security`
and `security_model` fail with `security failure: expected startup failure`: they check that an
unreadable key file (`chmod 000`) is rejected, and root can read it anyway. The other tests pass as
root. In a root-only container you can run the suite as `nobody` after
`chmod -R o+rwX build`: `setpriv --reuid=65534 --regid=65534 --clear-groups env HOME=/tmp ctest --test-dir build`.

## 3. A published model's evaluation

The no-harm gate evaluates a checkpoint on validation data and 54 held-out suites and writes
`eval.json`. The published `evaluation/eval.json` next to each model is the reference to compare with.

```sh
python3 -m venv .venv-train && .venv-train/bin/pip install torch laya==0.3.20 datasets huggingface_hub
# the checkpoint (Laya format) and the f32 GGUF, laid out as the gate expects
.venv-train/bin/hf download Beko2210/statim-decide-en-large --local-dir dist/statim-decide-en-large
ln -s ../dist/statim-decide-en-large/checkpoint models/statim-decide-en-large
ln -s ../dist/statim-decide-en-large/statim-decide-en-large-f32.gguf models/statim-decide-en-large-f32.gguf
mkdir -p models/statim-decide-en-large-published && cp dist/statim-decide-en-large/evaluation/eval.json models/statim-decide-en-large-published/
# evaluate (about an hour on an RTX 3070; several hours on a CPU)
STATIM_BIN=build/statim STATIM_GATE_DEVICE=cpu .venv-train/bin/python tools/finetune/gate.py eval models/statim-decide-en-large
.venv-train/bin/python tools/finetune/gate.py compare models/statim-decide-en-large-published models/statim-decide-en-large
```

Expected for `statim-decide-en-large` 0.5.0 (`models/statim-decide-en-large/eval.json`):

| Suite | Accuracy | Rows |
|---|---|---|
| `test/typed_decisions` | 0.7680 | 2,000 |
| `test/banking77` | 0.9280 | 2,000 |
| `test/ag_news` | 0.9390 | 2,000 |
| `test/emotion` | 0.5880 | 2,000 |
| `amazon_massive_intent/en` | 0.8667 | 150 |
| `hwu64/en` | 0.8333 | 150 |

and for `statim-decide-multilingual-base` 0.7.0: typed-decisions 0.7630, Banking77 0.9140,
AG News 0.9295, Emotion 0.5040, MASSIVE mean over 12 languages 0.7995. The suites with 2,000 rows
are deterministic (first rows of the test split); the 150-row suites are seeded stratified samples
(`--seed 20260926`), so they are deterministic too. The `compare` step should report every suite
"within noise" and no significant difference in either direction.

Notes for a CPU-only machine:

- `huggingface_hub` 1.x (what an unpinned `pip install` gives) replaced `huggingface-cli` with `hf`;
  the old command only prints a deprecation notice and exits 1. Several patterns need one
  `--include` each (`--include "checkpoint/*" --include "evaluation/*"`).
- The default Linux `torch` wheel bundles CUDA; the venv is about 6 GB even on a machine without a GPU.
- `STATIM_GATE_DEVICE=cpu` only selects the device of the Statim server (MASSIVE and zero-shot
  suites). `eval_dev.py` and `eval_laya.py` load the Laya reference in PyTorch with
  `device="cuda"`; without a GPU, Laya prints `CUDA requested but not available. Falling back to CPU.`
  and continues on the CPU. That fallback is expected.
- Statim's exact GPU mode agrees with its CPU mode. The PyTorch reference on CPU and on CUDA can
  differ in a few borderline rows of 2,000 (well inside the ±2 standard errors the gate allows).

The four 2,000-row suites alone, without the full gate, for the multilingual model (the checkpoint
only, 0.65 GB):

```sh
.venv-train/bin/hf download Beko2210/statim-decide-multilingual-base --include "checkpoint/*" --include "evaluation/*" --local-dir dist/statim-decide-multilingual-base
.venv-train/bin/python tools/finetune/eval_laya.py dist/statim-decide-multilingual-base/checkpoint --n 2000 --head-max-len 512
```

This is the call `gate.py eval` makes. It prints one JSON line per suite; compare `accuracy` with
`heldout` in `dist/statim-decide-multilingual-base/evaluation/eval.json`. `--n` applies to AG News,
Emotion and Banking77 (first *n* test rows); typed-decisions always uses the whole test split
(400 states, 2,000 decisions). On a 4-core cloud VM without a GPU it takes about 20 minutes
(AG News 2.5 min, Emotion 1.3 min, Banking77 9 min, typed-decisions 6.5 min). Obtained there for 0.4.0:
typed-decisions 0.7585, Banking77 0.9035, AG News 0.9310, Emotion 0.5285.

## 4. The model beats its base checkpoint

```sh
tools/fetch_models.sh english                                    # convaiinnovations/laya -> models/laya, models/laya-english-f32.gguf
ln -sfn laya-english-f32.gguf models/laya-f32.gguf
STATIM_BIN=build/statim STATIM_GATE_DEVICE=cpu .venv-train/bin/python tools/finetune/gate.py eval models/laya
.venv-train/bin/python tools/finetune/gate.py compare models/laya models/statim-decide-en-large
```

Expected: `VERDICT: PROMOTE`, 11 significant gains, 0 significant regressions, the trained family
+18.6 points (rows) and the zero-shot family within noise. For the multilingual model against
`convaiinnovations/laya-multilingual`: 18 gains, 36 within noise, 0 regressions.

## 5. Speed

```sh
./build/statim serve --device cpu -m english=models/statim-decide-en-large-f32.gguf --port 8310 &
.venv-train/bin/python bench/bench_server.py --url http://127.0.0.1:8310 --concurrency 1 8 --requests 60
```

Numbers depend on the machine; the README's tables state the hardware for every row (RTX 3070,
Ryzen 7 5800X). Compare ratios, not absolute values: for example `--gpu-fast` roughly doubles the
Vulkan throughput of the English model, q8_0 on CUDA roughly doubles it against exact f32.

## 6. Training the model again

The full recipe and its data are in the repository: mixture built by `tools/finetune/build_mixture.py`
and `build_extra.py` from the licence-audited sources in `DATA_LICENSES.md`, then
`train_multitask.py` with the command printed in the README "Results" section (about 7 hours on an
RTX 3070 for the English model, 8-bit AdamW). Training is seeded but not bit-reproducible across
GPUs; a retrained model should pass the same gate with the same verdict, and its numbers should sit
within two standard errors of the published ones.
