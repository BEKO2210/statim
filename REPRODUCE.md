# Reproducing Statim's numbers

Every number Statim publishes comes from a script in this repository, run on data you can download.
This page lists the commands and the values you should see. Differences beyond the stated noise
are worth an issue: https://github.com/BEKO2210/statim/issues.

Hardware: a CPU with 8 GB of free RAM is enough for everything except the training recipe; a GPU
(Vulkan or CUDA) makes the evaluations about ten times faster and gives the same answers, because
Statim's GPU modes are exact f32 by default.

## 1. Ten minutes: the engine answers real tickets

Needs only the release binary and one published model.

```sh
# binary (Linux x86-64; or build from source, see README "Quick start")
curl -fLO https://github.com/BEKO2210/statim/releases/download/v0.5.0/statim-0.5.0-linux-x86_64-cpu.tar.gz
curl -fLO https://github.com/BEKO2210/statim/releases/download/v0.5.0/SHA256SUMS
sha256sum -c --ignore-missing SHA256SUMS && tar -xzf statim-0.5.0-linux-x86_64-cpu.tar.gz
# model (q8_0 for CPU, 357 MB)
curl -fLO https://huggingface.co/Beko2210/statim-decide-multilingual-base/resolve/main/statim-decide-multilingual-base-q8_0.gguf
./statim-0.5.0-linux-x86_64-cpu/statim serve -m multilingual=statim-decide-multilingual-base-q8_0.gguf --port 8080 &
python3 examples/ticket-triage/triage.py eval --limit 500 --concurrency 2 --seed 0
```

Expected (Banking77 test split, seeded stratified sample of 500 tickets, 77 intents in one
question): intent accuracy **0.896**; with `min_confidence` 0.9 the server escalates 4 % of the
tickets and accuracy on the rest is **0.915**. Exact values and the coverage curve are in
[examples/ticket-triage/README.md](examples/ticket-triage/README.md).

## 2. The engine matches the Python reference

The parity gates compare Statim's logits with the Laya reference implementation on 240 recorded
sequences (tests/data/golden_*.jsonl) and require agreement within 1e-4.

```sh
tools/fetch_models.sh multilingual english     # Laya base checkpoints -> models/*-f32.gguf (needs: pip install numpy safetensors gguf)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build
ctest --test-dir build                          # add -DSTATIM_VULKAN=ON / -DSTATIM_CUDA=ON for the GPU gates
```

Expected: `100% tests passed` (8 tests on CPU; 12 with a GPU build).

## 3. A published model's evaluation

The no-harm gate evaluates a checkpoint on validation data and 54 held-out suites and writes
`eval.json`. The published `evaluation/eval.json` next to each model is the reference to compare with.

```sh
python3 -m venv .venv-train && .venv-train/bin/pip install torch laya==0.3.20 datasets huggingface_hub
# the checkpoint (Laya format) and the f32 GGUF, laid out as the gate expects
huggingface-cli download Beko2210/statim-decide-en-large --local-dir dist/statim-decide-en-large
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

and for `statim-decide-multilingual-base` 0.4.0: typed-decisions 0.7585, Banking77 0.9035,
AG News 0.9315, Emotion 0.5265, MASSIVE mean over 12 languages 0.7717. The suites with 2,000 rows
are deterministic (first rows of the test split); the 150-row suites are seeded stratified samples
(`--seed 20260926`), so they are deterministic too. The `compare` step should report every suite
"within noise" and no significant difference in either direction; GPU and CPU runs agree because
the GPU mode is exact.

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
