# Statim and the Laya PyTorch reference on CPU: protocol, fixed before measuring

This protocol is committed before any timing is taken. The results are published next to it,
whatever they show, and docs/ORT.md gains a Laya column only from these measurements.

## Question
How fast does the reference implementation, the `laya` Python package on PyTorch, score the same
decisions as Statim and ONNX Runtime on the same CPU, with the same model and the same token ids?

## Fixed setup
| | |
|---|---|
| Machine | pop-os: AMD Ryzen 7 5800X, 8 cores, 16 threads, the machine of docs/ORT.md; otherwise idle (no builds, servers or other benchmarks running), `CUDA_VISIBLE_DEVICES=""` |
| Reference | `laya` 0.3.20 on PyTorch 2.14.0, CPU, float32, no autocast, `torch.inference_mode()`, `torch.set_num_threads(t)`, `torch.set_num_interop_threads(1)` |
| Statim, ORT | the docs/ORT.md numbers (Statim 0.9.2, onnxruntime 1.30.0) are not re-measured; Statim 0.9.4 `test_model_parity` is re-run in the same session as a control, so a drift of the machine shows up |
| Model | the shipped statim-decide-multilingual-base checkpoint (`models/laya-multilingual-v9`), the one docs/ORT.md uses |
| Inputs | `build-ort/golden_v9.jsonl`: the 30 states × 8 questions of `tests/data/golden_inputs.json`, already tokenized; identical token ids for every engine |

## Parity gate (before any timing)
The reference must reproduce the stored golden logits for the v9 checkpoint: argmax 240/240 and
max |Δlogit| ≤ 1e-3. If it does not, no timing is published and the mismatch is reported instead.

## Measurements
1. **Raw scoring.** One batch of 8 sequences per state, collated as Laya's own `collate_items`
   does, forward pass only (no tokenization, no answer decoding), at 1, 4 and 8 threads. One
   warm-up pass, then 3 passes; the median per state is stored. Reported like docs/ORT.md: mean ms
   per state over the 29 short states, and the 770-token state separately.
2. **Start-up and footprint.** A fresh process: import, load the checkpoint, answer one decision
   (the README quick-start ticket) through Laya's public API at 8 threads. Reported: wall time from
   process start to the answer, and peak RSS (`/proc/self/status` VmHWM). Install size: the
   site-packages of the environment that runs it, without the interpreter.

## Reporting rules
- Same table shape as docs/ORT.md, every cell with its source file under `bench/results/laya-cpu/`.
- If the reference is faster anywhere, the README and the site say so in the same place as the
  rest of the comparison.
- Jev publishes no speed numbers and cannot be run locally; it appears only with its published
  accuracy, and its speed is shown as "not published".
