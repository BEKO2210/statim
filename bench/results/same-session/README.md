# Statim, ONNX Runtime and Laya in one session (2026-10-03)

Raw scoring of the 30 golden states (8 questions each, identical token ids) on pop-os (Ryzen 7
5800X, idle, `CUDA_VISIBLE_DEVICES=""`), measured back to back so the three engines share one
machine state: Statim, then ONNX Runtime, then the Laya PyTorch reference, then Statim again.

- `statim-ort.json`: Statim (`test_model_parity` from the v0.9.3 source, release flags, model
  `laya-multilingual-v9-f32.gguf`) before and after, and ORT 1.30 (`laya-multilingual-v9-f32.onnx`),
  via `bench/ort_compare.py` (`statim_raw`, `ort_raw`); 1 warm-up and 3 passes, median per state.
- `laya.json`: `bench/laya_compare.py raw`, same protocol.

Mean ms per state over the 29 short states (at most 144 tokens) / the 770-token state:

| Threads | Statim (first) | ORT | Laya | Statim (last) |
|---:|---:|---:|---:|---:|
| 1 | 1,438 / 15,236 | 1,426 / 19,023 | 1,440 / 18,648 | 1,433 / 15,190 |
| 4 | 421 / 4,592 | 424 / 7,918 | 546 / 6,841 | 416 / 4,571 |
| 8 | 262 / 3,044 | 266 / 7,054 | 304 / 4,850 | 262 / 3,077 |

Parity in the same runs: Statim 960/960 argmax (four passes), max |Δlogit| 2.4e-5; ORT 240/240,
1.0e-5. The first and last Statim runs differ by at most 1.2 %, so the machine did not drift.
The chart in `assets/readme/compare.svg` uses the 8-thread rows.
