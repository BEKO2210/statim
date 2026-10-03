# Statim and the Laya PyTorch reference on CPU (2026-10-03)

Protocol, committed before any timing: [laya-speed-protocol.md](laya-speed-protocol.md). Raw files:
`bench/results/laya-cpu/` (`parity.json`, `raw.json`, `startup.json`, `statim-control.json`),
written by `bench/laya_compare.py` and, for the Statim control, the `statim_raw` function of
`bench/ort_compare.py`.

## Parity gate
The reference reproduces the stored golden logits of the shipped checkpoint exactly: argmax 240/240,
max |Δlogit| 0.0. Statim's control run in the same session: 960/960 (four passes over the 240 items),
max |Δlogit| 2.4e-5.

## Raw scoring (ms per state; one batch of 8 questions; median of 3 passes after 1 warm-up)
| Threads | Statim, short | Laya, short | Laya / Statim | Statim, 770 tok | Laya, 770 tok | Laya / Statim |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,447 | 1,451 | 1.00× | 15,401 | 18,718 | 1.22× |
| 4 | 420 | 545 | 1.30× | 4,572 | 6,951 | 1.52× |
| 8 | 271 | 299 | 1.10× | 3,120 | 4,866 | 1.56× |

"Short" is the mean over the 29 states of at most 144 tokens. The Statim column is today's control
run (`test_model_parity`, release flags, source of v0.9.3). It agrees with the docs/ORT.md values
measured for 0.9.2 within 1.5 % except 8 threads on short inputs (271 against 257 ms, +5 %), so the
machine had not drifted; the ORT columns of docs/ORT.md are not re-measured.

## Start-up and footprint (8 threads, the README quick-start ticket)
| | Statim | Laya (PyTorch) |
|---|---:|---:|
| Process start to first answer | 0.45 s (docs/ORT.md) | 5.03 s |
| Peak resident memory | 637 MiB in the start-up window (docs/ORT.md) | 2,653 MiB (VmHWM) |
| Install | 5.2 MiB executable | 5,981 MiB: the training environment that ran it (laya, torch 2.14 with CUDA libraries, and the rest of the training stack), so an upper bound for a minimal CPU install |

Laya's answer to the quick-start ticket equals Statim's to the fourth decimal: billing 0.995,
urgency score 1.583, refund 0.9261.

## Summary
On one thread the reference and Statim are level on short inputs. With more threads, and on long
inputs at any thread count, Statim is faster: 1.10× to 1.56×. It starts 11× faster and needs a quarter
of the memory, without Python.
