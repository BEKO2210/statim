# Clean-room reproduction, 2026-09-27

An independent run of [REPRODUCE.md](../../REPRODUCE.md) sections 1–3 on a machine that had never
seen the project: fresh clone, no models, no Python packages, no GPU. Commands were taken from the
document as written; where they failed, the failure is recorded and the smallest working fix was
used. The fixes are in the same pull request as this file.

## Machine

| | |
|---|---|
| CPU | Intel Xeon @ 2.10 GHz, 4 vCPUs (AVX2, AVX-512F, FMA, F16C) |
| RAM | 16 GB, no swap |
| GPU | none |
| OS | Ubuntu 24.04.4 LTS, kernel 6.18, running as root in a cloud container |
| Toolchain | GCC 13.3.0, CMake 3.28.3, Ninja, Python 3.11.15 |
| Network | HTTPS through a proxy; GitHub releases and Hugging Face reachable |
| Commits | `c5cd1c6` for sections 1–3; `8685d6c` (main after #12) for a second build and `ctest` |

## Results

| Check | Expected | Obtained | Match |
|---|---|---|---|
| v0.5.2 CPU tarball against `SHA256SUMS` | OK | OK | yes |
| q8_0 GGUF against the model repo's `SHA256SUMS` | (not in the document) | OK | yes |
| Ticket triage, intent accuracy (500, seed 0) | 0.896 | **0.8960** (448/500) | exact |
| Ticket triage, `min_confidence` 0.9 | 0.915 at 4 % escalated | **0.9146** (439/480), 4.0 % escalated | exact |
| Ticket triage, full coverage curve | `results.md` in the repo | identical in all six rows | exact |
| `ctest`, `c5cd1c6`, as root | 8/8 | 6/8 (`security`, `security_model` fail) | no, see problem 4 |
| `ctest`, `8685d6c`, as root | 8/8 in the document | 8/10 (same two fail) | no, see problems 4 and 5 |
| `ctest`, `8685d6c`, as `nobody` | — | **10/10** | yes |
| `eval_laya.py`, typed-decisions | 0.7585 | **0.7585** (n 2,000) | exact |
| `eval_laya.py`, Banking77 | 0.9035 | **0.9035** (n 2,000) | exact |
| `eval_laya.py`, AG News | 0.9315 | **0.9310** (n 2,000) | −1 row |
| `eval_laya.py`, Emotion | 0.5265 | **0.5285** (n 2,000) | +4 rows |

`eval_laya.py` ran with `--n 2000` (the published protocol), not the reduced `--n 400`. The AG News
and Emotion differences are 1 and 4 of 2,000 rows, far inside the gate's noise band (one standard
error is 0.006 on AG News and 0.011 on Emotion). They come from the Python reference running in
PyTorch on the CPU instead of CUDA. The two exact suites and the exact triage curve (which runs
through Statim, not PyTorch) are consistent with that.

Calibration also moved slightly: Banking77 ECE 0.0827 vs 0.085 published, AG News 0.0101 vs 0.0088,
Emotion 0.2107 vs 0.2127.

## Steps and timings

Wall-clock times on the machine above. The source build overlapped with the first ~80 s of the
triage evaluation, so the triage latency figures are slightly pessimistic.

| Step | Command (abridged) | Time |
|---|---|---|
| Release download + verify | `curl` tarball and `SHA256SUMS`, `sha256sum -c`, `tar -xzf` | 1 s |
| Model download + verify | `curl` q8_0 GGUF (357 MB), `sha256sum -c` against the HF `SHA256SUMS` | 7 s |
| Server start | `statim serve -m multilingual=…q8_0.gguf --port 8080` | model loaded in 0.47 s |
| Triage evaluation | `triage.py eval --limit 500 --concurrency 2 --seed 0` | 370 s (1.36 req/s, p50 1.46 s, p95 2.39 s) |
| Converter dependencies | `pip install numpy safetensors gguf` | a few seconds |
| Base checkpoints | `tools/fetch_models.sh multilingual english` (2.5 GB of GGUF) | 40 s |
| Configure, as documented | `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release` | fails after 2 s |
| Submodule | `git submodule update --init --recursive` | 6 s |
| Build | `cmake --build build` (52 targets) | 76 s |
| Tests | `ctest --test-dir build` | 169 s (`c5cd1c6`), 179 s (`8685d6c`) |
| Training venv | `pip install torch laya==0.3.20 datasets huggingface_hub` | 117 s, 5.9 GB |
| Published checkpoint | `hf download … --include "checkpoint/*" --include "evaluation/*"` | 11 s |
| Held-out suites | `eval_laya.py dist/…/checkpoint --n 2000 --head-max-len 512` | 19 min 10 s (AG News 145 s, Emotion 75 s, Banking77 532 s, typed-decisions 386 s, plus dataset downloads and model load) |

Installed versions from the unpinned `pip install`: torch 2.14.0, datasets 5.0.1,
huggingface_hub 1.33.0, numpy 2.4.6, gguf 0.19.0, safetensors 0.8.0.

## Problems found

Each item names what the document said, what happened, and what the fix in this pull request does.

1. **Section 1 pinned an older release.** It downloaded v0.5.0 while v0.5.2 is current. The v0.5.0
   files still exist, so the commands worked, but a reader reproducing "the current release" ran
   the wrong binary. Now v0.5.2. `examples/ticket-triage/README.md` still pointed at v0.4.0; also
   updated.
2. **Section 1 assumed a repository checkout without saying so.** The last command runs
   `examples/ticket-triage/triage.py`, but no step clones the repository. Following the block in an
   empty directory fails at that line. Following it inside a checkout leaves the tarball, the
   extracted directory and `SHA256SUMS` as untracked files in the repository root (of these, `.gitignore` covers
   only `*.gguf`). The block now starts with `git clone` and downloads into the gitignored `dist/`.
3. **Section 1 had no model checksum and no readiness wait.** The model repository publishes
   `SHA256SUMS`; the block now verifies the GGUF against it. It also waits for `/health` before the
   evaluation instead of racing the backgrounded server (harmless here because the model loads in
   0.5 s and the script downloads data first, but not guaranteed on slow disks).
4. **`ctest` fails as root.** `security` and `security_model` fail with
   `security failure: expected startup failure`. `tests/test_security.cpp:122-123` makes a key file
   unreadable with `chmod 000` and expects `load_key_file` to reject it; root reads it anyway. The
   same binaries pass as `nobody` (84 and 90 checks), and the full suite passes 10/10 as `nobody`.
   Root is the default in Docker and most cloud sandboxes, so a stranger is likely to hit this. The
   document now says so and gives the `setpriv` command. The durable fix belongs in the test (skip
   that one check when `geteuid() == 0`); this pull request only changes documentation.
5. **The expected test count was stale.** Section 2 said "8 tests on CPU; 12 with a GPU build".
   Since #12 (`server_microbatch`, `batch_parity_multilingual`) there are 10 on CPU, plus 4 for each
   of `-DSTATIM_VULKAN=ON` and `-DSTATIM_CUDA=ON`.
6. **Section 2 did not initialise the ggml submodule.** Unlike the README quick start, section 2
   has no `--recursive`. CMake stops with
   `The source directory …/third_party/ggml does not contain a CMakeLists.txt file`. Added
   `git submodule update --init --recursive`.
7. **The converter dependencies were only in a comment.** `pip install numpy safetensors gguf` is
   now a command, with a note on PEP 668 systems. (It worked here because Python 3.11 is a
   `/usr/local` build; Ubuntu's own `python3` refuses a system-wide `pip install`.)
8. **`huggingface-cli` no longer works.** The unpinned `pip install huggingface_hub` installs 1.x,
   whose `huggingface-cli` prints `Warning: huggingface-cli is deprecated and no longer works. Use hf
   instead.` and exits 1. Replaced by `hf download` in REPRODUCE.md and in the README model section.
   With the new CLI, several `--include` patterns after one flag fail (`File not found in repository
   … evaluation/%2A`); each pattern needs its own `--include`.
9. **`STATIM_GATE_DEVICE=cpu` suggests the whole gate runs on the CPU by choice.** It only sets the
   Statim server's device. `eval_dev.py` and `eval_laya.py` ask Laya for `device="cuda"` and rely on
   Laya's fallback (`CUDA requested but not available. Falling back to CPU.`). It works, but the
   warning looks like an error to a first-time reader; now documented as expected.
10. **The CPU/GPU agreement claim was too broad.** Section 3 said "GPU and CPU runs agree because the
    GPU mode is exact". That holds for Statim, not for the PyTorch reference that produces the four
    2,000-row numbers: on the CPU, AG News and Emotion moved by 1 and 4 rows. Reworded.
11. **No short path to the four headline numbers.** Section 3 only offered the full gate (54 suites,
    "several hours on a CPU"). Added the single `eval_laya.py` call that produces typed-decisions,
    Banking77, AG News and Emotion (20 minutes here), its expected output, and the fact that `--n`
    does not affect typed-decisions (always the full test split: 400 states, 2,000 decisions).
12. **The default `torch` wheel is large.** The venv is 5.9 GB on a CPU-only machine because the
    default Linux wheel bundles CUDA. Now stated.
13. **The triage evaluation rewrites committed files.** `examples/ticket-triage/results.json` and
    `results.md` are overwritten with the local CPU, URL and latency (accuracy rows identical). Now
    stated, with the `git checkout` to restore them.

Not reproduced here: the full no-harm gate (`gate.py eval` for 54 suites), section 4 (base
comparison), section 5 (speed) and section 6 (training).
