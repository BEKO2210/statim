# Fuzzing

Three [libFuzzer](https://llvm.org/docs/LibFuzzer.html) harnesses cover everything an untrusted
party can hand the server, plus the model files an operator loads:

| Harness | Input | What runs | A bug is |
|---|---|---|---|
| `fuzz_request` | raw `POST /v1/systemone` / `/batch` body | `parse_decide_request()` (the handler's own parsing and validation), `check_work()`, question validation, tokenization, prompt packing, a real forward pass on two tiny models (small requests), answer decoding, consensus fusion, response serialization. Each input runs as a single and as a batch request against a Metaspace and a ByteLevel model. | any exception other than `HttpError` 400/413/422 or `QuestionError` (the server would answer 500), a crash, a sanitizer report, a leak, a hang |
| `fuzz_tokenizer` | first byte: truncation (`< 128`: none, else 1–64 tokens); rest: text, any bytes | `Tokenizer::encode` of both tiny tokenizers and the real Laya tokenizers (when `models/*/tokenizer/tokenizer.json` exist or `$STATIM_FUZZ_TOKENIZERS` lists them) | a throw, an id outside the vocabulary, a truncated encoding that is not a prefix of the full one, a crash, a sanitizer report, a hang |
| `fuzz_gguf` | a whole model file | `Model::load`; if the file is accepted, an `Engine` and two requests (choice, score, noul; ensemble; calibration) | a crash or abort (`GGML_ASSERT`), a sanitizer report, a leak, a hang. Rejecting with an exception is the expected outcome. |

Everything, vendored ggml included, is built with AddressSanitizer and UndefinedBehaviorSanitizer
(`-fno-sanitize-recover`). Coverage feedback is limited to `statim_core` and `statim_tokenizer`, so
ggml's kernels run fast enough for real forward passes per input while memory errors that our
inputs cause inside ggml are still reported.

## Run

```bash
sudo apt-get install clang libclang-rt-18-dev ninja-build        # Ubuntu 24.04
cmake -S . -B build-fuzz -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DSTATIM_NATIVE=OFF \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DSTATIM_FUZZ=ON
cmake --build build-fuzz --target fuzz_request fuzz_tokenizer fuzz_gguf
fuzz/run.sh request 1800          # seconds; also: tokenizer, gguf
```

`fuzz/run.sh` sets the sanitizer options (`halt_on_error=1`, `detect_leaks=1`, the suppressions
below), the request dictionary, a 25 s per-input timeout and a 4 GiB RSS cap. New corpus entries go
to `build-fuzz/corpus-<name>/`; crash inputs to `build-fuzz/artifacts/`. CI runs every harness for
60 s on each push (`.github/workflows/ci.yml`, job `fuzz`).

Reproduce a crash with `build-fuzz/fuzz_<name> <file>`, or without sanitizers through the replay
driver of an ordinary build: `build/fuzz_replay_<name> <file-or-dir>...`.

## Layout

| Path | Contents |
|---|---|
| `fuzz_*.cpp` | the harnesses (`LLVMFuzzerTestOneInput`) |
| `replay_main.cpp` | runs a harness over files without libFuzzer; ctest uses it |
| `corpus/<name>/` | minimised seed corpus (`-set_cover_merge=1` of the long runs), replayed by ctest `fuzz_regressions_<name>` |
| `regressions/<name>/` | every input that once crashed, named after the bug; also replayed by ctest |
| `seeds/<name>/` | the hand-written starting inputs the corpus grew from |
| `data/tiny-*.gguf` | two tiny deterministic models (16-wide, 2 layers; Metaspace and ByteLevel BPE), written by `make_tiny_model.py` |
| `request.dict` | libFuzzer dictionary of request field names and values |
| `ubsan.supp` | UBSan suppressions — confirmed upstream ggml issues only, one line each with the reason |

Regenerate the tiny models with `python3 fuzz/make_tiny_model.py fuzz/data` (needs `numpy` and
`gguf`); they are deterministic. When a fuzzer finds a bug: fix it, copy the input to
`regressions/<name>/<what-it-was>.<ext>`, and add a focused unit test where the invariant is
semantic (e.g. `tests/test_model_validation.cpp` for model files).

## Minimise the corpus

```bash
for t in request tokenizer gguf; do
  mkdir -p /tmp/min-$t
  build-fuzz/fuzz_$t -set_cover_merge=1 -max_len=16384 /tmp/min-$t fuzz/corpus/$t build-fuzz/corpus-$t
  rm -rf fuzz/corpus/$t && mv /tmp/min-$t fuzz/corpus/$t
done
cp fuzz/data/tiny-*.gguf fuzz/corpus/gguf/   # keep the two valid models as seeds
```

The GGUF corpus is ~15 MB on disk but its files are near-duplicates of the two tiny models, so
Git's delta compression stores it in well under 1 MB.
