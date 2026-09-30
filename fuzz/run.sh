#!/usr/bin/env bash
# Run one Statim fuzzer with the project's options.
#   fuzz/run.sh <request|tokenizer|gguf> <seconds> [build-dir] [extra libFuzzer args...]
# Build first:
#   cmake -S . -B build-fuzz -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DSTATIM_NATIVE=OFF \
#         -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DSTATIM_FUZZ=ON
#   cmake --build build-fuzz --target fuzz_request fuzz_tokenizer fuzz_gguf
# The corpus grows in build-dir/corpus-<name> (the committed seeds and regressions stay read-only);
# crashes land in build-dir/artifacts/. Exit status is non-zero when the fuzzer found a bug.
set -euo pipefail
cd "$(dirname "$0")/.."
name=${1:?usage: fuzz/run.sh <request|tokenizer|gguf> <seconds> [build-dir] [args...]}
secs=${2:?seconds}
build=${3:-build-fuzz}
shift $(( $# < 3 ? $# : 3 ))
work="$build/corpus-$name"
mkdir -p "$work" "$build/artifacts"
# the suppressions path is quoted: sanitizer option values may not contain spaces otherwise
export ASAN_OPTIONS=${ASAN_OPTIONS:-detect_leaks=1:abort_on_error=1}
export UBSAN_OPTIONS=${UBSAN_OPTIONS:-halt_on_error=1:print_stacktrace=1:suppressions=\"$PWD/fuzz/ubsan.supp\"}
args=(-max_total_time="$secs" -timeout=25 -rss_limit_mb=4096 -print_final_stats=1
      -artifact_prefix="$build/artifacts/$name-")
[ "$name" = request ] && args+=(-dict=fuzz/request.dict)
[ "$name" != gguf ] && args+=(-max_len=16384)
seeds="fuzz/seeds/$name"
[ "$name" = gguf ] && seeds="fuzz/data"  # the tiny valid models
exec "$build/fuzz_$name" "${args[@]}" "$@" "$work" "$seeds" "fuzz/regressions/$name"
