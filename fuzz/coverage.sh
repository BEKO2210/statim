#!/usr/bin/env bash
# Replay fuzz inputs through an instrumented build and write a focused Statim coverage note.
#   fuzz/coverage.sh <request|tokenizer|gguf> <coverage-build-dir> <out-dir> [corpus-dir...]
set -euo pipefail

usage='usage: fuzz/coverage.sh <request|tokenizer|gguf> <coverage-build-dir> <out-dir> [corpus-dir...]'
[ "$#" -ge 3 ] || { echo "$usage" >&2; exit 2; }

name=$1
case "$name" in request|tokenizer|gguf) ;; *) echo "$usage" >&2; exit 2 ;; esac

root=$(cd "$(dirname "$0")/.." && pwd -P)
absolute_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$PWD" "$1" ;;
  esac
}

build=$(absolute_path "$2")
out=$(absolute_path "$3")
shift 3
extra=()
for dir in "$@"; do extra+=("$(absolute_path "$dir")"); done

replay="$build/fuzz_replay_$name"
[ -x "$replay" ] || { echo "coverage: replay driver not found: $replay" >&2; exit 2; }
mkdir -p "$out"

profdata_tool=${LLVM_PROFDATA:-llvm-profdata}
cov_tool=${LLVM_COV:-llvm-cov}
command -v "$profdata_tool" >/dev/null || { echo "coverage: tool not found: $profdata_tool" >&2; exit 2; }
command -v "$cov_tool" >/dev/null || { echo "coverage: tool not found: $cov_tool" >&2; exit 2; }

if [ "$name" = gguf ]; then seed="$root/fuzz/data"; else seed="$root/fuzz/seeds/$name"; fi
inputs=("$seed" "$root/fuzz/regressions/$name" "${extra[@]}")
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
raw="$tmp/coverage-$name.profraw"
data="$tmp/coverage-$name.profdata"
log="$tmp/coverage-$name.replay.log"
LLVM_PROFILE_FILE="$raw" "$replay" "${inputs[@]}" | tee "$log"
replayed=$(awk '/^replayed [0-9]+ inputs$/ { n=$2 } END { if (n == "") exit 1; print n }' "$log")
"$profdata_tool" merge -sparse "$raw" -o "$data"

sources=()
while IFS= read -r -d '' file; do sources+=("$file"); done < <(
  find "$root/src" "$root/include/statim" -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \) -print0
)
report="$out/coverage-$name.txt"
"$cov_tool" report "$replay" -instr-profile="$data" -show-region-summary \
  "${sources[@]}" > "$report"

read -r region_cov line_cov < <(
  awk '$1 == "TOTAL" { print $4, $10; found=1 } END { if (!found) exit 1 }' "$report"
)

case "$name" in
  request)
    intended=(src/security.cpp src/engine.cpp src/tokenizer.cpp include/statim/http_security.h)
    ;;
  tokenizer)
    intended=(src/tokenizer.cpp)
    ;;
  gguf)
    intended=(src/gguf_preflight.cpp src/model.cpp include/statim/mapped_file.h)
    ;;
esac

note="$out/coverage-$name.md"
{
  printf '## Fuzz coverage: `%s`\n\n' "$name"
  printf -- '- Inputs replayed: %s\n' "$replayed"
  printf -- '- Total line coverage: %s\n' "$line_cov"
  printf -- '- Total region coverage: %s\n\n' "$region_cov"
  printf '| Intended file | Line coverage |\n'
  printf '|---|---:|\n'
  for wanted in "${intended[@]}"; do
    coverage=$(awk -v wanted="$wanted" '
      $1 != "TOTAL" {
        file=$1
        gsub(/\\/, "/", file)
        if (file == wanted || (length(file) > length(wanted) && substr(file, length(file)-length(wanted)+1) == wanted)) value=$10
      }
      END { if (value == "") value="0.00% (not linked)"; print value }
    ' "$report")
    printf '| `%s` | %s |\n' "$wanted" "$coverage"
  done
} > "$note"
