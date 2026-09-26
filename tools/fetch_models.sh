#!/usr/bin/env bash
# Download the Laya checkpoints from Hugging Face and convert them to Statim GGUF (f32 reference).
#   tools/fetch_models.sh [multilingual|english|typed-decisions]...   (default: multilingual english)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
HF=https://huggingface.co/convaiinnovations
models=("${@:-multilingual english}")
[ $# -eq 0 ] && models=(multilingual english)
for m in "${models[@]}"; do
  case $m in
    multilingual) repo=laya-multilingual; dir=models/laya-multilingual; out=models/laya-multilingual ;;
    english) repo=laya; dir=models/laya; out=models/laya-english ;;
    typed-decisions) repo=laya-typed-decisions; dir=models/laya-typed-decisions; out=models/laya-typed-decisions ;;
    *) echo "unknown model $m" >&2; exit 2 ;;
  esac
  mkdir -p "$dir/encoder" "$dir/tokenizer"
  for f in model.safetensors rl_agent_config.json encoder/config.json tokenizer/tokenizer.json tokenizer/tokenizer_config.json; do
    [ -s "$dir/$f" ] || curl -fL --retry 3 -o "$dir/$f" "$HF/$repo/resolve/main/$f"
  done
  [ -s "$out-f32.gguf" ] || $PY tools/convert_laya.py "$dir" -o "$out-f32.gguf" --type f32 --embd-type f16 --name "$repo"
done
