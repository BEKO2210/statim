#!/usr/bin/env bash
# Download the Laya checkpoints from Hugging Face and convert them to Statim GGUF (f32 reference).
#   tools/fetch_models.sh [multilingual|english|typed-decisions]...   (default: multilingual english)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
HF=https://huggingface.co/convaiinnovations

verify_file() {
  printf '%s  %s\n' "$2" "$1" | sha256sum -c - >/dev/null
}

fetch_file() {
  local path=$1 url=$2 sha256=$3
  if [ -s "$path" ] && verify_file "$path" "$sha256"; then
    return
  fi
  if [ -e "$path" ]; then
    echo "checksum mismatch for cached $path; downloading it again" >&2
  fi
  curl -fL --retry 3 -o "$path" "$url"
  verify_file "$path" "$sha256"
}

models=("${@:-multilingual english}")
[ $# -eq 0 ] && models=(multilingual english)
for m in "${models[@]}"; do
  case $m in
    multilingual)
      repo=laya-multilingual; dir=models/laya-multilingual; out=models/laya-multilingual
      revision=e4e9ddf21a7b1903b7acffd8814ad4307bf63a67
      hashes=(9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204 25061739243b617ad88d1219ba6f8a9c86c5881ca28df024fa2d9b3b2fcc30c6 83f6916d13ef0f556ac461f28308dc2bffa7ebeadee8ec9e2db5812020ea5bb4 609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f 424b69444bf7b5809dc2cd2e36d0bd71b8055124dd24274d6db3c655d38205e7)
      ;;
    english)
      repo=laya; dir=models/laya; out=models/laya-english
      revision=55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851
      hashes=(891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c ae287b56bbcf5f8c4f4541ae9dfd00c914c4c48b940b8398c3058af37ba92bbd bf3ab80598fdccf414855a2ce80f22859e4492d06ca8a62ddd1cfb63972f8979 6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30 50044de60daaa73df97d262e15a40d4faf0160e7d742df64b377877a1320dd12)
      ;;
    typed-decisions)
      repo=laya-typed-decisions; dir=models/laya-typed-decisions; out=models/laya-typed-decisions
      revision=1a793eb568e6718f15941d08f85432581df534e3
      hashes=(4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e ebf0cd524d92342a6be5e48e9fca3d7c2babfb5a56ccd79d2171ef5d8c7f7be8 5268d24ad3b77c8151de5dcb0762ba4391619aad9ab0bda33e36fb083cfeae6d 6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30 08d4cf3ac4dca381759441b85b91a6d40e688471dcd33d15d6649eb0a9a854d1)
      ;;
    *) echo "unknown model $m" >&2; exit 2 ;;
  esac
  mkdir -p "$dir/encoder" "$dir/tokenizer"
  files=(model.safetensors rl_agent_config.json encoder/config.json tokenizer/tokenizer.json tokenizer/tokenizer_config.json)
  i=0
  for f in "${files[@]}"; do
    fetch_file "$dir/$f" "$HF/$repo/resolve/$revision/$f" "${hashes[$i]}"
    i=$((i + 1))
  done
  [ -s "$out-f32.gguf" ] || $PY tools/convert_laya.py "$dir" -o "$out-f32.gguf" --type f32 --embd-type f16 --name "$repo"
done
