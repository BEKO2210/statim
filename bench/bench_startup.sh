#!/usr/bin/env bash
# Cold start (process launch -> first answer) and peak RSS, Laya Python vs. Statim.
set -euo pipefail
cd "$(dirname "$0")/.."
REQ=${REQ:-bench/request_example.json}
PY=${PY:-.venv-ref/bin/python}
measure() {  # label, command...
  local label=$1; shift
  local out; out=$( { /usr/bin/time -f "%e %M" "$@" >/dev/null; } 2>&1 | tail -1 )
  echo "{\"engine\":\"$label\",\"wall_s\":${out% *},\"peak_rss_mb\":$(( ${out#* } / 1024 ))}"
}
sync
measure laya-python-fp32 $PY -c "import json,sys,laya; r=json.load(open('$REQ')); a=laya.load('$PWD/models/laya-multilingual', device='cpu'); print(a.system_one(r['state'], r['questions']))"
for w in f32 q8_0; do
  [ -f models/laya-multilingual-$w.gguf ] && measure statim-$w ./build/statim decide -m models/laya-multilingual-$w.gguf --threads 4 < "$REQ"
done
