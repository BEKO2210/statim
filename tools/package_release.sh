#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 BUILD_DIR cpu|vulkan VERSION [OUTPUT_DIR]" >&2
  exit 2
fi

build_dir=$1
flavor=$2
version=${3#v}
output_dir=${4:-dist}

if [[ -z "$version" || ! "$version" =~ ^[0-9A-Za-z._+-]+$ ]]; then
  echo "error: version must be a filesystem-safe tag such as v1.0.0" >&2
  exit 2
fi

case "$flavor" in
  cpu|vulkan) ;;
  *) echo "error: flavor must be cpu or vulkan" >&2; exit 2 ;;
esac

for binary in statim statim-quantize; do
  if [[ ! -x "$build_dir/$binary" ]]; then
    echo "error: missing executable $build_dir/$binary" >&2
    exit 1
  fi
done

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
archive_root="statim-${version}-linux-x86_64-${flavor}"
staging=$(mktemp -d)
trap 'rm -rf -- "$staging"' EXIT

mkdir -p "$staging/$archive_root/deploy" "$output_dir"
install -m 0755 "$build_dir/statim" "$build_dir/statim-quantize" "$staging/$archive_root/"
for file in LICENSE NOTICE LICENSE-MODEL.md COMMERCIAL.md README.md; do
  install -m 0644 "$repo_root/$file" "$staging/$archive_root/$file"
done
install -m 0644 "$repo_root/deploy/statim.service" "$staging/$archive_root/deploy/statim.service"

tar --sort=name \
  --mtime="@${SOURCE_DATE_EPOCH:-0}" \
  --owner=0 --group=0 --numeric-owner \
  -C "$staging" -czf "$output_dir/$archive_root.tar.gz" "$archive_root"
