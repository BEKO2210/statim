#!/usr/bin/env python3
"""Measure grounded generation + independent verification at 1/2/4/8 concurrency.

Start Ollama with ``OLLAMA_NUM_PARALLEL=8`` (or at least the largest tested
slot count). The benchmark does not start or stop the server.
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def vram():
    cmd = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
           "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, check=True, timeout=20)
        return result.stdout.strip() or "no compute processes"
    except (OSError, subprocess.SubprocessError) as exc:
        return "nvidia-smi unavailable: %s" % exc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", type=int, default=24, help="kept items per capability at each concurrency")
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--num-gpu", type=int, default=-1)
    ap.add_argument("--eval-cache", type=Path, default=None)
    ap.add_argument("--s1bench-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.items < 1:
        ap.error("--items must be positive")
    print("VRAM before:\n%s" % vram(), flush=True)
    results = []
    with tempfile.TemporaryDirectory(prefix="statim-synth-throughput-") as tmp:
        for concurrency in (1, 2, 4, 8):
            out = Path(tmp) / ("c%d" % concurrency)
            cmd = [sys.executable, str(HERE / "grounded.py"), "--out-dir", str(out),
                   "--per-capability", str(args.items), "--concurrency", str(concurrency),
                   "--host", args.host, "--num-gpu", str(args.num_gpu)]
            if args.cache:
                cmd.extend(("--cache", str(args.cache)))
            if args.eval_cache:
                cmd.extend(("--eval-cache", str(args.eval_cache)))
            if args.s1bench_dir:
                cmd.extend(("--s1bench-dir", str(args.s1bench_dir)))
            start = time.monotonic()
            subprocess.run(cmd, check=True)
            elapsed = time.monotonic() - start
            manifest = json.loads((out / "manifest.json").read_text())
            kept = manifest["items_per_capability"] * 2
            row = {"concurrency": concurrency, "kept": kept, "seconds": round(elapsed, 3),
                   "items_per_hour": round(kept * 3600 / elapsed, 2), "vram": vram()}
            results.append(row)
            print(json.dumps(row), flush=True)
    print("\nconcurrency\tkept\tseconds\titems/h")
    for row in results:
        print("{concurrency}\t{kept}\t{seconds}\t{items_per_hour}".format(**row))
    print("\nVRAM after:\n%s" % vram())


if __name__ == "__main__":
    main()
