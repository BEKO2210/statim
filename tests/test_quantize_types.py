#!/usr/bin/env python3
"""statim-quantize accepts only the matrix types validate() accepts."""
import argparse
import subprocess
import tempfile
from pathlib import Path

# ggml enum order, matching statim::matrix_type_names().
ACCEPTED = "f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0, q2_K, q3_K, q4_K, q5_K, q6_K, bf16"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "out.gguf")

        def run(extra):
            return subprocess.run([args.binary, args.model, out, *extra], capture_output=True, text=True)

        for extra in (["iq4_nl"], ["q8_0", "--embd", "iq4_nl"], ["nope"]):
            got = run(extra)
            bad = extra[-1]
            assert got.returncode == 2, (extra, got.returncode, got.stderr)
            assert "unsupported type '%s' (accepted: %s)" % (bad, ACCEPTED) in got.stderr, got.stderr
            assert not Path(out).exists(), extra
        for extra in (["q4_0"], ["q8_0", "--embd", "q8_0"], ["q4_K"], ["f16"], ["f32"]):
            got = run(extra)
            assert got.returncode == 0, (extra, got.returncode, got.stderr)
            Path(out).unlink()
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
