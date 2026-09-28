#!/usr/bin/env python3
"""ctest fixture: write the PEFT test adapters and convert them with tools/convert_lora.py.

    python tests/lora/build_fixtures.py --base models/laya-multilingual-f32.gguf --out build/lora

Outputs <out>/zero.gguf (category emotion) and <out>/random.gguf (category sentiment).
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    py = sys.executable
    subprocess.check_call([py, os.path.join(HERE, "make_adapters.py"), "--base", a.base, "--out", a.out])
    for name, category in (("zero", "emotion"), ("random", "sentiment")):
        subprocess.check_call([py, os.path.join(ROOT, "tools", "convert_lora.py"), os.path.join(a.out, "peft-" + name),
                               "-o", os.path.join(a.out, name + ".gguf"), "--base", a.base, "--name", name,
                               "--category", category])


if __name__ == "__main__":
    main()
