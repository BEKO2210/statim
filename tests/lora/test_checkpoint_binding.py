#!/usr/bin/env python3
"""Tiny end-to-end LoRA binding: equal vectors, different matrices and checkpoint SHA-256."""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from safetensors.numpy import save_file

sys.dont_write_bytecode = True

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONVERT = os.path.join(ROOT, "tools", "convert_lora.py")


def run(cmd, ok=True):
    p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, STATIM_DEVICE="cpu"))
    if (p.returncode == 0) != ok:
        raise RuntimeError("command %s\nstdout:\n%s\nstderr:\n%s" % (cmd, p.stdout, p.stderr))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--quantize", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)

    spec = importlib.util.spec_from_file_location("make_tiny_model", os.path.join(ROOT, "fuzz", "make_tiny_model.py"))
    tiny = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tiny)

    with tempfile.TemporaryDirectory(dir=a.work) as d:
        base1, base2 = os.path.join(d, "base-1.gguf"), os.path.join(d, "base-2.gguf")
        legacy = os.path.join(d, "base-legacy.gguf")
        tiny.write(base1, "metaspace", matrix_variant=1, record_checkpoint_sha256=True)
        tiny.write(base2, "metaspace", matrix_variant=2, record_checkpoint_sha256=True)
        tiny.write(legacy, "metaspace", matrix_variant=1, record_checkpoint_sha256=False)

        adapter_dir = os.path.join(d, "adapter")
        os.makedirs(adapter_dir)
        json.dump({"peft_type": "LORA", "r": 2, "lora_alpha": 2, "bias": "none",
                   "target_modules": ["Wqkv"]}, open(os.path.join(adapter_dir, "adapter_config.json"), "w"))
        rng = np.random.default_rng(7)
        prefix = "base_model.model.encoder.layers.0.attn.Wqkv"
        save_file({prefix + ".lora_A.weight": rng.standard_normal((2, tiny.D)).astype(np.float32),
                   prefix + ".lora_B.weight": rng.standard_normal((3 * tiny.D, 2)).astype(np.float32)},
                  os.path.join(adapter_dir, "adapter_model.safetensors"))
        adapter = os.path.join(d, "adapter.gguf")
        run([sys.executable, CONVERT, adapter_dir, "-o", adapter, "--base", base1])

        info1 = json.loads(run([a.binary, "info", "-m", base1]).stdout)
        info2 = json.loads(run([a.binary, "info", "-m", base2]).stdout)
        assert info1["fingerprint"] == info2["fingerprint"]
        assert info1["checkpoint_sha256"] != info2["checkpoint_sha256"]
        run([a.binary, "info", "-m", base1, "--adapter", "tiny=" + adapter])
        refused = run([a.binary, "info", "-m", base2, "--adapter", "tiny=" + adapter], ok=False)
        assert "the vectors match but the matrices do not" in refused.stderr, refused.stderr
        run([a.binary, "info", "-m", legacy, "--adapter", "tiny=" + adapter])

        quantized = os.path.join(d, "base-q8_0.gguf")
        run([a.quantize, base1, quantized, "q8_0"])
        qinfo = json.loads(run([a.binary, "info", "-m", quantized]).stdout)
        assert qinfo["checkpoint_sha256"] == info1["checkpoint_sha256"]
        run([a.binary, "info", "-m", quantized, "--adapter", "tiny=" + adapter])

    print("PASS: tiny checkpoint SHA-256 binding and quantizer KV preservation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
