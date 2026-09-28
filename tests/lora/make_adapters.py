#!/usr/bin/env python3
"""Write the PEFT-style test adapters for tests/test_lora.cpp and tests/test_server_lora.py.

    python tests/lora/make_adapters.py --base models/laya-multilingual-f32.gguf --out build/lora

Creates <out>/peft-zero and <out>/peft-random (adapter_config.json + adapter_model.safetensors,
the layout PEFT's save_pretrained writes) with rank-4 LoRA on attn.Wqkv, attn.Wo, mlp.Wi and
mlp.Wo of every encoder layer. The zero adapter has lora_B = 0 (PEFT's initialisation). The
random adapter's values come from an integer hash, not a floating-point RNG, so every platform
and numpy version writes the same bytes; layer 0 mlp.Wo keeps lora_B = 0 to cover the
"exact no-op pair" path. Only numpy, safetensors and gguf are needed (no torch).
"""
import argparse
import json
import os
import sys

import numpy as np
from safetensors.numpy import save_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from convert_lora import gguf_header  # noqa: E402

TARGETS = ("attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo")
RANK, ALPHA, SIGMA = 4, 8.0, 0.03


def hashed_uniform(shape, stream):
    """Uniform values in [-1, 1) with 24 bits of precision, exact in float32 (splitmix64)."""
    n = int(np.prod(shape))
    with np.errstate(over="ignore"):
        x = np.arange(n, dtype=np.uint64) + np.uint64(stream) * np.uint64(0x9E3779B97F4A7C15)
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    u = (x >> np.uint64(40)).astype(np.float64) / float(1 << 23) - 1.0
    return u.astype(np.float32).reshape(shape)


def write(out_dir, tensors):
    os.makedirs(out_dir, exist_ok=True)
    save_file(tensors, os.path.join(out_dir, "adapter_model.safetensors"))
    cfg = {"peft_type": "LORA", "task_type": "FEATURE_EXTRACTION", "r": RANK, "lora_alpha": ALPHA,
           "lora_dropout": 0.0, "bias": "none", "use_rslora": False, "use_dora": False, "fan_in_fan_out": False,
           "target_modules": ["Wqkv", "Wo", "Wi"], "modules_to_save": None, "rank_pattern": {}, "alpha_pattern": {}}
    with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Statim model GGUF (for the projection shapes)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    shapes = {n: (sh[1], sh[0]) for n, sh in gguf_header(a.base)[1].items() if len(sh) == 2}  # torch [out, in]
    n_layer = 1 + max(int(n.split(".")[2]) for n in shapes if n.startswith("encoder.layers."))
    zero, rand = {}, {}
    stream = 1
    for l in range(n_layer):
        for t in TARGETS:
            out_f, in_f = shapes["encoder.layers.%d.%s.weight" % (l, t)]
            key = "base_model.model.encoder.layers.%d.%s" % (l, t)
            A = hashed_uniform((RANK, in_f), stream) * np.float32(SIGMA * 1.7320508)  # std SIGMA
            B = hashed_uniform((out_f, RANK), stream + 1) * np.float32(SIGMA * 1.7320508)
            stream += 2
            zero[key + ".lora_A.weight"] = A
            zero[key + ".lora_B.weight"] = np.zeros_like(B)
            rand[key + ".lora_A.weight"] = A
            rand[key + ".lora_B.weight"] = np.zeros_like(B) if (l == 0 and t == "mlp.Wo") else B
    write(os.path.join(a.out, "peft-zero"), zero)
    write(os.path.join(a.out, "peft-random"), rand)
    print("wrote %s/peft-zero and %s/peft-random (%d layers, rank %d)" % (a.out, a.out, n_layer, RANK))


if __name__ == "__main__":
    main()
