#!/usr/bin/env python3
"""tools/convert_lora.py: scaling (lora_alpha / r, rsLoRA, rank_pattern / alpha_pattern), key
formats, and every rejected input. numpy, safetensors and gguf only.

    python tests/lora/test_convert_lora.py --base models/laya-multilingual-f32.gguf
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from safetensors.numpy import save_file

import gguf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONVERT = os.path.join(ROOT, "tools", "convert_lora.py")
failures = []


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        failures.append(what)


def adapter(d, tensors, **cfg):
    os.makedirs(d, exist_ok=True)
    save_file(tensors, os.path.join(d, "adapter_model.safetensors"))
    c = {"peft_type": "LORA", "r": 2, "lora_alpha": 4, "bias": "none", "target_modules": ["Wqkv", "Wo", "Wi"]}
    c.update(cfg)
    json.dump(c, open(os.path.join(d, "adapter_config.json"), "w"))
    return d


def convert(d, out, *extra):
    return subprocess.run([sys.executable, CONVERT, d, "-o", out] + list(extra), capture_output=True, text=True)


def pair(prefix, r, n_in, n_out, seed):
    rng = np.random.default_rng(seed)
    return {prefix + ".lora_A.weight": rng.standard_normal((r, n_in)).astype(np.float32),
            prefix + ".lora_B.weight": rng.standard_normal((n_out, r)).astype(np.float32)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "out.gguf")
        wqkv = "base_model.model.encoder.layers.1.attn.Wqkv"
        wo = "base_model.model.encoder.layers.2.mlp.Wo"
        t = dict(pair(wqkv, 2, 768, 2304, 1), **pair(wo, 3, 1152, 768, 2))

        # scaling: default alpha/r, rank_pattern + alpha_pattern for layer 2 mlp.Wo
        d = adapter(os.path.join(tmp, "scale"), t, rank_pattern={"encoder.layers.2.mlp.Wo": 3},
                    alpha_pattern={r"layers\.2\.mlp\.Wo": 9, "Wo": 1})
        r = convert(d, out, "--base", a.base, "--category", "emotion", "--category", "sentiment")
        check(r.returncode == 0, "converts attention and MLP pairs with --base (%s)" % r.stderr.strip()[-200:])
        rd = gguf.GGUFReader(out)
        got = {x.name: np.array(x.data).reshape(tuple(reversed([int(s) for s in x.shape]))) for x in rd.tensors}
        check(np.array_equal(got["encoder.layers.1.attn.Wqkv.weight.lora_a"], t[wqkv + ".lora_A.weight"]), "lora_a stored as is")
        check(np.allclose(got["encoder.layers.1.attn.Wqkv.weight.lora_b"], t[wqkv + ".lora_B.weight"] * 2.0, rtol=0, atol=0),
              "lora_b pre-scaled by lora_alpha / r = 2")
        check(np.allclose(got["encoder.layers.2.mlp.Wo.weight.lora_b"], t[wo + ".lora_B.weight"] * 3.0, rtol=0, atol=0),
              "rank_pattern / alpha_pattern (full name, regex, first match wins): scale 9 / 3")
        check(rd.get_field("statim.format").contents() == "statim-lora-v1"
              and rd.get_field("statim.lora.categories").contents() == ["emotion", "sentiment"]
              and rd.get_field("statim.lora.base_name").contents() == "laya-multilingual",
              "metadata: format, categories, base name")

        d = adapter(os.path.join(tmp, "rs"), dict(pair(wqkv, 4, 768, 2304, 3)), r=4, lora_alpha=8, use_rslora=True)
        check(convert(d, out).returncode == 0, "rsLoRA converts")
        b = [np.array(x.data) for x in gguf.GGUFReader(out).tensors if x.name.endswith("lora_b")][0]
        check(np.allclose(b.reshape(2304, 4), pair(wqkv, 4, 768, 2304, 3)[wqkv + ".lora_B.weight"] * 4.0, rtol=0, atol=0),
              "rsLoRA scale lora_alpha / sqrt(r) = 4")

        # key without the base_model.model prefix and with an adapter name (PEFT in-memory state_dict)
        raw = {k.replace("base_model.model.", "").replace(".weight", ".default.weight"): v for k, v in pair(wqkv, 2, 768, 2304, 4).items()}
        check(convert(adapter(os.path.join(tmp, "keys"), raw), out).returncode == 0, "keys without prefix, with adapter name")

        # rejections
        for name, tensors, cfg, extra, needle in [
            ("dora", pair(wqkv, 2, 768, 2304, 5), {"use_dora": True}, [], "DoRA"),
            ("bias", pair(wqkv, 2, 768, 2304, 5), {"bias": "lora_only"}, [], "biases"),
            ("mts", pair(wqkv, 2, 768, 2304, 5), {"modules_to_save": ["classifier"]}, [], "modules_to_save"),
            ("head", pair("base_model.model.head.layers.0.linear1", 2, 768, 3072, 5), {}, [], "unsupported tensor"),
            ("emb", pair("base_model.model.encoder.embeddings.tok_embeddings", 2, 768, 768, 5), {}, [], "unsupported tensor"),
            ("half", {wqkv + ".lora_A.weight": np.zeros((2, 768), np.float32)}, {}, [], "without its partner"),
            ("rank", pair(wqkv, 3, 768, 2304, 5), {}, [], "adapter_config says r=2"),
            ("shape", pair(wqkv, 2, 768, 999, 5), {}, ["--base", a.base], "but the base weight is 768 -> 2304"),
            ("layer", pair("base_model.model.encoder.layers.99.attn.Wo", 2, 768, 768, 5), {}, ["--base", a.base],
             "base model has no tensor"),
            ("peft", pair(wqkv, 2, 768, 2304, 5), {"peft_type": "IA3"}, [], "not LORA"),
        ]:
            r = convert(adapter(os.path.join(tmp, name), tensors, **cfg), out, *extra)
            check(r.returncode != 0 and needle in r.stderr, "rejects %s (%s)" % (name, r.stderr.strip().splitlines()[-1] if r.stderr else "no error"))
        r = convert(os.path.join(tmp, "missing"), out)
        check(r.returncode != 0 and "not found" in r.stderr, "missing adapter directory")
        r = convert(adapter(os.path.join(tmp, "notbase"), pair(wqkv, 2, 768, 2304, 5)), out, "--base",
                    os.path.join(tmp, "scale", "adapter_config.json"))
        check(r.returncode != 0 and "not a GGUF file" in r.stderr, "--base must be a Statim model")
    print("FAIL: %d checks" % len(failures) if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
