#!/usr/bin/env python3
"""tools/convert_lora.py: scaling (lora_alpha / r, rsLoRA, rank_pattern / alpha_pattern), key
formats, f16 and bf16 factors, the base fingerprint, and every rejected input. numpy, safetensors
and gguf only.

    python tests/lora/test_convert_lora.py --base models/laya-multilingual-f32.gguf [--binary build/statim]

With --binary the recorded fingerprint is compared with the engine's (statim info).
"""
import argparse
import json
import os
import re
import struct
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


def convert(d, out, *extra, base=True):
    cmd = [sys.executable, CONVERT, d, "-o", out] + list(extra) + (["--base", BASE] if base else [])
    return subprocess.run(cmd, capture_output=True, text=True)


def save_raw(path, tensors):
    """safetensors file from {name: (dtype, shape, bytes)}, for dtypes numpy cannot write (BF16)."""
    header, blobs, offset = {}, [], 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    h = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def pair(prefix, r, n_in, n_out, seed):
    rng = np.random.default_rng(seed)
    return {prefix + ".lora_A.weight": rng.standard_normal((r, n_in)).astype(np.float32),
            prefix + ".lora_B.weight": rng.standard_normal((n_out, r)).astype(np.float32)}


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--binary", help="statim executable: compare the fingerprint with the engine's")
    a = ap.parse_args()
    BASE = a.base
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "out.gguf")
        wqkv = "base_model.model.encoder.layers.1.attn.Wqkv"
        wo = "base_model.model.encoder.layers.2.mlp.Wo"
        t = dict(pair(wqkv, 2, 768, 2304, 1), **pair(wo, 3, 1152, 768, 2))

        # scaling: default alpha/r, rank_pattern + alpha_pattern for layer 2 mlp.Wo
        d = adapter(os.path.join(tmp, "scale"), t, rank_pattern={"encoder.layers.2.mlp.Wo": 3},
                    alpha_pattern={r"layers\.2\.mlp\.Wo": 9, "Wo": 1})
        r = convert(d, out, "--category", "emotion", "--category", "sentiment")
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
        fp = rd.get_field("statim.lora.base_fingerprint").contents()
        check(len(fp) == 64 and all(c in "0123456789abcdef" for c in fp), "metadata: base fingerprint (%s...)" % fp[:16])
        if a.binary:
            info = json.loads(subprocess.run([a.binary, "info", "-m", a.base], capture_output=True, text=True,
                                             env=dict(os.environ, STATIM_DEVICE="cpu"), check=True).stdout)
            check(info["fingerprint"] == fp, "the recorded fingerprint is the engine's (statim info: %s...)" % info["fingerprint"][:16])

        # f16 and bf16 factors widen to f32 exactly (safetensors.numpy cannot read bf16)
        t16 = {k: v.astype(np.float16) for k, v in pair(wqkv, 2, 768, 2304, 6).items()}
        d = adapter(os.path.join(tmp, "f16"), t16)
        check(convert(d, out).returncode == 0 and np.array_equal(
            [np.array(x.data) for x in gguf.GGUFReader(out).tensors if x.name.endswith("lora_a")][0].reshape(2, 768),
            t16[wqkv + ".lora_A.weight"].astype(np.float32)), "f16 factors convert exactly")
        tb = {k: (v.view(np.uint32) >> 16).astype(np.uint16) for k, v in pair(wqkv, 2, 768, 2304, 7).items()}
        d = adapter(os.path.join(tmp, "bf16"), {})
        save_raw(os.path.join(d, "adapter_model.safetensors"), {k: ("BF16", v.shape, v.tobytes()) for k, v in tb.items()})
        want = (tb[wqkv + ".lora_A.weight"].astype(np.uint32) << 16).view(np.float32)
        r = convert(d, out)
        check(r.returncode == 0 and np.array_equal(
            [np.array(x.data) for x in gguf.GGUFReader(out).tensors if x.name.endswith("lora_a")][0].reshape(2, 768), want),
            "bf16 factors convert exactly (%s)" % r.stderr.strip()[-120:])
        for init in ("gaussian", "eva", True, "mica"):
            r = convert(adapter(os.path.join(tmp, "init-%s" % init), pair(wqkv, 2, 768, 2304, 8), init_lora_weights=init), out)
            check(r.returncode == 0, "init_lora_weights=%r (base weights unchanged, plain LoRA at inference) converts" % (init,))
        # VeLoRA and MonteCLoRA evaluate as plain LoRA; their training-only state is skipped
        for name, cfg, extra in [("velora", {"velora_config": {"num_groups": 32}}, ".lora_velora_embed"),
                                 ("monteclora", {"monteclora_config": {"num_samples": 4}}, ".lora_monteclora_sampler.std_prior")]:
            t = dict(pair(wqkv, 2, 768, 2304, 9))
            t[wqkv + extra] = np.ones(24, np.float32)
            r = convert(adapter(os.path.join(tmp, name), t, **cfg), out)
            names = sorted(x.name for x in gguf.GGUFReader(out).tensors) if r.returncode == 0 else []
            check(names == ["encoder.layers.1.attn.Wqkv.weight.lora_a", "encoder.layers.1.attn.Wqkv.weight.lora_b"],
                  "%s converts as plain LoRA, training-only state skipped (%s)" % (name, r.stderr.strip()[-120:]))

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
            ("shape", pair(wqkv, 2, 768, 999, 5), {}, [], "but the base weight is 768 -> 2304"),
            ("layer", pair("base_model.model.encoder.layers.99.attn.Wo", 2, 768, 768, 5), {}, [], "base model has no tensor"),
            ("peft", pair(wqkv, 2, 768, 2304, 5), {"peft_type": "IA3"}, [], "not LORA"),
            ("fifo", pair(wqkv, 2, 768, 2304, 5), {"fan_in_fan_out": True}, [], "fan_in_fan_out=true"),
            ("alora", pair(wqkv, 2, 768, 2304, 5), {"alora_invocation_tokens": [5, 6]}, [], "alora_invocation_tokens"),
            ("qalora", pair(wqkv, 2, 768, 2304, 5), {"use_qalora": True}, [], "use_qalora"),
            ("bdlora", pair(wqkv, 2, 768, 2304, 5), {"use_bdlora": {"nblocks": 2}}, [], "use_bdlora"),
            ("kasa", pair(wqkv, 2, 768, 2304, 5), {"kasa_config": {"beta": 0.1}}, [], "kasa_config"),
            ("arrow", pair(wqkv, 2, 768, 2304, 5), {"arrow_config": {"top_k": 2}}, [], "arrow_config"),
            ("nor", pair(wqkv, 2, 768, 2304, 5), {"r": None}, [], "r must be a positive integer"),
            ("strr", pair(wqkv, 2, 768, 2304, 5), {"r": "2"}, [], "r must be a positive integer"),
            ("pissa", pair(wqkv, 2, 768, 2304, 5), {"init_lora_weights": "pissa_niter_4"}, [], "changes the base weights"),
            ("olora", pair(wqkv, 2, 768, 2304, 5), {"init_lora_weights": "olora"}, [], "changes the base weights"),
            ("corda", pair(wqkv, 2, 768, 2304, 5), {"init_lora_weights": "corda"}, [], "changes the base weights"),
            ("loraga", pair(wqkv, 2, 768, 2304, 5), {"init_lora_weights": "lora_ga"}, [], "changes the base weights"),
            ("loftq", pair(wqkv, 2, 768, 2304, 5), {"init_lora_weights": "loftq"}, [], "changes the base weights"),
            ("lorabias", pair(wqkv, 2, 768, 2304, 5), {"lora_bias": True}, [], "biases"),
            ("tokens", pair(wqkv, 2, 768, 2304, 5), {"trainable_token_indices": [5]}, [], "trainable_token_indices"),
            ("replicate", pair(wqkv, 2, 768, 2304, 5), {"layer_replication": [[0, 4]]}, [], "layer_replication"),
            ("params", pair(wqkv, 2, 768, 2304, 5), {"target_parameters": ["mlp.Wi.weight"]}, [], "target_parameters"),
            ("f64", {k: v.astype(np.float64) for k, v in pair(wqkv, 2, 768, 2304, 5).items()}, {}, [], "must be F32, F16 or BF16"),
        ]:
            r = convert(adapter(os.path.join(tmp, name), tensors, **cfg), out, *extra)
            check(r.returncode != 0 and needle in r.stderr, "rejects %s (%s)" % (name, r.stderr.strip().splitlines()[-1] if r.stderr else "no error"))
        # malformed safetensors headers
        a16 = np.ones((2, 768), np.float32).tobytes()
        for name, entries, payload, needle in [
            ("negative", {wqkv + ".lora_A.weight": ("F32", [2, 768], [-8, len(a16) - 8])}, a16, "invalid shape or data offsets"),
            ("beyond", {wqkv + ".lora_A.weight": ("F32", [2, 768], [0, len(a16) + 4])}, a16, "invalid shape or data offsets"),
            ("short", {wqkv + ".lora_A.weight": ("F32", [2, 768], [0, len(a16) - 4])}, a16, "needs 6144"),
            ("overlap", {wqkv + ".lora_A.weight": ("F32", [2, 768], [0, len(a16)]),
                         wqkv + ".lora_B.weight": ("F32", [2, 768], [4, len(a16) + 4])}, a16 + b"\0" * 4, "overlap"),
        ]:
            d = adapter(os.path.join(tmp, name), {})
            h = json.dumps({k: {"dtype": t, "shape": sh, "data_offsets": off} for k, (t, sh, off) in entries.items()}).encode()
            with open(os.path.join(d, "adapter_model.safetensors"), "wb") as f:
                f.write(struct.pack("<Q", len(h)) + h + payload)
            r = convert(d, out)
            check(r.returncode != 0 and needle in r.stderr, "rejects safetensors with %s offsets (%s)" % (
                name, r.stderr.strip().splitlines()[-1] if r.stderr else "no error"))
        r = convert(os.path.join(tmp, "missing"), out)
        check(r.returncode != 0 and "not found" in r.stderr, "missing adapter directory")
        r = convert(adapter(os.path.join(tmp, "notbase"), pair(wqkv, 2, 768, 2304, 5)), out, "--base",
                    os.path.join(tmp, "scale", "adapter_config.json"), base=False)
        check(r.returncode != 0 and "not a GGUF file" in r.stderr, "--base must be a Statim model")
        r = convert(adapter(os.path.join(tmp, "nobase"), pair(wqkv, 2, 768, 2304, 5)), out, base=False)
        check(r.returncode != 0 and "--base" in r.stderr, "--base is required")

    # the auto-routing families exist three times: engine, converter, the gate's category suites
    src = open(os.path.join(ROOT, "src", "engine.cpp"), encoding="utf-8").read()
    block = src[src.index("kFamilies = {"):]
    engine = re.findall(r'\{"(\w+)", \{', block[:block.index("};")])
    bench = open(os.path.join(ROOT, "bench", "eval_categories.py"), encoding="utf-8").read()
    held = bench[bench.index("HELD_OUT = {"):]
    suites = re.findall(r'^    "(\w+)": \[', held[:held.index("\n}\n")], re.M)
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import convert_lora
    check(len(engine) == 14 and engine == list(convert_lora.CATEGORIES) == suites,
          "the 14 families agree in src/engine.cpp, tools/convert_lora.py and bench/eval_categories.py")
    print("FAIL: %d checks" % len(failures) if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
