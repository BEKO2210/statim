#!/usr/bin/env python3
"""Convert a PEFT LoRA adapter (adapter_config.json + adapter_model.safetensors) for the Laya
encoder into a Statim LoRA adapter GGUF.

    python tools/convert_lora.py runs/emotion-lora -o models/emotion.lora.gguf \
        --base models/laya-multilingual-f32.gguf --category emotion

Supported targets: the ModernBERT encoder's attention (attn.Wqkv, attn.Wo) and MLP (mlp.Wi,
mlp.Wo) projections of every layer. Any other LoRA module, DoRA, trained biases or
modules_to_save are rejected: Statim cannot represent them, and silently dropping them would
serve a different model than the one that was evaluated.

File layout (statim.format = "statim-lora-v1"):
  <base tensor>.lora_a  f32, torch shape [r, in]   (ggml ne [in, r])
  <base tensor>.lora_b  f32, torch shape [out, r]  (ggml ne [r, out]), pre-multiplied by the
                        PEFT scale (lora_alpha / r, or lora_alpha / sqrt(r) with use_rslora;
                        rank_pattern / alpha_pattern are honoured per module)
so the merged weight is W' = W + lora_b @ lora_a.

--base checks every tensor shape against a Statim model GGUF and records its general.name;
the engine refuses to load the adapter onto a model with a different name.
--category (repeatable) names the question families the adapter serves in "adapter": "auto"
routing; without it the adapter name is used (see docs/API.md, "LoRA adapters").
"""
import argparse
import json
import math
import os
import re
import struct
import sys

import numpy as np
from safetensors.numpy import load_file

import gguf

TARGETS = ("attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo")
# PEFT key: [base_model.model.][encoder.|model.]layers.<i>.<module>.lora_<A|B>[.<adapter>].weight
KEY_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.(attn\.Wqkv|attn\.Wo|mlp\.Wi|mlp\.Wo)\.lora_([AB])(?:\.[^.]+)?\.weight$")
CATEGORIES = ("sentiment", "emotion", "complaint", "nli", "safety", "reading", "similarity", "topic",
              "intent", "stance", "formality", "urgency", "fact_check", "pii")


def fail(msg):
    sys.exit("convert_lora: " + msg)


def gguf_header(path):
    """(string KVs, {tensor: ggml shape}) of a GGUF file. gguf.GGUFReader decodes every array
    element (a 256k-token vocabulary takes ~15 s); this skips arrays and reads only the header."""
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with open(path, "rb") as f:
        def u(fmt):
            return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]

        def string():
            return f.read(u("Q")).decode("utf-8")

        def skip(t):
            if t == 8:
                f.seek(u("Q"), 1)
            elif t == 9:
                et, n = u("I"), u("Q")
                if et in sizes:
                    f.seek(sizes[et] * n, 1)
                else:
                    for _ in range(n):
                        skip(et)
            else:
                f.seek(sizes[t], 1)

        if f.read(4) != b"GGUF":
            fail("%s is not a GGUF file" % path)
        u("I")
        n_tensors, n_kv = u("Q"), u("Q")
        kv = {}
        for _ in range(n_kv):
            k, t = string(), u("I")
            if t == 8:
                kv[k] = string()
            else:
                skip(t)
        shapes = {}
        for _ in range(n_tensors):
            name = string()
            nd = u("I")
            shapes[name] = tuple(u("Q") for _ in range(nd))
            u("I")
            u("Q")
    return kv, shapes


def pattern_value(patterns, module_name, default):
    """PEFT rank_pattern / alpha_pattern: the first key k with re.match(r"(.*\.)?(k)$", module_name),
    as peft.utils.other.get_pattern_key does."""
    for k, v in (patterns or {}).items():
        if re.match(r"(.*\.)?(%s)$" % k, module_name):
            return v
    return default


def read_adapter(adapter_dir):
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    w_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.exists(cfg_path):
        fail("%s not found" % cfg_path)
    if not os.path.exists(w_path):
        fail("%s not found (adapter_model.bin is not supported; save with safe_serialization=True)" % w_path)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("peft_type", "LORA").upper() != "LORA":
        fail("peft_type %r is not LORA" % cfg.get("peft_type"))
    if cfg.get("use_dora"):
        fail("DoRA adapters are not supported (use_dora=true)")
    if cfg.get("bias", "none") != "none":
        fail("adapters that train biases are not supported (bias=%r)" % cfg.get("bias"))
    if cfg.get("modules_to_save"):
        fail("modules_to_save=%r: fully trained modules cannot be expressed as a LoRA delta" % cfg["modules_to_save"])
    return cfg, load_file(w_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adapter_dir")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--base", help="Statim model GGUF to check shapes against (recommended)")
    ap.add_argument("--name", help="adapter name (default: directory name)")
    ap.add_argument("--category", action="append", default=[], choices=CATEGORIES,
                    help="question family served in auto routing (repeatable)")
    a = ap.parse_args()

    cfg, weights = read_adapter(a.adapter_dir)
    r_default = int(cfg["r"])
    alpha_default = float(cfg.get("lora_alpha", r_default))
    rslora = bool(cfg.get("use_rslora", False))
    fan_in_fan_out = bool(cfg.get("fan_in_fan_out", False))

    pairs = {}
    for key, arr in weights.items():
        m = KEY_RE.search(key)
        if not m:
            fail("unsupported tensor %r: only the encoder's %s projections can carry LoRA" % (key, ", ".join(TARGETS)))
        layer, module, ab = int(m.group(1)), m.group(2), m.group(3)
        slot = pairs.setdefault((layer, module), {})
        # module name as PEFT's rank_pattern / alpha_pattern see it (without the PeftModel prefix)
        slot["name"] = key[:m.end(2)].removeprefix("base_model.model.")
        if ab in slot:
            fail("duplicate lora_%s for layers.%d.%s" % (ab, layer, module))
        slot[ab] = arr.astype(np.float32)
    if not pairs:
        fail("no LoRA tensors found")

    base_shapes, base_name = {}, None
    if a.base:
        kv, base_shapes = gguf_header(a.base)  # ggml order: (in, out)
        if kv.get("statim.format") != "statim-decision-v1":
            fail("%s is not a Statim decision model" % a.base)
        base_name = kv.get("general.name", "")

    name = a.name or os.path.basename(os.path.normpath(a.adapter_dir))
    w = gguf.GGUFWriter(a.out, "laya")
    w.add_name(name)
    w.add_type("adapter")
    w.add_string("adapter.type", "lora")
    w.add_string("statim.format", "statim-lora-v1")
    w.add_uint32("statim.lora.rank", r_default)
    w.add_float32("statim.lora.alpha", alpha_default)
    w.add_bool("statim.lora.rslora", rslora)
    w.add_array("statim.lora.categories", a.category or [name])
    if base_name is not None:
        w.add_string("statim.lora.base_name", base_name)

    total = 0
    for (layer, module) in sorted(pairs):
        slot = pairs[(layer, module)]
        if set(slot) != {"A", "B", "name"}:
            fail("layers.%d.%s has lora_%s without its partner" % (layer, module, "".join(sorted(set(slot) - {"name"}))))
        A, B = slot["A"], slot["B"]
        if fan_in_fan_out:  # Conv1D-style storage; ModernBERT uses nn.Linear, but honour the flag
            A, B = B.T.copy(), A.T.copy()
        if A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[1]:
            fail("layers.%d.%s: lora_A %s and lora_B %s do not form a rank-r product" % (layer, module, A.shape, B.shape))
        rank = A.shape[0]
        mkey = slot["name"]
        r = int(pattern_value(cfg.get("rank_pattern"), mkey, r_default))
        alpha = float(pattern_value(cfg.get("alpha_pattern"), mkey, alpha_default))
        if r != rank:
            fail("%s: tensor rank %d but adapter_config says r=%d" % (mkey, rank, r))
        scale = alpha / (math.sqrt(r) if rslora else r)
        tname = "encoder.layers.%d.%s.weight" % (layer, module)
        if a.base:
            want = base_shapes.get(tname)
            if want is None:
                fail("base model has no tensor %s" % tname)
            if want != (A.shape[1], B.shape[0]):
                fail("%s: adapter maps %d -> %d but the base weight is %d -> %d"
                     % (tname, A.shape[1], B.shape[0], want[0], want[1]))
        w.add_tensor(tname + ".lora_a", np.ascontiguousarray(A))
        w.add_tensor(tname + ".lora_b", np.ascontiguousarray(B * np.float32(scale)))
        total += A.size + B.size
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("wrote %s (%d LoRA pairs, rank %d, %.2f MB)" % (a.out, len(pairs), r_default, os.path.getsize(a.out) / 1e6))


if __name__ == "__main__":
    main()
