#!/usr/bin/env python3
"""Convert a PEFT LoRA adapter (adapter_config.json + adapter_model.safetensors) for the Laya
encoder into a Statim LoRA adapter GGUF.

    python tools/convert_lora.py runs/emotion-lora -o models/emotion.lora.gguf \
        --base models/laya-multilingual-f32.gguf --category emotion

Supported targets: the ModernBERT encoder's attention (attn.Wqkv, attn.Wo) and MLP (mlp.Wi,
mlp.Wo) projections of every layer, as plain LoRA (W' = W + B·A) in f32, f16 or bf16. Anything
else is rejected: LoRA on other modules, trained biases, modules_to_save, fan_in_fan_out, the LoRA
variants whose inference differs from W + B·A (DoRA, aLoRA, QALoRA, BD-LoRA, KaSA, Arrow) and
initialisations that change the base weights (PiSSA, OLoRA, CorDA, LoRA-GA, LoftQ) unless
save_pretrained converted the adapter into a plain LoRA. Statim cannot represent these, and
silently dropping them would serve a different model than the one that was evaluated. VeLoRA,
MonteCLoRA and MiCA only change training (PEFT's eval forward and merge are plain LoRA), so they
convert; their training-only tensors (lora_velora_*, lora_monteclora_*) are skipped.

File layout (statim.format = "statim-lora-v1"):
  <base tensor>.lora_a  f32, torch shape [r, in]   (ggml ne [in, r])
  <base tensor>.lora_b  f32, torch shape [out, r]  (ggml ne [r, out]), pre-multiplied by the
                        PEFT scale (lora_alpha / r, or lora_alpha / sqrt(r) with use_rslora;
                        rank_pattern / alpha_pattern are honoured per module)
so the merged weight is W' = W + lora_b @ lora_a.

--base (required) is the Statim model GGUF the adapter was trained on: every tensor shape is
checked against it, and its fingerprint (SHA-256 over its vectors: normalisation weights and biases,
see checkpoint_fingerprint), optional statim.checkpoint_sha256 content identity, and general.name
are recorded. The engine loads the adapter only onto a matching model. Both hashes are the same for
every weight type of one checkpoint; the content identity also covers matrices.
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

import hashlib

import numpy as np

import gguf

TARGETS = ("attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo")
# PEFT key: [base_model.model.][encoder.|model.]layers.<i>.<module>.lora_<A|B>[.<adapter>].weight
KEY_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.(attn\.Wqkv|attn\.Wo|mlp\.Wi|mlp\.Wo)\.lora_([AB])(?:\.[^.]+)?\.weight$")
# LoraConfig fields that select a LoRA variant whose inference is not W + B·A (checked against
# src/peft/tuners/lora/variants.py on peft main, September 2026). use_qalora pools the input. The
# other variants PEFT marks "is_lora_variant" (velora_config, monteclora_config, and
# init_lora_weights="mica") evaluate and merge as plain LoRA and are accepted.
VARIANTS = ("use_dora", "alora_invocation_tokens", "use_qalora", "use_bdlora", "kasa_config", "arrow_config")
# state that VeLoRA and MonteCLoRA keep for training only (eval forward and merge ignore it)
TRAINING_ONLY = re.compile(r"\.lora_(velora|monteclora)_")
# init_lora_weights values that change the base weights (PiSSA, OLoRA, CorDA, LoRA-GA) or replace
# them with quantized ones (LoftQ)
BASE_CHANGING_INITS = ("pissa", "olora", "corda", "lora_ga", "loftq")
CATEGORIES = ("sentiment", "emotion", "complaint", "nli", "safety", "reading", "similarity", "topic",
              "intent", "stance", "formality", "urgency", "fact_check", "pii")


def fail(msg):
    sys.exit("convert_lora: " + msg)


def gguf_header(path):
    """(string and u32 KVs, {tensor: ggml shape}, {tensor: (ggml type, absolute data offset)}) of
    a GGUF file. gguf.GGUFReader decodes every array element (a 256k-token vocabulary takes ~15 s);
    this skips arrays and reads only the header."""
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
            elif t == 4:
                kv[k] = u("I")
            else:
                skip(t)
        shapes, where = {}, {}
        for _ in range(n_tensors):
            name = string()
            nd = u("I")
            shapes[name] = tuple(u("Q") for _ in range(nd))
            where[name] = (u("I"), u("Q"))
        align = kv.get("general.alignment", 32)
        data = (f.tell() + align - 1) // align * align
        where = {name: (t, data + off) for name, (t, off) in where.items()}
    return kv, shapes, where


def checkpoint_fingerprint(path, shapes, where):
    """The engine's checkpoint fingerprint (src/model.cpp, checkpoint_fingerprint): SHA-256 over every
    vector (ggml shape with ne[1..3] = 1: normalisation weights and biases), in byte order of the
    names, each as its name and a NUL byte, the element count (u64 little-endian) and the values as
    little-endian f32."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for name in sorted((n for n, sh in shapes.items() if all(d == 1 for d in sh[1:])), key=lambda n: n.encode()):
            ttype, offset = where[name]
            n = math.prod(shapes[name])
            width = {0: 4, 1: 2, 30: 2}.get(ttype)  # ggml F32, F16, BF16
            if width is None:
                fail("%s: vector %s has ggml type %d, expected f32, f16 or bf16" % (path, name, ttype))
            f.seek(offset)
            raw = f.read(width * n)
            if len(raw) != width * n:
                fail("truncated model file %s" % path)
            if ttype == 0:
                values = np.frombuffer(raw, dtype="<f4")
            elif ttype == 1:
                values = np.frombuffer(raw, dtype="<f2").astype("<f4")
            else:
                values = (np.frombuffer(raw, dtype="<u2").astype("<u4") << 16).view("<f4")
            h.update(name.encode() + b"\0" + struct.pack("<Q", n) + values.tobytes())
    return h.hexdigest()


def load_safetensors(path):
    """{name: f32 array} of a .safetensors file (8-byte header length, JSON header, data). Read
    directly because safetensors.numpy cannot load bf16, the dtype PEFT saves an adapter in when the
    model was trained in bf16. f16 and bf16 widen to f32 exactly. Offsets are checked: in bounds,
    exactly dtype x shape bytes, no two tensors overlapping."""
    with open(path, "rb") as f:
        raw = f.read()
    n = struct.unpack("<Q", raw[:8])[0] if len(raw) >= 8 else len(raw)
    if 8 + n > len(raw):
        fail("%s is not a safetensors file" % path)
    try:
        header = json.loads(raw[8:8 + n])
    except ValueError:
        fail("%s is not a safetensors file" % path)
    if not isinstance(header, dict):
        fail("%s is not a safetensors file" % path)
    data = memoryview(raw)[8 + n:]
    width = {"F32": 4, "F16": 2, "BF16": 2}
    out, spans = {}, []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = info.get("dtype") if isinstance(info, dict) else None
        shape = info.get("shape") if isinstance(info, dict) else None
        offsets = info.get("data_offsets") if isinstance(info, dict) else None
        if dtype not in width:
            fail("%s: tensor %s is %s; LoRA factors must be F32, F16 or BF16" % (path, name, dtype))
        if not (isinstance(shape, list) and all(type(d) is int and d >= 0 for d in shape)
                and isinstance(offsets, list) and len(offsets) == 2 and all(type(o) is int for o in offsets)
                and 0 <= offsets[0] <= offsets[1] <= len(data)):
            fail("%s: tensor %s has an invalid shape or data offsets" % (path, name))
        begin, end = offsets
        if end - begin != width[dtype] * math.prod(shape):
            fail("%s: tensor %s has %d bytes, but %s %s needs %d" % (path, name, end - begin, dtype, shape,
                                                                     width[dtype] * math.prod(shape)))
        spans.append((begin, end, name))
        chunk = data[begin:end]
        if dtype == "F32":
            values = np.frombuffer(chunk, dtype="<f4")
        elif dtype == "F16":
            values = np.frombuffer(chunk, dtype="<f2").astype(np.float32)
        else:
            values = (np.frombuffer(chunk, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        out[name] = values.astype(np.float32).reshape(shape)
    spans.sort()
    for (_, end0, name0), (begin1, _, name1) in zip(spans, spans[1:]):
        if begin1 < end0:
            fail("%s: tensors %s and %s overlap" % (path, name0, name1))
    return out


def pattern_value(patterns, module_name, default):
    r"""PEFT rank_pattern / alpha_pattern: the first key k with re.match(r"(.*\.)?(k)$", module_name),
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
    for key in VARIANTS:
        if cfg.get(key):
            fail("%s=%r: this LoRA variant computes something other than W + B·A and is not supported" % (key, cfg[key]))
    init = cfg.get("init_lora_weights", True)
    if isinstance(init, str) and init.lower().startswith(BASE_CHANGING_INITS):
        fail("init_lora_weights=%r changes the base weights, so the adapter only fits that changed base; convert it "
             "into a plain LoRA with save_pretrained(..., path_initial_model_for_weight_conversion=...) "
             "(not possible for LoftQ)" % init)
    if cfg.get("fan_in_fan_out"):
        fail("fan_in_fan_out=true: PEFT sets it for Conv1D weights stored as (in, out); the encoder's projections are "
             "nn.Linear, so the adapter was not trained for this model")
    if cfg.get("bias", "none") != "none" or cfg.get("lora_bias"):
        fail("adapters that train biases are not supported (bias=%r, lora_bias=%r)" % (cfg.get("bias"), cfg.get("lora_bias")))
    for key in ("modules_to_save", "trainable_token_indices"):
        if cfg.get(key):
            fail("%s=%r: fully trained modules or tokens cannot be expressed as a LoRA delta" % (key, cfg[key]))
    for key in ("layer_replication", "target_parameters"):
        if cfg.get(key):
            fail("%s=%r changes which weights LoRA applies to in a way Statim cannot represent" % (key, cfg[key]))
    return cfg, load_safetensors(w_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adapter_dir")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--base", required=True, help="the Statim model GGUF the adapter was trained on")
    ap.add_argument("--name", help="adapter name (default: directory name)")
    ap.add_argument("--category", action="append", default=[], choices=CATEGORIES,
                    help="question family served in auto routing (repeatable)")
    a = ap.parse_args()

    cfg, weights = read_adapter(a.adapter_dir)
    r_default, alpha_default = cfg.get("r"), cfg.get("lora_alpha", cfg.get("r"))
    if type(r_default) is not int or r_default < 1:
        fail("adapter_config.json: r must be a positive integer, got %r" % (r_default,))
    if type(alpha_default) not in (int, float):
        fail("adapter_config.json: lora_alpha must be a number, got %r" % (alpha_default,))
    alpha_default = float(alpha_default)
    rslora = bool(cfg.get("use_rslora", False))

    pairs = {}
    for key, arr in weights.items():
        if TRAINING_ONLY.search(key):
            continue
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

    kv, base_shapes, where = gguf_header(a.base)  # ggml order: (in, out)
    if kv.get("statim.format") != "statim-decision-v1":
        fail("%s is not a Statim decision model" % a.base)
    base_name = kv.get("general.name", "")
    fingerprint = checkpoint_fingerprint(a.base, base_shapes, where)

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
    w.add_string("statim.lora.base_name", base_name)
    w.add_string("statim.lora.base_fingerprint", fingerprint)
    if "statim.checkpoint_sha256" in kv:
        w.add_string("statim.lora.base_checkpoint_sha256", kv["statim.checkpoint_sha256"])

    total = 0
    for (layer, module) in sorted(pairs):
        slot = pairs[(layer, module)]
        if set(slot) != {"A", "B", "name"}:
            fail("layers.%d.%s has lora_%s without its partner" % (layer, module, "".join(sorted(set(slot) - {"name"}))))
        A, B = slot["A"], slot["B"]
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
