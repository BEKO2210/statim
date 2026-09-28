#!/usr/bin/env python3
"""Write tiny, deterministic Statim decision models for the fuzzers.

The files have the exact metadata keys and tensor layout of tools/convert_laya.py output,
but a 16-wide, 2-layer encoder and a toy BPE vocabulary, so one forward pass takes
microseconds. They are the GGUF fuzzer's seeds and the request fuzzer's runtime model.

    python3 fuzz/make_tiny_model.py fuzz/data
"""
import json
import os
import sys

import numpy as np
from gguf import GGUFWriter

D, N_HEAD, N_FF, N_LAYER = 16, 2, 16, 2
HEAD_LAYERS, HEAD_N_HEAD, HEAD_FF, ACT_HIDDEN, N_ACT = 2, 2, 32, 8, 2
SPECIALS = ["<pad>", "<sep>", "<cls>", "<unk>", "<mask>"]


def byte_level_alphabet():
    # GPT-2 bytes_to_unicode()
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return [chr(c) for c in cs]


def tokenizer(kind):
    if kind == "metaspace":
        pieces = SPECIALS + ["<0x%02X>" % b for b in range(256)] + ["▁"] + list("abcdefghijklmnopqrstuvwxyz")
        merges = ["▁ t", "t h", "▁t h", "▁th e", "a n", "an d", "▁ a", "▁a nd", "i n", "▁ in"]
        normalizer = {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}
        pre = {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": True}
        model = {"type": "BPE", "dropout": None, "unk_token": "<unk>", "continuing_subword_prefix": None,
                 "end_of_word_suffix": None, "fuse_unk": True, "byte_fallback": True, "ignore_merges": False}
    else:
        pieces = SPECIALS + byte_level_alphabet()
        merges = ["Ġ t", "t h", "Ġt h", "Ġth e", "a n", "an d", "Ġ a", "Ġa nd", "i n", "Ġ in"]
        normalizer = {"type": "NFC"}
        pre = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True}
        model = {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "",
                 "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False, "ignore_merges": False}
    for m in merges:  # every merge's parts and result must be pieces
        a, b = m.split(" ")
        for piece in (a, b, a + b):
            if piece not in pieces:
                pieces.append(piece)
    added = [{"id": i, "content": s, "single_word": False, "lstrip": s == "<mask>", "rstrip": False,
              "normalized": False, "special": True} for i, s in enumerate(SPECIALS)]
    return pieces, merges, normalizer, pre, model, added


def write(path, kind):
    rng = np.random.default_rng(1234 if kind == "metaspace" else 5678)
    pieces, merges, normalizer, pre, model, added = tokenizer(kind)
    w = GGUFWriter(path, "laya")
    w.add_name("tiny-" + kind)
    w.add_string("statim.format", "statim-decision-v1")
    w.add_uint32("laya.encoder.n_embd", D)
    w.add_uint32("laya.encoder.n_layer", N_LAYER)
    w.add_uint32("laya.encoder.n_head", N_HEAD)
    w.add_uint32("laya.encoder.n_ff", N_FF)
    w.add_uint32("laya.encoder.local_window", 16)
    w.add_array("laya.encoder.layer_is_global", [i % 2 == 0 for i in range(N_LAYER)])
    w.add_float32("laya.encoder.rope_theta_global", 160000.0)
    w.add_float32("laya.encoder.rope_theta_local", 10000.0)
    w.add_float32("laya.encoder.norm_eps", 1e-5)
    w.add_string("laya.encoder.activation", "gelu")
    w.add_uint32("laya.encoder.max_position", 512)
    w.add_uint32("laya.head.n_layer", HEAD_LAYERS)
    w.add_uint32("laya.head.n_head", HEAD_N_HEAD)
    w.add_uint32("laya.head.n_ff", HEAD_FF)
    w.add_uint32("laya.head.n_act", N_ACT)
    w.add_uint32("laya.max_len", 128)
    w.add_uint32("laya.head_max_len", 64)
    w.add_array("laya.temperature", [1.0, 1.5, 0.8])
    w.add_string("laya.temperature_by_options", json.dumps({"choice:2": 1.2}))
    w.add_string("laya.lang_temperatures", json.dumps({"de": {"temperature": [1.1, 1.0, 1.0], "temperature_by_options": {"noul:2": 0.9}}}))
    w.add_array("tokenizer.statim.tokens", pieces)
    w.add_array("tokenizer.statim.merges", merges)
    w.add_string("tokenizer.statim.normalizer", json.dumps(normalizer, ensure_ascii=False))
    w.add_string("tokenizer.statim.pre_tokenizer", json.dumps(pre, ensure_ascii=False))
    w.add_string("tokenizer.statim.added_tokens", json.dumps(added, ensure_ascii=False))
    w.add_string("tokenizer.statim.model", json.dumps(model))
    w.add_int32("tokenizer.statim.cls_id", 2)
    w.add_int32("tokenizer.statim.sep_id", 1)
    w.add_int32("tokenizer.statim.mask_id", 4)
    w.add_int32("tokenizer.statim.pad_id", 0)
    w.add_string("tokenizer.statim.mask_token", "<mask>")

    def t(name, *shape):
        w.add_tensor(name, (rng.standard_normal(shape) * 0.2).astype(np.float32))

    V = len(pieces)
    w.add_tensor("encoder.embeddings.tok_embeddings.weight", (rng.standard_normal((V, D)) * 0.2).astype(np.float16))
    t("encoder.embeddings.norm.weight", D)
    for l in range(N_LAYER):
        p = "encoder.layers.%d." % l
        if l:
            t(p + "attn_norm.weight", D)
        t(p + "attn.Wqkv.weight", 3 * D, D)
        t(p + "attn.Wo.weight", D, D)
        t(p + "mlp_norm.weight", D)
        t(p + "mlp.Wi.weight", 2 * N_FF, D)
        t(p + "mlp.Wo.weight", D, N_FF)
    t("encoder.final_norm.weight", D)
    t("type_emb.weight", 3, D)
    for l in range(HEAD_LAYERS):
        p = "head.layers.%d." % l
        t(p + "norm1.weight", D)
        t(p + "norm1.bias", D)
        t(p + "self_attn.in_proj_weight", 3 * D, D)
        t(p + "self_attn.in_proj_bias", 3 * D)
        t(p + "self_attn.out_proj.weight", D, D)
        t(p + "self_attn.out_proj.bias", D)
        t(p + "norm2.weight", D)
        t(p + "norm2.bias", D)
        t(p + "linear1.weight", HEAD_FF, D)
        t(p + "linear1.bias", HEAD_FF)
        t(p + "linear2.weight", D, HEAD_FF)
        t(p + "linear2.bias", D)
    t("scorer.0.weight", D)
    t("scorer.0.bias", D)
    t("scorer.1.weight", D, D)
    t("scorer.1.bias", D)
    t("scorer.3.weight", 1, D)
    t("scorer.3.bias", 1)
    t("act_head.0.weight", ACT_HIDDEN, D + 4)
    t("act_head.0.bias", ACT_HIDDEN)
    t("act_head.2.weight", N_ACT, ACT_HIDDEN)
    t("act_head.2.bias", N_ACT)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "fuzz/data"
    os.makedirs(out, exist_ok=True)
    for kind in ("metaspace", "bytelevel"):
        write(os.path.join(out, "tiny-%s.gguf" % kind), kind)
