#!/usr/bin/env python3
"""Convert a Laya checkpoint directory (model.safetensors + rl_agent_config.json + encoder/ +
tokenizer/) into one self-contained Statim GGUF file. The output records
statim.checkpoint_sha256, a SHA-256 over every source tensor as little-endian f32, so all output
weight types made from the same checkpoint have the same content identity.

    python tools/convert_laya.py models/laya-multilingual -o models/laya-multilingual-q8_0.gguf --type q8_0

--type: f32 | f16 | q8_0 | q4_0 (applies to large 2-D matmul weights; norms, biases and small
tensors stay f32). The token embedding follows --embd-type (default: same as --type).
"""
import argparse
import hashlib
import json
import os
import struct
import sys

import numpy as np
from safetensors.numpy import load_file

import gguf
from gguf import GGMLQuantizationType as QT
from gguf import quants

ARCH = "laya"
QMAP = {"f32": QT.F32, "f16": QT.F16, "q8_0": QT.Q8_0, "q4_0": QT.Q4_0, "q5_0": QT.Q5_0}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def rope_thetas(ecfg):
    rp = ecfg.get("rope_parameters")
    if isinstance(rp, dict) and "full_attention" in rp:
        return float(rp["full_attention"]["rope_theta"]), float(rp["sliding_attention"]["rope_theta"])
    # transformers 4.x attribute names and ModernBERT defaults
    return float(ecfg.get("global_rope_theta", 160000.0)), float(ecfg.get("local_rope_theta", 10000.0))


def tokenizer_meta(w, tok_dir):
    tj = load_json(os.path.join(tok_dir, "tokenizer.json"))
    tc = load_json(os.path.join(tok_dir, "tokenizer_config.json"))
    model = tj["model"]
    assert model["type"] == "BPE", "only BPE tokenizers are supported"
    vocab = model["vocab"]
    added = tj.get("added_tokens", [])
    n = max(max(vocab.values()), max((a["id"] for a in added), default=-1)) + 1
    tokens = [""] * n
    for piece, i in vocab.items():
        tokens[i] = piece
    for a in added:
        tokens[a["id"]] = a["content"]
    merges = [m if isinstance(m, str) else "%s %s" % (m[0], m[1]) for m in model.get("merges", [])]
    w.add_array("tokenizer.statim.tokens", tokens)
    w.add_array("tokenizer.statim.merges", merges)
    # Normalizer / pre-tokenizer / added tokens are small; keep their exact HF JSON so the C++
    # side interprets them with the same code path as tokenizer.json.
    w.add_string("tokenizer.statim.normalizer", json.dumps(tj.get("normalizer"), ensure_ascii=False))
    w.add_string("tokenizer.statim.pre_tokenizer", json.dumps(tj.get("pre_tokenizer"), ensure_ascii=False))
    w.add_string("tokenizer.statim.added_tokens", json.dumps(added, ensure_ascii=False))
    w.add_string("tokenizer.statim.model", json.dumps({k: v for k, v in model.items() if k not in ("vocab", "merges")}))

    def tid(name):
        t = tc.get(name)
        if isinstance(t, dict):
            t = t.get("content")
        if t is None:
            return -1
        if t in vocab:
            return vocab[t]
        for a in added:
            if a["content"] == t:
                return a["id"]
        return -1

    w.add_int32("tokenizer.statim.cls_id", tid("cls_token"))
    w.add_int32("tokenizer.statim.sep_id", tid("sep_token"))
    w.add_int32("tokenizer.statim.mask_id", tid("mask_token"))
    w.add_int32("tokenizer.statim.pad_id", tid("pad_token"))
    mt = tc.get("mask_token")
    w.add_string("tokenizer.statim.mask_token", mt["content"] if isinstance(mt, dict) else mt)


def quantizable(name, arr):
    return arr.ndim == 2 and arr.shape[1] % 32 == 0 and arr.shape[0] >= 64 and name.endswith("weight") \
        and "norm" not in name and not name.startswith("type_emb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--type", default="f16", choices=sorted(QMAP))
    ap.add_argument("--embd-type", default=None, choices=sorted(QMAP))
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    d = a.model_dir
    cfg = load_json(os.path.join(d, "rl_agent_config.json"))
    ecfg = load_json(os.path.join(d, "encoder", "config.json"))
    assert ecfg.get("model_type") == "modernbert", "encoder must be ModernBERT-family"
    weights = load_file(os.path.join(d, "model.safetensors"))

    w = gguf.GGUFWriter(a.out, ARCH)
    name = a.name or os.path.basename(os.path.normpath(d))
    w.add_name(name)
    w.add_string("general.license", "apache-2.0")
    w.add_string("general.source.url", "https://huggingface.co/convaiinnovations/" + name)
    w.add_string("general.base_model", cfg.get("encoder", ""))
    w.add_string("statim.format", "statim-decision-v1")

    g_theta, l_theta = rope_thetas(ecfg)
    n_layer = int(ecfg["num_hidden_layers"])
    layer_types = ecfg.get("layer_types") or [
        "full_attention" if i % int(ecfg["global_attn_every_n_layers"]) == 0 else "sliding_attention"
        for i in range(n_layer)]
    w.add_uint32("laya.encoder.n_embd", int(ecfg["hidden_size"]))
    w.add_uint32("laya.encoder.n_layer", n_layer)
    w.add_uint32("laya.encoder.n_head", int(ecfg["num_attention_heads"]))
    w.add_uint32("laya.encoder.n_ff", int(ecfg["intermediate_size"]))
    w.add_uint32("laya.encoder.local_window", int(ecfg["local_attention"]))
    w.add_array("laya.encoder.layer_is_global", [t == "full_attention" for t in layer_types])
    w.add_float32("laya.encoder.rope_theta_global", g_theta)
    w.add_float32("laya.encoder.rope_theta_local", l_theta)
    w.add_float32("laya.encoder.norm_eps", float(ecfg.get("norm_eps", 1e-5)))
    w.add_string("laya.encoder.activation", ecfg.get("hidden_activation", "gelu"))
    w.add_uint32("laya.encoder.max_position", int(ecfg.get("max_position_embeddings", 8192)))

    d_model = int(ecfg["hidden_size"])
    w.add_uint32("laya.head.n_layer", int(cfg.get("head_layers", 2)))
    w.add_uint32("laya.head.n_head", max(1, d_model // 64))
    w.add_uint32("laya.head.n_ff", 4 * d_model)
    w.add_uint32("laya.head.n_act", len(cfg.get("act_costs", {})) + 1)
    w.add_uint32("laya.max_len", int(cfg.get("max_len", 512)))
    w.add_uint32("laya.head_max_len", int(cfg.get("head_max_len", 192)))
    temps = weights.get("temperature")
    w.add_array("laya.temperature", [float(t) for t in cfg.get("temperature", [1.0, 1.0, 1.0])])
    w.add_string("laya.temperature_by_options", json.dumps(cfg.get("temperature_by_options", {})))
    w.add_string("laya.lang_temperatures", json.dumps(cfg.get("lang_temperatures", {})))

    tokenizer_meta(w, os.path.join(d, "tokenizer"))

    qt = QMAP[a.type]
    et = QMAP[a.embd_type or a.type]
    n_q = 0
    checkpoint_hash = hashlib.sha256()
    for tname in sorted(weights, key=lambda n: n.encode()):
        if tname == "temperature":
            continue
        arr = np.ascontiguousarray(weights[tname].astype("<f4", copy=False))
        checkpoint_hash.update(tname.encode() + b"\0")
        checkpoint_hash.update(struct.pack("<Q", arr.size))
        checkpoint_hash.update(memoryview(arr).cast("B"))
        if tname == "encoder.embeddings.tok_embeddings.weight":
            target = et
        elif quantizable(tname, arr):
            target = qt
        else:
            target = QT.F32
        if target in (QT.F32,):
            w.add_tensor(tname, arr)
        elif target == QT.F16:
            w.add_tensor(tname, arr.astype(np.float16))
        else:
            data = quants.quantize(arr, target)
            w.add_tensor(tname, data, raw_shape=data.shape, raw_dtype=target)
            n_q += 1
    w.add_string("statim.checkpoint_sha256", checkpoint_hash.hexdigest())
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("wrote %s (%s, %d quantized tensors, %.1f MB)" % (a.out, a.type, n_q, os.path.getsize(a.out) / 1e6))


if __name__ == "__main__":
    main()
