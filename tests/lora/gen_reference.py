#!/usr/bin/env python3
"""PyTorch reference for tests/test_lora.cpp: merge the random test adapter into the official
Laya model in PyTorch (W += lora_alpha / r * B @ A), then record

  tests/data/golden_lora_random.jsonl   logits of every (state, question) item of STATE_INDICES,
                                        one item per forward, like tools/gen_golden.py
  tests/data/lora_random_weights.json   sampled merged weight values plus the float64 sum of the
                                        delta per adapted tensor, and a digest of the adapter

Needs torch and the laya package (see tools/finetune); CI only reads the committed outputs.

    .venv-train/bin/python tests/lora/gen_reference.py --adapter build/lora/peft-random
"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch
from safetensors.torch import load_file

import laya
from laya.common import collate_items

STATE_INDICES = [0, 1, 2, 3, 5, 11, 21, 24, 25, 29]  # en, dict, de, es, zh, chat, nested dict, "", "ok", truncated
SAMPLES_PER_TENSOR = 64


def adapter_digest(path):
    h = hashlib.sha256()
    w = load_file(path)
    for k in sorted(w):
        h.update(k.encode())
        h.update(w[k].float().numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="PEFT directory written by make_adapters.py (peft-random)")
    ap.add_argument("--model", default="laya-multilingual")
    a = ap.parse_args()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    torch.set_num_threads(os.cpu_count())
    cfg = json.load(open(os.path.join(a.adapter, "adapter_config.json")))
    scale = float(cfg["lora_alpha"]) / int(cfg["r"])
    lora = load_file(os.path.join(a.adapter, "adapter_model.safetensors"))

    agent = laya.load(os.path.join(root, "models", a.model), device="cpu")
    assert agent.dtype == torch.float32 and not agent.amp_enabled
    params = dict(agent.model.named_parameters())
    rng = np.random.default_rng(0)
    weights = {"adapter_sha256": adapter_digest(os.path.join(a.adapter, "adapter_model.safetensors")), "tensors": {}}
    with torch.no_grad():
        for key in sorted(k for k in lora if k.endswith(".lora_A.weight")):
            module = key[len("base_model.model."):-len(".lora_A.weight")]  # encoder.layers.N.attn.Wqkv
            A, B = lora[key].float(), lora[key.replace("lora_A", "lora_B")].float()
            W = params[module + ".weight"]
            delta = scale * (B @ A)
            W += delta
            flat = W.reshape(-1)
            idx = rng.choice(flat.numel(), SAMPLES_PER_TENSOR, replace=False)
            weights["tensors"][module + ".weight"] = {
                "delta_sum": float(delta.double().sum()),
                "index": [int(i) for i in idx],
                "value": [float(flat[int(i)]) for i in idx],
            }

    inputs = json.load(open(os.path.join(root, "tests", "data", "golden_inputs.json"), encoding="utf-8"))
    questions = inputs["questions"]
    ids = list(questions)
    internal = {q: agent._to_internal(questions[q]) for q in ids}
    out = os.path.join(root, "tests", "data", "golden_lora_random.jsonl")
    n = 0
    with open(out, "w") as f:
        for si in STATE_INDICES:
            items = agent._encode_state(inputs["states"][si], ids, internal)
            for j, qid in enumerate(ids):
                it = items[j]
                b = collate_items([[it]], agent.tok.pad_token_id)
                with torch.no_grad():
                    logits, act = agent.model(b["input_ids"], b["attention_mask"], b["marker_pos"],
                                              b["marker_mask"], b["qtype"])
                f.write(json.dumps({"state_index": si, "question": qid, "ids": it["ids"], "markers": it["markers"],
                                    "qtype": it["qtype"], "logits": logits[0, :len(it["markers"])].tolist(),
                                    "act": torch.softmax(act.float(), -1)[0].tolist()}, ensure_ascii=False) + "\n")
                n += 1
    with open(os.path.join(root, "tests", "data", "lora_random_weights.json"), "w") as f:
        json.dump(weights, f)
    print("wrote %d items to %s" % (n, out))


if __name__ == "__main__":
    main()
