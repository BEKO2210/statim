#!/usr/bin/env python3
"""Uniform model soup: average the weights of fine-tuned runs of one recipe (Wortsman et al., 2022,
"Model soups", arXiv 2203.05482).

The members must share the start checkpoint, the architecture and the tokenizer, and differ only
in seed or batch order. The soup has the size and speed of one member. Its temperatures are the
geometric mean of the members' fitted temperatures. That is an approximation, so recalibrate before
publishing probabilities. The gate decides whether the soup ships (accuracy is unaffected by
temperatures).

    tools/finetune/soup.py models/soup models/run-a models/run-b models/run-c
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def geometric_mean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("members", nargs="+")
    a = ap.parse_args(argv)
    if len(a.members) < 2:
        raise SystemExit("a soup needs at least two members")
    if os.path.exists(a.out):
        raise SystemExit("refusing to overwrite %s" % a.out)
    cfgs = [json.load(open(os.path.join(m, "rl_agent_config.json"))) for m in a.members]
    bases = {(c.get("training_multitask") or {}).get("base") for c in cfgs}
    encoders = {c.get("encoder") for c in cfgs}
    toks = {sha256(os.path.join(m, "tokenizer", "tokenizer.json")) for m in a.members}
    if len(bases) != 1 or len(encoders) != 1 or len(toks) != 1:
        raise SystemExit("members differ in start checkpoint, encoder or tokenizer: %s %s %d tokenizers"
                         % (bases, encoders, len(toks)))

    total, keys = None, None
    for m in a.members:
        state = load_file(os.path.join(m, "model.safetensors"))
        if keys is None:
            keys = {k: (v.shape, v.dtype) for k, v in state.items()}
            total = {k: v.float().clone() for k, v in state.items()}
            continue
        if {k: (v.shape, v.dtype) for k, v in state.items()} != keys:
            raise SystemExit("%s: tensors differ from the first member" % m)
        for k, v in state.items():
            total[k] += v.float()
    soup = {k: (v / len(a.members)).to(keys[k][1]).contiguous() for k, v in total.items()}

    os.makedirs(a.out)
    save_file(soup, os.path.join(a.out, "model.safetensors"))
    for sub in ("encoder", "tokenizer"):
        shutil.copytree(os.path.join(a.members[0], sub), os.path.join(a.out, sub))
    cfg = json.loads(json.dumps(cfgs[0]))
    temps = [c.get("temperature") for c in cfgs]
    if all(isinstance(t, list) and len(t) == len(temps[0]) for t in temps):
        cfg["temperature"] = [geometric_mean(vals) for vals in zip(*temps)]
    shared = set.intersection(*(set(c.get("temperature_by_options") or {}) for c in cfgs))
    cfg["temperature_by_options"] = {b: geometric_mean([c["temperature_by_options"][b] for c in cfgs])
                                     for b in sorted(shared)}
    cfg["model_name"] = os.path.basename(a.out.rstrip("/"))
    cfg["soup"] = {"method": "uniform weight average (Wortsman et al., 2022)",
                   "members": [{"path": m, "model_sha256": sha256(os.path.join(m, "model.safetensors")),
                                "best_epoch": (c.get("training_multitask") or {}).get("best_epoch")}
                               for m, c in zip(a.members, cfgs)],
                   "temperatures": "geometric mean of the members (not refitted)"}
    json.dump(cfg, open(os.path.join(a.out, "rl_agent_config.json"), "w"), indent=2)
    print("soup of %d members -> %s (%d tensors)" % (len(a.members), a.out, len(soup)))


if __name__ == "__main__":
    sys.exit(main())
