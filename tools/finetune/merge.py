#!/usr/bin/env python3
"""Merge Laya checkpoints fine-tuned from the same base.

    linear   theta = sum_i w_i * theta_i                       (model soup, Wortsman et al. 2022)
    task     theta = base + sum_i w_i * (theta_i - base)        (task arithmetic, Ilharco et al. 2023)
    ties     task vectors trimmed to the top --density fraction by magnitude, sign elected per
             parameter, disjoint mean of the agreeing entries, scaled by w (Yadav et al. 2023)

The output directory copies encoder/tokenizer and the config of --config-from (default: the last
model); temperatures should be refit afterwards.

    .venv-train/bin/python tools/finetune/merge.py --base models/laya-multilingual \\
        --models models/laya-multilingual-banking77 models/laya-multilingual-multitask \\
        --weights 0.5 0.5 --method task --out models/merge-test
"""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def ties(base, deltas, weights, density):
    # trim: keep the largest-magnitude `density` fraction of each task vector
    trimmed = []
    for d in deltas:
        k = max(1, int(d.numel() * density))
        thresh = d.abs().flatten().kthvalue(d.numel() - k + 1).values if k < d.numel() else 0.0
        trimmed.append(torch.where(d.abs() >= thresh, d, torch.zeros_like(d)))
    stacked = torch.stack([w * t for w, t in zip(weights, trimmed)])
    sign = torch.sign(stacked.sum(0))
    agree = (torch.sign(stacked) == sign) & (stacked != 0)
    num = (stacked * agree).sum(0)
    cnt = agree.sum(0).clamp(min=1)
    return base + num / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, required=True)
    ap.add_argument("--method", choices=["linear", "task", "ties"], default="task")
    ap.add_argument("--density", type=float, default=0.2, help="ties: fraction of each task vector kept")
    ap.add_argument("--config-from", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert len(a.models) == len(a.weights)
    base = load_file(os.path.join(a.base, "model.safetensors"))
    sds = [load_file(os.path.join(m, "model.safetensors")) for m in a.models]
    out = {}
    for k, b in base.items():
        b32 = b.float()
        xs = [sd[k].float() for sd in sds]
        if a.method == "linear":
            v = sum(w * x for w, x in zip(a.weights, xs))
        elif a.method == "task":
            v = b32 + sum(w * (x - b32) for w, x in zip(a.weights, xs))
        else:
            deltas = [x - b32 for x in xs]
            v = b32 if all(torch.count_nonzero(d) == 0 for d in deltas) else ties(b32, deltas, a.weights, a.density)
        out[k] = v.to(b.dtype).contiguous()
    os.makedirs(a.out, exist_ok=True)
    save_file(out, os.path.join(a.out, "model.safetensors"))
    src = a.config_from or a.models[-1]
    for sub in ("encoder", "tokenizer"):
        shutil.copytree(os.path.join(src, sub), os.path.join(a.out, sub), dirs_exist_ok=True)
    cfg = json.load(open(os.path.join(src, "rl_agent_config.json")))
    cfg["merge"] = {"method": a.method, "base": a.base, "models": a.models, "weights": a.weights,
                    **({"density": a.density} if a.method == "ties" else {})}
    json.dump(cfg, open(os.path.join(a.out, "rl_agent_config.json"), "w"), indent=2)
    print("wrote", a.out, cfg["merge"])


if __name__ == "__main__":
    main()
