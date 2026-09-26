#!/usr/bin/env python3
"""Fine-tune a Laya checkpoint on Banking77 with Laya's RLCD recipe, single GPU.

Recipe as in Laya's notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb (soft cross-entropy
+ noisy-logit policy gradient on a proper scoring reward, AdamW, cosine schedule, post-hoc
temperature), with these changes:

- Banking77 *train* only; the first 400 *test* rows used by bench/eval_accuracy.py are never seen.
- 77 options need a larger option budget: head_max_len 192/256 cuts every option to 1-2 subword
  tokens. Banking items are built with head_max_len 512 and the output config keeps 512.
- Option order is shuffled per item, so the model learns to read the intent names instead of
  their positions in the list.
- Replay of LocalLLaMA/typed-decisions train (soft targets) against forgetting general decisions.
- Token embeddings are frozen: mmBERT's 256k-row table is ~90 % of the parameters, and AdamW's
  weight decay would shrink rows of every language the Banking77 data never touches.

    .venv-train/bin/python tools/finetune/train_banking77.py models/laya-multilingual models/laya-multilingual-banking77
"""
import argparse
import json
import math
import os
import random
import shutil
import time

import torch
from safetensors.torch import load_file, save_file

from laya.agent import _fix_tokenizer_config, _load_tokenizer
from laya.common import QTYPES, build_model, build_sequence, proper_reward, render_options, temp_bucket

INSTR = "Which banking intent does `message` express?"
BANK_HEAD = 512
SEED = 20260926


def bank_items(tok, rows, labels, max_len, rng, shuffle_options=True):
    out = []
    for r in rows:
        keys = list(labels)
        if shuffle_options:
            rng.shuffle(keys)
        gold = r["label_text"].replace("_", " ")
        q = {"t": "choice", "ins": INSTR, "crit": {k: None for k in keys}}
        seq, markers = build_sequence(tok, {"message": r["text"]}, q, max_len, BANK_HEAD)
        if len(markers) != len(keys):
            continue
        target = [1.0 if k == gold else 0.0 for k in keys]
        out.append({"ids": seq, "markers": markers, "qtype": QTYPES["choice"], "target": target, "src": "banking77"})
    return out


def typed_items(tok, rows, max_len, head_max_len):
    out = []
    for row in rows:
        state, questions, gold = json.loads(row["state"]), json.loads(row["questions"]), json.loads(row["gold"])
        for qid, qd in questions.items():
            if qid not in gold:
                continue
            t, crit = qd["type"], qd.get("criteria", {})
            if t == "choice" and isinstance(crit, list):
                crit = {c: None for c in crit}
            gp = gold[qid]["probabilities"]
            if t == "choice":
                target = [gp.get(k, 0.0) for k in crit]
            elif t == "noul":
                target = [gp.get("false", 0.5), gp.get("true", 0.5)]
            else:
                target = [gp.get(str(i), 0.0) for i in range(len(crit) if isinstance(crit, list) else 4)]
            s = sum(target)
            target = [v / s for v in target] if s > 0 else [1.0 / len(target)] * len(target)
            q = {"t": t, "ins": qd["instructions"], "crit": crit}
            seq, markers = build_sequence(tok, state, q, max_len, head_max_len)
            if len(markers) != len(render_options(q)) or len(markers) != len(target):
                continue
            out.append({"ids": seq, "markers": markers, "qtype": QTYPES[t], "target": target, "src": "typed"})
    return out


def collate(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax))
    for i, it in enumerate(items):
        ids[i, :len(it["ids"])] = torch.tensor(it["ids"])
        att[i, :len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, :k] = torch.tensor(it["target"])
    return ids, att, mpos, mmask, torch.tensor([it["qtype"] for it in items]), target


def batches(items, max_tokens, max_rows):
    """Length-sorted buckets under a padded-token budget, then shuffled."""
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    out, cur, cur_max = [], [], 0
    for i in order:
        L = len(items[i]["ids"])
        if cur and (max(cur_max, L) * (len(cur) + 1) > max_tokens or len(cur) >= max_rows):
            out.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, L)
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def predict(model, items, pad_id, device, dtype):
    model.eval()
    res = []
    for b in batches(items, 16384, 32):
        chunk = [items[i] for i in b]
        ids, att, mpos, mmask, qt, _ = collate(chunk, pad_id)
        with torch.autocast("cuda", dtype=dtype):
            logits, _ = model(ids.to(device), att.to(device), mpos.to(device), mmask.to(device), qt.to(device))
        lg = logits.float().cpu()
        for r, i in enumerate(b):
            res.append((i, lg[r, :len(items[i]["markers"])].tolist()))
    model.train()
    res.sort()
    return [z for _, z in res]


def fit_temp(pairs):
    if len(pairs) < 10:
        return None
    kmax = max(len(z) for z, _ in pairs)
    Z = torch.full((len(pairs), kmax), -1e4)
    T = torch.zeros((len(pairs), kmax))
    for i, (z, t) in enumerate(pairs):
        Z[i, :len(z)] = torch.tensor(z)
        T[i, :len(t)] = torch.tensor(t)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.5, 5.0).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr-encoder", type=float, default=2.5e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--max-tokens", type=int, default=4096, help="padded tokens per micro-batch")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--calib", type=int, default=500, help="Banking77 train rows held out for dev/calibration")
    ap.add_argument("--no-replay", action="store_true")
    a = ap.parse_args()

    from datasets import load_dataset

    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    device = torch.device("cuda")
    base = os.path.abspath(a.base)
    _fix_tokenizer_config(base)
    cfg = json.load(open(os.path.join(base, "rl_agent_config.json")))
    tok = _load_tokenizer(os.path.join(base, "tokenizer"), cfg)
    dtype = torch.bfloat16 if cfg.get("amp_dtype") == "bf16" else torch.float16
    max_len = cfg.get("max_len", 512)

    bank = list(load_dataset("mteb/banking77", split="train"))
    labels = sorted({r["label_text"].replace("_", " ") for r in bank})
    assert len(labels) == 77
    rng.shuffle(bank)
    dev_rows, train_rows = bank[:a.calib], bank[a.calib:]
    train = bank_items(tok, train_rows, labels, max_len, rng)
    # dev in the evaluation's fixed (sorted) order, like bench/eval_accuracy.py
    dev = bank_items(tok, dev_rows, labels, max_len, rng, shuffle_options=False)
    typed_dev = []
    if not a.no_replay:
        td = list(load_dataset("LocalLLaMA/typed-decisions", "all", split="train"))
        rng.shuffle(td)
        n_hold = len(td) // 10
        typed_dev = typed_items(tok, td[:n_hold], max_len, cfg.get("head_max_len", 192))
        train += typed_items(tok, td[n_hold:], max_len, cfg.get("head_max_len", 192))
    n_bank = sum(it["src"] == "banking77" for it in train)
    print(f"train items: {len(train)} ({n_bank} banking77, {len(train) - n_bank} typed-decisions) | "
          f"dev: {len(dev)} banking77, {len(typed_dev)} typed-decisions", flush=True)

    model = build_model(cfg, encoder_dir=os.path.join(base, "encoder"))
    model.load_state_dict(load_file(os.path.join(base, "model.safetensors")), strict=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device).train()

    frozen = 0
    enc_params, head_params = [], []
    for n, p in model.named_parameters():
        if "embeddings.tok_embeddings" in n:
            p.requires_grad_(False)
            frozen += p.numel()
        elif n.startswith("encoder."):
            enc_params.append(p)
        else:
            head_params.append(p)
    print(f"frozen token embeddings: {frozen / 1e6:.1f}M params; trainable: "
          f"{sum(p.numel() for p in enc_params + head_params) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr_encoder},
                             {"params": head_params, "lr": a.lr_head}], weight_decay=0.01)
    per_epoch = math.ceil(len(batches(train, a.max_tokens, 64)) / a.accum)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=per_epoch * a.epochs, eta_min=1e-6)

    def dev_acc():
        z = predict(model, dev, tok.pad_token_id, device, dtype)
        return sum(max(range(len(l)), key=l.__getitem__) == it["target"].index(1.0) for l, it in zip(z, dev)) / len(dev)

    acc0 = dev_acc()
    print(f"dev banking77 accuracy before training: {acc0:.4f}", flush=True)
    best, best_epoch, log = -1.0, 0, []
    os.makedirs(a.out, exist_ok=True)
    tmp_best = os.path.join(a.out, "best.safetensors")
    t0 = time.time()
    for epoch in range(a.epochs):
        sigma = 0.4 + (0.1 - 0.4) * (epoch / max(1, a.epochs - 1))
        bl = batches(train, a.max_tokens, 64)
        rng.shuffle(bl)
        opt.zero_grad(set_to_none=True)
        run_loss, n = 0.0, 0
        for step, b in enumerate(bl):
            ids, att, mpos, mmask, qt, target = collate([train[i] for i in b], tok.pad_token_id)
            ids, att, mpos, mmask, qt, target = (x.to(device) for x in (ids, att, mpos, mmask, qt, target))
            with torch.autocast("cuda", dtype=dtype):
                logits, act = model(ids, att, mpos, mmask, qt)
            logits = logits.float()
            k = mmask.sum(-1, keepdim=True).float()
            eps = torch.randn((4,) + logits.shape, device=device) * sigma * mmask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mmask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mmask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), qt, mmask, w_sph=0.75, w_rps=1.0)
                adv = (r - r.mean(0, keepdim=True)) / ((r - r.mean(0, keepdim=True)).std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mmask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mmask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + loss_ce) / a.accum + 0.0 * act.sum()
            loss.backward()
            if (step + 1) % a.accum == 0 or step + 1 == len(bl):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            run_loss += loss_ce.item()
            n += 1
            if step % 200 == 0:
                print(f"  epoch {epoch + 1} step {step}/{len(bl)} ce {run_loss / n:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
        acc = dev_acc()
        log.append({"epoch": epoch + 1, "train_ce": round(run_loss / n, 4), "dev_banking77": round(acc, 4),
                    "seconds": round(time.time() - t0)})
        print(f"=== epoch {epoch + 1}: train ce {run_loss / n:.4f} | dev banking77 {acc:.4f} | {time.time() - t0:.0f}s",
              flush=True)
        if acc > best:
            best, best_epoch = acc, epoch + 1
            save_file({k: v.contiguous().cpu() for k, v in model.state_dict().items()}, tmp_best)

    # restore the best epoch, then fit temperatures on held-out items only
    model.load_state_dict(load_file(tmp_best), strict=True)
    os.remove(tmp_best)
    calib = dev + typed_dev
    zs = predict(model, calib, tok.pad_token_id, device, dtype)
    by_type, by_bucket = {}, {}
    for z, it in zip(zs, calib):
        by_type.setdefault(it["qtype"], []).append((z, it["target"]))
        by_bucket.setdefault(temp_bucket(it["qtype"], len(z)), []).append((z, it["target"]))
    temps = list(cfg.get("temperature", [1.0, 1.0, 1.0]))
    for qt, pairs in by_type.items():
        t = fit_temp(pairs)
        if t:
            temps[qt] = t
    tbo = {b: t for b, pairs in by_bucket.items() if (t := fit_temp(pairs))}
    print("temperatures:", [round(t, 3) for t in temps], {k: round(v, 3) for k, v in tbo.items()}, flush=True)

    save_file({k: v.half().contiguous().cpu() for k, v in model.state_dict().items()},
              os.path.join(a.out, "model.safetensors"))
    for sub in ("encoder", "tokenizer"):
        shutil.copytree(os.path.join(base, sub), os.path.join(a.out, sub), dirs_exist_ok=True)
    cfg.update({"head_max_len": BANK_HEAD, "temperature": temps, "temperature_by_options": tbo,
                "model_name": "laya-multilingual-banking77", "fine_tuned": True,
                "training_banking77": {"base": os.path.basename(base.rstrip("/")), "best_epoch": best_epoch,
                                       "dev_banking77_before": round(acc0, 4), "log": log,
                                       "replay": not a.no_replay, "frozen": "token embeddings",
                                       "lr": [a.lr_encoder, a.lr_head], "epochs": a.epochs, "seed": SEED}})
    json.dump(cfg, open(os.path.join(a.out, "rl_agent_config.json"), "w"), indent=2)
    print(f"saved {a.out} (best epoch {best_epoch}, dev banking77 {best:.4f})", flush=True)


if __name__ == "__main__":
    main()
