#!/usr/bin/env python3
"""LoRA specialist for one decision category: PEFT LoRA on the encoder of a Statim (Laya)
checkpoint, trained on that category's rows of a built mixture, saved as a PEFT adapter that
tools/convert_lora.py turns into a Statim adapter GGUF (`statim serve --adapter`).

    .venv-train/bin/pip install peft    # 0.21.0 tested; not part of the training venv's install line
    .venv-train/bin/python tools/finetune/train_lora.py dist/statim-decide-multilingual-base/checkpoint \\
        --mixture data/mixture-v8.jsonl.gz --category emotion             # writes models/lora/emotion
    .venv/bin/python tools/convert_lora.py models/lora/emotion -o models/lora/emotion.lora.gguf --category emotion \\
        --base dist/statim-decide-multilingual-base/statim-decide-multilingual-base-f32.gguf

tools/finetune/lora_experiment.py runs this, the conversion and the held-out evaluation of base and
adapter for several categories and decides with the promotion gate's Holm rule.

What is trained:
- LoRA (W' = W + B·A, lora_alpha / r scale) on attn.Wqkv, attn.Wo, mlp.Wi and mlp.Wo of every
  encoder layer (22 x 4 = 88 modules on the released models), bias "none", no modules_to_save, no
  DoRA: exactly what an adapter GGUF can carry (convert_lora rejects anything else). The wrapped
  modules and the trainable parameters are asserted after get_peft_model. The token embeddings and
  the decision head (head.*, type_emb, scorer, act_head) stay frozen and bit-identical.
- The head is also kept in eval mode. Its dropout (0.1; the encoder's dropouts are 0.0, so it is the
  only one in the model) would otherwise perturb the logits during training only. Statim never drops,
  so the adapter is fitted to the head as it runs at inference. LoRA's own dropout
  (--lora-dropout) is the only training noise besides the RLCD sampling.
- init: --init true (PEFT default, B = 0: step 0 is exactly the base model) or gaussian.

Data: the rows of one category from a built mixture (jsonl.gz rows {state, q, target, src, lang},
read line by line; only matching rows are kept). A row belongs to the category when its src equals
"v6/<id>/<config or 'default'>" of an enabled (use: true) entry of --registry whose category string
passes the test in CATEGORY_TESTS (the same tests as registry.task_for; e.g. fact_check takes
10-fact-check and 10-claim-*, because its held-out suite is ClaimBuster check-worthiness). Rows
whose src the registry does not know (v5 rows, synthetic rows, sources of a newer registry) are
skipped and counted. Sequences are built as in train_multitask.py (max_len of the checkpoint,
head_max_len 512 like bench/eval_categories.py); items whose marker count does not match their
options are dropped.

Dev: --dev-items items held out per source in proportion to its size, chosen by a hash of the
normalised state (the same state_key as train_multitask.py), so the split does not depend on row
order or --limit-items and no dev state occurs in train (every question of a dev state is in dev).
Dev accuracy (argmax) is measured before training, after every epoch and every --eval-every
optimiser steps; the best adapter is kept. The base model was trained on most of the mixture, dev
rows included, so dev accuracy only ranks adapter checkpoints; the held-out verdict comes from
bench/eval_categories.py (lora_experiment.py).

Loss and optimiser: the RLCD + CE step of train_multitask.py (noisy-logit policy gradient on
laya's proper_reward with advantage normalisation, plus soft cross-entropy; sigma 0.4 -> 0.1 over
the epochs). AdamW on the LoRA parameters only, --lr 2e-4, linear warmup (--warmup fraction), then
cosine to 1e-6, gradient clipping at 1.0, --accum micro-batches per step. --weight-decay defaults to
0.0: decay on A and B pulls the product towards zero, i.e. towards the base model, and the recipe
already selects by dev accuracy; 0.01 (the full fine-tune's value) is a reasonable alternative.
Batches are train_banking77.batches (length-sorted, a padded-token budget --max-tokens and at most
--max-rows rows per micro-batch, shuffled).

Device and precision: --device auto (cuda if available) | cuda | cpu. --amp auto uses bf16 autocast
on a CUDA device that supports it (Ampere and newer, e.g. the RTX 3070) and fp32 elsewhere; there is
no fp16 autocast, because the recipe has no loss scaling. Base weights are fp32 in memory, as in the
full fine-tunes. --base-dtype fp16 or bf16 stores the frozen encoder matrices (projections and token
embeddings; norms and the head stay fp32) in 16 bits and needs bf16 autocast. fp16 is lossless for
the released checkpoints (model.safetensors is F16) and, under bf16 autocast, feeds the matmuls the
same bf16 values as fp32 storage; bf16 rounds the F16 weights to 8 significant bits, so the adapter
is trained against a slightly different base than the one Statim serves. Either saves about 0.6 GB.
Gradient checkpointing is on (encoder layers, non-reentrant, as in the full fine-tunes). Head
checkpointing is on as well: the head is frozen, but the encoder's gradient still flows through it,
so its activations are needed; checkpointing trades one extra forward pass of the two head layers
(about 3 % of a training step) for not storing them (measured below).

Sizing for an RTX 3070 (8 GB):
- recommended (the defaults): --max-tokens 8192 --max-rows 64 --accum 2 (about 16k padded tokens,
  or up to 128 rows, per optimiser step), r 16:
      .venv-train/bin/python tools/finetune/train_lora.py dist/statim-decide-multilingual-base/checkpoint \\
          --mixture data/mixture-v8.jsonl.gz --category emotion --device cuda \\
          --max-tokens 8192 --max-rows 64 --accum 2 --epochs 2
- estimate (not measured on a GPU): fp32 weights 1.29 GB (322M parameters, 197M of them the token
  embedding); LoRA parameters, gradients and AdamW states under 60 MB at r 16 (3.4M parameters);
  CUDA context and allocator slack about 0.5 GB; checkpointed layer inputs 22 x T x 768 x 4 B
  (0.55 GB at T = 8192 tokens) plus the head's 2 x T x 768 x 4 B; the recomputed layer during
  backward, with bf16 activations and the 4-D attention masks ModernBERT builds for sdpa, well under
  1 GB at 8 x 1024 tokens (the worst shape of an 8192-token budget). About 3-3.5 GB in total, so
  8192 tokens leave room for the desktop's share of the card; 16384 tokens should still fit (about
  4.5 GB) and halve the micro-batch count. The full fine-tune of this model (125M trainable
  parameters, 2 GB of AdamW states) already fit at --max-tokens 4096.
- check before a long run: --profile-batch 8x1024 runs two training steps on a synthetic batch of
  that shape and prints the peak memory (torch.cuda.max_memory_allocated on CUDA, VmHWM on CPU).
- measured on CPU (4 cores, fp32, torch 2.14): see the report of the PR that added this file; the
  CPU peak is higher than the GPU's for the same shape (fp32 activations, no bf16).

Output (--out, default models/lora/<category>, so convert_lora's default adapter name and category
are right): adapter_config.json + adapter_model.safetensors of the best adapter (save_pretrained,
safetensors) and train_lora.json (category, arguments, mixture path and SHA-256, base checkpoint path
and SHA-256 of its model.safetensors, rows per source, dev accuracy per evaluation, whether the initial
zero-delta adapter won, best epoch, wall time, peak memory, library versions).

Smoke run: --limit-items 64 --dev-items 16 --max-steps 4 --max-tokens 1024 --device cpu.
"""
import argparse
import collections
import contextlib
import gzip
import hashlib
import json
import math
import os
import random
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

REGISTRY = os.path.join(HERE, "sources", "v6-keep.json")
SEED = 20260926  # train_banking77.SEED (not imported: that module needs torch and laya)
HEAD_MAX_LEN = 512  # train_banking77.BANK_HEAD, and what bench/eval_categories.py sends
# convert_lora.CATEGORIES = the keys of bench/eval_categories.HELD_OUT (test_train_lora.py checks both)
CATEGORIES = ("sentiment", "emotion", "complaint", "nli", "safety", "reading", "similarity", "topic",
              "intent", "stance", "formality", "urgency", "fact_check", "pii")
TARGETS = ("attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo")
LORA_PARAM = re.compile(r"encoder\.layers\.\d+\.(attn\.Wqkv|attn\.Wo|mlp\.Wi|mlp\.Wo)\.lora_[AB]\.default\.weight")
FROZEN = ("encoder.embeddings.", "head.", "type_emb.", "scorer.", "act_head.")


def _parts(cat):
    return [p.strip() for p in cat.split(";")]


# Registry category string -> does it belong to the suite? Same tests as registry.task_for; a source
# may serve two suites ("3-complaint; 1-sentiment" is complaint and sentiment).
CATEGORY_TESTS = {
    "sentiment": lambda c: "1-sentiment" in c,
    "emotion": lambda c: "2-emotion" in c,
    "complaint": lambda c: "3-complaint" in c,
    "nli": lambda c: "4-nli" in c,
    "safety": lambda c: c.startswith("5-"),
    "reading": lambda c: any(p.startswith("6-") for p in _parts(c)),
    "similarity": lambda c: any(p.startswith("7-") for p in _parts(c)),
    "topic": lambda c: any(p.startswith("8-") for p in _parts(c)),
    "intent": lambda c: any(p.startswith("9-") for p in _parts(c)),
    "stance": lambda c: "10-stance" in c or "10-argument" in c,
    "formality": lambda c: "10-formality" in c,
    "urgency": lambda c: "10-urgency" in c,
    "fact_check": lambda c: "10-fact-check" in c or "10-claim" in c,
    "pii": lambda c: "10-pii" in c,
}


# --------------------------------------------------------------------------- data (stdlib only)

def source_name(entry):
    """registry.source_name: the src of every mixture row built from `entry`."""
    return "v6/%s/%s" % (entry["id"], entry.get("config") or "default")


def category_sources(registry, category):
    """(src names of the category, {src: category string} of every enabled entry). Exact src names:
    a config may contain "/", so the src is never split."""
    with open(registry, encoding="utf-8") as f:
        raw = json.load(f)
    raw = raw if isinstance(raw, list) else raw["sources"]
    enabled = {source_name(e): e.get("category", "") for e in raw if e.get("use") is True}
    test = CATEGORY_TESTS[category]
    return {src for src, cat in enabled.items() if test(cat)}, enabled


def select_rows(path, chosen, enabled, limit=0):
    """Rows of a built mixture (jsonl.gz, or plain jsonl) whose src is in `chosen`, streamed line by
    line. Returns (rows, stats); stats counts rows per chosen source, rows of other enabled sources,
    and rows whose src the registry does not know (with the most frequent such src values)."""
    rows, per_source = [], collections.Counter()
    other, unknown = 0, collections.Counter()
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            src = r.get("src", "")
            if src in chosen:
                rows.append({k: r.get(k) for k in ("state", "q", "target", "src", "lang")})
                per_source[src] += 1
                if limit and len(rows) >= limit:
                    break
            elif src in enabled:
                other += 1
            else:
                unknown[src] += 1
    stats = {"rows": len(rows), "per_source": dict(sorted(per_source.items())), "other_category_rows": other,
             "unknown_src_rows": sum(unknown.values()), "unknown_src_top": dict(unknown.most_common(10)),
             "limited": bool(limit and len(rows) >= limit)}
    return rows, stats


def state_key(state):
    """train_multitask.state_key: the held-out comparison key of a mixture state."""
    return " ".join(json.dumps(state, sort_keys=True, ensure_ascii=False).split()).lower()


def split_dev(rows, n_dev):
    """(train indices, dev indices). Each source gets a dev quota in proportion to its row count
    (largest remainder); within a source, states are taken in the order of the SHA-256 of their
    state_key until the quota is met. Every row of a dev state is dev, so no dev state is in train.
    Deterministic and independent of row order. At most half of the rows go to dev."""
    total = len(rows)
    n_dev = max(0, min(n_dev, total // 2))
    keys = [state_key(r["state"]) for r in rows]
    if not n_dev:
        return list(range(total)), []
    by_src = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_src[r["src"]].append(i)
    share = {s: n_dev * len(ix) / total for s, ix in by_src.items()}
    quota = {s: int(v) for s, v in share.items()}
    for s in sorted(share, key=lambda s: (quota[s] - share[s], s))[:n_dev - sum(quota.values())]:
        quota[s] += 1
    digest = {k: hashlib.sha256(k.encode("utf-8")).hexdigest() for k in set(keys)}
    dev_states = set()
    for s in sorted(by_src):
        count = collections.Counter(keys[i] for i in by_src[s])
        taken = 0
        for k in sorted(count, key=lambda k: (digest[k], k)):
            if taken >= quota[s]:
                break
            dev_states.add(k)
            taken += count[k]
    dev = [i for i in range(total) if keys[i] in dev_states]
    train = [i for i in range(total) if keys[i] not in dev_states]
    return train, dev


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def peak_rss_mb():
    """VmHWM of this process in MB (Linux), or ru_maxrss where /proc is missing."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


def reset_peak_rss():
    """Reset VmHWM to the current RSS (Linux >= 4.0, /proc/self/clear_refs "5"). False if unsupported."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


# --------------------------------------------------------------------------- model (torch, peft, laya)

def load_base(base):
    """(agent config, tokenizer, DecisionModel with the checkpoint's weights in fp32), loaded like
    train_multitask.py."""
    from safetensors.torch import load_file
    from laya.agent import _fix_tokenizer_config, _load_tokenizer
    from laya.common import build_model
    base = os.path.abspath(base)
    _fix_tokenizer_config(base)
    with open(os.path.join(base, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    tok = _load_tokenizer(os.path.join(base, "tokenizer"), cfg)
    model = build_model(cfg, encoder_dir=os.path.join(base, "encoder"))
    model.load_state_dict(load_file(os.path.join(base, "model.safetensors")), strict=True)
    return cfg, tok, model


def make_items(tok, rows, max_len, head_max_len):
    """Training items as in train_multitask.mixture_data; returns (items, dropped) where dropped
    counts rows whose marker count does not match their options or target."""
    import numpy as np
    from laya.common import QTYPES, build_sequence, render_options
    items, dropped = [], 0
    for r in rows:
        qd = r["q"]
        q = {"t": qd["type"], "ins": qd["instructions"],
             "crit": qd.get("criteria") or ({} if qd["type"] == "noul" else None)}
        seq, markers = build_sequence(tok, r["state"], q, max_len, head_max_len)
        if len(markers) != len(render_options(q)) or len(markers) != len(r["target"]):
            dropped += 1
            continue
        items.append({"ids": np.asarray(seq, dtype=np.int32), "markers": markers, "qtype": QTYPES[q["t"]],
                      "target": r["target"], "src": r["src"], "lang": r.get("lang") or ""})
    return items, dropped


def lora_config(r, alpha, dropout, init):
    from peft import LoraConfig
    return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=["Wqkv", "Wo", "Wi"],
                      bias="none", modules_to_save=None, use_dora=False, init_lora_weights=init, task_type=None)


def check_lora(pm, n_layers):
    """Assert that exactly encoder.layers.<i>.{attn.Wqkv, attn.Wo, mlp.Wi, mlp.Wo} carry LoRA and that
    only their lora_A / lora_B are trainable (nothing of the head, embeddings, type_emb, scorer or
    act_head). Returns (wrapped module count, trainable parameter count)."""
    from peft.tuners.lora import LoraLayer
    wrapped = {n.removeprefix("base_model.model.") for n, m in pm.named_modules() if isinstance(m, LoraLayer)}
    want = {"encoder.layers.%d.%s" % (i, t) for i in range(n_layers) for t in TARGETS}
    if wrapped != want:
        raise SystemExit("LoRA wraps the wrong modules: missing %s, unexpected %s"
                         % (sorted(want - wrapped)[:4], sorted(wrapped - want)[:4]))
    trainable = [(n.removeprefix("base_model.model."), p) for n, p in pm.named_parameters() if p.requires_grad]
    bad = [n for n, _ in trainable if not LORA_PARAM.fullmatch(n) or n.startswith(FROZEN)]
    if bad or len(trainable) != 2 * len(want):
        raise SystemExit("unexpected trainable parameters: %s (%d tensors, want %d)" % (bad[:4], len(trainable), 2 * len(want)))
    return len(wrapped), sum(p.numel() for _, p in trainable)


def cast_encoder_matrices(model, dtype):
    """Store the frozen encoder matrices (projections, token embeddings) in `dtype`; norms stay fp32."""
    import torch.nn as nn
    for m in model.encoder.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            m.weight.data = m.weight.data.to(dtype)


def set_train(pm, model):
    """Training mode for the encoder and LoRA; the frozen head stays in eval mode (no dropout)."""
    pm.train()
    if model.head is not None:
        model.head.eval()


def prepare(model, a, device):
    """Base dtype, gradient checkpointing, LoRA (checked), device and training mode. Returns the PeftModel."""
    import torch
    from peft import get_peft_model
    if a.base_dtype != "fp32":
        cast_encoder_matrices(model, {"fp16": torch.float16, "bf16": torch.bfloat16}[a.base_dtype])
    if a.grad_checkpointing:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = a.grad_checkpointing and a.head_checkpointing
    torch.manual_seed(a.seed)  # LoRA's A init
    pm = get_peft_model(model, lora_config(a.r, a.lora_alpha, a.lora_dropout, True if a.init == "true" else a.init))
    pm.lora_counts = check_lora(pm, model.encoder.config.num_hidden_layers)
    pm.to(device)
    set_train(pm, model)
    return pm


def pick_device(name):
    import torch
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda: no CUDA device (torch %s)" % torch.__version__)
    return torch.device(name)


def pick_amp(name, device):
    """bf16 autocast dtype, or None for fp32."""
    import torch
    if name == "off":
        return None
    if name == "bf16":
        return torch.bfloat16
    ok = device.type == "cuda" and torch.cuda.is_bf16_supported()
    return torch.bfloat16 if ok else None


def autocast(device, amp_dtype):
    import torch
    return torch.autocast(device.type, dtype=amp_dtype) if amp_dtype is not None else contextlib.nullcontext()


def rlcd_loss(logits, act, mmask, qt, target, sigma, accum):
    """The training step's loss from train_multitask.py: four noisy copies of the logits, laya's
    proper_reward (log + 0.75 spherical, minus RPS for score questions) as reward, normalised
    advantages, Gaussian log-likelihood of the noise as policy term, plus soft cross-entropy.
    Returns (loss / accum, cross-entropy)."""
    import torch
    from laya.common import proper_reward
    logits = logits.float()
    k = mmask.sum(-1, keepdim=True).float()
    eps = torch.randn((4,) + logits.shape, device=logits.device) * sigma * mmask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mmask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mmask, -1e4), -1)
    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), qt, mmask, w_sph=0.75, w_rps=1.0)
        adv = (r - r.mean(0, keepdim=True)) / ((r - r.mean(0, keepdim=True)).std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mmask).sum(-1) / (2 * sigma ** 2)
    loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mmask, -1e4), -1)).sum(-1).mean()
    return (-(adv * logp).mean() + loss_ce) / accum + 0.0 * act.sum(), loss_ce


def predict(pm, model, items, pad_id, device, amp_dtype):
    """Raw logits per item, in input order (train_banking77.predict with a device and autocast choice)."""
    import torch
    from train_banking77 import batches, collate
    pm.eval()
    out = [None] * len(items)
    with torch.no_grad():
        for b in batches(items, 16384, 32):
            ids, att, mpos, mmask, qt, _ = collate([items[i] for i in b], pad_id)
            with autocast(device, amp_dtype):
                logits, _ = pm(ids.to(device), att.to(device), mpos.to(device), mmask.to(device), qt.to(device))
            lg = logits.float().cpu()
            for row, i in enumerate(b):
                out[i] = lg[row, :len(items[i]["markers"])].tolist()
    set_train(pm, model)
    return out


def accuracy(items, logits):
    """(argmax accuracy, {lang: accuracy})."""
    hits = collections.defaultdict(list)
    for it, z in zip(items, logits):
        gold = max(range(len(it["target"])), key=it["target"].__getitem__)
        hits[it.get("lang") or "?"].append(max(range(len(z)), key=z.__getitem__) == gold)
    every = [h for v in hits.values() for h in v]
    return round(sum(every) / len(every), 4), {k: round(sum(v) / len(v), 4) for k, v in sorted(hits.items())}


def train(pm, model, train_items, dev_items, pad_id, device, amp_dtype, a, log=print):
    """Train LoRA; keep the adapter with the best dev accuracy (the last one without dev items) and
    load it into pm. Returns the record for train_lora.json."""
    import torch
    from train_banking77 import batches, collate
    rng = random.Random(a.seed)
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    per_epoch = math.ceil(len(batches(train_items, a.max_tokens, a.max_rows)) / a.accum)
    total = per_epoch * a.epochs
    if a.max_steps:
        total = min(total, a.max_steps)
    warm = int(total * a.warmup)
    floor = min(1.0, 1e-6 / a.lr)

    def lr_factor(step):
        if step < warm:
            return (step + 1) / warm
        t = (step - warm) / max(1, total - warm)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    def snapshot():
        return {n: p.detach().to("cpu", copy=True) for n, p in pm.named_parameters() if p.requires_grad}

    def dev_eval():
        return accuracy(dev_items, predict(pm, model, dev_items, pad_id, device, amp_dtype))

    t0 = time.time()
    dev_before = None
    if dev_items:
        acc, by_lang = dev_eval()
        dev_before = {"dev_acc": acc, "dev_by_lang": by_lang}
        log("dev before training: %.4f %s" % (acc, by_lang), flush=True)
    best = {"dev_acc": dev_before["dev_acc"] if dev_before else -1.0, "epoch": 0, "update": 0,
            "state": snapshot() if dev_before else None, "saved_initial": bool(dev_before)}
    history, state = [], {"updates": 0, "ce": 0.0, "n": 0, "last_eval": -1}

    def evaluate(epoch):
        entry = {"epoch": epoch, "update": state["updates"], "train_ce": round(state["ce"] / max(1, state["n"]), 4),
                 "lr": sched.get_last_lr()[0], "seconds": round(time.time() - t0, 1)}
        if dev_items:
            entry["dev_acc"], entry["dev_by_lang"] = dev_eval()
        if not dev_items or entry["dev_acc"] > best["dev_acc"]:
            best.update(dev_acc=entry.get("dev_acc"), epoch=epoch, update=state["updates"], state=snapshot(),
                        saved_initial=False)
        history.append(entry)
        log("=== epoch %d update %d: train ce %.4f | dev %s | %.0fs" % (
            epoch, state["updates"], entry["train_ce"], entry.get("dev_acc"), entry["seconds"]), flush=True)
        state.update(ce=0.0, n=0, last_eval=state["updates"])

    log("training: %d items, %d updates planned (%d per epoch), warmup %d" % (len(train_items), total, per_epoch, warm),
        flush=True)
    done = False
    for epoch in range(1, a.epochs + 1):
        sigma = 0.4 + (0.1 - 0.4) * ((epoch - 1) / max(1, a.epochs - 1))
        bl = batches(train_items, a.max_tokens, a.max_rows)
        rng.shuffle(bl)
        opt.zero_grad(set_to_none=True)
        for step, b in enumerate(bl):
            ids, att, mpos, mmask, qt, target = (x.to(device) for x in collate([train_items[i] for i in b], pad_id))
            with autocast(device, amp_dtype):
                logits, act = pm(ids, att, mpos, mmask, qt)
            loss, ce = rlcd_loss(logits, act, mmask, qt, target, sigma, a.accum)
            loss.backward()
            state["ce"] += ce.item()
            state["n"] += 1
            if (step + 1) % a.accum == 0 or step + 1 == len(bl):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                state["updates"] += 1
                if state["updates"] % 50 == 0:
                    log("  epoch %d update %d/%d ce %.4f lr %.2e %.0fs" % (
                        epoch, state["updates"], total, state["ce"] / max(1, state["n"]), sched.get_last_lr()[0],
                        time.time() - t0), flush=True)
                if a.eval_every and state["updates"] % a.eval_every == 0:
                    evaluate(epoch)
                if a.max_steps and state["updates"] >= a.max_steps:
                    done = True
                    break
        if state["last_eval"] != state["updates"]:
            evaluate(epoch)
        if done:
            break
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if n in best["state"]:
                p.copy_(best["state"][n].to(p.device))
    return {"dev_before": dev_before, "log": history, "updates": state["updates"], "updates_planned": total,
            "best": {k: v for k, v in best.items() if k != "state"}, "train_seconds": round(time.time() - t0, 1)}


def profile(a):
    """--profile-batch ROWSxLEN: two training steps on a synthetic batch of that shape; prints peak memory."""
    import gc
    import numpy as np
    import torch
    from train_banking77 import collate
    rows, length = (int(x) for x in a.profile_batch.lower().split("x"))
    cfg, tok, model = load_base(a.base)
    device = pick_device(a.device)
    amp_dtype = pick_amp(a.amp, device)
    if a.base_dtype != "fp32" and amp_dtype is None:
        raise SystemExit("--base-dtype %s needs bf16 autocast (--amp bf16, or auto on a CUDA device with bf16)" % a.base_dtype)
    pm = prepare(model, a, device)
    rng = np.random.default_rng(a.seed)
    k = 6
    items = []
    for _ in range(rows):
        ids = rng.integers(10, tok.vocab_size, length).astype(np.int32)
        markers = [1 + 2 * j for j in range(k)]
        ids[0], ids[markers] = tok.cls_token_id, tok.mask_token_id
        items.append({"ids": ids, "markers": markers, "qtype": 0, "target": [1.0] + [0.0] * (k - 1)})
    batch = [x.to(device) for x in collate(items, tok.pad_token_id)]
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated() / 2 ** 20
    else:
        before = rss_mb()
        reset_peak_rss()
    times = []
    for _ in range(2):
        t = time.time()
        ids, att, mpos, mmask, qt, target = batch
        with autocast(device, amp_dtype):
            logits, act = pm(ids, att, mpos, mmask, qt)
        loss, _ = rlcd_loss(logits, act, mmask, qt, target, 0.4, 1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(round(time.time() - t, 2))
    if device.type == "cuda":
        peak = {"allocated_before_mb": round(before), "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20),
                "max_reserved_mb": round(torch.cuda.max_memory_reserved() / 2 ** 20)}
    else:
        peak = {"rss_before_mb": round(before), "peak_rss_mb": round(peak_rss_mb())}
    print(json.dumps({"profile_batch": "%dx%d" % (rows, length), "tokens": rows * length, "device": str(device),
                      "amp": str(amp_dtype), "base_dtype": a.base_dtype, "grad_checkpointing": a.grad_checkpointing,
                      "head_checkpointing": bool(model.head_checkpointing), "r": a.r, "step_seconds": times, **peak}),
          flush=True)
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 epilog="See the module docstring for the recipe and the RTX 3070 sizing.")
    ap.add_argument("base", help="Statim/Laya checkpoint directory (model.safetensors, rl_agent_config.json, encoder/, tokenizer/)")
    ap.add_argument("--mixture", help="built mixture (jsonl.gz rows {state, q, target, src, lang})")
    ap.add_argument("--category", required=True, choices=CATEGORIES)
    ap.add_argument("--registry", default=REGISTRY, help="v6-keep.json that maps a row's src to its category")
    ap.add_argument("--out", default=None, help="adapter directory (default models/lora/<category>)")
    ap.add_argument("--r", type=int, default=16, help="LoRA rank")
    ap.add_argument("--lora-alpha", type=float, default=32.0, help="LoRA alpha (scale alpha / r)")
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--init", choices=["true", "gaussian"], default="true",
                    help="init_lora_weights: true (B = 0, PEFT default) or gaussian")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup", type=float, default=0.06, help="fraction of the optimiser steps with linear warmup")
    ap.add_argument("--max-tokens", type=int, default=8192, help="padded tokens per micro-batch")
    ap.add_argument("--max-rows", type=int, default=64, help="rows per micro-batch")
    ap.add_argument("--accum", type=int, default=2, help="micro-batches per optimiser step")
    ap.add_argument("--dev-items", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=0, help="also evaluate dev every N optimiser steps (0 = per epoch only)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimiser steps (0 = all epochs)")
    ap.add_argument("--limit-items", type=int, default=0, help="use only the first N rows of the category (smoke runs)")
    ap.add_argument("--max-len", type=int, default=0, help="sequence length (default: the checkpoint's max_len)")
    ap.add_argument("--head-max-len", type=int, default=HEAD_MAX_LEN)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--amp", choices=["auto", "bf16", "off"], default="auto",
                    help="auto: bf16 autocast on a CUDA device with bf16 support, fp32 otherwise")
    ap.add_argument("--base-dtype", choices=["fp32", "fp16", "bf16"], default="fp32",
                    help="storage of the frozen encoder matrices (fp16/bf16 need bf16 autocast)")
    ap.add_argument("--no-grad-checkpointing", dest="grad_checkpointing", action="store_false")
    ap.add_argument("--no-head-checkpointing", dest="head_checkpointing", action="store_false")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--profile-batch", default=None, metavar="ROWSxLEN",
                    help="run two training steps on a synthetic batch of this shape, print peak memory, exit")
    a = ap.parse_args(argv)
    if not a.profile_batch and not a.mixture:
        ap.error("--mixture is required")
    if a.accum < 1 or a.max_rows < 1 or a.max_tokens < 1 or a.r < 1 or a.epochs < 1:
        ap.error("--accum, --max-rows, --max-tokens, --r and --epochs must be positive")
    a.out = a.out or os.path.join("models", "lora", a.category)
    return a


def main(argv=None):
    a = parse_args(argv)
    if a.profile_batch:
        return profile(a)
    t0 = time.time()
    chosen, enabled = category_sources(a.registry, a.category)
    if not chosen:
        raise SystemExit("no enabled registry entry of category %s in %s" % (a.category, a.registry))
    rows, stats = select_rows(a.mixture, chosen, enabled, a.limit_items)
    print("category %s: %d rows from %d of %d registry sources | rows of other categories %d | unknown src %d %s"
          % (a.category, stats["rows"], len(stats["per_source"]), len(chosen), stats["other_category_rows"],
             stats["unknown_src_rows"], stats["unknown_src_top"] or ""), flush=True)
    for src, n in stats["per_source"].items():
        print("  %6d  %s" % (n, src), flush=True)
    if len(rows) < 2:
        raise SystemExit("too few rows of category %s in %s" % (a.category, a.mixture))
    train_ix, dev_ix = split_dev(rows, a.dev_items)

    import torch
    import peft
    import transformers
    import laya
    torch.manual_seed(a.seed)
    base = os.path.abspath(a.base)
    cfg, tok, model = load_base(base)
    max_len = a.max_len or cfg.get("max_len", 512)
    train_items, drop_t = make_items(tok, [rows[i] for i in train_ix], max_len, a.head_max_len)
    dev_items, drop_d = make_items(tok, [rows[i] for i in dev_ix], max_len, a.head_max_len)
    per_source = {src: {"rows": n, "train": 0, "dev": 0} for src, n in stats["per_source"].items()}
    for part, items in (("train", train_items), ("dev", dev_items)):
        for it in items:
            per_source[it["src"]][part] += 1
    del rows
    print("items: %d train, %d dev (%d rows dropped: marker count) | max_len %d, head_max_len %d"
          % (len(train_items), len(dev_items), drop_t + drop_d, max_len, a.head_max_len), flush=True)
    if not train_items:
        raise SystemExit("no training items left")

    device = pick_device(a.device)
    amp_dtype = pick_amp(a.amp, device)
    if a.base_dtype != "fp32" and amp_dtype is None:
        raise SystemExit("--base-dtype %s needs bf16 autocast (--amp bf16, or auto on a CUDA device with bf16)" % a.base_dtype)
    pm = prepare(model, a, device)
    wrapped, n_trainable = pm.lora_counts
    print("LoRA r=%d alpha=%g on %d modules: %d trainable parameters | device %s, amp %s, base %s, checkpointing "
          "encoder %s head %s" % (a.r, a.lora_alpha, wrapped, n_trainable, device, amp_dtype, a.base_dtype,
                                  a.grad_checkpointing, model.head_checkpointing), flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    result = train(pm, model, train_items, dev_items, tok.pad_token_id, device, amp_dtype, a)

    os.makedirs(a.out, exist_ok=True)
    pm.save_pretrained(a.out, safe_serialization=True)
    with open(os.path.join(a.out, "adapter_config.json"), encoding="utf-8") as f:
        saved = json.load(f)
    if saved.get("bias") != "none" or saved.get("modules_to_save") or saved.get("use_dora"):
        raise SystemExit("saved adapter_config.json is not a plain LoRA: %s" % saved)
    peak = (round(torch.cuda.max_memory_allocated() / 2 ** 20) if device.type == "cuda" else round(peak_rss_mb()))
    record = {
        "category": a.category, "base": base, "base_sha256": sha256_file(os.path.join(base, "model.safetensors")),
        "mixture": os.path.abspath(a.mixture), "mixture_sha256": sha256_file(a.mixture),
        "registry": os.path.abspath(a.registry),
        "registry_sha256": sha256_file(a.registry), "sources": per_source,
        "selection": {k: v for k, v in stats.items() if k != "per_source"},
        "items": {"train": len(train_items), "dev": len(dev_items), "dropped_marker_mismatch": drop_t + drop_d},
        "max_len": max_len, "head_max_len": a.head_max_len,
        "lora": {"r": a.r, "lora_alpha": a.lora_alpha, "lora_dropout": a.lora_dropout, "init": a.init,
                 "target_modules": ["Wqkv", "Wo", "Wi"], "wrapped_modules": wrapped, "trainable_parameters": n_trainable},
        "device": str(device), "amp": str(amp_dtype), "base_dtype": a.base_dtype,
        "gradient_checkpointing": a.grad_checkpointing, "head_checkpointing": bool(model.head_checkpointing),
        "head_mode": "eval (frozen, dropout off)", **result,
        "peak_memory_mb": peak, "peak_memory_kind": "cuda max_memory_allocated" if device.type == "cuda" else "process VmHWM",
        "seconds": round(time.time() - t0, 1), "args": vars(a),
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "peft": peft.__version__,
                     "transformers": transformers.__version__, "laya": getattr(laya, "__version__", "?")},
    }
    with open(os.path.join(a.out, "train_lora.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, ensure_ascii=False)
    best = result["best"]
    print("saved %s (best: epoch %s, update %s, dev %s; before %s) in %.0fs, peak %d MB" % (
        a.out, best["epoch"], best["update"], best["dev_acc"], (result["dev_before"] or {}).get("dev_acc"),
        time.time() - t0, peak), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
