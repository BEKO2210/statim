#!/usr/bin/env python3
"""Multi-task fine-tune of a Laya checkpoint: Banking77 + MASSIVE intents (51 languages) +
multilingual sentiment (12 languages), with typed-decisions replay and distillation replay.

Same recipe as train_banking77.py (RLCD loss, frozen token embeddings, best epoch by held-out dev,
temperatures refit on held-out items). Additions:

- Train splits only. Any train row whose text also occurs in a test split of the same dataset is
  dropped, so bench/eval_accuracy.py and bench/eval_multilingual.py stay clean.
- Instructions are drawn from a few paraphrases per task (the evaluation wording is one of them),
  so the model learns the task rather than one prompt string.
- Distillation questions that overlap a trained task (sentiment) are left out; anchoring them to
  the base model would pull against the task's gold labels.
- Best epoch = mean dev accuracy over the tasks (plus the held-out mixture slice with --mixture).
- Optional per-epoch task budgets (--budget, temperature-style mixing as in T5 / UniMax: small tasks
  are up-sampled, large ones sub-sampled per epoch), linear warmup (--warmup), and an exponential
  moving average of the weights (--ema), evaluated next to the raw weights every epoch.

    .venv-train/bin/python tools/finetune/train_multitask.py models/laya-multilingual models/laya-multilingual-multitask
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_banking77 import (BANK_HEAD, DISTILL_QUESTIONS, SEED, bank_items, batches,  # noqa: E402
                             collate, distill_items, fit_temp, predict, typed_items)
from source_policy import validate_mixture_dev  # noqa: E402

from laya.agent import _fix_tokenizer_config, _load_tokenizer  # noqa: E402
from laya.common import QTYPES, build_model, build_sequence, proper_reward, render_options, temp_bucket  # noqa: E402

MASSIVE_URL = "https://huggingface.co/datasets/mteb/amazon_massive_intent/resolve/main/%s/%s.json.gz"
SENT_URL = "https://raw.githubusercontent.com/tyqiangz/multilingual-sentiment-datasets/main/data/%s/%s.csv"
SENT_LANGS = ["arabic", "chinese", "english", "french", "german", "hindi", "indonesian", "italian",
              "japanese", "malay", "portuguese", "spanish"]
SENT_OPTIONS = ["negative", "neutral", "positive"]
INTENT_INSTR = ["Which intent does `utterance` express?", "What does the user want in `utterance`?",
                "Classify the intent of `utterance`."]
SENT_INSTR = ["What is the sentiment of `text`?", "What sentiment does the author of `text` express?",
              "Is `text` negative, neutral or positive?"]


def load(kind, url):
    from datasets import load_dataset
    return list(load_dataset(kind, data_files={"x": url}, split="x"))


def choice_item(tok, state, instr, keys, gold, max_len, head_max_len, src):
    q = {"t": "choice", "ins": instr, "crit": {k: None for k in keys}}
    seq, markers = build_sequence(tok, state, q, max_len, head_max_len)
    if len(markers) != len(keys):
        return None
    return {"ids": np.asarray(seq, dtype=np.int32), "markers": markers, "qtype": QTYPES["choice"],
            "target": [1.0 if k == gold else 0.0 for k in keys], "src": src}


V6_CATEGORIES = {"1": "sentiment", "2": "emotion", "3": "complaint", "4": "nli", "5": "safety", "6": "reading",
                 "7": "similarity", "8": "topic", "9": "intent", "10": "other"}


def v6_categories():
    """Source id -> category family for mixture v6 rows (src "v6/<id>/<config>"), read from the registry
    so the mixture file itself stays plain."""
    reg = os.path.join(os.path.dirname(__file__), "sources", "v6-keep.json")
    if not os.path.exists(reg):
        return {}
    out = {}
    for e in json.load(open(reg)):
        num = e.get("category", "").split(";")[0].strip().split("-")[0]
        out[e["id"]] = V6_CATEGORIES.get(num, "other")
    return out


def mixture_src(row, cats):
    """Budget key of a mixture row: "mixture" for v5 rows, "mixture:<category>" for v6 rows,
    "mixture:synth" for synthetic gap data."""
    src = row.get("src", "")
    if src.startswith("synth"):
        return "mixture:synth"  # local-generator gap data gets its own share of the mixture budget
    if not src.startswith("v6/"):
        return "mixture"
    sid = src[3:].rsplit("/", 1)[0]
    return "mixture:" + (row.get("category") or cats.get(sid, "other"))


def state_key(state):
    """Held-out comparison key of a mixture state; the same key filters train rows and distillation texts."""
    return " ".join(json.dumps(state, sort_keys=True, ensure_ascii=False).split()).lower()


def mixture_data(tok, path, max_len, head_max_len, n_dev, limit, rng):
    """Broad typed-decision mixture from build_mixture.py or mixture_v6/build.py. The first n_dev items
    are held out (v6 writes its dev slice first), and every later item whose state also occurs in the
    held-out slice is dropped (several questions of one source can share a document). Returns
    (train, dev, dev_states), dev_states being the state_key() of every held-out state."""
    import gzip
    import itertools
    train, dev, dev_states, leaked = [], [], set(), 0
    cats = v6_categories()
    # Streamed: mixture v7 has ~1M rows, which as parsed dicts alone would take ~2 GB of RAM.
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in itertools.islice(f, n_dev + limit) if limit else f:
            r = json.loads(line)
            in_dev = len(dev) < n_dev
            if not in_dev and state_key(r["state"]) in dev_states:
                leaked += 1
                continue
            qd = r["q"]
            q = {"t": qd["type"], "ins": qd["instructions"], "crit": qd.get("criteria") or ({} if qd["type"] == "noul" else None)}
            seq, markers = build_sequence(tok, r["state"], q, max_len, head_max_len)
            if len(markers) != len(render_options(q)) or len(markers) != len(r["target"]):
                continue
            it = {"ids": np.asarray(seq, dtype=np.int32), "markers": markers, "qtype": QTYPES[q["t"]], "target": r["target"],
                  "src": mixture_src(r, cats)}
            if in_dev:
                dev.append(it)
                dev_states.add(state_key(r["state"]))
            else:
                train.append(it)
    by_cat = {}
    for it in train:
        by_cat[it["src"]] = by_cat.get(it["src"], 0) + 1
    print(f"mixture: {len(train)} train, {len(dev)} dev, {leaked} train items dropped (state in dev)"
          + (f" | {by_cat}" if len(by_cat) > 1 else ""), flush=True)
    return train, dev, dev_states


def massive_data(tok, langs, per_lang, dev_langs, dev_per_lang, max_len, rng):
    train, dev, dropped = [], [], 0
    for lang in langs:
        rows = load("json", MASSIVE_URL % ("train", lang))
        test_texts = {r["text"] for r in load("json", MASSIVE_URL % ("test", lang))}
        clean = [r for r in rows if r["text"] not in test_texts]
        dropped += len(rows) - len(clean)
        rng.shuffle(clean)
        labels = sorted({r["label_text"].replace("_", " ") for r in rows})
        n_dev = dev_per_lang if lang in dev_langs else 0
        for i, r in enumerate(clean[:per_lang + n_dev]):
            gold = r["label_text"].replace("_", " ")
            if i < n_dev:  # dev: evaluation wording and sorted option order
                it = choice_item(tok, {"utterance": r["text"]}, INTENT_INSTR[0], labels, gold, max_len, BANK_HEAD, "massive")
                (dev.append(it) if it else None)
            else:
                keys = list(labels)
                rng.shuffle(keys)
                it = choice_item(tok, {"utterance": r["text"]}, rng.choice(INTENT_INSTR), keys, gold, max_len, BANK_HEAD, "massive")
                (train.append(it) if it else None)
    return train, dev, dropped


def sentiment_data(tok, per_lang, dev_per_lang, max_len, head_max_len, rng):
    train, dev, dropped = [], [], 0
    for lang in SENT_LANGS:
        rows = [r for r in load("csv", SENT_URL % (lang, "train")) if r["text"] and r["label"] in SENT_OPTIONS]
        test_texts = {r["text"] for r in load("csv", SENT_URL % (lang, "test"))}
        clean = [r for r in rows if r["text"] not in test_texts]
        dropped += len(rows) - len(clean)
        rng.shuffle(clean)
        for i, r in enumerate(clean[:per_lang + dev_per_lang]):
            if i < dev_per_lang:
                it = choice_item(tok, {"text": r["text"]}, SENT_INSTR[0], SENT_OPTIONS, r["label"], max_len, head_max_len, "sentiment")
                (dev.append(it) if it else None)
            else:
                keys = list(SENT_OPTIONS)
                rng.shuffle(keys)
                it = choice_item(tok, {"text": r["text"]}, rng.choice(SENT_INSTR), keys, r["label"], max_len, head_max_len, "sentiment")
                (train.append(it) if it else None)
    return train, dev, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr-encoder", type=float, default=2.5e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--massive-per-lang", type=int, default=400)
    ap.add_argument("--optim", choices=["adamw", "adamw8bit"], default="adamw")
    ap.add_argument("--max-len", type=int, default=0, help="override the checkpoint's max_len (e.g. 1024 for ModernBERT)")
    ap.add_argument("--massive-langs", default=None, help="comma-separated MASSIVE languages (default: all 51)")
    ap.add_argument("--sentiment-per-lang", type=int, default=1200)
    ap.add_argument("--distill", type=int, default=6000)
    ap.add_argument("--budget", default=None,
                    help="items per epoch per task, e.g. banking77=12000,massive=12000,sentiment=8000,typed=4000,distill=3000")
    ap.add_argument("--warmup", type=float, default=0.0, help="fraction of steps with linear LR warmup")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of the trainable weights (0 = off)")
    ap.add_argument("--tag", default="multitask")
    ap.add_argument("--patience", type=int, default=0,
                    help="stop after this many epochs without a new best dev mean (0 = run all epochs)")
    ap.add_argument("--mixture", default=None, help="gzipped JSONL from build_mixture.py")
    ap.add_argument("--mixture-dev", type=int, default=1000)
    ap.add_argument("--mixture-limit", type=int, default=0, help="use at most this many mixture items (0 = all)")
    ap.add_argument("--mixture-temperature", type=float, default=2.0,
                    help="with --budget mixture=N and a v6 mixture: split N over the categories in proportion to"
                         " size^(1/T) (T5-style temperature mixing; 1 = proportional, large = uniform)")
    ap.add_argument("--train-seed", type=int, default=0,
                    help="seed for initialisation and batch order only (0 = SEED); data splits keep SEED, so seeds stay comparable")
    ap.add_argument("--calibrate-only", action="store_true",
                    help="no training: refit the temperatures of an already trained model (e.g. a soup) on the"
                         " same dev items training uses, and save it")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="keep activations instead of recomputing them (faster; needs a large GPU, e.g. 80 GB)")
    ap.add_argument("--clean", action="store_true",
                    help="commercial-clean data only: no tyqiangz sentiment (aggregates research-only corpora),"
                         " distillation texts from the licence-filtered mixture instead of AG News / tweet_eval")
    a = ap.parse_args()
    validate_mixture_dev(a.mixture, a.mixture_dev)
    from datasets import load_dataset

    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    device = torch.device("cuda")
    base = os.path.abspath(a.base)
    _fix_tokenizer_config(base)
    cfg = json.load(open(os.path.join(base, "rl_agent_config.json")))
    tok = _load_tokenizer(os.path.join(base, "tokenizer"), cfg)
    dtype = torch.bfloat16 if cfg.get("amp_dtype") == "bf16" else torch.float16
    if a.max_len:
        cfg["max_len"] = a.max_len  # saved with the model, so inference uses the same budget
    max_len, hml = cfg.get("max_len", 512), cfg.get("head_max_len", 192)

    # ---- data
    bank = list(load_dataset("mteb/banking77", split="train"))
    labels77 = sorted({r["label_text"].replace("_", " ") for r in bank})
    rng.shuffle(bank)
    bank_dev = bank_items(tok, bank[:500], labels77, max_len, rng, shuffle_options=False)
    train = bank_items(tok, bank[500:], labels77, max_len, rng)
    langs = [x["path"].split("/")[-1].replace(".json.gz", "") for x in json.load(
        __import__("urllib.request", fromlist=["urlopen"]).urlopen(
            "https://huggingface.co/api/datasets/mteb/amazon_massive_intent/tree/main/train"))]
    if a.massive_langs:
        langs = [l for l in langs if l in set(a.massive_langs.split(","))]
    eval_langs = {"de", "en", "fr", "es", "it", "tr", "pl", "ru", "ja", "zh-CN", "ar", "hi"} & set(langs)
    m_train, m_dev, m_drop = massive_data(tok, langs, a.massive_per_lang, eval_langs, 50, max_len, rng)
    if a.clean:
        if not a.mixture:
            raise SystemExit("--clean needs --mixture (licence-filtered data from build_mixture.py)")
        s_train, s_dev, s_drop = [], [], 0
    else:
        s_train, s_dev, s_drop = sentiment_data(tok, a.sentiment_per_lang, 50, max_len, hml, rng)
    train += m_train + s_train
    td = list(load_dataset("LocalLLaMA/typed-decisions", "all", split="train"))
    rng.shuffle(td)
    typed_dev = typed_items(tok, td[:len(td) // 10], max_len, hml)
    train += typed_items(tok, td[len(td) // 10:], max_len, hml)
    x_dev, x_dev_states = [], set()
    if a.mixture:
        x_train, x_dev, x_dev_states = mixture_data(tok, a.mixture, max_len, BANK_HEAD, a.mixture_dev, a.mixture_limit, rng)
        train += x_train

    model = build_model(cfg, encoder_dir=os.path.join(base, "encoder"))
    model.load_state_dict(load_file(os.path.join(base, "model.safetensors")), strict=True)
    model.to(device)
    if a.distill:
        import train_banking77 as tb
        tb.DISTILL_QUESTIONS[:] = [q for q in DISTILL_QUESTIONS if "sentiment" not in q[1]]
        if a.clean:
            import gzip
            with gzip.open(a.mixture, "rt", encoding="utf-8") as f:  # never a held-out dev state
                texts = [t for t in (json.loads(line)["state"] for line in f)
                         if isinstance(t, str) and 20 < len(t) < 2000 and state_key(t) not in x_dev_states]
            rng.shuffle(texts)
            texts = texts[:a.distill]
        else:
            tw = [r["text"] for r in load_dataset("cardiffnlp/tweet_eval", "sentiment", split="train")]
            news = [r["text"] for r in load_dataset("fancyzhx/ag_news", split="train")]
            rng.shuffle(tw)
            rng.shuffle(news)
            texts = tw[:a.distill * 3 // 4] + news[:a.distill - a.distill * 3 // 4]
        dist = distill_items(tok, texts, max_len, hml, rng)
        zs = predict(model, [dict(it, target=[0.0] * len(it["markers"])) for it in dist], tok.pad_token_id, device, dtype)
        for it, z in zip(dist, zs):
            it["target"] = torch.softmax(torch.tensor(z), -1).tolist()
        train += dist
    for it in train:  # int32 token arrays: ~8x less RAM than Python int lists (full MASSIVE is ~576k items)
        if not isinstance(it["ids"], np.ndarray):
            it["ids"] = np.asarray(it["ids"], dtype=np.int32)
    counts = {}
    for it in train:
        counts[it["src"]] = counts.get(it["src"], 0) + 1
    print(f"train {len(train)} {counts} | dev banking {len(bank_dev)}, massive {len(m_dev)}, sentiment {len(s_dev)}, "
          f"typed {len(typed_dev)} | dropped as test duplicates: massive {m_drop}, sentiment {s_drop}", flush=True)

    if a.train_seed:  # data and dev splits above used SEED; only initialisation and batch order change
        torch.manual_seed(a.train_seed)
        rng = random.Random(a.train_seed)
    # ---- optimiser
    if not a.no_grad_ckpt:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    model.train()
    enc_params, head_params = [], []
    for n, p in model.named_parameters():
        if "embeddings.tok_embeddings" in n:
            p.requires_grad_(False)
        elif n.startswith("encoder."):
            enc_params.append(p)
        else:
            head_params.append(p)
    groups = [{"params": enc_params, "lr": a.lr_encoder}, {"params": head_params, "lr": a.lr_head}]
    if a.optim == "adamw8bit":
        # 8-bit optimizer states (Dettmers et al., 2022): ~2 GB less for ModernBERT-large, which
        # otherwise does not fit an 8 GB GPU; same training quality reported as 32-bit AdamW
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(groups, weight_decay=0.01)
    else:
        opt = torch.optim.AdamW(groups, weight_decay=0.01)
    by_src = {}
    for i, it in enumerate(train):
        by_src.setdefault(it["src"], []).append(i)
    budget = {k: int(v) for k, v in (kv.split("=") for kv in a.budget.split(","))} if a.budget else None
    if budget and "mixture" in budget:
        # v6: one number for the whole mixture, shared over its categories by temperature mixing;
        # explicit mixture:<category>=N entries win.
        cats_n = {k: len(v) for k, v in by_src.items() if k.startswith("mixture:") and k not in budget}
        if cats_n:
            t = max(a.mixture_temperature, 1e-6)
            w = {k: n ** (1.0 / t) for k, n in cats_n.items()}
            tot = sum(w.values())
            for k in cats_n:
                budget[k] = max(1, int(round(budget["mixture"] * w[k] / tot)))
            print("mixture budget per category:", {k: budget[k] for k in sorted(cats_n)}, flush=True)

    def epoch_indices():
        if not budget:
            return list(range(len(train)))
        out = []
        for src, idx in by_src.items():
            want = budget.get(src, len(idx))
            pool = []
            while len(pool) < want:  # whole passes first, then a random remainder
                chunk = list(idx)
                rng.shuffle(chunk)
                pool += chunk
            out += pool[:want]
        return out

    per_epoch = math.ceil(len(batches([train[i] for i in epoch_indices()], a.max_tokens, 64)) / a.accum)
    total = per_epoch * a.epochs
    warm = int(total * a.warmup)
    floor = 1e-6 / a.lr_encoder

    def lr_factor(step):
        if step < warm:
            return (step + 1) / warm
        t = (step - warm) / max(1, total - warm)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    ema = {n: p.detach().clone() for n, p in trainable} if a.ema else None

    def swap_ema():
        # exchange live and EMA weights in place (call twice to restore)
        with torch.no_grad():
            for n, p in trainable:
                tmp = p.detach().clone()
                p.copy_(ema[n])
                ema[n].copy_(tmp)

    def acc(items):
        z = predict(model, items, tok.pad_token_id, device, dtype)
        return sum(max(range(len(l)), key=l.__getitem__) == max(range(len(it["target"])), key=it["target"].__getitem__)
                   for l, it in zip(z, items)) / len(items)

    def dev_scores():
        d = {"banking77": round(acc(bank_dev), 4), "massive": round(acc(m_dev), 4)}
        if s_dev:
            d["sentiment"] = round(acc(s_dev), 4)
        if x_dev:
            d["mixture"] = round(acc(x_dev), 4)
            cats = sorted({it["src"] for it in x_dev if it["src"].startswith("mixture:")})
            if cats:  # per-category dev accuracy is logged, not part of the model-selection mean
                print("  mixture dev by category:", {c.split(":", 1)[1]: round(acc([it for it in x_dev if it["src"] == c]), 3) for c in cats}, flush=True)
        return d

    d0 = dev_scores()
    print("dev before:", d0, flush=True)
    best, best_epoch, log = -1.0, 0, []
    os.makedirs(a.out, exist_ok=True)
    tmp_best = os.path.join(a.out, "best.safetensors")
    t0 = time.time()
    since_best = 0
    if a.calibrate_only:  # the weights are final; only the temperatures below are refitted
        save_file({k: v.contiguous().cpu() for k, v in model.state_dict().items()}, tmp_best)
        best, best_epoch = sum(d0.values()) / len(d0), "calibrate-only"
    for epoch in range(0 if a.calibrate_only else a.epochs):
        sigma = 0.4 + (0.1 - 0.4) * (epoch / max(1, a.epochs - 1))
        ep_idx = epoch_indices()
        bl = [[ep_idx[j] for j in b] for b in batches([train[i] for i in ep_idx], a.max_tokens, 64)]
        rng.shuffle(bl)
        opt.zero_grad(set_to_none=True)
        run_loss, n = 0.0, 0
        for step, b in enumerate(bl):
            ids, att, mpos, mmask, qt, target = (x.to(device) for x in collate([train[i] for i in b], tok.pad_token_id))
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
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mmask, -1e4), -1)).sum(-1).mean()
            loss = (-(adv * logp).mean() + loss_ce) / a.accum + 0.0 * act.sum()
            loss.backward()
            if (step + 1) % a.accum == 0 or step + 1 == len(bl):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                if ema is not None:
                    with torch.no_grad():
                        for nm, p in trainable:
                            ema[nm].mul_(a.ema).add_(p.detach(), alpha=1 - a.ema)
            run_loss += loss_ce.item()
            n += 1
            if step % 500 == 0:
                print(f"  epoch {epoch + 1} step {step}/{len(bl)} ce {run_loss / n:.4f} {time.time() - t0:.0f}s", flush=True)
        improved = False
        cands = [("raw", dev_scores())]
        if ema is not None:
            swap_ema()
            cands.append(("ema", dev_scores()))
            swap_ema()
        for kind, d in cands:
            mean = sum(d.values()) / len(d)
            log.append({"epoch": epoch + 1, "weights": kind, "train_ce": round(run_loss / n, 4), **d,
                        "mean": round(mean, 4), "seconds": round(time.time() - t0)})
            print(f"=== epoch {epoch + 1} [{kind}]: ce {run_loss / n:.4f} | dev {d} mean {mean:.4f} | "
                  f"{time.time() - t0:.0f}s", flush=True)
            if mean > best:
                improved = True
                best, best_epoch = mean, f"{epoch + 1}/{kind}"
                if kind == "ema":
                    swap_ema()
                save_file({k: v.contiguous().cpu() for k, v in model.state_dict().items()}, tmp_best)
                if kind == "ema":
                    swap_ema()
        since_best = 0 if improved else since_best + 1
        if a.patience and since_best >= a.patience:
            print(f"early stop: no new best dev mean for {since_best} epoch(s); best {best:.4f} at {best_epoch}", flush=True)
            break

    model.load_state_dict(load_file(tmp_best), strict=True)
    os.remove(tmp_best)
    calib = bank_dev + m_dev + s_dev + typed_dev + x_dev
    zs = predict(model, calib, tok.pad_token_id, device, dtype)
    by_type, by_bucket = {}, {}
    for z, it in zip(zs, calib):
        by_type.setdefault(it["qtype"], []).append((z, it["target"]))
        by_bucket.setdefault(temp_bucket(it["qtype"], len(z)), []).append((z, it["target"]))
    temps = list(cfg.get("temperature", [1.0, 1.0, 1.0]))
    for qt, pairs in by_type.items():
        if (t := fit_temp(pairs)):
            temps[qt] = t
    tbo = {b: t for b, pairs in by_bucket.items() if (t := fit_temp(pairs))}
    print("temperatures:", [round(t, 3) for t in temps], {k: round(v, 3) for k, v in tbo.items()}, flush=True)
    save_file({k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}, os.path.join(a.out, "model.safetensors"))
    for sub in ("encoder", "tokenizer"):
        shutil.copytree(os.path.join(base, sub), os.path.join(a.out, sub), dirs_exist_ok=True)
    cfg.update({"head_max_len": BANK_HEAD, "temperature": temps, "temperature_by_options": tbo,
                "model_name": "laya-multilingual-" + a.tag, "fine_tuned": True,
                "training_multitask": {"base": os.path.basename(base.rstrip("/")), "best_epoch": best_epoch,
                                       "dev_before": d0, "log": log, "counts": counts, "distill": a.distill,
                                       "massive_per_lang": a.massive_per_lang, "massive_langs": a.massive_langs, "sentiment_per_lang": a.sentiment_per_lang,
                                       "lr": [a.lr_encoder, a.lr_head], "epochs": a.epochs, "seed": SEED,
                                       "budget": budget, "warmup": a.warmup, "ema": a.ema, "patience": a.patience, "optim": a.optim,
                                       "mixture": a.mixture and os.path.basename(a.mixture), "clean": a.clean}})
    json.dump(cfg, open(os.path.join(a.out, "rl_agent_config.json"), "w"), indent=2)
    print(f"saved {a.out} (best epoch {best_epoch}, dev mean {best:.4f})", flush=True)


if __name__ == "__main__":
    main()
