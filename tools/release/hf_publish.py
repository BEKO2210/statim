#!/usr/bin/env python3
"""Package a Statim Decide model for Hugging Face: GGUF files, the checkpoint, licences and a model
card whose every number is read from the gate evaluation (eval.json), never typed by hand.

    .venv-train/bin/python tools/release/hf_publish.py models/laya-multilingual-big1 \
        --name statim-decide-multilingual-base --version 0.4.0 --base models/laya-multilingual \
        --out dist/statim-decide-multilingual-base            # build and check
    ... --upload Beko2210/statim-decide-multilingual-base     # then publish

The card states the protocol of every number (split, rows, sampling) next to it.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GITHUB = "https://github.com/BEKO2210/statim"

MODELS = {
    "statim-decide-multilingual-base": {
        "display": "Statim Decide Multilingual Base",
        "encoder": "mmBERT-base", "encoder_id": "jhu-clsp/mmBERT-base", "encoder_licence": "MIT",
        "base_model": "convaiinnovations/laya-multilingual",
        "serve_key": "multilingual",
    },
    "statim-decide-en-large": {
        "display": "Statim Decide EN Large",
        "encoder": "ModernBERT-large", "encoder_id": "answerdotai/ModernBERT-large", "encoder_licence": "Apache-2.0",
        "base_model": "convaiinnovations/laya",
        "serve_key": "english",
    },
}
DATASETS = ["PolyAI/banking77", "AmazonScience/massive", "LocalLLaMA/typed-decisions",
            "tasksource/tasksource-jev-typed-decisions", "nvidia/Nemotron-Safety-Guard-Dataset-v3",
            "l3cube-pune/IndicGuard", "PolyAI/minds14", "benayas/snips"]
SUITES = {  # held-out key prefix -> (display name, protocol, trained on its train split?)
    "test/typed_decisions": ("typed-decisions", "test split, first 2,000 decisions; its train split is replay data", True),
    "test/banking77": ("Banking77", "test split, first 2,000 rows, all 77 intents in one question", True),
    "amazon_massive_intent": ("MASSIVE intents", "mean over 12 languages, 150 seeded stratified test rows each", True),
    "test/ag_news": ("AG News", "zero-shot (never trained on), first 2,000 test rows", False),
    "test/emotion": ("DAIR Emotion", "zero-shot, first 2,000 test rows", False),
    "hwu64": ("HWU64 intents", "English, 150 rows; rows overlapping MASSIVE removed", False),
    "sib200": ("SIB-200 topics", "zero-shot, mean over 4 languages, 150 rows each", False),
    "multilingual_sentiments": ("Sentiment", "zero-shot, mean over 12 languages, 150 rows each", False),
    "multi_hatecheck": ("HateCheck", "zero-shot, mean over 11 languages, 150 rows each", False),
    "belebele": ("Belebele reading", "zero-shot, mean over 4 languages, 150 rows each", False),
}
LANGS = ["ar", "de", "en", "es", "fr", "hi", "it", "ja", "pl", "ru", "tr", "zh", "id", "ms", "pt", "nl", "fa"]


def sh(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def suite_score(heldout, key):
    if key in heldout:
        return heldout[key]["acc"], 1
    rows = [v["acc"] for k, v in heldout.items() if k.startswith(key + "/")]
    return (sum(rows) / len(rows), len(rows)) if rows else (None, 0)


def gate_counts(base, model):
    gain = noise = loss = 0
    for k in sorted(set(base) & set(model)):
        x, y = base[k], model[k]
        se = math.sqrt(x["acc"] * (1 - x["acc"]) / x["n"] + y["acc"] * (1 - y["acc"]) / y["n"])
        d = y["acc"] - x["acc"]
        gain, noise, loss = (gain + 1, noise, loss) if d > 2 * se else (gain, noise, loss + 1) if d < -2 * se else (gain, noise + 1, loss)
    return gain, noise, loss


def card(a, meta, ev, base_ev, files):
    held, bheld = ev["heldout"], (base_ev or {}).get("heldout", {})
    rows, index = [], []
    for key, (name, proto, trained) in SUITES.items():
        s, n = suite_score(held, key)
        if s is None:
            continue
        b, _ = suite_score(bheld, key) if bheld else (None, 0)
        rows.append(f"| {name} | {'trained' if trained else 'held out'} | **{s:.4f}** | {'' if b is None else f'{b:.4f}'} | {proto} |")
        index.append({"task": {"type": "text-classification"},
                      "dataset": {"name": f"{name} ({proto})", "type": name.lower().replace(' ', '_')},
                      "metrics": [{"type": "accuracy", "value": round(s, 4)}]})
    g = gate_counts(bheld, held) if bheld else None
    tm = meta.get("training_multitask", {})
    per_suite = "\n".join(f"| `{k}` | {v['acc']:.4f} | {v['n']} |" for k, v in sorted(held.items()))
    files_md = "\n".join(f"| `{f}` | {size / 1e9:.2f} GB | {use} |" for f, size, use in files)
    front = {
        "license": "other", "license_name": "statim-weights",
        "license_link": f"https://huggingface.co/{a.upload or 'Beko2210/' + a.name}/blob/main/LICENSE-MODEL.md",
        "language": LANGS, "library_name": "gguf", "pipeline_tag": "zero-shot-classification",
        "base_model": a.info["base_model"], "datasets": DATASETS,
        "tags": ["statim", "gguf", "decision-making", "text-classification", "zero-shot-classification", a.info["encoder"]],
        "model-index": [{"name": a.name, "results": index}],
    }
    import yaml  # noqa: PLC0415 (only needed here)
    head = "---\n" + yaml.safe_dump(front, sort_keys=False, allow_unicode=True) + "---\n"
    gate_line = (f"Against the checkpoint it was trained from, on {sum(g)} held-out suites: **{g[0]} significant "
                 f"gains, {g[1]} within noise, {g[2]} regressions** (two combined binomial standard errors).") if g else ""
    return head + textwrap.dedent(f"""
    # {a.info['display']}

    A decision model for [Statim]({GITHUB}), the native C++ engine for typed decisions: ask any text a
    **choice**, a **score** or a **yes/no** question and get calibrated answers from one forward pass, on
    CPU or GPU, without Python at runtime. Version **{a.version}**, fine-tuned from
    [`{a.info['base_model']}`](https://huggingface.co/{a.info['base_model']}) ({a.info['encoder']} encoder).

    ## Quick start

    ```sh
    # Statim release binary: {GITHUB}/releases
    huggingface-cli download {a.upload or 'Beko2210/' + a.name} {a.name}-q8_0.gguf --local-dir models
    ./statim serve -m {a.info['serve_key']}=models/{a.name}-q8_0.gguf --port 8080
    curl -s localhost:8080/v1/systemone -d '{{
      "state": {{"subject": "Duplicate charge on invoice #4411",
                "body": "We were billed twice for March. Please refund the second charge."}},
      "questions": {{
        "department": {{"type": "choice", "instructions": "Which department should handle this?",
          "criteria": {{"billing": "invoices, refunds", "technical": "bugs, outages", "sales": "pricing, contracts"}}}},
        "urgency": {{"type": "score", "instructions": "How urgent is this request?",
          "criteria": ["not urgent", "soon", "critical"]}},
        "refund": {{"type": "noul", "instructions": "Does the user explicitly request a refund?"}}}}}}'
    ```

    Open `http://127.0.0.1:8080/` for the playground. API reference: [docs/API.md]({GITHUB}/blob/main/docs/API.md).

    ## Files

    | File | Size | Use |
    |---|---|---|
    {files_md}

    Checksums in `SHA256SUMS`. The f32 file reproduces the reference implementation within 1e-4 on
    Statim's parity tests; q8_0 is smaller and faster on CPU with slightly different logits.

    ## Evaluation

    Measured by Statim's no-harm gate ([`tools/finetune/gate.py`]({GITHUB}/blob/main/tools/finetune/gate.py))
    on held-out test data the model selection never looked at. {gate_line}

    | Suite | Role | This model | Base checkpoint | Protocol |
    |---|---|---|---|---|
    {chr(10).join(rows)}

    Published systems under the same protocol, for orientation: typed-decisions meraGPT 0.768,
    laya-typed-decisions 0.766, Jev 0.727; AG News zero-shot Laya 0.950, GPT-3 (CARP) 0.926, Jev 0.881;
    Banking77 supervised MPNet 0.941; MASSIVE XLM-R base 0.857 (full train set). Sources:
    [docs/ROADMAP.md]({GITHUB}/blob/main/docs/ROADMAP.md).

    <details><summary>All {len(held)} held-out suites</summary>

    | Suite | Accuracy | Rows |
    |---|---|---|
    {per_suite}

    </details>

    Reproduce these numbers: [REPRODUCE.md]({GITHUB}/blob/main/REPRODUCE.md).

    ## Training

    Multi-task fine-tuning with `train_multitask.py --clean` from checkpoint `{tm.get('base', '?')}`, best
    epoch `{tm.get('best_epoch', '?')}` selected on validation data only. Training data: only sources whose
    licence permits commercial use and imposes no ShareAlike or copyleft terms (Banking77, MASSIVE,
    typed-decisions replay, a licence-audited tasksource mixture, Nemotron-Safety-Guard, IndicGuard,
    MINDS-14, SNIPS), every one listed with its licence in
    [DATA_LICENSES.md](DATA_LICENSES.md). Evaluation test rows were removed from the training data.

    ## Intended use and limits

    - Classification-style decisions over short texts and JSON: routing, triage, moderation, intent,
      yes/no checks, ordinal ratings. It does not generate text.
    - Accuracy varies by task and language (see the table). Reading comprehension (Belebele) and
      semantic similarity are weak; do not use it for them without your own evaluation.
    - Use the confidence: Statim's `min_confidence` option marks low-confidence answers with
      `escalate: true` so a person can review them. Do not automate high-stakes decisions about people
      without human review.

    ## Licence

    The weights may be used under any one of: PolyForm Noncommercial 1.0.0, PolyForm Small Business
    1.0.0 (free commercial use below 100 people and 1 M USD revenue), PolyForm Free Trial 1.0.0 (any
    company, fewer than 32 days), or a Statim commercial licence
    ([COMMERCIAL.md]({GITHUB}/blob/main/COMMERCIAL.md)). Texts in [LICENSE-MODEL.md](LICENSE-MODEL.md).
    The Statim engine is Apache-2.0.

    Built on [Laya]({'https://huggingface.co/' + a.info['base_model']}) (Apache-2.0) and
    [{a.info['encoder']}](https://huggingface.co/{a.info['encoder_id']}) ({a.info['encoder_licence']}).
    Training data attribution: Banking77 (Casanueva et al., 2020, PolyAI), MASSIVE (FitzGerald et al.,
    2022, Amazon), and the CC-BY sources in DATA_LICENSES.md. Statim is independent and not affiliated
    with the Laya authors.
    """).replace("\n    ", "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--name", required=True, choices=sorted(MODELS))
    ap.add_argument("--version", required=True)
    ap.add_argument("--base", help="base checkpoint dir with eval.json, for the gate summary")
    ap.add_argument("--out", required=True)
    ap.add_argument("--upload", help="Hugging Face repo id; without it nothing is uploaded")
    a = ap.parse_args()
    a.info = MODELS[a.name]
    ev = json.load(open(os.path.join(a.model_dir, "eval.json")))
    base_ev = json.load(open(os.path.join(a.base, "eval.json"))) if a.base else None
    meta = json.load(open(os.path.join(a.model_dir, "rl_agent_config.json")))
    os.makedirs(a.out, exist_ok=True)
    py = os.path.join(ROOT, ".venv", "bin", "python")
    files = []
    for qtype, extra, use in (("f32", ["--embd-type", "f16"], "reference precision, exact on GPU"),
                              ("q8_0", [], "recommended for CPU: 4x smaller")):
        path = os.path.join(a.out, f"{a.name}-{qtype}.gguf")
        if not os.path.exists(path):
            sh([py, "tools/convert_laya.py", a.model_dir, "-o", path, "--type", qtype, *extra, "--name", a.name])
        files.append((os.path.basename(path), os.path.getsize(path), use))
    ck = os.path.join(a.out, "checkpoint")
    if not os.path.exists(ck):
        os.makedirs(ck)
        for rel in ("model.safetensors", "rl_agent_config.json", "encoder", "tokenizer"):
            src = os.path.join(a.model_dir, rel)
            (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, os.path.join(ck, rel))
    files.append(("checkpoint/", sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(ck) for f in fs),
                  "Laya-format checkpoint for fine-tuning and the Python reference"))
    os.makedirs(os.path.join(a.out, "evaluation"), exist_ok=True)
    shutil.copy2(os.path.join(a.model_dir, "eval.json"), os.path.join(a.out, "evaluation", "eval.json"))
    for f in ("LICENSE-MODEL.md", "DATA_LICENSES.md", "NOTICE"):
        shutil.copy2(os.path.join(ROOT, f), os.path.join(a.out, f))
    open(os.path.join(a.out, "README.md"), "w").write(card(a, meta, ev, base_ev, files))
    sums = []
    for d, _, fs in os.walk(a.out):
        for f in sorted(fs):
            p = os.path.join(d, f)
            if f != "SHA256SUMS":
                sums.append(f"{sha256(p)}  {os.path.relpath(p, a.out)}")
    open(os.path.join(a.out, "SHA256SUMS"), "w").write("\n".join(sorted(sums, key=lambda s: s[66:])) + "\n")
    print(f"built {a.out}: {len(sums)} files")
    if a.upload:
        from huggingface_hub import HfApi  # noqa: PLC0415
        api = HfApi()
        api.create_repo(a.upload, repo_type="model", private=False, exist_ok=True)
        info = api.upload_folder(repo_id=a.upload, folder_path=a.out, repo_type="model",
                                 commit_message=f"{a.info['display']} {a.version}")
        api.create_tag(a.upload, tag=f"v{a.version}", revision=info.oid, exist_ok=True)
        print(f"uploaded https://huggingface.co/{a.upload} (tag v{a.version})")


if __name__ == "__main__":
    sys.exit(main())
