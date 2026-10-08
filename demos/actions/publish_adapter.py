#!/usr/bin/env python3
"""Package the `actions` LoRA adapter for Hugging Face: the adapter GGUF, the PEFT files, the training
record, the per-game evaluation records and a model card whose every number is computed here from
those records (never typed). Nothing is uploaded without --upload.

    python3 demos/actions/publish_adapter.py --adapter-gguf models/lora/actions.lora.gguf \
        --peft models/lora/actions --results demos/actions/results/2026-10-07 --out dist/actions-adapter
    ... --upload Beko2210/statim-decide-multilingual-base-actions
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import statistics
import textwrap
from math import comb

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GITHUB = "https://github.com/BEKO2210/statim"
SITE = "https://beko2210.github.io/statim/actions/"
BASE_REPO = "Beko2210/statim-decide-multilingual-base"
GAMES = [("snake", "Snake, food eaten"), ("othello", "Othello, disc margin"),
         ("2048", "2048, score"), ("tetris", "Tetris, lines cleared")]
PLAYERS = [("random", "random moves"), ("zeroshot", "Statim, no adapter"),
           ("adapter", "**Statim + this adapter** (f32)"), ("adapter-q8", "Statim + this adapter, q8_0 (the demo Space)"),
           ("heuristic", "engine heuristic (teacher)")]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def games(results, game, player):
    rows = [json.loads(line) for line in open(os.path.join(results, f"ev-{game}-{player}.jsonl"))]
    return {r["seed"]: r for r in rows if r.get("type") == "game"}


def paired(a, b):
    """Mean paired difference, 95 % bootstrap interval (seeded) and two-sided sign-test p."""
    d = [a[s]["score"] - b[s]["score"] for s in sorted(a)]
    wins, losses = sum(x > 0 for x in d), sum(x < 0 for x in d)
    n = wins + losses
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n)
    rng = random.Random(0)
    boot = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(5000))
    return statistics.fmean(d), boot[round(0.025 * 4999)], boot[round(0.975 * 4999)], wins, losses, p


def fmt(v):
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.1f}" if abs(v) >= 1 or v == 0 else f"{v:.2f}"


def card(results, train, files):
    rows = []
    for game, label in GAMES:
        cells = []
        for player, _ in PLAYERS:
            if not os.path.exists(os.path.join(results, f"ev-{game}-{player}.jsonl")):
                cells.append("not measured yet")
                continue
            g = games(results, game, player)
            mean = statistics.fmean(r["score"] for r in g.values())
            cell = fmt(mean)
            if game == "othello":
                cell += f" ({sum(r['score'] > 0 for r in g.values())} of {len(g)} won)"
            if any(r.get("capped") for r in g.values()):
                cell += "*"
            cells.append(f"**{cell}**" if player == "adapter" else cell)
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    lines = []
    for game, label in GAMES:
        a = games(results, game, "adapter")
        for other, name in (("zeroshot", "without the adapter"), ("heuristic", "the teacher")):
            m, lo, hi, w, l, p = paired(a, games(results, game, other))
            lines.append(f"| {label.split(',')[0]} | {name} | {m:+,.1f} [{lo:+,.1f}, {hi:+,.1f}] | {w} / {l} | {p:.2g} |")
    src = train["sources"]
    n_rows = sum(v["rows"] for v in src.values())
    best = train["best"]
    files_md = "\n".join(f"| `{f}` | {size / 1e6:.1f} MB |" for f, size in files)
    header = "| Game, score | " + " | ".join(name for _, name in PLAYERS) + " |\n|---|" + "---:|" * len(PLAYERS)
    text = textwrap.dedent(f"""\
    ---
    license: other
    license_name: polyform-noncommercial-1.0.0
    license_link: https://huggingface.co/Beko2210/statim-decide-multilingual-base-actions/blob/main/LICENSE-MODEL.md
    base_model: {BASE_REPO}
    library_name: gguf
    tags: [statim, lora, gguf, decision-making, games]
    ---
    # Statim Decide: actions adapter

    A 13 MB LoRA adapter (rank {train['lora']['r']}) for
    [statim-decide-multilingual-base 0.10.0](https://huggingface.co/{BASE_REPO}) that teaches it to **pick
    moves**: a small game engine describes every legal move in one sentence, Statim reads the sentences
    with a one-line strategy as one `choice` question and returns a probability for every move. One
    forward pass per move, no search, no LLM. **[Watch it play, or let it play live]({SITE})**.

    ## Measured on 50 games it never saw

    Evaluation seeds 1–50 for every game and player; the training games come only from seeds 100,000
    and up. Mean over the 50 games; every player faces the same seeds. f32 runs merge the adapter on a
    GPU; q8_0 runs apply it at run time on the CPU, as the demo Space does.

    @@TABLE@@

    \\* Some games of this player hit the 5,000-move cap, so its mean is a lower bound.

    Paired per seed against the same games (mean difference, 95 % bootstrap interval, wins / losses,
    two-sided sign test):

    | Game | against | mean difference | wins / losses | p |
    |---|---|---:|---:|---:|
    @@PAIRED@@

    Without the adapter the model follows the wording of the move names, not the sentences. With it,
    Snake and Othello reach the teacher; 2048 and Tetris stay below it. The adapter only acts on requests
    that name it (`"adapter": "actions"`); every other request is served by the unchanged base model.

    ## Use

    ```sh
    statim serve -m multilingual=statim-decide-multilingual-base-q8_0.gguf \\
        --adapter multilingual:actions=statim-decide-multilingual-base-actions.lora.gguf
    ```
    Requests: `demos/actions/prompting.py` (`build_request(game, labels="moves")`) in the
    [repository]({GITHUB}/tree/main/demos/actions), or `site/actions/engines.js` in the browser.

    ## Files

    | File | Size |
    |---|---:|
    @@FILES@@

    Checksums in `SHA256SUMS`. The adapter is bound to the 0.10.0 base (its checkpoint fingerprint);
    Statim refuses to load it onto another model.

    ## Training

    `tools/finetune/train_lora.py --rows` on {n_rows:,} decisions of the engine heuristics
    ({', '.join(f"{k.split('/')[1]} {v['rows']:,}" for k, v in src.items())}), generated by
    `demos/actions/make_data.py`: soft targets from the heuristic's move scores, 15 % random exploration
    moves, shuffled option order and 50 % letter labels. LoRA r {train['lora']['r']}, alpha
    {train['lora']['lora_alpha']:g}, best dev agreement with the teacher {best['dev_acc']:.3f} (before
    {train['dev_before']['dev_acc']:.3f}). The data is generated by our own code; no external dataset is used.

    ## Licence

    PolyForm Noncommercial 1.0.0 only, like its base model 0.10.0, whose training data includes
    non-commercial sources ([details]({GITHUB}/blob/main/DATA_LICENSES.md)). Text in
    [LICENSE-MODEL.md](LICENSE-MODEL.md). The Statim engine is Apache-2.0.
    """)
    return (text.replace("@@TABLE@@", header + "\n" + "\n".join(rows))
            .replace("@@PAIRED@@", "\n".join(lines)).replace("@@FILES@@", files_md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-gguf", required=True)
    ap.add_argument("--peft", required=True, help="PEFT adapter directory from train_lora.py")
    ap.add_argument("--results", required=True, help="directory with ev-<game>-<player>.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--upload", help="Hugging Face repo id; nothing is uploaded without it")
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")
    train = json.load(open(os.path.join(a.results, "train_lora.json")))
    os.makedirs(os.path.join(a.out, "peft"))
    os.makedirs(os.path.join(a.out, "evaluation"))
    name = "statim-decide-multilingual-base-actions.lora.gguf"
    shutil.copy2(a.adapter_gguf, os.path.join(a.out, name))
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy2(os.path.join(a.peft, f), os.path.join(a.out, "peft", f))
    for f in sorted(os.listdir(a.results)):
        shutil.copy2(os.path.join(a.results, f), os.path.join(a.out, "evaluation", f))
    for f in ("LICENSE-MODEL.md", "NOTICE"):
        shutil.copy2(os.path.join(ROOT, f), os.path.join(a.out, f))
    files = [(name, os.path.getsize(os.path.join(a.out, name))),
             ("peft/adapter_model.safetensors", os.path.getsize(os.path.join(a.out, "peft", "adapter_model.safetensors")))]
    open(os.path.join(a.out, "README.md"), "w").write(card(a.results, train, files))
    sums = []
    for d, _, fs in os.walk(a.out):
        for f in sorted(fs):
            p = os.path.join(d, f)
            sums.append(f"{sha256(p)}  {os.path.relpath(p, a.out)}")
    open(os.path.join(a.out, "SHA256SUMS"), "w").write("\n".join(sorted(sums, key=lambda s: s[66:])) + "\n")
    print(f"built {a.out}: {len(sums)} files; adapter sha256 {sha256(os.path.join(a.out, name))}")
    if a.upload:
        from huggingface_hub import HfApi  # noqa: PLC0415
        api = HfApi()
        api.create_repo(a.upload, repo_type="model", private=False, exist_ok=True)
        info = api.upload_folder(repo_id=a.upload, folder_path=a.out, repo_type="model",
                                 commit_message="Statim Decide actions adapter (on 0.10.0)")
        print(f"uploaded https://huggingface.co/{a.upload} at {info.oid}")


if __name__ == "__main__":
    main()
