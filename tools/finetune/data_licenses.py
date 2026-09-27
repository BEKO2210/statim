#!/usr/bin/env python3
"""Generate DATA_LICENSES.md: every data source behind released Statim weights, with its licence.

    python3 tools/finetune/data_licenses.py data/mixture-v5.manifest.json tools/finetune/licence_audit.json \
        --v6 data/mixture-v6.manifest.json --synth data/synth-v2.manifest.json > DATA_LICENSES.md
"""
import argparse
import json
import re

POLICY = """\
Released Statim weights are trained only on data whose licence permits commercial use **and** does
not impose ShareAlike or copyleft terms on the model: Apache-2.0, MIT, BSD, CC0, CC-BY, ODC-By,
AFL-3.0. Excluded: non-commercial or research-only terms, ShareAlike (CC-BY-SA), copyleft (GPL,
AGPL, MPL, ODbL), custom or unknown terms. Datasets used only to *measure* the model are listed
separately; the model never trains on them."""

BASE = [
    ("Laya English checkpoint", "https://huggingface.co/convaiinnovations/laya", "Apache-2.0"),
    ("Laya multilingual checkpoint", "https://huggingface.co/convaiinnovations/laya-multilingual", "Apache-2.0"),
    ("ModernBERT-large (encoder of the English checkpoint)", "https://huggingface.co/answerdotai/ModernBERT-large", "Apache-2.0"),
    ("mmBERT-base (encoder of the multilingual checkpoint)", "https://huggingface.co/jhu-clsp/mmBERT-base", "MIT"),
]

TASKS = [
    ("Banking77, train split", "https://huggingface.co/datasets/PolyAI/banking77", "CC-BY-4.0",
     "77 banking intents. Attribution: Casanueva et al., 2020, PolyAI."),
    ("MASSIVE, train split, 51 languages", "https://huggingface.co/datasets/AmazonScience/massive", "CC-BY-4.0",
     "Loaded from the mteb/amazon_massive_intent mirror. Attribution: FitzGerald et al., 2022, Amazon."),
    ("typed-decisions, train split", "https://huggingface.co/datasets/LocalLLaMA/typed-decisions", "Apache-2.0",
     "Replay data."),
    ("tasksource typed-decisions mixture", "https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions",
     "per source, see below", "Only rows marked commercial whose every listed licence is permissive."),
]

EVAL_ONLY = [
    ("Banking77 test", "https://huggingface.co/datasets/PolyAI/banking77", "CC-BY-4.0"),
    ("MASSIVE test / validation", "https://huggingface.co/datasets/AmazonScience/massive", "CC-BY-4.0"),
    ("typed-decisions test", "https://huggingface.co/datasets/LocalLLaMA/typed-decisions", "Apache-2.0"),
    ("AG News test", "https://huggingface.co/datasets/fancyzhx/ag_news", "unknown"),
    ("DAIR Emotion test / validation", "https://huggingface.co/datasets/dair-ai/emotion", "unknown on HF"),
    ("tyqiangz multilingual sentiment test / valid", "https://github.com/tyqiangz/multilingual-sentiment-datasets",
     "aggregate; includes research-only corpora"),
]

NOT_IN_RELEASES = [
    ("tyqiangz multilingual sentiment (train)", "Aggregates Amazon Reviews Multi (academic research only), Yelp, IMDB, "
     "SemEval and DAIR Emotion; the repository's Apache-2.0 tag does not cover the underlying data."),
    ("AG News and tweet_eval texts for distillation", "Licence unknown; replaced by texts from the licence-filtered mixture."),
]


def source_link(sid):
    """Markdown link for a v6 registry id: Hub dataset, GitHub repository or Zenodo record."""
    if sid.startswith("github:"):
        repo = "/".join(sid[7:].split()[0].split("/")[:2])
        return f"[{sid[7:]}](https://github.com/{repo})"
    if sid.startswith("zenodo:"):
        record = re.match(r"\d+", sid[7:])
        return f"[{sid[7:]}](https://zenodo.org/records/{record.group(0)})" if record else sid
    if "|" in sid or " " in sid:
        return sid.replace("|", "/")
    return f"[{sid}](https://huggingface.co/datasets/{sid})"


def v6_section(man):
    """Training sources of a mixture v6 build: only sources that loaded and kept items."""
    used = sorted((r for r in man["sources"] if r.get("kept") and "error" not in r), key=lambda r: r["id"].lower())
    out = ["", f"### Mixture v6 sources ({len(used)} sources, {sum(r['kept'] for r in used):,} items, "
               f"at most {man['per_source_cap']} per source)", "",
           "Licence checked at the source for every entry (registry `tools/finetune/sources/v6-keep.json`, "
           "review notes and attribution in `tools/finetune/sources/v6-research.md`). Every text that occurs "
           f"in an evaluation suite was removed first ({man.get('eval_texts', 0):,} banned texts).", "",
           "| Source | Category | Licence | Languages | Items |", "|---|---|---|---|---:|"]
    for r in used:
        cats = ", ".join(c.strip().split("-", 1)[-1] for c in str(r.get("category", "")).split(";") if c.strip())
        langs = r.get("languages") or []
        lang = ", ".join(langs[:8]) + (f" +{len(langs) - 8}" if len(langs) > 8 else "")
        out.append(f"| {source_link(r['id'])} | {cats} | {r.get('licence', '')} | {lang} | {r['kept']:,} |")
    return out


def synth_section(man):
    """Synthetic gap data: generator, licence of the generator, verification, counts."""
    ol = man.get("ollama") or {}
    by_task = (man.get("counts") or {}).get("by_task") or {}
    rate = man.get("verify_acceptance_rate")
    return ["", f"### Synthetic gap data ({man.get('items', 0):,} items)", "",
            f"Generated on the maintainer's machine by `{man.get('generator_model', '')}` through Ollama "
            f"(licence {ol.get('general_license', 'see model card')}, digest `{str(ol.get('digest', ''))[:12]}`), "
            "for categories without enough licence-clean human data. The prompts contain no training examples; "
            "every item passed a blind second answer by the same model that had to match its gold label"
            + (f" ({rate:.0%} of verified items agreed)" if isinstance(rate, (int, float)) else "")
            + f". Prompt version `{man.get('prompts_version', '')}`.", "",
            "Items per task: " + ", ".join(f"{t} {n:,}" for t, n in sorted(by_task.items())) + "."]


def main():
    ap = argparse.ArgumentParser(description="Generate DATA_LICENSES.md")
    ap.add_argument("manifest", help="mixture v5 manifest (build_mixture.py)")
    ap.add_argument("audit", nargs="?", help="licence audit of the v5 mixture (licence_audit.json)")
    ap.add_argument("--v6", help="mixture v6 manifest (tools/finetune/mixture_v6/build.py)")
    ap.add_argument("--synth", help="synthetic gap data manifest (tools/synth/generate.py)")
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    rows = sorted(((s, n, man.get("licenses", {}).get(s, "")) for s, n in man["per_source"].items() if n),
                  key=lambda x: x[0].lower())
    out = ["# Data licences", "", POLICY, "",
           "## Base models", "", "| Model | Licence |", "|---|---|"]
    out += [f"| [{n}]({u}) | {l} |" for n, u, l in BASE]
    out += ["", "## Training data of released weights", "", "| Dataset | Licence | Notes |", "|---|---|---|"]
    out += [f"| [{n}]({u}) | {l} | {note} |" for n, u, l, note in TASKS]
    for d in man.get("extra", []):
        langs = ", ".join(d.get("languages", [])[:12]) if isinstance(d.get("languages"), list) else ""
        out.insert(out.index("| [tasksource typed-decisions mixture](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions) "
                             "| per source, see below | Only rows marked commercial whose every listed licence is permissive. |"),
                   f"| [{d['id']}](https://huggingface.co/datasets/{d['id']}) | {d.get('licence', '').upper()} | "
                   f"{d.get('items', '')} items{'; ' + langs if langs else ''}; evidence: [card revision]({d.get('licence_evidence_url', '')}) |")
    out += ["", f"### Mixture sources ({len(rows)} sources, {sum(n for _, n, _ in rows):,} items, "
                f"at most {man['per_source_cap']} per source)", "",
            "Licence as recorded per row by tasksource; source names follow its build manifest.", "",
            "| Source | Licence | Items |", "|---|---|---:|"]
    out += [f"| `{s}` | {l} | {n} |" for s, n, l in rows]
    st = man.get("stats", {})
    out += ["", "Filter statistics of this build: " + ", ".join(f"{k.replace('_', ' ')} {v:,}" for k, v in st.items()) + ".",
            ]
    if a.v6:
        out += v6_section(json.load(open(a.v6)))
    if a.synth:
        out += synth_section(json.load(open(a.synth)))
    out += ["", "## Used only for evaluation (never trained on)", "", "| Dataset | Licence |", "|---|---|"]
    out += [f"| [{n}]({u}) | {l} |" for n, u, l in EVAL_ONLY]
    out += ["", "## Not used for released weights", "", "| Data | Reason |", "|---|---|"]
    out += [f"| {n} | {r} |" for n, r in NOT_IN_RELEASES]
    if a.audit:
        audit = json.load(open(a.audit))
        ex = audit["exclude"]
        out += ["", f"### Mixture sources excluded by the licence audit ({len(ex)})", "",
                "Their recorded tags looked permissive, but the origin of the text or labels is not: "
                + audit["meta"]["method"] + ".", "", "| Source | Reason |", "|---|---|"]
        out += [f"| `{s}` | {r.replace('|', '/')} |" for s, r in sorted(ex.items(), key=lambda x: x[0].lower())]
    out += ["", "Research checkpoints trained before this policy (the Banking77 v1/v3 and multi-task runs "
            "documented in the README) used some of the data above and are not released.", ""]
    print("\n".join(out))


if __name__ == "__main__":
    main()
