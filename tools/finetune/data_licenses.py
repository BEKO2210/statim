#!/usr/bin/env python3
"""Generate DATA_LICENSES.md: every data source behind released Statim weights, with its licence.

    python3 tools/finetune/data_licenses.py data/mixture-v5.manifest.json tools/finetune/licence_audit.json > DATA_LICENSES.md
"""
import json
import sys

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


def main():
    man = json.load(open(sys.argv[1]))
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
            "", "## Used only for evaluation (never trained on)", "", "| Dataset | Licence |", "|---|---|"]
    out += [f"| [{n}]({u}) | {l} |" for n, u, l in EVAL_ONLY]
    out += ["", "## Not used for released weights", "", "| Data | Reason |", "|---|---|"]
    out += [f"| {n} | {r} |" for n, r in NOT_IN_RELEASES]
    if len(sys.argv) > 2:
        audit = json.load(open(sys.argv[2]))
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
