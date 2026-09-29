#!/usr/bin/env python3
"""Package one promoted Statim LoRA adapter for Hugging Face.

All measurements and training facts in the package come from the experiment output. Nothing is
uploaded unless --upload is supplied explicitly.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile

try:
    from tools.release.hf_publish import MODELS
except ModuleNotFoundError:  # direct execution puts tools/release, not the repository, on sys.path
    from hf_publish import MODELS


ROOT = Path(__file__).resolve().parents[2]
GITHUB = "https://github.com/BEKO2210/statim"
PROTOCOL_KEYS = ("n", "seed", "z", "alpha", "created", "argv")
LICENCE_NAMES = {
    "mit": "MIT",
    "apache-2.0": "Apache-2.0",
    "cc-by-4.0": "CC-BY-4.0",
    "cc0-1.0": "CC0-1.0",
    "bsd-3-clause": "BSD-3-Clause",
    "odc-by": "ODC-By",
}

GGUF_VALUE_FORMATS = {
    0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
    7: "?", 10: "Q", 11: "q", 12: "d",
}


def read_gguf_metadata(path):
    """Read a GGUF header and metadata values without reading tensor descriptors or data."""
    with open(path, "rb") as stream:
        def unpack(fmt):
            size = struct.calcsize("<" + fmt)
            data = stream.read(size)
            if len(data) != size:
                raise ValueError(f"truncated GGUF metadata in {path}")
            return struct.unpack("<" + fmt, data)[0]

        def string():
            size = unpack("Q")
            data = stream.read(size)
            if len(data) != size:
                raise ValueError(f"truncated GGUF string in {path}")
            return data.decode("utf-8")

        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:
                element_kind = unpack("I")
                count = unpack("Q")
                return [value(element_kind) for _ in range(count)]
            fmt = GGUF_VALUE_FORMATS.get(kind)
            if fmt is None:
                raise ValueError(f"unsupported GGUF metadata type {kind} in {path}")
            return unpack(fmt)

        if stream.read(4) != b"GGUF":
            raise ValueError(f"not a GGUF file: {path}")
        version = unpack("I")
        if version not in (1, 2, 3):
            raise ValueError(f"unsupported GGUF version {version} in {path}")
        unpack("Q")  # tensor count
        metadata_count = unpack("Q")
        metadata = {}
        for _ in range(metadata_count):
            key = string()
            metadata[key] = value(unpack("I"))
        return metadata


def adapter_metadata(path):
    metadata = read_gguf_metadata(path)
    result = {
        key.removeprefix("statim.lora."): metadata[key]
        for key in (
            "statim.lora.base_fingerprint",
            "statim.lora.base_name",
            "statim.lora.base_checkpoint_sha256",
        )
        if key in metadata
    }
    for required in ("base_fingerprint", "base_name"):
        if not isinstance(result.get(required), str) or not result[required]:
            raise ValueError(f"adapter GGUF has no valid statim.lora.{required}: {path}")
    return result


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_root(path):
    path = Path(path).resolve()
    if path.is_file():
        path = path.parent
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def sanitize_string(value, roots):
    """Remove machine-local absolute paths, including paths embedded in shell commands."""
    result = value
    for root in sorted({str(Path(p).resolve()) for p in roots if p}, key=len, reverse=True):
        result = result.replace(root + os.sep, "").replace(root, ".")

    def quoted(match):
        quote, path = match.group(1), match.group(2)
        return quote + (os.path.basename(path.rstrip("/")) or ".") + quote

    result = re.sub(r"(['\"])(/[^'\"]+)\1", quoted, result)
    result = re.sub(
        r"(?<![:\w/])/(?:[^\s'\"]+)",
        lambda match: os.path.basename(match.group(0).rstrip("/")) or ".",
        result,
    )
    return result


def sanitize(value, roots):
    if isinstance(value, dict):
        return {sanitize_string(k, roots): sanitize(v, roots) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v, roots) for v in value]
    if isinstance(value, str):
        return sanitize_string(value, roots)
    return value


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def copy_sanitized_json(source, destination, roots):
    value = read_json(source)
    cleaned = sanitize(value, roots)
    if cleaned == value:
        shutil.copyfile(source, destination)
    else:
        write_json(destination, cleaned)


def source_records(train, registry):
    by_key = {f"v6/{item['id']}/{item['config']}": item for item in registry}
    records = []
    for key, counts in train["sources"].items():
        if key not in by_key:
            raise ValueError(f"training source not found in registry by id/config: {key}")
        entry = by_key[key]
        records.append((key, counts, entry))
    return records


def baseline_scores(category, languages):
    text = (ROOT / "docs" / "BASELINES.md").read_text(encoding="utf-8")
    found = {}
    row = re.compile(r"^\|\s*" + re.escape(category) + r"\s*\(([^)]+)\)\s*\|(.+)$")
    for line in text.splitlines():
        match = row.match(line)
        if not match:
            continue
        columns = [part.strip().replace("**", "") for part in match.group(2).split("|")]
        found[match.group(1)] = float(columns[1])
    missing = sorted(set(languages) - set(found))
    if missing:
        raise ValueError(f"Qwen3-8B baseline missing for {category}: {', '.join(missing)}")
    return found


def decision_rule():
    text = (ROOT / "docs" / "ADAPTERS.md").read_text(encoding="utf-8")
    match = re.search(r"^- \*\*Decision\*\* \(`gate\.adapter_decision`\): (.+(?:\n  .+)*)", text, re.M)
    if not match:
        raise ValueError("could not find adapter decision rule in docs/ADAPTERS.md")
    return " ".join(line.strip() for line in match.group(1).splitlines())


def load_checks(args, adapter_path, adapter_meta):
    if args.skip_load_check:
        return "not checked (--skip-load-check)", []
    version_run = subprocess.run(
        [str(args.statim), "version"], cwd=ROOT, text=True, capture_output=True
    )
    if version_run.returncode:
        raise RuntimeError(f"statim version failed: {version_run.stderr.strip()}")
    version = version_run.stdout.strip()
    checks = []
    for base in args.base_gguf:
        base_run = subprocess.run(
            [str(args.statim), "info", "-m", str(base)],
            cwd=ROOT, text=True, capture_output=True,
        )
        if base_run.returncode:
            detail = base_run.stderr.strip() or base_run.stdout.strip()
            raise RuntimeError(f"base info check failed for {base}: {detail}")
        try:
            fingerprint = json.loads(base_run.stdout)["fingerprint"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError(f"could not read fingerprint from statim info for {base}") from exc
        if fingerprint != adapter_meta["base_fingerprint"]:
            raise RuntimeError(
                f"base fingerprint mismatch for {base}: adapter "
                f"{adapter_meta['base_fingerprint']}, base {fingerprint}"
            )
        command = [str(args.statim), "info", "-m", str(base), "--adapter", f"{args.category}={adapter_path}"]
        run = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        if run.returncode:
            detail = run.stderr.strip() or run.stdout.strip()
            raise RuntimeError(f"adapter load check failed for {base}: {detail}")
        try:
            info = json.loads(run.stdout)
            mode = info["adapter"]["mode"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError(f"could not read adapter mode from statim info for {base}") from exc
        checks.append((Path(base).name, mode))
    return version, checks


def yaml_front_matter(args, name, cells):
    repo_id = args.upload or f"{args.base_repo.split('/', 1)[0]}/{name}"
    results = []
    if args.protocol["eval_skip"] > 0:
        protocol = (
            f"n={args.scored_items} fresh: draw {args.protocol['n']}, "
            f"skip {args.protocol['eval_skip']}, seed={args.protocol['seed']}, "
            f"z={args.protocol['z']}, alpha={args.protocol['alpha']}"
        )
    else:
        protocol = (
            f"n={args.scored_items}, seed={args.protocol['seed']}, "
            f"z={args.protocol['z']}, alpha={args.protocol['alpha']}"
        )
    for cell in cells:
        results.append({
            "task": {"type": "text-classification"},
            "dataset": {"name": f"{args.category} {cell['lang']} ({protocol})", "type": args.category},
            "metrics": [{"type": "accuracy", "value": cell["adapter_acc"]}],
        })
    front = {
        "license": "other",
        "license_name": "statim-weights",
        "license_link": f"https://huggingface.co/{repo_id}/blob/main/LICENSE-MODEL.md",
        "base_model": args.base_repo,
        "base_model_relation": "adapter",
        "library_name": "gguf",
        "language": [cell["lang"] for cell in cells],
        "tags": ["statim", "lora", "adapter", "gguf", args.category],
        "pipeline_tag": "zero-shot-classification",
        "model-index": [{"name": name, "results": results}],
    }
    return "---\n" + "\n".join(
        f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in front.items()
    ) + "\n---\n"


def fmt(value, digits=4):
    return f"{value:.{digits}f}"


def category_display(category):
    return {"pii": "PII", "fact_check": "Fact-check"}.get(category, category.capitalize())


def source_url(entry):
    for field in ("url", "homepage", "pinned_url"):
        if entry.get(field):
            return entry[field]
    source_id = public_name(entry["id"])
    match = re.fullmatch(r"github:([^/\s]+)/([^/\s]+)", source_id)
    if match:
        return f"https://github.com/{match.group(1)}/{match.group(2)}"
    match = re.fullmatch(r"zenodo:(\d+).*", source_id)
    if match:
        return f"https://zenodo.org/records/{match.group(1)}"
    if re.fullmatch(r"[^/:\s]+/[^/\s]+", source_id):
        return f"https://huggingface.co/datasets/{source_id}"
    return None


def public_name(text):
    """A registry id or licence without its trailing working note in parentheses."""
    return re.sub(r"\s*\([^()]*\)\s*$", "", text).strip()


def licence_display(entry):
    licence = public_name(entry["licence"])
    return LICENCE_NAMES.get(licence.lower(), licence)


def source_link(entry):
    url = source_url(entry)
    return f"[`{public_name(entry['id'])}`]({url})" if url else f"`{public_name(entry['id'])}`"


def jsonl_cells(path):
    cells = {}
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {number} of {path}") from exc
            language = item.get("lang")
            if language and language != "macro" and "skipped" not in item:
                if "accuracy" not in item or "pool_items_sha256" not in item:
                    raise ValueError(f"incomplete language cell {language!r} in {path}")
                cells[language] = item
    if not cells:
        raise ValueError(f"no language cells in {path}")
    return cells


def published_checks(args, experiment_base, experiment_adapter):
    experiment = (jsonl_cells(experiment_base), jsonl_cells(experiment_adapter))
    results = []
    for weights, base_path, adapter_path in args.check:
        base_cells, adapter_cells = jsonl_cells(base_path), jsonl_cells(adapter_path)
        if set(base_cells) != set(adapter_cells):
            raise ValueError(f"published check language sets differ for {weights}")
        base_mean = sum(cell["accuracy"] for cell in base_cells.values()) / len(base_cells)
        adapter_mean = sum(cell["accuracy"] for cell in adapter_cells.values()) / len(adapter_cells)
        same = True
        for actual, expected in ((base_cells, experiment[0]), (adapter_cells, experiment[1])):
            if set(actual) != set(expected) or any(
                actual[lang][key] != expected[lang][key]
                for lang in actual for key in ("accuracy", "pool_items_sha256")
            ):
                same = False
        results.append({
            "weights": weights, "base_path": base_path, "adapter_path": adapter_path,
            "base_mean": base_mean, "adapter_mean": adapter_mean, "same": same,
        })
    return results


def adapter_notice(title, args, sources):
    required = [
        line for line in (ROOT / "LICENSE-MODEL.md").read_text(encoding="utf-8").splitlines()
        if line.startswith("Required Notice:")
    ]
    if not required:
        raise ValueError("LICENSE-MODEL.md contains no Required Notice lines")
    lines = [title, "", *required, "", (
        "The weights are licensed under PolyForm Noncommercial 1.0.0, PolyForm Small Business "
        "1.0.0 or PolyForm Free Trial 1.0.0, at the user's choice (LICENSE-MODEL.md), or a "
        f"commercial licence ({GITHUB}/blob/main/COMMERCIAL.md)."
    ), "", f"It adapts https://huggingface.co/{args.base_repo}.", (
        "Laya (https://github.com/NandhaKishorM/laya) is Apache-2.0; Statim is independent "
        "and not affiliated with or endorsed by the Laya authors."
    ), "", "Training data"]
    repeated = {entry["id"] for _, _, entry in sources
                if sum(other["id"] == entry["id"] for _, _, other in sources) > 1}
    for _, counts, entry in sources:
        licence = licence_display(entry)
        wording = (f"licensed under CC BY {licence.removeprefix('CC-BY-')}"
                   if licence.startswith("CC-BY-") else f"licensed under {licence}")
        link = source_url(entry)
        name = (f"{public_name(entry['id'])} ({entry['config']})" if entry["id"] in repeated
                else public_name(entry["id"]))
        linked = f"{name}, {link}" if link else name
        lines.append(f"- {linked}: {wording}; {counts['rows']:,} rows used.")
    return "\n".join(lines) + "\n"


def quick_start_example(category):
    if category == "pii":
        state = {"text": "Hi, this is Anna Schmidt. Call me back at +49 170 1234567."}
        question = {"pii": {"type": "noul", "instructions": "Does the text contain a phone number?"}}
    elif category == "emotion":
        state = {"text": "They cancelled my flight again and nobody even apologised!"}
        question = {"emotion": {"type": "choice", "instructions": "Which emotion is expressed most strongly?",
                                "criteria": ["anger", "disgust", "fear", "joy", "sadness", "surprise"]}}
    else:
        state = {"text": "The claim is supported by the cited evidence."}
        question = {category: {"type": "noul", "instructions": "Should this text receive a yes decision?"}}
    return state, question


def resolved_name(args):
    default = (args.upload.rsplit("/", 1)[-1] if args.upload
               else f"{args.base_repo.rsplit('/', 1)[-1]}-{args.category}")
    name = args.name or default
    if args.upload and name != args.upload.rsplit("/", 1)[-1]:
        raise ValueError(
            f"--name {name!r} does not match the repository name in --upload {args.upload!r}"
        )
    return name


def scored_items(cells):
    counts = []
    for cell in cells:
        base = cell.get("n_base")
        adapter = cell.get("n_adapter")
        if base != adapter:
            raise ValueError(
                f"scored item counts differ for {cell.get('lang', '<unknown>')}: "
                f"n_base={base!r}, n_adapter={adapter!r}"
            )
        if not isinstance(base, int) or isinstance(base, bool) or base < 1:
            raise ValueError(
                f"invalid scored item count for {cell.get('lang', '<unknown>')}: {base!r}"
            )
        counts.append(base)
    if len(set(counts)) != 1:
        raise ValueError(f"scored item counts differ between cells: {counts}")
    return counts[0]


def card(args, name, category_record, train, sources, baselines, adapter_sha, peft_sha,
         statim_version, checks, adapter_meta, published):
    cells = category_record["cells"]
    args.protocol = {key: args.summary[key] for key in PROTOCOL_KEYS}
    args.protocol["eval_skip"] = args.summary.get("eval_skip", 0)
    args.scored_items = scored_items(cells)
    expected = args.protocol["n"] - args.protocol["eval_skip"]
    if args.scored_items != expected:
        raise ValueError(f"cells scored {args.scored_items} items, but the draw of {args.protocol['n']} "
                         f"minus skip {args.protocol['eval_skip']} is {expected}")
    head = yaml_front_matter(args, name, cells)
    same_items_as_baselines = args.protocol["eval_skip"] == 0
    if not same_items_as_baselines:
        baselines = {}
    rows = []
    for cell in cells:
        qwen = fmt(baselines[cell["lang"]], 3) if same_items_as_baselines else ""
        rows.append(
            f"| {cell['lang']} | {fmt(cell['base_acc'])} | **{fmt(cell['adapter_acc'])}** | "
            f"{cell['delta'] * 100:+.2f} | {cell['verdict']}" + (f" | {qwen} |" if same_items_as_baselines else " |")
        )
    base_mean = sum(cell["base_acc"] for cell in cells) / len(cells)
    adapter_mean = sum(cell["adapter_acc"] for cell in cells) / len(cells)
    delta_mean = sum(cell["delta"] for cell in cells) / len(cells)
    qwen_values = [baselines.get(cell["lang"]) for cell in cells]
    qwen_mean = (sum(qwen_values) / len(qwen_values)
                 if all(value is not None for value in qwen_values) else None)
    rows.append(
        f"| **Mean** | {fmt(base_mean)} | **{fmt(adapter_mean)}** | {delta_mean * 100:+.2f} | —"
        + (f" | {'' if qwen_mean is None else fmt(qwen_mean, 3)} |" if same_items_as_baselines else " |")
    )
    table_head = ("| Language | Base | Adapter | Change (points) | Verdict | Qwen3-8B zero-shot |\n"
                  "|---|---:|---:|---:|---|---:|" if same_items_as_baselines else
                  "| Language | Base | Adapter | Change (points) | Verdict |\n|---|---:|---:|---:|---|")
    pooled_note = ("A single cell of {} items rarely clears {} SE on its own; the decision uses the pooled "
                   "family.\n\n".format(args.scored_items, f"{args.protocol['z']:g}") if len(cells) > 1 else "")
    qwen_note = ("Statim is trained on this category, while [Qwen3-8B runs zero-shot]"
                 f"({GITHUB}/blob/main/docs/BASELINES.md)." if same_items_as_baselines else
                 "Qwen3-8B was measured on the first draw's items only ([BASELINES.md]"
                 f"({GITHUB}/blob/main/docs/BASELINES.md)), so it is not shown next to these fresh items.")
    source_rows = [
        f"| {source_link(entry)} (`{entry['config']}`) | {counts['rows']:,} | "
        f"{licence_display(entry)} |"
        for _, counts, entry in sources
    ]
    lora = train["lora"]
    commands = "\n".join(f"- `{key}`: `{value}`" for key, value in category_record["commands"].items())
    trained_languages = {lang for _, _, entry in sources for lang in entry.get("languages", [])}
    untrained = [cell["lang"] for cell in cells if cell["lang"] not in trained_languages]
    untrained_line = ""
    if untrained:
        untrained_line = "\n- Evaluated languages with no category training data: " + ", ".join(untrained) + "."
    base_name = args.base_repo.rsplit("/", 1)[-1]
    base_info = MODELS.get(base_name)
    if base_info is None:
        raise ValueError(f"unknown base repository name: {base_name}")
    q8 = next((Path(path).name for path in args.base_gguf if "q8_0" in Path(path).name.lower()),
              f"{base_name}-q8_0.gguf")
    repo_id = args.upload or f"{args.base_repo.split('/', 1)[0]}/{name}"
    family = category_record["family"]
    z_label = f"{args.protocol['z']:g}"
    pooling_keys = ("delta", "se", "flag")
    if all(family["rows"].get(key) == family["suites"].get(key) for key in pooling_keys):
        family_result = (
            f"The pooled family change is **{family['rows']['delta'] * 100:+.2f} points** with "
            f"**{z_label} SE = {args.protocol['z'] * family['rows']['se'] * 100:.2f} points** "
            f"({family['rows']['flag']})."
        )
    else:
        family_result = (
            f"Pooled by rows, the family change is **{family['rows']['delta'] * 100:+.2f} points** with\n"
            f"**{z_label} SE = {args.protocol['z'] * family['rows']['se'] * 100:.2f} points** "
            f"({family['rows']['flag']}). Pooled by suites, it is "
            f"**{family['suites']['delta'] * 100:+.2f} points** with\n"
            f"**{z_label} SE = {args.protocol['z'] * family['suites']['se'] * 100:.2f} points** "
            f"({family['suites']['flag']})."
        )
    version_display = re.sub(r"^statim\s+", "", statim_version, flags=re.I)
    checked = []
    for filename, mode in checks:
        mode_display = {"merge": "merged at load", "merged": "merged at load",
                        "runtime": "runtime LoRA"}.get(mode, mode)
        if "f32" in filename.lower():
            checked.append(f"the published f32 file loads it {mode_display}")
        elif "q8_0" in filename.lower():
            checked.append(f"the q8_0 file as {mode_display}")
        else:
            checked.append(f"`{filename}` loads it as {mode_display}")
    if not checked:
        checked_text = "Load checks were skipped"
    elif len(checked) == 1:
        checked_text = f"Checked with Statim {version_display}: {checked[0]}"
    else:
        checked_text = f"Checked with Statim {version_display}: {', '.join(checked[:-1])}, {checked[-1]}"
    served = re.search(r"-(f32|f16|bf16|q8_0|q5_0|q4_0)\.gguf", category_record.get("commands", {}).get("serve", ""))
    experiment_weights = served.group(1) if served else None
    published_rows = []
    for item in published:
        mode = next((mode for filename, mode in checks
                     if item["weights"].lower() in filename.lower()), "—")
        mode = {"merge": "merged at load", "merged": "merged at load",
                "runtime": "runtime LoRA"}.get(mode, mode)
        published_rows.append(
            f"| {item['weights']} | {mode} | {fmt(item['base_mean'])} | "
            f"{fmt(item['adapter_mean'])} | "
            f"{(item['adapter_mean'] - item['base_mean']) * 100:+.2f} | "
            f"{('yes' if item['same'] else 'no') if experiment_weights in (None, item['weights']) else '— (experiment: ' + experiment_weights + ')'} |"
        )
    state, questions = quick_start_example(args.category)
    state_json = json.dumps(state, ensure_ascii=False)
    questions_json = json.dumps(questions, ensure_ascii=False)
    request_json = json.dumps({"state": state, "questions": questions, "adapter": args.category}, ensure_ascii=False)
    auto_request_json = json.dumps({"state": state, "questions": questions, "adapter": "auto"}, ensure_ascii=False)
    targets = ", ".join(f"`{module}`" for module in lora["target_modules"])
    if args.protocol["eval_skip"] > 0:
        results_intro = (
            f"Each cell uses {args.scored_items} fresh items: a seeded stratified draw of "
            f"{args.protocol['n']} (seed {args.protocol['seed']}) whose first "
            f"{args.protocol['eval_skip']} items, scored by an earlier run, are skipped. "
            "That earlier run is reported in docs/ADAPTERS.md."
        )
    else:
        results_intro = f"Each cell uses {args.scored_items} items (seed {args.protocol['seed']})."
    return head + f"""
# {base_info['display']}: {category_display(args.category)} adapter

A LoRA adapter that improves the {category_display(args.category)} decisions of
[`{args.base_repo}`](https://huggingface.co/{args.base_repo}) **{args.base_version}**.
{checked_text}. LoRA adapters need Statim 0.8.0 or later.

## Quick start

```sh
hf download {args.base_repo} {q8} --local-dir models
hf download {repo_id} {name}.lora.gguf --local-dir models
statim serve -m multilingual=models/{q8} --adapter multilingual:{args.category}=models/{name}.lora.gguf --port 8080

curl -s localhost:8080/v1/systemone -d '{request_json}'
curl -s localhost:8080/v1/systemone -d '{auto_request_json}'
```

Python (Python SDK 0.8.3 or later):

```python
from statim import Client
client = Client("http://127.0.0.1:8080")
client.decide({state_json}, {questions_json}, adapter="{args.category}")
```

## Results (experiment's f32 run)

{results_intro}

{table_head}
{chr(10).join(rows)}

{family_result}

{pooled_note}Decision rule: {decision_rule()}{'' if args.protocol['z'] == 2 else ' This run used ' + z_label + ' standard errors.'}

{qwen_note}

## Checked on the published files

| Weights | Adapter mode | Base mean | Adapter mean | Change (points) | Cell for cell as in the experiment |
|---|---|---:|---:|---:|---|
{chr(10).join(published_rows)}

## Training

| Source | Rows | Licence |
|---|---:|---|
{chr(10).join(source_rows)}

LoRA rank **{lora['r']}**, alpha **{lora['lora_alpha']}**, dropout **{lora['lora_dropout']}**;
target modules {targets}; **{lora['wrapped_modules']:,}** wrapped modules and
**{lora['trainable_parameters']:,}** trainable parameters.

- Items: **{train['items']['train']:,} train**, **{train['items']['dev']:,} dev**.
- Updates: **{train['updates']:,}**.
- Dev accuracy: **{train['dev_before']['dev_acc']}** before, **{train['best']['dev_acc']}** after.
- Time: **{train['seconds']} seconds**; peak memory: **{train['peak_memory_mb']:,} MB**.

## Provenance

- Adapter GGUF SHA-256: `{adapter_sha}`
- PEFT safetensors SHA-256: `{peft_sha}`
- Training checkpoint `model.safetensors` SHA-256: `{train['base_sha256']}`
- Base fingerprint: `{adapter_meta['base_fingerprint']}`
- Training mixture SHA-256: `{train['mixture_sha256']}`
- Source registry SHA-256: `{train['registry_sha256']}`

Experiment commands (paths relative to the Statim repository):

{commands}

Protocol and experiment: [docs/ADAPTERS.md]({GITHUB}/blob/main/docs/ADAPTERS.md). Full reproduction
instructions: [REPRODUCE.md]({GITHUB}/blob/main/REPRODUCE.md).

## Intended use and limits

- This adapter only helps its category; route requests with `"{args.category}"` or `"auto"`.
- Bound to `{args.base_repo}` {args.base_version} by the base fingerprint `{adapter_meta['base_fingerprint'][:16]}…` (`statim.lora.base_fingerprint`, SHA-256 over the checkpoint's norm and bias tensors); Statim refuses the adapter on a base whose fingerprint differs.{(' The full checkpoint SHA-256 is also checked when both the adapter and base file carry `statim.lora.base_checkpoint_sha256`.' if adapter_meta.get('base_checkpoint_sha256') else '')}
- Languages outside the evaluated list are untested.
- Each language cell has {args.scored_items} items.
- Do not automate decisions about people without human review.{untrained_line}

## Licence

The weights may be used under any one of: PolyForm Noncommercial 1.0.0, PolyForm Small Business
1.0.0 (free commercial use below 100 people and 1 M USD revenue), PolyForm Free Trial 1.0.0 (any
company, fewer than 32 days), or a Statim commercial licence
([COMMERCIAL.md]({GITHUB}/blob/main/COMMERCIAL.md)). Texts in [LICENSE-MODEL.md](LICENSE-MODEL.md).
The Statim engine is Apache-2.0.

Training data attribution is listed source by source above, with the row count and licence read
from the experiment registry.
"""


def write_sums(out):
    entries = []
    for path in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        entries.append(f"{sha256(path)}  {path.relative_to(out).as_posix()}")
    (out / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")
    return entries


def install_staging(staging, out):
    """Replace a package directory, tolerating sandbox metadata mounts at workspace roots."""
    if not out.exists():
        staging.rename(out)
        return
    if not out.is_dir():
        out.unlink()
        staging.rename(out)
        return
    preserved = {".git", ".agents", ".codex"}
    for path in out.iterdir():
        if path.name in preserved:
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    for path in staging.iterdir():
        destination = out / path.name
        if path.is_dir():
            shutil.copytree(path, destination)
        else:
            shutil.copy2(path, destination)
    shutil.rmtree(staging)


def upload(args, out, binary_paths):
    from huggingface_hub import HfApi  # imported only for an explicit --upload

    api = HfApi()
    api.create_repo(args.upload, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=args.upload, folder_path=str(out), repo_type="model")
    info = api.model_info(args.upload, files_metadata=True)
    expected = {}
    for line in (out / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        expected[relative] = digest
    remote = {}
    for sibling in info.siblings:
        lfs = getattr(sibling, "lfs", None)
        digest = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        if digest:
            remote[sibling.rfilename] = digest
    for relative in binary_paths:
        local = expected[relative]
        if remote.get(relative) != local:
            raise RuntimeError(
                f"remote LFS SHA-256 mismatch for {relative}: local {local}, remote {remote.get(relative)}"
            )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--category", required=True)
    parser.add_argument("--base-repo", required=True)
    parser.add_argument("--base-version", required=True)
    parser.add_argument("--base-gguf", action="append", default=[], type=Path)
    parser.add_argument(
        "--check", action="append", default=[], metavar="WEIGHTS=BASE_JSONL,ADAPTER_JSONL",
        help="evaluation outputs for the published base file without and with the adapter",
    )
    parser.add_argument("--statim", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--name")
    parser.add_argument("--upload", help="Hugging Face repository id; without it nothing is uploaded")
    parser.add_argument("--skip-load-check", action="store_true", help="tests only")
    args = parser.parse_args(argv)
    parsed = []
    for spec in args.check:
        try:
            weights, paths = spec.split("=", 1)
            base, adapter = paths.split(",", 1)
        except ValueError:
            parser.error(f"invalid --check {spec!r}; expected WEIGHTS=BASE_JSONL,ADAPTER_JSONL")
        if not weights or not base or not adapter:
            parser.error(f"invalid --check {spec!r}; expected WEIGHTS=BASE_JSONL,ADAPTER_JSONL")
        parsed.append((weights, Path(base), Path(adapter)))
    args.check = parsed
    return args


def main(argv=None):
    args = parse_args(argv)
    work = args.work_dir.resolve()
    try:
        summary = read_json(work / "summary.json")
        record = summary.get("categories", {}).get(args.category)
        if record is None:
            print(f"refusing to package {args.category}: category is absent from summary.json", file=sys.stderr)
            return 1
        if record.get("status") != "ok" or record.get("promote") is not True:
            print(
                f"refusing to package {args.category}: status={record.get('status')!r}, "
                f"promote={record.get('promote')!r}; {record.get('reason', '')}".rstrip(),
                file=sys.stderr,
            )
            return 1
        if not args.base_gguf and not args.skip_load_check:
            raise ValueError("--base-gguf is required unless --skip-load-check is given")
        name = resolved_name(args)
        required = [
            work / f"{args.category}.lora.gguf",
            work / args.category / "adapter_config.json",
            work / args.category / "adapter_model.safetensors",
            work / args.category / "train_lora.json",
            work / f"{args.category}.base.jsonl",
            work / f"{args.category}.adapter.jsonl",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise ValueError("missing experiment files: " + ", ".join(missing))
        train = read_json(required[3])
        adapter_meta = adapter_metadata(required[0])
        registry = read_json(ROOT / "tools" / "finetune" / "sources" / "v6-keep.json")
        sources = source_records(train, registry)
        # BASELINES measured the first draw's items; a fresh-item run is not compared with them
        baselines = ({} if summary.get("eval_skip", 0) > 0 else
                     baseline_scores(args.category, [cell["lang"] for cell in record["cells"]]))
        version, checks = load_checks(args, required[0], adapter_meta)
        published = published_checks(args, required[4], required[5])
        roots = [ROOT, repo_root(work)]
        clean_record = sanitize(record, roots)
        clean_summary = sanitize(
            {key: summary[key] for key in PROTOCOL_KEYS}
            | {"eval_skip": summary.get("eval_skip", 0), "category": args.category}
            | record,
            roots,
        )
        clean_train = sanitize(train, roots)
        args.summary = clean_summary

        parent = args.out.resolve().parent
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=parent))
        try:
            (staging / "peft").mkdir()
            (staging / "training").mkdir()
            (staging / "evaluation").mkdir()
            shutil.copyfile(required[0], staging / f"{name}.lora.gguf")
            copy_sanitized_json(required[1], staging / "peft" / "adapter_config.json", roots)
            shutil.copyfile(required[2], staging / "peft" / "adapter_model.safetensors")
            for source, destination in (
                (required[4], staging / "evaluation" / f"{args.category}.base.jsonl"),
                (required[5], staging / "evaluation" / f"{args.category}.adapter.jsonl"),
            ):
                original = source.read_text(encoding="utf-8")
                cleaned = sanitize_string(original, roots)
                if cleaned == original:
                    shutil.copyfile(source, destination)
                else:
                    destination.write_text(cleaned, encoding="utf-8")
            for item in published:
                for source, suffix in ((item["base_path"], "base"),
                                       (item["adapter_path"], "adapter")):
                    destination = (
                        staging / "evaluation" / f"published-{item['weights']}-{suffix}.jsonl"
                    )
                    original = source.read_text(encoding="utf-8")
                    destination.write_text(sanitize_string(original, roots), encoding="utf-8")
            shutil.copyfile(ROOT / "LICENSE-MODEL.md", staging / "LICENSE-MODEL.md")
            write_json(staging / "training" / "train_lora.json", clean_train)
            write_json(staging / "evaluation" / "summary.json", clean_summary)
            adapter_sha = sha256(staging / f"{name}.lora.gguf")
            peft_sha = sha256(staging / "peft" / "adapter_model.safetensors")
            readme = card(args, name, clean_record, clean_train, sources, baselines, adapter_sha,
                          peft_sha, sanitize_string(version, roots), checks, adapter_meta, published)
            (staging / "README.md").write_text(sanitize_string(readme, roots), encoding="utf-8")
            base_name = args.base_repo.rsplit("/", 1)[-1]
            if base_name not in MODELS:
                raise ValueError(f"unknown base repository name: {base_name}")
            title = f"{MODELS[base_name]['display']}: {category_display(args.category)} adapter"
            (staging / "NOTICE").write_text(
                adapter_notice(title, args, sources), encoding="utf-8"
            )
            write_sums(staging)
            install_staging(staging, args.out)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        print(f"built {args.out}: {sum(1 for p in args.out.rglob('*') if p.is_file())} files")
        if args.upload:
            upload(args, args.out, [f"{name}.lora.gguf", "peft/adapter_model.safetensors"])
            print(f"uploaded https://huggingface.co/{args.upload}")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
