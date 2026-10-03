"""Shared, stdlib-only source-policy matching helpers."""

from __future__ import annotations

import re
import hashlib
import json
from pathlib import Path


INPUT_RELATIVE_PATHS = (
    "tools/finetune/sources/policy.json",
    "tools/finetune/sources/v6-keep.json",
    "tools/finetune/licence_audit.json",
    "tools/finetune/source_policy.py",
    "tools/finetune/check_sources.py",
    "tools/finetune/build_mixture.py",
    "tools/finetune/build_extra.py",
    "tools/finetune/build_v9.py",
    "tools/finetune/mixture_v6/build.py",
    "tools/finetune/mixture_v6/loaders.py",
    "tools/finetune/mixture_v6/registry.py",
    "tools/finetune/mixture_v6/templates.py",
    "tools/finetune/mixture_v6/languages.py",
    "tools/finetune/mixture_v6/eval_texts.py",
)

ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_hashes(root):
    root = Path(root)
    return {name: file_sha256(root / name) for name in INPUT_RELATIVE_PATHS}


def validate_mixture_dev(path, requested):
    """Refuse a v9 mixture when its recorded dev prefix is not used verbatim."""
    if not path:
        return
    mixture = Path(path)
    stem = mixture.name.removesuffix(".jsonl.gz")
    adjacent = [Path(str(mixture).replace(".jsonl.gz", ".manifest.json")),
                Path(str(mixture) + ".manifest.json")]
    documented = ROOT / "docs" / "reproductions" / (stem + ".manifest.json")
    names = adjacent + [documented]
    for manifest_path in dict.fromkeys(names):
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path == documented:
            expected = manifest.get("output_sha256")
            if not isinstance(expected, str) or not mixture.is_file() or file_sha256(mixture) != expected:
                continue
        if manifest.get("version") != 9:
            continue
        if requested != manifest.get("dev_prefix_rows"):
            raise ValueError("--mixture-dev must equal v9 manifest dev_prefix_rows (%s)" %
                             manifest.get("dev_prefix_rows"))
        return


def config_tokens(config):
    """Return the individual configs represented by a registry config string."""
    if config is None:
        return {"default"}
    return {token.strip() for token in re.split(r"[|,]", str(config)) if token.strip()}


def exclusion_matches(registry, source_id, config, exclusion):
    """Whether a policy exclusion covers a registry record or emitted source name.

    v5 and extra policy ids are families, so an exclusion deliberately covers all
    names beginning with that id (for example HelpSteer also covers HelpSteer2).
    v6 ids are dataset ids and configs may represent several configs with ``|``
    or ``,``.
    """
    if exclusion.get("registry") != registry:
        return False
    excluded_id = exclusion.get("id")
    if registry in {"v5", "extra"}:
        if not isinstance(source_id, str) or not source_id.startswith(excluded_id):
            return False
    elif source_id != excluded_id:
        return False
    return ("config" not in exclusion or exclusion.get("configs_only") is not True or
            exclusion["config"] in config_tokens(config))


def audit_excludes_family(audit, family):
    """Whether licence_audit's independent deny list rejects a v5 family."""
    excluded = audit.get("exclude", {})
    return isinstance(excluded, dict) and family in excluded


def tasksource_family_matches(src, family, record):
    """Whether a tasksource name is an explicitly admitted member of ``family``.

    ``sources`` is the closed list taken from the pinned tasksource snapshot.  The
    structural check documents the intended family relation, while the explicit
    membership check prevents new subsets at another revision being admitted.
    """
    if not isinstance(src, str) or not isinstance(family, str):
        return False
    known = record.get("sources")
    if not isinstance(known, list) or src not in known:
        return False
    base = family.rstrip("/")
    if src == base or src.startswith(base + "/"):
        return True
    # Two tasksource families use non-slash subset names in the pinned snapshot.
    return family in {"commonsense_qa", "spartqa-"} and src.startswith(family)


def matching_tasksource_families(src, families):
    """Return enabled audit families which admit ``src`` under shared semantics."""
    return [name for name, record in families.items()
            if record.get("use") is True and tasksource_family_matches(src, name, record)]


def tasksource_families(audit):
    """Return audit family records joined to their pinned concrete source names."""
    names = audit.get("source_names", {})
    return {family: {**record, "sources": names.get(family)}
            for family, record in audit.get("keep_families", {}).items()}


def row_is_excluded(registry, source_id, config, row, policy):
    """Whether ``row`` hits any rows-scoped exclusion in policy.json."""
    for exclusion in policy.get("exclusions", []):
        if exclusion.get("scope") != "rows" or not exclusion_matches(
                registry, source_id, config, exclusion):
            continue
        for field, denied in exclusion.get("row_filter", {}).items():
            if field not in row:
                raise ValueError(f"row for {registry}:{source_id} lacks policy filter column {field!r}")
            values = denied if isinstance(denied, list) else [denied]
            actual = row[field]
            if field in {"lang", "language"}:
                actual = str(actual).strip().lower().replace("_", "-").split("-", 1)[0]
                values = [str(value).strip().lower().replace("_", "-").split("-", 1)[0]
                          for value in values]
            if actual in values:
                return True
    return False
