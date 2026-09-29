#!/usr/bin/env python3
"""Freeze and check the Statim HTTP API v1 contract (frozen as of 0.9.0).

The normalized contract records operations, request and response schemas, the
Error shape, query/header parameters, validation bounds, defaults, and the
documented /metrics families.  Breaking changes are removed operations or
statuses; removed request fields; newly-required request fields; narrowed or
changed request types, validation, or defaults; removed request enum values;
removed response fields; response fields made optional or changed in type; any
Error-shape change; and removed, renamed, or retyped metric families.  New
operations, optional request fields, response fields, statuses, enum values,
looser request bounds, and metric families are additive.  Response-bound and
response-default changes, descriptions, and examples are reported as notes.
"""

import argparse
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import sys

try:
    import yaml
except ImportError:
    print("PyYAML is required (pip install PyYAML)", file=sys.stderr)
    raise SystemExit(77)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OPENAPI = ROOT / "docs/openapi.yaml"
DEFAULT_CONTRACT = ROOT / "docs/api-v1.contract.json"
METHODS = {"get", "put", "post", "delete", "options", "head", "patch", "trace"}
MISSING = object()
SCHEMA_KEYS = {
    "type", "properties", "required", "enum", "const", "items", "prefixItems",
    "oneOf", "anyOf", "allOf", "additionalProperties", "not", "maxItems",
    "minItems", "maxLength", "minLength", "maximum", "minimum",
    "exclusiveMaximum", "exclusiveMinimum", "maxProperties", "minProperties",
    "pattern", "format", "multipleOf", "default",
}
MAX_BOUNDS = ("maxItems", "maxLength", "maximum", "exclusiveMaximum", "maxProperties")
MIN_BOUNDS = ("minItems", "minLength", "minimum", "exclusiveMinimum", "minProperties")
RESPONSE_NOTE_KEYS = set(MAX_BOUNDS + MIN_BOUNDS +
                         ("multipleOf", "pattern", "format", "default"))


def pointer(document, ref):
    if not ref.startswith("#/"):
        raise ValueError("only local OpenAPI references are supported: " + ref)
    value = document
    for part in ref[2:].split("/"):
        value = value[part.replace("~1", "/").replace("~0", "~")]
    return value


def dereference(document, value, stack=()):
    if not isinstance(value, dict):
        return value
    if "$ref" in value:
        ref = value["$ref"]
        if ref in stack:
            return {"$ref": ref}
        merged = copy.deepcopy(pointer(document, ref))
        merged.update({k: v for k, v in value.items() if k != "$ref"})
        return dereference(document, merged, stack + (ref,))
    return {k: dereference(document, v, stack) for k, v in value.items()}


def normalize_schema(document, schema):
    schema = dereference(document, schema or {})
    out = {}
    for key in sorted(SCHEMA_KEYS & schema.keys()):
        value = schema[key]
        if key == "properties":
            out[key] = {name: normalize_schema(document, child)
                        for name, child in sorted(value.items())}
        elif key in {"items", "additionalProperties", "not"} and isinstance(value, dict):
            out[key] = normalize_schema(document, value)
        elif key in {"oneOf", "anyOf", "allOf", "prefixItems"}:
            out[key] = [normalize_schema(document, child) for child in value]
        elif key in {"required", "enum"}:
            out[key] = sorted(value, key=lambda item: json.dumps(item, sort_keys=True))
        else:
            out[key] = value
    return out


def normalize_parameter(document, parameter):
    parameter = dereference(document, parameter)
    return {
        "name": parameter["name"],
        "in": parameter["in"],
        "required": bool(parameter.get("required", False)),
        "schema": normalize_schema(document, parameter.get("schema", {})),
    }


def response_schema(document, response):
    response = dereference(document, response)
    content = response.get("content", {})
    return {media: normalize_schema(document, body.get("schema", {}))
            for media, body in sorted(content.items())}


def metric_families(document):
    try:
        response = document["paths"]["/metrics"]["get"]["responses"]["200"]
        response = dereference(document, response)
        example = response["content"]["text/plain"]["example"]
    except (KeyError, TypeError):
        return []
    if not isinstance(example, str):
        return []
    families = {}
    for name, kind in re.findall(
            r"(?m)^\s*#\s*TYPE\s+([a-zA-Z_:][a-zA-Z0-9_:]*)\s+(\w+)\s*$", example):
        families[name] = kind
    return [{"name": name, "kind": kind} for name, kind in sorted(families.items())]


def build_contract(document):
    operations = {}
    for path, path_item in sorted(document.get("paths", {}).items()):
        for method, operation in sorted(path_item.items()):
            if method not in METHODS:
                continue
            name = method.upper() + " " + path
            parameters = path_item.get("parameters", []) + operation.get("parameters", [])
            parameters = [normalize_parameter(document, p) for p in parameters]
            parameters = [p for p in parameters if p["in"] in {"query", "header"}]
            parameters.sort(key=lambda p: (p["in"], p["name"]))
            request = None
            body = operation.get("requestBody")
            if body:
                body = dereference(document, body)
                request = {
                    "required": bool(body.get("required", False)),
                    "content": {media: normalize_schema(document, spec.get("schema", {}))
                                for media, spec in sorted(body.get("content", {}).items())},
                }
            operations[name] = {
                "parameters": parameters,
                "request": request,
                "responses": {str(status): response_schema(document, response)
                              for status, response in sorted(operation.get("responses", {}).items())},
            }
    error = normalize_schema(document, document["components"]["schemas"]["Error"])
    annotations = {}

    def collect(value, path="$"):
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = jpath(path, str(key))
                if key in {"description", "summary", "example", "examples"}:
                    encoded = json.dumps(child, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                    annotations[child_path] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
                else:
                    collect(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                collect(child, jpath(path, index))

    collect(document)
    return {"format": 1, "title": "Statim HTTP API v1", "error": error,
            "metric_families": metric_families(document), "operations": operations,
            "annotations": dict(sorted(annotations.items()))}


def jpath(base, key):
    if isinstance(key, int):
        return f"{base}[{key}]"
    if key.isidentifier():
        return f"{base}.{key}"
    return f"{base}[{json.dumps(key)}]"


def type_set(schema):
    value = schema.get("type")
    if value is None:
        return None
    return set(value if isinstance(value, list) else [value])


def rendered(value):
    if value is MISSING:
        return "absent"
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def compare_bound(old, new, path, keyword, direction, mode, breaks, notes):
    before, after = old.get(keyword, MISSING), new.get(keyword, MISSING)
    if before == after:
        return
    where = jpath(path, keyword)
    if mode == "response":
        notes.append(f"{where}: response bound changed from {rendered(before)} to {rendered(after)}")
        return
    if after is MISSING:
        notes.append(f"{where}: request bound removed (was {rendered(before)})")
        return
    if before is MISSING:
        breaks.append(f"{where}: request bound added: {rendered(after)}")
        return
    try:
        tighter = after < before if direction == "max" else after > before
        looser = after > before if direction == "max" else after < before
    except TypeError:
        tighter = looser = False
    level = notes if looser else breaks
    change = "loosened" if looser else "tightened" if tighter else "changed"
    level.append(f"{where}: request bound {change} from {rendered(before)} to {rendered(after)}")


def compare_multiple_of(old, new, path, mode, breaks, notes):
    before, after = old.get("multipleOf", MISSING), new.get("multipleOf", MISSING)
    if before == after:
        return
    where = jpath(path, "multipleOf")
    if mode == "response":
        notes.append(f"{where}: response bound changed from {rendered(before)} to {rendered(after)}")
        return
    if after is MISSING:
        notes.append(f"{where}: request bound removed (was {rendered(before)})")
        return
    if before is MISSING:
        breaks.append(f"{where}: request bound added: {rendered(after)}")
        return
    # A multiple whose ratio to the old value is integral accepts a subset;
    # the reverse is a relaxation. Fractions avoid binary-float modulo errors.
    try:
        before_fraction = Fraction(str(before))
        after_fraction = Fraction(str(after))
        tighter = (after_fraction > 0 and before_fraction > 0 and
                   (after_fraction / before_fraction).denominator == 1)
        looser = (after_fraction > 0 and before_fraction > 0 and
                  (before_fraction / after_fraction).denominator == 1)
    except (TypeError, ValueError, ZeroDivisionError):
        tighter = looser = False
    level = notes if looser and not tighter else breaks
    change = "loosened" if level is notes else "tightened" if tighter else "changed"
    level.append(f"{where}: request bound {change} from {rendered(before)} to {rendered(after)}")


def compare_constraint(old, new, path, keyword, mode, breaks, notes):
    before, after = old.get(keyword, MISSING), new.get(keyword, MISSING)
    if before == after:
        return
    where = jpath(path, keyword)
    if mode == "response":
        notes.append(f"{where}: response validation changed from {rendered(before)} to {rendered(after)}")
    elif after is MISSING:
        notes.append(f"{where}: request validation removed (was {rendered(before)})")
    else:
        breaks.append(f"{where}: request validation changed from {rendered(before)} to {rendered(after)}")


def compare_default(old, new, path, mode, breaks, notes):
    before, after = old.get("default", MISSING), new.get("default", MISSING)
    if before == after:
        return
    message = (f"{jpath(path, 'default')}: {mode} default changed from "
               f"{rendered(before)} to {rendered(after)}")
    (breaks if mode == "request" else notes).append(message)


def without_response_notes(value):
    if isinstance(value, dict):
        return {key: without_response_notes(child) for key, child in value.items()
                if key not in RESPONSE_NOTE_KEYS}
    if isinstance(value, list):
        return [without_response_notes(child) for child in value]
    return value


def compare_schema(old, new, path, mode, breaks, notes):
    old_types, new_types = type_set(old), type_set(new)
    if old_types != new_types:
        where = jpath(path, "type")
        if mode == "request" and old_types and new_types and old_types < new_types:
            notes.append(f"{where}: request type widened from {sorted(old_types)} to {sorted(new_types)}")
        else:
            breaks.append(f"{where}: {mode} type changed from {old.get('type')} to {new.get('type')}")

    old_enum = set(map(json.dumps, old.get("enum", [])))
    new_enum = set(map(json.dumps, new.get("enum", [])))
    if mode == "request" and "enum" in new and "enum" not in old:
        # a field that accepted any value of its type now accepts only the listed ones
        breaks.append(f"{jpath(path, 'enum')}: request enum added: values now restricted to {sorted(new_enum)}")
        old_enum = new_enum
    for value in sorted(old_enum - new_enum):
        breaks.append(f"{jpath(path, 'enum')}: {mode} enum value removed: {value}")
    for value in sorted(new_enum - old_enum):
        notes.append(f"{jpath(path, 'enum')}: {mode} enum value added: {value}")
    if old.get("const", MISSING) != new.get("const", MISSING):
        if mode == "response" and "const" not in old:
            notes.append(f"{jpath(path, 'const')}: response constant added")
        else:
            breaks.append(f"{jpath(path, 'const')}: {mode} constant changed")

    for keyword in MAX_BOUNDS:
        compare_bound(old, new, path, keyword, "max", mode, breaks, notes)
    for keyword in MIN_BOUNDS:
        compare_bound(old, new, path, keyword, "min", mode, breaks, notes)
    compare_multiple_of(old, new, path, mode, breaks, notes)
    for keyword in ("pattern", "format"):
        compare_constraint(old, new, path, keyword, mode, breaks, notes)
    compare_default(old, new, path, mode, breaks, notes)

    old_props, new_props = old.get("properties", {}), new.get("properties", {})
    old_required, new_required = set(old.get("required", [])), set(new.get("required", []))
    for name in sorted(old_props.keys() - new_props.keys()):
        breaks.append(f"{jpath(jpath(path, 'properties'), name)}: {mode} field removed")
    for name in sorted(new_props.keys() - old_props.keys()):
        field_path = jpath(jpath(path, "properties"), name)
        if mode == "request" and name in new_required:
            breaks.append(f"{field_path}: new required request field")
        else:
            notes.append(f"{field_path}: new {mode} field")
    for name in sorted(old_props.keys() & new_props.keys()):
        compare_schema(old_props[name], new_props[name],
                       jpath(jpath(path, "properties"), name), mode, breaks, notes)
    for name in sorted((new_required - old_required) & old_props.keys()):
        if mode == "request":
            breaks.append(f"{jpath(jpath(path, 'required'), name)}: request field made required")
        else:
            notes.append(f"{jpath(jpath(path, 'required'), name)}: response field made required")
    if mode == "request":
        for name in sorted((old_required - new_required) & old_props.keys()):
            notes.append(f"{jpath(jpath(path, 'required'), name)}: request field made optional")
    if mode == "response":
        for name in sorted((old_required - new_required) & old_props.keys()):
            breaks.append(f"{jpath(jpath(path, 'required'), name)}: response field made optional")

    for keyword in ("items", "additionalProperties", "not"):
        if isinstance(old.get(keyword), dict) and isinstance(new.get(keyword), dict):
            compare_schema(old[keyword], new[keyword], jpath(path, keyword), mode, breaks, notes)
        elif old.get(keyword) != new.get(keyword):
            breaks.append(f"{jpath(path, keyword)}: {mode} schema changed")
    for keyword in ("oneOf", "anyOf", "allOf", "prefixItems"):
        a, b = old.get(keyword, []), new.get(keyword, [])
        if len(a) != len(b):
            breaks.append(f"{jpath(path, keyword)}: {mode} alternatives changed")
        for index, (left, right) in enumerate(zip(a, b)):
            compare_schema(left, right, jpath(jpath(path, keyword), index), mode, breaks, notes)


def compare_contract(old, new):
    breaks, notes = [], []
    old_annotations = old.get("annotations", {})
    new_annotations = new.get("annotations", {})
    for path in sorted(old_annotations.keys() | new_annotations.keys()):
        if old_annotations.get(path) != new_annotations.get(path):
            notes.append(f"{path}: description or example changed")
    if without_response_notes(old.get("error")) != without_response_notes(new.get("error")):
        breaks.append("$.error: Error body shape changed")
    old_metrics = {item["name"]: item["kind"] for item in old.get("metric_families", [])}
    new_metrics = {item["name"]: item["kind"] for item in new.get("metric_families", [])}
    for name in sorted(old_metrics.keys() - new_metrics.keys()):
        breaks.append(f"{jpath('$.metric_families', name)}: metric family removed")
    for name in sorted(new_metrics.keys() - old_metrics.keys()):
        notes.append(f"{jpath('$.metric_families', name)}: metric family added ({new_metrics[name]})")
    for name in sorted(old_metrics.keys() & new_metrics.keys()):
        if old_metrics[name] != new_metrics[name]:
            breaks.append(f"{jpath('$.metric_families', name)}: metric kind changed from "
                          f"{old_metrics[name]} to {new_metrics[name]}")
    old_ops, new_ops = old.get("operations", {}), new.get("operations", {})
    for name in sorted(old_ops.keys() - new_ops.keys()):
        breaks.append(f"{jpath('$.operations', name)}: operation removed")
    for name in sorted(new_ops.keys() - old_ops.keys()):
        notes.append(f"{jpath('$.operations', name)}: operation added")
    for name in sorted(old_ops.keys() & new_ops.keys()):
        before, after = old_ops[name], new_ops[name]
        base = jpath("$.operations", name)
        # Parameter removal, requirement and schema narrowing can break callers.
        bp = {(p["in"], p["name"]): p for p in before["parameters"]}
        ap = {(p["in"], p["name"]): p for p in after["parameters"]}
        for key in sorted(bp.keys() - ap.keys()):
            breaks.append(f"{jpath(base, 'parameters')}[{key[0]}:{key[1]}]: parameter removed")
        for key in sorted(ap.keys() - bp.keys()):
            level = breaks if ap[key]["required"] else notes
            level.append(f"{jpath(base, 'parameters')}[{key[0]}:{key[1]}]: " +
                         ("required parameter added" if ap[key]["required"] else "optional parameter added"))
        for key in sorted(bp.keys() & ap.keys()):
            ppath = f"{jpath(base, 'parameters')}[{key[0]}:{key[1]}]"
            if not bp[key]["required"] and ap[key]["required"]:
                breaks.append(ppath + ": parameter made required")
            compare_schema(bp[key]["schema"], ap[key]["schema"], jpath(ppath, "schema"),
                           "request", breaks, notes)
        if before["request"] is not None and after["request"] is None:
            breaks.append(jpath(base, "request") + ": request body removed")
        elif before["request"] is None and after["request"] is not None:
            level = breaks if after["request"]["required"] else notes
            level.append(jpath(base, "request") + (": required request body added" if after["request"]["required"]
                                                   else ": optional request body added"))
        elif before["request"] is not None and after["request"] is not None:
            if not before["request"]["required"] and after["request"]["required"]:
                breaks.append(jpath(base, "request.required") + ": request body made required")
            bc, ac = before["request"]["content"], after["request"]["content"]
            for media in sorted(bc.keys() - ac.keys()):
                breaks.append(f"{jpath(jpath(base, 'request.content'), media)}: request media type removed")
            for media in sorted(ac.keys() - bc.keys()):
                notes.append(f"{jpath(jpath(base, 'request.content'), media)}: request media type added")
            for media in sorted(bc.keys() & ac.keys()):
                compare_schema(bc[media], ac[media], jpath(jpath(base, "request.content"), media),
                               "request", breaks, notes)
        old_responses, new_responses = before["responses"], after["responses"]
        for status in sorted(old_responses.keys() - new_responses.keys()):
            breaks.append(f"{jpath(jpath(base, 'responses'), status)}: documented status removed")
        for status in sorted(new_responses.keys() - old_responses.keys()):
            notes.append(f"{jpath(jpath(base, 'responses'), status)}: documented status added")
        for status in sorted(old_responses.keys() & new_responses.keys()):
            bc, ac = old_responses[status], new_responses[status]
            for media in sorted(bc.keys() - ac.keys()):
                breaks.append(f"{jpath(jpath(jpath(base, 'responses'), status), media)}: response media type removed")
            for media in sorted(ac.keys() - bc.keys()):
                notes.append(f"{jpath(jpath(jpath(base, 'responses'), status), media)}: response media type added")
            for media in sorted(bc.keys() & ac.keys()):
                compare_schema(bc[media], ac[media],
                               jpath(jpath(jpath(base, "responses"), status), media),
                               "response", breaks, notes)
    return breaks, notes


def load_yaml(path):
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument("--openapi", type=Path, default=DEFAULT_OPENAPI)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args(argv)
    current = build_contract(load_yaml(args.openapi))
    if args.write:
        args.contract.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.contract}")
        return 0
    try:
        frozen = json.loads(args.contract.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"BREAKING $.contract: frozen contract missing: {args.contract}")
        return 1
    breaks, notes = compare_contract(frozen, current)
    for note in notes:
        print("NOTE " + note)
    for item in breaks:
        print("BREAKING " + item)
    if breaks:
        print(f"API v1 contract check failed: {len(breaks)} breaking change(s)")
        return 1
    print(f"API v1 contract check passed ({len(notes)} additive note(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
