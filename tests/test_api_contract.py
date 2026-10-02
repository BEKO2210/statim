#!/usr/bin/env python3
"""Live OpenAPI example contract test for a CPU Statim server.

Every named POST request example is exercised.  Every response example is
validated statically, including examples without a matching request.  Some
responses are intentionally not induced live: inferenceFailed needs fault
injection; inferenceDeadline and queueDeadline depend on machine timing;
inferenceCancelled needs a client disconnect; busy and /ready saturation need a
race with admitted work; responseFinal can only occur if the conservative size
estimate underestimates serialization.  HTTP framing, parser/limit failures,
saturation, timeouts, and disconnect handling are exercised by security_http.

Exit 77 means localhost binding or the Python validation dependencies are not
available.  Run manually with: python3 tests/test_api_contract.py --binary build/statim
"""

import argparse
import copy
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

try:
    import yaml
    import jsonschema
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012
except ImportError as exc:
    # CI sets STATIM_CONTRACT_REQUIRED=1: there a missing dependency is a failure, not a skip
    required = os.environ.get("STATIM_CONTRACT_REQUIRED") == "1"
    print(f"{'FAIL' if required else 'SKIP'}: API contract test requires PyYAML and jsonschema ({exc})", flush=True)
    raise SystemExit(1 if required else 77)


ROOT = Path(__file__).resolve().parents[1]
OPENAPI_URI = "urn:statim:openapi"
METHODS = {"get", "put", "post", "delete", "options", "head", "patch", "trace"}


def resolve(document, value):
    while isinstance(value, dict) and "$ref" in value:
        ref = value["$ref"]
        if not ref.startswith("#/"):
            raise AssertionError("external reference in OpenAPI: " + ref)
        target = document
        for part in ref[2:].split("/"):
            target = target[part.replace("~1", "/").replace("~0", "~")]
        value = target
    return value


def content_examples(content):
    if "examples" in content:
        return content["examples"]
    if "example" in content:
        return {"example": {"value": content["example"]}}
    return {}


class SchemaValidator:
    def __init__(self, openapi):
        self.openapi = openapi
        resource = Resource(contents=openapi, specification=DRAFT202012)
        self.registry = Registry().with_resource(OPENAPI_URI, resource)

    def validate(self, instance, schema, label):
        def absolute_refs(value):
            if isinstance(value, dict):
                return {key: (OPENAPI_URI + item if key == "$ref" and item.startswith("#")
                              else absolute_refs(item)) for key, item in value.items()}
            if isinstance(value, list):
                return [absolute_refs(item) for item in value]
            return value
        schema = absolute_refs(copy.deepcopy(schema))
        try:
            jsonschema.Draft202012Validator(schema, registry=self.registry).validate(instance)
        except jsonschema.ValidationError as exc:
            where = "/".join(map(str, exc.absolute_path)) or "$"
            raise AssertionError(f"{label}: schema validation failed at {where}: {exc.message}") from exc


def static_response_examples(openapi, validator):
    count = 0
    for path, path_item in openapi["paths"].items():
        for method, operation in path_item.items():
            if method not in METHODS:
                continue
            for status, raw_response in operation.get("responses", {}).items():
                response = resolve(openapi, raw_response)
                for media, content in response.get("content", {}).items():
                    schema = content.get("schema", {})
                    for name, example in content_examples(content).items():
                        validator.validate(example["value"], schema,
                                           f"{method.upper()} {path} {status} {name}")
                        count += 1
    print(f"static response schemas: {count} examples valid", flush=True)


def reserve_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def request(port, method, path, payload=None, token=None):
    headers = {}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    body = payload if isinstance(payload, (bytes, str)) else (
        json.dumps(payload, separators=(",", ":")) if payload is not None else None)
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    conn.request(method, path, body=body, headers=headers)
    response = conn.getresponse()
    data = response.read()
    result = response.status, data, dict(response.getheaders())
    conn.close()
    return result


class Server:
    def __init__(self, binary, arguments, log_path):
        env = dict(os.environ, STATIM_DEVICE="cpu")
        env.pop("STATIM_API_KEY", None)
        env.pop("STATIM_GPU_FAST", None)
        self.log_path = log_path
        for _ in range(5):
            self.port = reserve_port()
            self.log = log_path.open("w+")
            command = [binary, "serve", "--device", "cpu", "--threads", "2", "--port",
                       str(self.port), "--no-access-log"] + arguments
            self.proc = subprocess.Popen(command, env=env, stdout=self.log, stderr=self.log)
            if self._healthy():
                return
            self.log.close()
            if "cannot listen" not in log_path.read_text(encoding="utf-8"):
                raise AssertionError(log_path.read_text(encoding="utf-8"))
        raise AssertionError("no free localhost port after five attempts")

    def _healthy(self):
        for _ in range(2400):
            if self.proc.poll() is not None:
                return False
            try:
                if request(self.port, "GET", "/health")[0] == 200:
                    return True
            except (OSError, http.client.HTTPException):
                time.sleep(0.05)
        raise AssertionError("server did not become healthy\n" + self.log_path.read_text(encoding="utf-8"))

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.log.close()


def json_body(data, label):
    try:
        return json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AssertionError(f"{label}: response is not JSON: {data[:200]!r}") from exc


def compare_example(actual, expected, tolerance, path="$", largest=None):
    if largest is None:
        largest = [0.0, "$", 0.0, 0.0]
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise AssertionError(f"{path}: expected object, got {type(actual).__name__}")
        if set(actual) != set(expected):
            raise AssertionError(f"{path}: keys differ: expected {sorted(expected)}, got {sorted(actual)}")
        for key in expected:
            compare_example(actual[key], expected[key], tolerance, f"{path}.{key}", largest)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{path}: expected array of length {len(expected)}")
        for index, (got, want) in enumerate(zip(actual, expected)):
            compare_example(got, want, tolerance, f"{path}[{index}]", largest)
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if type(actual) is not type(expected) or actual != expected:
            raise AssertionError(f"{path}: expected {expected!r}, got {actual!r}")
    elif isinstance(expected, (int, float)):
        if isinstance(actual, bool) or type(actual) is not type(expected):
            raise AssertionError(f"{path}: expected {type(expected).__name__}, got {type(actual).__name__}")
        difference = abs(float(actual) - float(expected))
        if difference > largest[0]:
            largest[:] = [difference, path, float(expected), float(actual)]
    else:
        raise AssertionError(f"{path}: unsupported example type {type(expected).__name__}")
    return largest


def response_entry(openapi, path, status):
    return resolve(openapi, openapi["paths"][path]["post"]["responses"][str(status)])


def run_post_examples(openapi, validator, port, tolerance):
    total = 0
    for path in ("/v1/systemone", "/v1/systemone/batch"):
        operation = openapi["paths"][path]["post"]
        requests = operation["requestBody"]["content"]["application/json"]["examples"]
        for name, request_example in requests.items():
            expected_status = request_example["x-expect-status"]
            response = response_entry(openapi, path, expected_status)
            content = response["content"]["application/json"]
            if name not in content.get("examples", {}):
                raise AssertionError(f"POST {path} request example {name!r} expects HTTP {expected_status}, "
                                     f"but that response has no example with the same name")
            expected = content["examples"][name]["value"]
            status, raw, _ = request(port, "POST", path, request_example["value"])
            label = f"POST {path} example {name}"
            if status != expected_status:
                raise AssertionError(f"{label}: expected HTTP {expected_status}, got {status}: {raw[:500]!r}")
            actual = json_body(raw, label)
            if status == 200:
                validator.validate(actual, content["schema"], label)
                largest = compare_example(actual, expected, tolerance)
                print(f"{label}: HTTP 200, max numeric deviation {largest[0]:.8g} at {largest[1]}", flush=True)
                if largest[0] > tolerance:
                    raise AssertionError(
                        f"{label}: documentation drift at {largest[1]}: expected {largest[2]:.8g}, "
                        f"got {largest[3]:.8g}; max deviation {largest[0]:.8g} exceeds "
                        f"tolerance {tolerance:g}")
            else:
                if actual != expected:
                    raise AssertionError(f"{label}: expected error {expected!r}, got {actual!r}")
                print(f"{label}: HTTP {status}, exact error body", flush=True)
            total += 1
    return total


def schema_for(openapi, path, status, media="application/json"):
    response = resolve(openapi, openapi["paths"][path]["get"]["responses"][str(status)])
    return response["content"][media]["schema"]


def run_get_checks(openapi, validator, normal, keyed):
    for path in ("/health", "/ready", "/v1/models"):
        status, raw, _ = request(normal, "GET", path)
        assert status == 200, (path, status, raw[:500])
        validator.validate(json_body(raw, path), schema_for(openapi, path, 200), "GET " + path)
        print(f"GET {path}: HTTP 200, schema valid", flush=True)

    status, raw, headers = request(normal, "GET", "/metrics")
    content_type = next((v for k, v in headers.items() if k.lower() == "content-type"), "")
    assert status == 200 and content_type == "text/plain; version=0.0.4", (status, content_type)
    assert raw.decode("utf-8"), "GET /metrics returned an empty body"
    print("GET /metrics: HTTP 200, Prometheus content type", flush=True)

    status, raw, headers = request(normal, "GET", "/")
    content_type = next((v for k, v in headers.items() if k.lower() == "content-type"), "")
    assert status == 200 and content_type.startswith("text/html"), (status, content_type)
    assert b"Statim Playground" in raw
    status, raw, _ = request(keyed, "GET", "/")
    assert status == 404 and json_body(raw, "disabled playground") == {"detail": "not found"}
    print("GET /: playground HTML and --no-playground 404 valid", flush=True)

    unauthorized = {"detail": "invalid or missing bearer token"}
    simple = {"state": "x", "questions": {}}
    batch = {"states": [], "questions": {}}
    for method, path, payload in (("POST", "/v1/systemone", simple),
                                  ("POST", "/v1/systemone/batch", batch),
                                  ("GET", "/v1/models", None), ("GET", "/metrics", None)):
        status, raw, _ = request(keyed, method, path, payload)
        assert status == 401 and json_body(raw, method + " " + path) == unauthorized
    print("keyed server: all documented 401 examples match exactly", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="build/statim")
    parser.add_argument("--english", default="models/laya-english-f32.gguf")
    parser.add_argument("--multilingual", default="models/laya-multilingual-f32.gguf")
    parser.add_argument("--openapi", type=Path, default=ROOT / "docs/openapi.yaml")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    openapi = yaml.safe_load(args.openapi.read_text(encoding="utf-8"))
    validator = SchemaValidator(openapi)
    static_response_examples(openapi, validator)
    try:
        reserve_port()
    except PermissionError:
        if os.environ.get("STATIM_CONTRACT_REQUIRED") == "1":
            print("FAIL: cannot bind localhost, and STATIM_CONTRACT_REQUIRED=1", flush=True)
            return 1
        print("SKIP: sandbox cannot bind localhost; static response examples passed", flush=True)
        return 77

    with tempfile.TemporaryDirectory(prefix="statim-api-contract-") as directory:
        tmp = Path(directory)
        key_file = tmp / "api-key"
        key_file.write_text("contract-test-key-0123456789abcdef\n", encoding="ascii")
        # Both checkpoints and no --consensus: the documented examples use language routing when
        # "model" is omitted, and opt into consensus per request ("model": "consensus").
        normal = Server(args.binary, ["-m", "english=" + args.english,
                                      "-m", "multilingual=" + args.multilingual], tmp / "normal.log")
        keyed = None
        try:
            keyed = Server(args.binary, ["-m", "multilingual=" + args.multilingual,
                                         "--api-key-file", str(key_file), "--no-playground"],
                           tmp / "keyed.log")
            total = run_post_examples(openapi, validator, normal.port, args.tolerance)
            run_get_checks(openapi, validator, normal.port, keyed.port)
            print(f"API contract: {total} POST request examples passed", flush=True)
        finally:
            if keyed is not None:
                keyed.close()
            normal.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
