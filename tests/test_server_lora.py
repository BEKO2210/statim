#!/usr/bin/env python3
"""CPU live-server test for LoRA adapter selection ("adapter": name | "auto" | "none" | null).

Needs the adapters written by tests/lora/build_fixtures.py (ctest fixture "lora"): zero.gguf
(lora_B = 0, category emotion) and random.gguf (category sentiment).

Server A: -m multilingual=BASE -m english=BASE, adapters multilingual:zero and multilingual:random
(merge mode). Checks selection, auto routing, model selection by adapter, the batch endpoint,
/v1/models, /metrics and request errors.
Server B: same weights with --adapter-mode runtime and micro-batching; concurrent requests for
different adapters must each get their own adapter's answers.
Server C: no adapters; responses keep their pre-adapter shape.
Startup errors: unknown model, reserved name, bad file.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

STATE = "Worst experience ever. The support agent hung up on me twice and I am furious."
STATES = [STATE, "Great product, would buy again. Delivery was fast.", "Il pagamento non va a buon fine, errore 402."]
Q_SENTIMENT = {"sentiment": {"type": "choice", "instructions": "What is the sentiment of the message?",
                             "criteria": ["positive", "neutral", "negative"]}}
Q_EMOTION = {"emotion": {"type": "choice", "instructions": "Which emotion does the writer express?",
                         "criteria": ["joy", "anger", "sadness", "fear"]}}
Q_URGENCY = {"urgency": {"type": "score", "instructions": "How urgent is this request?",
                         "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]}}
Q_MIXED = dict(Q_SENTIMENT, **Q_URGENCY)
Q_ALL = dict(Q_MIXED, churn={"type": "noul", "instructions": "Does the user threaten to cancel or leave?"})

failures = []


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        failures.append(what)


def reserve_port():
    probe = socket.socket()
    try:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]
    finally:
        probe.close()


def call(port, method, path, payload=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    conn.request(method, path, json.dumps(payload) if payload is not None else None,
                 {"Content-Type": "application/json"})
    res = conn.getresponse()
    body = res.read()
    conn.close()
    try:
        return res.status, json.loads(body)
    except ValueError:
        return res.status, body.decode()


def decide(port, questions, state=STATE, **extra):
    payload = dict(state=state, questions=questions, return_logits=True, **extra)
    status, body = call(port, "POST", "/v1/systemone", payload)
    assert status == 200, (status, body)
    return body


class Server:
    def __init__(self, binary, args, log_path):
        self.port = reserve_port()
        env = dict(os.environ, STATIM_DEVICE="cpu")
        for k in ("STATIM_API_KEY", "STATIM_GPU_FAST"):
            env.pop(k, None)
        self.log = open(log_path, "w+")
        self.log_path = log_path
        self.proc = subprocess.Popen([binary, "serve", "--device", "cpu", "--threads", "2", "--port", str(self.port),
                                      "--no-access-log"] + args, env=env, stdout=self.log, stderr=self.log)
        for _ in range(1200):
            if self.proc.poll() is not None:
                raise AssertionError(Path(log_path).read_text())
            try:
                if call(self.port, "GET", "/health")[0] == 200:
                    return
            except (OSError, http.client.HTTPException):
                time.sleep(0.05)
        raise AssertionError("server did not become healthy")

    def close(self):
        self.proc.terminate()
        self.proc.wait(timeout=30)
        self.log.close()


def max_logit_diff(a, b):
    d = 0.0
    for qid, ans in a["answers"].items():
        for x, y in zip(ans["logits"], b["answers"][qid]["logits"]):
            d = max(d, abs(x - y))
    return d


def test_server_a(binary, model, adapters, tmp):
    print("server A: merge mode")
    srv = Server(binary, ["-m", "multilingual=" + model, "-m", "english=" + model,
                          "--adapter", "multilingual:zero=" + str(adapters / "zero.gguf"),
                          "--adapter", "multilingual:random=" + str(adapters / "random.gguf")], tmp / "a.log")
    try:
        p = srv.port
        status, models = call(p, "GET", "/v1/models")
        by_id = {m["id"]: m for m in models["data"]}
        ad = {a["id"]: a for a in by_id["multilingual"]["adapters"]}
        check(status == 200 and set(ad) == {"zero", "random"} and by_id["english"]["adapters"] == [],
              "/v1/models lists the adapters per model")
        check(ad["zero"]["categories"] == ["emotion"] and ad["random"]["categories"] == ["sentiment"]
              and ad["random"]["mode"] == "merge" and ad["random"]["pairs_applied"] == 87 and ad["zero"]["bytes"] == 0,
              "/v1/models adapter details (categories, mode, pairs applied, bytes)")

        base = decide(p, Q_ALL, model="multilingual")
        check(base["routing"]["adapter"] is None and base["routing"]["adapter_reason"] == "none",
              "no adapter field: base weights, routing.adapter null")
        for value in (None, "none"):
            r = decide(p, Q_ALL, model="multilingual", adapter=value)
            check(r["answers"] == base["answers"] and r["routing"]["adapter"] is None,
                  "adapter %r: identical to the base" % (value,))
        zero = decide(p, Q_ALL, model="multilingual", adapter="zero")
        check(zero["answers"] == base["answers"], "zero adapter: answers bit-identical to the base")
        check(zero["routing"]["adapter"] == "zero" and zero["routing"]["adapter_reason"] == "requested",
              "zero adapter: routing names it")
        rnd = decide(p, Q_ALL, model="multilingual", adapter="random")
        check(max_logit_diff(rnd, base) > 1e-2 and rnd["routing"]["adapter"] == "random",
              "random adapter: logits differ from the base (max %.3f)" % max_logit_diff(rnd, base))

        # a named adapter picks the only model that has it, even for English text
        en = decide(p, Q_SENTIMENT, state="Thanks, that fixed it. Have a great day!")
        r = decide(p, Q_SENTIMENT, state="Thanks, that fixed it. Have a great day!", adapter="random")
        check(en["routing"]["model"] == "english" and r["routing"]["model"] == "multilingual"
              and r["routing"]["reason"] == "adapter", "a named adapter routes to the model that carries it")

        # auto routing by question family
        r = decide(p, Q_SENTIMENT, model="multilingual", adapter="auto")
        want = decide(p, Q_SENTIMENT, model="multilingual", adapter="random")
        check(r["routing"]["adapter"] == "random" and r["routing"]["adapter_reason"] == "auto:sentiment"
              and r["answers"] == want["answers"], "auto: sentiment question -> random (category sentiment)")
        r = decide(p, Q_EMOTION, model="multilingual", adapter="auto")
        check(r["routing"]["adapter"] == "zero" and r["routing"]["adapter_reason"] == "auto:emotion",
              "auto: emotion question -> zero (category emotion)")
        r = decide(p, Q_MIXED, model="multilingual", adapter="auto")
        check(r["routing"]["adapter"] is None and r["routing"]["adapter_reason"] == "auto:no-family"
              and r["answers"] == decide(p, Q_MIXED, model="multilingual")["answers"],
              "auto: mixed families -> base weights")
        r = decide(p, Q_URGENCY, model="multilingual", adapter="auto")
        check(r["routing"]["adapter"] is None and r["routing"]["adapter_reason"] == "auto:urgency:no-adapter",
              "auto: family without an adapter -> base weights")
        r = decide(p, Q_SENTIMENT, model="english", adapter="auto")
        check(r["routing"]["model"] == "english" and r["routing"]["adapter"] is None
              and r["routing"]["adapter_reason"] == "auto:sentiment:no-adapter", "auto on a model without adapters")

        # batch endpoint
        status, batch = call(p, "POST", "/v1/systemone/batch",
                             {"states": STATES, "questions": Q_SENTIMENT, "model": "multilingual", "adapter": "random",
                              "return_logits": True})
        singles = [decide(p, Q_SENTIMENT, state=s, model="multilingual", adapter="random") for s in STATES]
        check(status == 200 and all(b["routing"]["adapter"] == "random" for b in batch["results"])
              and max(max_logit_diff(b, s) for b, s in zip(batch["results"], singles)) <= 1e-4,
              "batch endpoint applies the adapter to every state")

        # errors
        for payload, code, needle, what in [
            ({"model": "multilingual", "adapter": "nope"}, 422,
             "unknown adapter 'nope' for model 'multilingual' (loaded: zero, random)", "unknown adapter"),
            ({"adapter": "nope"}, 422,  # English text routes to the english model
             "unknown adapter 'nope' for model 'english' (none loaded; other models: multilingual:zero, multilingual:random)",
             "unknown adapter, language-routed model"),
            ({"adapter": 5}, 422, "adapter must be null or a string", "adapter of the wrong type"),
            ({"adapter": "x" * 300}, 422, "adapter must be null or a string", "overlong adapter name"),
            ({"model": "english", "adapter": "zero"}, 422, "unknown adapter 'zero' for model 'english' (none loaded;",
             "adapter the explicit model does not carry"),
            ({"model": "consensus", "adapter": "zero"}, 422, "cannot be combined with consensus", "adapter with consensus"),
        ]:
            status, body = call(p, "POST", "/v1/systemone", dict(state=STATE, questions=Q_SENTIMENT, **payload))
            check(status == code and needle in body.get("detail", ""), "%s -> %d (%s)" % (what, status, body.get("detail")))
        status, body = call(p, "POST", "/v1/systemone", {"state": STATE, "questions": Q_SENTIMENT, "model": "consensus",
                                                         "adapter": "auto"})
        check(status == 200 and body["routing"]["adapter"] is None and body["routing"]["adapter_reason"] == "auto:consensus",
              "consensus with adapter auto uses the base weights")

        status, metrics = call(p, "GET", "/metrics")
        check('statim_adapter_info{model="multilingual",adapter="random",mode="merge"' in metrics,
              "/metrics exports statim_adapter_info")
        return {"base": base, "zero": zero, "random": rnd}
    finally:
        srv.close()


def test_server_b(binary, model, adapters, tmp, ref):
    print("server B: runtime mode + micro-batching")
    srv = Server(binary, ["-m", "multilingual=" + model, "--adapter-mode", "runtime",
                          "--adapter", "zero=" + str(adapters / "zero.gguf"),
                          "--adapter", "random=" + str(adapters / "random.gguf"),
                          "--batch-window-ms", "30", "--max-batch", "16", "--workers", "2"], tmp / "b.log")
    try:
        p = srv.port
        status, models = call(p, "GET", "/v1/models")
        check(models["data"][0]["adapters"][1]["mode"] == "runtime", "runtime mode reported in /v1/models")
        kinds = ["base", "zero", "random"] * 4
        with ThreadPoolExecutor(max_workers=len(kinds)) as pool:
            futures = [pool.submit(decide, p, Q_ALL, STATE, adapter=None if k == "base" else k) for k in kinds]
            results = [f.result() for f in futures]
        worst = max(max_logit_diff(r, ref[k]) for r, k in zip(results, kinds))
        check(all(r["routing"]["adapter"] == (None if k == "base" else k) for r, k in zip(results, kinds)),
              "concurrent requests keep their adapter")
        check(worst <= 1e-4, "runtime + micro-batched answers match merge mode per adapter (max |dlogit| %.2e)" % worst)
    finally:
        srv.close()


def test_server_plain(binary, model, tmp):
    print("server C: no adapters")
    srv = Server(binary, ["-m", "multilingual=" + model], tmp / "c.log")
    try:
        p = srv.port
        r = decide(p, Q_SENTIMENT)
        check("adapter" not in r["routing"] and "adapter_reason" not in r["routing"],
              "without adapters the routing object is unchanged")
        check("adapters" not in call(p, "GET", "/v1/models")[1]["data"][0], "without adapters /v1/models is unchanged")
        r = decide(p, Q_SENTIMENT, adapter="auto")
        check(r["routing"]["adapter"] is None and r["routing"]["adapter_reason"] == "auto:sentiment:no-adapter",
              "a request that sets adapter gets the routing fields")
        status, body = call(p, "POST", "/v1/systemone", {"state": STATE, "questions": Q_SENTIMENT, "adapter": "emotion"})
        check(status == 422 and "unknown adapter 'emotion' for model 'multilingual' (none loaded)" in body["detail"],
              "named adapter on a server without adapters -> 422")
    finally:
        srv.close()


def test_startup_errors(binary, model, adapters):
    print("startup errors")
    for args, needle in [
        (["--adapter", "nosuch:x=" + str(adapters / "zero.gguf")], "no model named 'nosuch'"),
        (["--adapter", "auto=" + str(adapters / "zero.gguf")], "must be 1-64 of"),
        (["--adapter", "a=" + model], "is not a Statim LoRA adapter"),
        (["--adapter", "a"], "--adapter expects"),
        (["--adapter-mode", "fast"], "--adapter-mode must be merge or runtime"),
    ]:
        r = subprocess.run([binary, "serve", "-m", "multilingual=" + model, "--port", str(reserve_port())] + args,
                           capture_output=True, text=True, timeout=120, env=dict(os.environ, STATIM_DEVICE="cpu"))
        check(r.returncode != 0 and needle in r.stderr, "%s -> exit %d" % (" ".join(args)[:60], r.returncode))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapters", required=True, help="directory with zero.gguf and random.gguf")
    a = ap.parse_args()
    adapters = Path(a.adapters)
    if not (os.path.exists(a.binary) and os.path.exists(a.model) and (adapters / "random.gguf").exists()):
        print("skipping: binary, model or adapters missing")
        return 77
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ref = test_server_a(a.binary, a.model, adapters, tmp)
        test_server_b(a.binary, a.model, adapters, tmp, ref)
        test_server_plain(a.binary, a.model, tmp)
        test_startup_errors(a.binary, a.model, adapters)
    print("FAIL: %d checks" % len(failures) if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
