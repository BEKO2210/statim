#!/usr/bin/env python3
"""Concurrency stress test for Statim server under ThreadSanitizer (READINESS P1 #46).

Tests concurrent execution with 32 clients under:
- HTTP worker pool
- admission (in_flight)
- engine pool and lease handling (weights switching: base, adapter A, adapter B)
- micro-batching (collector and workers)
- custom SGEMM op barrier
- cancellation and client disconnect
- calibration cache
- metrics, readiness, and health polling

Checks:
- All non-cancelled answers equal reference answers computed sequentially (tolerance 1e-4)
- No 5xx errors except documented 503s
- Server cleanly shuts down with SIGTERM (exit code 0)
- Server stderr contains no ThreadSanitizer, AddressSanitizer, or UB warnings
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

STATE_1 = "Great product, would buy again. Delivery was fast."
STATE_2 = "Worst experience ever. The support agent hung up on me twice and I am furious."
STATE_3 = "Il pagamento non va a buon fine, errore 402."
STATE_4 = "Can you send me a quote for 50 seats of the enterprise plan?"

# This test looks for races and wrong answers, not for the deadline: under ThreadSanitizer a CI
# runner serves about 1 request/s, and one queued request outlived the 120 s default. The clients
# wait as long as the server may take, so a slow runner cannot fail the test with a client timeout.
INFERENCE_TIMEOUT_S = 900
CLIENT_TIMEOUT_S = INFERENCE_TIMEOUT_S + 60

Q_SENTIMENT = {
    "sentiment": {
        "type": "choice",
        "instructions": "What is the sentiment of the message?",
        "criteria": ["positive", "neutral", "negative"],
    }
}

Q_EMOTION = {
    "emotion": {
        "type": "choice",
        "instructions": "Which emotion does the writer express?",
        "criteria": ["joy", "anger", "sadness", "fear"],
    }
}


def reserve_port():
    probe = socket.socket()
    try:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]
    finally:
        probe.close()


def assert_answers_equal(a, b):
    assert a.get("model") == b.get("model"), (a.get("model"), b.get("model"))
    assert a.get("routing") == b.get("routing"), (a.get("routing"), b.get("routing"))
    assert a["answers"].keys() == b["answers"].keys()
    for qid in a["answers"]:
        left, right = a["answers"][qid], b["answers"][qid]
        assert left["type"] == right["type"], (qid, left["type"], right["type"])
        if left["type"] == "choice":
            assert left["choice"] == right["choice"], (qid, left["choice"], right["choice"])
        for field in ("probabilities",):
            if field in left and field in right:
                assert left[field].keys() == right[field].keys()
                for key in left[field]:
                    assert abs(left[field][key] - right[field][key]) <= 1e-4, (qid, key, left, right)
        if "logits" in left and "logits" in right:
            for l_val, r_val in zip(left["logits"], right["logits"]):
                assert abs(l_val - r_val) <= 1e-4, (qid, left["logits"], right["logits"])
        for field in ("score", "noul", "confidence", "answer_confidence"):
            if field in left and field in right:
                assert abs(left[field] - right[field]) <= 1e-4, (qid, field, left, right)
        if "action" in left and "action" in right:
            assert abs(left["action"]["act_probability"] - right["action"]["act_probability"]) <= 1e-4, (
                qid,
                left["action"],
                right["action"],
            )


def assert_batch_answers_equal(a, b):
    assert "results" in a and "results" in b, (a.keys(), b.keys())
    assert len(a["results"]) == len(b["results"]), (len(a["results"]), len(b["results"]))
    for left, right in zip(a["results"], b["results"]):
        assert_answers_equal(left, right)


def ensure_adapters(model_path, adapters_dir):
    """Ensure zero.gguf and random.gguf exist in adapters_dir."""
    p_adapters = Path(adapters_dir)
    zero_p = p_adapters / "zero.gguf"
    random_p = p_adapters / "random.gguf"
    if zero_p.is_file() and random_p.is_file():
        return zero_p, random_p

    p_adapters.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    build_fixtures = root / "tests" / "lora" / "build_fixtures.py"
    if not build_fixtures.is_file():
        raise FileNotFoundError(f"Fixture builder not found: {build_fixtures}")

    cmd = [sys.executable, str(build_fixtures), "--base", str(model_path), "--out", str(p_adapters)]
    print(f"Building LoRA fixtures: {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)
    if not (zero_p.is_file() and random_p.is_file()):
        raise RuntimeError("Failed to build LoRA fixtures")
    return zero_p, random_p


def make_templates():
    """Build the request templates covering single/batch, adapters, calibration, disconnect, and polling."""
    return [
        # 0: Single request, base weights
        {
            "type": "single",
            "payload": {"state": STATE_1, "questions": Q_SENTIMENT, "return_logits": True},
            "desc": "single base sentiment",
        },
        # 1: Single request, adapter A (zero)
        {
            "type": "single",
            "payload": {"state": STATE_2, "questions": Q_EMOTION, "adapter": "zero", "return_logits": True},
            "desc": "single adapter zero emotion",
        },
        # 2: Single request, adapter B (random)
        {
            "type": "single",
            "payload": {"state": STATE_1, "questions": Q_SENTIMENT, "adapter": "random", "return_logits": True},
            "desc": "single adapter random sentiment",
        },
        # 3: Batch request, adapter A (zero)
        {
            "type": "batch",
            "payload": {
                "states": [STATE_2, STATE_3],
                "questions": Q_EMOTION,
                "adapter": "zero",
                "return_logits": True,
            },
            "desc": "batch adapter zero emotion",
        },
        # 4: Batch request, adapter B (random)
        {
            "type": "batch",
            "payload": {
                "states": [STATE_1, STATE_4],
                "questions": Q_SENTIMENT,
                "adapter": "random",
                "return_logits": True,
            },
            "desc": "batch adapter random sentiment",
        },
        # 5: Batch request, base weights
        {
            "type": "batch",
            "payload": {
                "states": [STATE_1, STATE_2],
                "questions": Q_SENTIMENT,
                "return_logits": True,
            },
            "desc": "batch base sentiment",
        },
        # 6: Repeated calibration request, base weights (calibration cache test)
        {
            "type": "single",
            "payload": {
                "state": STATE_1,
                "questions": Q_SENTIMENT,
                "calibrate": True,
                "return_logits": True,
            },
            "desc": "single base sentiment calibrate",
        },
        # 7: Client disconnect mid-way
        {
            "type": "disconnect",
            "payload": {
                "state": STATE_2,
                "questions": Q_EMOTION,
                "adapter": "zero",
            },
            "desc": "client disconnect mid-way",
        },
        # 8: Poll /metrics
        {
            "type": "metrics",
            "path": "/metrics",
            "desc": "poll /metrics",
        },
        # 9: Poll /ready
        {
            "type": "ready",
            "path": "/ready",
            "desc": "poll /ready",
        },
        # 10: Poll /health
        {
            "type": "health",
            "path": "/health",
            "desc": "poll /health",
        },
        # 11: Repeated calibration request, adapter B (calibration cache under adapter)
        {
            "type": "single",
            "payload": {
                "state": STATE_1,
                "questions": Q_SENTIMENT,
                "adapter": "random",
                "calibrate": True,
                "return_logits": True,
            },
            "desc": "single adapter random sentiment calibrate",
        },
    ]


def run_single_request(port, payload, timeout=CLIENT_TIMEOUT_S):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        body = json.dumps(payload)
        conn.request("POST", "/v1/systemone", body, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = resp.read().decode("utf-8")
        return resp.status, data
    finally:
        conn.close()


def run_batch_request(port, payload, timeout=CLIENT_TIMEOUT_S):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        body = json.dumps(payload)
        conn.request("POST", "/v1/systemone/batch", body, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = resp.read().decode("utf-8")
        return resp.status, data
    finally:
        conn.close()


def run_disconnect_request(port, payload):
    body_bytes = json.dumps(payload).encode("utf-8")
    req = (
        f"POST /v1/systemone HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{port}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body_bytes)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode("utf-8") + body_bytes
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=5)
        s.sendall(req)
        # Small delay so server reads request and begins micro-batching / inference
        time.sleep(0.01)
        s.close()
    except (OSError, http.client.HTTPException):
        pass


def run_poll_request(port, path, timeout=10):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        data = resp.read().decode("utf-8")
        return resp.status, data
    finally:
        conn.close()


def compute_sequential_references(port, templates):
    """Compute reference answers sequentially beforehand for every deterministic request."""
    references = {}
    print("Computing sequential reference answers...", flush=True)
    for idx, t in enumerate(templates):
        ttype = t["type"]
        if ttype == "single":
            status, body_text = run_single_request(port, t["payload"])
            assert status == 200, f"Sequential reference failed for template {idx} ({t['desc']}): status {status}\n{body_text}"
            references[idx] = json.loads(body_text)
        elif ttype == "batch":
            status, body_text = run_batch_request(port, t["payload"])
            assert status == 200, f"Sequential reference failed for template {idx} ({t['desc']}): status {status}\n{body_text}"
            references[idx] = json.loads(body_text)
    print(f"Computed {len(references)} sequential reference answers.", flush=True)
    return references


def main():
    parser = argparse.ArgumentParser(description="Statim server concurrency stress test")
    parser.add_argument("--binary", required=True, help="Path to statim binary")
    parser.add_argument("--model", required=True, help="Path to laya-multilingual-f32.gguf")
    parser.add_argument("--adapters", default=None, help="Path to adapters directory")
    parser.add_argument("--threads", type=int, default=4, help="Threads for server (default: 4)")
    parser.add_argument("--clients", type=int, default=32, help="Number of concurrent clients (default: 32)")
    parser.add_argument("--requests-per-client", type=int, default=12, help="Requests per client (default: 12)")
    args = parser.parse_args()

    # Preflight check: binary and model exist
    if not (os.path.isfile(args.binary) and os.path.isfile(args.model)):
        print(f"SKIP: binary '{args.binary}' or model '{args.model}' not found", flush=True)
        return 77

    # Check port binding
    try:
        port = reserve_port()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    except (OSError, PermissionError):
        print("SKIP: cannot bind loopback port in current environment", flush=True)
        return 77

    # Setup adapters
    adapters_dir = args.adapters
    if not adapters_dir:
        # Check standard locations or use a local lora dir
        candidate = Path("build/lora")
        if candidate.is_dir() and (candidate / "zero.gguf").is_file():
            adapters_dir = str(candidate)
        else:
            candidate = Path(os.path.dirname(args.binary)) / "lora"
            adapters_dir = str(candidate)

    try:
        zero_path, random_path = ensure_adapters(args.model, adapters_dir)
    except Exception as exc:
        print(f"SKIP: could not prepare LoRA adapters: {exc}", flush=True)
        return 77

    log_path = Path(f"/tmp/statim-concurrency-{port}.log")
    server_cmd = [
        args.binary,
        "serve",
        "-m",
        f"multilingual={args.model}",
        "--adapter",
        f"multilingual:zero={zero_path}",
        "--adapter",
        f"multilingual:random={random_path}",
        "--workers",
        "2",
        "--max-concurrent",
        "64",
        "--batch-window-ms",
        "5",
        "--max-batch",
        "16",
        "--threads",
        str(args.threads),
        "--port",
        str(port),
        "--no-access-log",
        "--inference-timeout",
        str(INFERENCE_TIMEOUT_S),
    ]

    env = dict(os.environ, STATIM_DEVICE="cpu", CUDA_VISIBLE_DEVICES="")
    env.pop("STATIM_API_KEY", None)
    env.pop("STATIM_GPU_FAST", None)

    print(f"Starting server on port {port} with command:\n  {' '.join(server_cmd)}", flush=True)
    server_log = open(log_path, "w+")
    proc = subprocess.Popen(server_cmd, env=env, stdout=server_log, stderr=server_log)

    try:
        # Wait for /health 200
        healthy = False
        for _ in range(600):
            if proc.poll() is not None:
                server_log.flush()
                log_content = log_path.read_text()
                raise AssertionError(f"Server died prematurely with code {proc.returncode}:\n{log_content}")
            try:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
                conn.request("GET", "/health")
                r = conn.getresponse()
                body = r.read().decode("utf-8")
                conn.close()
                if r.status == 200:
                    health_obj = json.loads(body)
                    if health_obj.get("status") == "ok":
                        healthy = True
                        break
            except (OSError, http.client.HTTPException):
                time.sleep(0.05)

        if not healthy:
            raise AssertionError(f"Server did not become healthy within 30s. Log:\n{log_path.read_text()}")

        print("Server is healthy. Beginning concurrency verification...", flush=True)
        templates = make_templates()
        references = compute_sequential_references(port, templates)

        n_clients = args.clients
        requests_per_client = args.requests_per_client
        total_requests = n_clients * requests_per_client
        n_templates = len(templates)

        request_counts = {t["desc"]: 0 for t in templates}
        counts_by_type = {"single": 0, "batch": 0, "disconnect": 0, "poll": 0}
        error_list = []

        print(
            f"Launching {n_clients} concurrent clients, {requests_per_client} requests each ({total_requests} total requests)...",
            flush=True,
        )
        t_start = time.monotonic()

        def client_worker(client_id):
            client_errors = []
            for step in range(requests_per_client):
                tpl_idx = (client_id + step) % n_templates
                tpl = templates[tpl_idx]
                ttype = tpl["type"]

                try:
                    if ttype == "single":
                        status, body_text = run_single_request(port, tpl["payload"])
                        if status == 200:
                            data = json.loads(body_text)
                            assert_answers_equal(data, references[tpl_idx])
                        elif status == 503:
                            # Verify documented 503
                            obj = json.loads(body_text)
                            detail = obj.get("detail", "")
                            if detail not in ("server busy, try again later", "inference queue deadline exceeded"):
                                client_errors.append(f"Client {client_id} step {step} ({tpl['desc']}) got undocumented 503: {body_text}")
                        else:
                            client_errors.append(f"Client {client_id} step {step} ({tpl['desc']}) unexpected status {status}: {body_text}")

                    elif ttype == "batch":
                        status, body_text = run_batch_request(port, tpl["payload"])
                        if status == 200:
                            data = json.loads(body_text)
                            assert_batch_answers_equal(data, references[tpl_idx])
                        elif status == 503:
                            obj = json.loads(body_text)
                            detail = obj.get("detail", "")
                            if detail not in ("server busy, try again later", "inference queue deadline exceeded"):
                                client_errors.append(f"Client {client_id} step {step} ({tpl['desc']}) got undocumented 503: {body_text}")
                        else:
                            client_errors.append(f"Client {client_id} step {step} ({tpl['desc']}) unexpected status {status}: {body_text}")

                    elif ttype == "disconnect":
                        run_disconnect_request(port, tpl["payload"])

                    elif ttype in ("metrics", "ready", "health"):
                        status, body_text = run_poll_request(port, tpl["path"], timeout=10)
                        if ttype == "metrics":
                            if status != 200 or "statim_requests_total" not in body_text:
                                client_errors.append(f"Client {client_id} step {step} /metrics returned status {status}")
                        elif ttype == "ready":
                            if status not in (200, 503):
                                client_errors.append(f"Client {client_id} step {step} /ready returned status {status}: {body_text}")
                        elif ttype == "health":
                            if status != 200 or "ok" not in body_text:
                                client_errors.append(f"Client {client_id} step {step} /health returned status {status}: {body_text}")

                except Exception as exc:
                    client_errors.append(f"Client {client_id} step {step} ({tpl['desc']}) exception: {exc}")

            return client_id, client_errors

        with ThreadPoolExecutor(max_workers=n_clients) as pool:
            futures = [pool.submit(client_worker, cid) for cid in range(n_clients)]
            for fut in futures:
                cid, errs = fut.result()
                if errs:
                    error_list.extend(errs)

        elapsed = time.monotonic() - t_start

        # Compute summary counts
        for cid in range(n_clients):
            for step in range(requests_per_client):
                tpl = templates[(cid + step) % n_templates]
                request_counts[tpl["desc"]] += 1
                if tpl["type"] in ("single", "batch", "disconnect"):
                    counts_by_type[tpl["type"]] += 1
                else:
                    counts_by_type["poll"] += 1

        print(f"\n--- Concurrency Run Summary ---")
        print(f"Runtime: {elapsed:.2f} s ({total_requests / elapsed:.1f} req/s)")
        print(f"Clients: {n_clients}, Requests per client: {requests_per_client}, Total: {total_requests}")
        print("Request counts by category:")
        for cat, cnt in counts_by_type.items():
            print(f"  {cat:12s}: {cnt:4d}")
        print("Request counts by template:")
        for desc, cnt in request_counts.items():
            print(f"  {desc:40s}: {cnt:4d}")

        if error_list:
            print(f"\nFAIL: {len(error_list)} errors encountered during concurrency wave:")
            for err in error_list[:20]:
                print(f"  - {err}")
            if len(error_list) > 20:
                print(f"  ... and {len(error_list) - 20} more errors")
            raise AssertionError(f"Encountered {len(error_list)} errors during concurrency test")

        # Verify server is still alive and responsive after wave
        status, health_body = run_poll_request(port, "/health")
        assert status == 200 and "ok" in health_body, f"Server unhealthy after test wave: {status} {health_body}"

        # Check /metrics one last time
        status, metrics_text = run_poll_request(port, "/metrics")
        assert status == 200, f"/metrics failed after test wave: {status}"

        print("All client requests verified successfully against sequential references.")

    finally:
        print("Stopping server with SIGTERM...", flush=True)
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                raise AssertionError("Server timed out after SIGTERM (exceeded 30s)")

        server_log.close()
        exit_code = proc.poll()
        log_text = log_path.read_text()

        # Clean up temporary log file
        try:
            log_path.unlink(missing_ok=True)
        except OSError:
            pass

        assert exit_code == 0, f"Server exited with non-zero status {exit_code}. Server log:\n{log_text}"

        # Sanitizer checks
        assert "WARNING: ThreadSanitizer" not in log_text, f"ThreadSanitizer warning in server log:\n{log_text}"
        assert "ERROR: AddressSanitizer" not in log_text, f"AddressSanitizer error in server log:\n{log_text}"
        assert "runtime error:" not in log_text, f"UBSan error in server log:\n{log_text}"

    print("Server exited cleanly (status 0). No sanitizer errors or races detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
