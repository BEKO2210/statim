"""Client tests.

Integration calls use the unauthenticated server at STATIM_URL (default
http://127.0.0.1:8190, multilingual CPU model) and a keyed server at
STATIM_AUTH_URL (default http://127.0.0.1:8191, key sdk-test-key).

Retry sequencing uses a local HTTP server so admission control does not have
to be saturated.
"""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from statim import (
    AuthenticationError,
    BadRequestError,
    ChoiceAnswer,
    Client,
    PayloadTooLargeError,
    ScoreAnswer,
    ServiceUnavailableError,
    TransportError,
    UnprocessableEntityError,
    YesNoAnswer,
)

BASE_URL = os.environ.get("STATIM_URL", "http://127.0.0.1:8190")
AUTH_URL = os.environ.get("STATIM_AUTH_URL", "http://127.0.0.1:8191")
AUTH_KEY = os.environ.get("STATIM_API_KEY_TEST", "sdk-test-key")

NOUL_BODY = {
    "model": "laya-multilingual",
    "answers": {
        "refund": {
            "type": "noul",
            "noul": 0.8,
            "confidence": 0.8,
            "answer_confidence": 0.8,
            "action": {"act_probability": 1.0},
        }
    },
    "usage": {"input_tokens": 10, "output_tokens": 0},
    "routing": {
        "model": "multilingual",
        "reason": "requested",
        "engine": "statim",
        "weights": "f32",
    },
}

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs and outages",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical"],
    },
    "refund": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?",
    },
}
STATE = {
    "subject": "Duplicate charge on invoice #4411",
    "body": "We were billed twice for March. Please refund the duplicate today.",
}


class _ScriptServer(ThreadingHTTPServer):
    hits: int
    last_body: bytes
    last_headers: dict[str, str]
    last_path: str


def _mock(script):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_GET(self):
            self._handle()

        def do_POST(self):
            self._handle()

        def _handle(self):
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length) if length else b""
            server = self.server
            assert isinstance(server, _ScriptServer)
            server.hits += 1
            server.last_body = body
            server.last_path = self.path
            server.last_headers = {key.lower(): value for key, value in self.headers.items()}
            status, extra, payload = script(server.hits, self)
            raw = payload.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            for key, value in extra.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    server = _ScriptServer(("127.0.0.1", 0), Handler)
    server.hits = 0
    server.last_body = b""
    server.last_headers = {}
    server.last_path = ""
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}", thread


@pytest.fixture(scope="session")
def base_url() -> str:
    client = Client(BASE_URL, timeout=5)
    try:
        health = client.health()
    except TransportError as exc:
        pytest.fail(f"Statim is not reachable at {BASE_URL} ({exc})")
    assert health.status == "ok"
    assert health.version == "0.3.0"
    return BASE_URL


@pytest.fixture(scope="session")
def auth() -> tuple[str, str]:
    client = Client(AUTH_URL, timeout=5)
    try:
        health = client.health()
    except TransportError as exc:
        pytest.fail(f"keyed Statim is not reachable at {AUTH_URL} ({exc})")
    assert health.status == "ok"
    return AUTH_URL, AUTH_KEY


def test_constructor_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        Client("not a url")
    with pytest.raises(ValueError):
        Client("http://127.0.0.1:1", timeout=0)
    with pytest.raises(ValueError):
        Client("http://127.0.0.1:1", max_retries=-1)
    client = Client("http://127.0.0.1:9", api_key="", timeout=1)
    assert client.api_key is None


def test_yes_no_probabilities_and_options_round_trip() -> None:
    seen: dict[str, Any] = {}

    def script(hit: int, handler: BaseHTTPRequestHandler):
        server = handler.server
        assert isinstance(server, _ScriptServer)
        seen["body"] = json.loads(server.last_body)
        seen["headers"] = server.last_headers
        assert hit == 1
        return 200, {"X-Request-Id": "opt-1", "X-Inference-Time-Ms": "1.50"}, json.dumps(NOUL_BODY)

    server, url, thread = _mock(script)
    try:
        client = Client(url, api_key="secret", timeout=5, max_retries=0)
        decision = client.decide(
            "hello",
            {"refund": {"type": "noul", "instructions": "Refund?"}},
            model=None,
            calibrate=False,
            ensemble=1,
            request_id="opt-1",
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert seen["body"] == {
        "state": "hello",
        "questions": {"refund": {"type": "noul", "instructions": "Refund?"}},
        "calibrate": False,
        "ensemble": 1,
    }
    assert seen["headers"]["x-request-id"] == "opt-1"
    assert seen["headers"]["authorization"] == "Bearer secret"
    assert "request_id" not in seen["body"]
    answer = decision.answers["refund"]
    assert isinstance(answer, YesNoAnswer)
    assert answer.type == "noul"
    assert answer.noul == 0.8
    assert answer.yes is True
    assert answer.probabilities == {"yes": 0.8, "no": pytest.approx(0.2)}
    assert answer.confidence == 0.8
    assert answer.escalate is None
    assert decision.request_id == "opt-1"
    assert decision.inference_time_ms == 1.5
    assert decision.usage.output_tokens == 0
    assert decision.routing.engine == "statim"


def test_selective_prediction_transport_serialization_and_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[tuple[str, dict[str, Any]]] = []
    selective_body = {
        **NOUL_BODY,
        "answers": {
            "refund": {**NOUL_BODY["answers"]["refund"], "escalate": True},
            "topic": {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"billing": 0.9, "other": 0.1},
                "confidence": 0.53,
                "answer_confidence": 0.9,
                "action": {"act_probability": 1.0},
                "escalate": False,
            },
            "urgency": {
                "type": "score",
                "score": 0.2,
                "legend": {"0": "low", "1": "high"},
                "probabilities": {"0": 0.8, "1": 0.2},
                "confidence": 0.28,
                "answer_confidence": 0.8,
                "action": {"act_probability": 1.0},
                "escalate": True,
            },
        },
    }

    class Response:
        status = 200
        headers = {"X-Request-Id": "selective"}

        def __init__(self, payload: Any) -> None:
            self.raw = json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def read(self) -> bytes:
            return self.raw

    def urlopen(request: Any, timeout: float):
        del timeout
        body = json.loads(request.data)
        requests.append((request.full_url, body))
        if request.full_url.endswith("/batch"):
            payload = {"results": [selective_body]}
        else:
            payload = selective_body if "min_confidence" in body else NOUL_BODY
        return Response(payload)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    client = Client("http://statim.invalid", max_retries=0)
    questions = {"refund": {"type": "noul", "instructions": "Refund?"}}
    decision = client.decide("hello", questions, min_confidence=0.85)
    batch = client.decide_batch(["hello"], questions, min_confidence=0.75)
    unchanged = client.decide("hello", questions)

    assert requests == [
        ("http://statim.invalid/v1/systemone", {"state": "hello", "questions": questions, "min_confidence": 0.85}),
        (
            "http://statim.invalid/v1/systemone/batch",
            {"states": ["hello"], "questions": questions, "min_confidence": 0.75},
        ),
        ("http://statim.invalid/v1/systemone", {"state": "hello", "questions": questions}),
    ]
    assert decision.answers["refund"].escalate is True
    assert decision.answers["topic"].escalate is False
    assert decision.answers["urgency"].escalate is True
    assert batch.results[0].answers["refund"].escalate is True
    assert unchanged.answers["refund"].escalate is None


def test_retries_honor_retry_after_and_stop_after_success() -> None:
    def script(hit: int, handler: BaseHTTPRequestHandler):
        echoed = handler.headers["X-Request-Id"]
        if hit < 3:
            return 503, {"Retry-After": "0", "X-Request-Id": echoed}, '{"detail":"server busy, try again later"}'
        return 200, {"X-Request-Id": echoed, "X-Inference-Time-Ms": "3.00"}, json.dumps(NOUL_BODY)

    server, url, thread = _mock(script)
    try:
        started = time.monotonic()
        decision = Client(url, timeout=5, max_retries=2, backoff=30).decide(
            "x",
            {"q": {"type": "noul", "instructions": "?"}},
            request_id="retry-ok",
        )
        elapsed = time.monotonic() - started
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    assert server.hits == 3
    assert elapsed < 2
    assert decision.request_id == "retry-ok"
    assert isinstance(decision.answers["refund"], YesNoAnswer)


def test_retry_after_seconds_are_slept_and_exhaustion_raises() -> None:
    def script(hit: int, handler: BaseHTTPRequestHandler):
        return 503, {"Retry-After": "1", "X-Request-Id": "busy"}, '{"detail":"server busy, try again later"}'

    server, url, thread = _mock(script)
    try:
        started = time.monotonic()
        with pytest.raises(ServiceUnavailableError) as caught:
            Client(url, timeout=5, max_retries=1, backoff=30).decide(
                "x", {"q": {"type": "noul", "instructions": "?"}}, request_id="busy-call"
            )
        elapsed = time.monotonic() - started
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    # One retry sleeps Retry-After (1s), not the 30s backoff.
    assert server.hits == 2
    assert 0.8 <= elapsed < 5
    err = caught.value
    assert str(err) == "server busy, try again later"
    assert err.detail == "server busy, try again later"
    assert err.status == 503
    assert err.retry_after == 1
    assert err.request_id == "busy"


def test_backoff_when_retry_after_is_absent() -> None:
    def script(hit: int, handler: BaseHTTPRequestHandler):
        if hit == 1:
            return 503, {"X-Request-Id": "b"}, '{"detail":"server busy, try again later"}'
        return 200, {"X-Request-Id": "b", "X-Inference-Time-Ms": "0.10"}, json.dumps(NOUL_BODY)

    server, url, thread = _mock(script)
    try:
        Client(url, timeout=5, max_retries=2, backoff=0).decide(
            "x", {"q": {"type": "noul", "instructions": "?"}}
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    assert server.hits == 2


def test_non_503_is_not_retried() -> None:
    def script(hit: int, handler: BaseHTTPRequestHandler):
        return 400, {"X-Request-Id": "once"}, '{"detail":"\'state\' is required"}'

    server, url, thread = _mock(script)
    try:
        with pytest.raises(BadRequestError) as caught:
            Client(url, timeout=5, max_retries=5).decide(None, {})
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    assert server.hits == 1
    assert caught.value.request_id == "once"
    assert caught.value.detail == "'state' is required"


def test_ready_503_is_not_an_error_and_is_not_retried() -> None:
    def script(hit: int, handler: BaseHTTPRequestHandler):
        return 503, {}, '{"ready": false}'

    server, url, thread = _mock(script)
    try:
        ready = Client(url, timeout=5, max_retries=5).ready()
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    assert server.hits == 1
    assert ready.ready is False


def test_status_mapping() -> None:
    cases = [
        (401, AuthenticationError, "invalid or missing bearer token"),
        (413, PayloadTooLargeError, "too many questions"),
        (422, UnprocessableEntityError, "ensemble must be an integer between 1 and 8"),
    ]

    def script(hit: int, handler: BaseHTTPRequestHandler):
        status, message = cases[hit - 1][0], cases[hit - 1][2]
        return status, {}, json.dumps({"detail": message})

    server, url, thread = _mock(script)
    try:
        client = Client(url, timeout=5, max_retries=3)
        for status, cls, message in cases:
            with pytest.raises(cls) as caught:
                client.models()
            assert caught.value.status == status
            assert caught.value.detail == message
            assert str(caught.value) == message
            assert caught.value.request_id is None
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_unexpected_option_and_request_id_type() -> None:
    client = Client("http://127.0.0.1:9", timeout=1)
    with pytest.raises(TypeError, match="unexpected option"):
        client.decide("x", {}, not_a_field=True)
    with pytest.raises(TypeError, match="request_id"):
        client.decide("x", {}, request_id=12)  # type: ignore[arg-type]


def test_health_ready_and_models(base_url: str) -> None:
    client = Client(base_url, timeout=10)
    health = client.health()
    assert health.status == "ok"
    assert health.version == "0.3.0"
    assert client.ready().ready is True
    listed = client.models()
    assert listed.object == "list"
    assert len(listed.data) == 1
    model = listed.data[0]
    assert model.id == "multilingual"
    assert model.object == "model"
    assert model.owned_by == "statim"
    assert model.device == "cpu"
    assert model.weights == "f32"
    assert model.max_len == 1024
    assert model.layers > 0 and model.hidden > 0 and model.vocab > 0
    assert model.source
    keyed = Client(base_url, api_key="unused-because-auth-is-off", timeout=10)
    assert keyed.models().data[0].id == "multilingual"


def test_handler_errors_on_the_real_server(base_url: str) -> None:
    client = Client(base_url, timeout=30)
    with pytest.raises(BadRequestError) as bad:
        client.decide(None, {}, request_id="err-400")
    assert bad.value.status == 400
    assert bad.value.detail == "'state' is required"
    assert bad.value.request_id == "err-400"

    with pytest.raises(BadRequestError) as replaced:
        client.decide(None, {}, request_id="bad id")
    assert replaced.value.request_id is not None
    assert len(replaced.value.request_id) == 16
    assert replaced.value.request_id != "bad id"

    with pytest.raises(UnprocessableEntityError) as invalid:
        client.decide(
            "x",
            {"q": {"type": "noul", "instructions": "?"}},
            ensemble=0,
            request_id="err-422",
        )
    assert invalid.value.detail == "ensemble must be an integer between 1 and 8"
    assert invalid.value.request_id == "err-422"

    with pytest.raises(UnprocessableEntityError) as unknown:
        client.decide("x", {"q": {"type": "maybe", "instructions": "?"}})
    assert unknown.value.detail == "unknown question type; use choice, score or noul"

    too_many = {f"q{i}": {"type": "noul", "instructions": "?"} for i in range(65)}
    with pytest.raises(PayloadTooLargeError) as large:
        client.decide("x", too_many, request_id="err-413")
    assert large.value.status == 413
    assert large.value.detail is not None
    assert large.value.detail.startswith("too many questions")
    assert large.value.request_id == "err-413"


def test_empty_batch(base_url: str) -> None:
    client = Client(base_url, timeout=60)
    batch = client.decide_batch([], {}, request_id="empty-batch")
    assert batch.results == ()
    assert batch.request_id == "empty-batch"
    assert batch.inference_time_ms is not None
    assert batch.inference_time_ms >= 0


def test_decide_and_batch_against_the_server(base_url: str) -> None:
    client = Client(base_url, timeout=180)
    decision = client.decide(
        STATE,
        QUESTIONS,
        model="multilingual",
        return_logits=True,
        request_id="live-decide",
    )
    assert decision.model
    assert decision.request_id == "live-decide"
    assert decision.inference_time_ms is not None and decision.inference_time_ms > 0
    assert decision.usage.output_tokens == 0
    assert decision.usage.input_tokens > 0
    assert decision.routing.model == "multilingual"
    assert decision.routing.reason == "requested"
    assert decision.routing.engine == "statim"
    assert decision.routing.weights == "f32"

    department = decision.answers["department"]
    assert isinstance(department, ChoiceAnswer)
    assert department.type == "choice"
    assert department.choice in department.probabilities
    assert department.choice in {"billing", "technical"}
    assert abs(sum(department.probabilities.values()) - 1) < 0.02
    assert 0 <= department.confidence <= 1
    assert department.logits is not None
    assert len(department.logits) == len(department.probabilities)
    assert 0 <= department.action.act_probability <= 1

    urgency = decision.answers["urgency"]
    assert isinstance(urgency, ScoreAnswer)
    assert isinstance(urgency.score, float)
    assert set(urgency.probabilities) == {"0", "1", "2"}
    assert urgency.legend["0"] == "not urgent"
    assert urgency.logits is not None
    assert len(urgency.logits) == 3
    assert 0 <= urgency.confidence <= 1

    refund = decision.answers["refund"]
    assert isinstance(refund, YesNoAnswer)
    assert 0 <= refund.noul <= 1
    assert refund.probabilities["yes"] == refund.noul
    assert refund.probabilities["no"] == pytest.approx(1 - refund.noul)
    assert refund.yes is (refund.noul >= 0.5)
    assert refund.logits is not None
    assert len(refund.logits) == 2
    assert 0 <= refund.confidence <= 1

    batch = client.decide_batch(
        ["Please refund the duplicate charge today.", "The login page returns a 500 error."],
        {
            "topic": {
                "type": "choice",
                "instructions": "What is this about?",
                "criteria": {"billing": "payments and refunds", "technical": "bugs and outages"},
            }
        },
        model="multilingual",
        request_id="live-batch",
    )
    assert batch.request_id == "live-batch"
    assert len(batch.results) == 2
    labels = []
    for item in batch.results:
        assert item.request_id == "live-batch"
        assert item.routing.reason == "requested"
        topic = item.answers["topic"]
        assert isinstance(topic, ChoiceAnswer)
        assert topic.logits is None
        assert topic.choice in {"billing", "technical"}
        labels.append(topic.choice)
    assert len(labels) == 2


def test_authentication_on_the_keyed_server(auth: tuple[str, str]) -> None:
    url, key = auth
    missing = Client(url, timeout=30)
    with pytest.raises(AuthenticationError) as no_key:
        missing.models()
    assert no_key.value.status == 401
    assert no_key.value.detail == "invalid or missing bearer token"
    assert str(no_key.value) == "invalid or missing bearer token"
    assert no_key.value.request_id is None

    wrong = Client(url, api_key="not-the-key", timeout=30)
    with pytest.raises(AuthenticationError) as denied:
        wrong.decide(
            "hello",
            {"q": {"type": "noul", "instructions": "Is this a greeting?"}},
            request_id="auth-decide",
        )
    assert denied.value.detail == "invalid or missing bearer token"
    # Pre-routing rejection does not echo X-Request-Id.
    assert denied.value.request_id is None

    authed = Client(url, api_key=key, timeout=30)
    listed = authed.models()
    assert listed.data[0].id == "multilingual"
    assert listed.data[0].device == "cpu"
    assert missing.health().status == "ok"
    assert missing.ready().ready is True
