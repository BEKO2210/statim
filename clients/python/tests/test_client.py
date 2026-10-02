"""Client tests.

Integration calls use the unauthenticated server at STATIM_URL (default
http://127.0.0.1:8190, multilingual CPU model) and a keyed server at
STATIM_AUTH_URL (default http://127.0.0.1:8191, key sdk-test-key-0123456789abcdef0123).

Retry sequencing uses a local HTTP server so admission control does not have
to be saturated.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from statim import (
    Adapter,
    AuthenticationError,
    BadRequestError,
    ChoiceAnswer,
    Client,
    PayloadTooLargeError,
    ScoreAnswer,
    ServiceUnavailableError,
    StatimError,
    TransportError,
    UnprocessableEntityError,
    YesNoAnswer,
)

BASE_URL = os.environ.get("STATIM_URL", "http://127.0.0.1:8190")
AUTH_URL = os.environ.get("STATIM_AUTH_URL", "http://127.0.0.1:8191")
AUTH_KEY = os.environ.get("STATIM_API_KEY_TEST", "sdk-test-key-0123456789abcdef0123")
ADAPTER_URL = os.environ.get("STATIM_ADAPTER_URL")
ADAPTER_NAME = os.environ.get("STATIM_ADAPTER_NAME")
AUTO_FAMILIES = {
    "sentiment",
    "emotion",
    "complaint",
    "nli",
    "safety",
    "reading",
    "similarity",
    "topic",
    "intent",
    "stance",
    "formality",
    "urgency",
    "fact_check",
    "pii",
}

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
    assert re.fullmatch(r"\d+\.\d+\.\d+", health.version)
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
        client = Client(url, api_key="sdk-unit-key-0123456789abcdef0123", timeout=5, max_retries=0)
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
    assert seen["headers"]["authorization"] == "Bearer sdk-unit-key-0123456789abcdef0123"
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


def test_adapter_transport_and_response_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[tuple[str, dict[str, Any]]] = []

    class Response:
        status = 200
        headers: dict[str, str] = {}

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
        routing = dict(NOUL_BODY["routing"])
        routing.update(adapter=None, adapter_reason="none")
        if body.get("adapter") == "emotion":
            routing.update(reason="adapter", adapter="emotion", adapter_reason="requested")
        elif body.get("adapter") == "auto":
            category = next(iter(body["questions"]))
            if category == "emotion":
                routing.update(adapter="emotion", adapter_reason="auto:emotion")
            elif category == "no_family":
                routing.update(adapter_reason="auto:no-family")
            else:
                routing.update(adapter_reason="auto:emotion:no-adapter")
        payload = {**NOUL_BODY, "routing": routing}
        return Response({"results": [payload]} if request.full_url.endswith("/batch") else payload)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    client = Client("http://statim.invalid", max_retries=0)
    questions = {"emotion": {"type": "noul", "instructions": "Emotion?"}}
    selected = client.decide("hello", questions, adapter="emotion")
    batch = client.decide_batch(["hello"], questions, adapter="auto")
    omitted = client.decide("hello", questions, adapter=None)
    no_family = client.decide(
        "hello", {"no_family": {"type": "noul", "instructions": "Unknown?"}}, adapter="auto"
    )
    no_adapter = client.decide(
        "hello", {"emotion_missing": {"type": "noul", "instructions": "Emotion?"}}, adapter="auto"
    )
    base = client.decide("hello", questions, adapter="none")
    long_name = "é" * 129
    long_adapter = client.decide("hello", questions, adapter=long_name)

    assert requests == [
        ("http://statim.invalid/v1/systemone", {"state": "hello", "questions": questions, "adapter": "emotion"}),
        (
            "http://statim.invalid/v1/systemone/batch",
            {"states": ["hello"], "questions": questions, "adapter": "auto"},
        ),
        ("http://statim.invalid/v1/systemone", {"state": "hello", "questions": questions}),
        (
            "http://statim.invalid/v1/systemone",
            {
                "state": "hello",
                "questions": {"no_family": {"type": "noul", "instructions": "Unknown?"}},
                "adapter": "auto",
            },
        ),
        (
            "http://statim.invalid/v1/systemone",
            {
                "state": "hello",
                "questions": {"emotion_missing": {"type": "noul", "instructions": "Emotion?"}},
                "adapter": "auto",
            },
        ),
        ("http://statim.invalid/v1/systemone", {"state": "hello", "questions": questions, "adapter": "none"}),
        (
            "http://statim.invalid/v1/systemone",
            {"state": "hello", "questions": questions, "adapter": long_name},
        ),
    ]
    assert selected.routing.adapter == "emotion"
    assert selected.routing.adapter_reason == "requested"
    assert selected.routing.reason == "adapter"
    assert batch.results[0].routing.adapter == "emotion"
    assert batch.results[0].routing.adapter_reason == "auto:emotion"
    assert omitted.routing.adapter is None
    assert omitted.routing.adapter_reason == "none"
    assert no_family.routing.adapter is None
    assert no_family.routing.adapter_reason == "auto:no-family"
    assert no_adapter.routing.adapter is None
    assert no_adapter.routing.adapter_reason == "auto:emotion:no-adapter"
    assert base.routing.adapter is None
    assert base.routing.adapter_reason == "none"
    assert long_adapter.routing.adapter is None

    with pytest.raises(TypeError, match="adapter"):
        client.decide("hello", questions, adapter=12)


def test_models_parse_adapters_and_reject_malformed(monkeypatch: pytest.MonkeyPatch) -> None:
    model = {
        "id": "multilingual",
        "object": "model",
        "owned_by": "statim",
        "source": "laya-multilingual",
        "weights": "q8_0",
        "layers": 22,
        "hidden": 768,
        "max_len": 1024,
        "vocab": 250002,
        "device": "cpu",
    }
    adapter = {
        "id": "emotion",
        "source": "emotion-lora",
        "mode": "runtime",
        "rank": 4,
        "alpha": 8.0,
        "pairs": 88,
        "pairs_applied": 87,
        "categories": ["emotion", "sentiment"],
        "bytes": 3456,
    }
    payloads = [
        {"object": "list", "data": [model, {**model, "id": "with-adapter", "adapters": [adapter]}]},
        {"object": "list", "data": [{**model, "adapters": None}]},
        {"object": "list", "data": [{**model, "adapters": {"bad": True}}]},
        {"object": "list", "data": [{**model, "adapters": [{key: value for key, value in adapter.items() if key != "id"}]}]},
        {"object": "list", "data": [{**model, "adapters": [{**adapter, "rank": "4"}]}]},
    ]

    class Response:
        status = 200
        headers: dict[str, str] = {}

        def __enter__(self):
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(payloads.pop(0)).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: Response())
    client = Client("http://statim.invalid", max_retries=0)
    listed = client.models()
    assert listed.data[0].adapters == ()
    parsed = listed.data[1].adapters[0]
    assert isinstance(parsed, Adapter)
    assert parsed.id == "emotion"
    assert parsed.alpha == 8.0
    assert parsed.categories == ("emotion", "sentiment")
    assert parsed.bytes == 3456
    for message in ("adapters must be an array", "adapters must be an array", "missing id", "adapter.rank"):
        with pytest.raises(StatimError, match=message):
            client.models()


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
    assert re.fullmatch(r"\d+\.\d+\.\d+", health.version)
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

    wrong = Client(url, api_key="wrong-sdk-key-0123456789abcdef01", timeout=30)
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


@pytest.mark.skipif(not ADAPTER_URL or not ADAPTER_NAME, reason="adapter server is not configured")
def test_adapter_against_the_server() -> None:
    assert ADAPTER_URL is not None and ADAPTER_NAME is not None
    client = Client(ADAPTER_URL, timeout=180)
    listed = client.models()
    matches = [(model, adapter) for model in listed.data for adapter in model.adapters if adapter.id == ADAPTER_NAME]
    assert matches, f"adapter {ADAPTER_NAME!r} is not listed"
    model, adapter = matches[0]
    assert adapter.categories, f"adapter {ADAPTER_NAME!r} has no categories"
    category = adapter.categories[0]
    questions = {category: {"type": "noul", "instructions": f"Is this about {category}?"}}

    requested = client.decide("adapter live test", questions, model=model.id, adapter=ADAPTER_NAME)
    assert requested.routing.adapter == ADAPTER_NAME
    assert requested.routing.adapter_reason == "requested"

    base = client.decide("adapter live test", questions, model=model.id, adapter="none")
    assert base.routing.adapter is None
    assert base.routing.adapter_reason == "none"

    if category not in AUTO_FAMILIES:
        pytest.skip(f"adapter category {category!r} is not one of the 14 auto-rule families")
    automatic = client.decide("adapter live test", questions, model=model.id, adapter="auto")
    assert automatic.routing.adapter == ADAPTER_NAME
    assert automatic.routing.adapter_reason == f"auto:{category}"
