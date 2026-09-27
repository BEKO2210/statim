"""HTTP client for ``POST /v1/systemone`` and the operational endpoints.

The runtime dependency is the Python standard library. ``503`` responses from
``decide`` and ``decide_batch`` are retried with backoff. Every other status
fails immediately. ``Retry-After`` (seconds) wins over the computed backoff;
admission saturation sends ``Retry-After: 1``.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import replace
from typing import Any, Mapping

from statim._version import __version__
from statim.errors import (
    AuthenticationError,
    BadRequestError,
    PayloadTooLargeError,
    ServiceUnavailableError,
    StatimError,
    TransportError,
    UnprocessableEntityError,
)
from statim.types import (
    BatchResult,
    Decision,
    Health,
    ModelList,
    Ready,
    parse_batch,
    parse_decision,
    parse_health,
    parse_model_list,
    parse_ready,
)

_DECISION_OPTIONS = frozenset(
    {
        "model",
        "lang",
        "ensemble",
        "ensemble_margin",
        "calibrate",
        "return_logits",
        "max_len",
        "head_max_len",
        "request_id",
    }
)
_BACKOFF_CAP_SECONDS = 30.0
_STATUS_ERRORS: dict[int, type[StatimError]] = {
    400: BadRequestError,
    401: AuthenticationError,
    413: PayloadTooLargeError,
    422: UnprocessableEntityError,
    503: ServiceUnavailableError,
}


class _RawResponse:
    __slots__ = ("status", "headers", "body", "payload")

    def __init__(self, status: int, headers: Any, body: bytes, payload: Any) -> None:
        self.status = status
        self.headers = headers
        self.body = body
        self.payload = payload


def _header(headers: Any, name: str) -> str | None:
    if headers is None:
        return None
    value = headers.get(name)
    if value is None or value == "":
        return None
    return str(value)


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a delta-seconds ``Retry-After``. HTTP-date values are ignored."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        return None
    if seconds < 0:
        return None
    return seconds


def _request_id_from(headers: Any) -> str | None:
    return _header(headers, "X-Request-Id")


def _inference_ms(headers: Any) -> float | None:
    raw = _header(headers, "X-Inference-Time-Ms")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _detail(payload: Any) -> str | None:
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str):
            return detail
    return None


def _error_from(response: _RawResponse) -> StatimError:
    detail = _detail(response.payload)
    if detail is not None:
        message = detail
    else:
        text = response.body.decode("utf-8", "replace").strip()
        message = text[:500] if text else f"HTTP {response.status}"
    request_id = _request_id_from(response.headers)
    cls = _STATUS_ERRORS.get(response.status, StatimError)
    if cls is ServiceUnavailableError:
        return ServiceUnavailableError(
            message,
            status=response.status,
            detail=detail,
            request_id=request_id,
            retry_after=_parse_retry_after(_header(response.headers, "Retry-After")),
        )
    return cls(message, status=response.status, detail=detail, request_id=request_id)


def _new_request_id() -> str:
    # uuid4 is 36 chars of hex and hyphens, inside the server's 1–128 allowlist.
    return str(uuid.uuid4())


def _check_options(options: Mapping[str, Any]) -> None:
    unknown = sorted(set(options) - _DECISION_OPTIONS)
    if unknown:
        raise TypeError(f"unexpected option(s): {', '.join(unknown)}")


def _check_request_id(request_id: Any) -> None:
    if request_id is not None and not isinstance(request_id, str):
        raise TypeError("request_id must be a string")


def _body_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Drop ``request_id`` and ``None``. ``False`` and ``0`` are sent."""
    return {key: value for key, value in options.items() if key != "request_id" and value is not None}


class Client:
    """Statim HTTP client.

    :param base_url: Absolute ``http`` or ``https`` origin, for example
        ``http://127.0.0.1:8080``. A trailing slash is removed.
    :param api_key: Bearer token. ``None`` or ``""`` sends no
        ``Authorization`` header. Required for decisions, ``/v1/models``, and
        ``/metrics`` when the server was started with keys. ``/health`` and
        ``/ready`` stay open.
    :param timeout: Socket timeout in seconds for every request. The server's
        inference deadline defaults to 120 seconds; the first decision after
        process start is slower because weights are faulted in from disk.
    :param max_retries: Extra attempts after a ``503`` from ``decide`` or
        ``decide_batch``. ``0`` disables retries. Other statuses are never
        retried, and ``ready()`` is never retried.
    :param backoff: Base delay in seconds when a ``503`` has no ``Retry-After``.
        Attempt ``n`` (starting at 0) sleeps ``min(backoff * 2**n, 30)``.
        A numeric ``Retry-After`` replaces that delay.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 120,
        *,
        max_retries: int = 2,
        backoff: float = 0.5,
    ) -> None:
        if not isinstance(base_url, str) or not base_url:
            raise ValueError("base_url must be an absolute http or https URL")
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute http or https URL")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or None
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be a positive number of seconds")
        self.timeout = float(timeout)
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be an integer >= 0")
        self.max_retries = max_retries
        if isinstance(backoff, bool) or not isinstance(backoff, (int, float)) or backoff < 0:
            raise ValueError("backoff must be a number >= 0")
        self.backoff = float(backoff)

    def decide(self, state: Any, questions: Mapping[str, Any], **options: Any) -> Decision:
        """Score one state.

        ``questions`` maps a question id to an object with ``type``
        (``choice``, ``score``, or ``noul``) and ``instructions``. Choice and
        score questions also need ``criteria``. Optional keyword arguments are
        ``model``, ``lang``, ``ensemble``, ``ensemble_margin``, ``calibrate``,
        ``return_logits``, ``max_len``, ``head_max_len``, and ``request_id``.
        ``None`` omits an optional field. An unknown keyword raises ``TypeError``.

        ``request_id`` is sent as ``X-Request-Id``. When omitted, the client
        generates a UUID. The value on the returned :class:`Decision` is the
        id the server echoed, which differs when the supplied id is not 1–128
        ASCII letters, digits, ``.``, ``_``, or ``-``.
        """
        _check_options(options)
        _check_request_id(options.get("request_id", None))
        if not isinstance(questions, Mapping):
            raise TypeError("questions must be a mapping")
        body: dict[str, Any] = {"state": state, "questions": dict(questions)}
        body.update(_body_options(options))
        request_id = options.get("request_id")
        response = self._request(
            "POST",
            "/v1/systemone",
            body,
            request_id=request_id if isinstance(request_id, str) else None,
            retry_on_503=True,
        )
        if response.status != 200:
            raise _error_from(response)
        decision = parse_decision(response.payload)
        return replace(
            decision,
            request_id=_request_id_from(response.headers),
            inference_time_ms=_inference_ms(response.headers),
        )

    def decide_batch(
        self,
        states: list[Any] | tuple[Any, ...],
        questions: Mapping[str, Any],
        **options: Any,
    ) -> BatchResult:
        """Score many states with one question set.

        ``states`` keeps its order in :attr:`BatchResult.results`. The same
        keyword options as :meth:`decide` are accepted. A limit failure on
        any state fails the whole call.
        """
        _check_options(options)
        _check_request_id(options.get("request_id", None))
        if isinstance(states, (str, bytes)) or not isinstance(states, (list, tuple)):
            raise TypeError("states must be a list")
        if not isinstance(questions, Mapping):
            raise TypeError("questions must be a mapping")
        body: dict[str, Any] = {"states": list(states), "questions": dict(questions)}
        body.update(_body_options(options))
        request_id = options.get("request_id")
        response = self._request(
            "POST",
            "/v1/systemone/batch",
            body,
            request_id=request_id if isinstance(request_id, str) else None,
            retry_on_503=True,
        )
        if response.status != 200:
            raise _error_from(response)
        parsed = parse_batch(response.payload)
        echoed = _request_id_from(response.headers)
        elapsed = _inference_ms(response.headers)
        results = tuple(replace(item, request_id=echoed, inference_time_ms=elapsed) for item in parsed.results)
        return replace(parsed, results=results, request_id=echoed, inference_time_ms=elapsed)

    def models(self) -> ModelList:
        """List checkpoints in load order. Requires the bearer key when auth is on."""
        response = self._request("GET", "/v1/models", None, retry_on_503=False)
        if response.status != 200:
            raise _error_from(response)
        return parse_model_list(response.payload)

    def health(self) -> Health:
        """Liveness. Always unauthenticated. Returns ``status`` and ``version``."""
        response = self._request("GET", "/health", None, retry_on_503=False)
        if response.status != 200:
            raise _error_from(response)
        return parse_health(response.payload)

    def ready(self) -> Ready:
        """Readiness. HTTP 503 ``{"ready": false}`` returns :class:`Ready` and is not retried."""
        response = self._request("GET", "/ready", None, retry_on_503=False)
        if response.status in (200, 503) and isinstance(response.payload, dict) and "ready" in response.payload:
            return parse_ready(response.payload)
        raise _error_from(response)

    def _request(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | None,
        *,
        request_id: str | None = None,
        retry_on_503: bool,
    ) -> _RawResponse:
        url = self.base_url + path
        data: bytes | None = None
        headers = {
            "Accept": "application/json",
            "User-Agent": f"statim-python/{__version__}",
            "X-Request-Id": request_id if request_id is not None else _new_request_id(),
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if body is not None:
            data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        attempt = 0
        while True:
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                    response = _RawResponse(resp.status, resp.headers, raw, _decode_json(raw))
            except urllib.error.HTTPError as exc:
                raw = exc.read()
                response = _RawResponse(exc.code, exc.headers, raw, _decode_json(raw))
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
                raise TransportError(f"request failed: {reason}") from exc

            if (
                retry_on_503
                and response.status == 503
                and attempt < self.max_retries
            ):
                retry_after = _parse_retry_after(_header(response.headers, "Retry-After"))
                delay = retry_after if retry_after is not None else min(self.backoff * (2**attempt), _BACKOFF_CAP_SECONDS)
                time.sleep(delay)
                attempt += 1
                continue
            if response.status < 400 and not isinstance(response.payload, (dict, list)):
                raise StatimError(
                    "response body is not valid JSON",
                    status=response.status,
                    request_id=_request_id_from(response.headers),
                )
            return response


def _decode_json(raw: bytes) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
