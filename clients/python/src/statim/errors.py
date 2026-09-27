"""Errors raised by the Statim client.

HTTP 400, 401, 413, 422, and 503 map to their own types. The exception
message is the server's ``detail`` string when the body is
``{"detail": "..."}``. Any other status is :class:`StatimError`.
"""

from __future__ import annotations


class StatimError(Exception):
    """Base class for Statim client failures.

    :param message: Text returned by ``str(exc)``. For HTTP errors this is the
        server ``detail`` when that field is a string.
    :param status: HTTP status, or ``None`` when no response arrived.
    :param detail: The ``detail`` field, or ``None`` when the body had none.
    :param request_id: ``X-Request-Id`` echoed by the server. Pre-routing
        responses (authentication, framing) do not send one.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        detail: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail
        self.request_id = request_id


class TransportError(StatimError):
    """The host could not be reached, or the connection timed out."""


class BadRequestError(StatimError):
    """HTTP 400. The JSON body is malformed or a required field is missing."""


class AuthenticationError(StatimError):
    """HTTP 401. The bearer token is missing or does not match a configured key."""


class PayloadTooLargeError(StatimError):
    """HTTP 413. The body or a request limit (questions, options, tokens) was exceeded."""


class UnprocessableEntityError(StatimError):
    """HTTP 422. A question, budget, or routing field was rejected."""


class ServiceUnavailableError(StatimError):
    """HTTP 503. Admission control or the engine queue is saturated.

    The client retries these on ``decide`` and ``decide_batch`` before raising.
    ``retry_after`` is the ``Retry-After`` header in seconds when the server
    sent one (admission saturation sends ``1``). The queue-deadline 503 and
    ``GET /ready`` do not send the header. ``ready()`` does not raise this
    for the ``{"ready": false}`` body.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = 503,
        detail: str | None = None,
        request_id: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, status=status, detail=detail, request_id=request_id)
        self.retry_after = retry_after
