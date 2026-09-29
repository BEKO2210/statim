"""Typed views of the Statim HTTP API.

Wire shapes follow ``docs/openapi.yaml`` ``components.schemas``. A yes/no
question is type ``noul`` on the wire; the client exposes it as
:class:`YesNoAnswer` and fills ``probabilities`` from ``noul`` because the
response body only carries that one probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from statim.errors import StatimError

Answer = "ChoiceAnswer | ScoreAnswer | YesNoAnswer"


@dataclass(frozen=True, slots=True)
class Action:
    """Action-head output. The HTTP API does not name the action."""

    act_probability: float


@dataclass(frozen=True, slots=True)
class Usage:
    """Token accounting. ``output_tokens`` is always 0."""

    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class Routing:
    """Which checkpoint ran.

    ``reason`` is ``requested``, ``lang:en``, ``lang:other``, ``default``,
    ``consensus``, or ``adapter``. ``engine`` is ``statim``. ``adapter`` is
    ``None`` for both an absent key and JSON null; ``adapter_reason`` tells
    them apart with ``None`` or ``"none"``.
    """

    model: str
    reason: str
    engine: str
    weights: str
    adapter: str | None = None
    adapter_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ChoiceAnswer:
    """A ``choice`` question. ``choice`` keeps the JSON type of the winning label."""

    choice: Any
    probabilities: dict[str, float]
    confidence: float
    answer_confidence: float
    action: Action
    logits: tuple[float, ...] | None = None
    logits_by_model: dict[str, tuple[float, ...]] | None = None
    escalate: bool | None = None
    type: str = "choice"


@dataclass(frozen=True, slots=True)
class ScoreAnswer:
    """A ``score`` question. ``score`` is the expected level, not a probability."""

    score: float
    legend: dict[str, Any]
    probabilities: dict[str, float]
    confidence: float
    answer_confidence: float
    action: Action
    logits: tuple[float, ...] | None = None
    logits_by_model: dict[str, tuple[float, ...]] | None = None
    escalate: bool | None = None
    type: str = "score"


@dataclass(frozen=True, slots=True)
class YesNoAnswer:
    """A ``noul`` question: does the statement hold?

    ``noul`` is the probability of the true side, rounded by the server.
    ``probabilities`` is derived here: ``yes`` is ``noul`` and ``no`` is
    ``1 - noul``. ``yes`` is true when ``noul >= 0.5``. ``confidence`` is the
    server value (the larger of the two sides).
    """

    noul: float
    yes: bool
    probabilities: dict[str, float]
    confidence: float
    answer_confidence: float
    action: Action
    logits: tuple[float, ...] | None = None
    logits_by_model: dict[str, tuple[float, ...]] | None = None
    escalate: bool | None = None
    type: str = "noul"


@dataclass(frozen=True, slots=True)
class Decision:
    """One ``POST /v1/systemone`` result.

    ``request_id`` and ``inference_time_ms`` come from response headers, not
    from the JSON body. ``inference_time_ms`` is ``None`` when the server did
    not send ``X-Inference-Time-Ms``.
    """

    model: str
    answers: dict[str, ChoiceAnswer | ScoreAnswer | YesNoAnswer]
    usage: Usage
    routing: Routing
    request_id: str | None = None
    inference_time_ms: float | None = None


@dataclass(frozen=True, slots=True)
class BatchResult:
    """``POST /v1/systemone/batch``. ``results`` follows the request's ``states`` order."""

    results: tuple[Decision, ...]
    request_id: str | None = None
    inference_time_ms: float | None = None


@dataclass(frozen=True, slots=True)
class Health:
    """``GET /health``. ``status`` is ``ok`` when the process is up."""

    status: str
    version: str


@dataclass(frozen=True, slots=True)
class Ready:
    """``GET /ready``. ``ready`` is false when the admission cap is full (HTTP 503)."""

    ready: bool


@dataclass(frozen=True, slots=True)
class Adapter:
    """One LoRA adapter attached to a loaded checkpoint."""

    id: str
    source: str
    mode: str
    rank: int
    alpha: float
    pairs: int
    pairs_applied: int
    categories: tuple[str, ...]
    bytes: int


@dataclass(frozen=True, slots=True)
class Model:
    """One loaded checkpoint from ``GET /v1/models``."""

    id: str
    object: str
    owned_by: str
    source: str
    weights: str
    layers: int
    hidden: int
    max_len: int
    vocab: int
    device: str
    adapters: tuple[Adapter, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelList:
    """``GET /v1/models`` body. ``object`` is ``list``."""

    object: str
    data: tuple[Model, ...]


def _fail(message: str) -> StatimError:
    return StatimError(f"malformed response: {message}")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise _fail(f"{name} must be an object")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(f"{name} must be a number")
    return float(value)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(f"{name} must be an integer")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise _fail(f"{name} must be a string")
    return value


def _field(obj: Mapping[str, Any], key: str) -> Any:
    if key not in obj:
        raise _fail(f"missing {key}")
    return obj[key]


def _optional_nullable_string(obj: Mapping[str, Any], key: str, name: str) -> str | None:
    if key not in obj or obj[key] is None:
        return None
    return _string(obj[key], name)


def _probabilities(value: Any) -> dict[str, float]:
    raw = _mapping(value, "probabilities")
    return {str(key): _number(item, f"probabilities.{key}") for key, item in raw.items()}


def _float_tuple(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise _fail(f"{name} must be an array")
    return tuple(_number(item, name) for item in value)


def _logits(obj: Mapping[str, Any]) -> tuple[float, ...] | None:
    if "logits" not in obj:
        return None
    return _float_tuple(obj["logits"], "logits")


def _logits_by_model(obj: Mapping[str, Any]) -> dict[str, tuple[float, ...]] | None:
    if "logits_by_model" not in obj:
        return None
    raw = _mapping(obj["logits_by_model"], "logits_by_model")
    return {str(key): _float_tuple(item, f"logits_by_model.{key}") for key, item in raw.items()}


def _escalate(obj: Mapping[str, Any]) -> bool | None:
    if "escalate" not in obj:
        return None
    value = obj["escalate"]
    if not isinstance(value, bool):
        raise _fail("escalate must be a boolean")
    return value


def _action(obj: Mapping[str, Any]) -> Action:
    raw = _mapping(_field(obj, "action"), "action")
    return Action(act_probability=_number(_field(raw, "act_probability"), "action.act_probability"))


def _scalar(value: Any, name: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)):
        return value
    raise _fail(f"{name} must be a string, number, boolean, or null")


def parse_answer(value: Any) -> ChoiceAnswer | ScoreAnswer | YesNoAnswer:
    """Parse one answer object into a choice, score, or yes/no result."""
    obj = _mapping(value, "answer")
    kind = _string(_field(obj, "type"), "answer.type")
    shared = {
        "confidence": _number(_field(obj, "confidence"), "confidence"),
        "answer_confidence": _number(_field(obj, "answer_confidence"), "answer_confidence"),
        "action": _action(obj),
        "escalate": _escalate(obj),
        "logits": _logits(obj),
        "logits_by_model": _logits_by_model(obj),
    }
    if kind == "choice":
        return ChoiceAnswer(
            choice=_scalar(_field(obj, "choice"), "choice"),
            probabilities=_probabilities(_field(obj, "probabilities")),
            **shared,
        )
    if kind == "score":
        legend = _mapping(_field(obj, "legend"), "legend")
        return ScoreAnswer(
            score=_number(_field(obj, "score"), "score"),
            legend={str(key): item for key, item in legend.items()},
            probabilities=_probabilities(_field(obj, "probabilities")),
            **shared,
        )
    if kind == "noul":
        # The wire object has a single probability. Derive the pair here so
        # callers can read yes/no the same way they read choice probabilities.
        noul = _number(_field(obj, "noul"), "noul")
        return YesNoAnswer(
            noul=noul,
            yes=noul >= 0.5,
            probabilities={"yes": noul, "no": 1.0 - noul},
            **shared,
        )
    raise _fail(f"unknown answer type {kind!r}")


def parse_decision(value: Any) -> Decision:
    """Parse a decision object. Header fields stay at their defaults."""
    obj = _mapping(value, "decision")
    answers_raw = _mapping(_field(obj, "answers"), "answers")
    answers = {str(key): parse_answer(item) for key, item in answers_raw.items()}
    usage_raw = _mapping(_field(obj, "usage"), "usage")
    routing_raw = _mapping(_field(obj, "routing"), "routing")
    return Decision(
        model=_string(_field(obj, "model"), "model"),
        answers=answers,
        usage=Usage(
            input_tokens=_integer(_field(usage_raw, "input_tokens"), "usage.input_tokens"),
            output_tokens=_integer(_field(usage_raw, "output_tokens"), "usage.output_tokens"),
        ),
        routing=Routing(
            model=_string(_field(routing_raw, "model"), "routing.model"),
            reason=_string(_field(routing_raw, "reason"), "routing.reason"),
            engine=_string(_field(routing_raw, "engine"), "routing.engine"),
            weights=_string(_field(routing_raw, "weights"), "routing.weights"),
            adapter=_optional_nullable_string(routing_raw, "adapter", "routing.adapter"),
            adapter_reason=(
                _string(routing_raw["adapter_reason"], "routing.adapter_reason")
                if "adapter_reason" in routing_raw
                else None
            ),
        ),
    )


def parse_batch(value: Any) -> BatchResult:
    """Parse ``{"results": [...]}`` without header metadata."""
    obj = _mapping(value, "batch")
    results = _field(obj, "results")
    if not isinstance(results, list):
        raise _fail("results must be an array")
    return BatchResult(results=tuple(parse_decision(item) for item in results))


def parse_health(value: Any) -> Health:
    obj = _mapping(value, "health")
    return Health(
        status=_string(_field(obj, "status"), "status"),
        version=_string(_field(obj, "version"), "version"),
    )


def parse_ready(value: Any) -> Ready:
    obj = _mapping(value, "ready")
    ready = _field(obj, "ready")
    if not isinstance(ready, bool):
        raise _fail("ready must be a boolean")
    return Ready(ready=ready)


def parse_adapter(value: Any) -> Adapter:
    obj = _mapping(value, "adapter")
    categories = _field(obj, "categories")
    if not isinstance(categories, list):
        raise _fail("categories must be an array")
    return Adapter(
        id=_string(_field(obj, "id"), "adapter.id"),
        source=_string(_field(obj, "source"), "adapter.source"),
        mode=_string(_field(obj, "mode"), "adapter.mode"),
        rank=_integer(_field(obj, "rank"), "adapter.rank"),
        alpha=_number(_field(obj, "alpha"), "adapter.alpha"),
        pairs=_integer(_field(obj, "pairs"), "adapter.pairs"),
        pairs_applied=_integer(_field(obj, "pairs_applied"), "adapter.pairs_applied"),
        categories=tuple(_string(item, "adapter.categories") for item in categories),
        bytes=_integer(_field(obj, "bytes"), "adapter.bytes"),
    )


def parse_model(value: Any) -> Model:
    obj = _mapping(value, "model")
    adapters = obj.get("adapters", [])
    if not isinstance(adapters, list):
        raise _fail("adapters must be an array")
    return Model(
        id=_string(_field(obj, "id"), "id"),
        object=_string(_field(obj, "object"), "object"),
        owned_by=_string(_field(obj, "owned_by"), "owned_by"),
        source=_string(_field(obj, "source"), "source"),
        weights=_string(_field(obj, "weights"), "weights"),
        layers=_integer(_field(obj, "layers"), "layers"),
        hidden=_integer(_field(obj, "hidden"), "hidden"),
        max_len=_integer(_field(obj, "max_len"), "max_len"),
        vocab=_integer(_field(obj, "vocab"), "vocab"),
        device=_string(_field(obj, "device"), "device"),
        adapters=tuple(parse_adapter(item) for item in adapters),
    )


def parse_model_list(value: Any) -> ModelList:
    obj = _mapping(value, "models")
    data = _field(obj, "data")
    if not isinstance(data, list):
        raise _fail("data must be an array")
    return ModelList(
        object=_string(_field(obj, "object"), "object"),
        data=tuple(parse_model(item) for item in data),
    )
