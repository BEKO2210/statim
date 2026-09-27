#!/usr/bin/env python3
"""Route real Banking77 support tickets with a Statim server.

``demo`` asks intent, urgency, and refund in one request for five tickets.
``eval`` measures intent accuracy, the selective-prediction curve, and
throughput. Ticket text is only the Banking77 test split (PolyAI/banking77,
CC-BY-4.0); nothing in this file invents a customer message.

Python 3.10+. Uses the standard library and the in-repo ``statim`` client
(``pip install ./clients/python``, or the checkout on ``sys.path``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import textwrap
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache" / "banking77-test.parquet"
RESULTS_JSON = HERE / "results.json"
RESULTS_MD = HERE / "results.md"

# Official test split, 3,080 rows. ``main`` is the parquet conversion once it is
# merged; the open pull-request ref is the same file on PolyAI/banking77 today.
PARQUET_URLS = (
    "https://huggingface.co/datasets/PolyAI/banking77/resolve/main/data/test-00000-of-00001.parquet",
    "https://huggingface.co/datasets/PolyAI/banking77/resolve/refs%2Fpr%2F7/data/test-00000-of-00001.parquet",
)

# 77 intent names do not fit a 256-token option budget: the server then keeps a
# few subwords of each option. This checkpoint family stores 512; send it
# explicitly so a server started from an older default still sees every intent.
HEAD_MAX_LEN = 512
INTENT_INSTRUCTIONS = "Which banking intent does `message` express?"
URGENCY_LEVELS = ("can wait", "today", "immediately")
REFUND_INSTRUCTIONS = "Does the customer ask for money back?"
CURVE_THRESHOLDS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)
DEMO_TICKETS = 5


def load_client():
    try:
        from statim import ChoiceAnswer, Client, ScoreAnswer, YesNoAnswer
    except ImportError:
        src = None
        for parent in HERE.parents:
            candidate = parent / "clients" / "python" / "src"
            if (candidate / "statim" / "__init__.py").is_file():
                src = candidate
                break
        if src is None:
            raise SystemExit(
                "statim client not found. From the repository root run:\n"
                "  pip install ./clients/python"
            )
        sys.path.insert(0, str(src))
        from statim import ChoiceAnswer, Client, ScoreAnswer, YesNoAnswer
    return Client, ChoiceAnswer, ScoreAnswer, YesNoAnswer


Client, ChoiceAnswer, ScoreAnswer, YesNoAnswer = load_client()


@dataclass(frozen=True, slots=True)
class Ticket:
    text: str
    label: str
    phrase: str


def readable(name: str) -> str:
    """Dataset intent id with underscores written as spaces."""
    return name.replace("_", " ")


def intent_question(phrases: list[str]) -> dict[str, Any]:
    return {
        "type": "choice",
        "instructions": INTENT_INSTRUCTIONS,
        "criteria": {phrase: None for phrase in phrases},
    }


def demo_questions(phrases: list[str]) -> dict[str, Any]:
    return {
        "intent": intent_question(phrases),
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this request?",
            "criteria": list(URGENCY_LEVELS),
        },
        "refund": {"type": "noul", "instructions": REFUND_INSTRUCTIONS},
    }


def eval_questions(phrases: list[str]) -> dict[str, Any]:
    return {"intent": intent_question(phrases)}


# --- Banking77 parquet (standard library) -----------------------------------

class _Compact:
    """Thrift compact protocol, enough for a Parquet footer and page headers."""

    _TRUE, _FALSE, _BYTE, _I16, _I32, _I64, _DOUBLE, _BINARY, _LIST, _SET, _MAP, _STRUCT = range(1, 13)

    def __init__(self, buf: bytes, offset: int = 0) -> None:
        self.buf = buf
        self.i = offset
        self._last = [0]

    def _u8(self) -> int:
        value = self.buf[self.i]
        self.i += 1
        return value

    def _varint(self) -> int:
        number = shift = 0
        while True:
            byte = self._u8()
            number |= (byte & 0x7F) << shift
            if byte < 128:
                return number
            shift += 7

    def _zigzag(self) -> int:
        number = self._varint()
        return (number >> 1) ^ -(number & 1)

    def _raw(self, size: int) -> bytes:
        chunk = self.buf[self.i : self.i + size]
        if len(chunk) != size:
            raise ValueError("truncated parquet thrift value")
        self.i += size
        return chunk

    def read_struct(self) -> dict[int, Any]:
        fields: dict[int, Any] = {}
        self._last.append(0)
        while True:
            header = self._u8()
            if header == 0:
                self._last.pop()
                return fields
            kind = header & 0x0F
            delta = header >> 4
            field_id = self._zigzag() if delta == 0 else self._last[-1] + delta
            self._last[-1] = field_id
            fields[field_id] = self._value(kind)

    def _value(self, kind: int) -> Any:
        if kind == self._TRUE:
            return True
        if kind == self._FALSE:
            return False
        if kind == self._BYTE:
            return self._u8()
        if kind in (self._I16, self._I32, self._I64):
            return self._zigzag()
        if kind == self._DOUBLE:
            import struct

            return struct.unpack("<d", self._raw(8))[0]
        if kind == self._BINARY:
            return self._raw(self._varint())
        if kind in (self._LIST, self._SET):
            header = self._u8()
            element = header & 0x0F
            size = header >> 4
            if size == 15:
                size = self._varint()
            return [self._value(element) for _ in range(size)]
        if kind == self._MAP:
            header = self._u8()
            size = header >> 4
            if size == 15:
                size = self._varint()
            types = self._u8()
            key_type, value_type = types >> 4, types & 0x0F
            return [(self._value(key_type), self._value(value_type)) for _ in range(size)]
        if kind == self._STRUCT:
            return self.read_struct()
        raise ValueError(f"unsupported thrift type {kind}")


def _snappy_decompress(src: bytes) -> bytes:
    index = 0

    def varint() -> int:
        nonlocal index
        number = shift = 0
        while True:
            byte = src[index]
            index += 1
            number |= (byte & 0x7F) << shift
            if byte < 128:
                return number
            shift += 7

    expected = varint()
    out = bytearray()
    while index < len(src) and len(out) < expected:
        tag = src[index]
        index += 1
        kind = tag & 3
        if kind == 0:
            literal = tag >> 2
            if literal < 60:
                length = literal + 1
            elif literal == 60:
                length = src[index] + 1
                index += 1
            elif literal == 61:
                length = int.from_bytes(src[index : index + 2], "little") + 1
                index += 2
            elif literal == 62:
                length = int.from_bytes(src[index : index + 3], "little") + 1
                index += 3
            else:
                length = int.from_bytes(src[index : index + 4], "little") + 1
                index += 4
            out += src[index : index + length]
            index += length
        else:
            if kind == 1:
                length = ((tag >> 2) & 7) + 4
                offset = ((tag & 0xE0) << 3) | src[index]
                index += 1
            elif kind == 2:
                length = (tag >> 2) + 1
                offset = int.from_bytes(src[index : index + 2], "little")
                index += 2
            else:
                length = (tag >> 2) + 1
                offset = int.from_bytes(src[index : index + 4], "little")
                index += 4
            if offset <= 0 or offset > len(out):
                raise ValueError("invalid snappy copy")
            start = len(out) - offset
            for step in range(length):
                out.append(out[start + step])
    if len(out) != expected:
        raise ValueError(f"snappy length {len(out)} != {expected}")
    return bytes(out)


def _decompress(codec: int, payload: bytes) -> bytes:
    if codec == 0:
        return payload
    if codec == 1:
        return _snappy_decompress(payload)
    if codec == 2:
        return zlib.decompress(payload, zlib.MAX_WBITS | 16)
    raise ValueError(f"unsupported parquet compression codec {codec}")


def _hybrid(buf: bytes, bit_width: int, count: int, *, prefixed: bool) -> tuple[list[int], int]:
    """RLE/bit-packing hybrid. Returns values and the number of bytes consumed."""
    if count == 0:
        return [], 0
    if bit_width == 0:
        return [0] * count, 4 if prefixed else 0
    if prefixed:
        block = int.from_bytes(buf[:4], "little")
        data = buf[4 : 4 + block]
        consumed = 4 + block
    else:
        data = buf
        consumed = 0
    values: list[int] = []
    cursor = 0
    mask = (1 << bit_width) - 1
    while len(values) < count:
        header = shift = 0
        while True:
            byte = data[cursor]
            cursor += 1
            header |= (byte & 0x7F) << shift
            if byte < 128:
                break
            shift += 7
        if header & 1 == 0:
            run = header >> 1
            width_bytes = (bit_width + 7) // 8
            raw = data[cursor : cursor + width_bytes]
            cursor += width_bytes
            value = int.from_bytes(raw, "little") & mask
            values.extend([value] * run)
        else:
            groups = header >> 1
            group_count = groups * 8
            nbytes = (group_count * bit_width + 7) // 8
            raw = data[cursor : cursor + nbytes]
            cursor += nbytes
            for item in range(group_count):
                bit = item * bit_width
                acc = 0
                for offset in range(bit_width):
                    pos = bit + offset
                    acc |= ((raw[pos // 8] >> (pos % 8)) & 1) << offset
                values.append(acc)
    if not prefixed:
        consumed = cursor
    return values[:count], consumed


def _plain_values(raw: bytes, physical: int, count: int) -> list[Any]:
    if physical == 6:
        values: list[Any] = []
        cursor = 0
        for _ in range(count):
            length = int.from_bytes(raw[cursor : cursor + 4], "little")
            cursor += 4
            values.append(raw[cursor : cursor + length].decode("utf-8"))
            cursor += length
        return values
    if physical == 2:
        import struct

        return list(struct.unpack("<" + "q" * count, raw[: 8 * count]))
    if physical == 1:
        import struct

        return list(struct.unpack("<" + "i" * count, raw[: 4 * count]))
    raise ValueError(f"unsupported parquet physical type {physical}")


def _column_levels(schema: list[dict[int, Any]], name: str) -> tuple[int, int]:
    max_def = max_rep = 0
    for element in schema[1:]:
        if element[4].decode() != name:
            continue
        repetition = element.get(3, 0)
        if repetition != 0:
            max_def += 1
        if repetition == 2:
            max_rep += 1
        return max_def, max_rep
    raise ValueError(f"parquet column {name} is missing")


def _read_column(blob: bytes, schema: list[dict[int, Any]], meta: dict[int, Any]) -> list[Any]:
    name = meta[3][0].decode()
    physical = meta[1]
    codec = meta[4]
    expected = meta[5]
    compressed_size = meta[7]
    start = meta.get(11, meta[9])
    max_def, max_rep = _column_levels(schema, name)
    pos = start
    end = start + compressed_size
    dictionary: list[Any] | None = None
    values: list[Any] = []
    while pos < end and len(values) < expected:
        reader = _Compact(blob, pos)
        header = reader.read_struct()
        pos = reader.i
        page_type = header[1]
        uncompressed = header[2]
        compressed = header[3]
        page = _decompress(codec, blob[pos : pos + compressed])
        pos += compressed
        if len(page) != uncompressed:
            raise ValueError(f"{name}: page decompressed to {len(page)}, expected {uncompressed}")
        if page_type == 2:
            dictionary = _plain_values(page, physical, header[7][1])
            continue
        if page_type != 0:
            raise ValueError(f"{name}: unsupported parquet page type {page_type}")
        page_header = header[5]
        count = page_header[1]
        encoding = page_header[2]
        cursor = 0
        if max_rep:
            _, used = _hybrid(page[cursor:], max_rep.bit_length(), count, prefixed=True)
            cursor += used
        if max_def:
            levels, used = _hybrid(page[cursor:], max_def.bit_length(), count, prefixed=True)
            cursor += used
        else:
            levels = [0] * count
        body = page[cursor:]
        present = [level == max_def for level in levels] if max_def else [True] * count
        defined = sum(present)
        if encoding in (2, 8):
            if dictionary is None:
                raise ValueError(f"{name}: dictionary page missing")
            bit_width = body[0]
            indices = [0] * defined if bit_width == 0 else _hybrid(body[1:], bit_width, defined, prefixed=False)[0]
            stream = iter(indices)
            for flag in present:
                values.append(dictionary[next(stream)] if flag else None)
        elif encoding == 0:
            plain = _plain_values(body, physical, defined)
            stream = iter(plain)
            for flag in present:
                values.append(next(stream) if flag else None)
        else:
            raise ValueError(f"{name}: unsupported encoding {encoding}")
    if len(values) != expected:
        raise ValueError(f"{name}: decoded {len(values)} values, footer says {expected}")
    return values


def read_parquet(path: Path) -> tuple[list[str], list[int], list[str]]:
    blob = path.read_bytes()
    if blob[:4] != b"PAR1" or blob[-4:] != b"PAR1":
        raise ValueError(f"{path} is not a parquet file")
    footer_len = int.from_bytes(blob[-8:-4], "little")
    meta = _Compact(blob[-(8 + footer_len) : -8]).read_struct()
    schema = meta[2]
    texts: list[str] = []
    labels: list[int] = []
    for row_group in meta[4]:
        for column in row_group[1]:
            column_meta = column[3]
            name = column_meta[3][0].decode()
            decoded = _read_column(blob, schema, column_meta)
            if name == "text":
                texts.extend(str(item) for item in decoded)
            elif name == "label":
                labels.extend(int(item) for item in decoded)
    names: list[str] | None = None
    for item in meta.get(5, []):
        if item.get(1) == b"huggingface":
            info = json.loads(item[2].decode())
            names = list(info["info"]["features"]["label"]["names"])
            break
    if names is None:
        raise ValueError("parquet file has no Hugging Face label names")
    if len(texts) != len(labels):
        raise ValueError(f"text/label length mismatch ({len(texts)} vs {len(labels)})")
    if any(index < 0 or index >= len(names) for index in labels):
        raise ValueError("label index outside the Hugging Face class names")
    if any(not text for text in texts):
        raise ValueError("parquet file contains an empty ticket")
    return texts, labels, names


def _download(url: str, dest: Path) -> bool:
    request = urllib.request.Request(url, headers={"User-Agent": "statim-ticket-triage/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403, 404):
            return False
        raise
    if not payload.startswith(b"PAR1"):
        raise SystemExit(f"download from {url} is not a parquet file")
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(".partial")
    temporary.write_bytes(payload)
    temporary.replace(dest)
    return True


def load_tickets() -> tuple[list[Ticket], list[str]]:
    if not (CACHE.is_file() and CACHE.stat().st_size > 8 and CACHE.read_bytes()[:4] == b"PAR1"):
        for url in PARQUET_URLS:
            print(f"downloading Banking77 test split\n  {url}", file=sys.stderr)
            if _download(url, CACHE):
                break
        else:
            raise SystemExit(
                "could not download the Banking77 test parquet from Hugging Face.\n"
                "Tried:\n  " + "\n  ".join(PARQUET_URLS)
            )
    texts, label_ids, names = read_parquet(CACHE)
    if len(texts) != 3080 or len(names) != 77:
        raise SystemExit(
            f"expected the Banking77 test split (3080 rows, 77 intents), "
            f"got {len(texts)} rows and {len(names)} intents from {CACHE}"
        )
    tickets = [
        Ticket(text=text, label=names[index], phrase=readable(names[index]))
        for text, index in zip(texts, label_ids)
    ]
    phrases = sorted({ticket.phrase for ticket in tickets})
    if len(phrases) != 77:
        raise SystemExit("intent names did not become 77 distinct phrases")
    return tickets, phrases


def stratified(tickets: list[Ticket], limit: int, seed: int) -> list[Ticket]:
    """Seeded sample with every intent represented as evenly as ``limit`` allows."""
    if limit >= len(tickets):
        return list(tickets)
    rng = random.Random(seed)
    groups: dict[str, list[Ticket]] = {}
    for ticket in tickets:
        groups.setdefault(ticket.label, []).append(ticket)
    labels = sorted(groups)
    for label in labels:
        rng.shuffle(groups[label])
    if limit < len(labels):
        chosen = labels[:]
        rng.shuffle(chosen)
        return [groups[label][0] for label in sorted(chosen[:limit])]
    base, remainder = divmod(limit, len(labels))
    order = labels[:]
    rng.shuffle(order)
    extra = set(order[:remainder])
    picked: list[Ticket] = []
    for label in labels:
        count = base + (1 if label in extra else 0)
        picked.extend(groups[label][:count])
    rng.shuffle(picked)
    return picked


def demo_sample(tickets: list[Ticket], seed: int) -> list[Ticket]:
    rng = random.Random(seed)
    picked = tickets[:]
    rng.shuffle(picked)
    return picked[:DEMO_TICKETS]


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def make_client(url: str) -> Any:
    client = Client(url, timeout=180)
    try:
        health = client.health()
    except Exception as exc:
        raise SystemExit(f"cannot reach {url} ({exc}). Start statim serve first.") from exc
    ready = client.ready()
    if not ready.ready:
        raise SystemExit(f"{url} is up (health {health.status}) but not ready")
    return client


def _probability(answer: ChoiceAnswer) -> float:
    return float(answer.probabilities.get(str(answer.choice), answer.answer_confidence))


def _score_level(answer: ScoreAnswer) -> tuple[str, float]:
    best = max(answer.probabilities, key=answer.probabilities.get)
    legend = answer.legend.get(best, best)
    return str(legend), float(answer.probabilities[best])


def _flag(answer: Any) -> str:
    if answer.escalate is None:
        return "n/a"
    return "yes" if answer.escalate else "no"


def print_demo(tickets: list[Ticket], decisions: list[Any], min_confidence: float) -> None:
    for index, (ticket, decision) in enumerate(zip(tickets, decisions), start=1):
        intent = decision.answers["intent"]
        urgency = decision.answers["urgency"]
        refund = decision.answers["refund"]
        if not isinstance(intent, ChoiceAnswer) or not isinstance(urgency, ScoreAnswer) or not isinstance(refund, YesNoAnswer):
            raise SystemExit("server returned an unexpected answer type")
        level, level_p = _score_level(urgency)
        # Only the routing decision (intent) decides whether a person must look at the ticket.
        # Urgency is an ordinal score whose probability naturally spreads over neighbouring
        # levels, and refund is a side signal; their flags are shown but do not escalate.
        escalated = ["intent"] if intent.escalate else []
        if min_confidence <= 0:
            ticket_flag = "no (min_confidence 0)"
        elif escalated:
            ticket_flag = f"yes (intent below {min_confidence:.2f})"
        else:
            ticket_flag = f"no (intent at or above {min_confidence:.2f})"
        match = "match" if str(intent.choice) == ticket.phrase else "miss"
        print(f"[{index}/{len(tickets)}]")
        print(textwrap.fill(ticket.text, width=88, initial_indent="text     ", subsequent_indent="         "))
        print(
            f"intent   {intent.choice}   p={_probability(intent):.4f}"
            f"   gold={ticket.phrase} ({ticket.label})   {match}"
            f"   escalate={_flag(intent)}"
        )
        print(
            f"urgency  {level}   score={urgency.score:.4f}   p={level_p:.4f}"
            f"   escalate={_flag(urgency)}"
        )
        side = "yes" if refund.yes else "no"
        print(f"refund   {side}   noul={refund.noul:.4f}   escalate={_flag(refund)}")
        print(f"ticket   escalate={ticket_flag}")
        print()


def curve_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(predictions)
    rows = []
    for threshold in CURVE_THRESHOLDS:
        # Same rule as the server: escalate when answer_confidence < threshold.
        # A threshold of 0 escalates nothing, because a confidence is never < 0.
        kept = [item for item in predictions if item["answer_confidence"] >= threshold]
        correct = sum(item["correct"] for item in kept)
        rows.append(
            {
                "min_confidence": threshold,
                "coverage": (len(kept) / total) if total else 0.0,
                "answered": len(kept),
                "escalated": total - len(kept),
                "accuracy": (correct / len(kept)) if kept else None,
                "correct": correct,
            }
        )
    return rows


def markdown_table(report: dict[str, Any]) -> str:
    lines = [
        "| min_confidence | coverage | answered | accuracy | correct |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in report["curve"]:
        accuracy = "—" if row["accuracy"] is None else f"{row['accuracy']:.4f}"
        lines.append(
            f"| {row['min_confidence']:.1f} | {row['coverage']:.4f} | {row['answered']} | {accuracy} | {row['correct']} |"
        )
    latency = report["latency_ms"]
    lines.append("")
    lines.append(
        f"End-to-end request latency at concurrency {report['concurrency']}: "
        f"p50 {latency['p50']:.1f} ms, p95 {latency['p95']:.1f} ms. "
        f"Throughput {report['throughput_rps']:.3f} requests/s "
        f"({report['n']} requests in {report['wall_seconds']:.2f} s)."
    )
    return "\n".join(lines) + "\n"


def run_demo(args: argparse.Namespace, tickets: list[Ticket], phrases: list[str]) -> None:
    if not 0.0 <= args.min_confidence <= 1.0:
        raise SystemExit("--min-confidence must be between 0 and 1")
    client = make_client(args.url)
    chosen = demo_sample(tickets, args.seed)
    questions = demo_questions(phrases)
    decisions = []
    for ticket in chosen:
        decisions.append(
            client.decide(
                {"message": ticket.text},
                questions,
                model=args.model,
                min_confidence=args.min_confidence,
                head_max_len=HEAD_MAX_LEN,
            )
        )
    print_demo(chosen, decisions, args.min_confidence)


def run_eval(args: argparse.Namespace, tickets: list[Ticket], phrases: list[str]) -> None:
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be a positive integer")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be a positive integer")
    client = make_client(args.url)
    chosen = tickets if args.limit is None else stratified(tickets, args.limit, args.seed)
    questions = eval_questions(phrases)
    models = client.models()
    loaded = next((item for item in models.data if item.id == args.model), None)

    def score(ticket: Ticket) -> dict[str, Any]:
        started = time.perf_counter()
        decision = client.decide(
            {"message": ticket.text},
            questions,
            model=args.model,
            head_max_len=HEAD_MAX_LEN,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        answer = decision.answers["intent"]
        if not isinstance(answer, ChoiceAnswer):
            raise RuntimeError("intent answer was not a choice")
        choice = str(answer.choice)
        return {
            "gold": ticket.phrase,
            "label": ticket.label,
            "choice": choice,
            "probability": _probability(answer),
            "answer_confidence": float(answer.answer_confidence),
            "correct": choice == ticket.phrase,
            "latency_ms": elapsed_ms,
            "inference_ms": decision.inference_time_ms,
            "routing_model": decision.routing.model,
            "weights": decision.routing.weights,
            "checkpoint": decision.model,
        }

    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    workers = min(args.concurrency, len(chosen))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(score, ticket) for ticket in chosen]
        try:
            for done, future in enumerate(as_completed(futures), start=1):
                predictions.append(future.result())
                if done % 25 == 0 or done == len(chosen):
                    elapsed = time.perf_counter() - started
                    print(f"{done}/{len(chosen)} in {elapsed:.1f}s", file=sys.stderr, flush=True)
        except Exception:
            pool.shutdown(cancel_futures=True)
            raise
    wall = time.perf_counter() - started
    latencies = [item["latency_ms"] for item in predictions]
    correct = sum(item["correct"] for item in predictions)
    report: dict[str, Any] = {
        "dataset": "PolyAI/banking77",
        "split": "test",
        "licence": "CC-BY-4.0",
        "n_test": len(tickets),
        "n": len(predictions),
        "limit": args.limit,
        "sample": "full test split" if args.limit is None else "seeded stratified by intent",
        "seed": args.seed,
        "intents_in_sample": len({item["label"] for item in predictions}),
        "url": args.url,
        "model_requested": args.model,
        "checkpoint": predictions[0]["checkpoint"] if predictions else None,
        "routing_model": predictions[0]["routing_model"] if predictions else None,
        "weights": predictions[0]["weights"] if predictions else None,
        "loaded_model": None
        if loaded is None
        else {
            "id": loaded.id,
            "source": loaded.source,
            "weights": loaded.weights,
            "device": loaded.device,
            "max_len": loaded.max_len,
            "layers": loaded.layers,
            "hidden": loaded.hidden,
        },
        "head_max_len": HEAD_MAX_LEN,
        "cpu": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "accuracy": (correct / len(predictions)) if predictions else None,
        "correct": correct,
        "curve": curve_rows(predictions),
        "latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "unit": "end-to-end HTTP request, including queueing",
        },
        "concurrency": args.concurrency,
        "throughput_rps": (len(predictions) / wall) if wall else None,
        "wall_seconds": wall,
        "question": "intent",
    }
    table = markdown_table(report)
    RESULTS_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    RESULTS_MD.write_text(table, encoding="utf-8")
    full = report["curve"][0]
    print(f"intent accuracy {report['accuracy']:.4f}  ({correct}/{len(predictions)})")
    print(table, end="")
    print(f"full coverage accuracy {full['accuracy']:.4f}")
    print(f"wrote {RESULTS_JSON}")
    print(f"wrote {RESULTS_MD}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Route Banking77 test tickets with a Statim server.",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="Statim base URL (default %(default)s)")
    parser.add_argument("--model", default="multilingual", help="model id (default %(default)s)")
    parser.add_argument("--seed", type=int, default=0, help="sample seed (default %(default)s)")
    parser.add_argument("command", choices=("demo", "eval"))
    parser.add_argument(
        "--min-confidence",
        dest="min_confidence",
        type=float,
        default=0.8,
        help="demo only: escalate answers below this answer_confidence (default %(default)s)",
    )
    parser.add_argument("--limit", type=int, default=None, help="eval only: seeded stratified sample size")
    parser.add_argument("--concurrency", type=int, default=4, help="eval only: request thread pool size (default %(default)s)")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    tickets, phrases = load_tickets()
    if args.command == "demo":
        run_demo(args, tickets, phrases)
    else:
        run_eval(args, tickets, phrases)


if __name__ == "__main__":
    main(sys.argv[1:])
