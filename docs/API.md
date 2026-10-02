# Statim HTTP API

Statim serves typed decisions over HTTP. A request carries a state, which is text or any JSON value except null, and zero or more questions. Each question has type `choice`, `score`, `noul`, or `yes_no`. The server scores every option of every question in one forward pass and returns the Jev/Laya `POST /v1/systemone` object. `yes_no` is an alias of `noul` in requests; answers always use the wire name `noul`.

The server listens on `127.0.0.1:8080` unless `--host` or `--port` is set. Paths outside the list below, and the wrong method on a known path, return 404 `{"detail":"HTTP request failed"}`. When authentication is configured, a nonpublic unknown path is rejected with 401 before route lookup unless it has a valid bearer key.

`GET /health` reports the version compiled into the binary. In this tree that version is `0.9.2` (`tools/release/check_versions.py` keeps this document, the SDKs and the site in step with `CMakeLists.txt`).

Successful JSON bodies are compact. The field order shown here is the order the server writes. Read fields by name.

## Quick start

Load two checkpoints and start the server. Names given to `-m` are the ids used for routing and for `GET /v1/models`.

```shell
build/statim serve \
  -m english=models/laya-english-f32.gguf \
  -m multilingual=models/laya-multilingual-f32.gguf \
  --port 8080 --no-access-log
```

The same call in curl, Python, and JavaScript follows. On a cold process the first decision is slower, because the weights are faulted in from disk. On the CPU used for these samples that call took about 5 seconds the first time and about 400 ms after the weights were resident.

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  -H 'X-Request-Id: doc-example-001' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary @- <<'EOF'
{
  "state": {
    "subject": "Duplicate charge on invoice #4411",
    "body": "We were billed twice for March. Please refund the duplicate today or we will cancel."
  },
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which department should handle this request?",
      "criteria": {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages",
        "sales": "pricing, new contracts",
        "other": "everything else"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this request?",
      "criteria": [
        "not urgent",
        "soon",
        "critical deadline or blocking issue"
      ]
    },
    "refund": {
      "type": "noul",
      "instructions": "Does the user explicitly request a refund?"
    }
  }
}
EOF
```

```text
{"model":"laya","answers":{"department":{"type":"choice","choice":"billing","probabilities":{"billing":0.97,"technical":0.0095,"sales":0.0107,"other":0.0098},"confidence":0.8791,"answer_confidence":0.97,"action":{"act_probability":1.0}},"urgency":{"type":"score","score":1.3385,"legend":{"0":"not urgent","1":"soon","2":"critical deadline or blocking issue"},"probabilities":{"0":0.1022,"1":0.4571,"2":0.4407},"confidence":0.1334,"answer_confidence":0.4571,"action":{"act_probability":1.0}},"refund":{"type":"noul","noul":0.8306,"confidence":0.8306,"answer_confidence":0.8306,"action":{"act_probability":1.0}}},"usage":{"input_tokens":215,"output_tokens":0},"routing":{"model":"english","reason":"lang:en","engine":"statim","weights":"f32"}}
200
```

```python
import json
import requests

payload = json.loads(r"""
{
  "state": {
    "subject": "Duplicate charge on invoice #4411",
    "body": "We were billed twice for March. Please refund the duplicate today or we will cancel."
  },
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which department should handle this request?",
      "criteria": {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages",
        "sales": "pricing, new contracts",
        "other": "everything else"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this request?",
      "criteria": [
        "not urgent",
        "soon",
        "critical deadline or blocking issue"
      ]
    },
    "refund": {
      "type": "noul",
      "instructions": "Does the user explicitly request a refund?"
    }
  }
}
""")
response = requests.post(
    "http://127.0.0.1:8080/v1/systemone",
    json=payload,
    headers={"X-Request-Id": "doc-example-001"},
    timeout=120,
)
if response.headers["X-Request-Id"] != "doc-example-001":
    raise SystemExit("X-Request-Id was not echoed")
if not response.headers["Server-Timing"].startswith("inference;dur="):
    raise SystemExit("Server-Timing missing")
if float(response.headers["X-Inference-Time-Ms"]) < 0:
    raise SystemExit("X-Inference-Time-Ms missing")
print(response.text)
print(response.status_code)
```

```text
{"model":"laya","answers":{"department":{"type":"choice","choice":"billing","probabilities":{"billing":0.97,"technical":0.0095,"sales":0.0107,"other":0.0098},"confidence":0.8791,"answer_confidence":0.97,"action":{"act_probability":1.0}},"urgency":{"type":"score","score":1.3385,"legend":{"0":"not urgent","1":"soon","2":"critical deadline or blocking issue"},"probabilities":{"0":0.1022,"1":0.4571,"2":0.4407},"confidence":0.1334,"answer_confidence":0.4571,"action":{"act_probability":1.0}},"refund":{"type":"noul","noul":0.8306,"confidence":0.8306,"answer_confidence":0.8306,"action":{"act_probability":1.0}}},"usage":{"input_tokens":215,"output_tokens":0},"routing":{"model":"english","reason":"lang:en","engine":"statim","weights":"f32"}}
200
```

```javascript
const payload = {
  "state": {
    "subject": "Duplicate charge on invoice #4411",
    "body": "We were billed twice for March. Please refund the duplicate today or we will cancel."
  },
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which department should handle this request?",
      "criteria": {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages",
        "sales": "pricing, new contracts",
        "other": "everything else"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this request?",
      "criteria": [
        "not urgent",
        "soon",
        "critical deadline or blocking issue"
      ]
    },
    "refund": {
      "type": "noul",
      "instructions": "Does the user explicitly request a refund?"
    }
  }
};
const response = await fetch("http://127.0.0.1:8080/v1/systemone", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-Request-Id": "doc-example-001",
  },
  body: JSON.stringify(payload),
});
const text = await response.text();
if (response.headers.get("X-Request-Id") !== "doc-example-001") {
  throw new Error("X-Request-Id was not echoed");
}
if (!response.headers.get("Server-Timing").startsWith("inference;dur=")) {
  throw new Error("Server-Timing missing");
}
if (Number.isNaN(Number(response.headers.get("X-Inference-Time-Ms")))) {
  throw new Error("X-Inference-Time-Ms missing");
}
console.log(text);
console.log(String(response.status));
```

```text
{"model":"laya","answers":{"department":{"type":"choice","choice":"billing","probabilities":{"billing":0.97,"technical":0.0095,"sales":0.0107,"other":0.0098},"confidence":0.8791,"answer_confidence":0.97,"action":{"act_probability":1.0}},"urgency":{"type":"score","score":1.3385,"legend":{"0":"not urgent","1":"soon","2":"critical deadline or blocking issue"},"probabilities":{"0":0.1022,"1":0.4571,"2":0.4407},"confidence":0.1334,"answer_confidence":0.4571,"action":{"act_probability":1.0}},"refund":{"type":"noul","noul":0.8306,"confidence":0.8306,"answer_confidence":0.8306,"action":{"act_probability":1.0}}},"usage":{"input_tokens":215,"output_tokens":0},"routing":{"model":"english","reason":"lang:en","engine":"statim","weights":"f32"}}
200
```

The response headers on that call are:

| Header | When |
|---|---|
| `X-Request-Id` | Every response produced by the decision handler, including errors. A supplied value is echoed only if it is 1–128 ASCII letters, digits, `.`, `_`, or `-`; otherwise the server replaces it with 16 generated hexadecimal characters. |
| `X-Inference-Time-Ms` | Successful `POST /v1/systemone` and `POST /v1/systemone/batch` only. Milliseconds with two decimal places. |
| `Server-Timing` | Same responses. The value is `inference;dur=` followed by the same number. |
| `Retry-After` | `503` from the decision handler only. The value is `1`. |

Every response, from every route and including errors, also carries `X-Content-Type-Options: nosniff` and
`Referrer-Policy: no-referrer`. `GET /` (the playground) adds a `Content-Security-Policy` that allows only its own
inline script and style block by SHA-256, connections to the same origin, and no framing
(`frame-ancestors 'none'`, plus `X-Frame-Options: DENY`). The playground keeps an API key only in the
tab's `sessionStorage`, so it is gone when the tab closes.

HTTP framing and declared body size, bearer authentication, and route lookup happen before the decision handler. Responses produced there have no `X-Request-Id`.

## Endpoints

| Method and path | Auth | Body |
|---|---|---|
| `POST /v1/systemone` | bearer, if keys are configured | one decision |
| `POST /v1/systemone/batch` | bearer, if keys are configured | the same questions for many states |
| `GET /v1/models` | bearer, if keys are configured | loaded checkpoints |
| `GET /health` | no | process is up |
| `GET /ready` | no | admission control has room |
| `GET /metrics` | bearer, if keys are configured | Prometheus text |
| `GET /` | no | playground HTML, unless `--no-playground` |

## Stability: API v1

API v1 is the HTTP interface described by `docs/openapi.yaml` and frozen as of 0.9.0 in
`docs/api-v1.contract.json`. Removing an operation, documented status, or request field; requiring
a previously optional or new request field; narrowing or changing a request type; removing a
request enum value or adding an enum to a request field that had none; adding or tightening a request validation bound; adding or changing a request
`pattern` or `format`; adding, removing, or changing a request field default; removing a response
field, making one optional, or changing its type; removing or renaming a documented `/metrics`
family or changing its kind; and any change to the `Error` body shape are breaking changes. A
breaking change requires a new major version. Frozen validation includes `maxItems`, `minItems`,
`maxLength`, `minLength`, `maximum`, `minimum`, `exclusiveMaximum`, `exclusiveMinimum`,
`maxProperties`, `minProperties`, `multipleOf`, `pattern`, and `format`.

New operations, optional request fields, response fields, documented statuses, and enum values are
additive and remain compatible, as are removed or loosened request validation bounds and new metric
families. Response validation-bound and default changes are compatibility notes, not failures.
Descriptions and examples may be corrected without changing the frozen structural contract, except
that the `# TYPE <name> <kind>` metric families in the `/metrics` example are frozen. Error responses
retain the documented `{"detail": ...}` shape, and their `detail` wording is documented and tested,
but the wording is not frozen: clients should branch on HTTP status codes rather than error text.

CI runs the mutation suite in `tools/docs/test_api_contract.py`, then compares the current OpenAPI
document with the frozen contract using `python3 tools/docs/api_contract.py --check`. CTest's
`api_contract` test also sends every paired request/response example to a real CPU server and
validates all response examples against their JSON Schema 2020-12 schemas.

With no API keys configured, the bearer check is skipped and every path is open. `GET /health`, `GET /ready`, and `GET /` never require a key. When authentication is configured, `/metrics` and `/v1/models` return 401 unless the request has a configured bearer key.

## Decision request

`POST /v1/systemone` reads a JSON object.

| Field | Required | Meaning |
|---|---|---|
| `state` | yes | Text or any JSON value except null. A string is scored as written. Any other value is serialized with Python-style separators (`, ` and `: `), with object key order kept and non-ASCII characters kept. When the token budget cuts the state, a JSON array keeps the end of the sequence and every other value keeps the beginning. |
| `questions` | yes | Object. Each key is the question id returned in `answers`. At most 64 questions. |
| `model` | no | `english`, `multilingual`, `consensus`, an alias `convaiinnovations/laya-<id>`, or omitted. Any other string is ignored and routing runs as if `model` were omitted. See [Model routing and consensus](#model-routing-and-consensus). |
| `adapter` | no | `null`, `"none"`, `"auto"`, or the name of a LoRA adapter loaded for the selected model. See [LoRA adapters](#lora-adapters). Anything else is HTTP 422. |
| `lang` | no | String. Selects a per-language temperature when the checkpoint has one for that language (the part before `-`, compared case-insensitively). It does not select the checkpoint. The English and multilingual GGUF files shipped with this tree store an empty language-temperature table, so this field does not change scores on those files. |
| `ensemble` | no | Integer from 1 to 8 inclusive. Any other present value is HTTP 422 with `ensemble must be an integer between 1 and 8`. Omitting it uses the server default (1, unless `--ensemble` was set). Values above 1 re-score choice questions under extra cyclic option orders and average them. Score and noul questions are not rotated. |
| `ensemble_margin` | no | Number, clamped to 0..1. A non-number is ignored. The default is 1. Extra option orders run only when the gap between the top probability and the second is below this margin. A margin of 0 runs no extra orders. |
| `calibrate` | no | Boolean. A non-boolean is ignored. The server default is false unless `--calibrate` was set. When true, each choice question with at least two options is adjusted by the distribution that question produces on content-free copies of the state (empty string, `N/A`, and `[MASK]`). |
| `return_logits` | no | Boolean. A non-boolean is ignored. When true, each answer gains a `logits` array, in option order, before temperature scaling. Under consensus the field is `logits_by_model` instead. See below. |
| `min_confidence` | no | Number from 0 to 1 inclusive. A positive value enables selective prediction and overrides `--min-confidence`; `0` disables it for this request. Any other present value is HTTP 422 with `min_confidence must be a number from 0 to 1`. |
| `max_len` | no | Integer from 32 to 8192 inclusive. Anything else that is present, including a non-integer, is HTTP 422. Omitting it uses the checkpoint default (English 512, multilingual 1024 on these files). |
| `head_max_len` | no | Integer from 32 to 8192 inclusive, same error rule. This is the token budget for the question and its options. Checkpoint defaults on these files: English 192, multilingual 256. |

The effective state budget inside the engine is `max(max_len, head_max_len + 128)`, using the checkpoint default for a limit you omitted. Sending `max_len: 32` therefore does not cut the sequence to 32 tokens when `head_max_len` is 192.

Unknown top-level request fields are ignored for compatibility with `laya.serve`.

Each question is an object:

| Field | Required | Meaning |
|---|---|---|
| `type` | yes | `choice`, `score`, `noul`, or `yes_no`; `yes_no` is a request alias of `noul`. |
| `instructions` | yes | The text to answer. A non-string is serialized as JSON and used as text. |
| `criteria` | depends on type | Options. See the next section. |
| `labels` | no | Only valid on `noul` and its `yes_no` request alias. Maps `false` and `true` to two different non-empty strings. |

Unknown fields in a question definition are also ignored. They do not affect inference or calibration-cache keys.

How criteria become option text:

- Choice object: each entry is `label: description`. A null or empty description keeps the label only.
- Choice list: each item must be a string, number, boolean, or null. The option text is that scalar. Duplicate labels keep the first occurrence.
- Score list: entry `i` becomes `level i: <description>`. Null entries are rejected. The description may be any JSON value except null.
- Noul: optional object with keys `true` and `false` only (case-insensitive). The default lines are `false: no, the statement does not hold` and `true: yes, the statement holds`. `labels` replaces the words `false` and `true`.

## Response

A single decision is one object. `model` is the checkpoint name stored in the GGUF file (`general.name`), not the `-m` id. On these files that is `laya` or `laya-multilingual`. Consensus sets `model` to `consensus`.

`routing.model` is the id that was actually used: `english`, `multilingual`, or `consensus`. `routing.reason` is `requested`, `lang:en`, `lang:other`, `default` (only one checkpoint is loaded, or the two are not named `english` and `multilingual`), or `consensus`. `routing.engine` is `statim`. `routing.weights` is the weight type of the checkpoint that ran. For consensus it is the English checkpoint's weight type. `routing.reason` is `adapter` when a named LoRA adapter selected the model. When the server has LoRA adapters loaded or the request sets `adapter`, `routing.adapter` is the adapter that ran, or `null` for the base weights, and `routing.adapter_reason` says why (`none`, `requested`, `auto:<family>`, `auto:<family>:no-adapter`, `auto:no-family`, `auto:consensus`); see [LoRA adapters](#lora-adapters).

`usage.input_tokens` counts encoder tokens for the canonical option order. Extra ensemble views are not added. Consensus adds the two checkpoints together. `usage.output_tokens` is always 0.

Probabilities, scores, `noul`, `confidence`, `answer_confidence`, and `act_probability` are rounded to 4 decimal places on their own, so probabilities need not sum to 1.

`confidence` for choice and score is `1 - entropy / log(number of options)`, clamped to 0..1. A question with one option has confidence 1. `answer_confidence` is the highest option probability.

For noul, `noul` is the probability of the true side. `confidence` and `answer_confidence` are both `max(noul, 1 - noul)`.

`action.act_probability` is the checkpoint's action head. The published Laya configs train that head with an `escalate` cost. The HTTP response does not name the action. On the published checkpoints this head saturates at 1.0 whatever the confidence, so it is not a usable abstain signal; use `min_confidence` and the `escalate` flag (see [Selective prediction](#selective-prediction)) to decide when a person should look instead.

The English checkpoint applies a stored temperature that depends on the question type and how many options it has. The multilingual checkpoint uses temperature 1. Consensus averages the option log-probabilities of both checkpoints and decodes at temperature 1.

With `return_logits: true`, a single checkpoint adds `logits` in option order. Those values are the raw scores for one view. After an ensemble they are the averaged log-softmax values. Consensus does not put a fused `logits` array on the answer. It adds `logits_by_model`, with the raw logits of `english` and `multilingual`.

## Selective prediction

Selective prediction lets an application accept confident answers automatically and hand uncertain ones to a person or a larger model. Set `min_confidence` to a number from 0 to 1 in either decision request, or start the server with `--min-confidence P` to use a default for all requests. A request value overrides the server default; `min_confidence: 0` disables selective prediction for that request.

When the effective threshold is greater than zero, every answer that has `answer_confidence` also has an `escalate` boolean. It is `true` exactly when `answer_confidence < min_confidence`; confidence equal to the threshold is not escalated. When no positive threshold applies, `escalate` is absent, preserving the original response shape.

```json
{
  "state": "I was charged twice for the same transfer.",
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "What is the banking intent?",
      "criteria": {
        "cash_withdrawal": "cash withdrawal issue",
        "card_payment_fee_charged": "fee charged for a card payment",
        "cash_withdrawal_or_transfer_charged_twice": "cash withdrawal or transfer charged twice"
      }
    }
  },
  "min_confidence": 0.9
}
```

```json
{
  "model": "laya-multilingual",
  "answers": {
    "intent": {
      "type": "choice",
      "choice": "cash_withdrawal_or_transfer_charged_twice",
      "probabilities": {
        "cash_withdrawal": 0.03,
        "card_payment_fee_charged": 0.05,
        "cash_withdrawal_or_transfer_charged_twice": 0.92
      },
      "confidence": 0.7315,
      "answer_confidence": 0.92,
      "action": {"act_probability": 1.0},
      "escalate": false
    }
  },
  "usage": {"input_tokens": 42, "output_tokens": 0},
  "routing": {"model": "multilingual", "reason": "default", "engine": "statim", "weights": "f32"}
}
```

On 400 Banking77 test rows, the 0.4.0 model measured 91.5% accuracy at full coverage. Keeping the 90% most confident answers raised accuracy to 95.6%; keeping the 60% most confident raised it to 99.2%. Coverage is the share answered automatically; the remainder is the candidate escalation set. These measurements describe that evaluation set, not a universal calibration guarantee, so choose the threshold on representative validation data.

## Question types

The quick start response is the full shape. The calls below show the variants that change the schema.

A choice whose criteria is a list of labels. The winning `choice` is one of those labels, with its original JSON type.

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary "{\"state\":\"The app crashes when I open settings.\",\"model\":\"english\",\"questions\":{\"topic\":{\"type\":\"choice\",\"instructions\":\"Topic?\",\"criteria\":[\"billing\",\"technical\",\"other\"]}}}"
```

```text
{"model":"laya","answers":{"topic":{"type":"choice","choice":"technical","probabilities":{"billing":0.0653,"technical":0.8436,"other":0.091},"confidence":0.5086,"answer_confidence":0.8436,"action":{"act_probability":1.0}}},"usage":{"input_tokens":24,"output_tokens":0},"routing":{"model":"english","reason":"requested","engine":"statim","weights":"f32"}}
200
```

A list may contain numbers and null. The winning value keeps that JSON type. Probability keys are the JSON spelling of the label (`1`, `2`, `null`, `true`, `false`).

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary "{\"state\":\"Pick 2.\",\"model\":\"english\",\"questions\":{\"n\":{\"type\":\"choice\",\"instructions\":\"Which number is named?\",\"criteria\":[1,2,null]}}}"
```

```text
{"model":"laya","answers":{"n":{"type":"choice","choice":2,"probabilities":{"1":0.1383,"2":0.684,"null":0.1777},"confidence":0.235,"answer_confidence":0.684,"action":{"act_probability":1.0}}},"usage":{"input_tokens":21,"output_tokens":0},"routing":{"model":"english","reason":"requested","engine":"statim","weights":"f32"}}
200
```

Score is the expected index into `criteria`, starting at 0. `legend` maps the index, as a string, back to the criterion you sent. The quick start response shows a score next to a choice.

Noul with custom labels and criteria. The response still has a single number, `noul`. It does not echo the labels.

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary "{\"state\":\"Please refund the duplicate charge.\",\"model\":\"english\",\"questions\":{\"refund\":{\"type\":\"noul\",\"instructions\":\"Does the user explicitly request a refund?\",\"criteria\":{\"true\":\"the user asks for money back\",\"false\":\"no refund is requested\"},\"labels\":{\"true\":\"yes\",\"false\":\"no\"}}}}"
```

```text
{"model":"laya","answers":{"refund":{"type":"noul","noul":0.8266,"confidence":0.8266,"answer_confidence":0.8266,"action":{"act_probability":1.0}}},"usage":{"input_tokens":38,"output_tokens":0},"routing":{"model":"english","reason":"requested","engine":"statim","weights":"f32"}}
200
```

An empty `questions` object is valid. Nothing is scored. `input_tokens` is 0 and the inference time is 0.

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary "{\"state\":\"hello\",\"questions\":{}}"
```

```text
{"model":"laya","answers":{},"usage":{"input_tokens":0,"output_tokens":0},"routing":{"model":"english","reason":"lang:en","engine":"statim","weights":"f32"}}
200
```

## Batching

`POST /v1/systemone/batch` takes `states`, an array of up to 256 values, and one `questions` object applied to every state. The response is `{"results":[...]}` in the same order. Each element has the single-decision shape. The states share forward passes. One state that fails a limit fails the whole request.

When the server starts with a positive `--batch-window-ms`, compatible concurrent
`POST /v1/systemone` calls are also packed into shared forward passes, up to `--max-batch` states.
Compatibility requires the same resolved model, validated questions, language, and effective token,
ensemble, calibration, logits, and confidence options. Each call keeps its own request ID, routing,
usage, deadline, and single-decision JSON shape. Answers are identical to isolated execution within
the 1e-4 probability parity tolerance. Explicit `/v1/systemone/batch` calls are not delayed or
combined by this scheduler.

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone/batch \
  --data-binary @- <<'EOF'
{
  "states": [
    "Please refund the duplicate charge today.",
    "The login page returns a 500 error."
  ],
  "model": "english",
  "questions": {
    "topic": {
      "type": "choice",
      "instructions": "What is this about?",
      "criteria": {
        "billing": "payments and refunds",
        "technical": "bugs and outages"
      }
    }
  }
}
EOF
```

```text
{"results":[{"model":"laya","answers":{"topic":{"type":"choice","choice":"billing","probabilities":{"billing":0.932,"technical":0.068},"confidence":0.6415,"answer_confidence":0.932,"action":{"act_probability":1.0}}},"usage":{"input_tokens":33,"output_tokens":0},"routing":{"model":"english","reason":"requested","engine":"statim","weights":"f32"}},{"model":"laya","answers":{"topic":{"type":"choice","choice":"technical","probabilities":{"billing":0.0745,"technical":0.9255},"confidence":0.6175,"answer_confidence":0.9255,"action":{"act_probability":1.0}}},"usage":{"input_tokens":34,"output_tokens":0},"routing":{"model":"english","reason":"requested","engine":"statim","weights":"f32"}}]}
200
```

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone/batch \
  --data-binary "{\"states\":[],\"questions\":{}}"
```

```text
{"results":[]}
200
```

Python, same request as the two-state call:

```python
import json
import requests

payload = json.loads(r"""
{
  "states": [
    "Please refund the duplicate charge today.",
    "The login page returns a 500 error."
  ],
  "model": "english",
  "questions": {
    "topic": {
      "type": "choice",
      "instructions": "What is this about?",
      "criteria": {
        "billing": "payments and refunds",
        "technical": "bugs and outages"
      }
    }
  }
}
""")
response = requests.post("http://127.0.0.1:8080/v1/systemone/batch", json=payload, timeout=120)
print(response.text)
print(response.status_code)
```

```text
{"results":[{"model":"laya","answers":{"topic":{"type":"choice","choice":"billing","probabilities":{"billing":0.932,"technical":0.068},"confidence":0.6415,"answer_confidence":0.932,"action":{"act_probability":1.0}}},"usage":{"input_tokens":33,"output_tokens":0},"routing":{"model":"english","reason":"requested","engine":"statim","weights":"f32"}},{"model":"laya","answers":{"topic":{"type":"choice","choice":"technical","probabilities":{"billing":0.0745,"technical":0.9255},"confidence":0.6175,"answer_confidence":0.9255,"action":{"act_probability":1.0}}},"usage":{"input_tokens":34,"output_tokens":0},"routing":{"model":"english","reason":"requested","engine":"statim","weights":"f32"}}]}
200
```

## Model routing and consensus

When `model` is omitted, both an `english` and a `multilingual` checkpoint are loaded, and the server was started without `--consensus`, the server picks a checkpoint from the state text.

The state counts as English when its letters are almost all ASCII and the text has enough common English function words. Text shorter than four words counts as English if it contains any word. Text whose non-ASCII letters are more than a small fraction of the ASCII letters goes to the multilingual checkpoint. This is a heuristic. It is not a language identifier, and it does not read `lang`.

An English state therefore routes to `english` with reason `lang:en`. The quick start response is that case. A German sentence routes to `multilingual` with reason `lang:other`.

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary @- <<'EOF'
{
  "state": "Die Rechnung wurde doppelt belastet. Bitte erstatten Sie den Betrag noch heute.",
  "questions": {
    "refund": {
      "type": "noul",
      "instructions": "Does the user explicitly request a refund?"
    }
  }
}
EOF
```

```text
{"model":"laya-multilingual","answers":{"refund":{"type":"noul","noul":0.9657,"confidence":0.9657,"answer_confidence":0.9657,"action":{"act_probability":1.0}}},"usage":{"input_tokens":50,"output_tokens":0},"routing":{"model":"multilingual","reason":"lang:other","engine":"statim","weights":"f32"}}
200
```

`model` set to a loaded id uses that checkpoint and reason `requested`. The alias is the literal prefix `convaiinnovations/laya-` plus the `-m` id, so `convaiinnovations/laya-english` selects the model you named `english`. It does not look up the Hugging Face repository `convaiinnovations/laya`.

A string that matches nothing, including a Jev model id, is not an error. Routing continues as if `model` had been omitted.

The next call prints `routing` for four ways of asking for the English invoice. The answers match.

```python
import json
import requests

base = json.loads(r"""
{
  "state": {
    "subject": "Duplicate charge on invoice #4411",
    "body": "We were billed twice for March. Please refund the duplicate today or we will cancel."
  },
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which department should handle this request?",
      "criteria": {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages",
        "sales": "pricing, new contracts",
        "other": "everything else"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this request?",
      "criteria": [
        "not urgent",
        "soon",
        "critical deadline or blocking issue"
      ]
    },
    "refund": {
      "type": "noul",
      "instructions": "Does the user explicitly request a refund?"
    }
  }
}
""")
variants = {
    "omitted": dict(base),
    "english": {**base, "model": "english"},
    "alias": {**base, "model": "convaiinnovations/laya-english"},
    "unknown": {**base, "model": "jev-typed-decisions"},
}
answers = {}
for name, payload in variants.items():
    response = requests.post("http://127.0.0.1:8080/v1/systemone", json=payload, timeout=120)
    response.raise_for_status()
    body = response.json()
    routing = body["routing"]
    print(f"{name} {routing['model']} {routing['reason']} {body['model']}")
    answers[name] = body["answers"]
same = answers["omitted"] == answers["english"] == answers["alias"] == answers["unknown"]
print("answers-match" if same else "answers-differ")
```

```text
omitted english lang:en laya
english english requested laya
alias english requested laya
unknown english lang:en laya
answers-match
```

`model: "consensus"` runs both checkpoints and averages their option log-probabilities with equal weight. Both must be loaded. `usage.input_tokens` is the sum of the two passes. If either checkpoint is missing, the string `consensus` is an unknown id and ordinary routing runs.

Starting the server with `--consensus` uses this path when the request omits `model`. A request that names `english` or `multilingual` still selects that checkpoint.

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary @- <<'EOF'
{
  "state": {
    "subject": "Duplicate charge on invoice #4411",
    "body": "We were billed twice for March. Please refund the duplicate today or we will cancel."
  },
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which department should handle this request?",
      "criteria": {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages",
        "sales": "pricing, new contracts",
        "other": "everything else"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this request?",
      "criteria": [
        "not urgent",
        "soon",
        "critical deadline or blocking issue"
      ]
    },
    "refund": {
      "type": "noul",
      "instructions": "Does the user explicitly request a refund?"
    }
  },
  "model": "consensus"
}
EOF
```

```text
{"model":"consensus","answers":{"department":{"type":"choice","choice":"billing","probabilities":{"billing":1.0,"technical":0.0,"sales":0.0,"other":0.0},"confidence":0.9999,"answer_confidence":1.0,"action":{"act_probability":1.0}},"urgency":{"type":"score","score":1.6652,"legend":{"0":"not urgent","1":"soon","2":"critical deadline or blocking issue"},"probabilities":{"0":0.0223,"1":0.2901,"2":0.6875},"confidence":0.3614,"answer_confidence":0.6875,"action":{"act_probability":1.0}},"refund":{"type":"noul","noul":0.963,"confidence":0.963,"answer_confidence":0.963,"action":{"act_probability":1.0}}},"usage":{"input_tokens":433,"output_tokens":0},"routing":{"model":"consensus","reason":"consensus","engine":"statim","weights":"f32"}}
200
```

JavaScript, consensus on the refund question, with per-checkpoint logits:

```javascript
const payload = {
  "state": "Please refund the duplicate charge today.",
  "model": "consensus",
  "return_logits": true,
  "questions": {
    "refund": {
      "type": "noul",
      "instructions": "Does the user explicitly request a refund?"
    }
  }
};
const response = await fetch("http://127.0.0.1:8080/v1/systemone", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify(payload),
});
console.log(await response.text());
console.log(String(response.status));
```

```text
{"model":"consensus","answers":{"refund":{"type":"noul","noul":0.976,"confidence":0.976,"answer_confidence":0.976,"action":{"act_probability":1.0},"logits_by_model":{"english":[-2.530663013458252,1.1402685642242432],"multilingual":[-3.2405383586883545,0.49913012981414795]}}},"usage":{"input_tokens":82,"output_tokens":0},"routing":{"model":"consensus","reason":"consensus","engine":"statim","weights":"f32"}}
200
```

The same question on the English checkpoint with `return_logits`. The array is the raw option scores, false then true.

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary "{\"state\":\"Please refund the duplicate charge today.\",\"model\":\"english\",\"return_logits\":true,\"questions\":{\"refund\":{\"type\":\"noul\",\"instructions\":\"Does the user explicitly request a refund?\"}}}"
```

```text
{"model":"laya","answers":{"refund":{"type":"noul","noul":0.8642,"confidence":0.8642,"answer_confidence":0.8642,"action":{"act_probability":1.0},"logits":[-2.530663013458252,1.1402685642242432]}},"usage":{"input_tokens":41,"output_tokens":0},"routing":{"model":"english","reason":"requested","engine":"statim","weights":"f32"}}
200
```

## LoRA adapters

One base checkpoint can serve several small, per-category LoRA adapters. Each request picks one with `adapter`; the base weights, the tokenizer and every tensor an adapter does not touch are shared.

### Convert

`tools/convert_lora.py` reads a PEFT adapter directory (`adapter_config.json` and `adapter_model.safetensors`, as `save_pretrained` writes them) and writes a Statim adapter GGUF:

```bash
python tools/convert_lora.py runs/emotion-lora -o models/emotion.lora.gguf \
    --base models/laya-multilingual-f32.gguf --category emotion
```

- Targets: the encoder's attention (`attn.Wqkv`, `attn.Wo`) and MLP (`mlp.Wi`, `mlp.Wo`) projections, any subset of layers. Keys may carry the `base_model.model.` prefix and an adapter name (`lora_A.default.weight`).
- Scale: `lora_alpha / r`, or `lora_alpha / sqrt(r)` with `use_rslora`; `rank_pattern` and `alpha_pattern` are matched per module the way PEFT matches them. The scale is folded into `lora_b`, so the file stores `W' = W + lora_b · lora_a`.
- Factors may be f32, f16 or bf16 (the dtype PEFT saves when the model was trained in bf16); f16 and bf16 are widened to f32 exactly.
- Rejected with an error, because Statim cannot represent them and dropping them would serve a different model than the one that was evaluated:
  - LoRA on any other module (decision head, embeddings, classifier), `bias` other than `none`, `lora_bias`, `modules_to_save`, `trainable_token_indices`, `layer_replication`, `target_parameters`, a missing `lora_A` or `lora_B`, a rank that disagrees with the config;
  - `fan_in_fan_out`, which PEFT sets for Conv1D weights stored as (in, out), not for the encoder's `nn.Linear` projections;
  - the LoRA variants whose inference is not `W + B·A`: DoRA, aLoRA (`alora_invocation_tokens`), QALoRA, BD-LoRA, KaSA and Arrow. VeLoRA, MonteCLoRA and MiCA only change training (PEFT evaluates and merges them as plain LoRA), so they convert, and their training-only tensors are skipped;
  - `init_lora_weights` PiSSA, OLoRA, CorDA or LoRA-GA, which change the base weights at initialisation, so the saved adapter only fits that changed base, and LoftQ, which replaces them with quantized ones. PEFT's `save_pretrained(..., path_initial_model_for_weight_conversion=...)` turns the first four into a plain LoRA (saved with `init_lora_weights: true`), which converts.
- `tools/convert_laya.py` records `statim.checkpoint_sha256`: SHA-256 over every tensor it writes, in byte order of the tensor names, each as the name and a NUL byte, the element count (u64 little-endian), and the source values as little-endian f32 before f16 conversion or quantization. Every output `--type` made from one source checkpoint has the same value, including when `statim-quantize` re-encodes it.
- `--base` (required) is the Statim model the adapter was trained on. Every shape is checked against it, and its `general.name`, fingerprint and, when present, `statim.checkpoint_sha256` are recorded. The fingerprint is a SHA-256 over the checkpoint's vectors, every tensor of ggml shape `[n, 1, 1, 1]` (the normalisation weights and the biases of the decision head and scorer): in byte order of the names, each as its name and a NUL byte, the element count (u64 little-endian) and the values as little-endian f32. Full fine-tuning changes these weights and quantizing never touches them, so the fingerprint tells apart two fine-tunes that share a name and is the same for the f32, f16, q8_0 and q4_0 files of one checkpoint (whose names may differ). The full checkpoint SHA-256 also distinguishes checkpoints with equal vectors but different matrices, such as one with another LoRA already merged. The engine checks both values when both files have the full SHA-256; if either lacks it, compatibility with files converted before this key was added is preserved by checking only the fingerprint. `statim info -m model.gguf` prints `fingerprint` and, when present, `checkpoint_sha256`.
- `--category` (repeatable) lists the question families the adapter serves in `"adapter": "auto"`; without it the adapter name is used.

File layout (`statim.format` = `statim-lora-v1`): `encoder.layers.<l>.<target>.weight.lora_a` (f32, ggml ne `[in, r]`) and `.lora_b` (f32, ne `[r, out]`, pre-scaled), plus `statim.lora.rank`, `statim.lora.alpha`, `statim.lora.rslora`, `statim.lora.categories`, `statim.lora.base_name`, `statim.lora.base_fingerprint` and, for bases that have it, `statim.lora.base_checkpoint_sha256`. The engine always requires the recorded fingerprint and also requires the checkpoint SHA-256 to match when both files carry it.

### Load

```bash
statim serve -m multilingual=models/laya-multilingual-q8_0.gguf \
    --adapter multilingual:emotion=models/emotion.lora.gguf \
    --adapter multilingual:sentiment=models/sentiment.lora.gguf
```

`--adapter [model:]name=file.gguf` attaches an adapter to the `-m` model called `model` (the first `-m` model when omitted). Names are 1-64 characters of `A-Z a-z 0-9 _ . -`; `auto` and `none` are reserved; a name may repeat across models but not within one. Every adapter is loaded and checked at startup, so a bad file stops the server before it listens. `statim decide`, `bench` and `info` take one `--adapter name=file.gguf` as well.

`--adapter-mode merge|runtime` chooses how adapters are applied, for all of them. Without it, adapters on f32, f16 or bf16 weights are merged and adapters on quantized weights run as runtime LoRA; see [below](#merge-at-load-vs-runtime-lora).

Adapters share the base model's `--workers` slots and engines: a server with `--workers 2` runs at most two forward passes at once over the base and all its adapters together, and holds at most two engines (compute buffers) for them, so adapters add neither compute threads nor compute memory. An engine serves one weight set at a time; when a request needs one for another adapter (or the base) and none is idle for it, an idle engine of another weight set is dropped and a new one is built. `statim_engines` in `/metrics` shows the count. With `--batch-window-ms`, requests are only batched with requests for the same adapter.

### Select

| `adapter` | Weights used | `routing.adapter` | `routing.adapter_reason` |
|---|---|---|---|
| omitted, `null` or `"none"` | base | `null` | `none` |
| a loaded name, e.g. `"emotion"` | that adapter | `"emotion"` | `requested` |
| `"auto"` | the adapter serving the request's question family, else base | its name or `null` | `auto:<family>`, `auto:<family>:no-adapter` or `auto:no-family` |

Model routing runs first. An explicit `model` must carry the named adapter, otherwise the request fails with 422. Without an explicit `model`, a named adapter that exactly one loaded model carries selects that model (`routing.reason` = `adapter`); otherwise the usual language routing picks the model, and the adapter must exist there. `"model": "consensus"` accepts only `"auto"` (which then uses the base weights, `auto:consensus`) or no adapter. The `--consensus` default applies to requests without a named adapter only: a request that names one runs on that adapter's model.

```bash
curl -sS http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' --data-binary @- <<'EOF'
{"state": "Worst experience ever. The agent hung up on me twice.",
 "adapter": "auto",
 "questions": {"sentiment": {"type": "choice", "instructions": "What is the sentiment of the message?",
                             "criteria": ["positive", "neutral", "negative"]}}}
EOF
```

With one model loaded and an adapter of category `sentiment`, the response carries `"routing": {"model": "multilingual", "reason": "default", ..., "adapter": "sentiment", "adapter_reason": "auto:sentiment"}`. `"auto"` never selects the model: `routing.reason` comes from the usual model routing (`default`, `lang:en`, `lang:other` or `requested`). The access log records the adapter per request.

### The auto rule

The families are the decision categories of the training mixture. Each family has a fixed keyword list:

| Family | Keywords |
|---|---|
| `sentiment` | sentiment, polarity |
| `emotion` | emotion, emotions, emotional, feeling, feelings, mood |
| `complaint` | complaint, complaints, complain, complaining |
| `nli` | nli, entailment, entail, entails, contradiction, contradict, contradicts |
| `safety` | safety, unsafe, toxic, toxicity, harmful, moderation |
| `reading` | reading, comprehension, passage |
| `similarity` | similarity, similar, paraphrase, paraphrases |
| `topic` | topic, topics |
| `intent` | intent, intents, intention |
| `stance` | stance |
| `formality` | formality, formal, informal |
| `urgency` | urgency, urgent |
| `fact_check` | fact, facts, factual, claim, claims |
| `pii` | pii, personally |

1. A question's family comes from its ID first: the ID is split into lowercase ASCII words (runs of letters and digits, so `fact_check` is `fact` `check`). If those words hit the keywords of exactly one family, that is the family.
2. If the ID hits no family, the same test runs on `instructions` (when it is a string).
3. No hit, or hits in more than one family, means the question has no family.
4. The request's family is the family every question shares. One question without a family, or two questions of different families, route the whole request to the base weights (`auto:no-family`).
5. Among the chosen model's adapters, the first one (in `--adapter` order) whose categories contain the family is used; if none does, the base weights are used (`auto:<family>:no-adapter`).

The rule is deliberately literal and keyword-based: it is deterministic, costs nothing, and a caller who wants a family can name the question after it (`"sentiment": {...}`). Non-English instructions only match through the question ID. It does not route per question: a request that mixes families runs on the base weights, and a caller who wants per-family adapters for such a request sends one request per family.

### Merge at load vs. runtime LoRA

`merge` computes `W' = W + B·A` once per adapted projection when the adapter loads: the base weight is dequantized to f32, `B·A` is added with a ggml graph on the CPU, and the result is stored in the base weight's type (quantized types are quantized again; on the CPU, repackable types go to the same repack buffer as the base). Requests then run the unchanged base graph. `runtime` keeps the base weights and adds `B·(A·x)` after each adapted projection in the graph. Without `--adapter-mode`, adapters on float weights (f32, f16, bf16) are merged and adapters on quantized weights run as runtime LoRA, for the accuracy reason below.

Measured on 4 threads of an Intel Xeon @ 2.10 GHz, `laya-multilingual`, every projection of all 22 layers adapted, 10 short states from `tests/data/golden_inputs.json` with all 8 questions (one request per state, mean of 30):

| Base weights | Adapter | Mode | ms per request | Extra memory | Load time |
|---|---|---|---|---|---|
| f32 | – | base | 560–588 | – | – |
| f32 | rank 4 | merge | 519–557 | 438 MB | 0.7–0.8 s |
| f32 | rank 4 | runtime | 749 | 3.3 MB | 4 ms |
| f32 | rank 16 | runtime | 707 | 13.5 MB | 12 ms |
| q8_0 | – | base | 798–830 | – | – |
| q8_0 | rank 4 | merge | 812 | 116 MB | 1.2 s |
| q8_0 | rank 4 | runtime | 1067 | 3.3 MB | 4 ms |
| q8_0 | rank 16 | runtime | 934 | 13.5 MB | 10 ms |

- Latency: a merged adapter runs at base speed (the differences above are run-to-run noise). Runtime LoRA costs 17-28 % per request although rank 4 adds under 1 % of the FLOPs: it adds three graph nodes per adapted projection (264 for this model), and on short CPU inputs every node pays a thread barrier and a pass over the activations. On float weights merge is exact and costs nothing per request, so it is the default there.
- Accuracy on float weights: the two modes agree within 2.4e-5 in the logits.
- Accuracy on quantized weights: merge quantizes `W + B·A` again, so the adapter's delta is rounded to the quantization grid, and a LoRA delta is often smaller than one step of it. `lora_quantized` measures this on the 80 reference items with the random rank-4 test adapter (mean |Δlogit|, CPU):

  | Base | vs. the f32 base | Adapter effect (runtime) | Runtime vs. the f32 PyTorch merge | Merge vs. runtime |
  |---|---|---|---|---|
  | q8_0 | 0.085 | 0.480 | 0.081 | 0.070, 14 % of the effect |
  | q4_0 | 0.641 | 0.549 | 0.674 | 0.502, 91 % of the effect |

  Runtime LoRA keeps a quantized model with the adapter as close to the f32 reference as the quantized base is to the f32 base. Merging loses a seventh of the adapter's effect on q8_0 and nearly all of it on q4_0, where the test adapter's delta (about 0.004 per weight) is smaller than one quantization step; a trained adapter's delta is often smaller still. So runtime is the default on quantized weights, and `--adapter-mode merge` trades that accuracy for base latency. For the best quantized result, merge the adapter into the checkpoint in PyTorch and quantize once with `tools/convert_laya.py`.
- Memory: the price of merge is one copy of the adapted projections per adapter: 438 MB in f32, 116 MB in q8_0, whatever the rank. Runtime stores only the factors (MB). Fourteen category adapters on an f32 base cost about 6.1 GB merged; that is when `--adapter-mode runtime` pays off there too.
- Exactness: a LoRA pair whose `lora_a` or `lora_b` is all zeros is skipped in both modes (the base tensor stays in place), so an untrained adapter (PEFT initialises `lora_B` to zero) is bit-identical to the base on every weight type.
- Types that need an importance matrix to quantize (the IQ types) cannot be merged; runtime mode, their default, works.

### Inspect

`GET /v1/models` lists each model's adapters: `id`, `source` (the adapter's `general.name`), `mode`, `rank`, `alpha`, `pairs` (LoRA pairs in the file), `pairs_applied` (non-zero pairs), `categories` and `bytes` (memory added on top of the base). `/metrics` exports `statim_adapter_info{model,adapter,mode} 1` and `statim_adapter_bytes{model,adapter}` per adapter, and `statim_engines{model}` (engines alive for a model and its adapters, at most `--workers`). `statim info -m base.gguf --adapter name=file.gguf` prints the same fields; `statim info` also prints the model's `fingerprint`.

### Tests

`ctest -R lora` (CPU):

- `lora_fixtures` writes a zero and a random rank-4 PEFT adapter from an integer hash (no RNG, identical on every platform) and converts them.
- `lora_convert` checks scaling, f16 and bf16 factors, the recorded fingerprint against `statim info`, and every rejected input of the converter.
- `lora_parity` checks that the zero adapter's logits and `decide()` output are bit-identical to the base, and that the random adapter's merged weights and logits match a PyTorch reference that merged the same adapter into the official Laya model (`tests/lora/gen_reference.py`, committed as `tests/data/golden_lora_random.jsonl`) within 1e-4, in both modes. It also checks SHA-256 against the FIPS 180-2 vectors and that adapters for another checkpoint, without a fingerprint, or with shapes that do not fit are refused.
- `lora_quantized` quantizes the base to q8_0 and q4_0 (`lora_quantize_*`) and checks the table above: same fingerprint, runtime as the default, runtime within 1.5 times the base's quantization noise of the f32 reference, merged weights equal to `quantize(dequantize(W) + B·A)` bit for bit, and repacked merged weights computing what the file layout computes.
- `server_lora` covers selection, auto routing, model selection by adapter, the batch endpoint, micro-batching across adapters, `/v1/models`, `/metrics`, authentication before adapter checks, a named adapter under `--consensus`, one engine for the base and two adapters with `--workers 1`, and the error cases.

CI runs them with `ctest -R lora -V`, so the measured differences are in the log.

## Option budget

Options share `head_max_len` tokens with the question text. When they do not fit, each option is cut to `max(4, (head_max_len - 16) / number of options)` tokens. With the English default of 192, a choice of about 77 labels is cut to one or two subwords per label, so the model does not read the label text. Set `head_max_len` to 512 for that kind of question. The state budget grows with it, because the engine keeps at least `head_max_len + 128` tokens in total.

Raising the budget does not change answers when the options already fit. The invoice questions fit in the English default. Sending `max_len` 512 and `head_max_len` 512 returns the same answers as leaving both fields out.

```python
import json
import requests

payload = json.loads(r"""
{
  "state": {
    "subject": "Duplicate charge on invoice #4411",
    "body": "We were billed twice for March. Please refund the duplicate today or we will cancel."
  },
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which department should handle this request?",
      "criteria": {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages",
        "sales": "pricing, new contracts",
        "other": "everything else"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this request?",
      "criteria": [
        "not urgent",
        "soon",
        "critical deadline or blocking issue"
      ]
    },
    "refund": {
      "type": "noul",
      "instructions": "Does the user explicitly request a refund?"
    }
  },
  "model": "english"
}
""")
plain = requests.post("http://127.0.0.1:8080/v1/systemone", json=payload, timeout=120)
raised = requests.post(
    "http://127.0.0.1:8080/v1/systemone",
    json={**payload, "max_len": 512, "head_max_len": 512},
    timeout=120,
)
print("same-answers" if plain.json()["answers"] == raised.json()["answers"] else "answers-differ")
print(plain.status_code)
print(raised.status_code)
```

```text
same-answers
200
200
```

If the option markers still do not fit in the effective `max_len`, the handler returns 422. The message has no colon after the question id. One hundred short options with both limits set to 32 returns that status. One hundred options is within the count limit. The failure is the token budget.

```python
import requests

criteria = {f"o{i}": f"description {i}" for i in range(100)}
response = requests.post(
    "http://127.0.0.1:8080/v1/systemone",
    json={
        "state": "x",
        "max_len": 32,
        "head_max_len": 32,
        "questions": {"q": {"type": "choice", "instructions": "pick one", "criteria": criteria}},
    },
    timeout=120,
)
print(response.text)
print(response.status_code)
```

```text
{"detail":"question 'q' options exceed head_max_len=32"}
422
```

## Ensemble and calibration

`ensemble` greater than 1 builds extra cyclic rotations of choice options and averages log-softmax scores in label order. `usage.input_tokens` stays at the canonical pass. The two calls below use the same three-way question. The probabilities move.

```python
import json
import requests

payload = {
    "state": "I was charged twice and also the site is down.",
    "model": "english",
    "questions": json.loads(r"""
{
  "topic": {
    "type": "choice",
    "instructions": "What is this about?",
    "criteria": {
      "billing": "payments and refunds",
      "technical": "bugs and outages",
      "other": "anything else"
    }
  }
}
"""),
}
for ensemble in (1, 3):
    response = requests.post(
        "http://127.0.0.1:8080/v1/systemone",
        json={**payload, "ensemble": ensemble, "ensemble_margin": 1},
        timeout=120,
    )
    print(response.text)
    print(response.status_code)
```

```text
{"model":"laya","answers":{"topic":{"type":"choice","choice":"billing","probabilities":{"billing":0.7798,"technical":0.2015,"other":0.0187},"confidence":0.4619,"answer_confidence":0.7798,"action":{"act_probability":1.0}}},"usage":{"input_tokens":42,"output_tokens":0},"routing":{"model":"english","reason":"requested","engine":"statim","weights":"f32"}}
200
{"model":"laya","answers":{"topic":{"type":"choice","choice":"billing","probabilities":{"billing":0.678,"technical":0.2976,"other":0.0244},"confidence":0.3494,"answer_confidence":0.678,"action":{"act_probability":1.0}}},"usage":{"input_tokens":42,"output_tokens":0},"routing":{"model":"english","reason":"requested","engine":"statim","weights":"f32"}}
200
```

`ensemble` is validated as an integer from 1 through 8; values outside that range and non-integers return 422. A non-boolean `calibrate` is ignored. This request uses the largest valid ensemble, a margin that is clamped to 1, and a string `calibrate` value that is ignored. It returns the empty-question response.

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary "{\"state\":\"hi there\",\"questions\":{},\"ensemble\":8,\"ensemble_margin\":5,\"calibrate\":\"yes\"}"
```

```text
{"model":"laya","answers":{},"usage":{"input_tokens":0,"output_tokens":0},"routing":{"model":"english","reason":"lang:en","engine":"statim","weights":"f32"}}
200
```

Calibration changes choice probabilities. The same two-way question, without and with `calibrate`.

```python
import requests

payload = {
    "state": "Please refund the duplicate charge today.",
    "model": "english",
    "questions": {
        "topic": {
            "type": "choice",
            "instructions": "What is this about?",
            "criteria": {"billing": "payments and refunds", "technical": "bugs and outages"},
        }
    },
}
for flag in (False, True):
    response = requests.post(
        "http://127.0.0.1:8080/v1/systemone",
        json={**payload, "calibrate": flag},
        timeout=120,
    )
    print(response.text)
    print(response.status_code)
```

```text
{"model":"laya","answers":{"topic":{"type":"choice","choice":"billing","probabilities":{"billing":0.932,"technical":0.068},"confidence":0.6415,"answer_confidence":0.932,"action":{"act_probability":1.0}}},"usage":{"input_tokens":33,"output_tokens":0},"routing":{"model":"english","reason":"requested","engine":"statim","weights":"f32"}}
200
{"model":"laya","answers":{"topic":{"type":"choice","choice":"billing","probabilities":{"billing":0.9122,"technical":0.0878},"confidence":0.5709,"answer_confidence":0.9122,"action":{"act_probability":1.0}}},"usage":{"input_tokens":33,"output_tokens":0},"routing":{"model":"english","reason":"requested","engine":"statim","weights":"f32"}}
200
```

## Authentication

Set keys with `STATIM_API_KEY` (comma-separated) or `--api-key-file` (one key per line). Comma-separated values and file lines are trimmed and blank values are skipped; in files, trimmed lines beginning with `#` are also skipped. A key must contain 1–4096 printable ASCII characters and no whitespace.

Authentication is fail-closed per configured source. If `STATIM_API_KEY` is present but empty or contains no valid key, startup aborts. Each `--api-key-file` must be readable and contain at least one valid key; a missing, unreadable, empty, or comment-only file aborts startup even if another source supplied a valid key. With neither source configured, startup is allowed only on `localhost`, `::1`, or an IPv4 address in `127.0.0.0/8`. Other hosts, including `0.0.0.0`, `::`, LAN addresses, and hostnames, exit with status 2 before model loading. `--allow-unauthenticated` explicitly permits an unauthenticated non-loopback listener and retains the `auth_off_on_network` warning and `"auth":false,"auth_status":"off"` listening log.

The client sends `Authorization: Bearer <key>`. The comparison is constant-time over the full header. A missing header, a wrong scheme, or a wrong key is:

```json
{"detail":"invalid or missing bearer token"}
```

That response is HTTP 401. It is produced before the route handler and therefore has no `X-Request-Id`.

The check covers both decision endpoints, `GET /metrics`, and `GET /v1/models`. `GET /health`, `GET /ready`, and `GET /` remain public. Exactly one `Authorization` header is required when auth is on.

## Errors

Handler errors are JSON objects with one string field, `detail`. HTTP framing, declared body size, bearer authentication, and body prohibition on operational endpoints are checked before routing. Decision checks then run in this order: authentication, the concurrency cap, body reception and JSON preflight, request-field sizes and question shape, integer budgets, states and per-state limits, aggregate work/capacity estimates, engine queueing, and inference.

| Status | Trigger | Body |
|---|---|---|
| 400 | body is not JSON | `{"detail":"request body must be valid JSON"}` |
| 400 | request body reception fails before an oversize condition is identified | `{"detail":"request body incomplete or exceeds limit"}` |
| 400 | duplicate key in any JSON object | `{"detail":"duplicate JSON object key"}` |
| 400 | multiple `Content-Length` headers, or both `Content-Length` and `Transfer-Encoding` | `{"detail":"ambiguous HTTP body framing"}` |
| 400 | empty or non-decimal `Content-Length` | `{"detail":"invalid Content-Length"}` |
| 400 | a public/operational endpoint receives a nonempty or transfer-encoded body | `{"detail":"this endpoint does not accept a request body"}` |
| 400 | top-level JSON or a question definition is not an object | `{"detail":"expected JSON object"}` |
| 400 | `questions` is missing or is not an object | `{"detail":"'questions' must be an object"}` |
| 400 | `state` is missing or null | `{"detail":"'state' is required"}` |
| 400 | batch request has no `states` array | `{"detail":"request body must contain a 'states' array"}` |
| 401 | a key is configured and `Authorization` does not match `Bearer <key>` | `{"detail":"invalid or missing bearer token"}` |
| 413 | body is larger than 2 MiB | `{"detail":"request body exceeds 2 MiB"}` for a declared oversize body; `{"detail":"request body incomplete or exceeds limit"}` when streaming crosses the limit |
| 413 | JSON nesting exceeds `--max-json-depth` (default 64; hard maximum 128) | `{"detail":"JSON nesting too deep"}` |
| 413 | JSON nodes exceed `--max-json-nodes` (default 100,000) | `{"detail":"too many JSON nodes"}` |
| 413 | one object exceeds `--max-object-members` (default 1,024) | `{"detail":"too many JSON object members"}` |
| 413 | a JSON object key exceeds 4,096 bytes | `{"detail":"JSON key exceeds 4096 bytes"}` |
| 413 | more than 64 questions | `{"detail":"too many questions"}` |
| 413 | more than 100 choice options in one question | `{"detail":"too many choice options for 'q' (101 > 100)"}` |
| 413 | more than 32 score levels in one question | `{"detail":"too many score levels for 'q' (33 > 32)"}` |
| 413 | choice options plus score levels across questions exceed 512 | `{"detail":"too many answer options across questions (540 > 512)"}` |
| 413 | state is longer than 50,000 Unicode code points | `{"detail":"state too large (50001 > 50000 chars)"}` |
| 413 | batch has more than 256 states | `{"detail":"too many states (257 > 256)"}` |
| 413 | question ID exceeds 256 bytes | `{"detail":"question ID exceeds 256 bytes"}` |
| 413 | instructions exceed 16,384 bytes | `{"detail":"instructions exceed 16384 bytes"}` |
| 413 | an object option/label key exceeds 1,024 bytes | `{"detail":"label exceeds 1024 bytes"}` |
| 413 | a criterion or label value exceeds 4,096 bytes | `{"detail":"criterion or label exceeds 4096 bytes"}` |
| 413 | aggregate evaluated rows exceed `--max-request-work` (default 4,096) | `{"detail":"request exceeds state/question/view work limit"}` |
| 413 | rows × effective sequence length exceed `--max-request-tokens` (default 1,048,576) | `{"detail":"request exceeds aggregate token budget"}` |
| 413 | the conservative attention estimate exceeds `--max-attention-mib` (default 1,024 MiB) | `{"detail":"request exceeds attention memory budget"}` |
| 413 | the conservative response estimate exceeds `--max-response-bytes` (default 16,777,216) | `{"detail":"request exceeds response byte budget"}` |
| 413 | the final serialized response exceeds `--max-response-bytes` | `{"detail":"response exceeds byte budget"}` |
| 413 | a defensive state/question/model cardinality check fails | `{"detail":"request exceeds state/question/model limits"}` |
| 422 | `model` or `lang` is not a string of at most 256 bytes | `{"detail":"model must be a string of at most 256 bytes"}` (the field name changes) |
| 422 | `adapter` is not null or a string of at most 256 bytes | `{"detail":"adapter must be null or a string of at most 256 bytes"}` |
| 422 | `adapter` names no adapter of the selected model | `{"detail":"unknown adapter 'x' for model 'multilingual' (loaded: emotion, sentiment)"}`; the parenthesis says `none loaded` and lists other models' adapters as `other models: <model>:<adapter>, ...` when there are any |
| 422 | a named `adapter` with `"model": "consensus"` | `{"detail":"adapter 'emotion' cannot be combined with consensus"}` |
| 422 | `max_len` or `head_max_len` is not an integer from 32 to 8192 | `{"detail":"max_len must be an integer between 32 and 8192"}` (the field name changes) |
| 422 | `ensemble` is not an integer from 1 to 8 | `{"detail":"ensemble must be an integer between 1 and 8"}` |
| 422 | `min_confidence` is not a number from 0 to 1 | `{"detail":"min_confidence must be a number from 0 to 1"}` |
| 422 | effective `max(max_len, head_max_len + 128)` exceeds the selected model's capacity | `{"detail":"effective max_len exceeds model capacity (head_max_len needs 128 state tokens)"}` |
| 422 | a question definition is invalid, or its options do not fit the token budget | `{"detail":"question '<id>': ..."}` or `{"detail":"question '<id>' options exceed head_max_len=<n>"}` |
| 422 | the cooperative inference deadline expires (default 120 seconds from admission) | `{"detail":"inference deadline exceeded"}` |
| 422 | the client disconnects while its request waits in a micro-batch (`--batch-window-ms`) | `{"detail":"inference cancelled"}` |
| 500 | an unexpected exception during the decision handler | `{"detail":"inference failed"}` |
| 500 | an unexpected exception outside the decision handler | `{"detail":"internal server error"}` |
| 503 | more than `--max-concurrent` requests (default 16) are already in the handler | `{"detail":"server busy, try again later"}` plus `Retry-After: 1` |
| 503 | no model worker becomes available before the inference deadline | `{"detail":"inference queue deadline exceeded"}` |

`GET /ready` uses a different 503 body, `{"ready":false}`, and does not set `Retry-After`. See [Metrics and other operations](#metrics-and-other-operations).

Question validation messages:

| `detail` | Trigger |
|---|---|
| `expected JSON object` (HTTP 400) | the question value is not an object |
| `unknown question type; use choice, score, noul or yes_no` | `type` is missing, not a string, or not one of those four |
| `question requires instructions` | `instructions` is missing |
| `question '<id>': a choice question takes 'criteria' as a dict of label -> description, or a list of labels` | choice `criteria` is missing or not an object or list |
| `question '<id>': a choice question needs at least one criterion` | the object or list is empty |
| `question '<id>': choice label <i> must be a scalar (a string, number or null)` | a list entry is an object or array |
| `question '<id>': a score question takes 'criteria' as a list of level descriptions, index 0 first` | score `criteria` is not a list |
| `question '<id>': a score question needs at least one level` | the list is empty |
| `question '<id>': score level <i> is null; give every level a description` | a level is null |
| `question '<id>': a noul question takes 'criteria' as a dict with optional 'true'/'false' descriptions, or omits it` | noul `criteria` is present and is not an object |
| `question '<id>': a noul question takes 'criteria' keyed only 'true'/'false'` | an object key is something else |
| `question '<id>': 'labels' is only supported for noul questions` | `labels` on a choice or score |
| `question '<id>': noul labels must map exactly 'false' and 'true' to distinct non-empty strings` | `labels` is not that map |
| `question '<id>' options exceed head_max_len=<n>` | option markers do not fit in the effective token budget |

HTTP 500 is the handler's catch-all. The samples below did not hit an unexpected exception, so that body was not returned by the server used here. The string is what the handler writes. The log line on stderr contains the underlying message. The client does not receive it.

The calls below are the bodies in the tables.

An error produced inside the decision handler still echoes `X-Request-Id`. Pre-routing framing, declared-size, authentication, body-prohibition, and route errors do not. Errors do not set `Server-Timing`; only the admission-saturation 503 sets `Retry-After: 1`.

```python
import requests

response = requests.post(
    "http://127.0.0.1:8080/v1/systemone",
    json={"questions": {}},
    headers={"X-Request-Id": "doc-error-001"},
    timeout=30,
)
if response.headers["X-Request-Id"] != "doc-error-001":
    raise SystemExit("request id was not echoed on an error")
if "Server-Timing" in response.headers or "Retry-After" in response.headers:
    raise SystemExit("success timing headers were set on an error")
print(response.text)
print(response.status_code)
```

```text
{"detail":"'state' is required"}
400
```

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary "not json"
```

```text
{"detail":"request body must be valid JSON"}
400
```

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary "[1,2]"
```

```text
{"detail":"expected JSON object"}
400
```

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary "{\"state\":\"x\"}"
```

```text
{"detail":"'questions' must be an object"}
400
```

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary "{\"questions\":{}}"
```

```text
{"detail":"'state' is required"}
400
```

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary "{\"state\":null,\"questions\":{}}"
```

```text
{"detail":"'state' is required"}
400
```

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary "{\"state\":\"x\",\"questions\":[]}"
```

```text
{"detail":"'questions' must be an object"}
400
```

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone/batch \
  --data-binary "{\"questions\":{\"refund\":{\"type\":\"noul\",\"instructions\":\"Does the user explicitly request a refund?\"}}}"
```

```text
{"detail":"request body must contain a 'states' array"}
400
```

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary "{\"state\":\"x\",\"questions\":{},\"max_len\":31}"
```

```text
{"detail":"max_len must be an integer between 32 and 8192"}
422
```

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary "{\"state\":\"x\",\"questions\":{},\"max_len\":8193}"
```

```text
{"detail":"max_len must be an integer between 32 and 8192"}
422
```

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary "{\"state\":\"x\",\"questions\":{},\"head_max_len\":32.5}"
```

```text
{"detail":"head_max_len must be an integer between 32 and 8192"}
422
```

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary "{\"state\":\"x\",\"questions\":{\"q\":{\"type\":\"rank\",\"instructions\":\"x\"}}}"
```

```text
{"detail":"unknown question type; use choice, score, noul or yes_no"}
422
```

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary "{\"state\":\"x\",\"questions\":{\"q\":{\"type\":\"noul\"}}}"
```

```text
{"detail":"question requires instructions"}
422
```

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary "{\"state\":\"x\",\"questions\":{\"q\":\"noul\"}}"
```

```text
{"detail":"expected JSON object"}
400
```

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary "{\"state\":\"x\",\"questions\":{\"q\":{\"type\":\"choice\",\"instructions\":\"x\",\"criteria\":{}}}}"
```

```text
{"detail":"question 'q': a choice question needs at least one criterion"}
422
```

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary "{\"state\":\"x\",\"questions\":{\"q\":{\"type\":\"choice\",\"instructions\":\"x\",\"criteria\":\"a, b\"}}}"
```

```text
{"detail":"question 'q': a choice question takes 'criteria' as a dict of label -> description, or a list of labels"}
422
```

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary "{\"state\":\"x\",\"questions\":{\"q\":{\"type\":\"choice\",\"instructions\":\"x\",\"criteria\":[{\"a\":1}]}}}"
```

```text
{"detail":"question 'q': choice label 0 must be a scalar (a string, number or null)"}
422
```

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary "{\"state\":\"x\",\"questions\":{\"q\":{\"type\":\"score\",\"instructions\":\"x\",\"criteria\":{\"a\":\"b\"}}}}"
```

```text
{"detail":"question 'q': a score question takes 'criteria' as a list of level descriptions, index 0 first"}
422
```

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary "{\"state\":\"x\",\"questions\":{\"q\":{\"type\":\"score\",\"instructions\":\"x\",\"criteria\":[]}}}"
```

```text
{"detail":"question 'q': a score question needs at least one level"}
422
```

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary "{\"state\":\"x\",\"questions\":{\"q\":{\"type\":\"score\",\"instructions\":\"x\",\"criteria\":[\"ok\",null]}}}"
```

```text
{"detail":"question 'q': score level 1 is null; give every level a description"}
422
```

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary "{\"state\":\"x\",\"questions\":{\"q\":{\"type\":\"noul\",\"instructions\":\"x\",\"criteria\":[\"maybe\"]}}}"
```

```text
{"detail":"question 'q': a noul question takes 'criteria' as a dict with optional 'true'/'false' descriptions, or omits it"}
422
```

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary "{\"state\":\"x\",\"questions\":{\"q\":{\"type\":\"noul\",\"instructions\":\"x\",\"criteria\":{\"maybe\":\"x\"}}}}"
```

```text
{"detail":"question 'q': a noul question takes 'criteria' keyed only 'true'/'false'"}
422
```

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary "{\"state\":\"x\",\"questions\":{\"q\":{\"type\":\"noul\",\"instructions\":\"x\",\"labels\":{\"false\":\"no\",\"true\":\"no\"}}}}"
```

```text
{"detail":"question 'q': noul labels must map exactly 'false' and 'true' to distinct non-empty strings"}
422
```

```bash
curl -sS -w '\n%{http_code}\n' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/systemone \
  --data-binary "{\"state\":\"x\",\"questions\":{\"q\":{\"type\":\"choice\",\"instructions\":\"x\",\"criteria\":[\"a\"],\"labels\":{\"true\":\"y\",\"false\":\"n\"}}}}"
```

```text
{"detail":"question 'q': 'labels' is only supported for noul questions"}
422
```

Counts and sizes that are awkward to paste as JSON are generated in Python.

```python
import requests

url = "http://127.0.0.1:8080/v1/systemone"
batch = "http://127.0.0.1:8080/v1/systemone/batch"

def show(response):
    print(response.status_code)
    print(response.text)

questions = {f"q{i}": {"type": "noul", "instructions": "x"} for i in range(65)}
show(requests.post(url, json={"state": "x", "questions": questions}, timeout=30))

criteria = {f"o{i}": "d" for i in range(101)}
show(requests.post(url, json={"state": "x", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": criteria}}}, timeout=30))

levels = [f"level {i}" for i in range(33)]
show(requests.post(url, json={"state": "x", "questions": {"q": {"type": "score", "instructions": "x", "criteria": levels}}}, timeout=30))

many = {}
for i in range(6):
    many[f"q{i}"] = {"type": "choice", "instructions": "x", "criteria": {f"o{j}": "d" for j in range(90)}}
show(requests.post(url, json={"state": "x", "questions": many}, timeout=30))

show(requests.post(url, json={"state": "a" * 50001, "questions": {}}, timeout=30))
show(requests.post(batch, json={"states": ["x"] * 257, "questions": {}}, timeout=60))

blob = '{"state":"' + ("a" * (2 * 1024 * 1024)) + '","questions":{}}'
response = requests.post(url, data=blob.encode(), headers={"Content-Type": "application/json"}, timeout=30)
print(response.status_code)
print(response.text)
```

```text
413
{"detail":"too many questions"}
413
{"detail":"too many choice options for 'q' (101 > 100)"}
413
{"detail":"too many score levels for 'q' (33 > 32)"}
413
{"detail":"too many answer options across questions (540 > 512)"}
413
{"detail":"state too large (50001 > 50000 chars)"}
413
{"detail":"too many states (257 > 256)"}
413
{"detail":"request body exceeds 2 MiB"}
```

Noul questions do not add to the 512-option total. Only choice entries and score levels do. A question that is missing `type` or `criteria` is not counted toward those option caps. It still fails later with 422 if the definition is invalid. The state limit counts Unicode code points, not bytes. For a non-string state the count is the code points of the serialized JSON.

The concurrency cap is separate from option limits. The default is 16 requests inside the handler (`--max-concurrent`). The default worker count is 1, so requests inside the cap wait for the worker. The next request receives 503. While the cap is full, `GET /ready` returns 503 with `{"ready":false}` and no `Retry-After`.

```python
import threading
import time
import requests
from concurrent.futures import ThreadPoolExecutor

payload = {"state": {"subject": "Duplicate charge on invoice #4411", "body": "We were billed twice for March. Please refund the duplicate today or we will cancel."}, "questions": {"department": {"type": "choice", "instructions": "Which department should handle this request?", "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages", "sales": "pricing, new contracts", "other": "everything else"}}, "urgency": {"type": "score", "instructions": "How urgent is this request?", "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]}, "refund": {"type": "noul", "instructions": "Does the user explicitly request a refund?"}}, "model": "english"}
url = "http://127.0.0.1:8080/v1/systemone"
ready = {"status": None, "body": None, "retry": None}
stop = threading.Event()

def poll():
    while not stop.is_set():
        response = requests.get("http://127.0.0.1:8080/ready", timeout=5)
        if response.status_code == 503:
            ready["status"] = 503
            ready["body"] = response.text
            ready["retry"] = response.headers.get("Retry-After")
            return
        time.sleep(0.02)

def one():
    response = requests.post(url, json=payload, timeout=180)
    return response.status_code, response.headers.get("Retry-After"), response.text

watcher = threading.Thread(target=poll)
watcher.start()
with ThreadPoolExecutor(max_workers=24) as pool:
    results = [f.result() for f in [pool.submit(one) for _ in range(24)]]
stop.set()
watcher.join(timeout=5)

busy = [item for item in results if item[0] == 503]
ok = [item for item in results if item[0] == 200]
if not busy or not ok:
    raise SystemExit(f"expected both 200 and 503, got {sorted(set(item[0] for item in results))}")
status, retry, text = busy[0]
if retry != "1" or text != '{"detail":"server busy, try again later"}':
    raise SystemExit(f"unexpected 503: {retry} {text}")
if ready["body"] != '{"ready":false}' or ready["retry"] is not None:
    raise SystemExit(f"unexpected ready: {ready}")
print(text)
print(status)
print("retry")
print(retry)
print(ready["body"])
print(ready["status"])
```

```text
{"detail":"server busy, try again later"}
503
retry
1
{"ready":false}
503
```

## Limits and server controls

Limits count UTF-8 bytes unless the table says Unicode code points. Unknown request and question fields are ignored, but still count toward the body, JSON-node, object-member, and key-size limits.

| Limit or flag | Default | Meaning |
|---|---:|---|
| Request body | 2 MiB | Length-framed and chunked bodies |
| `--max-json-depth` | 64 | Nested containers including the root; configurable only up to 128 |
| `--max-json-nodes` | 100,000 | Containers, scalar values, and object keys |
| `--max-object-members` | 1,024 | Members in any one object |
| JSON object key | 4,096 bytes | Includes arbitrary state keys |
| States / questions | 256 / 64 | Per batch / per request |
| State | 50,000 code points | String value, or Python-compatible serialization of a structured state |
| Choice / score / total options | 100 / 32 / 512 | Per choice / per score / across choice and score questions |
| Question ID / `model` / `lang` | 256 bytes each | `model` and `lang` must also be strings |
| Instructions | 16,384 bytes | String bytes, or compact rendered JSON size |
| Criterion or label value | 4,096 bytes | Object label keys are limited to 1,024 bytes |
| `max_len`, `head_max_len` | checkpoint defaults | Explicit request values and nonzero CLI defaults must be integers 32–8,192; CLI 0 selects checkpoint defaults |
| `ensemble` / `--ensemble` | 1 | Integer 1–8 |
| `min_confidence` / `--min-confidence` | unset | Number 0–1; a positive effective value adds `escalate` to answers below the threshold |
| `--max-request-work` | 4,096 | `states × views × models`; choice ensembles, three calibration views, and two consensus models count |
| `--max-request-tokens` | 1,048,576 | Work rows × effective sequence length |
| `--max-attention-mib` | 1,024 MiB | Conservative packed-graph attention estimate |
| `--max-response-bytes` | 16,777,216 | Conservative preflight estimate and final serialized response |
| `--max-concurrent` | 16 | Admitted decision requests; valid range 1–256 |
| `--workers` | 1 | Inference workers per model; valid range 1–64 |
| `--batch-window-ms` | 0 | Collection window for compatible concurrent single-state requests; 0 disables micro-batching |
| `--max-batch` | 16 | Maximum states in a server-created micro-batch; valid range 1–256 |
| `--http-queue` | 32 | Pending sockets beyond the fixed `max-concurrent + 4` HTTP workers |
| `--request-timeout` | 30 seconds | Absolute combined header/body read deadline |
| Keep-alive / write timeout | 2 seconds idle, 100 requests / 30 seconds | Fixed server settings |
| `--inference-timeout` | 120 seconds | From admission through body read, queue wait, and cooperative inference |

All numeric CLI arguments use strict non-negative decimal integer parsing and reject values above 2,147,483,647. Limit/deadline/queue flags must be positive. Other serve controls are `--host`, `--port`, `--device`, `--gpu-fast`, `--threads`, `--calibrate`, `--consensus`, `--allow-unauthenticated`, `--no-access-log`, `--no-playground`, `--max-len N` and `--head-max-len N` (server-wide default token budgets; a request's `max_len` and `head_max_len` override them), and repeatable `-m [name=]model.gguf` / `--api-key-file FILE`. Environment controls are `STATIM_API_KEY`, `STATIM_DEVICE`, `STATIM_GPU_FAST=1`, and `STATIM_LOG=debug`.

Aggregate budgets deliberately use upper bounds, so short text can be rejected when the requested sequence budget is large. Effective sequence length is `max(max_len, head_max_len + 128)` and must fit each selected model's positional capacity. The attention estimate is `2 × min(32 × length, 8192) × length × max(encoder_heads, head_heads) × 4` bytes. The response estimate reserves 1,024 bytes per state plus 4,096 bytes per question and eight times each serialized question and ID size.

## Metrics and other operations

`GET /health` is liveness. It is 200 when the process is accepting connections. Models are loaded before the socket opens, so a listening server has already loaded its checkpoints.

```bash
curl -sS -w '\n%{http_code}\n' http://127.0.0.1:8080/health
```

```text
{"status":"ok","version":"0.9.2"}
200
```

`GET /ready` is 200 `{"ready":true}` when fewer than `max_concurrent` requests are in the handler. The saturation example above shows the 503 body.

```bash
curl -sS -w '\n%{http_code}\n' http://127.0.0.1:8080/ready
```

```text
{"ready":true}
200
```

`GET /v1/models` lists checkpoints in load order. `id` is the `-m` name. `source` is the GGUF `general.name`. `max_len` is the checkpoint default. `head_max_len` is not in this object. `device` is the actual compute device used by that model. When the server was started with `--adapter`, every entry also has `adapters`, the model's LoRA adapters (possibly empty; fields in [LoRA adapters](#lora-adapters)). This endpoint requires the bearer key when authentication is configured.

```bash
curl -sS -w '\n%{http_code}\n' http://127.0.0.1:8080/v1/models
```

```text
{"object":"list","data":[{"id":"english","object":"model","owned_by":"statim","source":"laya","weights":"f32","layers":28,"hidden":1024,"max_len":512,"vocab":50368,"device":"cpu"},{"id":"multilingual","object":"model","owned_by":"statim","source":"laya-multilingual","weights":"f32","layers":22,"hidden":768,"max_len":1024,"vocab":256000,"device":"cpu"}]}
200
```

`GET /` is the playground when it is enabled, which is the default. The page is a single HTML document titled Statim Playground. It calls `/health`, `/ready`, `/v1/models`, and `POST /v1/systemone` from the browser. `--no-playground` makes `GET /` return 404 `{"detail":"not found"}`. That flag was not set on the server that produced these samples.

```python
import requests

response = requests.get("http://127.0.0.1:8080/", timeout=30)
if response.status_code != 200:
    raise SystemExit(response.status_code)
if not response.headers["Content-Type"].startswith("text/html"):
    raise SystemExit(response.headers["Content-Type"])
if "<title>Statim Playground</title>" not in response.text:
    raise SystemExit("title missing")
print(response.headers["Content-Type"])
print(response.status_code)
print("playground-ok")
```

```text
text/html; charset=utf-8
200
playground-ok
```

`GET /metrics` is Prometheus text, `Content-Type: text/plain; version=0.0.4`. It requires the bearer key when authentication is configured. Counters move. The lines that are always present:

| Line | Meaning |
|---|---|
| `statim_requests_total{code="<status>"}` | Decision-handler responses only. Health, ready, models, metrics, and the playground are not counted. A status appears after the first response with that code. |
| `statim_request_duration_ms_bucket{le="..."}` | End-to-end handler latency in milliseconds. Bucket bounds are 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, and `+Inf`. |
| `statim_request_duration_ms_sum`, `statim_request_duration_ms_count` | Histogram sum and count. |
| `statim_input_tokens_total` | Sum of `usage.input_tokens` on successful decisions. |
| `statim_rejected_busy_total` | Requests rejected with 503. |
| `statim_batch_size_sum`, `statim_batch_size_count` | Summary of states in server-created micro-batch jobs. |
| `statim_batch_wait_ms_sum`, `statim_batch_wait_ms_count` | Summary of admitted-request time waiting for compatible peers. |
| `statim_in_flight` | Requests currently in the handler. |
| `statim_uptime_seconds` | Seconds since the process started listening. |
| `statim_workers_busy{model="<id>"}` | Workers currently running inference for that checkpoint, one line per checkpoint under one `# TYPE statim_workers_busy gauge`. |
| `statim_model_info{model="...",weights="...",version="..."} 1` | One line per checkpoint under one `# TYPE statim_model_info gauge`. `version` is the build version. |

```text
# HELP statim_requests_total Inference requests by HTTP status.
# TYPE statim_requests_total counter
# HELP statim_request_duration_ms End-to-end request latency.
# TYPE statim_request_duration_ms histogram
statim_request_duration_ms_bucket{le="5"} N
statim_request_duration_ms_bucket{le="+Inf"} N
statim_request_duration_ms_sum N
statim_request_duration_ms_count N
# TYPE statim_input_tokens_total counter
statim_input_tokens_total N
# TYPE statim_rejected_busy_total counter
statim_rejected_busy_total N
# HELP statim_batch_size States executed in server-created micro-batches.
# TYPE statim_batch_size summary
statim_batch_size_sum N
statim_batch_size_count N
# HELP statim_batch_wait_ms Time an admitted request waited for compatible peers.
# TYPE statim_batch_wait_ms summary
statim_batch_wait_ms_sum N
statim_batch_wait_ms_count N
# TYPE statim_in_flight gauge
statim_in_flight N
# TYPE statim_uptime_seconds gauge
statim_uptime_seconds N
# TYPE statim_workers_busy gauge
statim_workers_busy{model="english"} N
# TYPE statim_model_info gauge
statim_model_info{model="english",weights="f32",version="0.9.2"} 1
```

Label values are escaped as the text format requires (backslash, double quote and newline), so any `-m` name is safe. With LoRA adapters loaded, `statim_engines`, `statim_adapter_info` and `statim_adapter_bytes` follow; see [LoRA adapters](#lora-adapters).

`N` stands for a live number. The check below reads the endpoint and requires those names.

```python
import requests

response = requests.get("http://127.0.0.1:8080/metrics", timeout=30)
text = response.text
required = [
    "# HELP statim_requests_total Inference requests by HTTP status.",
    "# TYPE statim_requests_total counter",
    "# HELP statim_request_duration_ms End-to-end request latency.",
    "# TYPE statim_request_duration_ms histogram",
    'statim_request_duration_ms_bucket{le="5"}',
    'statim_request_duration_ms_bucket{le="10"}',
    'statim_request_duration_ms_bucket{le="25"}',
    'statim_request_duration_ms_bucket{le="50"}',
    'statim_request_duration_ms_bucket{le="100"}',
    'statim_request_duration_ms_bucket{le="250"}',
    'statim_request_duration_ms_bucket{le="500"}',
    'statim_request_duration_ms_bucket{le="1000"}',
    'statim_request_duration_ms_bucket{le="2500"}',
    'statim_request_duration_ms_bucket{le="5000"}',
    'statim_request_duration_ms_bucket{le="10000"}',
    'statim_request_duration_ms_bucket{le="+Inf"}',
    "statim_request_duration_ms_sum ",
    "statim_request_duration_ms_count ",
    "# TYPE statim_input_tokens_total counter",
    "statim_input_tokens_total ",
    "# TYPE statim_rejected_busy_total counter",
    "statim_rejected_busy_total ",
    "# HELP statim_batch_size States executed in server-created micro-batches.",
    "# TYPE statim_batch_size summary",
    "statim_batch_size_sum ",
    "statim_batch_size_count ",
    "# HELP statim_batch_wait_ms Time an admitted request waited for compatible peers.",
    "# TYPE statim_batch_wait_ms summary",
    "statim_batch_wait_ms_sum ",
    "statim_batch_wait_ms_count ",
    "# TYPE statim_in_flight gauge",
    "statim_in_flight ",
    "# TYPE statim_uptime_seconds gauge",
    "statim_uptime_seconds ",
    "# TYPE statim_workers_busy gauge",
    'statim_workers_busy{model="english"}',
    'statim_workers_busy{model="multilingual"}',
    "# TYPE statim_model_info gauge",
    'statim_model_info{model="english",weights="f32",version="0.9.2"} 1',
    'statim_model_info{model="multilingual",weights="f32",version="0.9.2"} 1',
]
missing = [line for line in required if line not in text]
if response.status_code != 200 or response.headers["Content-Type"] != "text/plain; version=0.0.4":
    missing.append("status-or-content-type")
if missing:
    raise SystemExit("MISSING " + " | ".join(missing))
print(response.headers["Content-Type"])
print(response.status_code)
print("metrics-ok")
```

```text
text/plain; version=0.0.4
200
metrics-ok
```

An unknown path, and the wrong method on a known path, are JSON 404 responses. With authentication enabled, nonpublic paths require a valid key before this route lookup.

```bash
curl -sS -w '\n%{http_code}\n' http://127.0.0.1:8080/no-such
```

```text
{"detail":"HTTP request failed"}
404
```

```bash
curl -sS -w '\n%{http_code}\n' http://127.0.0.1:8080/v1/systemone
```

```text
{"detail":"HTTP request failed"}
404
```

## Models

The ids `english` and `multilingual` are the names passed to `-m`. Automatic routing looks those names up. Any other `-m` names load and appear in `/v1/models`, and a request can select them with `model`, but the English-text heuristic only runs when checkpoints with those two ids are both loaded. Otherwise the first loaded checkpoint is used and `routing.reason` is `default`.

The files used for the samples are converted Laya checkpoints. Their GGUF metadata, and the Hugging Face model cards fetched for this document, say:

| File | GGUF `general.name` | Source | Licence | Encoder |
|---|---|---|---|---|
| `laya-english-f32.gguf` | `laya` | [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) | Apache-2.0 | [answerdotai/ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large), Apache-2.0 |
| `laya-multilingual-f32.gguf` | `laya-multilingual` | [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) | Apache-2.0 | [jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base), MIT |

`general.license` in both GGUF files is `apache-2.0`. `general.source.url` points at the Hugging Face repositories above. Token budgets in those files match the upstream `rl_agent_config.json` documents: English `max_len` 512 and `head_max_len` 192, multilingual `max_len` 1024 and `head_max_len` 256.
