# statim

Official Python client for the [Statim](../../README.md) HTTP API. The package
is named `statim`, uses only the standard library, and targets the API
documented in [`docs/API.md`](../../docs/API.md) (server 0.3.0).

```sh
pip install ./clients/python
```

```python
from statim import Client, YesNoAnswer

client = Client("http://127.0.0.1:8080", timeout=120)
decision = client.decide(
    {
        "subject": "Duplicate charge on invoice #4411",
        "body": "We were billed twice for March. Please refund the duplicate today.",
    },
    {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this request?",
            "criteria": {
                "billing": "invoices, payments, refunds",
                "technical": "bugs and outages",
            },
        },
        "refund": {
            "type": "noul",
            "instructions": "Does the user explicitly request a refund?",
        },
    },
    model="multilingual",
    request_id="example-001",
)
print(decision.answers["department"].choice)
refund = decision.answers["refund"]
assert isinstance(refund, YesNoAnswer)
print(refund.yes, refund.probabilities, refund.confidence)
print(decision.request_id, decision.inference_time_ms)
```

`Client(base_url, api_key=None, timeout=120, *, max_retries=2, backoff=0.5)`.

| Method | Endpoint |
|---|---|
| `decide(state, questions, **options)` | `POST /v1/systemone` |
| `decide_batch(states, questions, **options)` | `POST /v1/systemone/batch` |
| `models()` | `GET /v1/models` |
| `health()` | `GET /health` |
| `ready()` | `GET /ready` |

Options are `model`, `lang`, `ensemble`, `ensemble_margin`, `calibrate`,
`return_logits`, `max_len`, `head_max_len`, and `request_id`. Pass `None` to
omit a field. Question type `noul` is the yes/no question; the parsed object
is `YesNoAnswer`. Its `noul` field is the server probability that the
statement holds, and `probabilities` is `{"yes": noul, "no": 1 - noul}`.

`request_id` is sent as `X-Request-Id`. When omitted, the client generates a
UUID. The id on the result is the one the server echoed. Values outside 1–128
ASCII letters, digits, `.`, `_`, and `-` are replaced by the server.

With an API key, requests send `Authorization: Bearer <key>`. `/health` and
`/ready` never require one.

## Errors and retries

The exception message is the server's `detail` string.

| Status | Exception | `Retry-After` |
|---|---|---|
| 400 | `BadRequestError` | no |
| 401 | `AuthenticationError` | no |
| 413 | `PayloadTooLargeError` | no |
| 422 | `UnprocessableEntityError` | no |
| 503 | `ServiceUnavailableError` (`retry_after` in seconds) | admission saturation sends `1` |

`decide` and `decide_batch` retry `503` only. The delay is the `Retry-After`
value when it is a number of seconds, otherwise `backoff * 2**attempt` capped
at 30 seconds. `max_retries` is the number of extra attempts (default 2).
`ready()` returns `Ready(ready=False)` for `503 {"ready": false}` and does not
retry. Connection failures raise `TransportError` and are not retried.

```python
from statim import ServiceUnavailableError

try:
    client.decide("hello", {"q": {"type": "noul", "instructions": "Greeting?"}})
except ServiceUnavailableError as exc:
    print(exc.detail, exc.retry_after, exc.request_id)
```

## Tests

Start the server, then run pytest from this directory. The suite expects an
unauthenticated server on port 8190 and a second server on port 8191 whose
only key is `sdk-test-key`.

From the repository root, start the unauthenticated server. Start a second
process the same way on port 8191 with `--api-key-file` containing
`sdk-test-key`.

```sh
env -u STATIM_API_KEY ./build/statim serve \
  -m multilingual=models/laya-multilingual-f32.gguf \
  --host 127.0.0.1 --port 8190 --device cpu
```

```sh
python3 -m pytest
```

`STATIM_URL` and `STATIM_AUTH_URL` override the two base URLs.

The client source is Apache-2.0, the same as the engine. Model weights are
covered by `LICENSE-MODEL.md`, not by this package.
