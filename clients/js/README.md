# @statim/client

Official TypeScript client for the [Statim](../../README.md) HTTP API. ESM,
`fetch`, no runtime dependencies. Declaration files come from `tsc` and follow
the schemas in [`docs/openapi.yaml`](../../docs/openapi.yaml). The package
targets the same API as server 0.8.7.

```sh
npm install
npx tsc -p tsconfig.json
```

```js
import { Client } from "@statim/client";

const client = new Client("http://127.0.0.1:8080", null, 120);
const decision = await client.decide(
  {
    subject: "Duplicate charge on invoice #4411",
    body: "We were billed twice for March. Please refund the duplicate today.",
  },
  {
    department: {
      type: "choice",
      instructions: "Which department should handle this request?",
      criteria: {
        billing: "invoices, payments, refunds",
        technical: "bugs and outages",
      },
    },
    refund: {
      type: "noul",
      instructions: "Does the user explicitly request a refund?",
    },
  },
  { model: "multilingual", request_id: "example-001" },
);
console.log(decision.answers.department.choice);
const refund = decision.answers.refund;
if (refund.type === "noul") {
  console.log(refund.yes, refund.probabilities, refund.confidence);
}
console.log(decision.request_id, decision.inference_time_ms);
```

`new Client(baseUrl, apiKey = null, timeout = 120, { max_retries: 2, backoff: 0.5 })`.

| Method | Endpoint |
|---|---|
| `decide(state, questions, options)` | `POST /v1/systemone` |
| `decide_batch(states, questions, options)` | `POST /v1/systemone/batch` |
| `models()` | `GET /v1/models` |
| `health()` | `GET /health` |
| `ready()` | `GET /ready` |

Options match the Python client: `model`, `adapter`, `lang`, `ensemble`, `ensemble_margin`,
`calibrate`, `return_logits`, `min_confidence`, `max_len`, `head_max_len`, `request_id`. `null`
and `undefined` omit a field; request base weights explicitly with
`{ adapter: "none" }`. Question type `noul` is returned as `YesNoAnswer` (`type` stays `"noul"`). `probabilities.yes` is the server's
`noul` value and `probabilities.no` is `1 - noul`.
Every answer type has optional `escalate`, present only when a positive
confidence threshold applied to the response.

Select a LoRA adapter by name, or use `{ adapter: "auto" }` to match the
question family. A name needs a server started with that adapter, for example
`--adapter multilingual:emotion=emotion.lora.gguf`. `"auto"` works on any
server and falls back to the base weights.

```js
const result = await client.decide(state, questions, { adapter: "auto" });
console.log(result.routing.adapter, result.routing.adapter_reason); // e.g. emotion auto:emotion
console.log((await client.models()).data.flatMap((m) => m.adapters.map((a) => a.id)));
```

`request_id` is sent as `X-Request-Id`. When omitted, the client generates a
UUID. The id on the result is the one the server echoed.

A non-empty `apiKey` sends `Authorization: Bearer <key>`. `/health` and
`/ready` never require one.

## Errors and retries

The exception message is the server's `detail` string.

| Status | Exception |
|---|---|
| 400 | `BadRequestError` |
| 401 | `AuthenticationError` |
| 413 | `PayloadTooLargeError` |
| 422 | `UnprocessableEntityError` |
| 503 | `ServiceUnavailableError` (`retry_after` seconds) |

`decide` and `decide_batch` retry `503` only. The delay is numeric
`Retry-After` when present, otherwise `backoff * 2**attempt` capped at 30
seconds. `max_retries` counts extra attempts. `ready()` returns
`{ ready: false }` for `503 {"ready": false}` and does not retry. Connection
and timeout failures raise `TransportError`.

## Tests

Build, start the same two servers as the Python suite (ports 8190 and 8191),
then:

```sh
npx tsc -p tsconfig.json
node --test test/client.test.js
```

`STATIM_URL` and `STATIM_AUTH_URL` override the base URLs. `STATIM_API_KEY_TEST`
overrides the key for port 8191 (default `sdk-test-key`).

Source is Apache-2.0, the same licence as the engine.
