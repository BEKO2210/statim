import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import test from "node:test";

import {
  AuthenticationError,
  BadRequestError,
  Client,
  PayloadTooLargeError,
  ServiceUnavailableError,
  UnprocessableEntityError,
} from "../dist/index.js";

const BASE_URL = process.env.STATIM_URL ?? "http://127.0.0.1:8190";
const AUTH_URL = process.env.STATIM_AUTH_URL ?? "http://127.0.0.1:8191";
const AUTH_KEY = process.env.STATIM_API_KEY_TEST ?? "sdk-test-key-0123456789abcdef0123";
const ADAPTER_URL = process.env.STATIM_ADAPTER_URL;
const ADAPTER_NAME = process.env.STATIM_ADAPTER_NAME;
const AUTO_FAMILIES = new Set([
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
]);

const NOUL_BODY = {
  model: "laya-multilingual",
  answers: {
    refund: {
      type: "noul",
      noul: 0.8,
      confidence: 0.8,
      answer_confidence: 0.8,
      action: { act_probability: 1 },
    },
  },
  usage: { input_tokens: 10, output_tokens: 0 },
  routing: { model: "multilingual", reason: "requested", engine: "statim", weights: "f32" },
};

const QUESTIONS = {
  department: {
    type: "choice",
    instructions: "Which department should handle this request?",
    criteria: { billing: "invoices, payments, refunds", technical: "bugs and outages" },
  },
  urgency: {
    type: "score",
    instructions: "How urgent is this request?",
    criteria: ["not urgent", "soon", "critical"],
  },
  refund: { type: "noul", instructions: "Does the user explicitly request a refund?" },
};

const STATE = {
  subject: "Duplicate charge on invoice #4411",
  body: "We were billed twice for March. Please refund the duplicate today.",
};

function mock(script) {
  const state = { hits: 0, lastBody: "", lastHeaders: {} };
  const server = http.createServer(async (req, res) => {
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    state.hits += 1;
    state.lastBody = Buffer.concat(chunks).toString("utf8");
    state.lastHeaders = req.headers;
    const result = script(state.hits, req, state.lastBody);
    const raw = Buffer.from(result.body);
    res.writeHead(result.status, {
      "Content-Type": "application/json",
      "Content-Length": String(raw.length),
      ...result.headers,
    });
    res.end(raw);
  });
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      resolve({
        url: `http://127.0.0.1:${address.port}`,
        state,
        close: () => new Promise((done) => server.close(() => done())),
      });
    });
  });
}

test("declarations export the OpenAPI answer types", () => {
  const dts = fs.readFileSync(new URL("../dist/index.d.ts", import.meta.url), "utf8");
  for (const name of [
    "ChoiceAnswer",
    "Adapter",
    "ScoreAnswer",
    "YesNoAnswer",
    "NoulAnswer",
    "Decision",
    "SystemOneRequest",
    "BatchRequest",
    "ServiceUnavailableError",
    "Client",
  ]) {
    assert.match(dts, new RegExp(`\\b${name}\\b`));
  }
});

test("constructor rejects a bad URL", () => {
  assert.throws(() => new Client("not a url"), TypeError);
  assert.throws(() => new Client("http://127.0.0.1:1", null, 0), TypeError);
  const client = new Client("http://127.0.0.1:9", "", 1);
  assert.equal(client.apiKey, null);
});

test("yes/no probabilities and option encoding", async () => {
  const server = await mock((hit, req, body) => {
    assert.equal(hit, 1);
    assert.equal(req.headers["x-request-id"], "opt-1");
    assert.equal(req.headers.authorization, "Bearer sdk-unit-key-0123456789abcdef0123");
    assert.deepEqual(JSON.parse(body), {
      state: "hello",
      questions: { refund: { type: "noul", instructions: "Refund?" } },
      calibrate: false,
      ensemble: 1,
    });
    return {
      status: 200,
      headers: { "X-Request-Id": "opt-1", "X-Inference-Time-Ms": "1.50" },
      body: JSON.stringify(NOUL_BODY),
    };
  });
  try {
    const decision = await new Client(server.url, "sdk-unit-key-0123456789abcdef0123", 5, { max_retries: 0 }).decide(
      "hello",
      { refund: { type: "noul", instructions: "Refund?" } },
      { calibrate: false, ensemble: 1, request_id: "opt-1" },
    );
    const answer = decision.answers.refund;
    assert.equal(answer.type, "noul");
    assert.equal(answer.noul, 0.8);
    assert.equal(answer.yes, true);
    assert.equal(answer.probabilities.yes, 0.8);
    assert.ok(Math.abs(answer.probabilities.no - 0.2) < 1e-12);
    assert.equal(answer.escalate, undefined);
    assert.equal(decision.request_id, "opt-1");
    assert.equal(decision.inference_time_ms, 1.5);
  } finally {
    await server.close();
  }
});

test("selective prediction transport serialization and parsing", async () => {
  const requests = [];
  const originalFetch = globalThis.fetch;
  const selectiveBody = {
    ...NOUL_BODY,
    answers: {
      refund: { ...NOUL_BODY.answers.refund, escalate: true },
      topic: {
        type: "choice",
        choice: "billing",
        probabilities: { billing: 0.9, other: 0.1 },
        confidence: 0.53,
        answer_confidence: 0.9,
        action: { act_probability: 1 },
        escalate: false,
      },
      urgency: {
        type: "score",
        score: 0.2,
        legend: { 0: "low", 1: "high" },
        probabilities: { 0: 0.8, 1: 0.2 },
        confidence: 0.28,
        answer_confidence: 0.8,
        action: { act_probability: 1 },
        escalate: true,
      },
    },
  };
  globalThis.fetch = async (url, init) => {
    const body = JSON.parse(init.body);
    requests.push([String(url), body]);
    const payload = String(url).endsWith("/batch")
      ? { results: [selectiveBody] }
      : "min_confidence" in body
        ? selectiveBody
        : NOUL_BODY;
    return new Response(JSON.stringify(payload), {
      status: 200,
      headers: { "Content-Type": "application/json", "X-Request-Id": "selective" },
    });
  };
  try {
    const client = new Client("http://statim.invalid", null, 5, { max_retries: 0 });
    const questions = { refund: { type: "noul", instructions: "Refund?" } };
    const decision = await client.decide("hello", questions, { min_confidence: 0.85 });
    const batch = await client.decide_batch(["hello"], questions, { min_confidence: 0.75 });
    const unchanged = await client.decide("hello", questions);
    assert.deepEqual(requests, [
      ["http://statim.invalid/v1/systemone", { state: "hello", questions, min_confidence: 0.85 }],
      ["http://statim.invalid/v1/systemone/batch", { states: ["hello"], questions, min_confidence: 0.75 }],
      ["http://statim.invalid/v1/systemone", { state: "hello", questions }],
    ]);
    assert.equal(decision.answers.refund.escalate, true);
    assert.equal(decision.answers.topic.escalate, false);
    assert.equal(decision.answers.urgency.escalate, true);
    assert.equal(batch.results[0].answers.refund.escalate, true);
    assert.equal(unchanged.answers.refund.escalate, undefined);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("adapter transport and response parsing", async () => {
  const requests = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    const body = JSON.parse(init.body);
    requests.push([String(url), body]);
    const routing = { ...NOUL_BODY.routing, adapter: null, adapter_reason: "none" };
    if (body.adapter === "emotion") {
      Object.assign(routing, { reason: "adapter", adapter: "emotion", adapter_reason: "requested" });
    }
    if (body.adapter === "auto") {
      const category = Object.keys(body.questions)[0];
      if (category === "emotion") Object.assign(routing, { adapter: "emotion", adapter_reason: "auto:emotion" });
      else if (category === "no_family") Object.assign(routing, { adapter_reason: "auto:no-family" });
      else Object.assign(routing, { adapter_reason: "auto:emotion:no-adapter" });
    }
    const payload = { ...NOUL_BODY, routing };
    return new Response(JSON.stringify(String(url).endsWith("/batch") ? { results: [payload] } : payload));
  };
  try {
    const client = new Client("http://statim.invalid", null, 5, { max_retries: 0 });
    const questions = { emotion: { type: "noul", instructions: "Emotion?" } };
    const selected = await client.decide("hello", questions, { adapter: "emotion" });
    const batch = await client.decide_batch(["hello"], questions, { adapter: "auto" });
    const omitted = await client.decide("hello", questions, { adapter: null });
    const noFamilyQuestions = { no_family: { type: "noul", instructions: "Unknown?" } };
    const noFamily = await client.decide("hello", noFamilyQuestions, { adapter: "auto" });
    const noAdapterQuestions = { emotion_missing: { type: "noul", instructions: "Emotion?" } };
    const noAdapter = await client.decide("hello", noAdapterQuestions, { adapter: "auto" });
    const base = await client.decide("hello", questions, { adapter: "none" });
    const longName = "é".repeat(129);
    const longAdapter = await client.decide("hello", questions, { adapter: longName });
    assert.deepEqual(requests, [
      ["http://statim.invalid/v1/systemone", { state: "hello", questions, adapter: "emotion" }],
      ["http://statim.invalid/v1/systemone/batch", { states: ["hello"], questions, adapter: "auto" }],
      ["http://statim.invalid/v1/systemone", { state: "hello", questions }],
      ["http://statim.invalid/v1/systemone", { state: "hello", questions: noFamilyQuestions, adapter: "auto" }],
      ["http://statim.invalid/v1/systemone", { state: "hello", questions: noAdapterQuestions, adapter: "auto" }],
      ["http://statim.invalid/v1/systemone", { state: "hello", questions, adapter: "none" }],
      ["http://statim.invalid/v1/systemone", { state: "hello", questions, adapter: longName }],
    ]);
    assert.equal(selected.routing.adapter, "emotion");
    assert.equal(selected.routing.adapter_reason, "requested");
    assert.equal(selected.routing.reason, "adapter");
    assert.equal(batch.results[0].routing.adapter, "emotion");
    assert.equal(batch.results[0].routing.adapter_reason, "auto:emotion");
    assert.equal(omitted.routing.adapter, null);
    assert.equal(omitted.routing.adapter_reason, "none");
    assert.equal(noFamily.routing.adapter, null);
    assert.equal(noFamily.routing.adapter_reason, "auto:no-family");
    assert.equal(noAdapter.routing.adapter, null);
    assert.equal(noAdapter.routing.adapter_reason, "auto:emotion:no-adapter");
    assert.equal(base.routing.adapter, null);
    assert.equal(base.routing.adapter_reason, "none");
    assert.equal(longAdapter.routing.adapter, null);
    await assert.rejects(() => client.decide("hello", questions, { adapter: 12 }), TypeError);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("models parse adapters and reject malformed adapters", async () => {
  const model = {
    id: "multilingual",
    object: "model",
    owned_by: "statim",
    source: "laya-multilingual",
    weights: "q8_0",
    layers: 22,
    hidden: 768,
    max_len: 1024,
    vocab: 250002,
    device: "cpu",
  };
  const adapter = {
    id: "emotion",
    source: "emotion-lora",
    mode: "runtime",
    rank: 4,
    alpha: 8,
    pairs: 88,
    pairs_applied: 87,
    categories: ["emotion", "sentiment"],
    bytes: 3456,
  };
  const adapterWithoutId = Object.fromEntries(Object.entries(adapter).filter(([key]) => key !== "id"));
  const payloads = [
    { object: "list", data: [model, { ...model, id: "with-adapter", adapters: [adapter] }] },
    { object: "list", data: [{ ...model, adapters: null }] },
    { object: "list", data: [{ ...model, adapters: { bad: true } }] },
    { object: "list", data: [{ ...model, adapters: [adapterWithoutId] }] },
    { object: "list", data: [{ ...model, adapters: [{ ...adapter, rank: "4" }] }] },
  ];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response(JSON.stringify(payloads.shift()));
  try {
    const client = new Client("http://statim.invalid", null, 5, { max_retries: 0 });
    const listed = await client.models();
    assert.deepEqual(listed.data[0].adapters, []);
    assert.deepEqual(listed.data[1].adapters[0], adapter);
    await assert.rejects(() => client.models(), /adapters must be an array/);
    await assert.rejects(() => client.models(), /adapters must be an array/);
    await assert.rejects(() => client.models(), /missing id/);
    await assert.rejects(() => client.models(), /adapter.rank must be an integer/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("retries honor Retry-After 0 and then succeed", async () => {
  const server = await mock((hit, req) => {
    const echoed = req.headers["x-request-id"];
    if (hit < 3) {
      return {
        status: 503,
        headers: { "Retry-After": "0", "X-Request-Id": echoed },
        body: JSON.stringify({ detail: "server busy, try again later" }),
      };
    }
    return {
      status: 200,
      headers: { "X-Request-Id": echoed, "X-Inference-Time-Ms": "3.00" },
      body: JSON.stringify({
        ...NOUL_BODY,
        answers: { q: NOUL_BODY.answers.refund },
      }),
    };
  });
  try {
    const started = Date.now();
    const decision = await new Client(server.url, null, 5, { max_retries: 2, backoff: 30 }).decide(
      "x",
      { q: { type: "noul", instructions: "?" } },
      { request_id: "retry-ok" },
    );
    assert.ok(Date.now() - started < 2000);
    assert.equal(server.state.hits, 3);
    assert.equal(decision.request_id, "retry-ok");
    assert.equal(decision.answers.q.type, "noul");
  } finally {
    await server.close();
  }
});

test("Retry-After seconds are slept and exhaustion raises", async () => {
  const server = await mock(() => ({
    status: 503,
    headers: { "Retry-After": "1", "X-Request-Id": "busy" },
    body: JSON.stringify({ detail: "server busy, try again later" }),
  }));
  try {
    const started = Date.now();
    await assert.rejects(
      () =>
        new Client(server.url, null, 5, { max_retries: 1, backoff: 30 }).decide(
          "x",
          { q: { type: "noul", instructions: "?" } },
          { request_id: "busy-call" },
        ),
      (err) => {
        assert.ok(err instanceof ServiceUnavailableError);
        assert.equal(err.message, "server busy, try again later");
        assert.equal(err.detail, "server busy, try again later");
        assert.equal(err.status, 503);
        assert.equal(err.retry_after, 1);
        assert.equal(err.request_id, "busy");
        return true;
      },
    );
    const elapsed = Date.now() - started;
    assert.ok(elapsed >= 800 && elapsed < 5000);
    assert.equal(server.state.hits, 2);
  } finally {
    await server.close();
  }
});

test("backoff runs when Retry-After is absent", async () => {
  const server = await mock((hit, req) => {
    if (hit === 1) {
      return { status: 503, headers: { "X-Request-Id": "b" }, body: '{"detail":"server busy, try again later"}' };
    }
    return {
      status: 200,
      headers: { "X-Request-Id": req.headers["x-request-id"], "X-Inference-Time-Ms": "0.10" },
      body: JSON.stringify({ ...NOUL_BODY, answers: { q: NOUL_BODY.answers.refund } }),
    };
  });
  try {
    await new Client(server.url, null, 5, { max_retries: 2, backoff: 0 }).decide("x", {
      q: { type: "noul", instructions: "?" },
    });
    assert.equal(server.state.hits, 2);
  } finally {
    await server.close();
  }
});

test("non-503 is not retried", async () => {
  const server = await mock(() => ({
    status: 400,
    headers: { "X-Request-Id": "once" },
    body: JSON.stringify({ detail: "'state' is required" }),
  }));
  try {
    await assert.rejects(
      () => new Client(server.url, null, 5, { max_retries: 5 }).decide(null, {}),
      (err) => {
        assert.ok(err instanceof BadRequestError);
        assert.equal(err.request_id, "once");
        assert.equal(err.detail, "'state' is required");
        return true;
      },
    );
    assert.equal(server.state.hits, 1);
  } finally {
    await server.close();
  }
});

test("ready 503 is not retried", async () => {
  const server = await mock(() => ({ status: 503, headers: {}, body: '{"ready":false}' }));
  try {
    const ready = await new Client(server.url, null, 5, { max_retries: 5 }).ready();
    assert.equal(ready.ready, false);
    assert.equal(server.state.hits, 1);
  } finally {
    await server.close();
  }
});

test("status mapping", async () => {
  const cases = [
    [401, AuthenticationError, "invalid or missing bearer token"],
    [413, PayloadTooLargeError, "too many questions"],
    [422, UnprocessableEntityError, "ensemble must be an integer between 1 and 8"],
  ];
  const server = await mock((hit) => {
    const [status, , message] = cases[hit - 1];
    return { status, headers: {}, body: JSON.stringify({ detail: message }) };
  });
  try {
    const client = new Client(server.url, null, 5, { max_retries: 3 });
    for (const [status, cls, message] of cases) {
      await assert.rejects(
        () => client.models(),
        (err) => {
          assert.ok(err instanceof cls);
          assert.equal(err.status, status);
          assert.equal(err.detail, message);
          assert.equal(err.message, message);
          assert.equal(err.request_id, null);
          return true;
        },
      );
    }
  } finally {
    await server.close();
  }
});

test("unexpected option", async () => {
  const client = new Client("http://127.0.0.1:9", null, 1);
  await assert.rejects(
    () => client.decide("x", {}, { not_a_field: true }),
    (err) => err instanceof TypeError && /unexpected option/.test(err.message),
  );
});

test("health, ready, and models on the real server", async () => {
  const client = new Client(BASE_URL, null, 10);
  const health = await client.health();
  assert.equal(health.status, "ok");
  assert.match(health.version, /^\d+\.\d+\.\d+$/);
  assert.equal((await client.ready()).ready, true);
  const listed = await client.models();
  assert.equal(listed.object, "list");
  assert.equal(listed.data.length, 1);
  const model = listed.data[0];
  assert.equal(model.id, "multilingual");
  assert.equal(model.object, "model");
  assert.equal(model.owned_by, "statim");
  assert.equal(model.device, "cpu");
  assert.equal(model.weights, "f32");
  assert.equal(model.max_len, 1024);
  assert.ok(model.layers > 0 && model.hidden > 0 && model.vocab > 0);
  const keyed = new Client(BASE_URL, "unused-because-auth-is-off", 10);
  assert.equal((await keyed.models()).data[0].id, "multilingual");
});

test("handler errors on the real server", async () => {
  const client = new Client(BASE_URL, null, 30);
  await assert.rejects(
    () => client.decide(null, {}, { request_id: "err-400" }),
    (err) => {
      assert.ok(err instanceof BadRequestError);
      assert.equal(err.status, 400);
      assert.equal(err.detail, "'state' is required");
      assert.equal(err.request_id, "err-400");
      return true;
    },
  );
  await assert.rejects(
    () => client.decide(null, {}, { request_id: "bad id" }),
    (err) => {
      assert.ok(err instanceof BadRequestError);
      assert.equal(err.request_id?.length, 16);
      assert.notEqual(err.request_id, "bad id");
      return true;
    },
  );
  await assert.rejects(
    () =>
      client.decide("x", { q: { type: "noul", instructions: "?" } }, { ensemble: 0, request_id: "err-422" }),
    (err) => {
      assert.ok(err instanceof UnprocessableEntityError);
      assert.equal(err.detail, "ensemble must be an integer between 1 and 8");
      assert.equal(err.request_id, "err-422");
      return true;
    },
  );
  await assert.rejects(
    () => client.decide("x", { q: { type: "maybe", instructions: "?" } }),
    (err) => {
      assert.ok(err instanceof UnprocessableEntityError);
      assert.equal(err.detail, "unknown question type; use choice, score or noul");
      return true;
    },
  );
  const tooMany = {};
  for (let i = 0; i < 65; i += 1) tooMany[`q${i}`] = { type: "noul", instructions: "?" };
  await assert.rejects(
    () => client.decide("x", tooMany, { request_id: "err-413" }),
    (err) => {
      assert.ok(err instanceof PayloadTooLargeError);
      assert.equal(err.status, 413);
      assert.ok(err.detail.startsWith("too many questions"));
      assert.equal(err.request_id, "err-413");
      return true;
    },
  );
});

test("empty batch", async () => {
  const batch = await new Client(BASE_URL, null, 60).decide_batch([], {}, { request_id: "empty-batch-js" });
  assert.deepEqual(batch.results, []);
  assert.equal(batch.request_id, "empty-batch-js");
  assert.ok(batch.inference_time_ms !== null && batch.inference_time_ms >= 0);
});

test("decide and batch against the server", async () => {
  const client = new Client(BASE_URL, null, 180);
  const decision = await client.decide(STATE, QUESTIONS, {
    model: "multilingual",
    return_logits: true,
    request_id: "live-decide-js",
  });
  assert.ok(decision.model);
  assert.equal(decision.request_id, "live-decide-js");
  assert.ok(decision.inference_time_ms > 0);
  assert.equal(decision.usage.output_tokens, 0);
  assert.ok(decision.usage.input_tokens > 0);
  assert.equal(decision.routing.model, "multilingual");
  assert.equal(decision.routing.reason, "requested");
  assert.equal(decision.routing.engine, "statim");
  assert.equal(decision.routing.weights, "f32");

  const department = decision.answers.department;
  assert.equal(department.type, "choice");
  assert.ok(department.choice === "billing" || department.choice === "technical");
  const sum = Object.values(department.probabilities).reduce((acc, value) => acc + value, 0);
  assert.ok(Math.abs(sum - 1) < 0.02);
  assert.ok(department.confidence >= 0 && department.confidence <= 1);
  assert.equal(department.logits.length, Object.keys(department.probabilities).length);

  const urgency = decision.answers.urgency;
  assert.equal(urgency.type, "score");
  assert.equal(typeof urgency.score, "number");
  assert.deepEqual(Object.keys(urgency.probabilities).sort(), ["0", "1", "2"]);
  assert.equal(urgency.legend["0"], "not urgent");
  assert.equal(urgency.logits.length, 3);

  const refund = decision.answers.refund;
  assert.equal(refund.type, "noul");
  assert.ok(refund.noul >= 0 && refund.noul <= 1);
  assert.equal(refund.probabilities.yes, refund.noul);
  assert.ok(Math.abs(refund.probabilities.no - (1 - refund.noul)) < 1e-12);
  assert.equal(refund.yes, refund.noul >= 0.5);
  assert.equal(refund.logits.length, 2);

  const batch = await client.decide_batch(
    ["Please refund the duplicate charge today.", "The login page returns a 500 error."],
    {
      topic: {
        type: "choice",
        instructions: "What is this about?",
        criteria: { billing: "payments and refunds", technical: "bugs and outages" },
      },
    },
    { model: "multilingual", request_id: "live-batch-js" },
  );
  assert.equal(batch.request_id, "live-batch-js");
  assert.equal(batch.results.length, 2);
  for (const item of batch.results) {
    assert.equal(item.request_id, "live-batch-js");
    assert.equal(item.routing.reason, "requested");
    const topic = item.answers.topic;
    assert.equal(topic.type, "choice");
    assert.equal(topic.logits, undefined);
    assert.ok(topic.choice === "billing" || topic.choice === "technical");
  }
});

test("authentication on the keyed server", async () => {
  const missing = new Client(AUTH_URL, null, 30);
  await assert.rejects(
    () => missing.models(),
    (err) => {
      assert.ok(err instanceof AuthenticationError);
      assert.equal(err.status, 401);
      assert.equal(err.detail, "invalid or missing bearer token");
      assert.equal(err.message, "invalid or missing bearer token");
      assert.equal(err.request_id, null);
      return true;
    },
  );
  const wrong = new Client(AUTH_URL, "wrong-sdk-key-0123456789abcdef01", 30);
  await assert.rejects(
    () =>
      wrong.decide(
        "hello",
        { q: { type: "noul", instructions: "Is this a greeting?" } },
        { request_id: "auth-decide" },
      ),
    (err) => {
      assert.ok(err instanceof AuthenticationError);
      assert.equal(err.detail, "invalid or missing bearer token");
      assert.equal(err.request_id, null);
      return true;
    },
  );
  const authed = new Client(AUTH_URL, AUTH_KEY, 30);
  const listed = await authed.models();
  assert.equal(listed.data[0].id, "multilingual");
  assert.equal(listed.data[0].device, "cpu");
  assert.equal((await missing.health()).status, "ok");
  assert.equal((await missing.ready()).ready, true);
});

test(
  "adapter against the server",
  { skip: !ADAPTER_URL || !ADAPTER_NAME },
  async (t) => {
    const client = new Client(ADAPTER_URL, null, 180);
    const listed = await client.models();
    const match = listed.data
      .flatMap((model) => model.adapters.map((adapter) => ({ model, adapter })))
      .find(({ adapter }) => adapter.id === ADAPTER_NAME);
    assert.ok(match, `adapter ${ADAPTER_NAME} is not listed`);
    assert.ok(match.adapter.categories.length > 0, `adapter ${ADAPTER_NAME} has no categories`);
    const category = match.adapter.categories[0];
    const questions = { [category]: { type: "noul", instructions: `Is this about ${category}?` } };

    const requested = await client.decide("adapter live test", questions, {
      model: match.model.id,
      adapter: ADAPTER_NAME,
    });
    assert.equal(requested.routing.adapter, ADAPTER_NAME);
    assert.equal(requested.routing.adapter_reason, "requested");

    const base = await client.decide("adapter live test", questions, { model: match.model.id, adapter: "none" });
    assert.equal(base.routing.adapter, null);
    assert.equal(base.routing.adapter_reason, "none");

    await t.test(
      "auto routes a documented family",
      { skip: AUTO_FAMILIES.has(category) ? false : `category ${category} is not one of the 14 auto-rule families` },
      async () => {
        const automatic = await client.decide("adapter live test", questions, {
          model: match.model.id,
          adapter: "auto",
        });
        assert.equal(automatic.routing.adapter, ADAPTER_NAME);
        assert.equal(automatic.routing.adapter_reason, `auto:${category}`);
      },
    );
  },
);
