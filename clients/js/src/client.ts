/**
 * Fetch client for the Statim HTTP API. No runtime dependencies.
 * `503` from `decide` and `decide_batch` is retried; every other status fails
 * immediately. A numeric `Retry-After` replaces the exponential backoff.
 */

import {
  AuthenticationError,
  BadRequestError,
  PayloadTooLargeError,
  ServiceUnavailableError,
  StatimError,
  TransportError,
  UnprocessableEntityError,
} from "./errors.js";
import { version } from "./version.js";
import {
  assertDecideOptions,
  bodyOptions,
  parseBatch,
  parseDecision,
  parseHealth,
  parseModelList,
  parseReady,
} from "./types.js";
import type {
  BatchResult,
  DecideOptions,
  Decision,
  Health,
  ModelList,
  Question,
  Ready,
} from "./types.js";

const BACKOFF_CAP_SECONDS = 30;

export interface ClientOptions {
  /** Extra attempts after a 503 from decide or decide_batch. Default 2. */
  max_retries?: number;
  /** Base delay in seconds when a 503 has no numeric Retry-After. Default 0.5. */
  backoff?: number;
}

interface RawResponse {
  status: number;
  headers: Headers;
  text: string;
  payload: unknown;
}

function header(headers: Headers, name: string): string | null {
  const value = headers.get(name);
  if (value == null || value === "") return null;
  return value;
}

function parseRetryAfter(value: string | null): number | null {
  if (value == null) return null;
  const text = value.trim();
  if (!text) return null;
  const seconds = Number(text);
  if (!Number.isFinite(seconds) || seconds < 0) return null;
  return seconds;
}

function detailOf(payload: unknown): string | null {
  if (typeof payload === "object" && payload !== null && !Array.isArray(payload)) {
    const detail = (payload as Record<string, unknown>).detail;
    if (typeof detail === "string") return detail;
  }
  return null;
}

function newRequestId(): string {
  return crypto.randomUUID();
}

function sleep(seconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, seconds * 1000));
}

function errorFrom(response: RawResponse): StatimError {
  const detail = detailOf(response.payload);
  const message = detail ?? (response.text.trim().slice(0, 500) || `HTTP ${response.status}`);
  const request_id = header(response.headers, "x-request-id");
  const options = { status: response.status, detail, request_id };
  switch (response.status) {
    case 400:
      return new BadRequestError(message, options);
    case 401:
      return new AuthenticationError(message, options);
    case 413:
      return new PayloadTooLargeError(message, options);
    case 422:
      return new UnprocessableEntityError(message, options);
    case 503:
      return new ServiceUnavailableError(message, {
        ...options,
        retry_after: parseRetryAfter(header(response.headers, "retry-after")),
      });
    default:
      return new StatimError(message, options);
  }
}

export class Client {
  readonly baseUrl: string;
  readonly apiKey: string | null;
  readonly timeout: number;
  readonly max_retries: number;
  readonly backoff: number;

  /**
   * @param baseUrl Absolute http or https URL. A trailing slash is removed.
   * @param apiKey Bearer token. Null or "" sends no Authorization header.
   * @param timeout Socket timeout in seconds. Default 120.
   */
  constructor(baseUrl: string, apiKey: string | null = null, timeout = 120, options: ClientOptions = {}) {
    let parsed: URL;
    try {
      parsed = new URL(baseUrl);
    } catch {
      throw new TypeError("base_url must be an absolute http or https URL");
    }
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      throw new TypeError("base_url must be an absolute http or https URL");
    }
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.apiKey = apiKey ? apiKey : null;
    if (typeof timeout !== "number" || !Number.isFinite(timeout) || timeout <= 0) {
      throw new TypeError("timeout must be a positive number of seconds");
    }
    this.timeout = timeout;
    const maxRetries = options.max_retries ?? 2;
    if (!Number.isInteger(maxRetries) || maxRetries < 0) {
      throw new TypeError("max_retries must be an integer >= 0");
    }
    this.max_retries = maxRetries;
    const backoff = options.backoff ?? 0.5;
    if (typeof backoff !== "number" || !Number.isFinite(backoff) || backoff < 0) {
      throw new TypeError("backoff must be a number >= 0");
    }
    this.backoff = backoff;
  }

  /** Score one state. See {@link DecideOptions}. */
  async decide(state: unknown, questions: Record<string, Question>, options: DecideOptions = {}): Promise<Decision> {
    assertDecideOptions(options);
    const response = await this.#request(
      "POST",
      "/v1/systemone",
      { state, questions, ...bodyOptions(options) },
      { retryOn503: true, requestId: options.request_id },
    );
    if (response.status !== 200) throw errorFrom(response);
    return attach(parseDecision(response.payload), response);
  }

  /** Score many states with one question set. Results keep `states` order. */
  async decide_batch(
    states: readonly unknown[],
    questions: Record<string, Question>,
    options: DecideOptions = {},
  ): Promise<BatchResult> {
    assertDecideOptions(options);
    if (!Array.isArray(states)) throw new TypeError("states must be a list");
    const response = await this.#request(
      "POST",
      "/v1/systemone/batch",
      { states: [...states], questions, ...bodyOptions(options) },
      { retryOn503: true, requestId: options.request_id },
    );
    if (response.status !== 200) throw errorFrom(response);
    const parsed = parseBatch(response.payload);
    const request_id = header(response.headers, "x-request-id");
    const inference_time_ms = inferenceMs(response.headers);
    return {
      results: parsed.results.map((item) => ({ ...item, request_id, inference_time_ms })),
      request_id,
      inference_time_ms,
    };
  }

  /** Loaded checkpoints, in load order. */
  async models(): Promise<ModelList> {
    const response = await this.#request("GET", "/v1/models", undefined, { retryOn503: false });
    if (response.status !== 200) throw errorFrom(response);
    return parseModelList(response.payload);
  }

  /** Liveness. Unauthenticated. */
  async health(): Promise<Health> {
    const response = await this.#request("GET", "/health", undefined, { retryOn503: false });
    if (response.status !== 200) throw errorFrom(response);
    return parseHealth(response.payload);
  }

  /** Readiness. `503 {"ready": false}` returns `{ready: false}` and is not retried. */
  async ready(): Promise<Ready> {
    const response = await this.#request("GET", "/ready", undefined, { retryOn503: false });
    if (
      (response.status === 200 || response.status === 503) &&
      typeof response.payload === "object" &&
      response.payload !== null &&
      "ready" in response.payload
    ) {
      return parseReady(response.payload);
    }
    throw errorFrom(response);
  }

  async #request(
    method: string,
    path: string,
    body: Record<string, unknown> | undefined,
    call: { retryOn503: boolean; requestId?: string },
  ): Promise<RawResponse> {
    const headers: Record<string, string> = {
      Accept: "application/json",
      "User-Agent": `statim-js/${version}`,
      "X-Request-Id": call.requestId ?? newRequestId(),
    };
    if (this.apiKey) headers.Authorization = `Bearer ${this.apiKey}`;
    let payload: string | undefined;
    if (body !== undefined) {
      payload = JSON.stringify(body);
      headers["Content-Type"] = "application/json";
    }
    let attempt = 0;
    while (true) {
      let response: Response;
      try {
        response = await fetch(this.baseUrl + path, {
          method,
          headers,
          body: payload,
          signal: AbortSignal.timeout(Math.max(1, Math.ceil(this.timeout * 1000))),
        });
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        throw new TransportError(`request failed: ${message}`);
      }
      const text = await response.text();
      const raw: RawResponse = { status: response.status, headers: response.headers, text, payload: decodeJson(text) };
      if (call.retryOn503 && raw.status === 503 && attempt < this.max_retries) {
        const retryAfter = parseRetryAfter(header(raw.headers, "retry-after"));
        const delay = retryAfter ?? Math.min(this.backoff * 2 ** attempt, BACKOFF_CAP_SECONDS);
        await sleep(delay);
        attempt += 1;
        continue;
      }
      if (raw.status < 400 && (raw.payload === null || typeof raw.payload !== "object")) {
        throw new StatimError("response body is not valid JSON", {
          status: raw.status,
          request_id: header(raw.headers, "x-request-id"),
        });
      }
      return raw;
    }
  }
}

function attach(decision: Decision, response: RawResponse): Decision {
  return {
    ...decision,
    request_id: header(response.headers, "x-request-id"),
    inference_time_ms: inferenceMs(response.headers),
  };
}

function inferenceMs(headers: Headers): number | null {
  const raw = header(headers, "x-inference-time-ms");
  if (raw == null) return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

function decodeJson(text: string): unknown {
  if (!text) return null;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return null;
  }
}
