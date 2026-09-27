/** HTTP errors from the Statim API. The message is the `detail` string. */

export class StatimError extends Error {
  readonly status: number | null;
  readonly detail: string | null;
  /** Echoed `X-Request-Id`, or null when the server did not send one. */
  readonly request_id: string | null;

  constructor(
    message: string,
    options: { status?: number | null; detail?: string | null; request_id?: string | null } = {},
  ) {
    super(message);
    this.name = new.target.name;
    this.status = options.status ?? null;
    this.detail = options.detail ?? null;
    this.request_id = options.request_id ?? null;
  }
}

export class TransportError extends StatimError {
  /** DNS, connection, or timeout failure before an HTTP status. */
}

export class BadRequestError extends StatimError {
  /** HTTP 400. Malformed JSON or a missing required field. */
}

export class AuthenticationError extends StatimError {
  /** HTTP 401. Missing or wrong bearer token. */
}

export class PayloadTooLargeError extends StatimError {
  /** HTTP 413. Body size or a request limit was exceeded. */
}

export class UnprocessableEntityError extends StatimError {
  /** HTTP 422. A question, budget, or routing field was rejected. */
}

export class ServiceUnavailableError extends StatimError {
  /**
   * HTTP 503 from admission control or the engine queue.
   * `retry_after` is the `Retry-After` header in seconds, when present.
   * `decide` and `decide_batch` retry these before raising.
   */
  readonly retry_after: number | null;

  constructor(
    message: string,
    options: {
      status?: number | null;
      detail?: string | null;
      request_id?: string | null;
      retry_after?: number | null;
    } = {},
  ) {
    super(message, options);
    this.retry_after = options.retry_after ?? null;
  }
}
