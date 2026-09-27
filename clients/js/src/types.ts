/**
 * Types for the Statim HTTP API.
 *
 * Wire members follow `docs/openapi.yaml` `components.schemas`
 * (Error, Action, Usage, Routing, ChoiceAnswer, ScoreAnswer, NoulAnswer,
 * Decision, Question, SystemOneRequest, BatchRequest, BatchResponse, Health,
 * Ready, Model, ModelList). `YesNoAnswer` is the SDK view of `NoulAnswer`:
 * the body only carries `noul`, and `probabilities` is filled in here.
 */

import { StatimError } from "./errors.js";

export interface ErrorBody {
  detail: string;
}

export interface Action {
  act_probability: number;
}

export interface Usage {
  input_tokens: number;
  output_tokens: number;
}

/** `reason` is requested, lang:en, lang:other, default, or consensus. */
export interface Routing {
  model: string;
  reason: string;
  engine: string;
  weights: string;
}

export type ChoiceValue = string | number | boolean | null;

export interface ChoiceAnswer {
  type: "choice";
  choice: ChoiceValue;
  probabilities: Record<string, number>;
  confidence: number;
  answer_confidence: number;
  action: Action;
  escalate?: boolean;
  logits?: number[];
  logits_by_model?: Record<string, number[]>;
}

export interface ScoreAnswer {
  type: "score";
  score: number;
  legend: Record<string, unknown>;
  probabilities: Record<string, number>;
  confidence: number;
  answer_confidence: number;
  action: Action;
  escalate?: boolean;
  logits?: number[];
  logits_by_model?: Record<string, number[]>;
}

/** Wire object for question type `noul`. */
export interface NoulAnswer {
  type: "noul";
  noul: number;
  confidence: number;
  answer_confidence: number;
  action: Action;
  escalate?: boolean;
  logits?: number[];
  logits_by_model?: Record<string, number[]>;
}

/** `noul` as a yes/no answer. `probabilities.yes` equals `noul`. */
export interface YesNoAnswer extends NoulAnswer {
  yes: boolean;
  probabilities: { yes: number; no: number };
}

export type Answer = ChoiceAnswer | ScoreAnswer | YesNoAnswer;

export interface Decision {
  model: string;
  answers: Record<string, Answer>;
  usage: Usage;
  routing: Routing;
  /** `X-Request-Id` echoed by the server, or null when it sent none. */
  request_id: string | null;
  /** `X-Inference-Time-Ms`, or null when the server sent none. */
  inference_time_ms: number | null;
}

export interface Question {
  type: "choice" | "score" | "noul";
  instructions: unknown;
  criteria?: unknown;
  labels?: { true?: string; false?: string };
}

export interface SystemOneRequest {
  state: unknown;
  questions: Record<string, Question>;
  model?: string;
  lang?: string;
  ensemble?: number;
  ensemble_margin?: number;
  calibrate?: boolean;
  return_logits?: boolean;
  min_confidence?: number;
  max_len?: number;
  head_max_len?: number;
}

export interface BatchRequest {
  states: unknown[];
  questions: Record<string, Question>;
  model?: string;
  lang?: string;
  ensemble?: number;
  ensemble_margin?: number;
  calibrate?: boolean;
  return_logits?: boolean;
  min_confidence?: number;
  max_len?: number;
  head_max_len?: number;
}

/** Wire batch body. The SDK method returns {@link BatchResult}. */
export interface BatchResponse {
  results: Decision[];
}

export interface BatchResult {
  results: Decision[];
  request_id: string | null;
  inference_time_ms: number | null;
}

export interface Health {
  status: string;
  version: string;
}

export interface Ready {
  ready: boolean;
}

export interface Model {
  id: string;
  object: string;
  owned_by: string;
  source: string;
  weights: string;
  layers: number;
  hidden: number;
  max_len: number;
  vocab: number;
  device: string;
}

export interface ModelList {
  object: string;
  data: Model[];
}

export interface DecideOptions {
  model?: string;
  lang?: string;
  ensemble?: number;
  ensemble_margin?: number;
  calibrate?: boolean;
  return_logits?: boolean;
  min_confidence?: number;
  max_len?: number;
  head_max_len?: number;
  request_id?: string;
}

const DECISION_OPTIONS = new Set([
  "model",
  "lang",
  "ensemble",
  "ensemble_margin",
  "calibrate",
  "return_logits",
  "min_confidence",
  "max_len",
  "head_max_len",
  "request_id",
]);

export function assertDecideOptions(options: DecideOptions): void {
  for (const key of Object.keys(options)) {
    if (!DECISION_OPTIONS.has(key)) throw new TypeError(`unexpected option(s): ${key}`);
  }
  const requestId = options.request_id;
  if (requestId !== undefined && typeof requestId !== "string") {
    throw new TypeError("request_id must be a string");
  }
}

export function bodyOptions(options: DecideOptions): Record<string, unknown> {
  const body: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(options)) {
    if (key === "request_id" || value === undefined || value === null) continue;
    body[key] = value;
  }
  return body;
}

function fail(message: string): never {
  throw new StatimError(`malformed response: ${message}`);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function field(obj: Record<string, unknown>, key: string): unknown {
  if (!(key in obj)) fail(`missing ${key}`);
  return obj[key];
}

function num(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) fail(`${name} must be a number`);
  return value;
}

function integer(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isInteger(value)) fail(`${name} must be an integer`);
  return value;
}

function str(value: unknown, name: string): string {
  if (typeof value !== "string") fail(`${name} must be a string`);
  return value;
}

function scalar(value: unknown, name: string): ChoiceValue {
  if (value === null || typeof value === "string" || typeof value === "boolean") return value;
  if (typeof value === "number" && Number.isFinite(value)) return value;
  return fail(`${name} must be a string, number, boolean, or null`);
}

function probabilities(value: unknown): Record<string, number> {
  if (!isRecord(value)) fail("probabilities must be an object");
  const out: Record<string, number> = {};
  for (const [key, item] of Object.entries(value)) out[key] = num(item, `probabilities.${key}`);
  return out;
}

function floatList(value: unknown, name: string): number[] {
  if (!Array.isArray(value)) fail(`${name} must be an array`);
  return value.map((item) => num(item, name));
}

function logitsOf(obj: Record<string, unknown>): number[] | undefined {
  if (!("logits" in obj)) return undefined;
  return floatList(obj.logits, "logits");
}

function logitsByModel(obj: Record<string, unknown>): Record<string, number[]> | undefined {
  if (!("logits_by_model" in obj)) return undefined;
  if (!isRecord(obj.logits_by_model)) fail("logits_by_model must be an object");
  const out: Record<string, number[]> = {};
  for (const [key, item] of Object.entries(obj.logits_by_model)) {
    out[key] = floatList(item, `logits_by_model.${key}`);
  }
  return out;
}

function actionOf(obj: Record<string, unknown>): Action {
  const raw = field(obj, "action");
  if (!isRecord(raw)) fail("action must be an object");
  return { act_probability: num(field(raw, "act_probability"), "action.act_probability") };
}

export function parseAnswer(value: unknown): Answer {
  if (!isRecord(value)) fail("answer must be an object");
  const kind = str(field(value, "type"), "answer.type");
  const shared = {
    confidence: num(field(value, "confidence"), "confidence"),
    answer_confidence: num(field(value, "answer_confidence"), "answer_confidence"),
    action: actionOf(value),
    ...("escalate" in value
      ? { escalate: typeof value.escalate === "boolean" ? value.escalate : fail("escalate must be a boolean") }
      : {}),
  };
  const logits = logitsOf(value);
  const byModel = logitsByModel(value);
  const extra = {
    ...(logits !== undefined ? { logits } : {}),
    ...(byModel !== undefined ? { logits_by_model: byModel } : {}),
  };
  if (kind === "choice") {
    const answer: ChoiceAnswer = {
      type: "choice",
      choice: scalar(field(value, "choice"), "choice"),
      probabilities: probabilities(field(value, "probabilities")),
      ...shared,
      ...extra,
    };
    return answer;
  }
  if (kind === "score") {
    const legendRaw = field(value, "legend");
    if (!isRecord(legendRaw)) fail("legend must be an object");
    const answer: ScoreAnswer = {
      type: "score",
      score: num(field(value, "score"), "score"),
      legend: { ...legendRaw },
      probabilities: probabilities(field(value, "probabilities")),
      ...shared,
      ...extra,
    };
    return answer;
  }
  if (kind === "noul") {
    const noul = num(field(value, "noul"), "noul");
    const answer: YesNoAnswer = {
      type: "noul",
      noul,
      yes: noul >= 0.5,
      probabilities: { yes: noul, no: 1 - noul },
      ...shared,
      ...extra,
    };
    return answer;
  }
  return fail(`unknown answer type ${JSON.stringify(kind)}`);
}

function parseUsage(value: unknown): Usage {
  if (!isRecord(value)) fail("usage must be an object");
  return {
    input_tokens: integer(field(value, "input_tokens"), "usage.input_tokens"),
    output_tokens: integer(field(value, "output_tokens"), "usage.output_tokens"),
  };
}

function parseRouting(value: unknown): Routing {
  if (!isRecord(value)) fail("routing must be an object");
  return {
    model: str(field(value, "model"), "routing.model"),
    reason: str(field(value, "reason"), "routing.reason"),
    engine: str(field(value, "engine"), "routing.engine"),
    weights: str(field(value, "weights"), "routing.weights"),
  };
}

export function parseDecision(value: unknown): Decision {
  if (!isRecord(value)) fail("decision must be an object");
  const answersRaw = field(value, "answers");
  if (!isRecord(answersRaw)) fail("answers must be an object");
  const answers: Record<string, Answer> = {};
  for (const [key, item] of Object.entries(answersRaw)) answers[key] = parseAnswer(item);
  return {
    model: str(field(value, "model"), "model"),
    answers,
    usage: parseUsage(field(value, "usage")),
    routing: parseRouting(field(value, "routing")),
    request_id: null,
    inference_time_ms: null,
  };
}

export function parseBatch(value: unknown): BatchResult {
  if (!isRecord(value)) fail("batch must be an object");
  const results = field(value, "results");
  if (!Array.isArray(results)) fail("results must be an array");
  return { results: results.map((item) => parseDecision(item)), request_id: null, inference_time_ms: null };
}

export function parseHealth(value: unknown): Health {
  if (!isRecord(value)) fail("health must be an object");
  return { status: str(field(value, "status"), "status"), version: str(field(value, "version"), "version") };
}

export function parseReady(value: unknown): Ready {
  if (!isRecord(value)) fail("ready must be an object");
  const ready = field(value, "ready");
  if (typeof ready !== "boolean") fail("ready must be a boolean");
  return { ready };
}

export function parseModel(value: unknown): Model {
  if (!isRecord(value)) fail("model must be an object");
  return {
    id: str(field(value, "id"), "id"),
    object: str(field(value, "object"), "object"),
    owned_by: str(field(value, "owned_by"), "owned_by"),
    source: str(field(value, "source"), "source"),
    weights: str(field(value, "weights"), "weights"),
    layers: integer(field(value, "layers"), "layers"),
    hidden: integer(field(value, "hidden"), "hidden"),
    max_len: integer(field(value, "max_len"), "max_len"),
    vocab: integer(field(value, "vocab"), "vocab"),
    device: str(field(value, "device"), "device"),
  };
}

export function parseModelList(value: unknown): ModelList {
  if (!isRecord(value)) fail("models must be an object");
  const data = field(value, "data");
  if (!Array.isArray(data)) fail("data must be an array");
  return { object: str(field(value, "object"), "object"), data: data.map((item) => parseModel(item)) };
}
