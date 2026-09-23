import { randomUUID } from "node:crypto";
import type { LLMProvider, ProviderCapabilities } from "./interface.js";
import type { ChatRequest, ChatStreamEvent, StopReason } from "./types.js";

export type ProviderRequestPurpose = "answer" | "summarize" | "recovery";

/** Caller-owned causal identity. Providers never discover or persist it implicitly. */
export interface ProviderTraceContext {
  readonly sessionId?: string;
  readonly runId?: string;
  readonly turnId?: string;
  readonly requestId: string;
  readonly operationId: string;
  readonly parentOperationId?: string;
  readonly purpose: ProviderRequestPurpose;
  readonly configurationRevisionId?: string;
  readonly promptRevisionId?: string;
}

interface ProviderTraceEventBase {
  readonly version: 1;
  readonly eventId: string;
  readonly observedAt: string;
  readonly provider: string;
  readonly requestedModel: string;
  readonly trace: ProviderTraceContext;
}

export interface ProviderTokenUsage {
  readonly inputTokens?: number;
  readonly outputTokens?: number;
  readonly cacheReadInputTokens?: number;
  readonly cacheWriteInputTokens?: number;
  /** Included in outputTokens when the provider documents it that way. */
  readonly reasoningTokens?: number;
  readonly totalTokens?: number;
}

export interface ProviderReportedCost {
  readonly amount: number;
  readonly currency?: string;
}

export type ProviderUsageAvailability = "reported" | "partial" | "unavailable";

interface ProviderAttemptIdentity {
  readonly attemptId: string;
  readonly attemptNumber: number;
  readonly endpointClass?: string;
}

export type ProviderTraceEvent =
  | (ProviderTraceEventBase & {
      readonly type: "provider_request_started";
      readonly messageCount: number;
      readonly toolCount: number;
    })
  | (ProviderTraceEventBase & {
      readonly type: "provider_request_completed";
      readonly durationMs: number;
      readonly stopReason?: StopReason;
      readonly rawProviderReason?: string;
    })
  | (ProviderTraceEventBase & {
      readonly type: "provider_request_failed";
      readonly durationMs: number;
      readonly errorName: string;
      readonly message: string;
    })
  | (ProviderTraceEventBase & {
      readonly type: "provider_attempt_started";
    } & ProviderAttemptIdentity)
  | (ProviderTraceEventBase & {
      readonly type: "provider_attempt_payload";
      /** HTTP JSON body or SDK command/input, before SDK-specific serialization. */
      readonly transport: "http_json" | "sdk_input" | "unavailable";
      readonly requestBody?: unknown;
      readonly unavailableReason?: "adapter_payload_capture_failed";
    } & ProviderAttemptIdentity)
  | (ProviderTraceEventBase & {
      readonly type: "provider_attempt_connected";
      readonly durationMs: number;
    } & ProviderAttemptIdentity)
  | (ProviderTraceEventBase & {
      readonly type: "provider_attempt_failed";
      readonly durationMs: number;
      readonly errorName: string;
      readonly message: string;
    } & ProviderAttemptIdentity)
  | (ProviderTraceEventBase & {
      readonly type: "provider_retry_wait";
      readonly delayMs: number;
      readonly reason: string;
    } & ProviderAttemptIdentity)
  | (ProviderTraceEventBase & {
      readonly type: "provider_attempt_first_output";
      readonly durationMs: number;
      readonly outputType: "assistant_text" | "thinking" | "tool_call";
    } & ProviderAttemptIdentity)
  | (ProviderTraceEventBase & {
      readonly type: "provider_attempt_completed";
      readonly durationMs: number;
      readonly stopReason?: StopReason;
      readonly rawProviderReason?: string;
    } & ProviderAttemptIdentity)
  | (ProviderTraceEventBase & {
      readonly type: "provider_response_metadata";
      readonly attemptId?: string;
      readonly attemptNumber?: number;
      readonly endpointClass?: string;
      readonly upstreamRequestId?: string;
      readonly actualModel?: string;
    })
  | (ProviderTraceEventBase & {
      readonly type: "provider_usage_reported";
      readonly attemptId?: string;
      readonly attemptNumber?: number;
      readonly endpointClass?: string;
      readonly availability: ProviderUsageAvailability;
      readonly reportKind?: "delta" | "cumulative";
      readonly usage?: ProviderTokenUsage;
      readonly providerReportedCost?: ProviderReportedCost;
    })
  | (ProviderTraceEventBase & {
      readonly type: "provider_request_mutated";
      readonly mutation: "tools_removed";
      readonly appliesToAttemptNumber: number;
      readonly reason: string;
    });

type ProviderTraceEventDetails<
  Event extends ProviderTraceEvent = ProviderTraceEvent,
> = Event extends ProviderTraceEventBase
  ? Omit<Event, keyof ProviderTraceEventBase>
  : never;

export type ProviderTraceObserver = (event: ProviderTraceEvent) => void;

function emitSafely(
  observer: ProviderTraceObserver | undefined,
  event: ProviderTraceEvent,
): void {
  try {
    observer?.(event);
  } catch {
    // Trace sinks are observational. They must not change provider behavior.
  }
}

function eventBase(
  provider: string,
  request: ChatRequest,
): ProviderTraceEventBase | undefined {
  if (!request.trace) return undefined;
  return {
    version: 1,
    eventId: randomUUID(),
    observedAt: new Date().toISOString(),
    provider,
    requestedModel: request.model,
    trace: request.trace,
  };
}

export function createProviderAttemptTraceHooks(options: {
  readonly provider: string;
  readonly request: ChatRequest;
  readonly endpointClass?: string | ((attemptNumber: number) => string);
  readonly requestPayload?: (attemptNumber: number) => {
    readonly transport: "http_json" | "sdk_input";
    readonly requestBody: unknown;
  };
}): Pick<
  import("./internal/withRetry.js").WithRetryOptions,
  "onAttemptStart" | "onAttemptSuccess" | "onAttemptFailure" | "onRetry"
> {
  const attemptIds = new Map<number, string>();
  const emit = (event: ProviderTraceEvent): void =>
    emitSafely(options.request.onTraceEvent, event);
  const base = (): ProviderTraceEventBase | undefined =>
    eventBase(options.provider, options.request);
  const attemptIdentity = (attempt: number): ProviderAttemptIdentity => {
    let attemptId = attemptIds.get(attempt);
    if (!attemptId) {
      attemptId = randomUUID();
      attemptIds.set(attempt, attemptId);
    }
    const attemptNumber = attempt + 1;
    const endpointClass =
      typeof options.endpointClass === "function"
        ? options.endpointClass(attemptNumber)
        : options.endpointClass;
    return {
      attemptId,
      attemptNumber,
      ...(endpointClass ? { endpointClass } : {}),
    };
  };

  return {
    onAttemptStart: ({ attempt }) => {
      const fields = base();
      if (fields) {
        emit({
          ...fields,
          type: "provider_attempt_started",
          ...attemptIdentity(attempt),
        });
        if (options.request.captureRequestPayload && options.requestPayload) {
          const payloadFields = base();
          if (!payloadFields) return;
          try {
            const payload = options.requestPayload(attempt + 1);
            emit({
              ...payloadFields,
              type: "provider_attempt_payload",
              ...attemptIdentity(attempt),
              transport: payload.transport,
              requestBody: structuredClone(payload.requestBody),
            });
          } catch {
            emit({
              ...payloadFields,
              type: "provider_attempt_payload",
              ...attemptIdentity(attempt),
              transport: "unavailable",
              unavailableReason: "adapter_payload_capture_failed",
            });
          }
        }
      }
    },
    onAttemptSuccess: ({ attempt, durationMs }) => {
      const fields = base();
      if (fields)
        emit({
          ...fields,
          type: "provider_attempt_connected",
          ...attemptIdentity(attempt),
          durationMs,
        });
    },
    onAttemptFailure: ({ attempt, durationMs, err }) => {
      const fields = base();
      if (fields)
        emit({
          ...fields,
          type: "provider_attempt_failed",
          ...attemptIdentity(attempt),
          durationMs,
          errorName: err instanceof Error ? err.name : "Error",
          message: err instanceof Error ? err.message : String(err),
        });
    },
    onRetry: ({ attempt, delayMs, err }) => {
      const fields = base();
      if (fields)
        emit({
          ...fields,
          type: "provider_retry_wait",
          ...attemptIdentity(attempt),
          delayMs,
          reason: err instanceof Error ? err.message : String(err),
        });
    },
  };
}

export function emitProviderResponseMetadata(options: {
  readonly provider: string;
  readonly request: ChatRequest;
  readonly upstreamRequestId?: string;
  readonly actualModel?: string;
  readonly endpointClass?: string;
}): void {
  if (
    !options.upstreamRequestId &&
    !options.actualModel &&
    !options.endpointClass
  ) {
    return;
  }
  const fields = eventBase(options.provider, options.request);
  if (!fields) return;
  emitSafely(options.request.onTraceEvent, {
    ...fields,
    type: "provider_response_metadata",
    ...(options.upstreamRequestId
      ? { upstreamRequestId: options.upstreamRequestId }
      : {}),
    ...(options.actualModel ? { actualModel: options.actualModel } : {}),
    ...(options.endpointClass ? { endpointClass: options.endpointClass } : {}),
  });
}

export function emitProviderUsage(options: {
  readonly provider: string;
  readonly request: ChatRequest;
  readonly availability: Exclude<ProviderUsageAvailability, "unavailable">;
  readonly reportKind: "delta" | "cumulative";
  readonly usage: ProviderTokenUsage;
  readonly providerReportedCost?: ProviderReportedCost;
  readonly endpointClass?: string;
}): void {
  const fields = eventBase(options.provider, options.request);
  if (!fields) return;
  emitSafely(options.request.onTraceEvent, {
    ...fields,
    type: "provider_usage_reported",
    availability: options.availability,
    reportKind: options.reportKind,
    usage: options.usage,
    ...(options.providerReportedCost
      ? { providerReportedCost: options.providerReportedCost }
      : {}),
    ...(options.endpointClass ? { endpointClass: options.endpointClass } : {}),
  });
}

export function emitProviderRequestMutation(options: {
  readonly provider: string;
  readonly request: ChatRequest;
  readonly mutation: "tools_removed";
  readonly appliesToAttemptNumber: number;
  readonly reason: string;
}): void {
  const fields = eventBase(options.provider, options.request);
  if (!fields) return;
  emitSafely(options.request.onTraceEvent, {
    ...fields,
    type: "provider_request_mutated",
    mutation: options.mutation,
    appliesToAttemptNumber: options.appliesToAttemptNumber,
    reason: options.reason,
  });
}

/**
 * Decorates a provider with logical-request observations. Retry/attempt events
 * remain provider-owned and can be added without changing this boundary.
 */
export function withProviderTracing(provider: LLMProvider): LLMProvider {
  return new TracedProvider(provider);
}

class TracedProvider implements LLMProvider {
  readonly name: string;

  constructor(private readonly provider: LLMProvider) {
    this.name = provider.name;
  }

  getCapabilities(): ProviderCapabilities {
    return this.provider.getCapabilities();
  }

  async *streamChat(request: ChatRequest): AsyncIterable<ChatStreamEvent> {
    const traceState = new ProviderRequestTraceState(this.name, request);
    try {
      for await (const event of this.provider.streamChat(traceState.request)) {
        traceState.observeStreamEvent(event);
        yield event;
      }
      traceState.complete();
    } catch (error) {
      traceState.fail(error);
      throw error;
    } finally {
      traceState.closeInterrupted();
    }
  }
}

type ProviderTerminal = {
  stopReason: StopReason;
  rawProviderReason?: string;
};

class ProviderRequestTraceState {
  readonly request: ChatRequest;
  private readonly startedAt = performance.now();
  private currentAttempt?: ProviderAttemptIdentity;
  private currentAttemptStartedAt = this.startedAt;
  private terminal?: ProviderTerminal;
  private usageObserved = false;
  private firstOutputObserved = false;
  private outcomeRecorded = false;
  private readonly failedAttemptIds = new Set<string>();

  constructor(
    private readonly provider: string,
    private readonly originalRequest: ChatRequest,
  ) {
    this.emitStarted();
    this.request = {
      ...originalRequest,
      onTraceEvent: (event) => this.observeProviderEvent(event),
    };
  }

  observeStreamEvent(event: ChatStreamEvent): void {
    this.captureFirstOutput(event);
    if ("type" in event && event.type === "terminal") {
      this.terminal = {
        stopReason: event.stopReason,
        rawProviderReason: event.rawProviderReason,
      };
    }
  }

  complete(): void {
    this.emitAttemptCompleted();
    this.emitUnavailableUsage();
    this.emitRequestOutcome("provider_request_completed");
    this.outcomeRecorded = true;
  }

  fail(error: unknown): void {
    this.emitAttemptFailure(error);
    this.emitUnavailableUsage();
    this.emitRequestOutcome("provider_request_failed", error);
    this.outcomeRecorded = true;
  }

  closeInterrupted(): void {
    if (this.outcomeRecorded) return;
    if (this.terminal) {
      this.emitAttemptCompleted();
      this.emitUnavailableUsage();
      this.emitRequestOutcome("provider_request_completed");
      return;
    }
    const error = new Error("Provider stream consumption cancelled");
    error.name = "AbortError";
    this.emitAttemptFailure(error);
    this.emitUnavailableUsage();
    this.emitRequestOutcome("provider_request_failed", error);
  }

  private observeProviderEvent(event: ProviderTraceEvent): void {
    this.captureAttempt(event);
    if (event.type === "provider_usage_reported") this.usageObserved = true;
    if (event.type === "provider_attempt_failed") {
      this.failedAttemptIds.add(event.attemptId);
    }
    emitSafely(this.originalRequest.onTraceEvent, this.attributeAttempt(event));
  }

  private captureAttempt(event: ProviderTraceEvent): void {
    if (
      event.type !== "provider_attempt_started" &&
      event.type !== "provider_attempt_connected"
    ) {
      return;
    }
    if (event.type === "provider_attempt_started") {
      this.currentAttemptStartedAt = performance.now();
    }
    this.currentAttempt = {
      attemptId: event.attemptId,
      attemptNumber: event.attemptNumber,
      ...(event.endpointClass ? { endpointClass: event.endpointClass } : {}),
    };
  }

  private attributeAttempt(event: ProviderTraceEvent): ProviderTraceEvent {
    if (
      (event.type !== "provider_response_metadata" &&
        event.type !== "provider_usage_reported") ||
      !this.currentAttempt ||
      event.attemptId !== undefined
    ) {
      return event;
    }
    return { ...event, ...this.currentAttempt };
  }

  private captureFirstOutput(event: ChatStreamEvent): void {
    if (this.firstOutputObserved || !("type" in event)) return;
    const outputType = providerOutputType(event.type);
    if (!outputType || !this.currentAttempt) return;
    this.firstOutputObserved = true;
    this.emit({
      type: "provider_attempt_first_output",
      ...this.currentAttempt,
      durationMs: performance.now() - this.currentAttemptStartedAt,
      outputType,
    });
  }

  private emitStarted(): void {
    this.emit({
      type: "provider_request_started",
      messageCount: this.originalRequest.messages.length,
      toolCount: this.originalRequest.tools?.length ?? 0,
    });
  }

  private emitAttemptCompleted(): void {
    if (!this.currentAttempt) return;
    this.emit({
      type: "provider_attempt_completed",
      ...this.currentAttempt,
      durationMs: performance.now() - this.currentAttemptStartedAt,
      ...this.terminal,
    });
  }

  private emitAttemptFailure(error: unknown): void {
    if (
      !this.currentAttempt ||
      this.failedAttemptIds.has(this.currentAttempt.attemptId)
    ) {
      return;
    }
    this.emit({
      type: "provider_attempt_failed",
      ...this.currentAttempt,
      durationMs: performance.now() - this.currentAttemptStartedAt,
      ...errorFields(error),
    });
  }

  private emitUnavailableUsage(): void {
    if (this.usageObserved) return;
    this.emit({
      type: "provider_usage_reported",
      availability: "unavailable",
      ...this.currentAttempt,
    });
    this.usageObserved = true;
  }

  private emitRequestOutcome(
    type: "provider_request_completed" | "provider_request_failed",
    error?: unknown,
  ): void {
    this.emit({
      type,
      durationMs: performance.now() - this.startedAt,
      ...(type === "provider_request_completed"
        ? this.terminal
        : errorFields(error)),
    } as ProviderTraceEvent);
  }

  private emit(event: ProviderTraceEventDetails): void {
    const base = eventBase(this.provider, this.originalRequest);
    if (!base) return;
    emitSafely(this.originalRequest.onTraceEvent, {
      ...base,
      ...event,
    } as ProviderTraceEvent);
  }
}

function providerOutputType(
  type: string,
): "assistant_text" | "thinking" | "tool_call" | undefined {
  if (type === "assistant_text") return "assistant_text";
  if (type === "thinking_delta" || type === "reasoning_summary") {
    return "thinking";
  }
  return type === "tool_calls" ? "tool_call" : undefined;
}

function errorFields(error: unknown): { errorName: string; message: string } {
  return {
    errorName: error instanceof Error ? error.name : "Error",
    message: error instanceof Error ? error.message : String(error),
  };
}
