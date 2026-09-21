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
      readonly attemptId: string;
      readonly attemptNumber: number;
    })
  | (ProviderTraceEventBase & {
      readonly type: "provider_attempt_connected";
      readonly attemptId: string;
      readonly attemptNumber: number;
      readonly durationMs: number;
    })
  | (ProviderTraceEventBase & {
      readonly type: "provider_attempt_failed";
      readonly attemptId: string;
      readonly attemptNumber: number;
      readonly durationMs: number;
      readonly errorName: string;
      readonly message: string;
    })
  | (ProviderTraceEventBase & {
      readonly type: "provider_retry_wait";
      readonly attemptId: string;
      readonly attemptNumber: number;
      readonly delayMs: number;
      readonly reason: string;
    })
  | (ProviderTraceEventBase & {
      readonly type: "provider_request_mutated";
      readonly mutation: "tools_removed";
      readonly appliesToAttemptNumber: number;
      readonly reason: string;
    });

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
}): Pick<
  import("./internal/withRetry.js").WithRetryOptions,
  "onAttemptStart" | "onAttemptSuccess" | "onAttemptFailure" | "onRetry"
> {
  const attemptIds = new Map<number, string>();
  const emit = (event: ProviderTraceEvent): void =>
    emitSafely(options.request.onTraceEvent, event);
  const base = (): ProviderTraceEventBase | undefined =>
    eventBase(options.provider, options.request);
  const attemptIdentity = (attempt: number) => {
    let attemptId = attemptIds.get(attempt);
    if (!attemptId) {
      attemptId = randomUUID();
      attemptIds.set(attempt, attemptId);
    }
    return { attemptId, attemptNumber: attempt + 1 };
  };

  return {
    onAttemptStart: ({ attempt }) => {
      const fields = base();
      if (fields)
        emit({
          ...fields,
          type: "provider_attempt_started",
          ...attemptIdentity(attempt),
        });
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

  private emitStarted(request: ChatRequest): void {
    const base = eventBase(this.name, request);
    if (!base) return;
    emitSafely(request.onTraceEvent, {
      ...base,
      type: "provider_request_started",
      messageCount: request.messages.length,
      toolCount: request.tools?.length ?? 0,
    });
  }

  private emitCompleted(
    request: ChatRequest,
    startedAt: number,
    terminal?: { stopReason: StopReason; rawProviderReason?: string },
  ): void {
    const base = eventBase(this.name, request);
    if (!base) return;
    emitSafely(request.onTraceEvent, {
      ...base,
      type: "provider_request_completed",
      durationMs: performance.now() - startedAt,
      ...terminal,
    });
  }

  private emitFailed(
    request: ChatRequest,
    startedAt: number,
    error: unknown,
  ): void {
    const base = eventBase(this.name, request);
    if (!base) return;
    emitSafely(request.onTraceEvent, {
      ...base,
      type: "provider_request_failed",
      durationMs: performance.now() - startedAt,
      errorName: error instanceof Error ? error.name : "Error",
      message: error instanceof Error ? error.message : String(error),
    });
  }

  private emitCancelled(request: ChatRequest, startedAt: number): void {
    const error = new Error("Provider stream consumption cancelled");
    error.name = "AbortError";
    this.emitFailed(request, startedAt, error);
  }

  async *streamChat(request: ChatRequest): AsyncIterable<ChatStreamEvent> {
    const startedAt = performance.now();
    this.emitStarted(request);

    let terminal:
      | { stopReason: StopReason; rawProviderReason?: string }
      | undefined;
    let outcomeRecorded = false;
    try {
      for await (const event of this.provider.streamChat(request)) {
        if ("type" in event && event.type === "terminal") {
          terminal = {
            stopReason: event.stopReason,
            rawProviderReason: event.rawProviderReason,
          };
        }
        yield event;
      }

      this.emitCompleted(request, startedAt, terminal);
      outcomeRecorded = true;
    } catch (error) {
      this.emitFailed(request, startedAt, error);
      outcomeRecorded = true;
      throw error;
    } finally {
      if (!outcomeRecorded) {
        if (terminal) {
          this.emitCompleted(request, startedAt, terminal);
        } else {
          this.emitCancelled(request, startedAt);
        }
      }
    }
  }
}
