import type { LLMProvider } from "../interface.js";
import type { ChatRequest, ChatStreamEvent } from "../types.js";
import { withProviderTracing, type ProviderTraceEvent } from "../trace.js";
import { createProviderRetryOptions } from "../internal/shared.js";
import { withRetry } from "../internal/withRetry.js";

const trace = {
  sessionId: "session-1",
  runId: "run-1",
  turnId: "turn-1",
  requestId: "request-1",
  operationId: "operation-1",
  purpose: "answer" as const,
};

function request(
  onTraceEvent: (event: ProviderTraceEvent) => void,
): ChatRequest {
  return {
    model: "model-1",
    messages: [{ role: "user", content: "hello" }],
    trace,
    onTraceEvent,
  };
}

describe("withProviderTracing", () => {
  it("assigns distinct identities to retry attempts and links retry waits", async () => {
    const events: ProviderTraceEvent[] = [];
    const tracedRequest = request((event) => events.push(event));
    let calls = 0;

    await expect(
      withRetry(
        async () => {
          calls++;
          if (calls === 1) throw new Error("temporary");
          return "connected";
        },
        createProviderRetryOptions({
          request: tracedRequest,
          model: "model-1",
          provider: "fixture",
          retryConfig: {
            maxRetries: 1,
            consecutive529Limit: 3,
            baseDelayMs: 0,
          },
          isRetryable: () => true,
        }),
      ),
    ).resolves.toBe("connected");

    expect(events.map((event) => event.type)).toEqual([
      "provider_attempt_started",
      "provider_attempt_failed",
      "provider_retry_wait",
      "provider_attempt_started",
      "provider_attempt_connected",
    ]);
    const firstAttemptEvents = events.slice(0, 3) as Array<
      ProviderTraceEvent & { attemptId: string }
    >;
    expect(
      new Set(firstAttemptEvents.map((event) => event.attemptId)).size,
    ).toBe(1);
    const secondAttempt = events[3] as ProviderTraceEvent & {
      attemptId: string;
    };
    expect(secondAttempt.attemptId).not.toBe(firstAttemptEvents[0].attemptId);
  });

  it("observes a completed logical request without changing stream events", async () => {
    const provider: LLMProvider = {
      name: "fixture",
      getCapabilities: () => ({ contextWindowTokens: 1000 }),
      async *streamChat(): AsyncIterable<ChatStreamEvent> {
        yield { type: "assistant_text", delta: "ok" };
        yield { type: "terminal", stopReason: "end_turn" };
      },
    };
    const events: ProviderTraceEvent[] = [];

    const streamed: ChatStreamEvent[] = [];
    for await (const event of withProviderTracing(provider).streamChat(
      request((event) => events.push(event)),
    )) {
      streamed.push(event);
    }

    expect(streamed).toHaveLength(2);
    expect(events.map((event) => event.type)).toEqual([
      "provider_request_started",
      "provider_request_completed",
    ]);
    expect(events[1]).toMatchObject({
      trace,
      stopReason: "end_turn",
      requestedModel: "model-1",
    });
  });

  it("completes the trace when consumption stops after the terminal event", async () => {
    const provider: LLMProvider = {
      name: "fixture",
      getCapabilities: () => ({ contextWindowTokens: 1000 }),
      async *streamChat(): AsyncIterable<ChatStreamEvent> {
        yield { type: "terminal", stopReason: "end_turn" };
        yield { type: "assistant_text", delta: "unread" };
      },
    };
    const events: ProviderTraceEvent[] = [];

    for await (const event of withProviderTracing(provider).streamChat(
      request((traceEvent) => events.push(traceEvent)),
    )) {
      if (event.type === "terminal") break;
    }

    expect(events.map((event) => event.type)).toEqual([
      "provider_request_started",
      "provider_request_completed",
    ]);
    expect(events[1]).toMatchObject({ stopReason: "end_turn" });
  });

  it("fails the trace when consumption stops before a terminal event", async () => {
    const provider: LLMProvider = {
      name: "fixture",
      getCapabilities: () => ({ contextWindowTokens: 1000 }),
      async *streamChat(): AsyncIterable<ChatStreamEvent> {
        yield { type: "assistant_text", delta: "partial" };
        yield { type: "terminal", stopReason: "end_turn" };
      },
    };
    const events: ProviderTraceEvent[] = [];

    for await (const _event of withProviderTracing(provider).streamChat(
      request((event) => events.push(event)),
    )) {
      break;
    }

    expect(events.map((event) => event.type)).toEqual([
      "provider_request_started",
      "provider_request_failed",
    ]);
    expect(events[1]).toMatchObject({
      errorName: "AbortError",
      message: "Provider stream consumption cancelled",
    });
  });

  it("isolates observer failures and reports provider failures", async () => {
    const failure = new Error("upstream failed");
    const provider: LLMProvider = {
      name: "fixture",
      getCapabilities: () => ({ contextWindowTokens: 1000 }),
      async *streamChat(): AsyncIterable<ChatStreamEvent> {
        throw failure;
      },
    };
    const observed: ProviderTraceEvent[] = [];
    let calls = 0;
    const traced = withProviderTracing(provider);

    await expect(async () => {
      for await (const _event of traced.streamChat(
        request((event) => {
          calls++;
          if (calls === 1) throw new Error("sink failed");
          observed.push(event);
        }),
      )) {
        // no events
      }
    }).rejects.toBe(failure);

    expect(observed).toHaveLength(1);
    expect(observed[0]).toMatchObject({
      type: "provider_request_failed",
      errorName: "Error",
      message: "upstream failed",
    });
  });
});
