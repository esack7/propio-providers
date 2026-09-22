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

  it("attributes first useful output and unavailable usage to the connected attempt", async () => {
    const events: ProviderTraceEvent[] = [];
    const provider: LLMProvider = {
      name: "fixture",
      getCapabilities: () => ({ contextWindowTokens: 1000 }),
      async *streamChat(streamRequest): AsyncIterable<ChatStreamEvent> {
        await withRetry(
          async () => "connected",
          createProviderRetryOptions({
            request: streamRequest,
            model: streamRequest.model,
            provider: "fixture",
            retryConfig: {
              maxRetries: 0,
              consecutive529Limit: 1,
              baseDelayMs: 0,
            },
            endpointClass: "fixture_stream",
            isRetryable: () => false,
          }),
        );
        yield { type: "assistant_text", delta: "ok" };
        yield { type: "terminal", stopReason: "end_turn" };
      },
    };

    for await (const _event of withProviderTracing(provider).streamChat(
      request((event) => events.push(event)),
    )) {
      // consume
    }

    const started = events.find(
      (event) => event.type === "provider_attempt_started",
    );
    const firstOutput = events.find(
      (event) => event.type === "provider_attempt_first_output",
    );
    const usage = events.find(
      (event) => event.type === "provider_usage_reported",
    );
    expect(started).toMatchObject({ endpointClass: "fixture_stream" });
    expect(firstOutput).toMatchObject({
      attemptId: started?.attemptId,
      attemptNumber: 1,
      endpointClass: "fixture_stream",
      outputType: "assistant_text",
    });
    expect(usage).toMatchObject({
      attemptId: started?.attemptId,
      availability: "unavailable",
    });
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
      "provider_usage_reported",
      "provider_request_completed",
    ]);
    expect(events[2]).toMatchObject({
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
      "provider_usage_reported",
      "provider_request_completed",
    ]);
    expect(events[2]).toMatchObject({ stopReason: "end_turn" });
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
      "provider_usage_reported",
      "provider_request_failed",
    ]);
    expect(events[2]).toMatchObject({
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

    expect(observed).toHaveLength(2);
    expect(observed[0]).toMatchObject({
      type: "provider_usage_reported",
      availability: "unavailable",
    });
    expect(observed[1]).toMatchObject({
      type: "provider_request_failed",
      errorName: "Error",
      message: "upstream failed",
    });
  });
});
