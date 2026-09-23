import { XaiProvider } from "../providers/xai.js";
import type { ProviderTraceEvent } from "../trace.js";
import { ProviderCapacityError, ProviderRateLimitError } from "../types.js";
import {
  OPENAI_COMPATIBLE_PROVIDER_TEST_ENV,
  OpenRouterTestFixture,
  ProviderAuthenticationError,
  ProviderError,
  registerAcceptsApiKeyTest,
  registerOpenAiCompatibleStreamErrorTests,
  registerOpenAiCompatibleToolResultExpansionTest,
  registerProviderTestLifecycle,
  setupOpenAiCompatibleProviderTests,
  type ChatRequest,
} from "./openAiCompatibleTestHelpers.js";

const { originalEnv, originalFetch } = OPENAI_COMPATIBLE_PROVIDER_TEST_ENV;
const DEFAULT_MODEL = "grok-4-1-fast-reasoning";
const DEFAULT_CONTEXT_WINDOW = 2_000_000;
const DEFAULT_REQUEST: ChatRequest = {
  model: DEFAULT_MODEL,
  messages: [{ role: "user", content: "Hi" }],
};

const createSseStream = OpenRouterTestFixture.createSseStream;

function createProvider(
  options: Partial<ConstructorParameters<typeof XaiProvider>[0]> = {},
): XaiProvider {
  return new XaiProvider({
    model: DEFAULT_MODEL,
    contextWindowTokens: DEFAULT_CONTEXT_WINDOW,
    apiKey: "xai-test",
    ...options,
  });
}

const {
  createRequest,
  expectStreamChatToThrow,
  expectRequestError,
  expectProviderErrorAndMessage,
  collectToolMessages,
} = setupOpenAiCompatibleProviderTests({
  createProvider,
  defaultRequest: DEFAULT_REQUEST,
});

describe("XaiProvider", () => {
  registerProviderTestLifecycle(originalEnv, originalFetch);

  describe("constructor", () => {
    registerAcceptsApiKeyTest({
      expectedName: "xai",
      createProvider: () =>
        new XaiProvider({
          model: "grok-4-1-fast-reasoning",
          contextWindowTokens: DEFAULT_CONTEXT_WINDOW,
          apiKey: "xai-test-key",
        }),
    });

    it("should use XAI_API_KEY env var when apiKey not in options", () => {
      process.env.XAI_API_KEY = "xai-env-key";
      const provider = new XaiProvider({
        model: "grok-4-1-fast-reasoning",
        contextWindowTokens: DEFAULT_CONTEXT_WINDOW,
      });
      expect(provider.name).toBe("xai");
    });

    it("should throw ProviderAuthenticationError when no API key is provided", () => {
      delete process.env.XAI_API_KEY;
      expect(() => {
        new XaiProvider({
          model: "grok-4-1-fast-reasoning",
          contextWindowTokens: DEFAULT_CONTEXT_WINDOW,
        });
      }).toThrow(ProviderAuthenticationError);
      expect(() => {
        new XaiProvider({
          model: "grok-4-1-fast-reasoning",
          contextWindowTokens: DEFAULT_CONTEXT_WINDOW,
        });
      }).toThrow(/API key|xAI/);
    });

    it("should report the configured context window for current Grok models", () => {
      const provider = new XaiProvider({
        model: "grok-4-1-fast-reasoning",
        contextWindowTokens: DEFAULT_CONTEXT_WINDOW,
        apiKey: "xai-test-key",
      });

      expect(provider.getCapabilities().contextWindowTokens).toBe(2_000_000);
    });

    it("should report the configured context window for newly configured xAI models", () => {
      const provider = new XaiProvider({
        model: "grok-4.3",
        contextWindowTokens: 1_000_000,
        apiKey: "xai-test-key",
      });

      expect(provider.getCapabilities().contextWindowTokens).toBe(1_000_000);
    });
  });

  describe("streamChat()", () => {
    it("should yield content deltas from mocked SSE stream", async () => {
      const chunks = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
        'data: {"choices":[{"delta":{"content":" Grok"}}]}\n\n',
        "data: [DONE]\n\n",
      ];
      globalThis.fetch = jest.fn().mockResolvedValue({
        ok: true,
        body: createSseStream(chunks),
      });

      const provider = createProvider();
      const request = createRequest({
        messages: [{ role: "user", content: "Hello" }],
      });
      const deltas: string[] = [];
      for await (const chunk of provider.streamChat(request)) {
        deltas.push(chunk.delta);
      }
      expect(deltas).toEqual(["Hello", " Grok"]);
    });

    it("should use the Responses API and emit thinking deltas when reasoning is requested", async () => {
      globalThis.fetch = jest.fn().mockResolvedValue({
        ok: true,
        body: createSseStream([
          'data: {"type":"response.reasoning_summary_text.delta","delta":"Planning. "}\n\n',
          'data: {"type":"response.output_text.delta","delta":"Answer."}\n\n',
          'data: {"type":"response.completed","response":{"status":"completed"}}\n\n',
        ]),
      });

      const provider = createProvider();
      const thinkingEvents: string[] = [];
      const assistantText: string[] = [];

      for await (const chunk of provider.streamChat(
        createRequest({ requestReasoning: true }),
      )) {
        if (chunk.type === "thinking_delta") {
          thinkingEvents.push(chunk.delta);
        }
        if (chunk.type === "assistant_text") {
          assistantText.push(chunk.delta);
        }
      }

      expect(thinkingEvents).toEqual(["Planning. "]);
      expect(assistantText).toEqual(["Answer."]);
      expect(fetch).toHaveBeenCalledWith(
        "https://api.x.ai/v1/responses",
        expect.objectContaining({
          method: "POST",
          headers: expect.objectContaining({
            Authorization: "Bearer xai-test",
          }),
        }),
      );
      const requestBody = JSON.parse(
        (fetch as jest.Mock).mock.calls[0][1].body,
      );
      expect(requestBody.stream).toBe(true);
      expect(requestBody.input).toEqual([
        { role: "user", content: [{ type: "input_text", text: "Hi" }] },
      ]);
    });

    it("should call the xAI API endpoint with correct auth header", async () => {
      const chunks = [
        'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n',
        "data: [DONE]\n\n",
      ];
      globalThis.fetch = jest.fn().mockResolvedValue({
        ok: true,
        body: createSseStream(chunks),
      });

      const provider = createProvider();
      for await (const chunk of provider.streamChat(createRequest())) {
        // consume
      }

      expect(fetch).toHaveBeenCalledWith(
        "https://api.x.ai/v1/chat/completions",
        expect.objectContaining({
          method: "POST",
          headers: expect.objectContaining({
            Authorization: "Bearer xai-test",
            "Content-Type": "application/json",
          }),
        }),
      );
    });

    registerOpenAiCompatibleToolResultExpansionTest({ collectToolMessages });

    registerOpenAiCompatibleStreamErrorTests({
      createProvider,
      expectRequestError,
      expectProviderErrorAndMessage,
      expectStreamChatToThrow,
      defaultRequest: DEFAULT_REQUEST,
      contextLengthErrorMessage:
        "This model's maximum context length is 131072 tokens. However, your messages resulted in 200000 tokens.",
    });

    it("should fall back to a regional endpoint when the global endpoint returns 503", async () => {
      const traceEvents: ProviderTraceEvent[] = [];
      const successChunks = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
        "data: [DONE]\n\n",
      ];
      const successStream = new ReadableStream({
        start(controller) {
          successChunks.forEach((c) =>
            controller.enqueue(new TextEncoder().encode(c)),
          );
          controller.close();
        },
      });

      globalThis.fetch = jest
        .fn()
        .mockResolvedValueOnce({
          ok: false,
          status: 503,
          text: () => Promise.resolve("upstream connect error"),
        })
        .mockResolvedValueOnce({
          ok: true,
          body: successStream,
        });

      const provider = new XaiProvider({
        model: "grok-4-1-fast-reasoning",
        contextWindowTokens: DEFAULT_CONTEXT_WINDOW,
        apiKey: "xai-test",
      });

      const deltas: string[] = [];
      for await (const chunk of provider.streamChat({
        model: "grok-4-1-fast-reasoning",
        messages: [{ role: "user", content: "Hi" }],
        captureRequestPayload: true,
        trace: {
          requestId: "request-1",
          operationId: "operation-1",
          purpose: "answer",
        },
        onTraceEvent: (event) => traceEvents.push(event),
      })) {
        deltas.push(chunk.delta);
      }

      expect(deltas).toEqual(["Hello"]);
      expect(traceEvents).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            type: "provider_attempt_payload",
            transport: "http_json",
            requestBody: expect.objectContaining({ stream: true }),
          }),
        ]),
      );
      expect(fetch).toHaveBeenNthCalledWith(
        1,
        "https://api.x.ai/v1/chat/completions",
        expect.any(Object),
      );
      expect(fetch).toHaveBeenNthCalledWith(
        2,
        "https://us-east-1.api.x.ai/v1/chat/completions",
        expect.any(Object),
      );
      expect(fetch).toHaveBeenCalledTimes(2);
      const attempts = traceEvents.filter(
        (event) => event.type === "provider_attempt_started",
      );
      expect(attempts).toHaveLength(2);
      expect(attempts.map((event) => event.endpointClass)).toEqual([
        "chat_completions:global",
        "chat_completions:us_east_1",
      ]);
      expect(new Set(attempts.map((event) => event.attemptId)).size).toBe(2);
    });

    it("retries rate limits on the global endpoint without regional bursts", async () => {
      const traceEvents: ProviderTraceEvent[] = [];
      const random = jest.spyOn(Math, "random").mockReturnValue(0.5);
      globalThis.fetch = jest.fn().mockResolvedValue({
        ok: false,
        status: 429,
        headers: new Map([["retry-after", "1"]]),
        text: () => Promise.resolve("rate limited"),
      });

      await expectStreamChatToThrow(
        createProvider({
          retryConfig: {
            maxRetries: 1,
            consecutive529Limit: 3,
            baseDelayMs: 10,
          },
        }),
        ProviderRateLimitError,
        {
          ...DEFAULT_REQUEST,
          trace: {
            requestId: "request-1",
            operationId: "operation-1",
            purpose: "answer",
          },
          onTraceEvent: (event) => traceEvents.push(event),
        },
      );

      expect(fetch).toHaveBeenCalledTimes(2);
      expect((fetch as jest.Mock).mock.calls.map(([url]) => url)).toEqual([
        "https://api.x.ai/v1/chat/completions",
        "https://api.x.ai/v1/chat/completions",
      ]);
      expect(
        traceEvents
          .filter((event) => event.type === "provider_attempt_started")
          .map((event) => event.endpointClass),
      ).toEqual(["chat_completions:global", "chat_completions:global"]);
      expect(
        traceEvents.find((event) => event.type === "provider_retry_wait"),
      ).toEqual(expect.objectContaining({ delayMs: 5 }));
      random.mockRestore();
    });

    it("does not retry cancelled requests", async () => {
      const abortError = new Error("The operation was aborted");
      abortError.name = "AbortError";
      globalThis.fetch = jest.fn().mockRejectedValue(abortError);

      await expectStreamChatToThrow(
        createProvider({
          retryConfig: {
            maxRetries: 3,
            consecutive529Limit: 3,
            baseDelayMs: 0,
          },
        }),
        /Request cancelled/,
      );

      expect(fetch).toHaveBeenCalledTimes(1);
    });

    it("counts consecutive 529 limits in complete regional sweeps", async () => {
      globalThis.fetch = jest.fn().mockResolvedValue({
        ok: false,
        status: 529,
        text: () => Promise.resolve("overloaded"),
      });

      await expectStreamChatToThrow(
        createProvider({
          retryConfig: {
            maxRetries: 5,
            consecutive529Limit: 2,
            baseDelayMs: 0,
          },
        }),
        ProviderCapacityError,
      );

      expect((fetch as jest.Mock).mock.calls.map(([url]) => url)).toEqual([
        "https://api.x.ai/v1/chat/completions",
        "https://us-east-1.api.x.ai/v1/chat/completions",
        "https://eu-west-1.api.x.ai/v1/chat/completions",
        "https://api.x.ai/v1/chat/completions",
        "https://us-east-1.api.x.ai/v1/chat/completions",
        "https://eu-west-1.api.x.ai/v1/chat/completions",
      ]);
    });
  });

  it("reports xAI response identity and token usage", async () => {
    const traceEvents: ProviderTraceEvent[] = [];
    globalThis.fetch = jest.fn().mockResolvedValue({
      ok: true,
      body: createSseStream([
        'data: {"id":"xai-response-1","model":"grok-4-1-fast-reasoning","choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18,"completion_tokens_details":{"reasoning_tokens":4}}}\n\n',
        "data: [DONE]\n\n",
      ]),
    });

    for await (const _event of createProvider().streamChat({
      ...DEFAULT_REQUEST,
      trace: {
        requestId: "request-1",
        operationId: "operation-1",
        purpose: "answer",
      },
      onTraceEvent: (event) => traceEvents.push(event),
    })) {
      // consume
    }

    expect(traceEvents).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: "provider_response_metadata",
          endpointClass: "chat_completions:global",
          upstreamRequestId: "xai-response-1",
        }),
        expect.objectContaining({
          type: "provider_usage_reported",
          availability: "reported",
          usage: expect.objectContaining({
            inputTokens: 11,
            outputTokens: 7,
            reasoningTokens: 4,
          }),
        }),
      ]),
    );
  });
});
