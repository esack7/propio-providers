import { OpenAiProvider } from "../providers/openai.js";
import type {
  ChatRequest,
  ChatStreamEvent,
  ToolCallsStreamEvent,
} from "../types.js";
import {
  ProviderAuthenticationError,
  ProviderCapacityError,
  ProviderContextLengthError,
  ProviderInvalidRequestError,
  ProviderModelNotFoundError,
  ProviderRateLimitError,
} from "../types.js";
import { OpenRouterTestFixture } from "./openrouterTestHelpers.js";

const DEFAULT_REQUEST: ChatRequest = {
  model: "gpt-5.5",
  messages: [{ role: "user", content: "Hello" }],
};

const originalFetch = globalThis.fetch;
const originalApiKey = process.env.OPENAI_API_KEY;
const createSseStream = OpenRouterTestFixture.createSseStream;

function createProvider(
  options: Partial<ConstructorParameters<typeof OpenAiProvider>[0]> = {},
): OpenAiProvider {
  return new OpenAiProvider({
    model: "gpt-5.5",
    contextWindowTokens: 1_050_000,
    apiKey: "openai-test-key",
    retryConfig: { maxRetries: 0, consecutive529Limit: 1 },
    ...options,
  });
}

function successfulResponse(chunks: string[]): Partial<Response> {
  return {
    ok: true,
    status: 200,
    body: createSseStream(chunks),
  };
}

async function collectEvents(
  provider: OpenAiProvider,
  request: ChatRequest = DEFAULT_REQUEST,
): Promise<ChatStreamEvent[]> {
  const events: ChatStreamEvent[] = [];
  for await (const event of provider.streamChat(request)) {
    events.push(event);
  }
  return events;
}

async function expectStreamError(
  response: Partial<Response>,
  expected: new (...args: any[]) => Error,
): Promise<void> {
  globalThis.fetch = jest.fn().mockResolvedValue(response);
  await expect(collectEvents(createProvider())).rejects.toThrow(expected);
}

describe("OpenAiProvider", () => {
  beforeEach(() => {
    delete process.env.OPENAI_API_KEY;
    globalThis.fetch = originalFetch;
  });

  afterAll(() => {
    globalThis.fetch = originalFetch;
    if (originalApiKey === undefined) {
      delete process.env.OPENAI_API_KEY;
    } else {
      process.env.OPENAI_API_KEY = originalApiKey;
    }
  });

  describe("constructor", () => {
    it("accepts an explicit API key and configured context window", () => {
      const provider = createProvider();
      expect(provider.name).toBe("openai");
      expect(provider.getCapabilities().contextWindowTokens).toBe(1_050_000);
    });

    it("uses OPENAI_API_KEY when no explicit key is supplied", () => {
      process.env.OPENAI_API_KEY = "openai-env-key";
      const provider = new OpenAiProvider({
        model: "gpt-5.4-mini",
        contextWindowTokens: 400_000,
      });
      expect(provider.name).toBe("openai");
    });

    it("rejects a missing API key", () => {
      expect(
        () =>
          new OpenAiProvider({
            model: "gpt-5.5",
            contextWindowTokens: 1_050_000,
          }),
      ).toThrow(ProviderAuthenticationError);
    });
  });

  describe("request and stream mapping", () => {
    it("streams assistant text and exactly one terminal event", async () => {
      globalThis.fetch = jest
        .fn()
        .mockResolvedValue(
          successfulResponse([
            'data: {"type":"response.output_text.delta","delta":"Hello"}\n\n',
            'data: {"type":"response.output_text.delta","delta":" OpenAI"}\n\n',
            'data: {"type":"response.completed","response":{"status":"completed"}}\n\n',
          ]),
        );

      const events = await collectEvents(createProvider());
      expect(events).toEqual([
        { type: "assistant_text", delta: "Hello" },
        { type: "assistant_text", delta: " OpenAI" },
        { type: "terminal", stopReason: "end_turn" },
      ]);
      expect(fetch).toHaveBeenCalledWith(
        "https://api.openai.com/v1/responses",
        expect.objectContaining({
          method: "POST",
          headers: expect.objectContaining({
            Authorization: "Bearer openai-test-key",
            "Content-Type": "application/json",
          }),
        }),
      );
    });

    it("maps messages, images, tools, reasoning options, and batched results", async () => {
      globalThis.fetch = jest
        .fn()
        .mockResolvedValue(
          successfulResponse([
            'data: {"type":"response.completed","response":{"status":"completed"}}\n\n',
          ]),
        );
      const signal = new AbortController().signal;
      const reasoningItem = {
        type: "reasoning",
        id: "rs_1",
        encrypted_content: "encrypted",
        summary: [],
      };

      await collectEvents(createProvider(), {
        model: "gpt-5.5",
        signal,
        requestReasoning: true,
        tools: [
          {
            type: "function",
            function: {
              name: "lookup",
              description: "Look something up",
              parameters: {
                type: "object",
                properties: { query: { type: "string" } },
                required: ["query"],
              },
            },
          },
        ],
        messages: [
          { role: "system", content: "Be concise" },
          {
            role: "user",
            content: "Inspect this",
            images: ["https://example.com/image.png", new Uint8Array([1, 2])],
          },
          {
            role: "assistant",
            content: "",
            reasoningContent: JSON.stringify([reasoningItem]),
            toolCalls: [
              {
                id: "call_1",
                function: { name: "lookup", arguments: { query: "value" } },
              },
            ],
          },
          {
            role: "tool",
            content: "",
            toolResults: [
              {
                toolCallId: "call_1",
                toolName: "lookup",
                content: "result",
              },
            ],
          },
        ],
      });

      const init = (fetch as jest.Mock).mock.calls[0][1];
      const body = JSON.parse(init.body);
      expect(init.signal).toBe(signal);
      expect(body).toMatchObject({
        model: "gpt-5.5",
        stream: true,
        store: false,
        include: ["reasoning.encrypted_content"],
        reasoning: { summary: "auto" },
        tools: [
          {
            type: "function",
            name: "lookup",
            description: "Look something up",
          },
        ],
      });
      expect(body.input).toEqual([
        { role: "system", content: "Be concise" },
        {
          role: "user",
          content: [
            { type: "input_text", text: "Inspect this" },
            {
              type: "input_image",
              image_url: "https://example.com/image.png",
            },
            {
              type: "input_image",
              image_url: "data:image/png;base64,AQI=",
            },
          ],
        },
        reasoningItem,
        {
          type: "function_call",
          call_id: "call_1",
          name: "lookup",
          arguments: '{"query":"value"}',
          status: "completed",
        },
        {
          type: "function_call_output",
          call_id: "call_1",
          output: "result",
        },
      ]);
    });

    it("assembles function calls, preserves reasoning state, and emits summaries", async () => {
      const reasoningItem = {
        type: "reasoning",
        id: "rs_1",
        encrypted_content: "opaque-value",
        summary: [],
      };
      globalThis.fetch = jest
        .fn()
        .mockResolvedValue(
          successfulResponse([
            'data: {"type":"response.reasoning_summary_text.delta","delta":"Checking the tool."}\n\n',
            `data: ${JSON.stringify({ type: "response.output_item.done", output_index: 0, item: reasoningItem })}\n\n`,
            'data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":""}}\n\n',
            'data: {"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\\"query\\":"}\n\n',
            'data: {"type":"response.function_call_arguments.delta","output_index":1,"delta":"\\"value\\"}"}\n\n',
            'data: {"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{\\"query\\":\\"value\\"}"}}\n\n',
            'data: {"type":"response.completed","response":{"status":"completed"}}\n\n',
          ]),
        );

      const events = await collectEvents(createProvider());
      expect(events[0]).toEqual({
        type: "reasoning_summary",
        summary: "Checking the tool.",
        source: "provider",
      });
      expect(events[1]).toEqual({
        type: "tool_calls",
        toolCalls: [
          {
            id: "call_1",
            function: { name: "lookup", arguments: { query: "value" } },
          },
        ],
        reasoningContent: JSON.stringify([
          reasoningItem,
          {
            type: "function_call",
            id: "fc_1",
            call_id: "call_1",
            name: "lookup",
            arguments: '{"query":"value"}',
          },
        ]),
      });
      expect(events[2]).toEqual({ type: "terminal", stopReason: "tool_use" });
    });

    it("replays function calls without reasoning items", async () => {
      const functionCallItem = {
        type: "function_call",
        id: "fc_1",
        call_id: "call_1",
        name: "lookup",
        arguments: '{"query":"value"}',
        status: "completed",
      };
      globalThis.fetch = jest
        .fn()
        .mockResolvedValue(
          successfulResponse([
            `data: ${JSON.stringify({ type: "response.output_item.done", output_index: 0, item: functionCallItem })}\n\n`,
            'data: {"type":"response.completed","response":{"status":"completed"}}\n\n',
          ]),
        );

      const events = await collectEvents(createProvider());
      const toolCallsEvent = events[0] as ToolCallsStreamEvent;
      expect(toolCallsEvent).toEqual({
        type: "tool_calls",
        toolCalls: [
          {
            id: "call_1",
            function: { name: "lookup", arguments: { query: "value" } },
          },
        ],
        reasoningContent: JSON.stringify([functionCallItem]),
      });

      globalThis.fetch = jest
        .fn()
        .mockResolvedValue(
          successfulResponse([
            'data: {"type":"response.completed","response":{"status":"completed"}}\n\n',
          ]),
        );
      await collectEvents(createProvider(), {
        model: "gpt-5.5",
        messages: [
          { role: "user", content: "Look this up" },
          {
            role: "assistant",
            content: "",
            toolCalls: toolCallsEvent.toolCalls,
            reasoningContent: toolCallsEvent.reasoningContent,
          },
          { role: "tool", content: "result", toolCallId: "call_1" },
        ],
      });

      const body = JSON.parse((fetch as jest.Mock).mock.calls[0][1].body);
      expect(body.input).toEqual([
        { role: "user", content: "Look this up" },
        functionCallItem,
        {
          type: "function_call_output",
          call_id: "call_1",
          output: "result",
        },
      ]);
    });

    it("replays interleaved reasoning and parallel function calls in output order", async () => {
      globalThis.fetch = jest
        .fn()
        .mockResolvedValue(
          successfulResponse([
            'data: {"type":"response.completed","response":{"status":"completed"}}\n\n',
          ]),
        );
      const replayItems = [
        {
          type: "reasoning",
          id: "rs_1",
          encrypted_content: "first",
          summary: [],
        },
        {
          type: "function_call",
          id: "fc_1",
          call_id: "call_1",
          name: "lookup",
          arguments: '{"query":"first"}',
          status: "completed",
        },
        {
          type: "reasoning",
          id: "rs_2",
          encrypted_content: "second",
          summary: [],
        },
        {
          type: "function_call",
          id: "fc_2",
          call_id: "call_2",
          name: "lookup",
          arguments: '{"query":"second"}',
          status: "completed",
        },
      ];

      await collectEvents(createProvider(), {
        model: "gpt-5.5",
        messages: [
          { role: "user", content: "Look up both" },
          {
            role: "assistant",
            content: "",
            reasoningContent: JSON.stringify(replayItems),
            toolCalls: [
              {
                id: "call_1",
                function: { name: "lookup", arguments: { query: "first" } },
              },
              {
                id: "call_2",
                function: { name: "lookup", arguments: { query: "second" } },
              },
            ],
          },
          { role: "tool", content: "first result", toolCallId: "call_1" },
          { role: "tool", content: "second result", toolCallId: "call_2" },
        ],
      });

      const body = JSON.parse((fetch as jest.Mock).mock.calls[0][1].body);
      expect(body.input).toEqual([
        { role: "user", content: "Look up both" },
        ...replayItems,
        {
          type: "function_call_output",
          call_id: "call_1",
          output: "first result",
        },
        {
          type: "function_call_output",
          call_id: "call_2",
          output: "second result",
        },
      ]);
    });

    it("passes arbitrary future model IDs through unchanged", async () => {
      globalThis.fetch = jest
        .fn()
        .mockResolvedValue(
          successfulResponse([
            'data: {"type":"response.completed","response":{"status":"completed"}}\n\n',
          ]),
        );
      const provider = createProvider({
        model: "gpt-future-general",
        contextWindowTokens: 2_000_000,
      });
      await collectEvents(provider, {
        model: "gpt-future-general",
        messages: [{ role: "user", content: "Hello" }],
      });

      const body = JSON.parse((fetch as jest.Mock).mock.calls[0][1].body);
      expect(body.model).toBe("gpt-future-general");
    });

    it.each([
      ["response.incomplete", "max_tokens"],
      ["response.failed", "error"],
      ["response.cancelled", "error"],
    ])("maps %s to %s", async (eventType, stopReason) => {
      globalThis.fetch = jest
        .fn()
        .mockResolvedValue(
          successfulResponse([
            `data: ${JSON.stringify({ type: eventType })}\n\n`,
          ]),
        );
      const events = await collectEvents(createProvider());
      expect(events.at(-1)).toEqual({ type: "terminal", stopReason });
    });
  });

  describe("errors and retries", () => {
    it("maps authentication failures", async () => {
      await expectStreamError(
        { ok: false, status: 401 },
        ProviderAuthenticationError,
      );
      await expectStreamError(
        { ok: false, status: 403 },
        ProviderAuthenticationError,
      );
    });

    it("maps rate limits and missing models", async () => {
      await expectStreamError(
        {
          ok: false,
          status: 429,
          headers: new Headers({ "retry-after": "3" }),
        },
        ProviderRateLimitError,
      );
      await expectStreamError(
        { ok: false, status: 404 },
        ProviderModelNotFoundError,
      );
    });

    it("maps context length and capacity failures", async () => {
      await expectStreamError(
        {
          ok: false,
          status: 400,
          text: async () =>
            JSON.stringify({
              error: { message: "Maximum context length exceeded" },
            }),
        },
        ProviderContextLengthError,
      );
      await expectStreamError(
        { ok: false, status: 503 },
        ProviderCapacityError,
      );
    });

    it("retries transient failures and emits diagnostics", async () => {
      const diagnostics: unknown[] = [];
      globalThis.fetch = jest
        .fn()
        .mockResolvedValueOnce({
          ok: false,
          status: 503,
          text: async () => "temporarily unavailable",
        })
        .mockResolvedValueOnce(
          successfulResponse([
            'data: {"type":"response.completed","response":{"status":"completed"}}\n\n',
          ]),
        );
      const provider = createProvider({
        retryConfig: {
          maxRetries: 1,
          consecutive529Limit: 2,
          baseDelayMs: 0,
        },
        onDiagnosticEvent: (event) => diagnostics.push(event),
      });

      await collectEvents(provider);
      expect(fetch).toHaveBeenCalledTimes(2);
      expect(diagnostics).toEqual([
        expect.objectContaining({
          type: "provider_retry",
          provider: "openai",
          model: "gpt-5.5",
          attemptNumber: 1,
        }),
      ]);
    });

    it("honors cancellation without retrying", async () => {
      globalThis.fetch = jest
        .fn()
        .mockRejectedValue(
          new DOMException("This operation was aborted", "AbortError"),
        );
      const provider = createProvider({
        retryConfig: { maxRetries: 2, consecutive529Limit: 2, baseDelayMs: 0 },
      });

      await expect(collectEvents(provider)).rejects.toThrow(
        "Request cancelled",
      );
      expect(fetch).toHaveBeenCalledTimes(1);
    });

    it("does not retry invalid requests", async () => {
      globalThis.fetch = jest.fn().mockResolvedValue({
        ok: false,
        status: 400,
        text: async () =>
          JSON.stringify({ error: { message: "Invalid request" } }),
      });
      const provider = createProvider({
        retryConfig: { maxRetries: 3, consecutive529Limit: 2, baseDelayMs: 0 },
      });
      await expect(collectEvents(provider)).rejects.toThrow(
        ProviderInvalidRequestError,
      );
      expect(fetch).toHaveBeenCalledTimes(1);
    });
  });
});
