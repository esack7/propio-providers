import { MetaProvider } from "../providers/meta.js";
import type { ChatRequest, ChatStreamEvent } from "../types.js";
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
  model: "muse-spark-1.1",
  messages: [{ role: "user", content: "Hello" }],
};

const originalFetch = globalThis.fetch;
const originalApiKey = process.env.META_API_KEY;
const originalGenericApiKey = process.env.MODEL_API_KEY;
const createSseStream = OpenRouterTestFixture.createSseStream;

function createProvider(
  options: Partial<ConstructorParameters<typeof MetaProvider>[0]> = {},
): MetaProvider {
  return new MetaProvider({
    model: "muse-spark-1.1",
    contextWindowTokens: 1_048_576,
    apiKey: "meta-test-key",
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
  provider: MetaProvider,
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

describe("MetaProvider", () => {
  beforeEach(() => {
    delete process.env.META_API_KEY;
    delete process.env.MODEL_API_KEY;
    globalThis.fetch = originalFetch;
  });

  afterAll(() => {
    globalThis.fetch = originalFetch;
    if (originalApiKey === undefined) {
      delete process.env.META_API_KEY;
    } else {
      process.env.META_API_KEY = originalApiKey;
    }
    if (originalGenericApiKey === undefined) {
      delete process.env.MODEL_API_KEY;
    } else {
      process.env.MODEL_API_KEY = originalGenericApiKey;
    }
  });

  describe("constructor", () => {
    it("accepts an explicit key and configured context window", () => {
      const provider = createProvider();
      expect(provider.name).toBe("meta");
      expect(provider.getCapabilities().contextWindowTokens).toBe(1_048_576);
    });

    it("uses META_API_KEY when no explicit key is supplied", () => {
      process.env.META_API_KEY = "meta-env-key";
      const provider = new MetaProvider({
        model: "muse-spark-1.1",
        contextWindowTokens: 1_048_576,
      });
      expect(provider.name).toBe("meta");
    });

    it("rejects a missing key", () => {
      expect(
        () =>
          new MetaProvider({
            model: "muse-spark-1.1",
            contextWindowTokens: 1_048_576,
          }),
      ).toThrow(ProviderAuthenticationError);
    });

    it("does not use the generic MODEL_API_KEY variable", () => {
      process.env.MODEL_API_KEY = "generic-model-key";
      expect(
        () =>
          new MetaProvider({
            model: "muse-spark-1.1",
            contextWindowTokens: 1_048_576,
          }),
      ).toThrow(ProviderAuthenticationError);
    });
  });

  describe("request and stream mapping", () => {
    it("targets Meta with the explicit key and streams assistant text", async () => {
      process.env.META_API_KEY = "meta-env-key";
      globalThis.fetch = jest
        .fn()
        .mockResolvedValue(
          successfulResponse([
            'data: {"type":"response.output_text.delta","delta":"Hello Meta"}\n\n',
            'data: {"type":"response.completed","response":{"status":"completed"}}\n\n',
          ]),
        );

      const events = await collectEvents(createProvider());
      expect(events).toEqual([
        { type: "assistant_text", delta: "Hello Meta" },
        { type: "terminal", stopReason: "end_turn" },
      ]);
      expect(fetch).toHaveBeenCalledWith(
        "https://api.meta.ai/v1/responses",
        expect.objectContaining({
          method: "POST",
          headers: expect.objectContaining({
            Authorization: "Bearer meta-test-key",
            "Content-Type": "application/json",
          }),
        }),
      );
    });

    it("routes commentary and complete reasoning summaries separately from the final answer", async () => {
      const commentaryItem = {
        type: "message",
        id: "msg_commentary",
        role: "assistant",
        phase: "commentary",
      };
      const finalItem = {
        type: "message",
        id: "msg_final",
        role: "assistant",
        phase: "final",
      };
      globalThis.fetch = jest
        .fn()
        .mockResolvedValue(
          successfulResponse([
            `data: ${JSON.stringify({ type: "response.output_item.added", output_index: 0, item: commentaryItem })}\n\n`,
            'data: {"type":"response.reasoning_summary_text.delta","delta":"Checking "}\n\n',
            'data: {"type":"response.reasoning_summary_text.delta","delta":"both sources."}\n\n',
            'data: {"type":"response.output_text.delta","output_index":0,"delta":"I will "}\n\n',
            'data: {"type":"response.output_text.delta","output_index":0,"delta":"check them."}\n\n',
            `data: ${JSON.stringify({ type: "response.output_item.added", output_index: 1, item: finalItem })}\n\n`,
            'data: {"type":"response.output_text.delta","output_index":1,"delta":"Here is "}\n\n',
            'data: {"type":"response.output_text.delta","output_index":1,"delta":"the answer."}\n\n',
            'data: {"type":"response.completed","response":{"status":"completed"}}\n\n',
          ]),
        );

      const events = await collectEvents(createProvider());
      expect(events).toEqual([
        {
          type: "reasoning_summary",
          summary: "Checking ",
          source: "provider",
        },
        {
          type: "reasoning_summary",
          summary: "both sources.",
          source: "provider",
        },
        { type: "thinking_delta", delta: "I will " },
        { type: "thinking_delta", delta: "check them." },
        { type: "assistant_text", delta: "Here is " },
        { type: "assistant_text", delta: "the answer." },
        { type: "terminal", stopReason: "end_turn" },
      ]);
    });

    it("maps images, tools, reasoning summaries, and batched tool results", async () => {
      globalThis.fetch = jest
        .fn()
        .mockResolvedValue(
          successfulResponse([
            'data: {"type":"response.completed","response":{"status":"completed"}}\n\n',
          ]),
        );
      const signal = new AbortController().signal;

      await collectEvents(createProvider(), {
        model: "muse-spark-1.1",
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
            content: "I will look that up.",
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
        model: "muse-spark-1.1",
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
      expect(body.reasoning.effort).toBeUndefined();
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
        {
          role: "assistant",
          phase: "commentary",
          content: [{ type: "output_text", text: "I will look that up." }],
        },
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

    it("preserves reasoning, commentary, and parallel calls in output order", async () => {
      const reasoningItem = {
        type: "reasoning",
        id: "rs_1",
        encrypted_content: "opaque",
        summary: [],
      };
      const commentaryItem = {
        type: "message",
        id: "msg_1",
        role: "assistant",
        status: "completed",
        phase: "commentary",
        content: [{ type: "output_text", text: "I will check both." }],
      };
      const firstCall = {
        type: "function_call",
        id: "fc_1",
        call_id: "call_1",
        name: "lookup",
        arguments: '{"query":"first"}',
        status: "completed",
      };
      const secondCall = {
        type: "function_call",
        id: "fc_2",
        call_id: "call_2",
        name: "lookup",
        arguments: '{"query":"second"}',
        status: "completed",
      };
      globalThis.fetch = jest
        .fn()
        .mockResolvedValue(
          successfulResponse([
            `data: ${JSON.stringify({ type: "response.output_item.added", output_index: 1, item: commentaryItem })}\n\n`,
            'data: {"type":"response.reasoning_summary_text.delta","delta":"Checking "}\n\n',
            'data: {"type":"response.reasoning_summary_text.delta","delta":"both."}\n\n',
            'data: {"type":"response.output_text.delta","output_index":1,"delta":"I will check "}\n\n',
            'data: {"type":"response.output_text.delta","output_index":1,"delta":"both."}\n\n',
            `data: ${JSON.stringify({ type: "response.output_item.done", output_index: 0, item: reasoningItem })}\n\n`,
            `data: ${JSON.stringify({ type: "response.output_item.done", output_index: 1, item: commentaryItem })}\n\n`,
            `data: ${JSON.stringify({ type: "response.output_item.done", output_index: 2, item: firstCall })}\n\n`,
            `data: ${JSON.stringify({ type: "response.output_item.done", output_index: 3, item: secondCall })}\n\n`,
            'data: {"type":"response.completed","response":{"status":"completed"}}\n\n',
          ]),
        );

      const events = await collectEvents(createProvider());
      expect(events.slice(0, 4)).toEqual([
        {
          type: "reasoning_summary",
          summary: "Checking ",
          source: "provider",
        },
        {
          type: "reasoning_summary",
          summary: "both.",
          source: "provider",
        },
        { type: "thinking_delta", delta: "I will check " },
        { type: "thinking_delta", delta: "both." },
      ]);
      expect(events[4]).toEqual({
        type: "tool_calls",
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
        reasoningContent: JSON.stringify([
          reasoningItem,
          commentaryItem,
          firstCall,
          secondCall,
        ]),
      });
      expect(events[5]).toEqual({ type: "terminal", stopReason: "tool_use" });
    });

    it("replays completed output items exactly once before tool results", async () => {
      const replayItems = [
        {
          type: "reasoning",
          id: "rs_1",
          encrypted_content: "opaque",
          summary: [],
        },
        {
          type: "message",
          id: "msg_1",
          role: "assistant",
          status: "completed",
          phase: "commentary",
          content: [{ type: "output_text", text: "I will check both." }],
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
          type: "function_call",
          id: "fc_2",
          call_id: "call_2",
          name: "lookup",
          arguments: '{"query":"second"}',
          status: "completed",
        },
      ];
      globalThis.fetch = jest
        .fn()
        .mockResolvedValue(
          successfulResponse([
            'data: {"type":"response.completed","response":{"status":"completed"}}\n\n',
          ]),
        );

      await collectEvents(createProvider(), {
        model: "muse-spark-1.1",
        messages: [
          { role: "user", content: "Look up both" },
          {
            role: "assistant",
            content: "I will check both.",
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

    it("places synthesized legacy commentary before replayed function calls", async () => {
      const reasoningItem = {
        type: "reasoning",
        id: "rs_1",
        encrypted_content: "opaque",
        summary: [],
      };
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
            'data: {"type":"response.completed","response":{"status":"completed"}}\n\n',
          ]),
        );

      await collectEvents(createProvider(), {
        model: "muse-spark-1.1",
        messages: [
          { role: "user", content: "Look this up" },
          {
            role: "assistant",
            content: "I will look that up.",
            reasoningContent: JSON.stringify([reasoningItem, functionCallItem]),
            toolCalls: [
              {
                id: "call_1",
                function: { name: "lookup", arguments: { query: "value" } },
              },
            ],
          },
          { role: "tool", content: "result", toolCallId: "call_1" },
        ],
      });

      const body = JSON.parse((fetch as jest.Mock).mock.calls[0][1].body);
      expect(body.input).toEqual([
        { role: "user", content: "Look this up" },
        reasoningItem,
        {
          role: "assistant",
          phase: "commentary",
          content: [{ type: "output_text", text: "I will look that up." }],
        },
        functionCallItem,
        {
          type: "function_call_output",
          call_id: "call_1",
          output: "result",
        },
      ]);
    });

    it("passes arbitrary future Meta model IDs through unchanged", async () => {
      globalThis.fetch = jest
        .fn()
        .mockResolvedValue(
          successfulResponse([
            'data: {"type":"response.completed","response":{"status":"completed"}}\n\n',
          ]),
        );
      const provider = createProvider({
        model: "muse-future",
        contextWindowTokens: 2_000_000,
      });
      await collectEvents(provider, {
        model: "muse-future",
        messages: [{ role: "user", content: "Hello" }],
      });

      const body = JSON.parse((fetch as jest.Mock).mock.calls[0][1].body);
      expect(body.model).toBe("muse-future");
      expect(provider.getCapabilities().contextWindowTokens).toBe(2_000_000);
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
    it("maps authentication, rate-limit, model, context, and capacity failures", async () => {
      await expectStreamError(
        { ok: false, status: 401 },
        ProviderAuthenticationError,
      );
      await expectStreamError(
        { ok: false, status: 403 },
        ProviderAuthenticationError,
      );
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

    it("retries transient failures with Meta diagnostics", async () => {
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
          provider: "meta",
          model: "muse-spark-1.1",
          attemptNumber: 1,
        }),
      ]);
    });

    it("does not retry cancellation or invalid requests", async () => {
      globalThis.fetch = jest
        .fn()
        .mockRejectedValueOnce(
          new DOMException("This operation was aborted", "AbortError"),
        );
      const provider = createProvider({
        retryConfig: { maxRetries: 2, consecutive529Limit: 2, baseDelayMs: 0 },
      });
      await expect(collectEvents(provider)).rejects.toThrow(
        "Request cancelled",
      );
      expect(fetch).toHaveBeenCalledTimes(1);

      globalThis.fetch = jest.fn().mockResolvedValue({
        ok: false,
        status: 400,
        text: async () =>
          JSON.stringify({ error: { message: "Invalid request" } }),
      });
      await expect(collectEvents(provider)).rejects.toThrow(
        ProviderInvalidRequestError,
      );
      expect(fetch).toHaveBeenCalledTimes(1);
    });
  });
});
