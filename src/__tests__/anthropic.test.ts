// Use jest.unstable_mockModule for ESM-compatible mocking of the Anthropic SDK.
// jest.mock() hoisting doesn't work reliably for default exports from CJS-with-__esModule modules.

class MockAPIError extends Error {
  status: number;
  headers: Record<string, string>;
  constructor(
    status: number,
    message: string,
    headers: Record<string, string> = {},
  ) {
    super(message);
    this.name = "APIError";
    this.status = status;
    this.headers = headers;
  }
}

const mockStream = jest.fn();
const MockAnthropicConstructor = jest.fn().mockImplementation(() => ({
  messages: { create: mockStream, stream: mockStream },
}));
(MockAnthropicConstructor as any).APIError = MockAPIError;

jest.unstable_mockModule("@anthropic-ai/sdk", () => ({
  default: MockAnthropicConstructor,
  Anthropic: MockAnthropicConstructor,
  APIError: MockAPIError,
}));

import {
  ChatMessage,
  ChatRequest,
  ChatStreamEvent,
  ProviderAuthenticationError,
  ProviderRateLimitError,
  ProviderCapacityError,
  ProviderModelNotFoundError,
  ProviderContextLengthError,
  ProviderError,
} from "../types.js";

let AnthropicProvider: any;

const SONNET_46_MODEL = "claude-sonnet-4-6";
const HAIKU_45_MODEL = "claude-haiku-4-5-20251001";
const OPUS_48_MODEL = "claude-opus-4-8";

beforeAll(async () => {
  const mod = await import("../providers/anthropic.js");
  AnthropicProvider = mod.AnthropicProvider;
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function createTestProvider(options: Record<string, unknown> = {}) {
  return new AnthropicProvider({
    model: SONNET_46_MODEL,
    contextWindowTokens: 200_000,
    apiKey: "test-key",
    ...options,
  });
}

function createChatRequest(
  content = "Hello",
  options: Partial<ChatRequest> = {},
): ChatRequest {
  return {
    messages: [{ role: "user", content }],
    model: SONNET_46_MODEL,
    ...options,
  };
}

function makeStream(events: object[]) {
  return {
    [Symbol.asyncIterator]: async function* () {
      for (const e of events) yield e;
    },
  };
}

function makeErrorStream(error: Error) {
  return {
    [Symbol.asyncIterator]: async function* (): AsyncGenerator<never> {
      throw error;
    },
  };
}

async function collectStream(
  provider: any,
  request: ChatRequest,
): Promise<ChatStreamEvent[]> {
  const events: ChatStreamEvent[] = [];
  for await (const e of provider.streamChat(request)) {
    events.push(e as ChatStreamEvent);
  }
  return events;
}

function getTerminalEvent(events: ChatStreamEvent[]) {
  return events.find((event) => event.type === "terminal") as any;
}

function getToolCallEvent(events: ChatStreamEvent[]) {
  return events.find((event) => event.type === "tool_calls") as any;
}

function getToolCallReasoningBlocks(events: ChatStreamEvent[]) {
  return JSON.parse(getToolCallEvent(events).reasoningContent);
}

function expectAdaptiveThinkingRequest(callArgs: any) {
  expect(callArgs.max_tokens).toBe(16384);
  expect(callArgs.thinking).toEqual(
    expect.objectContaining({
      type: "adaptive",
      display: "summarized",
    }),
  );
  expect(callArgs.thinking).not.toHaveProperty("budget_tokens");
  expect(callArgs.output_config).toEqual({ effort: "high" });
}

function textStreamEvents(text = "Hello!") {
  return [
    {
      type: "content_block_start",
      index: 0,
      content_block: { type: "text", text: "" },
    },
    {
      type: "content_block_delta",
      index: 0,
      delta: { type: "text_delta", text },
    },
    { type: "content_block_stop", index: 0 },
    {
      type: "message_delta",
      delta: { stop_reason: "end_turn", stop_sequence: null },
      usage: {},
    },
    { type: "message_stop" },
  ];
}

function toolCallStreamEvents(
  name = "my_tool",
  args = '{"x":1}',
  id = "call-1",
) {
  return [
    {
      type: "content_block_start",
      index: 0,
      content_block: { type: "tool_use", id, name, input: {} },
    },
    {
      type: "content_block_delta",
      index: 0,
      delta: { type: "input_json_delta", partial_json: args },
    },
    { type: "content_block_stop", index: 0 },
    {
      type: "message_delta",
      delta: { stop_reason: "tool_use", stop_sequence: null },
      usage: {},
    },
    { type: "message_stop" },
  ];
}

function emptyInputToolCallStreamEvents(name = "snapshot", id = "call-empty") {
  return [
    {
      type: "content_block_start",
      index: 0,
      content_block: { type: "tool_use", id, name, input: {} },
    },
    { type: "content_block_stop", index: 0 },
    {
      type: "message_delta",
      delta: { stop_reason: "tool_use", stop_sequence: null },
      usage: {},
    },
    { type: "message_stop" },
  ];
}

function thinkingToolCallStreamEvents(
  thinkingText = "Let me think",
  signature = "sig-abc",
  toolName = "my_tool",
  toolId = "call-1",
) {
  return [
    {
      type: "content_block_start",
      index: 0,
      content_block: { type: "thinking", thinking: "", signature: "" },
    },
    {
      type: "content_block_delta",
      index: 0,
      delta: { type: "thinking_delta", thinking: thinkingText },
    },
    {
      type: "content_block_delta",
      index: 0,
      delta: { type: "signature_delta", signature },
    },
    { type: "content_block_stop", index: 0 },
    ...toolCallStreamTail(toolName, toolId),
  ];
}

function redactedThinkingToolCallStreamEvents(
  data = "redacted-blob",
  toolName = "my_tool",
  toolId = "call-1",
) {
  return [
    {
      type: "content_block_start",
      index: 0,
      content_block: { type: "redacted_thinking", data },
    },
    ...toolCallStreamTail(toolName, toolId),
  ];
}

function toolCallStreamTail(toolName = "my_tool", toolId = "call-1") {
  return [
    {
      type: "content_block_start",
      index: 1,
      content_block: {
        type: "tool_use",
        id: toolId,
        name: toolName,
        input: {},
      },
    },
    {
      type: "content_block_delta",
      index: 1,
      delta: { type: "input_json_delta", partial_json: '{"x":1}' },
    },
    { type: "content_block_stop", index: 1 },
    {
      type: "message_delta",
      delta: { stop_reason: "tool_use", stop_sequence: null },
      usage: {},
    },
    { type: "message_stop" },
  ];
}

// ---------------------------------------------------------------------------

describe("AnthropicProvider", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockStream.mockReturnValue(makeStream(textStreamEvents()));
  });

  // ---------------------------------------------------------------------------
  describe("Initialization", () => {
    it("uses apiKey from options", () => {
      createTestProvider({ apiKey: "sk-test" });
      expect(MockAnthropicConstructor).toHaveBeenCalledWith(
        expect.objectContaining({ apiKey: "sk-test" }),
      );
    });

    it("falls back to ANTHROPIC_API_KEY env var", () => {
      process.env.ANTHROPIC_API_KEY = "env-key";
      new AnthropicProvider({ model: "m", contextWindowTokens: 1000 });
      expect(MockAnthropicConstructor).toHaveBeenCalledWith(
        expect.objectContaining({ apiKey: "env-key" }),
      );
      delete process.env.ANTHROPIC_API_KEY;
    });

    it("throws ProviderAuthenticationError when no key available", () => {
      delete process.env.ANTHROPIC_API_KEY;
      expect(
        () => new AnthropicProvider({ model: "m", contextWindowTokens: 1000 }),
      ).toThrow(ProviderAuthenticationError);
    });

    it("exposes name = anthropic", () => {
      expect(createTestProvider().name).toBe("anthropic");
    });

    it("exposes contextWindowTokens via getCapabilities", () => {
      const p = createTestProvider({ contextWindowTokens: 200_000 });
      expect(p.getCapabilities().contextWindowTokens).toBe(200_000);
    });
  });

  // ---------------------------------------------------------------------------
  describe("streamChat — text response", () => {
    it("yields assistant_text deltas", async () => {
      mockStream.mockReturnValue(makeStream(textStreamEvents("Hi!")));
      const events = await collectStream(
        createTestProvider(),
        createChatRequest(),
      );
      const textEvents = events.filter((e) => e.type === "assistant_text");
      expect(textEvents).toHaveLength(1);
      expect((textEvents[0] as any).delta).toBe("Hi!");
    });

    it("yields terminal event with end_turn stop reason", async () => {
      const events = await collectStream(
        createTestProvider(),
        createChatRequest(),
      );
      const terminal = events.find((e) => e.type === "terminal") as any;
      expect(terminal).toBeDefined();
      expect(terminal.stopReason).toBe("end_turn");
    });

    it("populates rawProviderReason on terminal event", async () => {
      const events = await collectStream(
        createTestProvider(),
        createChatRequest(),
      );
      const terminal = events.find((e) => e.type === "terminal") as any;
      expect(terminal.rawProviderReason).toBe("end_turn");
    });

    it("reads stop_reason from message_delta event", async () => {
      const customEvents = [...textStreamEvents("ok")];
      const idx = customEvents.findIndex(
        (e) => (e as any).type === "message_delta",
      );
      customEvents[idx] = {
        type: "message_delta",
        delta: { stop_reason: "max_tokens", stop_sequence: null },
        usage: {},
      };
      mockStream.mockReturnValue(makeStream(customEvents));
      const events = await collectStream(
        createTestProvider(),
        createChatRequest(),
      );
      const terminal = getTerminalEvent(events);
      expect(terminal.stopReason).toBe("max_tokens");
    });

    it("passes abort signal to the SDK call", async () => {
      const controller = new AbortController();
      await collectStream(
        createTestProvider(),
        createChatRequest("hi", { signal: controller.signal }),
      );
      expect(mockStream).toHaveBeenCalledWith(
        expect.any(Object),
        expect.objectContaining({ signal: controller.signal }),
      );
    });

    it("passes stream=true to the SDK call", async () => {
      await collectStream(createTestProvider(), createChatRequest());
      expect(mockStream).toHaveBeenCalledWith(
        expect.objectContaining({ stream: true }),
        expect.any(Object),
      );
    });

    it("uses max_tokens >= 16384 for non-reasoning requests", async () => {
      await collectStream(createTestProvider(), createChatRequest());
      const callArgs = mockStream.mock.calls[0][0];
      expect(callArgs.max_tokens).toBeGreaterThanOrEqual(16384);
    });
  });

  // ---------------------------------------------------------------------------
  describe("streamChat — tool calls", () => {
    it("yields tool_calls event with correct name, args and id", async () => {
      mockStream.mockReturnValue(
        makeStream(toolCallStreamEvents("search", '{"q":"test"}', "call-99")),
      );
      const events = await collectStream(
        createTestProvider(),
        createChatRequest(),
      );
      const toolEvent = events.find((e) => e.type === "tool_calls") as any;
      expect(toolEvent).toBeDefined();
      expect(toolEvent.toolCalls[0].function.name).toBe("search");
      expect(toolEvent.toolCalls[0].function.arguments).toEqual({ q: "test" });
      expect(toolEvent.toolCalls[0].id).toBe("call-99");
    });

    it("yields tool_calls for empty-input tools with no input_json_delta", async () => {
      mockStream.mockReturnValue(
        makeStream(emptyInputToolCallStreamEvents("browser_snapshot")),
      );
      const events = await collectStream(
        createTestProvider(),
        createChatRequest(),
      );
      const toolEvent = getToolCallEvent(events);
      expect(toolEvent).toBeDefined();
      expect(toolEvent.toolCalls[0].function.name).toBe("browser_snapshot");
      expect(toolEvent.toolCalls[0].function.arguments).toEqual({});
    });

    it("sets terminal stop_reason to tool_use", async () => {
      mockStream.mockReturnValue(makeStream(toolCallStreamEvents()));
      const events = await collectStream(
        createTestProvider(),
        createChatRequest(),
      );
      const terminal = getTerminalEvent(events);
      expect(terminal.stopReason).toBe("tool_use");
    });
  });

  // ---------------------------------------------------------------------------
  describe("streamChat — extended thinking", () => {
    it("yields thinking_delta events before tool_calls", async () => {
      mockStream.mockReturnValue(
        makeStream(thinkingToolCallStreamEvents("Hmm")),
      );
      const events = await collectStream(
        createTestProvider(),
        createChatRequest("think", { requestReasoning: true }),
      );
      const thinkingEvents = events.filter((e) => e.type === "thinking_delta");
      const toolEvent = events.find((e) => e.type === "tool_calls");
      expect(thinkingEvents.length).toBeGreaterThan(0);
      expect((thinkingEvents[0] as any).delta).toBe("Hmm");
      expect(events.indexOf(thinkingEvents[0])).toBeLessThan(
        events.indexOf(toolEvent!),
      );
    });

    it("uses adaptive thinking for Sonnet 4.6 reasoning requests", async () => {
      mockStream.mockReturnValue(
        makeStream(thinkingToolCallStreamEvents("my thoughts", "sig-123")),
      );
      const events = await collectStream(
        createTestProvider(),
        createChatRequest("think", { requestReasoning: true }),
      );
      const callArgs = mockStream.mock.calls[0][0];
      expectAdaptiveThinkingRequest(callArgs);
      const blocks = getToolCallReasoningBlocks(events);
      expect(blocks[0]).toMatchObject({
        thinking: "my thoughts",
        signature: "sig-123",
      });
    });

    it("uses adaptive thinking for Opus 4.8 reasoning requests", async () => {
      mockStream.mockReturnValue(makeStream(thinkingToolCallStreamEvents()));
      await collectStream(
        createTestProvider(),
        createChatRequest("think", {
          model: OPUS_48_MODEL,
          requestReasoning: true,
        }),
      );
      const callArgs = mockStream.mock.calls[0][0];
      expectAdaptiveThinkingRequest(callArgs);
    });

    it("keeps manual enabled thinking for Haiku 4.5 reasoning requests", async () => {
      mockStream.mockReturnValue(makeStream(thinkingToolCallStreamEvents()));
      await collectStream(
        createTestProvider(),
        createChatRequest("think", {
          model: HAIKU_45_MODEL,
          requestReasoning: true,
        }),
      );
      const callArgs = mockStream.mock.calls[0][0];
      expect(callArgs.max_tokens).toBeGreaterThan(
        callArgs.thinking.budget_tokens,
      );
      expect(callArgs.thinking).toEqual(
        expect.objectContaining({
          type: "enabled",
          budget_tokens: 10000,
        }),
      );
      expect(callArgs.output_config).toBeUndefined();
    });

    it("preserves redacted_thinking blocks in reasoningContent", async () => {
      mockStream.mockReturnValue(
        makeStream(redactedThinkingToolCallStreamEvents("redacted-123")),
      );
      const events = await collectStream(
        createTestProvider(),
        createChatRequest("think", { requestReasoning: true }),
      );
      const blocks = getToolCallReasoningBlocks(events);
      expect(blocks).toContainEqual({
        type: "redacted_thinking",
        data: "redacted-123",
      });
    });

    it("omits thinking param when requestReasoning is not set", async () => {
      await collectStream(createTestProvider(), createChatRequest());
      expect(mockStream.mock.calls[0][0].thinking).toBeUndefined();
      expect(mockStream.mock.calls[0][0].output_config).toBeUndefined();
    });

    it("does not replay thinking blocks when thinking is not requested", async () => {
      await collectStream(createTestProvider(), {
        model: "claude-sonnet-4-6",
        messages: [
          {
            role: "assistant",
            content: "",
            reasoningContent: JSON.stringify([
              { thinking: "prior reasoning", signature: "sig-prior" },
            ]),
            toolCalls: [
              {
                id: "tc-1",
                function: { name: "search", arguments: { q: "test" } },
              },
            ],
          },
          {
            role: "tool",
            content: "",
            toolResults: [
              { toolCallId: "tc-1", toolName: "search", content: "result" },
            ],
          },
          { role: "user", content: "continue" },
        ],
      });

      const assistantMessage = mockStream.mock.calls[0][0].messages[0];
      expect(
        assistantMessage.content.some(
          (block: any) => block.type === "thinking",
        ),
      ).toBe(false);
      expect(mockStream.mock.calls[0][0].thinking).toBeUndefined();
    });
  });

  // ---------------------------------------------------------------------------
  describe("message translation", () => {
    let provider: any;
    beforeEach(() => {
      provider = createTestProvider();
    });

    it("translates user message to user role with text block", () => {
      const result = provider.chatMessageToAnthropicMessage({
        role: "user",
        content: "Hello",
      } as ChatMessage);
      expect(result.role).toBe("user");
      expect(result.content).toContainEqual({ type: "text", text: "Hello" });
    });

    it("translates batched tool-result message to user role with tool_result blocks", () => {
      const result = provider.chatMessageToAnthropicMessage({
        role: "tool",
        content: "",
        toolResults: [{ toolCallId: "id-1", toolName: "t", content: "ok" }],
      } as ChatMessage);
      expect(result.role).toBe("user");
      expect(result.content).toContainEqual(
        expect.objectContaining({ type: "tool_result", tool_use_id: "id-1" }),
      );
    });

    it("translates legacy single-tool-result message", () => {
      const result = provider.chatMessageToAnthropicMessage({
        role: "tool",
        content: "result",
        toolCallId: "id-2",
      } as ChatMessage);
      expect(result.content).toContainEqual(
        expect.objectContaining({ type: "tool_result", tool_use_id: "id-2" }),
      );
    });

    it("includes tool_use blocks for assistant messages with toolCalls", () => {
      const result = provider.chatMessageToAnthropicMessage({
        role: "assistant",
        content: "Calling",
        toolCalls: [
          { id: "tc-1", function: { name: "fn", arguments: { a: 1 } } },
        ],
      } as ChatMessage);
      expect(result.content).toContainEqual(
        expect.objectContaining({ type: "tool_use", id: "tc-1", name: "fn" }),
      );
    });

    it("throws ProviderError when a tool call has no id", () => {
      expect(() =>
        provider.chatMessageToAnthropicMessage({
          role: "assistant",
          content: "",
          toolCalls: [{ function: { name: "fn", arguments: {} } }],
        } as ChatMessage),
      ).toThrow(ProviderError);
    });

    it("inserts ThinkingBlockParam before tool_use when replaying reasoningContent", () => {
      const blocks = [{ thinking: "reasoning text", signature: "sig-xyz" }];
      const result = provider.chatMessageToAnthropicMessage({
        role: "assistant",
        content: "",
        reasoningContent: JSON.stringify(blocks),
        toolCalls: [{ id: "tc-1", function: { name: "fn", arguments: {} } }],
      } as ChatMessage);
      expect(result.content[0]).toMatchObject({
        type: "thinking",
        thinking: "reasoning text",
        signature: "sig-xyz",
      });
    });

    it("replays redacted_thinking blocks from reasoningContent", () => {
      const result = provider.chatMessageToAnthropicMessage({
        role: "assistant",
        content: "",
        reasoningContent: JSON.stringify([
          { type: "redacted_thinking", data: "redacted-123" },
        ]),
        toolCalls: [{ id: "tc-1", function: { name: "fn", arguments: {} } }],
      } as ChatMessage);
      expect(result.content[0]).toMatchObject({
        type: "redacted_thinking",
        data: "redacted-123",
      });
    });

    it("does NOT insert thinking block when message has no toolCalls", () => {
      const result = provider.chatMessageToAnthropicMessage({
        role: "assistant",
        content: "hi",
        reasoningContent: JSON.stringify([{ thinking: "t", signature: "s" }]),
      } as ChatMessage);
      expect(result.content.some((b: any) => b.type === "thinking")).toBe(
        false,
      );
    });

    it("uses non-whitespace fallback instead of empty text block", () => {
      const result = provider.chatMessageToAnthropicMessage({
        role: "user",
        content: "",
      } as ChatMessage);
      const textBlock = result.content.find((b: any) => b.type === "text");
      expect(textBlock?.text).toBe(".");
    });

    it("concatenates multiple system messages into a single system string", async () => {
      await collectStream(createTestProvider(), {
        messages: [
          { role: "system", content: "Part A" },
          { role: "system", content: "Part B" },
          { role: "user", content: "Hi" },
        ],
        model: "claude-sonnet-4-6",
      });
      const callArgs = mockStream.mock.calls[0][0];
      expect(callArgs.system).toContain("Part A");
      expect(callArgs.system).toContain("Part B");
    });

    it("detects jpeg magic bytes for Uint8Array images", () => {
      const jpegBytes = new Uint8Array([0xff, 0xd8, 0xff, 0xe0]);
      expect(provider.resolveImageData(jpegBytes).mediaType).toBe("image/jpeg");
    });

    it("detects png magic bytes for Uint8Array images", () => {
      const pngBytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47]);
      expect(provider.resolveImageData(pngBytes).mediaType).toBe("image/png");
    });

    it("parses data URL media type and strips the header prefix", () => {
      const result = provider.resolveImageData("data:image/webp;base64,abc123");
      expect(result.mediaType).toBe("image/webp");
      expect(result.data).toBe("abc123");
    });

    it("rejects unsupported data URL media types", () => {
      expect(() =>
        provider.resolveImageData("data:image/svg+xml;base64,abc123"),
      ).toThrow(ProviderError);
    });
  });

  // ---------------------------------------------------------------------------
  describe("tool translation", () => {
    it("translates ChatTool to Anthropic Tool with input_schema", () => {
      const result = createTestProvider().chatToolToAnthropicTool({
        type: "function",
        function: {
          name: "get_weather",
          description: "Gets weather",
          parameters: {
            type: "object",
            properties: { city: { type: "string" } },
            required: ["city"],
          },
        },
      });
      expect(result).toMatchObject({
        name: "get_weather",
        description: "Gets weather",
        input_schema: expect.objectContaining({ type: "object" }),
      });
    });
  });

  // ---------------------------------------------------------------------------
  describe("error translation", () => {
    function throwFromStream(error: Error) {
      mockStream.mockReturnValue(makeErrorStream(error));
      return collectStream(createTestProvider(), createChatRequest());
    }

    it.each([
      [503, "Service unavailable"],
      [529, "Overloaded"],
    ])(
      "retries a failed stream connection on status %i and succeeds on the second attempt",
      async (status, message) => {
        const randomSpy = jest.spyOn(Math, "random").mockReturnValue(0);
        mockStream
          .mockRejectedValueOnce(new MockAPIError(status, message))
          .mockResolvedValueOnce(makeStream(textStreamEvents("Recovered!")));

        try {
          const events = await collectStream(
            createTestProvider(),
            createChatRequest(),
          );

          expect(mockStream).toHaveBeenCalledTimes(2);
          expect(
            events.filter((e) => e.type === "assistant_text"),
          ).toHaveLength(1);
          expect(
            (events.find((e) => e.type === "assistant_text") as any).delta,
          ).toBe("Recovered!");
        } finally {
          randomSpy.mockRestore();
        }
      },
    );

    it("maps status 401 to ProviderAuthenticationError", async () => {
      await expect(
        throwFromStream(new MockAPIError(401, "Unauthorized")),
      ).rejects.toBeInstanceOf(ProviderAuthenticationError);
    });

    it("maps authentication message to ProviderAuthenticationError", async () => {
      await expect(
        throwFromStream(new MockAPIError(400, "authentication failed")),
      ).rejects.toBeInstanceOf(ProviderAuthenticationError);
    });

    it("maps status 429 to ProviderRateLimitError", async () => {
      await expect(
        throwFromStream(new MockAPIError(429, "Rate limited")),
      ).rejects.toBeInstanceOf(ProviderRateLimitError);
    });

    it("includes retry-after seconds on ProviderRateLimitError", async () => {
      let caught: any;
      try {
        await throwFromStream(
          new MockAPIError(429, "Rate limited", { "retry-after": "30" }),
        );
      } catch (e) {
        caught = e;
      }
      expect(caught).toBeInstanceOf(ProviderRateLimitError);
      expect(caught.retryAfterSeconds).toBe(30);
    });

    it("maps status 529 to ProviderCapacityError", async () => {
      await expect(
        throwFromStream(new MockAPIError(529, "Overloaded")),
      ).rejects.toBeInstanceOf(ProviderCapacityError);
    });

    it("maps status 404 to ProviderModelNotFoundError", async () => {
      await expect(
        throwFromStream(new MockAPIError(404, "Model not found")),
      ).rejects.toBeInstanceOf(ProviderModelNotFoundError);
    });

    it("maps status 400 + 'context length' to ProviderContextLengthError", async () => {
      await expect(
        throwFromStream(new MockAPIError(400, "context length exceeded")),
      ).rejects.toBeInstanceOf(ProviderContextLengthError);
    });

    it("maps non-specific APIError to ProviderError", async () => {
      await expect(
        throwFromStream(new MockAPIError(503, "Service unavailable")),
      ).rejects.toBeInstanceOf(ProviderError);
    });

    it("wraps plain Error as ProviderError", async () => {
      mockStream.mockReturnValue(makeErrorStream(new Error("network failure")));
      await expect(
        collectStream(createTestProvider(), createChatRequest()),
      ).rejects.toBeInstanceOf(ProviderError);
    });
  });
});
