import { GeminiProvider } from "../providers/gemini.js";
import {
  ProviderAuthenticationError,
  ProviderContextLengthError,
  ProviderError,
  ProviderModelNotFoundError,
  ProviderRateLimitError,
} from "../types.js";
import { ChatMessage, ChatRequest } from "../types.js";

const originalEnv = process.env;
const originalFetch = globalThis.fetch;

import {
  OpenRouterTestFixture,
  registerProviderTestLifecycle,
} from "./openrouterTestHelpers.js";

const createSseStream = OpenRouterTestFixture.createSseStream;

function createGeminiProvider(
  options: Partial<ConstructorParameters<typeof GeminiProvider>[0]> = {},
): GeminiProvider {
  return new GeminiProvider({
    model: "gemini-3.1-pro-preview",
    contextWindowTokens: 1_048_576,
    apiKey: "gemini-test-key",
    ...options,
  });
}

function setupGeminiSuccessMock(captureBody?: (body: unknown) => void): void {
  globalThis.fetch = OpenRouterTestFixture.setupFetchMock(
    ['data: {"choices":[{"delta":{"content":"ok"}}]}\n\n', "data: [DONE]\n\n"],
    captureBody,
  );
}

async function consumeGeminiStream(
  provider: GeminiProvider,
  request: ChatRequest,
): Promise<void> {
  for await (const _chunk of provider.streamChat(request)) {
    // consume
  }
}

async function collectGeminiTextStreams(
  provider: GeminiProvider,
  request: ChatRequest,
): Promise<{ thinkingEvents: string[]; assistantText: string[] }> {
  const thinkingEvents: string[] = [];
  const assistantText: string[] = [];

  for await (const chunk of provider.streamChat(request)) {
    if (chunk.type === "thinking_delta") {
      thinkingEvents.push(chunk.delta);
    }
    if (chunk.type === "assistant_text") {
      assistantText.push(chunk.delta);
    }
  }

  return { thinkingEvents, assistantText };
}

async function captureGeminiRequestBody(
  request: ChatRequest,
  sseLines: string[] = [
    'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n',
    "data: [DONE]\n\n",
  ],
): Promise<unknown> {
  let capturedBody: unknown = null;
  globalThis.fetch = OpenRouterTestFixture.setupFetchMock(sseLines, (body) => {
    capturedBody = body;
  });

  const provider = createGeminiProvider();
  await consumeGeminiStream(provider, request);
  return capturedBody;
}

async function expectGeminiThinkingStream(
  sseLines: string[],
  expected: { thinkingEvents: string[]; assistantText: string[] },
): Promise<void> {
  globalThis.fetch = jest.fn().mockResolvedValue({
    ok: true,
    body: createSseStream(sseLines),
  });

  const provider = createGeminiProvider();
  const result = await collectGeminiTextStreams(provider, {
    model: "gemini-3.1-pro-preview",
    messages: [{ role: "user", content: "Hi" }],
    requestReasoning: true,
  });

  expect(result.thinkingEvents).toEqual(expected.thinkingEvents);
  expect(result.assistantText).toEqual(expected.assistantText);
}

describe("GeminiProvider", () => {
  registerProviderTestLifecycle(originalEnv, originalFetch);

  async function expectGeminiStreamChatToThrow(
    fetchSetup: () => void,
    ErrorClass: new (...args: unknown[]) => Error,
    model = "gemini-3.1-pro-preview",
  ): Promise<void> {
    fetchSetup();
    const provider = createGeminiProvider({ model });
    await expect(async () => {
      for await (const _chunk of provider.streamChat({
        model,
        messages: [{ role: "user", content: "Hi" }],
      })) {
        // consume
      }
    }).rejects.toThrow(ErrorClass);
  }

  describe("constructor", () => {
    it("should accept API key from options", () => {
      const provider = createGeminiProvider();
      expect(provider.name).toBe("gemini");
    });

    it("should use GEMINI_API_KEY env var when apiKey not in options", () => {
      process.env.GEMINI_API_KEY = "gemini-env-key";
      const provider = createGeminiProvider({ apiKey: undefined });
      expect(provider.name).toBe("gemini");
    });

    it("should fall back to GOOGLE_API_KEY when GEMINI_API_KEY is missing", () => {
      delete process.env.GEMINI_API_KEY;
      process.env.GOOGLE_API_KEY = "google-env-key";
      const provider = createGeminiProvider({ apiKey: undefined });
      expect(provider.name).toBe("gemini");
    });

    it("should throw ProviderAuthenticationError when no API key is provided", () => {
      delete process.env.GEMINI_API_KEY;
      delete process.env.GOOGLE_API_KEY;
      expect(() => {
        createGeminiProvider({ apiKey: undefined });
      }).toThrow(ProviderAuthenticationError);
      expect(() => {
        createGeminiProvider({ apiKey: undefined });
      }).toThrow(/API key|Gemini/);
    });

    it("should report the requested 1,048,576 token context window for gemini preview models", () => {
      const models = [
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview",
      ];

      for (const model of models) {
        const provider = createGeminiProvider({ model });
        expect(provider.getCapabilities().contextWindowTokens).toBe(1_048_576);
      }
    });

    it("should accept configured Gemini models without a local allow-list", () => {
      const provider = createGeminiProvider({
        model: "gemini-future-preview",
        contextWindowTokens: 2_000_000,
      });

      expect(provider.getCapabilities().contextWindowTokens).toBe(2_000_000);
    });
  });

  describe("streamChat()", () => {
    it("should translate messages, images, and batched tool results into the Gemini request body", async () => {
      let capturedBody: unknown = null;
      setupGeminiSuccessMock((body) => {
        capturedBody = body;
      });

      const provider = createGeminiProvider();

      const request: ChatRequest = {
        model: "gemini-3.1-pro-preview",
        messages: [
          { role: "system", content: "You are helpful" },
          {
            role: "user",
            content: "Look",
            images: ["data:image/png;base64,iVBORw0KGgo="],
          },
          {
            role: "assistant",
            content: "",
            toolCalls: [
              {
                id: "call_1",
                thoughtSignature: "sig-1",
                function: {
                  name: "get_weather",
                  arguments: { location: "NYC" },
                },
              },
            ],
          },
          {
            role: "tool",
            content: "",
            toolResults: [
              {
                toolCallId: "call_1",
                toolName: "get_weather",
                content: "sunny",
              },
              {
                toolCallId: "call_2",
                toolName: "get_time",
                content: "noon",
              },
            ],
          },
        ],
        tools: [
          {
            type: "function",
            function: {
              name: "get_weather",
              description: "Get weather",
              parameters: { type: "object", properties: {} },
            },
          },
        ],
      };

      await consumeGeminiStream(provider, request);

      expect(capturedBody).not.toBeNull();
      const body = capturedBody as {
        model: string;
        stream: boolean;
        messages: Array<{
          role: string;
          content: string | Array<unknown>;
          tool_call_id?: string;
        }>;
        tools?: unknown[];
      };
      expect(body.model).toBe("gemini-3.1-pro-preview");
      expect(body.stream).toBe(true);
      expect(body.tools).toHaveLength(1);
      expect(body.messages).toHaveLength(5);
      expect(body.messages[1].role).toBe("user");
      expect(Array.isArray(body.messages[1].content)).toBe(true);
      const userContent = body.messages[1].content as Array<{
        type: string;
        text?: string;
        image_url?: { url: string };
      }>;
      expect(userContent[0]).toEqual({ type: "text", text: "Look" });
      expect(userContent[1].type).toBe("image_url");
      expect(userContent[1].image_url?.url).toContain("data:image/png;base64");
      expect(body.messages[2]).toMatchObject({
        role: "assistant",
        tool_calls: [
          {
            id: "call_1",
            extra_content: {
              google: {
                thought_signature: "sig-1",
              },
            },
          },
        ],
      });
      expect(body.messages[3]).toMatchObject({
        role: "tool",
        content: "sunny",
        tool_call_id: "call_1",
      });
      expect(body.messages[4]).toMatchObject({
        role: "tool",
        content: "noon",
        tool_call_id: "call_2",
      });
    });

    it("should stream assistant text and tool calls from SSE chunks", async () => {
      const toolCallStart = JSON.stringify({
        choices: [
          {
            delta: {
              tool_calls: [
                {
                  index: 0,
                  id: "call_1",
                  extra_content: {
                    google: {
                      thought_signature: "sig-1",
                    },
                  },
                  function: { name: "get_weather" },
                },
              ],
            },
          },
        ],
      });
      const toolCallArgs1 = JSON.stringify({
        choices: [
          {
            delta: {
              tool_calls: [
                {
                  index: 0,
                  function: { arguments: '{"location":"NY' },
                },
              ],
            },
          },
        ],
      });
      const toolCallArgs2 = JSON.stringify({
        choices: [
          {
            delta: {
              tool_calls: [
                {
                  index: 0,
                  function: { arguments: 'C"}' },
                },
              ],
            },
          },
        ],
      });

      globalThis.fetch = jest.fn().mockResolvedValue({
        ok: true,
        body: createSseStream([
          'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
          'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
          `data: ${toolCallStart}\n\n`,
          `data: ${toolCallArgs1}\n\n`,
          `data: ${toolCallArgs2}\n\n`,
          "data: [DONE]\n\n",
        ]),
      });

      const provider = createGeminiProvider();

      const request: ChatRequest = {
        model: "gemini-3.1-pro-preview",
        messages: [{ role: "user", content: "Hi" }],
      };

      const deltas: string[] = [];
      let toolCalls: unknown[] | undefined;
      for await (const chunk of provider.streamChat(request)) {
        if ("type" in chunk && chunk.type === "assistant_text") {
          deltas.push(chunk.delta);
        }
        if ("type" in chunk && chunk.type === "tool_calls") {
          toolCalls = chunk.toolCalls;
        }
      }

      expect(deltas).toEqual(["Hello", " world"]);
      expect(toolCalls).toHaveLength(1);
      const parsedToolCalls = toolCalls as Array<{
        thoughtSignature?: string;
        function: { name: string; arguments: Record<string, string> };
      }>;
      expect(parsedToolCalls[0].function.name).toBe("get_weather");
      expect(parsedToolCalls[0].function.arguments.location).toBe("NYC");
      expect(parsedToolCalls[0].thoughtSignature).toBe("sig-1");
    });

    it("should request Gemini thought summaries when live thinking is requested", async () => {
      const capturedBody = (await captureGeminiRequestBody({
        model: "gemini-3.1-pro-preview",
        messages: [{ role: "user", content: "Hi" }],
        requestReasoning: true,
      })) as { extra_body: unknown };

      expect(capturedBody.extra_body).toEqual({
        google: {
          thinking_config: {
            include_thoughts: true,
          },
        },
      });
    });

    it("should emit Gemini thought summaries as live thinking deltas", async () => {
      await expectGeminiThinkingStream(
        [
          'data: {"choices":[{"delta":{"reasoning_content":"<thought>I should inspect "}}]}\n\n',
          'data: {"choices":[{"delta":{"extra_content":{"google":{"thought_summary":"the repo.</thought>"}}}}]}\n\n',
          'data: {"choices":[{"delta":{"content":"Final answer."}}]}\n\n',
          "data: [DONE]\n\n",
        ],
        {
          thinkingEvents: ["I should inspect ", "the repo."],
          assistantText: ["Final answer."],
        },
      );
    });

    it("should extract thought blocks from string content into thinking deltas", async () => {
      await expectGeminiThinkingStream(
        [
          'data: {"choices":[{"delta":{"content":"<thought>Analyzing UI\\n</thought>"}}]}\n\n',
          'data: {"choices":[{"delta":{"content":"\\n\\nFinal answer."}}]}\n\n',
          "data: [DONE]\n\n",
        ],
        {
          thinkingEvents: ["Analyzing UI\n"],
          assistantText: ["\n\nFinal answer."],
        },
      );
    });

    it("should stream thought blocks split across string content chunks", async () => {
      await expectGeminiThinkingStream(
        [
          'data: {"choices":[{"delta":{"content":"<thought>Analyzing "}}]}\n\n',
          'data: {"choices":[{"delta":{"content":"UI</thought>"}}]}\n\n',
          "data: [DONE]\n\n",
        ],
        { thinkingEvents: ["Analyzing UI"], assistantText: [] },
      );
    });

    it("should split Gemini content parts into thought and assistant streams", async () => {
      await expectGeminiThinkingStream(
        [
          'data: {"choices":[{"delta":{"content":[{"text":"Thinking part.","thought":true},{"text":"Answer part."}]}}]}\n\n',
          "data: [DONE]\n\n",
        ],
        {
          thinkingEvents: ["Thinking part."],
          assistantText: ["Answer part."],
        },
      );
    });

    it("should emit tool calls when Gemini uses camelCase toolCalls and stop finish reasons", async () => {
      const toolCallChunk = JSON.stringify({
        choices: [
          {
            delta: {
              toolCalls: [
                {
                  index: 0,
                  id: "call_2",
                  extra_content: {
                    google: {
                      thought_signature: "sig-2",
                    },
                  },
                  function: {
                    name: "list_files",
                    arguments: '{"path":"."}',
                  },
                },
              ],
            },
            finish_reason: "stop",
          },
        ],
      });

      globalThis.fetch = jest.fn().mockResolvedValue({
        ok: true,
        body: createSseStream([
          `data: ${toolCallChunk}\n\n`,
          "data: [DONE]\n\n",
        ]),
      });

      const provider = createGeminiProvider();

      const request: ChatRequest = {
        model: "gemini-3.1-pro-preview",
        messages: [{ role: "user", content: "What is in this repo?" }],
      };

      let toolCalls: unknown[] | undefined;
      for await (const chunk of provider.streamChat(request)) {
        if ("type" in chunk && chunk.type === "tool_calls") {
          toolCalls = chunk.toolCalls;
        }
      }

      expect(toolCalls).toHaveLength(1);
      expect(toolCalls?.[0]).toMatchObject({
        id: "call_2",
        thoughtSignature: "sig-2",
        function: {
          name: "list_files",
          arguments: { path: "." },
        },
      });
    });

    it("should not duplicate repeated Gemini tool call names", async () => {
      globalThis.fetch = jest.fn().mockResolvedValue({
        ok: true,
        body: createSseStream([
          'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read"}}]}}]}\n\n',
          'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"read","arguments":"{\\"path\\":\\"package.json\\"}"}}]}}]}\n\n',
          "data: [DONE]\n\n",
        ]),
      });

      const provider = createGeminiProvider();

      let toolCalls: unknown[] | undefined;
      for await (const chunk of provider.streamChat({
        model: "gemini-3.1-pro-preview",
        messages: [{ role: "user", content: "What is in this repo?" }],
      })) {
        if ("type" in chunk && chunk.type === "tool_calls") {
          toolCalls = chunk.toolCalls;
        }
      }

      expect(toolCalls?.[0]).toMatchObject({
        id: "call_1",
        function: {
          name: "read",
          arguments: { path: "package.json" },
        },
      });
    });

    it("should throw ProviderAuthenticationError on 401", async () => {
      await expectGeminiStreamChatToThrow(() => {
        globalThis.fetch = jest.fn().mockResolvedValue({
          ok: false,
          status: 401,
          headers: new Headers(),
        });
      }, ProviderAuthenticationError);
    });

    it("should throw ProviderRateLimitError on 429", async () => {
      await expectGeminiStreamChatToThrow(() => {
        globalThis.fetch = jest.fn().mockResolvedValue({
          ok: false,
          status: 429,
          headers: new Headers([["retry-after", "30"]]),
        });
      }, ProviderRateLimitError);
    });

    it("should throw ProviderModelNotFoundError on 404", async () => {
      await expectGeminiStreamChatToThrow(() => {
        globalThis.fetch = jest.fn().mockResolvedValue({
          ok: false,
          status: 404,
          headers: new Headers(),
        });
      }, ProviderModelNotFoundError);
    });

    it("should throw ProviderContextLengthError on context-length failures", async () => {
      await expectGeminiStreamChatToThrow(() => {
        globalThis.fetch = jest.fn().mockResolvedValue({
          ok: false,
          status: 400,
          headers: new Headers(),
          text: async () =>
            JSON.stringify({
              error: { message: "prompt is too long for the model" },
            }),
        });
      }, ProviderContextLengthError);
    });

    it("should translate network failures into ProviderError", async () => {
      await expectGeminiStreamChatToThrow(() => {
        globalThis.fetch = jest
          .fn()
          .mockRejectedValue(new Error("fetch failed"));
      }, ProviderError);
    });

    it("should pass request model overrides through to the upstream API", async () => {
      const capturedBody = (await captureGeminiRequestBody({
        model: "gemini-future-preview",
        messages: [{ role: "user", content: "Hi" }],
      })) as { model: string };

      expect(capturedBody.model).toBe("gemini-future-preview");
    });

    it("should preserve signed Gemini tool calls with thought signatures", async () => {
      const capturedBody = (await captureGeminiRequestBody({
        model: "gemini-3.1-pro-preview",
        messages: [
          {
            role: "assistant",
            content: "",
            toolCalls: [
              {
                id: "call_1",
                thoughtSignature: "sig-weather",
                function: {
                  name: "get_weather",
                  arguments: { location: "NYC" },
                },
              },
            ],
          },
          {
            role: "tool",
            content: "",
            toolResults: [
              {
                toolCallId: "call_1",
                toolName: "get_weather",
                content: "sunny",
              },
            ],
          },
        ],
      })) as {
        messages: Array<{
          role: string;
          tool_calls?: Array<{
            id: string;
            extra_content?: { google?: { thought_signature?: string } };
          }>;
        }>;
      };

      const assistant = capturedBody.messages.find(
        (message) => message.role === "assistant" && message.tool_calls?.length,
      );
      expect(assistant?.tool_calls?.[0]).toMatchObject({
        id: "call_1",
        extra_content: {
          google: {
            thought_signature: "sig-weather",
          },
        },
      });
      expect(
        capturedBody.messages.some(
          (message) =>
            message.role === "tool" &&
            (message as { tool_call_id?: string }).tool_call_id === "call_1",
        ),
      ).toBe(true);
    });
  });
});
