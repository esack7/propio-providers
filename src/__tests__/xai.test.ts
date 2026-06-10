import { XaiProvider } from "../providers/xai.js";
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
      })) {
        deltas.push(chunk.delta);
      }

      expect(deltas).toEqual(["Hello"]);
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
    });
  });
});
