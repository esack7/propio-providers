import {
  CloudflareProvider,
  normalizeCloudflareModelId,
} from "../providers/cloudflare.js";
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
const DEFAULT_MODEL = "cf/moonshotai/kimi-k2.6";
const DEFAULT_CONTEXT_WINDOW = 262_144;
const DEFAULT_ACCOUNT_ID = "test-account-id";
const DEFAULT_REQUEST: ChatRequest = {
  model: DEFAULT_MODEL,
  messages: [{ role: "user", content: "Hi" }],
};

const CLOUDFLARE_ENDPOINT = `https://api.cloudflare.com/client/v4/accounts/${DEFAULT_ACCOUNT_ID}/ai/v1/chat/completions`;

const createSseStream = OpenRouterTestFixture.createSseStream;

function createProvider(
  options: Partial<ConstructorParameters<typeof CloudflareProvider>[0]> = {},
): CloudflareProvider {
  return new CloudflareProvider({
    model: DEFAULT_MODEL,
    contextWindowTokens: DEFAULT_CONTEXT_WINDOW,
    apiKey: "cf-test-token",
    accountId: DEFAULT_ACCOUNT_ID,
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

function expectCloudflareConstructorAuthError(
  setupEnv: () => void,
  messagePattern: RegExp,
): void {
  setupEnv();
  const createMissingCredentialProvider = () =>
    new CloudflareProvider({
      model: DEFAULT_MODEL,
      contextWindowTokens: DEFAULT_CONTEXT_WINDOW,
    });
  expect(createMissingCredentialProvider).toThrow(ProviderAuthenticationError);
  expect(createMissingCredentialProvider).toThrow(messagePattern);
}

describe("normalizeCloudflareModelId", () => {
  it("should prefix cf/ model IDs with @", () => {
    expect(normalizeCloudflareModelId("cf/moonshotai/kimi-k2.6")).toBe(
      "@cf/moonshotai/kimi-k2.6",
    );
  });

  it("should leave already normalized model IDs unchanged", () => {
    expect(normalizeCloudflareModelId("@cf/moonshotai/kimi-k2.6")).toBe(
      "@cf/moonshotai/kimi-k2.6",
    );
  });

  it("should leave other model IDs unchanged", () => {
    expect(normalizeCloudflareModelId("some-other-model")).toBe(
      "some-other-model",
    );
  });
});

describe("CloudflareProvider", () => {
  registerProviderTestLifecycle(originalEnv, originalFetch);

  describe("constructor", () => {
    registerAcceptsApiKeyTest({
      expectedName: "cloudflare",
      createProvider: () =>
        new CloudflareProvider({
          model: DEFAULT_MODEL,
          contextWindowTokens: DEFAULT_CONTEXT_WINDOW,
          apiKey: "cf-test-token",
          accountId: DEFAULT_ACCOUNT_ID,
        }),
    });

    it("should use CLOUDFLARE_API_TOKEN env var when apiKey not in options", () => {
      process.env.CLOUDFLARE_API_TOKEN = "cf-env-token";
      process.env.CLOUDFLARE_ACCOUNT_ID = DEFAULT_ACCOUNT_ID;
      const provider = new CloudflareProvider({
        model: DEFAULT_MODEL,
        contextWindowTokens: DEFAULT_CONTEXT_WINDOW,
      });
      expect(provider.name).toBe("cloudflare");
    });

    it("should fall back to CLOUDFLARE_AUTH_TOKEN and CLOUDFLARE_API_KEY", () => {
      delete process.env.CLOUDFLARE_API_TOKEN;
      process.env.CLOUDFLARE_AUTH_TOKEN = "auth-token";
      process.env.CLOUDFLARE_ACCOUNT_ID = DEFAULT_ACCOUNT_ID;
      expect(
        () =>
          new CloudflareProvider({
            model: DEFAULT_MODEL,
            contextWindowTokens: DEFAULT_CONTEXT_WINDOW,
          }),
      ).not.toThrow();

      delete process.env.CLOUDFLARE_AUTH_TOKEN;
      process.env.CLOUDFLARE_API_KEY = "api-key";
      expect(
        () =>
          new CloudflareProvider({
            model: DEFAULT_MODEL,
            contextWindowTokens: DEFAULT_CONTEXT_WINDOW,
          }),
      ).not.toThrow();
    });

    it("should use CLOUDFLARE_ACCOUNT_ID env var when accountId not in options", () => {
      process.env.CLOUDFLARE_API_TOKEN = "cf-env-token";
      process.env.CLOUDFLARE_ACCOUNT_ID = "env-account-id";
      const provider = new CloudflareProvider({
        model: DEFAULT_MODEL,
        contextWindowTokens: DEFAULT_CONTEXT_WINDOW,
      });
      expect(provider.name).toBe("cloudflare");
    });

    it("should throw ProviderAuthenticationError when no API token is provided", () => {
      expectCloudflareConstructorAuthError(() => {
        delete process.env.CLOUDFLARE_API_TOKEN;
        delete process.env.CLOUDFLARE_AUTH_TOKEN;
        delete process.env.CLOUDFLARE_API_KEY;
        process.env.CLOUDFLARE_ACCOUNT_ID = DEFAULT_ACCOUNT_ID;
      }, /API token|Cloudflare/);
    });

    it("should throw ProviderAuthenticationError when no account ID is provided", () => {
      expectCloudflareConstructorAuthError(() => {
        process.env.CLOUDFLARE_API_TOKEN = "cf-env-token";
        delete process.env.CLOUDFLARE_ACCOUNT_ID;
      }, /account ID|Cloudflare/);
    });

    it("should report the configured context window for Kimi K2.6", () => {
      const provider = createProvider();
      expect(provider.getCapabilities().contextWindowTokens).toBe(262_144);
    });
  });

  describe("streamChat()", () => {
    it("should yield content deltas from mocked SSE stream", async () => {
      const chunks = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
        'data: {"choices":[{"delta":{"content":" Cloudflare"}}]}\n\n',
        "data: [DONE]\n\n",
      ];
      globalThis.fetch = jest.fn().mockResolvedValue({
        ok: true,
        body: createSseStream(chunks),
      });

      const provider = createProvider();
      const deltas: string[] = [];
      for await (const chunk of provider.streamChat(createRequest())) {
        if (chunk.type === "assistant_text") {
          deltas.push(chunk.delta);
        }
      }
      expect(deltas).toEqual(["Hello", " Cloudflare"]);
    });

    it("should call the Cloudflare account-scoped endpoint with correct auth header", async () => {
      globalThis.fetch = jest.fn().mockResolvedValue({
        ok: true,
        body: createSseStream([
          'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n',
          "data: [DONE]\n\n",
        ]),
      });

      for await (const _chunk of createProvider().streamChat(createRequest())) {
        // consume
      }

      expect(fetch).toHaveBeenCalledWith(
        CLOUDFLARE_ENDPOINT,
        expect.objectContaining({
          method: "POST",
          headers: expect.objectContaining({
            Authorization: "Bearer cf-test-token",
            Accept: "text/event-stream, application/json",
            "Content-Type": "application/json",
          }),
        }),
      );
    });

    it("should send cf/moonshotai/kimi-k2.6 as @cf/moonshotai/kimi-k2.6", async () => {
      const mockFetch = jest.fn().mockResolvedValue({
        ok: true,
        body: createSseStream([
          'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n',
        ]),
      });
      globalThis.fetch = mockFetch;

      for await (const _chunk of createProvider().streamChat(createRequest())) {
        // consume
      }

      const requestBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(requestBody.model).toBe("@cf/moonshotai/kimi-k2.6");
    });

    it("should emit streamed tool calls when finish_reason is tool_calls", async () => {
      globalThis.fetch = jest.fn().mockResolvedValue({
        ok: true,
        body: createSseStream([
          'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read_file","arguments":"{\\"path\\":\\".\\"}"}}]},"finish_reason":"tool_calls"}]}\n\n',
          "data: [DONE]\n\n",
        ]),
      });

      const provider = createProvider();
      let toolCalls: unknown[] | undefined;
      for await (const chunk of provider.streamChat(createRequest())) {
        if (chunk.type === "tool_calls") {
          toolCalls = chunk.toolCalls;
        }
      }

      expect(toolCalls).toHaveLength(1);
      expect(toolCalls?.[0]).toMatchObject({
        id: "call_1",
        function: {
          name: "read_file",
          arguments: { path: "." },
        },
      });
    });

    registerOpenAiCompatibleToolResultExpansionTest({ collectToolMessages });

    it("should serialize image attachments into OpenAI-compatible content parts", async () => {
      const mockFetch = jest.fn().mockResolvedValue({
        ok: true,
        body: createSseStream([
          'data: {"choices":[{"delta":{"content":"Seen"}}]}\n\n',
        ]),
      });
      globalThis.fetch = mockFetch;

      for await (const _chunk of createProvider().streamChat(
        createRequest({
          messages: [
            {
              role: "user",
              content: "Describe this image",
              images: ["data:image/png;base64,iVBORw0KGgo="],
            },
          ],
        }),
      )) {
        // consume
      }

      const requestBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      const userContent = requestBody.messages[0].content as Array<{
        type: string;
        text?: string;
        image_url?: { url: string };
      }>;
      expect(userContent).toHaveLength(2);
      expect(userContent[0].type).toBe("text");
      expect(userContent[1].type).toBe("image_url");
      expect(userContent[1].image_url?.url).toContain("data:image/png;base64");
    });

    it("should surface Cloudflare errors[].message on API failures", async () => {
      globalThis.fetch = jest.fn().mockResolvedValue({
        ok: false,
        status: 503,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              success: false,
              errors: [
                {
                  code: 7000,
                  message: "Workers AI capacity temporarily unavailable",
                },
              ],
            }),
          ),
      });

      const provider = createProvider({
        retryConfig: { maxRetries: 0, consecutive529Limit: 1 },
      });
      await expectStreamChatToThrow(provider, ProviderError);
      await expectStreamChatToThrow(
        provider,
        /Workers AI capacity temporarily unavailable/,
      );
    });

    registerOpenAiCompatibleStreamErrorTests({
      createProvider,
      expectRequestError,
      expectProviderErrorAndMessage,
      expectStreamChatToThrow,
      defaultRequest: DEFAULT_REQUEST,
      contextLengthErrorMessage:
        "This model's maximum context length is 262144 tokens. However, your messages resulted in 300000 tokens.",
    });
  });
});
