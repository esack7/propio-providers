import type { ChatMessage, ChatRequest, ChatStreamEvent } from "../types.js";
import {
  ProviderAuthenticationError,
  ProviderContextLengthError,
  ProviderError,
  ProviderModelNotFoundError,
  ProviderRateLimitError,
} from "../types.js";
import {
  OpenRouterTestFixture,
  registerAcceptsApiKeyTest,
  registerProviderTestLifecycle,
} from "./openrouterTestHelpers.js";

export {
  ProviderAuthenticationError,
  ProviderError,
  registerAcceptsApiKeyTest,
  registerProviderTestLifecycle,
};
export type { ChatRequest };
export { OpenRouterTestFixture };

export const OPENAI_COMPATIBLE_PROVIDER_TEST_ENV = {
  originalEnv: process.env,
  originalFetch: globalThis.fetch,
};

export type OpenAiCompatibleStreamingProvider = {
  streamChat(request: ChatRequest): AsyncIterable<ChatStreamEvent>;
};

export function setupOpenAiCompatibleProviderTests<
  TProvider extends OpenAiCompatibleStreamingProvider,
  TOptions = Record<string, unknown>,
>(config: {
  createProvider: (options?: Partial<TOptions>) => TProvider;
  defaultRequest: ChatRequest;
}) {
  function createRequest(overrides: Partial<ChatRequest> = {}): ChatRequest {
    return {
      ...config.defaultRequest,
      ...overrides,
      messages: overrides.messages ?? config.defaultRequest.messages,
    };
  }

  return {
    createRequest,
    ...createOpenAiCompatibleProviderTestHelpers({
      createProvider: config.createProvider,
      createRequest,
      defaultRequest: config.defaultRequest,
    }),
  };
}

export function registerOpenAiCompatibleToolResultExpansionTest(options: {
  collectToolMessages: (messages: ChatMessage[]) => Promise<unknown[]>;
}): void {
  it("should expand batched tool results into individual messages", async () => {
    const toolMessages = await options.collectToolMessages([
      { role: "user", content: "Test" },
      {
        role: "assistant",
        content: "",
        toolCalls: [
          { id: "call1", function: { name: "tool1", arguments: {} } },
        ],
      },
      {
        role: "tool",
        content: "",
        toolResults: [
          { toolCallId: "call1", toolName: "tool1", content: "result1" },
        ],
      },
    ]);

    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0]).toMatchObject({
      role: "tool",
      content: "result1",
      tool_call_id: "call1",
    });
  });
}

export function createOpenAiCompatibleProviderTestHelpers<
  TProvider extends OpenAiCompatibleStreamingProvider,
  TOptions = Record<string, unknown>,
>(config: {
  createProvider: (options?: Partial<TOptions>) => TProvider;
  createRequest: (overrides?: Partial<ChatRequest>) => ChatRequest;
  defaultRequest: ChatRequest;
}) {
  const createSseStream = OpenRouterTestFixture.createSseStream;

  async function expectStreamChatToThrow(
    provider: TProvider,
    matcher: string | RegExp | (new (...args: unknown[]) => unknown),
    request: ChatRequest = config.defaultRequest,
  ): Promise<void> {
    await expect(async () => {
      for await (const _chunk of provider.streamChat(request)) {
        // consume
      }
    }).rejects.toThrow(matcher as any);
  }

  async function expectRequestError(
    response: Record<string, unknown>,
    matcher: string | RegExp | (new (...args: unknown[]) => unknown),
    providerOptions: Partial<TOptions> = {},
  ): Promise<void> {
    globalThis.fetch = jest.fn().mockResolvedValue(response);
    await expectStreamChatToThrow(
      config.createProvider(providerOptions),
      matcher,
    );
  }

  async function expectProviderErrorAndMessage(
    response: Record<string, unknown>,
    messageMatcher: string | RegExp,
    providerOptions: Partial<TOptions> = {},
  ): Promise<void> {
    globalThis.fetch = jest.fn().mockResolvedValue(response);
    const provider = config.createProvider(providerOptions);
    await expectStreamChatToThrow(provider, ProviderError);
    await expectStreamChatToThrow(provider, messageMatcher);
  }

  async function collectToolMessages(
    messages: ChatMessage[],
  ): Promise<unknown[]> {
    const mockFetch = jest.fn().mockResolvedValue({
      ok: true,
      body: createSseStream([
        'data: {"choices":[{"delta":{"content":"Done"}}]}\n\n',
      ]),
    });
    globalThis.fetch = mockFetch;

    for await (const _chunk of config
      .createProvider()
      .streamChat(config.createRequest({ messages }))) {
      // consume
    }

    const requestBody = JSON.parse(mockFetch.mock.calls[0][1].body);
    return requestBody.messages.filter(
      (message: { role?: string }) => message.role === "tool",
    );
  }

  return {
    expectStreamChatToThrow,
    expectRequestError,
    expectProviderErrorAndMessage,
    collectToolMessages,
  };
}

export function registerOpenAiCompatibleStreamErrorTests<
  TOptions = Record<string, unknown>,
>(options: {
  createProvider: (
    providerOptions?: Partial<TOptions>,
  ) => OpenAiCompatibleStreamingProvider;
  expectRequestError: (
    response: Record<string, unknown>,
    matcher: string | RegExp | (new (...args: unknown[]) => unknown),
    providerOptions?: Partial<TOptions>,
  ) => Promise<void>;
  expectProviderErrorAndMessage: (
    response: Record<string, unknown>,
    messageMatcher: string | RegExp,
    providerOptions?: Partial<TOptions>,
  ) => Promise<void>;
  expectStreamChatToThrow: (
    provider: OpenAiCompatibleStreamingProvider,
    matcher: string | RegExp | (new (...args: unknown[]) => unknown),
    request?: ChatRequest,
  ) => Promise<void>;
  defaultRequest: ChatRequest;
  contextLengthErrorMessage: string;
}): void {
  it("should throw ProviderAuthenticationError on 401", async () => {
    await options.expectRequestError(
      { ok: false, status: 401 },
      ProviderAuthenticationError,
    );
  });

  it("should throw ProviderRateLimitError on 429", async () => {
    await options.expectRequestError(
      {
        ok: false,
        status: 429,
        headers: new Map([["retry-after", "30"]]),
      },
      ProviderRateLimitError,
    );
  });

  it("should throw ProviderModelNotFoundError on 404", async () => {
    await options.expectRequestError(
      { ok: false, status: 404 },
      ProviderModelNotFoundError,
    );
  });

  it("should throw ProviderError on 5xx", async () => {
    await options.expectProviderErrorAndMessage(
      {
        ok: false,
        status: 503,
        text: () => Promise.resolve("upstream connect error"),
      },
      /upstream connect error/,
      {
        retryConfig: { maxRetries: 0, consecutive529Limit: 1 },
      } as Partial<TOptions>,
    );
  });

  it("should throw ProviderError on network failure", async () => {
    globalThis.fetch = jest.fn().mockRejectedValue(new Error("fetch failed"));
    await options.expectStreamChatToThrow(
      options.createProvider(),
      ProviderError,
    );
  });

  it("should throw ProviderContextLengthError on 400 with context length message in body", async () => {
    await options.expectRequestError(
      {
        ok: false,
        status: 400,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              error: { message: options.contextLengthErrorMessage },
            }),
          ),
      },
      ProviderContextLengthError,
    );
  });

  it("should throw generic ProviderError on 400 without context length message", async () => {
    globalThis.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 400,
      text: () =>
        Promise.resolve(
          JSON.stringify({ error: { message: "Invalid request format" } }),
        ),
    });

    const provider = options.createProvider({
      retryConfig: { maxRetries: 0, consecutive529Limit: 1 },
    } as Partial<TOptions>);
    await options.expectStreamChatToThrow(provider, ProviderError);
    await expect(async () => {
      for await (const _chunk of provider.streamChat(options.defaultRequest)) {
        // consume
      }
    }).rejects.not.toThrow(ProviderContextLengthError);
  });
}
