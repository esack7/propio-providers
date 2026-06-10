import { OpenRouterProvider } from "../providers/openrouter.js";
import type { ChatMessage, ChatRequest, ChatStreamEvent } from "../types.js";

const DEFAULT_CHAT_REQUEST: ChatRequest = {
  model: "openai/gpt-3.5-turbo",
  messages: [{ role: "user", content: "Hi" }] satisfies ChatMessage[],
};
const DEFAULT_CONTEXT_WINDOW = 128_000;

type RetryConfig = {
  maxRetries: number;
  baseDelayMs: number;
  consecutive529Limit: number;
};

type StreamingProvider = {
  streamChat(request: ChatRequest): AsyncIterable<ChatStreamEvent>;
};

export function registerProviderTestLifecycle(
  originalEnv: NodeJS.ProcessEnv,
  originalFetch: typeof globalThis.fetch,
): void {
  beforeEach(() => {
    jest.resetModules();
    process.env = { ...originalEnv };
    globalThis.fetch = originalFetch;
  });

  afterAll(() => {
    process.env = originalEnv;
    globalThis.fetch = originalFetch;
  });
}

export function registerAcceptsApiKeyTest<T extends { name: string }>(options: {
  expectedName: string;
  createProvider: () => T;
}): void {
  it("should accept API key from options", () => {
    const provider = options.createProvider();
    expect(provider.name).toBe(options.expectedName);
  });
}

export class OpenRouterTestFixture {
  static createSseStream(chunks: string[]): ReadableStream<Uint8Array> {
    return new ReadableStream({
      start(controller) {
        chunks.forEach((chunk) =>
          controller.enqueue(new TextEncoder().encode(chunk)),
        );
        controller.close();
      },
    });
  }

  static setupFetchMock(
    chunks: string[],
    captureBody?: (body: unknown) => void,
  ): jest.Mock {
    const mock = jest
      .fn()
      .mockImplementation((_url: string, init?: RequestInit) => {
        if (captureBody && init?.body) {
          captureBody(JSON.parse(init.body as string));
        }
        return Promise.resolve({
          ok: true,
          body: OpenRouterTestFixture.createSseStream(chunks),
        });
      });
    globalThis.fetch = mock;
    return mock;
  }

  static createProvider(
    model = "openai/gpt-3.5-turbo",
    apiKey = "sk-test",
    retryConfig?: {
      maxRetries: number;
      baseDelayMs: number;
      consecutive529Limit: number;
    },
  ): OpenRouterProvider {
    return new OpenRouterProvider({
      model,
      contextWindowTokens: DEFAULT_CONTEXT_WINDOW,
      apiKey,
      retryConfig,
    });
  }

  static createRequest(overrides: Partial<ChatRequest> = {}): ChatRequest {
    return {
      ...DEFAULT_CHAT_REQUEST,
      ...overrides,
      messages: overrides.messages ?? DEFAULT_CHAT_REQUEST.messages,
    };
  }

  static async consumeStream(
    provider: StreamingProvider,
    request: ChatRequest = DEFAULT_CHAT_REQUEST,
  ): Promise<ChatStreamEvent[]> {
    const events: ChatStreamEvent[] = [];
    for await (const event of provider.streamChat(request)) {
      events.push(event);
    }
    return events;
  }

  static async collectDeltas(
    provider: StreamingProvider,
    request: ChatRequest = DEFAULT_CHAT_REQUEST,
  ): Promise<string[]> {
    return (await OpenRouterTestFixture.consumeStream(provider, request)).map(
      (event) => event.delta,
    );
  }

  static async expectStreamChatToThrow(
    provider: OpenRouterProvider,
    matcher: string | RegExp | (new (...args: unknown[]) => unknown),
    request: ChatRequest = DEFAULT_CHAT_REQUEST,
  ): Promise<void> {
    await expect(async () => {
      for await (const _chunk of provider.streamChat(request)) {
        // consume
      }
    }).rejects.toThrow(matcher as any);
  }

  /**
   * Sets up a fetch mock returning the given response, creates a default
   * provider, expects streamChat to throw the given matcher, and returns the
   * mock so callers can assert call counts.
   */
  static async expectErrorOnStatus(
    fetchResponse: Record<string, unknown>,
    errorMatcher: string | RegExp | (new (...args: unknown[]) => unknown),
    model = "openai/gpt-3.5-turbo",
    apiKey = "sk-test",
    retryConfig?: RetryConfig,
  ): Promise<jest.Mock> {
    const fetchMock = jest.fn().mockResolvedValue(fetchResponse);
    globalThis.fetch = fetchMock;
    const provider = new OpenRouterProvider({
      model,
      contextWindowTokens: DEFAULT_CONTEXT_WINDOW,
      apiKey,
      retryConfig,
    });
    await OpenRouterTestFixture.expectStreamChatToThrow(provider, errorMatcher);
    return fetchMock;
  }

  /**
   * Sets up a fetch mock that always returns the same failing response,
   * creates a provider with one retry, and expects the stream to throw the
   * given error class. Returns the mock so callers can assert call counts.
   */
  static async expectRetryError(
    fetchResponse: Record<string, unknown>,
    errorMatcher: string | RegExp | (new (...args: unknown[]) => unknown),
    retryConfig: RetryConfig = {
      maxRetries: 1,
      baseDelayMs: 0,
      consecutive529Limit: 3,
    },
  ): Promise<jest.Mock> {
    const fetchMock = jest.fn().mockResolvedValue(fetchResponse);
    globalThis.fetch = fetchMock;
    const provider = new OpenRouterProvider({
      model: "openai/gpt-3.5-turbo",
      contextWindowTokens: DEFAULT_CONTEXT_WINDOW,
      apiKey: "sk-test",
      retryConfig,
    });
    await OpenRouterTestFixture.expectStreamChatToThrow(provider, errorMatcher);
    return fetchMock;
  }

  /**
   * Runs streamChat inside a try/catch and returns the caught error (or
   * undefined if no error was thrown). Useful for asserting on error properties
   * that are not easily checked with `expect(...).rejects.toThrow`.
   */
  static async catchStreamError(
    provider: OpenRouterProvider,
    request: ChatRequest = DEFAULT_CHAT_REQUEST,
  ): Promise<unknown> {
    let caughtError: unknown;
    try {
      for await (const _chunk of provider.streamChat(request)) {
        // consume
      }
    } catch (error) {
      caughtError = error;
    }
    return caughtError;
  }

  /**
   * Creates a fetch mock that fails on the first call (with the given status)
   * and succeeds on the second call, streaming the given content back.
   * Also sets `globalThis.fetch` to the mock and creates a provider with one
   * retry. Returns both the mock and the provider so callers can make requests
   * and assert call counts.
   */
  static createRetrySuccessMock(
    failStatus: number,
    successContent = "Recovered",
    extraProviderOpts: Record<string, unknown> = {},
  ): { fetchMock: jest.Mock; provider: OpenRouterProvider } {
    const fetchMock = jest
      .fn()
      .mockResolvedValueOnce({ ok: false, status: failStatus })
      .mockResolvedValueOnce({
        ok: true,
        body: OpenRouterTestFixture.createSseStream([
          `data: {"choices":[{"delta":{"content":"${successContent}"}}]}\n\n`,
          "data: [DONE]\n\n",
        ]),
      });
    globalThis.fetch = fetchMock;
    const provider = new OpenRouterProvider({
      model: "openai/gpt-3.5-turbo",
      contextWindowTokens: DEFAULT_CONTEXT_WINDOW,
      apiKey: "sk-test",
      retryConfig: { maxRetries: 1, baseDelayMs: 0, consecutive529Limit: 3 },
      ...extraProviderOpts,
    });
    return { fetchMock, provider };
  }
}

export function mockOpenRouterSseStream(chunks: string[]): void {
  globalThis.fetch = jest.fn().mockResolvedValue({
    ok: true,
    body: OpenRouterTestFixture.createSseStream(chunks),
  });
}

export async function collectOpenRouterThinkingEvents(
  provider: StreamingProvider,
  chunks: string[],
  request: ChatRequest = DEFAULT_CHAT_REQUEST,
): Promise<string[]> {
  mockOpenRouterSseStream(chunks);

  const thinkingEvents: string[] = [];
  for await (const event of provider.streamChat(request)) {
    if (event.type === "thinking_delta") {
      thinkingEvents.push(event.delta);
    }
  }

  return thinkingEvents;
}

export async function collectOpenRouterThinkingAndToolEvents(
  provider: StreamingProvider,
  chunks: string[],
  request: ChatRequest,
): Promise<{ thinkingEvents: string[]; toolEvents: ChatStreamEvent[] }> {
  mockOpenRouterSseStream(chunks);

  const thinkingEvents: string[] = [];
  const toolEvents: ChatStreamEvent[] = [];
  for await (const event of provider.streamChat(request)) {
    if (event.type === "thinking_delta") {
      thinkingEvents.push(event.delta);
    }
    if (event.type === "tool_calls") {
      toolEvents.push(event);
    }
  }

  return { thinkingEvents, toolEvents };
}
