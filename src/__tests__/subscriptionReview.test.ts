import { createProvider } from "../factory.js";
import {
  OpenAiSubscriptionProvider,
  ChatGptPlanError,
  type OpenAiSubscriptionOptions,
} from "../providers/openaiSubscription.js";
import { OpenAiProvider } from "../providers/openai.js";
import { MetaProvider } from "../providers/meta.js";
import { withProviderTracing, type ProviderTraceEvent } from "../trace.js";
import {
  ProviderAuthenticationError,
  ProviderModelNotFoundError,
  ProviderRateLimitError,
  ProviderError,
  ProviderCapacityError,
  ProviderInvalidRequestError,
  type ChatRequest,
  type ChatStreamEvent,
} from "../types.js";
import type { LLMProvider } from "../interface.js";
const originalFetch = globalThis.fetch;
const request: ChatRequest = { model: "review-model", messages: [] };
const options = {
  model: request.model,
  contextWindowTokens: 128000,
  retryConfig: { maxRetries: 1, consecutive529Limit: 2, baseDelayMs: 0 },
};
const subscription = () =>
  new OpenAiSubscriptionProvider({
    ...options,
    getAccessToken: async () => "credential-sentinel",
  });
function sse(events: unknown[]) {
  return new Response(
    events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""),
  );
}
async function collect(provider: LLMProvider, input = request) {
  const events: ChatStreamEvent[] = [];
  for await (const event of provider.streamChat(input)) events.push(event);
  return events;
}
afterEach(() => {
  globalThis.fetch = originalFetch;
  jest.useRealTimers();
});

it.each([
  [401, ProviderAuthenticationError],
  [403, ProviderAuthenticationError],
  [404, ProviderModelNotFoundError],
  [429, ProviderRateLimitError],
  [503, ProviderCapacityError],
  [400, ProviderInvalidRequestError],
])("keeps HTTP %s errors typed and safe", async (status, ErrorType) => {
  globalThis.fetch = jest.fn().mockImplementation(
    async () =>
      new Response(
        JSON.stringify({
          error: { message: "credential-sentinel", param: "model" },
        }),
        {
          status: status as number,
          headers: { "x-request-id": "req-review", "retry-after": "4" },
        },
      ),
  );
  let caught: unknown;
  try {
    await collect(subscription());
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(ErrorType);
  expect(caught).toMatchObject({
    status,
    requestId: "req-review",
    param: "model",
  });
  expect(String(caught)).not.toContain("credential-sentinel");
  expect(JSON.stringify(caught)).not.toContain("credential-sentinel");
  if (status === 429) expect(caught).toMatchObject({ retryAfterSeconds: 4 });
  expect(fetch).toHaveBeenCalledTimes(status === 429 ? 2 : 1);
});

it("maps a non-JSON expired-token admission failure to authentication", async () => {
  globalThis.fetch = jest
    .fn()
    .mockResolvedValue(new Response("credential-sentinel", { status: 401 }));
  await expect(collect(subscription())).rejects.toBeInstanceOf(
    ProviderAuthenticationError,
  );
  expect(fetch).toHaveBeenCalledTimes(1);
});

it("maps invalid subscriber context to authentication even inside SSE", async () => {
  globalThis.fetch = jest.fn().mockResolvedValue(
    sse([
      {
        type: "response.failed",
        response: { error: { code: "subscription_sharing_invalid_user" } },
      },
    ]),
  );
  await expect(collect(subscription())).rejects.toBeInstanceOf(
    ProviderAuthenticationError,
  );
  expect(fetch).toHaveBeenCalledTimes(1);
});

it("retries generic rate-limit admission once with a fresh application token", async () => {
  const getAccessToken = jest
    .fn()
    .mockResolvedValueOnce("token-1")
    .mockResolvedValueOnce("token-2");
  globalThis.fetch = jest
    .fn()
    .mockResolvedValueOnce(
      new Response("rate limit", {
        status: 429,
        headers: { "retry-after": "4" },
      }),
    )
    .mockResolvedValueOnce(sse([{ type: "response.completed" }]));
  await collect(new OpenAiSubscriptionProvider({ ...options, getAccessToken }));
  expect(getAccessToken).toHaveBeenCalledTimes(2);
  expect(
    (fetch as jest.Mock).mock.calls.map(
      (call) => call[1].headers.Authorization,
    ),
  ).toEqual(["Bearer token-1", "Bearer token-2"]);
});

it("does not retry a plan usage limit presented as HTTP 429", async () => {
  globalThis.fetch = jest
    .fn()
    .mockResolvedValue(
      Response.json(
        { error: { code: "subscription_sharing_usage_limit_exceeded" } },
        { status: 429 },
      ),
    );
  await expect(collect(subscription())).rejects.toBeInstanceOf(
    ChatGptPlanError,
  );
  expect(fetch).toHaveBeenCalledTimes(1);
});

it.each([
  "subscription_sharing_usage_unavailable",
  "subscription_sharing_user_unavailable",
])(
  "bounds retries for documented temporary admission code %s",
  async (code) => {
    globalThis.fetch = jest
      .fn()
      .mockImplementation(async () =>
        Response.json({ error: { code } }, { status: 503 }),
      );
    await expect(collect(subscription())).rejects.toMatchObject({
      code,
      retryable: true,
    });
    expect(fetch).toHaveBeenCalledTimes(2);
  },
);

it.each([OpenAiProvider, MetaProvider])(
  "preserves API cancellation error with a custom signal reason for %p",
  async (Provider) => {
    const controller = new AbortController();
    const reason = { application: "cancelled" };
    globalThis.fetch = jest.fn().mockImplementation(async () => {
      controller.abort(reason);
      throw new DOMException("aborted", "AbortError");
    });
    const instance = new Provider({ ...options, apiKey: "api-fixture" });
    await expect(
      collect(instance, { ...request, signal: controller.signal }),
    ).rejects.toMatchObject({
      name: "ProviderError",
      message: "Request cancelled",
    });
    expect(fetch).toHaveBeenCalledTimes(1);
  },
);

it("preserves a subscription caller's exact custom cancellation reason", async () => {
  const controller = new AbortController();
  const reason = { application: "cancelled" };
  globalThis.fetch = jest.fn().mockImplementation(async () => {
    controller.abort(reason);
    throw new DOMException("aborted", "AbortError");
  });
  await expect(
    collect(subscription(), { ...request, signal: controller.signal }),
  ).rejects.toBe(reason);
  expect(fetch).toHaveBeenCalledTimes(1);
});

it("translates a transport AbortError without throwing from the error translator", async () => {
  globalThis.fetch = jest
    .fn()
    .mockRejectedValue(new DOMException("aborted", "AbortError"));
  await expect(collect(subscription())).rejects.toMatchObject({
    name: "ProviderError",
    message: "Request cancelled",
  });
  expect(fetch).toHaveBeenCalledTimes(1);
});

it("fails closed for JavaScript callers without a credential function", () => {
  const invalid = {
    ...options,
    apiKey: "api-fixture",
    getAccessToken: undefined,
  } as unknown as OpenAiSubscriptionOptions;
  expect(() => new OpenAiSubscriptionProvider(invalid)).toThrow(
    ProviderAuthenticationError,
  );
});

it("never reads either API-key environment variable when a token source is supplied", () => {
  const originalEnvironment = process.env;
  const reads: PropertyKey[] = [];
  process.env = new Proxy(originalEnvironment, {
    get(target, key) {
      reads.push(key);
      return Reflect.get(target, key);
    },
  });
  try {
    subscription();
  } finally {
    process.env = originalEnvironment;
  }
  expect(reads).not.toContain("OPENAI_API_KEY");
  expect(reads).not.toContain("PROPIO_UNUSED_SUBSCRIPTION_API_KEY");
});

it("records reported usage from failed SSE without admitting tools or terminal success", async () => {
  globalThis.fetch = jest.fn().mockResolvedValue(
    sse([
      {
        type: "response.output_item.done",
        output_index: 0,
        item: {
          type: "function_call",
          name: "lookup",
          call_id: "call1",
          arguments: "{}",
        },
      },
      {
        type: "response.failed",
        response: {
          id: "resp-failed",
          usage: { input_tokens: 123, output_tokens: 45, total_tokens: 168 },
          error: { code: "subscription_sharing_usage_limit_exceeded" },
        },
      },
    ]),
  );
  const traceEvents: ProviderTraceEvent[] = [];
  const visible: ChatStreamEvent[] = [];
  try {
    for await (const event of withProviderTracing(subscription()).streamChat({
      ...request,
      trace: {
        requestId: "review-request",
        operationId: "review-operation",
        purpose: "answer",
      },
      onTraceEvent: (event) => traceEvents.push(event),
    }))
      visible.push(event);
    throw new Error("Expected failed response");
  } catch (error) {
    expect(error).toBeInstanceOf(ChatGptPlanError);
  }
  expect(visible).toEqual([]);
  expect(traceEvents).toContainEqual(
    expect.objectContaining({
      type: "provider_usage_reported",
      availability: "reported",
      usage: { inputTokens: 123, outputTokens: 45, totalTokens: 168 },
    }),
  );
  expect(traceEvents).toContainEqual(
    expect.objectContaining({ type: "provider_request_failed" }),
  );
  expect(
    traceEvents.some((event) => event.type === "provider_request_completed"),
  ).toBe(false);
  expect(fetch).toHaveBeenCalledTimes(1);
});

it("allows long reasoning gaps and resets the five-minute default on each chunk", async () => {
  jest.useFakeTimers();
  let writer!: ReadableStreamDefaultController<Uint8Array>;
  let cancelled = false;
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      writer = controller;
    },
    cancel() {
      cancelled = true;
    },
  });
  const send = (event: unknown) =>
    writer.enqueue(
      new TextEncoder().encode(`data: ${JSON.stringify(event)}\n\n`),
    );
  globalThis.fetch = jest.fn().mockResolvedValue(new Response(stream));
  let settled = false;
  const result = collect(subscription()).finally(() => {
    settled = true;
  });
  // Attach a rejection observer before advancing the clock.
  const outcome = result.catch((error: unknown) => error);
  send({ type: "response.output_item.added" });
  await jest.advanceTimersByTimeAsync(240000);
  expect(settled).toBe(false);
  expect(cancelled).toBe(false);
  send({ type: "response.output_item.added", item: { type: "reasoning" } });
  await jest.advanceTimersByTimeAsync(240000);
  expect(settled).toBe(false);
  send({ type: "response.completed" });
  expect(await outcome).toEqual([
    expect.objectContaining({ type: "terminal" }),
  ]);
  expect(cancelled).toBe(true);
});

it("honors the factory's configurable idle bound and cleans up its reader", async () => {
  jest.useFakeTimers();
  let cancelled = false;
  globalThis.fetch = jest.fn().mockResolvedValue(
    new Response(
      new ReadableStream({
        cancel() {
          cancelled = true;
        },
      }),
    ),
  );
  const instance = createProvider(
    {
      name: "plan",
      type: "openai",
      auth: { mode: "chatgpt", accountId: "test-account" },
      defaultModel: request.model,
      models: [
        { key: request.model, name: "Review", contextWindowTokens: 128000 },
      ],
    },
    undefined,
    undefined,
    false,
    undefined,
    {
      openAiCredentialSource: () => ({ getAccessToken: async () => "fixture" }),
      openAiSubscription: { idleTimeoutMs: 25 },
    },
  );
  const result = expect(collect(instance)).rejects.toBeInstanceOf(
    ProviderError,
  );
  await jest.advanceTimersByTimeAsync(26);
  await result;
  expect(cancelled).toBe(true);
});

it.each([0, -1, 1.5, Infinity, 2147483648])(
  "rejects invalid idle timeout %s",
  (idleTimeoutMs) => {
    expect(
      () =>
        new OpenAiSubscriptionProvider({
          ...options,
          getAccessToken: async () => "fixture",
          idleTimeoutMs,
        }),
    ).toThrow("idleTimeoutMs");
  },
);
