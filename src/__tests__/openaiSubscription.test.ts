import {
  OpenAiSubscriptionProvider,
  ChatGptPlanError,
} from "../providers/openaiSubscription.js";
import {
  ProviderAuthenticationError,
  type ChatRequest,
  type ChatStreamEvent,
} from "../types.js";

const originalFetch = globalThis.fetch;
const request: ChatRequest = {
  model: "test-plan-model",
  messages: [
    { role: "system", content: "Help" },
    { role: "user", content: "Hello" },
  ],
};
function response(events: unknown[]) {
  return new Response(
    events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""),
    { headers: { "x-request-id": "req-poc" } },
  );
}
function provider(token = async () => "plan-token", idleTimeoutMs?: number) {
  return new OpenAiSubscriptionProvider({
    model: request.model,
    contextWindowTokens: 128000,
    getAccessToken: token,
    idleTimeoutMs,
    retryConfig: { maxRetries: 0, consecutive529Limit: 1 },
  });
}
async function collect(instance: OpenAiSubscriptionProvider, input = request) {
  const events: ChatStreamEvent[] = [];
  for await (const event of instance.streamChat(input)) events.push(event);
  return events;
}
afterEach(() => {
  globalThis.fetch = originalFetch;
});

it("uses application tokens per request, developer instructions, namespaced local tools, and stateless SSE", async () => {
  let tokenNumber = 0;
  globalThis.fetch = jest.fn().mockImplementation(async () =>
    response([
      { type: "response.output_text.delta", delta: "Hello" },
      { type: "response.completed", response: { status: "completed" } },
    ]),
  );
  const instance = provider(async () => `plan-token-${++tokenNumber}`);
  const input: ChatRequest = {
    ...request,
    tools: [
      {
        type: "function",
        function: {
          name: "lookup",
          description: "Lookup",
          parameters: { type: "object" },
        },
      },
    ],
  };
  await collect(instance, input);
  await collect(instance, input);
  const calls = (fetch as jest.Mock).mock.calls;
  expect(calls[0][1].headers.Authorization).toBe("Bearer plan-token-1");
  expect(calls[1][1].headers.Authorization).toBe("Bearer plan-token-2");
  const body = JSON.parse(calls[0][1].body);
  expect(body).toMatchObject({
    store: false,
    stream: true,
    input: [
      { role: "developer", content: "Help" },
      { role: "user", content: "Hello" },
    ],
    tools: [
      { type: "namespace", name: "functions", tools: [{ name: "lookup" }] },
    ],
  });
  expect(body).not.toHaveProperty("previous_response_id");
});

it("preserves namespaced tool and reasoning items through result replay", async () => {
  const call = {
    type: "function_call",
    id: "fc1",
    call_id: "call1",
    namespace: "functions",
    name: "lookup",
    arguments: "{}",
    status: "completed",
  };
  const reasoning = {
    type: "reasoning",
    id: "rs1",
    encrypted_content: "opaque",
    summary: [],
  };
  globalThis.fetch = jest
    .fn()
    .mockResolvedValueOnce(
      response([
        { type: "response.output_item.done", output_index: 0, item: reasoning },
        { type: "response.output_item.done", output_index: 1, item: call },
        { type: "response.completed", response: { status: "completed" } },
      ]),
    )
    .mockResolvedValueOnce(
      response([
        { type: "response.completed", response: { status: "completed" } },
      ]),
    );
  const instance = provider();
  const events = await collect(instance);
  const tools = events.find(
    (event) => "type" in event && event.type === "tool_calls",
  );
  if (!tools || !("type" in tools) || tools.type !== "tool_calls")
    throw new Error("No tools");
  expect(tools.toolCalls[0].function.name).toBe("lookup");
  await collect(instance, {
    ...request,
    messages: [
      ...request.messages,
      {
        role: "assistant",
        content: "",
        toolCalls: tools.toolCalls,
        reasoningContent: tools.reasoningContent,
      },
      { role: "tool", content: "ready", toolCallId: "call1" },
    ],
  });
  const body = JSON.parse((fetch as jest.Mock).mock.calls[1][1].body);
  expect(body.input).toEqual(
    expect.arrayContaining([
      reasoning,
      call,
      { type: "function_call_output", call_id: "call1", output: "ready" },
    ]),
  );
});

it.each([
  { events: [] },
  { events: [{ type: "response.output_text.delta", delta: "partial" }] },
  { events: [{ type: "response.done", response: { status: "completed" } }] },
])("rejects streams without response.completed: %j", async ({ events }) => {
  globalThis.fetch = jest.fn().mockResolvedValue(response(events));
  await expect(collect(provider())).rejects.toThrow(
    "without response.completed",
  );
});

it.each(["response.failed", "error"])(
  "retains subscription error metadata and never emits pending tools on %s",
  async (type) => {
    globalThis.fetch = jest.fn().mockResolvedValue(
      response([
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
          type,
          error: {
            code: "subscription_sharing_usage_limit_exceeded",
            message: "App limit reached",
          },
          response: {
            error: {
              code: "subscription_sharing_usage_limit_exceeded",
              message: "App limit reached",
            },
          },
        },
      ]),
    );
    const seen: ChatStreamEvent[] = [];
    try {
      for await (const event of provider().streamChat(request))
        seen.push(event);
      throw new Error("Expected failure");
    } catch (error) {
      expect(error).toMatchObject({
        name: "ChatGptPlanError",
        code: "subscription_sharing_usage_limit_exceeded",
        requestId: "req-poc",
        status: 200,
      });
    }
    expect(seen).toEqual([]);
  },
);

it("rejects incomplete responses", async () => {
  globalThis.fetch = jest
    .fn()
    .mockResolvedValue(
      response([
        { type: "response.incomplete", response: { status: "incomplete" } },
      ]),
    );
  await expect(collect(provider())).rejects.toThrow("incomplete");
});

it("preserves admission codes and request IDs", async () => {
  globalThis.fetch = jest.fn().mockResolvedValue(
    new Response(
      JSON.stringify({
        error: {
          code: "subscription_sharing_user_not_eligible",
          message: "Unavailable",
          param: "model",
        },
      }),
      { status: 403, headers: { "x-request-id": "req-admission" } },
    ),
  );
  await expect(collect(provider())).rejects.toMatchObject({
    code: "subscription_sharing_user_not_eligible",
    status: 403,
    requestId: "req-admission",
    param: "model",
  });
  expect(fetch).toHaveBeenCalledTimes(1);
});

it("does not fall back to API credentials when the account callback fails", async () => {
  globalThis.fetch = jest.fn();
  await expect(
    collect(
      provider(async () => {
        throw new ProviderAuthenticationError("Sign in");
      }),
    ),
  ).rejects.toThrow("Sign in");
  expect(fetch).not.toHaveBeenCalled();
});

it("rejects blank tokens", async () => {
  await expect(collect(provider(async () => ""))).rejects.toBeInstanceOf(
    ProviderAuthenticationError,
  );
});

it("keeps structured errors as provider errors", () => {
  expect(new ChatGptPlanError("limit").name).toBe("ChatGptPlanError");
});

it("finishes and cancels an open reader at response.completed", async () => {
  let cancelled = false;
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(
        new TextEncoder().encode(
          'data: {"type":"response.completed","response":{"status":"completed"}}\n\n',
        ),
      );
    },
    cancel() {
      cancelled = true;
    },
  });
  globalThis.fetch = jest.fn().mockResolvedValue(new Response(stream));
  expect((await collect(provider())).at(-1)?.type).toBe("terminal");
  expect(cancelled).toBe(true);
});

it("cancels an idle reader on caller abort without dispatching pending calls", async () => {
  let cancelled = false;
  const controller = new AbortController();
  globalThis.fetch = jest.fn().mockResolvedValue(
    new Response(
      new ReadableStream({
        cancel() {
          cancelled = true;
        },
      }),
    ),
  );
  const pending = collect(provider(), {
    ...request,
    signal: controller.signal,
  });
  setTimeout(() => controller.abort(), 20);
  await expect(pending).rejects.toMatchObject({ name: "AbortError" });
  expect(cancelled).toBe(true);
});

it("rejects an idle unconfirmed stream and closes its reader", async () => {
  jest.useFakeTimers();
  let cancelled = false;
  try {
    globalThis.fetch = jest.fn().mockResolvedValue(
      new Response(
        new ReadableStream({
          cancel() {
            cancelled = true;
          },
        }),
      ),
    );
    const pending = expect(collect(provider(undefined, 60000))).rejects.toThrow(
      "idle timeout",
    );
    await jest.advanceTimersByTimeAsync(60001);
    await pending;
    expect(cancelled).toBe(true);
  } finally {
    jest.useRealTimers();
  }
});
