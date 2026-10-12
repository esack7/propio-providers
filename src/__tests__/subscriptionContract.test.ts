import { createProvider } from "../factory.js";
import { validateProvidersConfig } from "../configValidation.js";
import { ProviderContextLengthError } from "../types.js";
import type { OpenAiProviderConfig } from "../config.js";
const config: OpenAiProviderConfig = {
  name: "plan",
  type: "openai",
  auth: { mode: "chatgpt", accountId: "registration-1" },
  models: [
    { key: "plan-model", name: "Plan model", contextWindowTokens: 128000 },
  ],
  defaultModel: "plan-model",
};
const oldFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = oldFetch;
});
function validate(auth: unknown, apiKey?: string) {
  return validateProvidersConfig({
    default: "plan",
    providers: [{ ...config, auth, apiKey }],
  });
}

it.each([
  undefined,
  { mode: "api-key" },
  { mode: "chatgpt", accountId: "registration-1" },
])("accepts additive access configuration %j", (auth) => {
  expect(validate(auth).default).toBe("plan");
});
it.each([
  { mode: "other" },
  { mode: "chatgpt", accountId: "" },
  { mode: "chatgpt" },
  null,
])("rejects invalid auth %j", (auth) => {
  expect(() => validate(auth)).toThrow();
});
it("rejects API secrets and requires injected credentials for plan access", () => {
  expect(() => validate(config.auth, "api-secret")).toThrow("billing");
  expect(() => createProvider(config)).toThrow("openAiCredentialSource");
});
it("pins the source and never consults OPENAI_API_KEY for plan access", async () => {
  const previous = process.env.OPENAI_API_KEY;
  process.env.OPENAI_API_KEY = "api-sentinel";
  const ids: string[] = [];
  const provider = createProvider(
    config,
    undefined,
    undefined,
    false,
    undefined,
    {
      openAiCredentialSource: (id) => {
        ids.push(id);
        return { getAccessToken: async () => "plan-sentinel" };
      },
    },
  );
  globalThis.fetch = jest
    .fn()
    .mockResolvedValue(new Response('data: {"type":"response.completed"}\n\n'));
  try {
    for await (const _ of provider.streamChat({
      model: config.defaultModel,
      messages: [],
    })) {
    }
  } finally {
    if (previous === undefined) delete process.env.OPENAI_API_KEY;
    else process.env.OPENAI_API_KEY = previous;
  }
  expect(ids).toEqual(["registration-1"]);
  expect((fetch as jest.Mock).mock.calls[0][1].headers.Authorization).toBe(
    "Bearer plan-sentinel",
  );
});
it("keeps context length errors typed and excludes echoed secrets", async () => {
  const provider = createProvider(
    config,
    undefined,
    undefined,
    false,
    undefined,
    {
      openAiCredentialSource: () => ({
        getAccessToken: async () => "plan-sentinel",
      }),
    },
  );
  globalThis.fetch = jest.fn().mockResolvedValue(
    Response.json(
      {
        error: { code: "context_length_exceeded", message: "plan-sentinel" },
      },
      { status: 400 },
    ),
  );
  try {
    for await (const _ of provider.streamChat({
      model: config.defaultModel,
      messages: [],
    })) {
    }
    throw new Error("Expected failure");
  } catch (error) {
    expect(error).toBeInstanceOf(ProviderContextLengthError);
    expect(String(error)).not.toContain("plan-sentinel");
  }
});
