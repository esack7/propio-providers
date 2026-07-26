# @propio-ai/providers

Provider adapters for LLM APIs with a unified streaming chat interface. Supports Anthropic (Claude), AWS Bedrock, Ollama, OpenRouter, OpenAI, Meta Model API, Google Gemini, xAI (Grok), and Cloudflare Workers AI.

Extracted from [propio-agent](https://github.com/esack7/propio-agent), which uses it as its provider layer.

## Install

```bash
npm install @propio-ai/providers
```

Requires Node.js >= 20. ESM only.

## Usage

```ts
import { createProvider, type ChatStreamEvent } from "@propio-ai/providers";

const provider = createProvider({
  name: "claude",
  type: "anthropic",
  models: [
    {
      name: "Claude Sonnet",
      key: "claude-sonnet-4-6",
      contextWindowTokens: 200000,
    },
  ],
  defaultModel: "claude-sonnet-4-6",
  apiKey: process.env.ANTHROPIC_API_KEY,
});

for await (const event of provider.streamChat({
  model: "claude-sonnet-4-6",
  messages: [{ role: "user", content: "Hello!" }],
})) {
  if ("type" in event && event.type === "assistant_text") {
    process.stdout.write(event.delta);
  }
}
```

### OpenAI

The first-party OpenAI provider uses the Responses API. Supply an API key directly or set `OPENAI_API_KEY`:

```ts
import { createProvider } from "@propio-ai/providers";

const openai = createProvider({
  name: "openai",
  type: "openai",
  models: [
    {
      name: "GPT-5.5",
      key: "gpt-5.5",
      contextWindowTokens: 1_050_000,
    },
    {
      name: "GPT-5.4",
      key: "gpt-5.4",
      contextWindowTokens: 1_050_000,
    },
    {
      name: "GPT-5.4 mini",
      key: "gpt-5.4-mini",
      contextWindowTokens: 400_000,
    },
  ],
  defaultModel: "gpt-5.5",
  apiKey: process.env.OPENAI_API_KEY,
});
```

Model support is configuration-driven. To adopt a newly generally available model, add its model ID, display name, and documented context window to `models`, then optionally select it as `defaultModel`. The provider does not contain a model-name allowlist or version-specific routing. Check [OpenAI's model catalog](https://developers.openai.com/api/docs/models) for current IDs and limits before changing configuration.

The provider streams assistant text, function calls, and OpenAI-provided reasoning summaries through the shared event contract. Tool-call continuation preserves encrypted OpenAI reasoning state internally; raw chain-of-thought is never exposed as an event.

### Meta Model API

The Meta provider uses Meta's OpenAI-compatible Responses API. Supply an API key directly or set `META_API_KEY`:

```ts
import { createProvider } from "@propio-ai/providers";

const meta = createProvider({
  name: "meta",
  type: "meta",
  models: [
    {
      name: "Muse Spark 1.1",
      key: "muse-spark-1.1",
      contextWindowTokens: 1_048_576,
    },
  ],
  defaultModel: "muse-spark-1.1",
  apiKey: process.env.META_API_KEY,
});
```

Meta requests stream from `https://api.meta.ai/v1/responses` without server-side response storage. Encrypted reasoning and completed output items are preserved internally so tool-call turns can be replayed in provider order. Model support remains configuration-driven; future Meta model IDs do not require a library update.

The provider reads only the namespaced `META_API_KEY` environment variable; it does not fall back to the generic `MODEL_API_KEY`. Meta continuation state can include completed assistant commentary items, so `reasoningContent` may contain plaintext user-visible commentary in addition to opaque reasoning and function-call state. Applications that persist `reasoningContent` should protect it as conversation content.

## API

### Factory

- `createProvider(config, modelKey?, onDiagnosticEvent?, debugLoggingEnabled?, retryConfig?)` — instantiate an `LLMProvider` from a `ProviderConfig`
- `extractModelFromConfig(config)` — read the default model key from a provider config

### Provider contract

`LLMProvider` exposes `name`, `getCapabilities()`, and `streamChat(request)`, which yields `ChatStreamEvent` values (`assistant_text`, `thinking_delta`, `tool_calls`, `status`, `reasoning_summary`, `terminal`).

`ProviderCapabilities.supportsSyntheticToolCallHistory` is `false` for providers (currently Gemini) that reject caller-fabricated assistant tool-call history; callers should inline such content into a user message instead.

### Configuration

- `validateProvidersConfig(value)` — validate an arbitrary parsed value as a `ProvidersConfig`
- `resolveProvider(config, name?)` / `resolveModelKey(provider, key?)`
- `getDefaultProviderModelSelection(config)` / `updateDefaultProviderModelSelection(config, providerName, modelKey?)`
- `loadProvidersConfig(filePath, options?)` / `loadProvidersConfigAsync(filePath, options?)` — load + validate from an explicit file path; `options.missingMessage` customizes the missing-file error
- `writeProvidersConfig(filePath, config)` — atomic write
- `updateDefaultProviderModelSelectionInFile(filePath, providerName, modelKey?)`

### Errors

`ProviderError` and subclasses `ProviderAuthenticationError`, `ProviderRateLimitError`, `ProviderCapacityError`, `ProviderModelNotFoundError`, `ProviderContextLengthError`, `ProviderInvalidRequestError`. A non-context-length HTTP 400 is reported as `ProviderInvalidRequestError` and is not retried by OpenAI-compatible providers or OpenRouter.

### Diagnostics

Pass a `ProviderDiagnosticListener` to `createProvider` to receive `ProviderDiagnosticEvent`s (currently `provider_retry`, emitted when a request is retried).

## Development

```bash
npm install
npm test              # unit tests
npm run test:integration  # live-API tests (needs provider credentials)
npm run build
```

## License

MIT
