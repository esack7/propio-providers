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

When reasoning is requested, explicit Meta reasoning summaries are emitted as
`reasoning_summary` events and Meta `commentary` message text is emitted as
`thinking_delta` events. Final answer text remains `assistant_text`. Opaque
encrypted reasoning is retained only for continuation and is never emitted as
visible output.

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

### Request tracing

`ChatRequest.trace` accepts caller-owned session, run, turn, request, and operation identities plus a request purpose. `ChatRequest.onTraceEvent` observes logical request and network-attempt lifecycle events without creating files or discovering application directories. Observer failures are isolated from provider behavior.

Providers created through the normal factory emit attempt start, connection, failure, and retry-wait records from the shared retry machinery. OpenRouter additionally records its final-retry tools-removal mutation. Wrap any `LLMProvider` with `withProviderTracing(provider)` to add logical request start/completion/failure records while preserving the streamed event contract. The wrapper is explicit so `createProvider` continues returning the concrete adapter type for compatibility.

Attempt records include a stable attempt ID, attempt number, endpoint class, connection time, first-useful-output time, and terminal duration. Response metadata records the upstream request ID and actual routed model when the provider exposes them. xAI regional fallback requests are separate attempts rather than an opaque operation inside one retry.

Usage records keep input, output, cache-read, cache-write, reasoning, and total token counters distinct. Each report is marked as cumulative or delta and as reported, partial, or unavailable; missing values are never converted to zero. OpenRouter's provider-reported USD charge is retained separately when present. Reasoning tokens remain part of output tokens for providers that document them as an output-token breakdown.

| Adapter    | Trace availability                                                                                                  |
| ---------- | ------------------------------------------------------------------------------------------------------------------- |
| Anthropic  | Message ID, actual model, cumulative input/output/cache/thinking usage, raw stop reason                             |
| Bedrock    | AWS request ID, cumulative input/output/cache usage, raw stop reason; no separately returned actual model           |
| Ollama     | Actual model and final prompt/evaluation counts; no upstream request ID                                             |
| OpenRouter | Response ID, routed model, cumulative usage and reported charge, raw stop reason, tools-removal mutation            |
| OpenAI     | Responses ID/model, cumulative input/output/cache/reasoning usage, raw response status                              |
| Meta       | Responses ID/model, cumulative input/output/cache/reasoning usage, raw response status                              |
| Gemini     | OpenAI-compatible or native Gemini usage fields, response ID/model when supplied, raw finish reason                 |
| xAI        | Response ID/model and usage when supplied, raw finish reason/status, distinct global and regional endpoint attempts |
| Cloudflare | OpenAI-compatible response ID/model and usage when supplied, raw finish reason                                      |

Adapter events are emitted only from fields actually returned by the installed SDK or wire response. When a traced logical request completes without any provider usage report, `withProviderTracing` emits one explicit `unavailable` usage record linked to the final attempt.

## Development

```bash
npm install
npm test              # unit tests
npm run test:integration  # live-API tests (needs provider credentials)
npm run build
```

## License

MIT
