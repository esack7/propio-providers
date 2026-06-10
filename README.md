# @propio/providers

Provider adapters for LLM APIs with a unified streaming chat interface. Supports Anthropic (Claude), AWS Bedrock, Ollama, OpenRouter, Google Gemini, xAI (Grok), and Cloudflare Workers AI.

Extracted from [propio-agent](https://github.com/esack7/propio-agent), which uses it as its provider layer.

## Install

```bash
npm install @propio/providers
```

Requires Node.js >= 20. ESM only.

## Usage

```ts
import { createProvider, type ChatStreamEvent } from "@propio/providers";

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

`ProviderError` and subclasses `ProviderAuthenticationError`, `ProviderRateLimitError`, `ProviderCapacityError`, `ProviderModelNotFoundError`, `ProviderContextLengthError`.

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
