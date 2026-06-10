import { LLMProvider } from "./interface.js";
import {
  ProviderConfig,
  OllamaProviderConfig,
  BedrockProviderConfig,
  OpenRouterProviderConfig,
  GeminiProviderConfig,
  XaiProviderConfig,
  CloudflareProviderConfig,
  AnthropicProviderConfig,
  Model,
} from "./config.js";
import type {
  ProviderDiagnosticListener,
  ProviderRetryConfig,
} from "./diagnostics.js";
import { OllamaProvider } from "./providers/ollama.js";
import { BedrockProvider } from "./providers/bedrock.js";
import { OpenRouterProvider } from "./providers/openrouter.js";
import { GeminiProvider } from "./providers/gemini.js";
import { XaiProvider } from "./providers/xai.js";
import { CloudflareProvider } from "./providers/cloudflare.js";
import { AnthropicProvider } from "./providers/anthropic.js";

/**
 * Factory function to create LLM provider instances from configuration.
 *
 * This factory encapsulates provider instantiation logic, allowing new providers to be added
 * without modifying the Agent class. The factory uses a switch statement on type field
 * to determine which provider class to instantiate.
 *
 * @param config - Provider configuration containing type field and provider-specific settings
 * @param modelKey - Optional model key override. If provided, uses this instead of config.defaultModel
 * @returns An LLMProvider interface instance configured according to the provided config
 * @throws Error if the provider type is unknown or unsupported
 *
 * @example
 * // Create an Ollama provider
 * const ollamaProvider = createProvider({
 *   name: 'local-ollama',
 *   type: 'ollama',
 *   models: [{ name: 'Llama', key: 'llama3.2', contextWindowTokens: 8192 }],
 *   defaultModel: 'llama3.2',
 *   host: 'http://localhost:11434'
 * });
 *
 * @example
 * // Create a Bedrock provider with specific model
 * const bedrockProvider = createProvider({
 *   name: 'bedrock',
 *   type: 'bedrock',
 *   models: [{ name: 'Claude 3.5', key: 'anthropic.claude-3-5-sonnet-20241022-v2:0', contextWindowTokens: 200000 }],
 *   defaultModel: 'anthropic.claude-3-5-sonnet-20241022-v2:0',
 *   region: 'us-west-2'
 * }, 'anthropic.claude-3-5-sonnet-20241022-v2:0');
 */
export function createProvider(
  config: ProviderConfig,
  modelKey?: string,
  onDiagnosticEvent?: ProviderDiagnosticListener,
  debugLoggingEnabled = false,
  retryConfig?: ProviderRetryConfig,
): LLMProvider {
  const model = modelKey || config.defaultModel;

  const resolveModelConfig = (): Model => {
    const modelConfig = config.models.find((entry) => entry.key === model);
    if (!modelConfig) {
      const availableModels = config.models
        .map((entry) => entry.key)
        .join(", ");
      throw new Error(
        `Model "${model}" not found in provider "${config.name}". Available: ${availableModels}`,
      );
    }
    return modelConfig;
  };

  // Switch statement pattern for mapping provider type to implementation.
  // Each case instantiates the appropriate provider class with extracted config.
  switch (config.type) {
    case "ollama": {
      const modelConfig = resolveModelConfig();
      return new OllamaProvider({
        model: model,
        contextWindowTokens: modelConfig.contextWindowTokens,
        host: (config as OllamaProviderConfig).host,
        retryConfig,
        onDiagnosticEvent,
      });
    }
    case "bedrock": {
      const modelConfig = resolveModelConfig();
      return new BedrockProvider({
        model: model,
        contextWindowTokens: modelConfig.contextWindowTokens,
        region: (config as BedrockProviderConfig).region,
        retryConfig,
        onDiagnosticEvent,
      });
    }
    case "openrouter": {
      const openRouterConfig = config as OpenRouterProviderConfig;
      const modelConfig = resolveModelConfig();
      return new OpenRouterProvider({
        model,
        contextWindowTokens: modelConfig.contextWindowTokens,
        apiKey: openRouterConfig.apiKey,
        httpReferer: openRouterConfig.httpReferer,
        xTitle: openRouterConfig.xTitle,
        provider: openRouterConfig.provider,
        fallbackModels: openRouterConfig.fallbackModels,
        debugEchoUpstreamBody: openRouterConfig.debugEchoUpstreamBody,
        debugLoggingEnabled,
        onDiagnosticEvent,
        retryConfig,
      });
    }
    case "gemini": {
      const geminiConfig = config as GeminiProviderConfig;
      const modelConfig = resolveModelConfig();
      return new GeminiProvider({
        model,
        contextWindowTokens: modelConfig.contextWindowTokens,
        apiKey: geminiConfig.apiKey,
        retryConfig,
        onDiagnosticEvent,
      });
    }
    case "xai": {
      const xaiConfig = config as XaiProviderConfig;
      const modelConfig = resolveModelConfig();
      return new XaiProvider({
        model,
        contextWindowTokens: modelConfig.contextWindowTokens,
        apiKey: xaiConfig.apiKey,
        retryConfig,
        onDiagnosticEvent,
      });
    }
    case "cloudflare": {
      const cloudflareConfig = config as CloudflareProviderConfig;
      const modelConfig = resolveModelConfig();
      return new CloudflareProvider({
        model,
        contextWindowTokens: modelConfig.contextWindowTokens,
        apiKey: cloudflareConfig.apiKey,
        accountId: cloudflareConfig.accountId,
        retryConfig,
        onDiagnosticEvent,
      });
    }
    case "anthropic": {
      const anthropicConfig = config as AnthropicProviderConfig;
      const modelConfig = resolveModelConfig();
      return new AnthropicProvider({
        model,
        contextWindowTokens: modelConfig.contextWindowTokens,
        apiKey: anthropicConfig.apiKey,
        retryConfig,
        onDiagnosticEvent,
      });
    }
    default:
      throw new Error(
        `Unknown provider type: "${(config as any).type}". Valid providers: ollama, bedrock, openrouter, gemini, xai, cloudflare, anthropic`,
      );
  }
}

/**
 * Extract the default model name from a provider configuration.
 *
 * This utility function provides a centralized way to extract the default model name from any
 * provider configuration type. All provider configs now have a top-level defaultModel field.
 *
 * @param config - The provider configuration object
 * @returns The default model key string
 *
 * @example
 * const model = extractModelFromConfig({
 *   name: 'ollama',
 *   type: 'ollama',
 *   models: [{ name: 'Llama', key: 'llama3.2', contextWindowTokens: 8192 }],
 *   defaultModel: 'llama3.2'
 * });
 * console.log(model); // 'llama3.2'
 */
export function extractModelFromConfig(config: ProviderConfig): string {
  return config.defaultModel;
}
