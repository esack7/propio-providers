/**
 * A model entry containing both human-readable name and technical key.
 * contextWindowTokens is required because providers do not maintain
 * hardcoded per-model capability tables.
 */
export interface Model {
  name: string;
  key: string;
  contextWindowTokens: number;
}

/**
 * Base provider configuration with common fields
 */
export interface BaseProviderConfig {
  name: string;
  type: string;
  models: Model[];
  defaultModel: string;
}

/**
 * Ollama provider configuration with flat structure
 */
export interface OllamaProviderConfig extends BaseProviderConfig {
  type: "ollama";
  host?: string;
}

/**
 * Bedrock provider configuration with flat structure
 */
export interface BedrockProviderConfig extends BaseProviderConfig {
  type: "bedrock";
  region?: string;
}

/**
 * OpenRouter provider configuration with flat structure
 */
export interface OpenRouterRoutingConfig {
  allowFallbacks?: boolean;
  order?: string[];
  requireParameters?: boolean;
}

export interface OpenRouterProviderConfig extends BaseProviderConfig {
  type: "openrouter";
  apiKey?: string;
  httpReferer?: string;
  xTitle?: string;
  provider?: OpenRouterRoutingConfig;
  fallbackModels?: string[];
  debugEchoUpstreamBody?: boolean;
}

/**
 * xAI (Grok) provider configuration using the OpenAI-compatible API at api.x.ai
 */
export interface XaiProviderConfig extends BaseProviderConfig {
  type: "xai";
  apiKey?: string;
}

/**
 * Cloudflare Workers AI provider configuration using the OpenAI-compatible API.
 */
export interface CloudflareProviderConfig extends BaseProviderConfig {
  type: "cloudflare";
  apiKey?: string;
  accountId?: string;
}

/**
 * Gemini provider configuration using Google's OpenAI-compatible API.
 */
export interface GeminiProviderConfig extends BaseProviderConfig {
  type: "gemini";
  apiKey?: string;
}

/**
 * Anthropic (Claude API) provider configuration
 */
export interface AnthropicProviderConfig extends BaseProviderConfig {
  type: "anthropic";
  apiKey?: string;
}

/**
 * Configuration for a single LLM provider (discriminated union)
 */
export type ProviderConfig =
  | OllamaProviderConfig
  | BedrockProviderConfig
  | OpenRouterProviderConfig
  | GeminiProviderConfig
  | XaiProviderConfig
  | CloudflareProviderConfig
  | AnthropicProviderConfig;

/**
 * Multi-provider configuration
 */
export interface ProvidersConfig {
  default: string;
  providers: ProviderConfig[];
}
