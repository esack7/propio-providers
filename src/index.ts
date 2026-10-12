// Provider factory
export { createProvider, extractModelFromConfig } from "./factory.js";

// Core provider contract
export type { LLMProvider, ProviderCapabilities } from "./interface.js";

// Wire types and error classes
export {
  ProviderError,
  ProviderAuthenticationError,
  ProviderRateLimitError,
  ProviderCapacityError,
  ProviderModelNotFoundError,
  ProviderContextLengthError,
  ProviderInvalidRequestError,
} from "./types.js";
export type {
  ToolResult,
  ChatMessage,
  ChatToolCall,
  ChatTool,
  ChatRequest,
  ChatResponse,
  ChatChunk,
  ChatStreamEvent,
  StopReason,
  ProviderReasoningSummarySource,
  AssistantTextStreamEvent,
  ThinkingDeltaStreamEvent,
  ToolCallsStreamEvent,
  StatusStreamEvent,
  ReasoningSummaryStreamEvent,
  StreamTerminalEvent,
} from "./types.js";

// Provider configuration types
export type {
  Model,
  BaseProviderConfig,
  OllamaProviderConfig,
  BedrockProviderConfig,
  OpenRouterRoutingConfig,
  OpenRouterProviderConfig,
  XaiProviderConfig,
  CloudflareProviderConfig,
  GeminiProviderConfig,
  AnthropicProviderConfig,
  OpenAiProviderConfig,
  MetaProviderConfig,
  ProviderConfig,
  ProvidersConfig,
} from "./config.js";

// Config validation (pure)
export {
  validateProvidersConfig,
  resolveProvider,
  resolveModelKey,
  getDefaultProviderModelSelection,
  updateDefaultProviderModelSelection,
} from "./configValidation.js";
export type { ProviderModelSelection } from "./configValidation.js";

// Config file helpers (explicit paths)
export {
  loadProvidersConfig,
  loadProvidersConfigAsync,
  writeProvidersConfig,
  updateDefaultProviderModelSelectionInFile,
} from "./configFile.js";
export type { LoadProvidersConfigOptions } from "./configFile.js";

// Diagnostics
export type {
  ProviderDiagnosticEvent,
  ProviderRetryDiagnosticEvent,
  ProviderDiagnosticListener,
  ProviderRetryConfig,
} from "./diagnostics.js";

// Explicit, headless request tracing
export { withProviderTracing } from "./trace.js";
export type {
  ProviderRequestPurpose,
  ProviderReportedCost,
  ProviderTokenUsage,
  ProviderTraceContext,
  ProviderTraceEvent,
  ProviderTraceObserver,
  ProviderUsageAvailability,
} from "./trace.js";

export {
  OpenAiSubscriptionProvider,
  ChatGptPlanError,
} from "./providers/openaiSubscription.js";
export type { OpenAiSubscriptionOptions } from "./providers/openaiSubscription.js";
export type { OpenAiAuthConfig } from "./config.js";
export type { ProviderRuntimeOptions } from "./factory.js";
