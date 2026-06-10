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
