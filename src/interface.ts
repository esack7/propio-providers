import { ChatRequest, ChatStreamEvent } from "./types.js";

/**
 * Provider capability information for the currently configured model.
 */
export interface ProviderCapabilities {
  readonly contextWindowTokens: number;
  /**
   * Whether the provider accepts synthetic (caller-fabricated) assistant
   * tool-call history that was never produced by the model. Providers that
   * verify tool-call provenance (e.g. Gemini's thought signatures) reject
   * such history; callers should inline that content into a user message
   * instead. undefined/true = supported.
   */
  readonly supportsSyntheticToolCallHistory?: boolean;
}

/**
 * LLMProvider interface defining the contract for all LLM provider implementations
 */
export interface LLMProvider {
  /**
   * Provider identifier
   */
  readonly name: string;

  /**
   * Return capability info for the active model. Implementations may use
   * internal lookup tables; callers can override via per-model config.
   */
  getCapabilities(): ProviderCapabilities;

  /**
   * Streaming chat completion
   * @param request - The chat request with messages, model, and optional tools
   * @returns AsyncIterable yielding ChatStreamEvent objects for assistant text/tool calls/status updates
   */
  streamChat(request: ChatRequest): AsyncIterable<ChatStreamEvent>;
}
