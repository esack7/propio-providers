/**
 * Tool result representation
 */
export interface ToolResult {
  toolCallId: string; // Provider-specific tool call ID
  toolName: string; // Name of the tool that was called
  content: string;
}

/**
 * Provider-agnostic message type
 */
export interface ChatMessage {
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  /**
   * Provider-specific reasoning state that some thinking models require to be
   * replayed during tool-call loops. It is not rendered to users.
   */
  reasoningContent?: string;
  toolCalls?: ChatToolCall[];
  toolCallId?: string; // For tool role messages: which tool call this is a result for (deprecated, use toolResults)
  toolResults?: ToolResult[]; // For batched tool results
  images?: (Uint8Array | string)[];
}

/**
 * Tool call representation
 */
export interface ChatToolCall {
  id?: string; // Provider-specific tool call ID (e.g., Bedrock toolUseId)
  /**
   * Gemini thought signature required for function-call rounds on Gemini 3.
   * Other providers ignore this field.
   */
  thoughtSignature?: string;
  function: {
    name: string;
    arguments: Record<string, any>;
  };
}

/**
 * Tool definition
 */
export interface ChatTool {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: {
      type: string;
      properties?: Record<string, any>;
      required?: string[];
      [key: string]: any;
    };
  };
}

/**
 * Chat request with all information needed for LLM request
 */
export interface ChatRequest {
  messages: ChatMessage[];
  model: string;
  tools?: ChatTool[];
  stream?: boolean;
  signal?: AbortSignal;
  iteration?: number;
  /**
   * Ask providers that support exposed reasoning to return live thinking
   * tokens. Providers that do not support this should ignore it.
   */
  requestReasoning?: boolean;
}

/**
 * Chat response from provider
 */
export interface ChatResponse {
  message: ChatMessage;
  stopReason: "end_turn" | "tool_use" | "max_tokens" | "stop_sequence";
}

export type ProviderReasoningSummarySource = "provider" | "agent";

export interface AssistantTextStreamEvent {
  type: "assistant_text";
  delta: string;
}

export interface ThinkingDeltaStreamEvent {
  type: "thinking_delta";
  delta: string;
}

export interface ToolCallsStreamEvent {
  type: "tool_calls";
  toolCalls: ChatToolCall[];
  reasoningContent?: string;
}

export interface StatusStreamEvent {
  type: "status";
  status: string;
  phase?: string;
}

export interface ReasoningSummaryStreamEvent {
  type: "reasoning_summary";
  summary: string;
  source: ProviderReasoningSummarySource;
}

/**
 * Normalized terminal event emitted when stream ends.
 * Maps provider-specific stop reasons to a canonical set.
 */
export type StopReason =
  | "end_turn"
  | "tool_use"
  | "max_tokens"
  | "stop_sequence"
  | "error";

export interface StreamTerminalEvent {
  type: "terminal";
  stopReason: StopReason;
  rawProviderReason?: string; // For diagnostics: provider's original reason string
}

/**
 * Backward-compatible streaming chunk shape.
 * New implementations should emit ChatStreamEvent variants with a `type`.
 */
export interface ChatChunk {
  delta: string;
  toolCalls?: ChatToolCall[];
}

/**
 * Streaming event model used across providers/agent runtime.
 */
export type ChatStreamEvent =
  | AssistantTextStreamEvent
  | ThinkingDeltaStreamEvent
  | ToolCallsStreamEvent
  | StatusStreamEvent
  | ReasoningSummaryStreamEvent
  | StreamTerminalEvent
  | ChatChunk;

/**
 * Base provider error class
 */
export class ProviderError extends Error {
  public originalError?: Error;

  constructor(message: string, originalError?: Error) {
    super(message);
    this.name = "ProviderError";
    this.originalError = originalError;
  }
}

/**
 * Authentication error
 */
export class ProviderAuthenticationError extends ProviderError {
  constructor(message: string, originalError?: Error) {
    super(message, originalError);
    this.name = "ProviderAuthenticationError";
  }
}

/**
 * Rate limit error with optional retry info
 */
export class ProviderRateLimitError extends ProviderError {
  public retryAfterSeconds?: number;

  constructor(
    message: string,
    retryAfterSeconds?: number,
    originalError?: Error,
  ) {
    super(message, originalError);
    this.name = "ProviderRateLimitError";
    this.retryAfterSeconds = retryAfterSeconds;
  }
}

/**
 * Capacity error for upstream resource exhaustion (e.g., OpenRouter 529)
 */
export class ProviderCapacityError extends ProviderError {
  constructor(message: string, originalError?: Error) {
    super(message, originalError);
    this.name = "ProviderCapacityError";
  }
}

/**
 * Model not found error
 */
export class ProviderModelNotFoundError extends ProviderError {
  public modelName: string;

  constructor(modelName: string, message: string, originalError?: Error) {
    super(message, originalError);
    this.name = "ProviderModelNotFoundError";
    this.modelName = modelName;
  }
}

/**
 * Context length exceeded error. Thrown when the prompt exceeds the model's
 * context window. The agent loop uses this signal to rebuild the prompt at
 * a tighter retry level instead of surfacing the error to the user.
 */
export class ProviderContextLengthError extends ProviderError {
  constructor(message: string, originalError?: Error) {
    super(message, originalError);
    this.name = "ProviderContextLengthError";
  }
}
