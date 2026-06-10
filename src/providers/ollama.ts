import { Ollama, Message, Tool, ToolCall } from "ollama";
import { LLMProvider, ProviderCapabilities } from "../interface.js";
import { createProviderCapabilities } from "../internal/capabilities.js";
import {
  ChatMessage,
  ChatRequest,
  ChatStreamEvent,
  ChatTool,
  ChatToolCall,
  ProviderError,
  ProviderAuthenticationError,
  ProviderModelNotFoundError,
  ProviderContextLengthError,
  ProviderCapacityError,
} from "../types.js";
import type { ProviderDiagnosticListener } from "../diagnostics.js";
import { withRetry } from "../internal/withRetry.js";
import {
  createProviderRetryOptions,
  expandToolResultMessages,
} from "../internal/shared.js";

interface OllamaStreamChunk {
  message: {
    content?: string;
    thinking?: string;
    tool_calls?: ToolCall[];
  };
  done_reason?: string;
}

interface OllamaStreamState {
  lastToolCalls?: ChatToolCall[];
  stopReason: string;
}

interface OllamaErrorPattern {
  readonly test: (errorMessage: string, lower: string) => boolean;
  readonly make: (
    errorMessage: string,
    originalError: Error | undefined,
    model: string,
  ) => ProviderError;
}

const OLLAMA_ERROR_PATTERNS: readonly OllamaErrorPattern[] = [
  {
    test: (_errorMessage, lower) =>
      lower.includes("context length") ||
      lower.includes("too many tokens") ||
      lower.includes("input is too long"),
    make: (errorMessage, originalError) =>
      new ProviderContextLengthError(
        `Context length exceeded: ${errorMessage}`,
        originalError,
      ),
  },
  {
    test: (errorMessage) =>
      errorMessage.includes("ECONNREFUSED") ||
      errorMessage.includes("ECONNRESET") ||
      errorMessage.includes("connect"),
    make: (errorMessage, originalError) =>
      new ProviderAuthenticationError(
        `Failed to connect to Ollama: ${errorMessage}`,
        originalError,
      ),
  },
  {
    test: (_errorMessage, lower) =>
      lower.includes("model not found") || lower.includes("pull"),
    make: (errorMessage, originalError, model) =>
      new ProviderModelNotFoundError(
        model,
        `Model ${model} not found: ${errorMessage}`,
        originalError,
      ),
  },
];

/**
 * Ollama implementation of LLMProvider
 */
export class OllamaProvider implements LLMProvider {
  readonly name = "ollama";
  private ollama: Ollama;
  private model: string;
  private capabilities: ProviderCapabilities;
  private retryConfig?: { maxRetries: number; consecutive529Limit: number };
  private onDiagnosticEvent?: ProviderDiagnosticListener;

  constructor(options: {
    model: string;
    contextWindowTokens: number;
    host?: string;
    retryConfig?: { maxRetries: number; consecutive529Limit: number };
    onDiagnosticEvent?: ProviderDiagnosticListener;
  }) {
    const isSandbox = process.env.IS_SANDBOX === "true";

    // Priority: OLLAMA_HOST (no conversion) > options.host (with conversion) > default (mode-specific)
    const host =
      process.env.OLLAMA_HOST || this.resolveHost(options.host, isSandbox);

    this.ollama = new Ollama({ host });
    this.model = options.model;
    this.capabilities = createProviderCapabilities(options.contextWindowTokens);
    this.retryConfig = options.retryConfig;
    this.onDiagnosticEvent = options.onDiagnosticEvent;
  }

  getCapabilities(): ProviderCapabilities {
    return this.capabilities;
  }

  /**
   * Resolve host with smart localhost→host.docker.internal conversion for sandbox mode
   */
  private resolveHost(
    configuredHost: string | undefined,
    isSandbox: boolean,
  ): string {
    // If no host configured, use mode-specific default
    if (!configuredHost) {
      return isSandbox
        ? "http://host.docker.internal:11434"
        : "http://localhost:11434";
    }

    // If host contains localhost and we're in sandbox, convert to host.docker.internal
    if (isSandbox && configuredHost.includes("localhost")) {
      return configuredHost.replace("localhost", "host.docker.internal");
    }

    // Otherwise use as-is
    return configuredHost;
  }

  private isRetryableError(err: unknown): boolean {
    if (err instanceof ProviderContextLengthError) return false;
    if (err instanceof ProviderModelNotFoundError) return false;
    return err instanceof ProviderError;
  }

  private mapOllamaDoneReason(doneReason: string): string {
    const reasonMap: Record<string, string> = {
      length: "max_tokens",
      stop: "end_turn",
    };
    return reasonMap[doneReason] ?? "end_turn";
  }

  private createOllamaMessages(request: ChatRequest): Message[] {
    return this.expandToolResults(request.messages).map((msg) =>
      this.chatMessageToOllamaMessage(msg),
    );
  }

  private createOllamaTools(request: ChatRequest): Tool[] | undefined {
    return request.tools?.map((tool) => this.chatToolToOllamaTool(tool));
  }

  private async createOllamaStream(request: ChatRequest) {
    const messages = this.createOllamaMessages(request);
    const tools = this.createOllamaTools(request);
    return await withRetry(
      () =>
        this.ollama.chat({
          model: request.model || this.model,
          messages,
          stream: true,
          ...(tools && { tools }),
        }),
      createProviderRetryOptions({
        request,
        model: this.model,
        provider: this.name,
        retryConfig: this.retryConfig,
        isRetryable: (err) => this.isRetryableError(err),
        onDiagnosticEvent: this.onDiagnosticEvent,
      }),
    );
  }

  private processOllamaChunk(
    chunk: OllamaStreamChunk,
    request: ChatRequest,
    state: OllamaStreamState,
  ): ChatStreamEvent[] {
    if (request.signal?.aborted) {
      throw new ProviderError("Request cancelled");
    }

    const events: ChatStreamEvent[] = [];
    if (chunk.message.thinking) {
      events.push({ type: "thinking_delta", delta: chunk.message.thinking });
    }
    if (chunk.message.content) {
      events.push({ type: "assistant_text", delta: chunk.message.content });
    }
    if (chunk.message.tool_calls) {
      state.lastToolCalls = chunk.message.tool_calls.map((toolCall) =>
        this.ollamaToolCallToChatToolCall(toolCall),
      );
    }
    if (chunk.done_reason) {
      state.stopReason = this.mapOllamaDoneReason(chunk.done_reason);
    }
    return events;
  }

  private buildOllamaToolCallsEvent(
    toolCalls: ChatToolCall[] | undefined,
  ): ChatStreamEvent | null {
    return toolCalls && toolCalls.length > 0
      ? {
          type: "tool_calls",
          toolCalls,
        }
      : null;
  }

  async *streamChat(request: ChatRequest): AsyncIterable<ChatStreamEvent> {
    try {
      if (request.signal?.aborted) {
        throw new ProviderError("Request cancelled");
      }

      const response = await this.createOllamaStream(request);
      const state: OllamaStreamState = { stopReason: "end_turn" };

      for await (const chunk of response) {
        yield* this.processOllamaChunk(
          chunk as OllamaStreamChunk,
          request,
          state,
        );
      }

      const toolCallsEvent = this.buildOllamaToolCallsEvent(
        state.lastToolCalls,
      );
      if (toolCallsEvent) {
        yield toolCallsEvent;
      }

      yield { type: "terminal", stopReason: state.stopReason as any };
    } catch (error) {
      throw this.translateError(error);
    }
  }

  /**
   * Expand batched tool results into separate messages for Ollama.
   * Ollama expects each tool result as a separate message, not batched like Bedrock.
   */
  private expandToolResults(messages: ChatMessage[]): ChatMessage[] {
    return expandToolResultMessages(
      messages,
      (toolResult) => toolResult.toolName,
    );
  }

  /**
   * Translate ChatMessage to Ollama Message format
   */
  private chatMessageToOllamaMessage(msg: ChatMessage): Message {
    const ollamaMsg: Message = {
      role: msg.role,
      content: msg.content,
    };

    if (msg.toolCalls) {
      ollamaMsg.tool_calls = msg.toolCalls.map((toolCall) => ({
        function: {
          name: toolCall.function.name,
          arguments: toolCall.function.arguments,
        },
      }));
    }

    // For tool role messages, Ollama expects tool_name field
    if (msg.role === "tool" && msg.toolCallId) {
      ollamaMsg.tool_name = msg.toolCallId;
    }

    if (msg.images) {
      // Handle images as mixed type array - needs casting for type compatibility
      ollamaMsg.images = msg.images as any;
    }

    return ollamaMsg;
  }

  /**
   * Translate Ollama Message to ChatMessage
   */
  private ollamaMessageToChatMessage(msg: Message): ChatMessage {
    const chatMsg: ChatMessage = {
      role: msg.role as "user" | "assistant" | "system" | "tool",
      content: msg.content,
    };

    if (msg.tool_calls && msg.tool_calls.length > 0) {
      chatMsg.toolCalls = msg.tool_calls.map((toolCall) =>
        this.ollamaToolCallToChatToolCall(toolCall),
      );
    }

    if (msg.images) {
      // Handle mixed array of strings and Uint8Array
      chatMsg.images = msg.images as (Uint8Array | string)[];
    }

    return chatMsg;
  }

  /**
   * Translate Ollama ToolCall to ChatToolCall
   */
  private ollamaToolCallToChatToolCall(toolCall: ToolCall): ChatToolCall {
    return {
      function: {
        name: toolCall.function.name,
        arguments: toolCall.function.arguments,
      },
    };
  }

  /**
   * Translate ChatTool to Ollama Tool format
   */
  private chatToolToOllamaTool(chatTool: ChatTool): Tool {
    return {
      type: chatTool.type,
      function: {
        name: chatTool.function.name,
        description: chatTool.function.description,
        parameters: chatTool.function.parameters,
      },
    };
  }

  /**
   * Determine stop reason from Ollama response
   */
  private getStopReason(
    message: Message,
  ): "end_turn" | "tool_use" | "max_tokens" | "stop_sequence" {
    if (message.tool_calls && message.tool_calls.length > 0) {
      return "tool_use";
    }
    return "end_turn";
  }

  /**
   * Translate errors to ProviderError types
   */
  private translateError(error: any): ProviderError {
    const originalError = error instanceof Error ? error : undefined;
    const errorMessage = originalError?.message ?? String(error);
    const lower = errorMessage.toLowerCase();

    for (const pattern of OLLAMA_ERROR_PATTERNS) {
      if (pattern.test(errorMessage, lower)) {
        return pattern.make(errorMessage, originalError, this.model);
      }
    }

    return new ProviderError(errorMessage, originalError);
  }
}
