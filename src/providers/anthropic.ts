import Anthropic from "@anthropic-ai/sdk";
import { ProviderCapabilities } from "../interface.js";
import { BaseProvider, BaseProviderOptions } from "../internal/baseProvider.js";
import {
  ChatMessage,
  ChatRequest,
  ChatStreamEvent,
  ChatTool,
  ChatToolCall,
  ProviderError,
  ProviderAuthenticationError,
  ProviderRateLimitError,
  ProviderModelNotFoundError,
  ProviderContextLengthError,
  ProviderCapacityError,
} from "../types.js";
import { withRetry } from "../internal/withRetry.js";
import { createProviderRetryOptions } from "../internal/shared.js";

// Default budget and min output headroom when extended thinking is enabled.
const THINKING_BUDGET_TOKENS = 10000;
const THINKING_OUTPUT_HEADROOM = 1000;
const DEFAULT_MAX_TOKENS = 16384;

type AnthropicReplayBlock =
  | {
      type: "thinking";
      thinking: string;
      signature: string;
    }
  | {
      type: "redacted_thinking";
      data: string;
    };

type AnthropicReplayBlockInput = {
  type?: "thinking" | "redacted_thinking";
  thinking?: string;
  signature?: string;
  data?: string;
};

interface AnthropicStreamState {
  toolCalls: ChatToolCall[];
  currentToolCall: Partial<ChatToolCall> | null;
  currentToolInputJson: string;
  currentToolInputFromStart: Record<string, unknown> | null;
  // Per-block thinking accumulation (text + signature for replay)
  currentThinkingText: string;
  currentThinkingSignature: string;
  inThinkingBlock: boolean;
  // Accumulated thinking blocks for tool-call replay.
  thinkingBlocks: AnthropicReplayBlock[];
  stopReason: "end_turn" | "tool_use" | "max_tokens" | "stop_sequence";
  rawStopReason: string;
}

export class AnthropicProvider extends BaseProvider {
  readonly name = "anthropic";
  private client: Anthropic;

  constructor(options: BaseProviderOptions & { apiKey?: string }) {
    const apiKey = options.apiKey ?? process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
      throw new ProviderAuthenticationError(
        "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable or provide apiKey in config.",
      );
    }
    super(options);
    this.client = new Anthropic({ apiKey });
  }

  // fallow-ignore-next-line complexity
  async *streamChat(request: ChatRequest): AsyncIterable<ChatStreamEvent> {
    try {
      // Concatenate multiple system messages rather than silently dropping all but the first.
      const systemMessages = request.messages.filter(
        (m) => m.role === "system",
      );
      const systemContent =
        systemMessages.length > 0
          ? systemMessages.map((m) => m.content).join("\n\n")
          : undefined;

      const replayThinkingBlocks = request.requestReasoning === true;
      const messages = request.messages
        .filter((m) => m.role !== "system")
        .map((msg) =>
          this.chatMessageToAnthropicMessage(msg, { replayThinkingBlocks }),
        );

      const anthropicTools = request.tools?.map((tool) =>
        this.chatToolToAnthropicTool(tool),
      );

      const thinkingBudget = request.requestReasoning
        ? THINKING_BUDGET_TOKENS
        : undefined;
      // max_tokens must exceed thinking.budget_tokens; use 16k floor for normal requests.
      const maxTokens = thinkingBudget
        ? Math.max(
            thinkingBudget + THINKING_OUTPUT_HEADROOM,
            DEFAULT_MAX_TOKENS,
          )
        : DEFAULT_MAX_TOKENS;

      const createStream = () =>
        this.client.messages.create(
          {
            model: request.model || this.model,
            max_tokens: maxTokens,
            system: systemContent,
            messages: messages as Anthropic.MessageParam[],
            tools: anthropicTools as Anthropic.Tool[] | undefined,
            thinking: thinkingBudget
              ? { type: "enabled", budget_tokens: thinkingBudget }
              : undefined,
            stream: true,
          },
          { signal: request.signal },
        );

      const stream = (await withRetry(
        createStream,
        this.createRetryOptions(request),
      )) as AsyncIterable<Anthropic.MessageStreamEvent>;

      const state: AnthropicStreamState = {
        toolCalls: [],
        currentToolCall: null,
        currentToolInputJson: "",
        currentToolInputFromStart: null,
        currentThinkingText: "",
        currentThinkingSignature: "",
        inThinkingBlock: false,
        thinkingBlocks: [],
        stopReason: "end_turn",
        rawStopReason: "end_turn",
      };

      for await (const event of stream) {
        if (event.type === "content_block_start") {
          this.handleContentBlockStart(
            event as Anthropic.RawContentBlockStartEvent,
            state,
          );
        } else if (event.type === "content_block_delta") {
          const { text, thinking } = this.handleContentBlockDelta(
            event as Anthropic.RawContentBlockDeltaEvent,
            state,
          );
          if (text) {
            yield { type: "assistant_text", delta: text };
          }
          if (thinking) {
            yield { type: "thinking_delta", delta: thinking };
          }
        } else if (event.type === "content_block_stop") {
          this.handleContentBlockStop(state);
        } else if (event.type === "message_delta") {
          // stop_reason lives on message_delta, not message_stop
          const deltaEvent = event as Anthropic.RawMessageDeltaEvent;
          if (deltaEvent.delta.stop_reason) {
            state.rawStopReason = deltaEvent.delta.stop_reason;
            state.stopReason = this.mapStopReason(deltaEvent.delta.stop_reason);
          }
        }
      }

      if (state.toolCalls.length > 0) {
        // Attach accumulated thinking blocks as reasoningContent for replay in the next turn.
        const reasoningContent =
          state.thinkingBlocks.length > 0
            ? JSON.stringify(state.thinkingBlocks)
            : undefined;
        yield {
          type: "tool_calls",
          toolCalls: state.toolCalls,
          reasoningContent,
        };
      }

      yield {
        type: "terminal",
        stopReason: state.stopReason,
        rawProviderReason: state.rawStopReason,
      };
    } catch (error) {
      throw this.translateError(error);
    }
  }

  private handleContentBlockStart(
    event: Anthropic.RawContentBlockStartEvent,
    state: AnthropicStreamState,
  ): void {
    const block = event.content_block;
    if (block.type === "tool_use") {
      const toolBlock = block as Anthropic.ToolUseBlock;
      state.currentToolCall = {
        id: toolBlock.id,
        function: {
          name: toolBlock.name,
          arguments: {},
        },
      };
      state.currentToolInputJson = "";
      state.currentToolInputFromStart = this.asToolInputObject(toolBlock.input);
    } else if (block.type === "thinking") {
      state.inThinkingBlock = true;
      state.currentThinkingText = "";
      state.currentThinkingSignature = "";
    } else if (block.type === "redacted_thinking") {
      state.thinkingBlocks.push({
        type: "redacted_thinking",
        data: block.data,
      });
    }
  }

  // fallow-ignore-next-line complexity
  private handleContentBlockDelta(
    event: Anthropic.RawContentBlockDeltaEvent,
    state: AnthropicStreamState,
  ): { text?: string; thinking?: string } {
    const delta = event.delta;

    if (delta.type === "text_delta") {
      return { text: (delta as Anthropic.TextDelta).text || undefined };
    } else if (delta.type === "thinking_delta") {
      const text = (delta as Anthropic.ThinkingDelta).thinking || "";
      state.currentThinkingText += text;
      return { thinking: text || undefined };
    } else if (delta.type === "signature_delta") {
      state.currentThinkingSignature +=
        (delta as Anthropic.SignatureDelta).signature || "";
    } else if (delta.type === "input_json_delta") {
      state.currentToolInputJson +=
        (delta as Anthropic.InputJSONDelta).partial_json || "";
    }
    return {};
  }

  private handleContentBlockStop(state: AnthropicStreamState): void {
    if (state.inThinkingBlock) {
      // Finalize the thinking block — keep text+signature for replay.
      if (state.currentThinkingText || state.currentThinkingSignature) {
        state.thinkingBlocks.push({
          type: "thinking",
          thinking: state.currentThinkingText,
          signature: state.currentThinkingSignature,
        });
      }
      state.inThinkingBlock = false;
      state.currentThinkingText = "";
      state.currentThinkingSignature = "";
      return;
    }

    if (!state.currentToolCall?.function) {
      return;
    }

    state.currentToolCall.function.arguments =
      this.parseToolInputArguments(state);

    state.toolCalls.push(state.currentToolCall as ChatToolCall);
    state.currentToolCall = null;
    state.currentToolInputJson = "";
    state.currentToolInputFromStart = null;
  }

  private parseToolInputArguments(
    state: AnthropicStreamState,
  ): Record<string, unknown> {
    const inputJson = state.currentToolInputJson.trim();
    if (inputJson.length === 0) {
      return state.currentToolInputFromStart ?? {};
    }

    try {
      return JSON.parse(inputJson) as Record<string, unknown>;
    } catch {
      return {
        raw: state.currentToolInputJson,
      };
    }
  }

  private asToolInputObject(value: unknown): Record<string, unknown> | null {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      return null;
    }
    return value as Record<string, unknown>;
  }

  private mapStopReason(
    reason: string | null,
  ): "end_turn" | "tool_use" | "max_tokens" | "stop_sequence" {
    switch (reason) {
      case "end_turn":
        return "end_turn";
      case "tool_use":
        return "tool_use";
      case "max_tokens":
        return "max_tokens";
      case "stop_sequence":
        return "stop_sequence";
      default:
        return "end_turn";
    }
  }

  private chatMessageToAnthropicMessage(
    msg: ChatMessage,
    options: { replayThinkingBlocks?: boolean } = {
      replayThinkingBlocks: true,
    },
  ): Anthropic.MessageParam {
    const content: Anthropic.ContentBlockParam[] = [];

    if (options.replayThinkingBlocks !== false) {
      this.appendThinkingBlocks(msg, content);
    }
    this.appendTextBlock(msg, content);
    this.appendToolUseBlocks(msg, content);
    this.appendImageBlocks(msg, content);
    this.appendToolResultBlocks(msg, content);

    return {
      role: msg.role === "tool" ? "user" : (msg.role as "user" | "assistant"),
      content: content.length > 0 ? content : [{ type: "text", text: "." }],
    };
  }

  // Replay thinking blocks on assistant turns preceding tool_result messages.
  // Anthropic requires the prior thinking content to be re-sent in tool-call loops.
  private appendThinkingBlocks(
    msg: ChatMessage,
    content: Anthropic.ContentBlockParam[],
  ): void {
    if (
      !(
        msg.role === "assistant" &&
        msg.reasoningContent &&
        msg.toolCalls?.length
      )
    ) {
      return;
    }
    try {
      const blocks = JSON.parse(
        msg.reasoningContent,
      ) as AnthropicReplayBlockInput[];
      for (const block of blocks) {
        if (this.appendRedactedThinkingBlock(block, content)) continue;
        this.appendThinkingReplayBlock(block, content);
      }
    } catch {
      // Malformed reasoningContent — skip rather than crash.
    }
  }

  private appendRedactedThinkingBlock(
    block: AnthropicReplayBlockInput,
    content: Anthropic.ContentBlockParam[],
  ): boolean {
    if (block.type !== "redacted_thinking" || typeof block.data !== "string") {
      return false;
    }

    content.push({ type: "redacted_thinking", data: block.data });
    return true;
  }

  private appendThinkingReplayBlock(
    block: AnthropicReplayBlockInput,
    content: Anthropic.ContentBlockParam[],
  ): boolean {
    if (block.type !== undefined && block.type !== "thinking") {
      return false;
    }

    if (
      typeof block.thinking !== "string" ||
      typeof block.signature !== "string"
    ) {
      return false;
    }

    content.push({
      type: "thinking",
      thinking: block.thinking,
      signature: block.signature,
    });
    return true;
  }

  private appendTextBlock(
    msg: ChatMessage,
    content: Anthropic.ContentBlockParam[],
  ): void {
    if (msg.content && msg.role !== "tool") {
      content.push({ type: "text", text: msg.content });
    }
  }

  private appendToolUseBlocks(
    msg: ChatMessage,
    content: Anthropic.ContentBlockParam[],
  ): void {
    if (!msg.toolCalls?.length) return;
    for (const toolCall of msg.toolCalls) {
      if (!toolCall.id) {
        throw new ProviderError(
          `Anthropic requires a tool call id but tool call "${toolCall.function.name}" has none`,
        );
      }
      content.push({
        type: "tool_use",
        id: toolCall.id,
        name: toolCall.function.name,
        input: toolCall.function.arguments,
      });
    }
  }

  private appendImageBlocks(
    msg: ChatMessage,
    content: Anthropic.ContentBlockParam[],
  ): void {
    if (!msg.images?.length) return;
    for (const image of msg.images) {
      const { data, mediaType } = this.resolveImageData(image);
      content.push({
        type: "image",
        source: { type: "base64", media_type: mediaType, data },
      });
    }
  }

  private appendToolResultBlocks(
    msg: ChatMessage,
    content: Anthropic.ContentBlockParam[],
  ): void {
    if (msg.role !== "tool") return;
    if (msg.toolResults?.length) {
      for (const toolResult of msg.toolResults) {
        content.push({
          type: "tool_result",
          tool_use_id: toolResult.toolCallId,
          content: toolResult.content,
        });
      }
    } else if (msg.toolCallId) {
      content.push({
        type: "tool_result",
        tool_use_id: msg.toolCallId,
        content: msg.content,
      });
    }
  }

  private resolveImageData(image: Uint8Array | string): {
    data: string;
    mediaType: "image/jpeg" | "image/png" | "image/gif" | "image/webp";
  } {
    if (typeof image === "string") {
      if (image.startsWith("data:")) {
        const semicolonIndex = image.indexOf(";");
        const commaIndex = image.indexOf(",");
        if (semicolonIndex === -1 || commaIndex === -1) {
          throw new ProviderError(
            "Anthropic image data URLs must include a base64 payload",
          );
        }

        const mediaType = image.slice(5, semicolonIndex);
        const data = image.slice(commaIndex + 1);
        const supportedMediaTypes = new Set([
          "image/jpeg",
          "image/png",
          "image/gif",
          "image/webp",
        ]);

        if (supportedMediaTypes.has(mediaType)) {
          return {
            mediaType: mediaType as
              | "image/jpeg"
              | "image/png"
              | "image/gif"
              | "image/webp",
            data,
          };
        }

        throw new ProviderError(
          `Anthropic does not support image data URL media type "${mediaType || "unknown"}"`,
        );
      }
      return { data: image, mediaType: "image/png" };
    }

    // Sniff magic bytes for Uint8Array images
    const mediaType = this.sniffImageMediaType(image);
    return { data: Buffer.from(image).toString("base64"), mediaType };
  }

  private sniffImageMediaType(
    bytes: Uint8Array,
  ): "image/jpeg" | "image/png" | "image/gif" | "image/webp" {
    if (bytes[0] === 0xff && bytes[1] === 0xd8) return "image/jpeg";
    if (bytes[0] === 0x89 && bytes[1] === 0x50) return "image/png"; // \x89P
    if (bytes[0] === 0x47 && bytes[1] === 0x49) return "image/gif"; // GI
    if (bytes[8] === 0x57 && bytes[9] === 0x45) return "image/webp"; // RIFF....WEBP
    return "image/png";
  }

  private chatToolToAnthropicTool(chatTool: ChatTool): Anthropic.Tool {
    return {
      name: chatTool.function.name,
      description: chatTool.function.description,
      input_schema: chatTool.function
        .parameters as Anthropic.Tool["input_schema"],
    };
  }

  private static readonly RETRYABLE_STATUSES = new Set([
    429, 500, 502, 503, 504, 529,
  ]);

  private createRetryOptions(request: ChatRequest) {
    return createProviderRetryOptions({
      request,
      model: this.model,
      provider: this.name,
      retryConfig: this.retryConfig
        ? { ...this.retryConfig, baseDelayMs: 500 }
        : undefined,
      isRetryable: (error) => this.isRetryableError(error),
      onDiagnosticEvent: this.onDiagnosticEvent,
    });
  }

  // fallow-ignore-next-line complexity
  private isRetryableError(error: unknown): boolean {
    if (error instanceof ProviderCapacityError) return true;
    if (error instanceof Anthropic.APIError) {
      return AnthropicProvider.RETRYABLE_STATUSES.has(error.status);
    }
    const name = (error as any)?.name ?? "";
    const message = error instanceof Error ? error.message : String(error);
    return (
      name.includes("timeout") ||
      message.includes("rate limit") ||
      message.includes("throttle") ||
      message.includes("ECONNREFUSED") ||
      message.includes("ENOTFOUND")
    );
  }

  // fallow-ignore-next-line complexity
  private translateError(error: unknown): Error {
    if (error instanceof Anthropic.APIError) {
      const status = error.status;

      if (status === 401 || error.message.includes("authentication")) {
        return new ProviderAuthenticationError(
          `Anthropic authentication failed: ${error.message}`,
          error,
        );
      }

      if (status === 429) {
        const retryAfter = error.headers?.["retry-after"];
        const retryAfterSeconds = retryAfter
          ? parseInt(retryAfter, 10)
          : undefined;
        return new ProviderRateLimitError(
          `Anthropic rate limit exceeded: ${error.message}`,
          retryAfterSeconds,
          error,
        );
      }

      if (status === 529) {
        return new ProviderCapacityError(
          `Anthropic overloaded: ${error.message}`,
          error,
        );
      }

      if (status === 404 || error.message.includes("Model not found")) {
        return new ProviderModelNotFoundError(
          this.model,
          `Anthropic model not found: ${error.message}`,
          error,
        );
      }

      if (status === 400 && error.message.includes("context length")) {
        return new ProviderContextLengthError(
          `Anthropic context length exceeded: ${error.message}`,
          error,
        );
      }

      return new ProviderError(
        `Anthropic API error (${status}): ${error.message}`,
        error,
      );
    }

    if (error instanceof Error) {
      return new ProviderError(
        `Anthropic provider error: ${error.message}`,
        error,
      );
    }

    return new ProviderError(`Anthropic provider error: ${String(error)}`);
  }
}
