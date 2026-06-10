import {
  ChatMessage,
  ChatRequest,
  ChatStreamEvent,
  ChatToolCall,
  ProviderAuthenticationError,
  ProviderError,
} from "../types.js";
import type { ProviderDiagnosticListener } from "../diagnostics.js";
import {
  accumulateOpenAIStreamToolCall,
  applyOpenAIMessageCore,
  buildOpenAIChatCompletionRequestBody,
  buildOpenAIStreamToolCalls,
  createOpenAIMessageWithImages,
  createOpenAIToolCall,
  parseJsonMaybe,
  parseOpenAIStreamToolCallArguments,
  readSseDataLines,
  type OpenAIMessageContentPart,
} from "../internal/shared.js";
import { fetchOpenAiCompatibleStreamReader } from "../internal/openAiStream.js";
import {
  OpenAiCompatibleProvider,
  type OpenAiCompatibleProviderOptions,
} from "../internal/openAiCompatibleProvider.js";
import type { ProviderCapabilities } from "../interface.js";

const GEMINI_API_URL =
  "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions";
const GEMINI_THOUGHT_TAG_PATTERN = /<\/?thought(?:\s[^>]*)?>/gi;
const GEMINI_THOUGHT_BLOCK_PATTERN =
  /<thought(?:\s[^>]*)?>([\s\S]*?)<\/thought>/gi;
const GEMINI_PARTIAL_THOUGHT_TAG_PATTERN = /<\/?(?:th(?:ought)?)?$/i;

interface OpenAIMessage {
  role: "user" | "assistant" | "system" | "tool";
  content: string | OpenAIMessageContentPart[];
  tool_calls?: Array<{
    id: string;
    type: "function";
    function: { name: string; arguments: string };
    extra_content?: {
      google?: {
        thought_signature?: string;
      };
    };
  }>;
  tool_call_id?: string;
}

interface OpenAITool {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: object;
  };
}

interface GeminiToolCallDelta {
  index?: number;
  id?: string;
  function?: { name?: string; arguments?: string };
  extra_content?: {
    google?: {
      thought_signature?: string;
    };
  };
  thought_signature?: string;
  thoughtSignature?: string;
}

interface GeminiAccumulatedToolCall {
  id?: string;
  name: string;
  argsString: string;
  thoughtSignature?: string;
}

interface GeminiStreamChoice {
  delta?: {
    content?:
      | string
      | Array<{
          text?: string;
          type?: string;
          thought?: boolean;
        }>;
    reasoning_content?: string;
    reasoning?: string;
    thought?: string;
    thought_summary?: string;
    reasoning_details?: Array<{
      type?: string;
      text?: string;
      summary?: string;
    }>;
    extra_content?: {
      google?: {
        reasoning_content?: string;
        thought?: string;
        thought_summary?: string;
        thinking?: string;
      };
    };
    tool_calls?: GeminiToolCallDelta[];
    toolCalls?: GeminiToolCallDelta[];
  };
  finish_reason?: string;
}

/**
 * Gemini implementation of LLMProvider using Google's OpenAI-compatible API.
 */
export class GeminiProvider extends OpenAiCompatibleProvider {
  readonly name = "gemini";
  private readonly model: string;
  private readonly apiKey: string;
  private readonly retryConfig?: {
    maxRetries: number;
    consecutive529Limit: number;
  };
  private readonly onDiagnosticEvent?: ProviderDiagnosticListener;
  private streamContentBuffer = "";

  constructor(options: OpenAiCompatibleProviderOptions) {
    super();
    const apiKey =
      options.apiKey ??
      process.env.GEMINI_API_KEY ??
      process.env.GOOGLE_API_KEY ??
      "";
    if (!apiKey || apiKey.trim() === "") {
      throw new ProviderAuthenticationError(
        "Gemini API key is required. Set GEMINI_API_KEY or GOOGLE_API_KEY, or pass apiKey in options.",
      );
    }
    this.model = options.model;
    this.configureCapabilities(options.contextWindowTokens);
    this.apiKey = apiKey;
    this.retryConfig = options.retryConfig;
    this.onDiagnosticEvent = options.onDiagnosticEvent;
  }

  getCapabilities(): ProviderCapabilities {
    // Gemini validates assistant tool-call history against thought
    // signatures, so fabricated tool-call rounds are rejected upstream.
    return {
      ...super.getCapabilities(),
      supportsSyntheticToolCallHistory: false,
    };
  }

  protected chatMessageToOpenAIMessage(msg: ChatMessage): OpenAIMessage {
    const out = createOpenAIMessageWithImages<OpenAIMessage>(msg);

    applyOpenAIMessageCore(out, msg);
    if (msg.toolCalls && msg.toolCalls.length > 0) {
      out.tool_calls = msg.toolCalls.map((tc) =>
        createOpenAIToolCall(
          tc,
          tc.thoughtSignature
            ? {
                extra_content: {
                  google: {
                    thought_signature: tc.thoughtSignature,
                  },
                },
              }
            : undefined,
        ),
      );
    }

    return out;
  }

  protected translateError(
    error: unknown,
    response?: Response,
    _responseBody?: string,
  ): ProviderError {
    return this.translateStandardOpenAiError(error, response, {
      model: this.model,
      authenticationMessage: "Invalid Gemini API key",
      rateLimitMessage: "Gemini rate limit exceeded",
      serviceErrorMessage: "Gemini service error",
      connectionErrorMessage: "Failed to connect to Gemini API",
      requestFailedMessage: "Gemini request failed",
    });
  }

  private mapFinishReason(finishReason: string): string {
    if (finishReason === "MAX_TOKENS") return "max_tokens";
    if (finishReason === "STOP") return "end_turn";
    if (finishReason === "TOOL_CALLS") return "tool_use";
    if (
      finishReason === "SAFETY" ||
      finishReason === "RECITATION" ||
      finishReason === "OTHER"
    )
      return "error";
    return "end_turn";
  }

  private extractThoughtSignature(toolCall: {
    extra_content?: { google?: { thought_signature?: string } };
    thought_signature?: string;
    thoughtSignature?: string;
  }): string | undefined {
    return (
      toolCall.extra_content?.google?.thought_signature ??
      toolCall.thought_signature ??
      toolCall.thoughtSignature
    );
  }

  private accumulateGeminiThoughtSignature(
    toolCall: GeminiToolCallDelta,
    toolCallsByIndex: Map<number, GeminiAccumulatedToolCall>,
  ): void {
    const thoughtSignature = this.extractThoughtSignature(toolCall);
    const index = toolCall.index ?? 0;
    const accumulated = toolCallsByIndex.get(index);
    if (accumulated && thoughtSignature && !accumulated.thoughtSignature) {
      accumulated.thoughtSignature = thoughtSignature;
    }
  }

  private processGeminiToolCallsDelta(
    toolCallsDelta: GeminiToolCallDelta[] | undefined,
    toolCallsByIndex: Map<number, GeminiAccumulatedToolCall>,
  ): void {
    if (!toolCallsDelta || !Array.isArray(toolCallsDelta)) {
      return;
    }

    for (const toolCall of toolCallsDelta) {
      accumulateOpenAIStreamToolCall(toolCall, toolCallsByIndex, () => ({
        name: "",
        argsString: "",
      }));
      this.normalizeRepeatedGeminiToolName(toolCall, toolCallsByIndex);
      this.accumulateGeminiThoughtSignature(toolCall, toolCallsByIndex);
    }
  }

  private normalizeRepeatedGeminiToolName(
    toolCall: GeminiToolCallDelta,
    toolCallsByIndex: Map<number, GeminiAccumulatedToolCall>,
  ): void {
    const name = toolCall.function?.name;
    if (!name) {
      return;
    }

    const accumulated = toolCallsByIndex.get(toolCall.index ?? 0);
    if (accumulated?.name === `${name}${name}`) {
      accumulated.name = name;
    }
  }

  private buildGeminiToolCallsEvent(
    toolCallsByIndex: Map<number, GeminiAccumulatedToolCall>,
  ): ChatStreamEvent | null {
    if (toolCallsByIndex.size === 0) {
      return null;
    }

    return {
      type: "tool_calls",
      toolCalls: buildOpenAIStreamToolCalls(toolCallsByIndex, (acc) => ({
        id: acc.id,
        thoughtSignature: acc.thoughtSignature,
        function: {
          name: acc.name || "",
          arguments: parseOpenAIStreamToolCallArguments(acc.argsString),
        },
      })),
    };
  }

  private extractGeminiThinkingText(
    delta: NonNullable<GeminiStreamChoice["delta"]>,
  ): string {
    const directThinking = [
      delta.reasoning_content,
      delta.reasoning,
      delta.thought,
      delta.thought_summary,
      delta.extra_content?.google?.reasoning_content,
      delta.extra_content?.google?.thought,
      delta.extra_content?.google?.thought_summary,
      delta.extra_content?.google?.thinking,
    ]
      .filter((value): value is string => typeof value === "string")
      .join("");

    const reasoningDetails =
      delta.reasoning_details
        ?.map((detail) => {
          if (typeof detail.text === "string") {
            return detail.text;
          }
          if (typeof detail.summary === "string") {
            return detail.summary;
          }
          return "";
        })
        .join("") ?? "";

    const thoughtParts = Array.isArray(delta.content)
      ? delta.content
          .filter((part) => part.thought === true)
          .map((part) => part.text ?? "")
          .join("")
      : "";

    return this.cleanGeminiThinkingText(
      `${directThinking}${reasoningDetails}${thoughtParts}`,
    );
  }

  private cleanGeminiThinkingText(text: string): string {
    return text.replace(GEMINI_THOUGHT_TAG_PATTERN, "");
  }

  private resetGeminiStreamState(): void {
    this.streamContentBuffer = "";
  }

  private processGeminiStreamingContent(chunk: string): {
    thinking: string;
    assistant: string;
  } {
    this.streamContentBuffer += chunk;
    const buffer = this.streamContentBuffer;

    let thinking = "";
    let assistant = "";
    let consumed = 0;

    GEMINI_THOUGHT_BLOCK_PATTERN.lastIndex = 0;
    for (const match of buffer.matchAll(GEMINI_THOUGHT_BLOCK_PATTERN)) {
      if (match.index === undefined) {
        continue;
      }

      assistant += buffer.slice(consumed, match.index);
      thinking += match[1] ?? "";
      consumed = match.index + match[0].length;
    }

    const tail = buffer.slice(consumed);
    const openThoughtMatch = tail.match(/<thought(?:\s[^>]*)?>/i);
    if (openThoughtMatch?.index !== undefined) {
      assistant += tail.slice(0, openThoughtMatch.index);
      this.streamContentBuffer = tail.slice(openThoughtMatch.index);
    } else {
      const partialTagMatch = tail.match(GEMINI_PARTIAL_THOUGHT_TAG_PATTERN);
      if (partialTagMatch?.index !== undefined) {
        assistant += tail.slice(0, partialTagMatch.index);
        this.streamContentBuffer = tail.slice(partialTagMatch.index);
      } else {
        assistant += tail;
        this.streamContentBuffer = "";
      }
    }

    return {
      thinking: this.cleanGeminiThinkingText(thinking),
      assistant: this.cleanGeminiThinkingText(assistant),
    };
  }

  private flushGeminiStreamContentBuffer(): {
    thinking: string;
    assistant: string;
  } {
    if (this.streamContentBuffer.length === 0) {
      return { thinking: "", assistant: "" };
    }

    const tail = this.streamContentBuffer;
    this.streamContentBuffer = "";
    const cleaned = this.cleanGeminiThinkingText(tail);
    if (/<thought/i.test(tail)) {
      return { thinking: cleaned, assistant: "" };
    }

    return { thinking: "", assistant: cleaned };
  }

  private extractGeminiAssistantText(
    delta: NonNullable<GeminiStreamChoice["delta"]>,
  ): string {
    if (typeof delta.content === "string") {
      return "";
    }

    if (!Array.isArray(delta.content)) {
      return "";
    }

    return this.cleanGeminiThinkingText(
      delta.content
        .filter((part) => part.thought !== true)
        .map((part) => part.text ?? "")
        .join(""),
    );
  }

  private extractGeminiContentText(
    delta: NonNullable<GeminiStreamChoice["delta"]>,
  ): { thinking: string; assistant: string } {
    if (typeof delta.content === "string") {
      return this.processGeminiStreamingContent(delta.content);
    }

    return {
      thinking: "",
      assistant: this.extractGeminiAssistantText(delta),
    };
  }

  private parseGeminiStreamChunk(
    data: string,
    toolCallsByIndex: Map<number, GeminiAccumulatedToolCall>,
  ): { events: ChatStreamEvent[]; stopReason?: string; done: boolean } {
    if (data === "[DONE]") {
      return { events: [], done: true };
    }

    const choice = parseJsonMaybe<{ choices?: GeminiStreamChoice[] }>(data)
      ?.choices?.[0];
    if (!choice?.delta) {
      return { events: [], done: false };
    }

    const events: ChatStreamEvent[] = [];
    const contentText = this.extractGeminiContentText(choice.delta);
    const thinkingText =
      this.extractGeminiThinkingText(choice.delta) + contentText.thinking;
    if (thinkingText.length > 0) {
      events.push({ type: "thinking_delta", delta: thinkingText });
    }

    const assistantText = contentText.assistant;
    if (assistantText.length > 0) {
      events.push({ type: "assistant_text", delta: assistantText });
    }
    this.processGeminiToolCallsDelta(
      choice.delta.tool_calls ?? choice.delta.toolCalls,
      toolCallsByIndex,
    );

    return {
      events,
      stopReason: choice.finish_reason
        ? this.mapFinishReason(choice.finish_reason)
        : undefined,
      done: false,
    };
  }

  private async createGeminiStreamReader(
    request: ChatRequest,
  ): Promise<ReadableStreamDefaultReader<Uint8Array>> {
    const body = this.buildGeminiRequestBody(request);
    return fetchOpenAiCompatibleStreamReader({
      body,
      signal: request.signal,
      fetchStream: (streamBody, signal) =>
        this.fetchGeminiStream(streamBody, signal),
      retryOptions: this.buildRetryOptions(
        request,
        this.model,
        this.retryConfig,
        this.onDiagnosticEvent,
      ),
      translateError: (error) => this.translateError(error),
    });
  }

  private buildGeminiRequestBody(
    request: ChatRequest,
  ): Record<string, unknown> {
    const effectiveModel = request.model || this.model;
    return buildOpenAIChatCompletionRequestBody({
      request,
      model: effectiveModel,
      mapMessage: (msg) => this.chatMessageToOpenAIMessage(msg),
      mapTool: (tool) => this.chatToolToOpenAITool(tool),
      extra: (body) => {
        if (request.requestReasoning) {
          body.extra_body = {
            google: {
              thinking_config: {
                include_thoughts: true,
              },
            },
          };
        }
      },
    });
  }

  private async fetchGeminiStream(
    body: Record<string, unknown>,
    signal: AbortSignal | undefined,
  ): Promise<Response> {
    const res = await fetch(GEMINI_API_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
      signal,
    });
    if (!res.ok) {
      let errorBody = "";
      try {
        errorBody = await res.text();
      } catch {
        /* ignore read failures */
      }
      throw this.translateError(
        new Error(errorBody || `HTTP ${res.status}`),
        res,
      );
    }
    return res;
  }

  async *streamChat(request: ChatRequest): AsyncIterable<ChatStreamEvent> {
    try {
      this.resetGeminiStreamState();
      const reader = await this.createGeminiStreamReader(request);
      const toolCallsByIndex = new Map<number, GeminiAccumulatedToolCall>();
      let stopReason: any = "end_turn";

      for await (const data of readSseDataLines(reader)) {
        const result = this.parseGeminiStreamChunk(data, toolCallsByIndex);
        if (result.stopReason) {
          stopReason = result.stopReason;
        }
        yield* result.events;
        if (result.done) {
          break;
        }
      }

      const flushedContent = this.flushGeminiStreamContentBuffer();
      if (flushedContent.thinking.length > 0) {
        yield {
          type: "thinking_delta",
          delta: flushedContent.thinking,
        };
      }
      if (flushedContent.assistant.length > 0) {
        yield {
          type: "assistant_text",
          delta: flushedContent.assistant,
        };
      }

      const toolCallsEvent = this.buildGeminiToolCallsEvent(toolCallsByIndex);
      if (toolCallsEvent) {
        yield toolCallsEvent;
      }

      yield { type: "terminal", stopReason };
    } catch (error) {
      if (error instanceof ProviderError) throw error;
      throw this.translateError(error);
    }
  }
}
