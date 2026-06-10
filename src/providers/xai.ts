import {
  ChatMessage,
  ChatRequest,
  ChatStreamEvent,
  ChatTool,
  ChatToolCall,
  ProviderError,
  ProviderAuthenticationError,
  type StopReason,
} from "../types.js";
import type { ProviderDiagnosticListener } from "../diagnostics.js";
import { withRetry } from "../internal/withRetry.js";
import {
  buildOpenAIChatCompletionRequestBody,
  expandToolResultMessages,
  parseJsonMaybe,
  parseOpenAIStreamToolCallArguments,
  readSseDataLines,
  type OpenAIStreamToolCallAccumulator,
} from "../internal/shared.js";
import { consumeOpenAiChatCompletionsStream } from "../internal/openAiStream.js";
import {
  OpenAiCompatibleProvider,
  type OpenAiCompatibleProviderOptions,
} from "../internal/openAiCompatibleProvider.js";

const XAI_CHAT_COMPLETIONS_API_URLS = [
  "https://api.x.ai/v1/chat/completions",
  "https://us-east-1.api.x.ai/v1/chat/completions",
  "https://eu-west-1.api.x.ai/v1/chat/completions",
] as const;

const XAI_RESPONSES_API_URLS = [
  "https://api.x.ai/v1/responses",
  "https://us-east-1.api.x.ai/v1/responses",
  "https://eu-west-1.api.x.ai/v1/responses",
] as const;

/**
 * xAI (Grok) implementation of LLMProvider using the OpenAI-compatible API at api.x.ai.
 */
export class XaiProvider extends OpenAiCompatibleProvider {
  readonly name = "xai";
  private readonly model: string;
  private readonly apiKey: string;
  private readonly retryConfig?: {
    maxRetries: number;
    consecutive529Limit: number;
  };
  private readonly onDiagnosticEvent?: ProviderDiagnosticListener;

  constructor(options: OpenAiCompatibleProviderOptions) {
    super();
    const apiKey = options.apiKey ?? process.env.XAI_API_KEY ?? "";
    if (!apiKey || apiKey.trim() === "") {
      throw new ProviderAuthenticationError(
        "xAI API key is required. Set XAI_API_KEY or pass apiKey in options.",
      );
    }
    this.retryConfig = options.retryConfig;
    this.onDiagnosticEvent = options.onDiagnosticEvent;
    this.model = options.model;
    this.configureCapabilities(options.contextWindowTokens);
    this.apiKey = apiKey;
  }

  private shouldRetryEndpoint(status?: number): boolean {
    return status !== undefined && status >= 500 && status < 600;
  }

  private isAbortError(translated: ProviderError): boolean {
    return (
      translated.message === "Request cancelled" ||
      translated.originalError?.name === "AbortError"
    );
  }

  private shouldContinueToNextEndpoint(
    error: unknown,
    translated: ProviderError,
  ): boolean {
    return !(error instanceof ProviderError) && !this.isAbortError(translated);
  }

  private async createEndpointResponseError(response: Response): Promise<{
    translated: ProviderError;
    retryable: boolean;
  }> {
    let errorBody = "";
    try {
      errorBody = await response.text();
    } catch {
      // ignore read failures
    }

    return {
      translated: this.translateError(
        new Error(errorBody || `HTTP ${response.status}`),
        response,
      ),
      retryable: this.shouldRetryEndpoint(response.status),
    };
  }

  private getContinuationError(error: unknown): ProviderError {
    const translated =
      error instanceof ProviderError ? error : this.translateError(error);
    if (!this.shouldContinueToNextEndpoint(error, translated)) {
      throw translated;
    }
    return translated;
  }

  private async createPostResponse(
    apiUrls: readonly string[],
    body: Record<string, unknown>,
    signal?: AbortSignal,
  ): Promise<Response> {
    let lastError: ProviderError | null = null;

    for (const apiUrl of apiUrls) {
      try {
        const response = await fetch(apiUrl, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${this.apiKey}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify(body),
          signal,
        });

        if (response.ok) {
          return response;
        }

        const { translated, retryable } =
          await this.createEndpointResponseError(response);
        if (!retryable) {
          throw translated;
        }
        lastError = translated;
      } catch (error) {
        lastError = this.getContinuationError(error);
      }
    }

    throw lastError ?? new ProviderError("xAI request failed");
  }

  async *streamChat(request: ChatRequest): AsyncIterable<ChatStreamEvent> {
    try {
      if (request.requestReasoning) {
        yield* this.streamResponsesChat(request);
        return;
      }

      const body = this.createChatCompletionRequestBody(request);
      const reader = await this.postStreamReader(
        request,
        XAI_CHAT_COMPLETIONS_API_URLS,
        body,
      );
      const toolCallsByIndex = new Map<
        number,
        OpenAIStreamToolCallAccumulator
      >();
      yield* this.consumeChatCompletionsStream(reader, toolCallsByIndex);
    } catch (error) {
      if (error instanceof ProviderError) throw error;
      throw this.translateError(error);
    }
  }

  private async *streamResponsesChat(
    request: ChatRequest,
  ): AsyncIterable<ChatStreamEvent> {
    const body = this.createResponsesRequestBody(request);
    const reader = await this.postStreamReader(
      request,
      XAI_RESPONSES_API_URLS,
      body,
    );

    yield* this.consumeResponsesStream(reader);
  }

  private async postStreamReader(
    request: ChatRequest,
    apiUrls: readonly string[],
    body: Record<string, unknown>,
  ): Promise<ReadableStreamDefaultReader<Uint8Array>> {
    const response = await withRetry(
      () => this.createPostResponse(apiUrls, body, request.signal),
      this.buildRetryOptions(
        request,
        this.model,
        this.retryConfig,
        this.onDiagnosticEvent,
      ),
    );

    const reader = this.getResponseReader(response);
    if (!reader) {
      throw this.translateError(new Error("No response body"));
    }

    return reader;
  }

  private createResponsesRequestBody(
    request: ChatRequest,
  ): Record<string, unknown> {
    const body: Record<string, unknown> = {
      model: request.model || this.model,
      input: this.chatMessagesToResponsesInput(request.messages),
      stream: true,
    };

    if (request.tools && request.tools.length > 0) {
      body.tools = request.tools.map((tool) =>
        this.chatToolToResponsesTool(tool),
      );
    }

    return body;
  }

  private chatToolToResponsesTool(tool: ChatTool): Record<string, unknown> {
    return {
      type: "function",
      name: tool.function.name,
      description: tool.function.description,
      parameters: tool.function.parameters ?? {
        type: "object",
        properties: {},
      },
    };
  }

  private serializeToolArguments(args: unknown): string {
    return typeof args === "string" ? args : JSON.stringify(args ?? {});
  }

  private chatMessagesToResponsesInput(
    messages: ChatMessage[],
  ): Record<string, unknown>[] {
    const input: Record<string, unknown>[] = [];

    for (const msg of expandToolResultMessages(messages)) {
      this.appendResponsesInputMessage(msg, input);
    }

    return input;
  }

  private appendResponsesInputMessage(
    msg: ChatMessage,
    input: Record<string, unknown>[],
  ): void {
    if (msg.role === "system") {
      input.push({ role: "system", content: msg.content ?? "" });
      return;
    }

    if (msg.role === "user") {
      this.appendResponsesUserMessage(msg, input);
      return;
    }

    if (msg.role === "assistant") {
      this.appendResponsesAssistantMessage(msg, input);
      return;
    }

    if (msg.role === "tool" && msg.toolCallId) {
      input.push({
        type: "function_call_output",
        call_id: msg.toolCallId,
        output: msg.content ?? "",
      });
    }
  }

  private appendResponsesUserMessage(
    msg: ChatMessage,
    input: Record<string, unknown>[],
  ): void {
    const content: Record<string, unknown>[] = [];
    if (msg.content) {
      content.push({ type: "input_text", text: msg.content });
    }
    input.push({
      role: "user",
      content:
        content.length > 0 ? content : [{ type: "input_text", text: "" }],
    });
  }

  private appendResponsesAssistantMessage(
    msg: ChatMessage,
    input: Record<string, unknown>[],
  ): void {
    if (msg.content) {
      input.push({ role: "assistant", content: msg.content });
    }

    if (!msg.toolCalls) {
      return;
    }

    for (const toolCall of msg.toolCalls) {
      const callId =
        toolCall.id ?? `call_${toolCall.function.name}_${input.length}`;
      input.push({
        type: "function_call",
        id: callId,
        call_id: callId,
        name: toolCall.function.name,
        arguments: this.serializeToolArguments(toolCall.function.arguments),
        status: "completed",
      });
    }
  }

  private async *consumeResponsesStream(
    reader: ReadableStreamDefaultReader<Uint8Array>,
  ): AsyncIterable<ChatStreamEvent> {
    const state: XaiResponsesStreamState = {
      functionCallsByOutputIndex: new Map(),
      hasFunctionCall: false,
    };
    let stopReason: StopReason = "end_turn";

    for await (const data of readSseDataLines(reader)) {
      const result = this.parseResponsesStreamLine(data, state);
      if (result.stopReason) {
        stopReason = result.stopReason;
      }
      yield* result.events;
    }

    yield { type: "terminal", stopReason };
  }

  private createChatCompletionRequestBody(
    request: ChatRequest,
  ): Record<string, unknown> {
    return buildOpenAIChatCompletionRequestBody({
      request,
      model: this.model,
      mapMessage: (msg) => this.chatMessageToOpenAIMessage(msg),
      mapTool: (tool) => this.chatToolToOpenAITool(tool),
    });
  }

  private getResponseReader(
    response: Response,
  ): ReadableStreamDefaultReader<Uint8Array> | null {
    return response.body?.getReader() ?? null;
  }

  private async *consumeChatCompletionsStream(
    reader: ReadableStreamDefaultReader<Uint8Array>,
    toolCallsByIndex: Map<number, OpenAIStreamToolCallAccumulator>,
  ): AsyncIterable<ChatStreamEvent> {
    yield* consumeOpenAiChatCompletionsStream(reader, toolCallsByIndex);
  }

  private mapXaiResponsesStopReason(
    status: string | undefined,
    hasFunctionCall: boolean,
  ): "end_turn" | "tool_use" | "max_tokens" | "error" {
    if (hasFunctionCall) {
      return "tool_use";
    }

    if (status === "incomplete") {
      return "max_tokens";
    }

    if (status === "failed" || status === "cancelled") {
      return "error";
    }

    return "end_turn";
  }

  private buildResponsesToolCallsEvent(
    toolCall: ResponsesFunctionCall,
  ): ChatStreamEvent | null {
    const name = toolCall.name ?? "";
    if (!name) {
      return null;
    }

    const argsString = toolCall.arguments ?? toolCall.argsString ?? "";
    const toolCalls: ChatToolCall[] = [
      {
        id: toolCall.call_id ?? toolCall.id,
        function: {
          name,
          arguments: parseOpenAIStreamToolCallArguments(argsString),
        },
      },
    ];

    return { type: "tool_calls", toolCalls };
  }

  private parseResponsesStreamLine(
    data: string,
    state: XaiResponsesStreamState,
  ): ResponsesStreamLineResult {
    if (data === "[DONE]") {
      return { events: [] };
    }

    const event = parseJsonMaybe<XaiResponsesStreamEvent>(data);
    if (!event?.type) {
      return { events: [] };
    }

    return this.handleResponsesStreamEvent(event, state);
  }

  // fallow-ignore-next-line complexity
  private handleResponsesStreamEvent(
    event: XaiResponsesStreamEvent,
    state: XaiResponsesStreamState,
  ): ResponsesStreamLineResult {
    switch (event.type) {
      case "response.reasoning_summary_text.delta":
      case "response.reasoning_text.delta":
        return this.handleResponsesReasoningDelta(event);
      case "response.output_text.delta":
        return this.handleResponsesOutputTextDelta(event);
      case "response.output_item.added":
        return this.handleResponsesOutputItemAdded(event, state);
      case "response.function_call_arguments.delta":
        return this.handleResponsesFunctionCallArgumentsDelta(event, state);
      case "response.output_item.done":
        return this.handleResponsesOutputItemDone(event, state);
      case "response.completed":
      case "response.done":
      case "response.incomplete":
        return this.handleResponsesTerminalEvent(event, state);
      case "response.failed":
        return {
          events: [],
          stopReason: this.mapXaiResponsesStopReason("failed", false),
        };
      default:
        return { events: [] };
    }
  }

  private handleResponsesReasoningDelta(
    event: XaiResponsesStreamEvent,
  ): ResponsesStreamLineResult {
    if (!event.delta) {
      return { events: [] };
    }

    return { events: [{ type: "thinking_delta", delta: event.delta }] };
  }

  private handleResponsesOutputTextDelta(
    event: XaiResponsesStreamEvent,
  ): ResponsesStreamLineResult {
    if (!event.delta) {
      return { events: [] };
    }

    return { events: [{ type: "assistant_text", delta: event.delta }] };
  }

  private handleResponsesOutputItemAdded(
    event: XaiResponsesStreamEvent,
    state: XaiResponsesStreamState,
  ): ResponsesStreamLineResult {
    const item = event.item;
    if (item?.type !== "function_call") {
      return { events: [] };
    }

    const outputIndex = event.output_index ?? 0;
    state.functionCallsByOutputIndex.set(outputIndex, {
      id: item.call_id ?? item.id,
      name: item.name ?? "",
      argsString: item.arguments ?? "",
    });
    return { events: [] };
  }

  private handleResponsesFunctionCallArgumentsDelta(
    event: XaiResponsesStreamEvent,
    state: XaiResponsesStreamState,
  ): ResponsesStreamLineResult {
    const accumulated = state.functionCallsByOutputIndex.get(
      event.output_index ?? 0,
    );
    if (accumulated && event.delta) {
      accumulated.argsString += event.delta;
    }
    return { events: [] };
  }

  // fallow-ignore-next-line complexity
  private handleResponsesOutputItemDone(
    event: XaiResponsesStreamEvent,
    state: XaiResponsesStreamState,
  ): ResponsesStreamLineResult {
    const item = event.item;
    if (item?.type !== "function_call") {
      return { events: [] };
    }

    const outputIndex = event.output_index ?? 0;
    const accumulated = state.functionCallsByOutputIndex.get(outputIndex);
    const toolCallsEvent = this.buildResponsesToolCallsEvent({
      id: item.id,
      call_id: item.call_id ?? accumulated?.id,
      name: item.name ?? accumulated?.name,
      arguments: item.arguments ?? accumulated?.argsString,
    });
    state.functionCallsByOutputIndex.delete(outputIndex);
    if (!toolCallsEvent) {
      return { events: [] };
    }

    state.hasFunctionCall = true;
    return { events: [toolCallsEvent] };
  }

  private handleResponsesTerminalEvent(
    event: XaiResponsesStreamEvent,
    state: XaiResponsesStreamState,
  ): ResponsesStreamLineResult {
    const status =
      event.type === "response.incomplete"
        ? "incomplete"
        : event.response?.status;
    return {
      events: [],
      stopReason: this.mapXaiResponsesStopReason(status, state.hasFunctionCall),
    };
  }

  protected translateError(
    error: unknown,
    response?: Response,
    _responseBody?: string,
  ): ProviderError {
    return this.translateStandardOpenAiError(error, response, {
      model: this.model,
      authenticationMessage: "Invalid xAI API key",
      rateLimitMessage: "xAI rate limit exceeded",
      serviceErrorMessage: "xAI service error",
      connectionErrorMessage: "Failed to connect to xAI API",
      requestFailedMessage: "xAI request failed",
    });
  }
}

interface ResponsesFunctionCall {
  id?: string;
  call_id?: string;
  name?: string;
  arguments?: string;
  argsString?: string;
}

interface ResponsesStreamLineResult {
  events: ChatStreamEvent[];
  stopReason?: "end_turn" | "tool_use" | "max_tokens" | "error";
}

interface XaiResponsesStreamState {
  functionCallsByOutputIndex: Map<number, ResponsesFunctionCall>;
  hasFunctionCall: boolean;
}

interface XaiResponsesStreamEvent {
  type?: string;
  delta?: string;
  output_index?: number;
  item?: {
    type?: string;
    id?: string;
    call_id?: string;
    name?: string;
    arguments?: string;
  };
  response?: {
    status?: string;
  };
}
