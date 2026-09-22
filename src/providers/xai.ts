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
import { withRetry, type WithRetryOptions } from "../internal/withRetry.js";
import {
  buildOpenAIChatCompletionRequestBody,
  createResponsesFunctionTool,
  expandToolResultMessages,
  parseJsonMaybe,
  parseOpenAIStreamToolCallArguments,
  readSseDataLines,
  serializeToolArguments,
  type OpenAIStreamToolCallAccumulator,
} from "../internal/shared.js";
import { consumeOpenAiChatCompletionsStream } from "../internal/openAiStream.js";
import {
  OpenAiCompatibleProvider,
  type OpenAiCompatibleProviderOptions,
  type OpenAiCompatibleRetryConfig,
} from "../internal/openAiCompatibleProvider.js";
import { emitOpenAiCompatibleTrace } from "../internal/providerMeasurements.js";

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
  private readonly retryConfig?: OpenAiCompatibleRetryConfig;
  private readonly onDiagnosticEvent?: ProviderDiagnosticListener;
  private readonly endpointFallbackErrors = new WeakSet<object>();

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

  private async createEndpointResponseError(
    response: Response,
  ): Promise<ProviderError> {
    let errorBody = "";
    try {
      errorBody = await response.text();
    } catch {
      // ignore read failures
    }

    const error = this.translateError(
      new Error(errorBody || `HTTP ${response.status}`),
      response,
    );
    if (response.status >= 500 && response.status < 600) {
      this.endpointFallbackErrors.add(error);
    }
    return error;
  }

  private async createPostResponse(
    apiUrl: string,
    body: Record<string, unknown>,
    signal?: AbortSignal,
  ): Promise<Response> {
    let response: Response;
    try {
      response = await fetch(apiUrl, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
        signal,
      });
    } catch (error) {
      const translated =
        error instanceof ProviderError ? error : this.translateError(error);
      if (
        !(error instanceof ProviderError) &&
        !this.isCancellation(translated)
      ) {
        this.endpointFallbackErrors.add(translated);
      }
      throw translated;
    }
    if (!response.ok) throw await this.createEndpointResponseError(response);
    return response;
  }

  async *streamChat(request: ChatRequest): AsyncIterable<ChatStreamEvent> {
    try {
      if (request.requestReasoning) {
        yield* this.streamResponsesChat(request);
        return;
      }

      const body = this.createChatCompletionRequestBody(request);
      const { reader, endpointClass } = await this.postStreamReader(
        request,
        XAI_CHAT_COMPLETIONS_API_URLS,
        body,
      );
      const toolCallsByIndex = new Map<
        number,
        OpenAIStreamToolCallAccumulator
      >();
      yield* this.consumeChatCompletionsStream(
        reader,
        toolCallsByIndex,
        request,
        endpointClass,
      );
    } catch (error) {
      if (error instanceof ProviderError) throw error;
      throw this.translateError(error);
    }
  }

  private async *streamResponsesChat(
    request: ChatRequest,
  ): AsyncIterable<ChatStreamEvent> {
    const body = this.createResponsesRequestBody(request);
    const { reader, endpointClass } = await this.postStreamReader(
      request,
      XAI_RESPONSES_API_URLS,
      body,
    );

    yield* this.consumeResponsesStream(reader, request, endpointClass);
  }

  private async postStreamReader(
    request: ChatRequest,
    apiUrls: readonly string[],
    body: Record<string, unknown>,
  ): Promise<{
    reader: ReadableStreamDefaultReader<Uint8Array>;
    endpointClass: string;
  }> {
    const retryState: XaiEndpointRetryState = {
      endpointIndex: 0,
      logicalRetryCount: 0,
      nextRetry: "logical_retry",
    };
    let endpointClass = this.xaiEndpointClass(apiUrls[0]!);
    const response = await withRetry(
      () => {
        const apiUrl = apiUrls[retryState.endpointIndex]!;
        endpointClass = this.xaiEndpointClass(apiUrl);
        return this.createPostResponse(apiUrl, body, request.signal);
      },
      this.buildEndpointRetryOptions(request, apiUrls, retryState),
    );

    const reader = this.getResponseReader(response);
    if (!reader) {
      throw this.translateError(new Error("No response body"));
    }

    return { reader, endpointClass };
  }

  private buildEndpointRetryOptions(
    request: ChatRequest,
    apiUrls: readonly string[],
    state: XaiEndpointRetryState,
  ): WithRetryOptions {
    const configuredRetries = this.retryConfig?.maxRetries ?? 3;
    const base = this.buildRetryOptions(
      request,
      this.model,
      this.endpointRetryConfig(apiUrls.length),
      this.onDiagnosticEvent,
      () => this.xaiEndpointClass(apiUrls[state.endpointIndex]!),
    );
    const baseOnRetry = base.onRetry;

    return {
      ...base,
      isRetryable: (error) => {
        if (this.isCancellation(error) || !base.isRetryable(error)) {
          return false;
        }
        if (
          this.isEndpointFallbackError(error) &&
          state.endpointIndex < apiUrls.length - 1
        ) {
          state.nextRetry = "endpoint_fallback";
          return true;
        }
        state.nextRetry = "logical_retry";
        return state.logicalRetryCount < configuredRetries;
      },
      getBackoffAttempt: () =>
        state.nextRetry === "endpoint_fallback"
          ? null
          : state.logicalRetryCount,
      onRetry: (context) => {
        baseOnRetry?.(context);
        if (state.nextRetry === "endpoint_fallback") {
          state.endpointIndex += 1;
        } else {
          state.endpointIndex = 0;
          state.logicalRetryCount += 1;
        }
      },
    };
  }

  private isEndpointFallbackError(error: unknown): boolean {
    return typeof error === "object" && error !== null
      ? this.endpointFallbackErrors.has(error)
      : false;
  }

  private isCancellation(error: unknown): boolean {
    return (
      error instanceof Error &&
      (error.name === "AbortError" ||
        error.message === "Request cancelled" ||
        (error instanceof ProviderError &&
          error.originalError?.name === "AbortError"))
    );
  }

  private endpointRetryConfig(endpointCount: number): {
    maxRetries: number;
    consecutive529Limit: number;
    baseDelayMs?: number;
  } {
    const configuredRetries = this.retryConfig?.maxRetries ?? 3;
    return {
      // A retry attempt now represents one physical endpoint call. Preserve
      // the previous number of complete global/regional sweeps.
      maxRetries: (configuredRetries + 1) * endpointCount - 1,
      consecutive529Limit:
        (this.retryConfig?.consecutive529Limit ?? 3) * endpointCount,
      ...(this.retryConfig?.baseDelayMs !== undefined
        ? { baseDelayMs: this.retryConfig.baseDelayMs }
        : {}),
    };
  }

  private xaiEndpointClass(apiUrl: string): string {
    const api = apiUrl.includes("/responses")
      ? "responses"
      : "chat_completions";
    const region = apiUrl.includes("us-east-1")
      ? "us_east_1"
      : apiUrl.includes("eu-west-1")
        ? "eu_west_1"
        : "global";
    return `${api}:${region}`;
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
      body.tools = request.tools.map(createResponsesFunctionTool);
    }

    return body;
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
        arguments: serializeToolArguments(toolCall.function.arguments),
        status: "completed",
      });
    }
  }

  private async *consumeResponsesStream(
    reader: ReadableStreamDefaultReader<Uint8Array>,
    request: ChatRequest,
    endpointClass: string,
  ): AsyncIterable<ChatStreamEvent> {
    const state: XaiResponsesStreamState = {
      functionCallsByOutputIndex: new Map(),
      hasFunctionCall: false,
    };
    let stopReason: StopReason = "end_turn";
    let rawProviderReason: string | undefined;
    const measurementState = { responseMetadataObserved: false };

    for await (const data of readSseDataLines(reader)) {
      const result = this.parseResponsesStreamLine(
        data,
        state,
        request,
        endpointClass,
        measurementState,
      );
      if (result.stopReason) {
        stopReason = result.stopReason;
        rawProviderReason = result.rawProviderReason;
      }
      yield* result.events;
    }

    yield { type: "terminal", stopReason, rawProviderReason };
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
    request: ChatRequest,
    endpointClass: string,
  ): AsyncIterable<ChatStreamEvent> {
    yield* consumeOpenAiChatCompletionsStream(reader, toolCallsByIndex, {
      provider: this.name,
      request,
      endpointClass,
    });
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
    request: ChatRequest,
    endpointClass: string,
    measurementState: { responseMetadataObserved: boolean },
  ): ResponsesStreamLineResult {
    if (data === "[DONE]") {
      return { events: [] };
    }

    const event = parseJsonMaybe<XaiResponsesStreamEvent>(data);
    if (!event?.type) {
      return { events: [] };
    }

    this.captureResponsesTrace(event, request, endpointClass, measurementState);

    return this.handleResponsesStreamEvent(event, state);
  }

  private captureResponsesTrace(
    event: XaiResponsesStreamEvent,
    request: ChatRequest,
    endpointClass: string,
    state: { responseMetadataObserved: boolean },
  ): void {
    const response = event.response;
    if (!response) return;
    emitOpenAiCompatibleTrace({
      provider: this.name,
      request,
      endpointClass,
      responseId: response.id,
      actualModel: response.model,
      usage: response.usage,
      includeResponseMetadata: !state.responseMetadataObserved,
    });
    if (response.id || response.model) state.responseMetadataObserved = true;
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
      rawProviderReason: status,
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

interface XaiEndpointRetryState {
  endpointIndex: number;
  logicalRetryCount: number;
  nextRetry: "endpoint_fallback" | "logical_retry";
}

interface ResponsesStreamLineResult {
  events: ChatStreamEvent[];
  stopReason?: "end_turn" | "tool_use" | "max_tokens" | "error";
  rawProviderReason?: string;
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
    id?: string;
    model?: string;
    status?: string;
    usage?: {
      input_tokens?: number;
      output_tokens?: number;
      total_tokens?: number;
      input_tokens_details?: { cached_tokens?: number };
      output_tokens_details?: { reasoning_tokens?: number };
    };
  };
}
