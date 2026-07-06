import type { ProviderDiagnosticListener } from "../diagnostics.js";
import type {
  ChatMessage,
  ChatRequest,
  ChatStreamEvent,
  ChatTool,
  ChatToolCall,
  StopReason,
} from "../types.js";
import {
  ProviderAuthenticationError,
  ProviderCapacityError,
  ProviderError,
} from "../types.js";
import {
  OpenAiCompatibleProvider,
  type OpenAiCompatibleProviderOptions,
} from "../internal/openAiCompatibleProvider.js";
import {
  expandToolResultMessages,
  imageToOpenAIUrl,
  parseJsonMaybe,
  parseOpenAIStreamToolCallArguments,
  readSseDataLines,
} from "../internal/shared.js";
import { withRetry } from "../internal/withRetry.js";

const OPENAI_RESPONSES_API_URL = "https://api.openai.com/v1/responses";

interface ResponsesFunctionCall {
  id?: string;
  call_id?: string;
  name?: string;
  arguments?: string;
}

interface ResponsesReasoningItem extends Record<string, unknown> {
  type: "reasoning";
}

interface ResponsesStreamEvent {
  type?: string;
  delta?: string;
  output_index?: number;
  item?: Record<string, unknown> & ResponsesFunctionCall;
  response?: { status?: string };
}

interface FunctionCallAccumulator {
  id?: string;
  callId?: string;
  name: string;
  arguments: string;
}

interface ResponsesStreamState {
  readonly functionCallsByOutputIndex: Map<number, FunctionCallAccumulator>;
  readonly reasoningItems: ResponsesReasoningItem[];
  stopReason: StopReason;
}

/** OpenAI implementation using the forward-looking Responses API. */
export class OpenAiProvider extends OpenAiCompatibleProvider {
  readonly name = "openai";
  private readonly model: string;
  private readonly apiKey: string;
  private readonly retryConfig?: OpenAiCompatibleProviderOptions["retryConfig"];
  private readonly onDiagnosticEvent?: ProviderDiagnosticListener;

  constructor(options: OpenAiCompatibleProviderOptions) {
    super();
    const apiKey = options.apiKey ?? process.env.OPENAI_API_KEY ?? "";
    if (!apiKey.trim()) {
      throw new ProviderAuthenticationError(
        "OpenAI API key is required. Set OPENAI_API_KEY or pass apiKey in options.",
      );
    }

    this.model = options.model;
    this.apiKey = apiKey;
    this.retryConfig = options.retryConfig;
    this.onDiagnosticEvent = options.onDiagnosticEvent;
    this.configureCapabilities(options.contextWindowTokens);
  }

  async *streamChat(request: ChatRequest): AsyncIterable<ChatStreamEvent> {
    try {
      const body = this.createRequestBody(request);
      const response = await withRetry(
        () => this.postResponse(body, request.signal),
        this.buildRetryOptions(
          request,
          this.model,
          this.retryConfig,
          this.onDiagnosticEvent,
        ),
      );
      const reader = response.body?.getReader();
      if (!reader) {
        throw new ProviderError("OpenAI response had no body");
      }

      yield* this.consumeResponsesStream(reader);
    } catch (error) {
      if (error instanceof ProviderError) {
        throw error;
      }
      throw this.translateError(error);
    }
  }

  private createRequestBody(request: ChatRequest): Record<string, unknown> {
    const body: Record<string, unknown> = {
      model: request.model || this.model,
      input: this.messagesToInput(request.messages),
      stream: true,
      store: false,
      include: ["reasoning.encrypted_content"],
    };

    if (request.tools?.length) {
      body.tools = request.tools.map((tool) => this.toolToResponseTool(tool));
    }

    if (request.requestReasoning) {
      body.reasoning = { summary: "auto" };
    }

    return body;
  }

  private messagesToInput(messages: ChatMessage[]): Record<string, unknown>[] {
    const input: Record<string, unknown>[] = [];
    for (const message of expandToolResultMessages(messages)) {
      this.appendMessageInput(message, input);
    }
    return input;
  }

  private appendMessageInput(
    message: ChatMessage,
    input: Record<string, unknown>[],
  ): void {
    switch (message.role) {
      case "tool":
        this.appendToolResultInput(message, input);
        break;
      case "assistant":
        this.appendAssistantInput(message, input);
        break;
      case "user":
        input.push(this.createUserInput(message));
        break;
      default:
        input.push({ role: message.role, content: message.content ?? "" });
    }
  }

  private appendToolResultInput(
    message: ChatMessage,
    input: Record<string, unknown>[],
  ): void {
    if (!message.toolCallId) {
      return;
    }
    input.push({
      type: "function_call_output",
      call_id: message.toolCallId,
      output: message.content ?? "",
    });
  }

  private createUserInput(message: ChatMessage): Record<string, unknown> {
    if (!message.images?.length) {
      return { role: "user", content: message.content ?? "" };
    }

    const content: Record<string, unknown>[] = [];
    if (message.content) {
      content.push({ type: "input_text", text: message.content });
    }
    for (const image of message.images) {
      content.push({
        type: "input_image",
        image_url: imageToOpenAIUrl(image),
      });
    }
    return { role: "user", content };
  }

  private appendAssistantInput(
    message: ChatMessage,
    input: Record<string, unknown>[],
  ): void {
    input.push(...this.parseReasoningContent(message.reasoningContent));

    if (message.content) {
      input.push({ role: "assistant", content: message.content });
    }

    for (const toolCall of message.toolCalls ?? []) {
      const callId = toolCall.id ?? `call_${toolCall.function.name}`;
      input.push({
        type: "function_call",
        id: callId,
        call_id: callId,
        name: toolCall.function.name,
        arguments: this.serializeArguments(toolCall.function.arguments),
        status: "completed",
      });
    }
  }

  private parseReasoningContent(
    reasoningContent: string | undefined,
  ): ResponsesReasoningItem[] {
    if (!reasoningContent) {
      return [];
    }

    const parsed = parseJsonMaybe<unknown>(reasoningContent);
    if (!Array.isArray(parsed)) {
      return [];
    }

    return parsed.filter(
      (item): item is ResponsesReasoningItem =>
        typeof item === "object" &&
        item !== null &&
        (item as { type?: unknown }).type === "reasoning",
    );
  }

  private serializeArguments(argumentsValue: unknown): string {
    return typeof argumentsValue === "string"
      ? argumentsValue
      : JSON.stringify(argumentsValue ?? {});
  }

  private toolToResponseTool(tool: ChatTool): Record<string, unknown> {
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

  private async postResponse(
    body: Record<string, unknown>,
    signal?: AbortSignal,
  ): Promise<Response> {
    let response: Response;
    try {
      response = await fetch(OPENAI_RESPONSES_API_URL, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
        signal,
      });
    } catch (error) {
      throw this.translateError(error);
    }

    if (response.ok) {
      return response;
    }

    let responseBody = "";
    try {
      responseBody = await response.text();
    } catch {
      // Preserve the status-based translation when the body cannot be read.
    }
    throw this.translateError(
      new Error(responseBody || `HTTP ${response.status}`),
      response,
      responseBody,
    );
  }

  private async *consumeResponsesStream(
    reader: ReadableStreamDefaultReader<Uint8Array>,
  ): AsyncIterable<ChatStreamEvent> {
    const state: ResponsesStreamState = {
      functionCallsByOutputIndex: new Map(),
      reasoningItems: [],
      stopReason: "end_turn",
    };

    for await (const data of readSseDataLines(reader)) {
      if (data === "[DONE]") {
        continue;
      }
      const event = parseJsonMaybe<ResponsesStreamEvent>(data);
      if (!event?.type) {
        continue;
      }
      const outputEvent = this.applyStreamEvent(event, state);
      if (outputEvent) {
        yield outputEvent;
      }
    }

    const toolCallsEvent = this.buildToolCallsEvent(state);
    if (toolCallsEvent) {
      yield toolCallsEvent;
    }
    yield { type: "terminal", stopReason: state.stopReason };
  }

  private applyStreamEvent(
    event: ResponsesStreamEvent,
    state: ResponsesStreamState,
  ): ChatStreamEvent | null {
    const visibleEvent = this.createVisibleStreamEvent(event);
    if (visibleEvent) {
      return visibleEvent;
    }

    this.captureStreamState(event, state);
    return null;
  }

  private createVisibleStreamEvent(
    event: ResponsesStreamEvent,
  ): ChatStreamEvent | null {
    switch (event.type) {
      case "response.output_text.delta":
        return event.delta
          ? { type: "assistant_text", delta: event.delta }
          : null;
      case "response.reasoning_summary_text.delta":
        return event.delta
          ? {
              type: "reasoning_summary",
              summary: event.delta,
              source: "provider",
            }
          : null;
      default:
        return null;
    }
  }

  private captureStreamState(
    event: ResponsesStreamEvent,
    state: ResponsesStreamState,
  ): void {
    if (this.captureStreamItemState(event, state)) {
      return;
    }
    this.captureTerminalState(event, state);
  }

  private captureStreamItemState(
    event: ResponsesStreamEvent,
    state: ResponsesStreamState,
  ): boolean {
    switch (event.type) {
      case "response.output_item.added":
        this.captureFunctionCall(event, state, false);
        return true;
      case "response.function_call_arguments.delta":
        this.captureFunctionArgumentsDelta(event, state);
        return true;
      case "response.output_item.done":
        this.captureCompletedItem(event, state);
        return true;
      default:
        return false;
    }
  }

  private captureTerminalState(
    event: ResponsesStreamEvent,
    state: ResponsesStreamState,
  ): void {
    switch (event.type) {
      case "response.completed":
      case "response.done":
        state.stopReason = this.mapStopReason(
          event.response?.status,
          state.functionCallsByOutputIndex.size > 0,
        );
        break;
      case "response.incomplete":
        state.stopReason = "max_tokens";
        break;
      case "response.failed":
      case "response.cancelled":
        state.stopReason = "error";
        break;
    }
  }

  private captureFunctionCall(
    event: ResponsesStreamEvent,
    state: ResponsesStreamState,
    replaceArguments: boolean,
  ): void {
    const item = event.item;
    if (item?.type !== "function_call") {
      return;
    }
    const index = event.output_index ?? 0;
    const current = state.functionCallsByOutputIndex.get(index) ?? {
      name: "",
      arguments: "",
    };
    this.mergeFunctionCall(current, item, replaceArguments);
    state.functionCallsByOutputIndex.set(index, current);
  }

  private mergeFunctionCall(
    current: FunctionCallAccumulator,
    item: ResponsesStreamEvent["item"],
    replaceArguments: boolean,
  ): void {
    if (!item) {
      return;
    }
    if (item.id !== undefined) {
      current.id = item.id;
    }
    if (item.call_id !== undefined) {
      current.callId = item.call_id;
    }
    if (item.name !== undefined) {
      current.name = item.name;
    }
    if (typeof item.arguments !== "string") {
      return;
    }
    if (replaceArguments || !current.arguments) {
      current.arguments = item.arguments;
    }
  }

  private captureFunctionArgumentsDelta(
    event: ResponsesStreamEvent,
    state: ResponsesStreamState,
  ): void {
    if (!event.delta) {
      return;
    }
    const index = event.output_index ?? 0;
    const current = state.functionCallsByOutputIndex.get(index) ?? {
      name: "",
      arguments: "",
    };
    current.arguments += event.delta;
    state.functionCallsByOutputIndex.set(index, current);
  }

  private captureCompletedItem(
    event: ResponsesStreamEvent,
    state: ResponsesStreamState,
  ): void {
    if (event.item?.type === "reasoning") {
      state.reasoningItems.push(event.item as ResponsesReasoningItem);
      return;
    }
    this.captureFunctionCall(event, state, true);
  }

  private buildToolCallsEvent(
    state: ResponsesStreamState,
  ): ChatStreamEvent | null {
    if (!state.functionCallsByOutputIndex.size) {
      return null;
    }

    const toolCalls: ChatToolCall[] = [...state.functionCallsByOutputIndex]
      .sort(([left], [right]) => left - right)
      .map(([, call]) => ({
        id: call.callId ?? call.id,
        function: {
          name: call.name,
          arguments: parseOpenAIStreamToolCallArguments(call.arguments),
        },
      }));

    return {
      type: "tool_calls",
      toolCalls,
      ...(state.reasoningItems.length
        ? { reasoningContent: JSON.stringify(state.reasoningItems) }
        : {}),
    };
  }

  private mapStopReason(
    status: string | undefined,
    hasFunctionCalls: boolean,
  ): StopReason {
    if (hasFunctionCalls) {
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

  protected translateError(
    error: unknown,
    response?: Response,
    _responseBody?: string,
  ): ProviderError {
    const originalError = this.createOriginalError(error);

    if (response?.status === 403) {
      return new ProviderAuthenticationError(
        "OpenAI API key is not authorized for this request",
        originalError,
      );
    }
    if (response?.status === 529) {
      return new ProviderCapacityError(
        "OpenAI capacity is temporarily exhausted",
        originalError,
      );
    }

    return this.translateStandardOpenAiError(error, response, {
      model: this.model,
      authenticationMessage: "Invalid OpenAI API key",
      rateLimitMessage: "OpenAI rate limit exceeded",
      serviceErrorMessage: "OpenAI service error",
      connectionErrorMessage: "Failed to connect to OpenAI API",
      requestFailedMessage: "OpenAI request failed",
    });
  }

  protected isRetryableError(error: unknown): boolean {
    if (
      error instanceof ProviderError &&
      error.message === "Request cancelled"
    ) {
      return false;
    }
    return super.isRetryableError(error);
  }
}
