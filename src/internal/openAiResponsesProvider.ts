import type { ProviderDiagnosticListener } from "../diagnostics.js";
import type {
  ChatMessage,
  ChatRequest,
  ChatStreamEvent,
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
} from "./openAiCompatibleProvider.js";
import {
  createResponsesFunctionTool,
  expandToolResultMessages,
  imageToOpenAIUrl,
  parseJsonMaybe,
  parseOpenAIStreamToolCallArguments,
  readSseDataLines,
  serializeToolArguments,
} from "./shared.js";
import { withRetry } from "./withRetry.js";
import { emitOpenAiCompatibleTrace } from "./providerMeasurements.js";

interface ResponsesFunctionCall {
  id?: string;
  call_id?: string;
  name?: string;
  arguments?: string;
}

interface ResponsesReasoningItem extends Record<string, unknown> {
  type: "reasoning";
}

interface ResponsesMessageItem extends Record<string, unknown> {
  type: "message";
}

type ResponsesReplayItem =
  | ResponsesReasoningItem
  | ResponsesMessageItem
  | (Record<string, unknown> &
      ResponsesFunctionCall & {
        type: "function_call";
      });

interface ResponsesStreamEvent {
  type?: string;
  delta?: string;
  output_index?: number;
  item?: Record<string, unknown> & ResponsesFunctionCall;
  error?: { code?: string; message?: string; param?: string };
  response?: {
    id?: string;
    model?: string;
    status?: string;
    error?: { code?: string; message?: string; param?: string };
    usage?: {
      input_tokens?: number;
      output_tokens?: number;
      total_tokens?: number;
      input_tokens_details?: { cached_tokens?: number };
      output_tokens_details?: { reasoning_tokens?: number };
    };
  };
}

interface FunctionCallAccumulator {
  id?: string;
  callId?: string;
  name: string;
  arguments: string;
}

interface ResponsesStreamState {
  readonly functionCallsByOutputIndex: Map<number, FunctionCallAccumulator>;
  readonly outputItemPhasesByOutputIndex: Map<number, string>;
  readonly replayItemsByOutputIndex: Map<number, ResponsesReplayItem>;
  stopReason: StopReason;
  rawStopReason?: string;
}

const RESPONSES_SUCCESS_EVENTS = new Set([
  "response.completed",
  "response.done",
]);
const RESPONSES_FAILURE_EVENTS = new Set([
  "response.failed",
  "response.cancelled",
]);

export interface OpenAiResponsesProviderProfile {
  readonly name: string;
  readonly apiUrl: string;
  readonly apiKeyEnv: string;
  readonly missingApiKeyMessage: string;
  readonly missingBodyMessage: string;
  readonly unauthorizedMessage: string;
  readonly capacityMessage: string;
  readonly authenticationMessage: string;
  readonly rateLimitMessage: string;
  readonly serviceErrorMessage: string;
  readonly connectionErrorMessage: string;
  readonly requestFailedMessage: string;
  /**
   * Preserve completed assistant commentary items for exact replay and mark
   * synthesized legacy assistant tool-call history as commentary.
   */
  readonly preserveAssistantCommentary?: boolean;
  /** Route Responses API commentary output to the shared thinking stream. */
  readonly routeCommentaryToThinking?: boolean;
  /** Subscription inference must explicitly complete before tools can execute. */
  readonly requireCompletedResponse?: boolean;
}

export interface OpenAiResponsesProviderOptions extends OpenAiCompatibleProviderOptions {
  /** Application-owned credentials, resolved for each network attempt. */
  readonly getBearerToken?: (signal?: AbortSignal) => Promise<string>;
  /** Subscription stream inactivity bound; does not apply to API providers. */
  readonly idleTimeoutMs?: number;
}

/** Shared implementation for providers exposing the OpenAI Responses wire API. */
export class OpenAiResponsesProvider extends OpenAiCompatibleProvider {
  readonly name: string;
  private readonly model: string;
  private readonly apiKey: string;
  private readonly getBearerToken?: OpenAiResponsesProviderOptions["getBearerToken"];
  private readonly retryConfig?: OpenAiCompatibleProviderOptions["retryConfig"];
  private readonly onDiagnosticEvent?: ProviderDiagnosticListener;
  private readonly profile: OpenAiResponsesProviderProfile;
  private readonly idleTimeoutMs: number;

  constructor(
    options: OpenAiResponsesProviderOptions,
    profile: OpenAiResponsesProviderProfile,
  ) {
    super();
    const apiKey = options.getBearerToken
      ? ""
      : (options.apiKey ?? process.env[profile.apiKeyEnv] ?? "");
    if (!options.getBearerToken && !apiKey.trim()) {
      throw new ProviderAuthenticationError(profile.missingApiKeyMessage);
    }

    this.name = profile.name;
    this.model = options.model;
    this.apiKey = apiKey;
    this.getBearerToken = options.getBearerToken;
    this.retryConfig = options.retryConfig;
    this.onDiagnosticEvent = options.onDiagnosticEvent;
    this.profile = profile;
    this.idleTimeoutMs = options.idleTimeoutMs ?? 300000;
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
          "responses",
          () => ({ transport: "http_json", requestBody: body }),
        ),
      );
      const reader = response.body?.getReader();
      if (!reader) {
        throw new ProviderError(this.profile.missingBodyMessage);
      }

      try {
        yield* this.consumeResponsesStream(reader, request, response);
      } finally {
        if (this.profile.requireCompletedResponse) {
          await reader.cancel().catch(() => {});
          reader.releaseLock();
        }
      }
    } catch (error) {
      this.throwIfSubscriptionAborted(request.signal);
      if (error instanceof ProviderError) {
        throw error;
      }
      throw this.translateError(error);
    }
  }

  protected createRequestBody(request: ChatRequest): Record<string, unknown> {
    const body: Record<string, unknown> = {
      model: request.model || this.model,
      input: this.messagesToInput(request.messages),
      stream: true,
      store: false,
      include: ["reasoning.encrypted_content"],
    };

    if (request.tools?.length) {
      body.tools = request.tools.map(createResponsesFunctionTool);
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
    const replayItems = this.parseReplayItems(message.reasoningContent);
    this.appendReplayItemsAndAssistantMessage(message, replayItems, input);

    const replayedCallIds = new Set(
      replayItems
        .filter((item) => item.type === "function_call")
        .flatMap((item) => [item.call_id, item.id])
        .filter((id): id is string => typeof id === "string"),
    );
    for (const toolCall of message.toolCalls ?? []) {
      const callId = toolCall.id ?? `call_${toolCall.function.name}`;
      if (replayedCallIds.has(callId)) {
        continue;
      }
      input.push({
        type: "function_call",
        call_id: callId,
        name: toolCall.function.name,
        arguments: serializeToolArguments(toolCall.function.arguments),
        status: "completed",
      });
    }
  }

  private appendReplayItemsAndAssistantMessage(
    message: ChatMessage,
    replayItems: ResponsesReplayItem[],
    input: Record<string, unknown>[],
  ): void {
    const hasReplayedAssistantMessage = replayItems.some(
      (item) => item.type === "message",
    );
    if (!message.content || hasReplayedAssistantMessage) {
      input.push(...replayItems);
      return;
    }

    const assistantMessage = this.createAssistantMessageInput(message);
    const firstFunctionCallIndex =
      this.profile.preserveAssistantCommentary === true
        ? replayItems.findIndex((item) => item.type === "function_call")
        : -1;
    if (firstFunctionCallIndex < 0) {
      input.push(...replayItems, assistantMessage);
      return;
    }

    input.push(
      ...replayItems.slice(0, firstFunctionCallIndex),
      assistantMessage,
      ...replayItems.slice(firstFunctionCallIndex),
    );
  }

  private createAssistantMessageInput(
    message: ChatMessage,
  ): Record<string, unknown> {
    if (
      this.profile.preserveAssistantCommentary === true &&
      message.toolCalls?.length
    ) {
      return {
        role: "assistant",
        phase: "commentary",
        content: [{ type: "output_text", text: message.content }],
      };
    }
    return { role: "assistant", content: message.content };
  }

  private parseReplayItems(
    reasoningContent: string | undefined,
  ): ResponsesReplayItem[] {
    if (!reasoningContent) {
      return [];
    }

    const parsed = parseJsonMaybe<unknown>(reasoningContent);
    if (!Array.isArray(parsed)) {
      return [];
    }

    return parsed.filter(
      (item): item is ResponsesReplayItem =>
        typeof item === "object" &&
        item !== null &&
        this.isReplayItemType(String((item as { type?: unknown }).type)),
    );
  }

  private isReplayItemType(type: string): boolean {
    return (
      type === "reasoning" ||
      type === "function_call" ||
      (type === "message" && this.profile.preserveAssistantCommentary === true)
    );
  }

  private async postResponse(
    body: Record<string, unknown>,
    signal?: AbortSignal,
  ): Promise<Response> {
    let response: Response;
    try {
      this.throwIfSubscriptionAborted(signal);
      const token = this.getBearerToken
        ? await this.getBearerToken(signal)
        : this.apiKey;
      this.throwIfSubscriptionAborted(signal);
      if (!token.trim()) {
        throw new ProviderAuthenticationError(
          this.profile.missingApiKeyMessage,
        );
      }
      response = await fetch(this.profile.apiUrl, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
        signal,
      });
    } catch (error) {
      this.throwIfSubscriptionAborted(signal);
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

  private throwIfSubscriptionAborted(signal?: AbortSignal): void {
    if (this.profile.requireCompletedResponse) signal?.throwIfAborted();
  }

  private async *consumeResponsesStream(
    reader: ReadableStreamDefaultReader<Uint8Array>,
    request: ChatRequest,
    response: Response,
  ): AsyncIterable<ChatStreamEvent> {
    const state: ResponsesStreamState = {
      functionCallsByOutputIndex: new Map(),
      outputItemPhasesByOutputIndex: new Map(),
      replayItemsByOutputIndex: new Map(),
      stopReason: "end_turn",
    };
    const measurementState = { responseMetadataObserved: false };
    let completed = false;

    const guard = this.profile.requireCompletedResponse
      ? { signal: request.signal, idleTimeoutMs: this.idleTimeoutMs }
      : undefined;
    for await (const data of readSseDataLines(reader, guard)) {
      const event = this.parseStreamEvent(data);
      if (!event) continue;
      this.captureTraceFields(event, request, measurementState);
      completed = this.trackSubscriptionCompletion(event, response, completed);
      const outputEvent = this.applyStreamEvent(event, state);
      if (outputEvent) {
        yield outputEvent;
      }
      if (completed && this.profile.requireCompletedResponse) break;
    }

    this.requireStreamCompletion(completed);

    const toolCallsEvent = this.buildToolCallsEvent(state);
    if (toolCallsEvent) {
      yield toolCallsEvent;
    }
    yield {
      type: "terminal",
      stopReason: state.stopReason,
      rawProviderReason: state.rawStopReason,
    };
  }

  private parseStreamEvent(data: string): ResponsesStreamEvent | null {
    if (data === "[DONE]") return null;
    const event = parseJsonMaybe<ResponsesStreamEvent>(data);
    if (event?.type) return event;
    if (this.profile.requireCompletedResponse)
      throw new ProviderError(
        "Malformed subscription stream; no tools were executed.",
      );
    return null;
  }

  private trackSubscriptionCompletion(
    event: ResponsesStreamEvent,
    response: Response,
    completed: boolean,
  ): boolean {
    if (!this.profile.requireCompletedResponse) return completed;
    return this.validateSubscriptionEvent(event, response) || completed;
  }

  private requireStreamCompletion(completed: boolean): void {
    if (this.profile.requireCompletedResponse && !completed) {
      throw new ProviderError(
        "Subscription stream ended without response.completed; no tools were executed.",
      );
    }
  }

  private validateSubscriptionEvent(
    event: ResponsesStreamEvent,
    response: Response,
  ): boolean {
    if (event.type === "response.incomplete") {
      throw new ProviderError(
        "Subscription response was incomplete; no tools were executed.",
      );
    }
    if (
      RESPONSES_FAILURE_EVENTS.has(event.type ?? "") ||
      event.type === "error"
    ) {
      const details = event.response?.error ??
        event.error ?? { code: event.type };
      throw this.translateError(
        new Error("Subscription stream failed."),
        response,
        JSON.stringify({ error: details }),
      );
    }
    return event.type === "response.completed";
  }

  private captureTraceFields(
    event: ResponsesStreamEvent,
    request: ChatRequest,
    state: { responseMetadataObserved: boolean },
  ): void {
    const response = event.response;
    if (!response) return;
    emitOpenAiCompatibleTrace({
      provider: this.name,
      request,
      endpointClass: "responses",
      responseId: response.id,
      actualModel: response.model,
      usage: response.usage,
      includeResponseMetadata: !state.responseMetadataObserved,
    });
    if (response.id || response.model) state.responseMetadataObserved = true;
  }

  private applyStreamEvent(
    event: ResponsesStreamEvent,
    state: ResponsesStreamState,
  ): ChatStreamEvent | null {
    const visibleEvent = this.createVisibleStreamEvent(event, state);
    if (visibleEvent) {
      return visibleEvent;
    }

    this.captureStreamState(event, state);
    return null;
  }

  private createVisibleStreamEvent(
    event: ResponsesStreamEvent,
    state: ResponsesStreamState,
  ): ChatStreamEvent | null {
    switch (event.type) {
      case "response.output_text.delta":
        if (!event.delta) {
          return null;
        }
        return this.isCommentaryOutput(event, state)
          ? { type: "thinking_delta", delta: event.delta }
          : { type: "assistant_text", delta: event.delta };
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
        this.captureOutputItemPhase(event, state);
        this.captureFunctionCall(event, state, false);
        return true;
      case "response.function_call_arguments.delta":
        this.captureFunctionArgumentsDelta(event, state);
        return true;
      case "response.output_item.done":
        this.captureOutputItemPhase(event, state);
        this.captureCompletedItem(event, state);
        return true;
      default:
        return false;
    }
  }

  private isCommentaryOutput(
    event: ResponsesStreamEvent,
    state: ResponsesStreamState,
  ): boolean {
    return (
      this.profile.routeCommentaryToThinking === true &&
      state.outputItemPhasesByOutputIndex.get(event.output_index ?? 0) ===
        "commentary"
    );
  }

  private captureOutputItemPhase(
    event: ResponsesStreamEvent,
    state: ResponsesStreamState,
  ): void {
    if (
      event.item?.type !== "message" ||
      typeof event.item.phase !== "string"
    ) {
      return;
    }
    state.outputItemPhasesByOutputIndex.set(
      event.output_index ?? 0,
      event.item.phase,
    );
  }

  private captureTerminalState(
    event: ResponsesStreamEvent,
    state: ResponsesStreamState,
  ): void {
    if (this.captureIncompleteState(event, state)) return;
    if (this.captureFailureState(event, state)) return;
    this.captureSuccessState(event, state);
  }

  private captureIncompleteState(
    event: ResponsesStreamEvent,
    state: ResponsesStreamState,
  ): boolean {
    if (event.type !== "response.incomplete") return false;
    state.rawStopReason = "incomplete";
    state.stopReason = "max_tokens";
    return true;
  }

  private captureFailureState(
    event: ResponsesStreamEvent,
    state: ResponsesStreamState,
  ): boolean {
    if (!RESPONSES_FAILURE_EVENTS.has(event.type ?? "")) return false;
    state.rawStopReason = event.response?.status ?? event.type;
    state.stopReason = "error";
    return true;
  }

  private captureSuccessState(
    event: ResponsesStreamEvent,
    state: ResponsesStreamState,
  ): void {
    if (!RESPONSES_SUCCESS_EVENTS.has(event.type ?? "")) return;
    state.rawStopReason = event.response?.status;
    state.stopReason = this.mapStopReason(
      event.response?.status,
      state.functionCallsByOutputIndex.size > 0,
    );
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
    if (
      event.item?.type === "reasoning" ||
      event.item?.type === "function_call" ||
      (event.item?.type === "message" &&
        this.profile.preserveAssistantCommentary === true)
    ) {
      state.replayItemsByOutputIndex.set(
        event.output_index ?? 0,
        event.item as ResponsesReplayItem,
      );
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

    const replayItems = [...state.replayItemsByOutputIndex]
      .sort(([left], [right]) => left - right)
      .map(([, item]) => item);

    // The Responses API requires prior output items to be replayed unchanged.
    // reasoningContent is provider-private continuation state. It can contain
    // opaque reasoning/function-call state and, for commentary-preserving
    // profiles, plaintext assistant message items in their original order.
    return {
      type: "tool_calls",
      toolCalls,
      ...(replayItems.length
        ? { reasoningContent: JSON.stringify(replayItems) }
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
        this.profile.unauthorizedMessage,
        originalError,
      );
    }
    if (response?.status === 503) {
      return new ProviderCapacityError(
        this.profile.capacityMessage,
        originalError,
      );
    }

    return this.translateStandardOpenAiError(error, response, {
      model: this.model,
      authenticationMessage: this.profile.authenticationMessage,
      rateLimitMessage: this.profile.rateLimitMessage,
      serviceErrorMessage: this.profile.serviceErrorMessage,
      connectionErrorMessage: this.profile.connectionErrorMessage,
      requestFailedMessage: this.profile.requestFailedMessage,
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
