import {
  BedrockRuntimeClient,
  ConverseCommand,
  ConverseStreamCommand,
} from "@aws-sdk/client-bedrock-runtime";
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

interface BedrockStreamState {
  toolCalls: ChatToolCall[];
  currentToolCall: Partial<ChatToolCall> | null;
  currentToolInput: string;
  stopReason: "end_turn" | "tool_use" | "max_tokens" | "stop_sequence";
}

interface BedrockErrorDetails {
  message: string;
  name: string;
  originalError?: Error;
}

/**
 * Bedrock implementation of LLMProvider
 */
export class BedrockProvider extends BaseProvider {
  readonly name = "bedrock";
  private client: BedrockRuntimeClient;

  constructor(options: BaseProviderOptions & { region?: string }) {
    super(options);
    this.client = new BedrockRuntimeClient({
      region: options.region ?? "us-east-1",
    });
  }

  async *streamChat(request: ChatRequest): AsyncIterable<ChatStreamEvent> {
    try {
      const command = this.createConverseStreamCommand(request);
      const response = await withRetry(
        () => this.client.send(command, { abortSignal: request.signal }),
        this.createRetryOptions(request),
      );
      const stream = this.getStreamFromResponse(response);

      if (!stream) {
        throw new Error("No stream output");
      }

      const state: BedrockStreamState = {
        toolCalls: [],
        currentToolCall: null,
        currentToolInput: "",
        stopReason: "end_turn",
      };

      for await (const event of stream as AsyncIterable<any>) {
        this.captureStopReason(event, state);
        const assistantText = this.handleStreamEvent(event, state);
        if (assistantText) {
          yield { type: "assistant_text", delta: assistantText };
        }
      }

      if (state.toolCalls.length > 0) {
        yield { type: "tool_calls", toolCalls: state.toolCalls };
      }

      // Emit normalized terminal event (Phase 4.5)
      yield { type: "terminal", stopReason: state.stopReason };
    } catch (error) {
      throw this.translateError(error);
    }
  }

  private createRetryOptions(request: ChatRequest) {
    return {
      maxRetries: this.retryConfig?.maxRetries ?? 3,
      baseDelayMs: 500,
      isRetryable: (error: unknown) => this.isRetryableError(error),
      consecutive529Limit: this.retryConfig?.consecutive529Limit ?? 3,
      onRetry: (ctx: {
        err: unknown;
        attempt: number;
        delayMs: number;
      }): void => this.emitRetryDiagnostic(request, ctx),
    };
  }

  private isRetryableError(error: unknown): boolean {
    const name = (error as any)?.name ?? "";
    const message = error instanceof Error ? error.message : String(error);
    return (
      name === "ThrottlingException" ||
      name === "ServiceUnavailableException" ||
      name === "InternalServerException" ||
      message.includes("rate limit") ||
      message.includes("throttl")
    );
  }

  private emitRetryDiagnostic(
    request: ChatRequest,
    ctx: { err: unknown; attempt: number; delayMs: number },
  ): void {
    this.onDiagnosticEvent?.({
      type: "provider_retry",
      provider: this.name,
      model: request.model || this.model,
      iteration: request.iteration ?? 0,
      reason: ctx.err instanceof Error ? ctx.err.message : String(ctx.err),
      attemptNumber: ctx.attempt + 1,
      delayMs: ctx.delayMs,
    });
  }

  private captureStopReason(event: any, state: BedrockStreamState): void {
    if (event.messageStop?.stopReason) {
      state.stopReason = this.mapStopReason(event.messageStop.stopReason);
    }
  }

  private createConverseStreamCommand(
    request: ChatRequest,
  ): ConverseStreamCommand {
    const systemMessage = request.messages.find((m) => m.role === "system");
    const messages = request.messages
      .filter((m) => m.role !== "system")
      .map((msg) => this.chatMessageToBedrockMessage(msg));
    const toolConfig = request.tools
      ? {
          tools: request.tools.map((tool) => this.chatToolToBedrockTool(tool)),
        }
      : undefined;

    return new ConverseStreamCommand({
      modelId: request.model || this.model,
      messages: messages as any,
      system: systemMessage ? [{ text: systemMessage.content }] : undefined,
      toolConfig: toolConfig as any,
    });
  }

  private getStreamFromResponse(
    response: unknown,
  ): AsyncIterable<any> | undefined {
    let stream = (response as any).stream || (response as any).output;

    if (
      !stream &&
      typeof response === "object" &&
      response !== null &&
      Symbol.asyncIterator in response
    ) {
      stream = response;
    }

    return stream as AsyncIterable<any> | undefined;
  }

  private handleStreamEvent(
    event: any,
    state: {
      toolCalls: ChatToolCall[];
      currentToolCall: Partial<ChatToolCall> | null;
      currentToolInput: string;
    },
  ): string | undefined {
    let assistantText: string | undefined;

    if (event.contentBlockDelta) {
      assistantText = this.handleContentBlockDelta(
        event.contentBlockDelta.delta,
        state,
      );
    }

    if (event.contentBlockStart) {
      this.handleContentBlockStart(event.contentBlockStart.start, state);
    }

    if (event.contentBlockStop) {
      this.handleContentBlockStop(state);
    }

    return assistantText;
  }

  private handleContentBlockDelta(
    delta: any,
    state: {
      toolCalls: ChatToolCall[];
      currentToolCall: Partial<ChatToolCall> | null;
      currentToolInput: string;
    },
  ): string | undefined {
    if (delta.text) {
      return delta.text;
    }

    if (delta.toolUse) {
      const partialInput = delta.toolUse.input;
      if (partialInput) {
        state.currentToolInput += partialInput;
      }
    }

    return undefined;
  }

  private handleContentBlockStart(
    start: any,
    state: {
      toolCalls: ChatToolCall[];
      currentToolCall: Partial<ChatToolCall> | null;
      currentToolInput: string;
    },
  ): void {
    const toolUse = start?.toolUse;
    if (!toolUse) {
      return;
    }

    state.currentToolCall = {
      id: toolUse.toolUseId,
      function: {
        name: toolUse.name,
        arguments: {},
      },
    };
    state.currentToolInput = "";
  }

  private handleContentBlockStop(state: {
    toolCalls: ChatToolCall[];
    currentToolCall: Partial<ChatToolCall> | null;
    currentToolInput: string;
  }): void {
    if (!state.currentToolCall?.function || !state.currentToolInput) {
      return;
    }

    try {
      state.currentToolCall.function.arguments = JSON.parse(
        state.currentToolInput,
      );
    } catch {
      state.currentToolCall.function.arguments = {
        raw: state.currentToolInput,
      };
    }

    state.toolCalls.push(state.currentToolCall as ChatToolCall);
    state.currentToolCall = null;
    state.currentToolInput = "";
  }

  /**
   * Check if a message is a system message
   */
  private isSystemMessage(msg: ChatMessage): boolean {
    return msg.role === "system";
  }

  /**
   * Translate ChatMessage to Bedrock Message format
   */
  private chatMessageToBedrockMessage(msg: ChatMessage): any {
    const contentBlocks: any[] = [];

    this.appendTextContentBlock(msg, contentBlocks);
    this.appendToolCallBlocks(msg, contentBlocks);
    this.appendImageBlocks(msg, contentBlocks);
    this.appendToolResultBlocks(msg, contentBlocks);

    return {
      role: msg.role === "tool" ? "user" : (msg.role as any),
      content: contentBlocks,
    };
  }

  private appendTextContentBlock(msg: ChatMessage, contentBlocks: any[]): void {
    if (msg.content && msg.role !== "tool") {
      contentBlocks.push({
        text: msg.content,
      } as any);
    }
  }

  private appendToolCallBlocks(msg: ChatMessage, contentBlocks: any[]): void {
    if (!msg.toolCalls || msg.toolCalls.length === 0) {
      return;
    }

    for (const toolCall of msg.toolCalls) {
      contentBlocks.push({
        toolUse: {
          toolUseId: toolCall.id || `${toolCall.function.name}-${Date.now()}`,
          name: toolCall.function.name,
          input: toolCall.function.arguments,
        },
      } as any);
    }
  }

  private appendImageBlocks(msg: ChatMessage, contentBlocks: any[]): void {
    if (!msg.images || msg.images.length === 0) {
      return;
    }

    for (const image of msg.images) {
      if (typeof image === "string") {
        this.appendBase64ImageBlock(image, contentBlocks);
      } else {
        contentBlocks.push({
          image: {
            format: "png" as any,
            source: {
              bytes: image,
            },
          },
        } as any);
      }
    }
  }

  private appendBase64ImageBlock(image: string, contentBlocks: any[]): void {
    if (!image.startsWith("data:")) {
      return;
    }

    const [header, data] = image.split(",");
    const mediaType = header.match(/:(.*?);/)?.[1] || "image/png";
    contentBlocks.push({
      image: {
        format: (mediaType.split("/")[1] || "png") as any,
        source: {
          bytes: Buffer.from(data, "base64"),
        },
      },
    } as any);
  }

  private appendToolResultBlocks(msg: ChatMessage, contentBlocks: any[]): void {
    if (msg.role !== "tool") {
      return;
    }

    if (msg.toolResults && msg.toolResults.length > 0) {
      for (const toolResult of msg.toolResults) {
        contentBlocks.push({
          toolResult: {
            toolUseId: toolResult.toolCallId,
            content: [{ text: toolResult.content }],
            status: "success",
          },
        } as any);
      }
      return;
    }

    if (msg.toolCallId) {
      contentBlocks.push({
        toolResult: {
          toolUseId: msg.toolCallId,
          content: [{ text: msg.content }],
          status: "success",
        },
      } as any);
    }
  }

  /**
   * Translate Bedrock Message to ChatMessage
   */
  private bedrockMessageToChatMessage(msg: any): ChatMessage {
    let content = "";
    const toolCalls: ChatToolCall[] = [];

    if (msg.content && Array.isArray(msg.content)) {
      for (const block of msg.content) {
        // Skip undefined or null blocks
        if (!block) {
          continue;
        }

        if (block.text) {
          content += block.text;
        }

        if (block.toolUse) {
          toolCalls.push({
            id: block.toolUse.toolUseId, // Store toolUseId for later reference
            function: {
              name: block.toolUse.name,
              arguments: block.toolUse.input || {},
            },
          });
        }
      }
    }

    const chatMsg: ChatMessage = {
      role: "assistant",
      content,
    };

    if (toolCalls.length > 0) {
      chatMsg.toolCalls = toolCalls;
    }

    return chatMsg;
  }

  /**
   * Translate ChatTool to Bedrock ToolSpecification
   */
  private chatToolToBedrockTool(chatTool: ChatTool): any {
    // Bedrock expects the JSON schema directly in the inputSchema.json field
    // Strip out descriptions from properties as Bedrock may not support them
    const schema = chatTool.function.parameters || {
      type: "object",
      properties: {},
    };

    // Clean the schema by removing descriptions from properties
    const cleanedSchema = this.cleanSchema(schema);

    return {
      toolSpec: {
        name: chatTool.function.name,
        description: chatTool.function.description,
        inputSchema: {
          json: cleanedSchema,
        },
      },
    };
  }

  /**
   * Clean JSON schema by removing unsupported fields like descriptions from properties
   */
  private isSchemaObject(schema: any): schema is Record<string, any> {
    return Boolean(schema) && typeof schema === "object";
  }

  private cleanSchema(schema: any): any {
    if (!this.isSchemaObject(schema)) {
      return schema;
    }

    const cleaned = { ...schema };

    if (this.isSchemaObject(cleaned.properties)) {
      cleaned.properties = Object.fromEntries(
        Object.entries(cleaned.properties).map(([key, value]) => [
          key,
          this.cleanSchemaProperty(value),
        ]),
      );
    }

    return cleaned;
  }

  private cleanSchemaProperty(value: any): any {
    if (!this.isSchemaObject(value)) {
      return value;
    }

    const cleaned: any = {};
    this.copySchemaField(cleaned, value, "type");
    this.copySchemaField(cleaned, value, "enum");

    if (value.items) {
      cleaned.items = this.cleanSchema(value.items);
    }
    if (value.properties) {
      cleaned.properties = this.cleanSchema(value.properties);
    }

    return cleaned;
  }

  private copySchemaField(
    target: Record<string, unknown>,
    source: Record<string, unknown>,
    field: string,
  ): void {
    if (source[field] !== undefined) {
      target[field] = source[field];
    }
  }

  /**
   * Map Bedrock stop reason to provider-agnostic format
   */
  private mapStopReason(
    bedrockStopReason: string | undefined,
  ): "end_turn" | "tool_use" | "max_tokens" | "stop_sequence" {
    switch (bedrockStopReason) {
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

  /**
   * Translate errors to ProviderError types
   */
  private isContextLengthError(message: string, name: string): boolean {
    const lower = message.toLowerCase();
    return (
      name.includes("ModelInputLimitExceededException") ||
      lower.includes("too many input tokens") ||
      lower.includes("context length") ||
      lower.includes("input is too long") ||
      lower.includes("prompt is too long") ||
      lower.includes("maximum number of tokens") ||
      lower.includes("exceeds the model")
    );
  }

  private getErrorDetails(error: any): BedrockErrorDetails {
    return {
      message: error instanceof Error ? error.message : String(error),
      name: (error as any).name || "",
      ...(error instanceof Error ? { originalError: error } : {}),
    };
  }

  private translateError(error: any): ProviderError {
    const details = this.getErrorDetails(error);
    return (
      this.translateContextLengthError(details) ??
      this.translateAuthenticationError(details) ??
      this.translateModelNotFoundError(details) ??
      this.translateRateLimitError(details) ??
      this.translateServiceError(details) ??
      new ProviderError(details.message, details.originalError)
    );
  }

  private translateContextLengthError(
    details: BedrockErrorDetails,
  ): ProviderContextLengthError | null {
    if (this.isContextLengthError(details.message, details.name)) {
      return new ProviderContextLengthError(
        `Context length exceeded: ${details.message}`,
        details.originalError,
      );
    }
    return null;
  }

  private translateAuthenticationError(
    details: BedrockErrorDetails,
  ): ProviderAuthenticationError | null {
    if (
      details.name.includes("ValidationException") ||
      details.message.includes("Invalid") ||
      details.message.includes("credentials")
    ) {
      return new ProviderAuthenticationError(
        `Bedrock authentication failed: ${details.message}`,
        details.originalError,
      );
    }
    return null;
  }

  private translateModelNotFoundError(
    details: BedrockErrorDetails,
  ): ProviderModelNotFoundError | null {
    if (
      details.name.includes("ResourceNotFoundException") ||
      details.message.includes("model not found")
    ) {
      return new ProviderModelNotFoundError(
        this.model,
        `Model ${this.model} not found in Bedrock: ${details.message}`,
        details.originalError,
      );
    }
    return null;
  }

  private translateRateLimitError(
    details: BedrockErrorDetails,
  ): ProviderRateLimitError | null {
    if (
      details.name.includes("ThrottlingException") ||
      details.message.includes("rate limit") ||
      details.message.includes("throttl")
    ) {
      return new ProviderRateLimitError(
        `Bedrock rate limited: ${details.message}`,
        undefined,
        details.originalError,
      );
    }
    return null;
  }

  private translateServiceError(
    details: BedrockErrorDetails,
  ): ProviderError | null {
    if (
      details.name.includes("ServiceUnavailableException") ||
      details.name.includes("InternalServerException")
    ) {
      return new ProviderError(
        `Bedrock service error: ${details.message}`,
        details.originalError,
      );
    }
    return null;
  }
}
