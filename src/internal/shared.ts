import type { ProviderDiagnosticListener } from "../diagnostics.js";
import type {
  ChatMessage,
  ChatRequest,
  ChatTool,
  ChatToolCall,
  ToolResult,
} from "../types.js";
import { ProviderCapacityError } from "../types.js";
import type { WithRetryOptions } from "./withRetry.js";

export interface OpenAIToolDefinition {
  readonly type: "function";
  readonly function: {
    readonly name: string;
    readonly description: string;
    readonly parameters: object;
  };
}

export interface OpenAIMessageCore {
  readonly role: "user" | "assistant" | "system" | "tool";
  content: unknown;
  reasoning_content?: string;
  tool_calls?: ReturnType<typeof createOpenAIToolCall>[];
  tool_call_id?: string;
}

export interface OpenAIMessageContentPart {
  type: "text" | "image_url";
  text?: string;
  image_url?: { url: string };
}

export interface OpenAIStreamToolCallAccumulator {
  id?: string;
  name: string;
  argsString: string;
}

export function expandToolResultMessages(
  messages: ChatMessage[],
  toolCallIdForResult: (toolResult: ToolResult) => string = (toolResult) =>
    toolResult.toolCallId,
): ChatMessage[] {
  const expanded: ChatMessage[] = [];

  for (const msg of messages) {
    if (msg.role === "tool" && msg.toolResults && msg.toolResults.length > 0) {
      for (const toolResult of msg.toolResults) {
        expanded.push({
          role: "tool",
          content: toolResult.content,
          toolCallId: toolCallIdForResult(toolResult),
        });
      }
      continue;
    }

    expanded.push(msg);
  }

  return expanded;
}

export function buildOpenAIChatCompletionRequestBody<TMessage, TTool>(options: {
  request: ChatRequest;
  model: string;
  mapMessage: (message: ChatMessage) => TMessage;
  mapTool?: (tool: ChatTool) => TTool;
  includeTools?: boolean;
  extra?: (body: Record<string, unknown>) => void;
}): Record<string, unknown> {
  const messages = expandToolResultMessages(options.request.messages).map(
    options.mapMessage,
  );

  const body: Record<string, unknown> = {
    model: options.request.model || options.model,
    messages,
    stream: true,
  };

  if (
    options.includeTools !== false &&
    options.mapTool &&
    options.request.tools &&
    options.request.tools.length > 0
  ) {
    body.tools = options.request.tools.map(options.mapTool);
  }

  options.extra?.(body);
  return body;
}

export function createProviderRetryOptions(options: {
  request: ChatRequest;
  model: string;
  provider: string;
  retryConfig?: {
    maxRetries: number;
    consecutive529Limit: number;
    baseDelayMs?: number;
  };
  isRetryable: (err: unknown) => boolean;
  onDiagnosticEvent?: ProviderDiagnosticListener;
}): WithRetryOptions {
  const retryOptions: WithRetryOptions = {
    maxRetries: options.retryConfig?.maxRetries ?? 3,
    isRetryable: options.isRetryable,
    is529: (err) => err instanceof ProviderCapacityError,
    consecutive529Limit: options.retryConfig?.consecutive529Limit ?? 3,
    onRetry: (ctx) =>
      options.onDiagnosticEvent?.({
        type: "provider_retry",
        provider: options.provider,
        model: options.request.model || options.model,
        iteration: options.request.iteration ?? 0,
        reason: ctx.err instanceof Error ? ctx.err.message : String(ctx.err),
        attemptNumber: ctx.attempt + 1,
        delayMs: ctx.delayMs,
      }),
  };

  if (options.retryConfig?.baseDelayMs !== undefined) {
    retryOptions.baseDelayMs = options.retryConfig.baseDelayMs;
  }

  return retryOptions;
}

function serializeToolArguments(args: unknown): string {
  return typeof args === "string" ? args : JSON.stringify(args ?? {});
}

export function accumulateOpenAIStreamToolCall<
  T extends OpenAIStreamToolCallAccumulator,
>(
  toolCall: {
    index?: number;
    id?: string;
    function?: { name?: string; arguments?: string };
  },
  toolCallsByIndex: Map<number, T>,
  createAccumulator: () => T,
): void {
  const idx = toolCall.index ?? 0;
  let acc = toolCallsByIndex.get(idx);
  if (!acc) {
    acc = createAccumulator();
    toolCallsByIndex.set(idx, acc);
  }
  if (toolCall.id) acc.id = toolCall.id;
  if (toolCall.function?.name) acc.name += toolCall.function.name;
  if (toolCall.function?.arguments != null) {
    acc.argsString += toolCall.function.arguments;
  }
}

export function parseOpenAIStreamToolCallArguments(
  argsString: string,
): Record<string, unknown> {
  if (!argsString) {
    return {};
  }

  try {
    return JSON.parse(argsString) as Record<string, unknown>;
  } catch {
    return { raw: argsString };
  }
}

export function buildOpenAIStreamToolCalls<
  T extends OpenAIStreamToolCallAccumulator,
  U,
>(toolCallsByIndex: Map<number, T>, mapToolCall: (toolCall: T) => U): U[] {
  const toolCalls: U[] = [];
  const indices = [...toolCallsByIndex.keys()].sort((a, b) => a - b);

  for (const index of indices) {
    const acc = toolCallsByIndex.get(index);
    if (!acc) {
      continue;
    }

    toolCalls.push(mapToolCall(acc));
  }

  return toolCalls;
}

export async function* readSseDataLines(
  reader: ReadableStreamDefaultReader<Uint8Array>,
): AsyncIterable<string> {
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        yield line.slice(6).trim();
      }
    }
  }
}

export function parseJsonMaybe<T>(value: string): T | null {
  try {
    return JSON.parse(value) as T;
  } catch {
    return null;
  }
}

export function createOpenAIToolCall(
  toolCall: ChatToolCall,
  extra?: Record<string, unknown>,
): {
  readonly id: string;
  readonly type: "function";
  readonly function: { readonly name: string; readonly arguments: string };
} & Record<string, unknown> {
  return {
    id: toolCall.id ?? `call_${toolCall.function.name}_${Date.now()}`,
    type: "function" as const,
    function: {
      name: toolCall.function.name,
      arguments: serializeToolArguments(toolCall.function.arguments),
    },
    ...(extra ?? {}),
  };
}

export function createOpenAIToolDefinition(
  tool: ChatTool,
): OpenAIToolDefinition {
  return {
    type: "function",
    function: {
      name: tool.function.name,
      description: tool.function.description,
      parameters: (tool.function.parameters ?? {
        type: "object",
        properties: {},
      }) as object,
    },
  };
}

export function applyOpenAIMessageCore<T extends OpenAIMessageCore>(
  out: T,
  msg: ChatMessage,
): T {
  if (msg.reasoningContent !== undefined) {
    out.reasoning_content = msg.reasoningContent;
  }
  if (msg.toolCalls && msg.toolCalls.length > 0) {
    out.tool_calls = msg.toolCalls.map((toolCall) =>
      createOpenAIToolCall(toolCall),
    );
  }
  if (msg.role === "tool" && msg.toolCallId) {
    out.tool_call_id = msg.toolCallId;
  }
  return out;
}

export function createOpenAIMessageWithImages<T extends OpenAIMessageCore>(
  msg: ChatMessage,
): T {
  const role = msg.role as T["role"];
  const out = {
    role,
    content: msg.content ?? "",
  } as T;

  if (msg.role === "user" && msg.images && msg.images.length > 0) {
    out.content = buildOpenAIImageContentParts(msg.content, msg.images);
  }

  return out;
}

function imageToOpenAIUrl(image: Uint8Array | string): string {
  if (typeof image === "string") {
    if (image.startsWith("data:")) {
      return image;
    }
    if (/^https?:\/\//i.test(image)) {
      return image;
    }
    return `data:image/png;base64,${image}`;
  }

  return `data:image/png;base64,${Buffer.from(image).toString("base64")}`;
}

function buildOpenAIImageContentParts(
  content: string | undefined,
  images: Array<Uint8Array | string>,
): OpenAIMessageContentPart[] {
  const parts: OpenAIMessageContentPart[] = [];
  if (content) {
    parts.push({ type: "text", text: content });
  }
  for (const image of images) {
    parts.push({
      type: "image_url",
      image_url: {
        url: imageToOpenAIUrl(image),
      },
    });
  }
  return parts;
}

export function normalizeErrorMessage(message: string): string {
  const trimmed = message.trim();
  if (!trimmed) {
    return "";
  }

  try {
    const parsed = JSON.parse(trimmed) as {
      error?: { message?: string };
      message?: string;
      errors?: Array<{ message?: string }>;
    };
    if (typeof parsed.error?.message === "string") {
      return parsed.error.message;
    }
    if (Array.isArray(parsed.errors) && parsed.errors.length > 0) {
      const errorMessages = parsed.errors
        .map((entry) => entry.message)
        .filter((entry): entry is string => typeof entry === "string");
      if (errorMessages.length > 0) {
        return errorMessages.join("; ");
      }
    }
    if (typeof parsed.message === "string") {
      return parsed.message;
    }
  } catch {
    // Fall through to the raw response body when it is not JSON.
  }

  return trimmed.replace(/\s+/g, " ");
}

export function isContextLengthError(message: string): boolean {
  const lower = message.toLowerCase();
  return (
    lower.includes("context length") ||
    lower.includes("context window") ||
    lower.includes("maximum context") ||
    lower.includes("token limit") ||
    lower.includes("too many tokens") ||
    lower.includes("input is too long") ||
    lower.includes("prompt is too long") ||
    lower.includes("exceeds the model") ||
    lower.includes("reduce your prompt")
  );
}

export function isAbortOrTransportError(message: string): boolean {
  return (
    message.includes("AbortError") ||
    message.includes("aborted") ||
    message.includes("ECONNREFUSED") ||
    message.includes("ETIMEDOUT") ||
    message.includes("fetch failed")
  );
}

export function parseRetryAfterSeconds(
  retryAfter: string | null | undefined,
): number | undefined {
  if (!retryAfter) {
    return undefined;
  }

  const parsed = Number.parseInt(retryAfter, 10);
  return Number.isNaN(parsed) ? undefined : parsed;
}
