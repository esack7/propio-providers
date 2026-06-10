import type { WithRetryOptions } from "./withRetry.js";
import { withRetry } from "./withRetry.js";
import type { ProviderError } from "../types.js";
import {
  accumulateOpenAIStreamToolCall,
  buildOpenAIStreamToolCalls,
  parseJsonMaybe,
  parseOpenAIStreamToolCallArguments,
  readSseDataLines,
  type OpenAIStreamToolCallAccumulator,
} from "./shared.js";
import type { ChatStreamEvent, StopReason } from "../types.js";

interface OpenAIChatCompletionsStreamToolCallDelta {
  index?: number;
  id?: string;
  function?: { name?: string; arguments?: string };
}

interface OpenAIChatCompletionsStreamChoice {
  delta?: {
    content?: string;
    tool_calls?: OpenAIChatCompletionsStreamToolCallDelta[];
  };
  finish_reason?: string;
}

function mapStandardOpenAiFinishReason(finishReason: string): string {
  const finishReasonMap: Record<string, string> = {
    length: "max_tokens",
    stop: "end_turn",
    tool_calls: "tool_use",
  };
  return finishReasonMap[finishReason] ?? "end_turn";
}

function parseOpenAiChatCompletionsStreamChoice(
  data: string,
): OpenAIChatCompletionsStreamChoice | undefined {
  return parseJsonMaybe<{ choices?: OpenAIChatCompletionsStreamChoice[] }>(data)
    ?.choices?.[0];
}

function buildToolCallsEventFromFinishReason(
  finishReason: string | undefined,
  toolCallsByIndex: Map<number, OpenAIStreamToolCallAccumulator>,
): ChatStreamEvent | null {
  if (finishReason !== "tool_calls") {
    return null;
  }

  const toolCalls = buildOpenAIStreamToolCalls(toolCallsByIndex, (acc) => ({
    id: acc.id,
    function: {
      name: acc.name || "",
      arguments: parseOpenAIStreamToolCallArguments(acc.argsString),
    },
  }));
  return toolCalls.length > 0 ? { type: "tool_calls", toolCalls } : null;
}

function accumulateOpenAiChatCompletionsToolCalls(
  toolCalls: OpenAIChatCompletionsStreamToolCallDelta[] | undefined,
  toolCallsByIndex: Map<number, OpenAIStreamToolCallAccumulator>,
): void {
  if (!toolCalls || !Array.isArray(toolCalls)) {
    return;
  }

  for (const toolCall of toolCalls) {
    accumulateOpenAIStreamToolCall(toolCall, toolCallsByIndex, () => ({
      name: "",
      argsString: "",
    }));
  }
}

function parseOpenAiChatCompletionsStreamLine(
  data: string,
  toolCallsByIndex: Map<number, OpenAIStreamToolCallAccumulator>,
): { events: ChatStreamEvent[]; stopReason?: string } {
  if (data === "[DONE]") {
    return { events: [] };
  }

  const choice = parseOpenAiChatCompletionsStreamChoice(data);
  if (!choice?.delta) {
    return { events: [] };
  }

  const events: ChatStreamEvent[] = [];
  const content = choice.delta.content;
  if (content != null && content !== "") {
    events.push({ type: "assistant_text", delta: content });
  }
  accumulateOpenAiChatCompletionsToolCalls(
    choice.delta.tool_calls,
    toolCallsByIndex,
  );

  const toolCallsEvent = buildToolCallsEventFromFinishReason(
    choice.finish_reason,
    toolCallsByIndex,
  );
  if (toolCallsEvent) {
    events.push(toolCallsEvent);
  }

  return {
    events,
    stopReason: choice.finish_reason
      ? mapStandardOpenAiFinishReason(choice.finish_reason)
      : undefined,
  };
}

export async function* consumeOpenAiChatCompletionsStream(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  toolCallsByIndex: Map<number, OpenAIStreamToolCallAccumulator>,
): AsyncIterable<ChatStreamEvent> {
  let stopReason: StopReason = "end_turn";

  for await (const data of readSseDataLines(reader)) {
    const result = parseOpenAiChatCompletionsStreamLine(data, toolCallsByIndex);
    if (result.stopReason) {
      stopReason = result.stopReason as StopReason;
    }
    for (const event of result.events) {
      yield event;
    }
  }

  yield { type: "terminal", stopReason };
}

export async function fetchOpenAiCompatibleStreamReader(options: {
  body: Record<string, unknown>;
  signal: AbortSignal | undefined;
  fetchStream: (
    body: Record<string, unknown>,
    signal: AbortSignal | undefined,
  ) => Promise<Response>;
  retryOptions: WithRetryOptions;
  translateError: (error: unknown) => ProviderError;
}): Promise<ReadableStreamDefaultReader<Uint8Array>> {
  const response = await withRetry(
    () => options.fetchStream(options.body, options.signal),
    options.retryOptions,
  );
  const reader = response.body?.getReader();
  if (!reader) {
    throw options.translateError(new Error("No response body"));
  }
  return reader;
}
