import { createProvider } from "../factory.js";
import type { MetaProviderConfig } from "../config.js";
import type { LLMProvider } from "../interface.js";
import { MetaProvider } from "../providers/meta.js";
import type {
  ChatMessage,
  ChatRequest,
  ChatTool,
  StopReason,
  ToolCallsStreamEvent,
} from "../types.js";
import {
  describeProviderIntegration,
  expectProviderStreamsAssistantText,
  requireEnv,
} from "./integrationHarness.js";

const MODEL = {
  name: "Muse Spark 1.1",
  key: "muse-spark-1.1",
  contextWindowTokens: 1_048_576,
} as const;

const REQUIRED_VALUE_TOOL: ChatTool = {
  type: "function",
  function: {
    name: "required_value",
    description: "Returns the required value for a supplied key.",
    parameters: {
      type: "object",
      properties: { key: { type: "string" } },
      required: ["key"],
    },
  },
};

interface MetaStreamTrace {
  assistantText: string;
  toolEvent?: ToolCallsStreamEvent;
  terminalReason?: StopReason;
}

async function collectMetaStream(
  provider: LLMProvider,
  request: ChatRequest,
): Promise<MetaStreamTrace> {
  const trace: MetaStreamTrace = { assistantText: "" };
  for await (const event of provider.streamChat(request)) {
    if (!("type" in event)) {
      continue;
    }
    if (event.type === "assistant_text") {
      trace.assistantText += event.delta;
    } else if (event.type === "tool_calls") {
      trace.toolEvent = event;
    } else if (event.type === "terminal") {
      trace.terminalReason = event.stopReason;
    }
  }
  return trace;
}

function expectValidReplayState(toolEvent: ToolCallsStreamEvent): void {
  const replayItems = JSON.parse(toolEvent.reasoningContent ?? "[]") as Array<
    Record<string, unknown>
  >;
  const reasoningIndex = replayItems.findIndex(
    (item) => item.type === "reasoning",
  );
  const functionCallIndex = replayItems.findIndex(
    (item) => item.type === "function_call",
  );
  expect(reasoningIndex).toBeGreaterThanOrEqual(0);
  expect(functionCallIndex).toBeGreaterThan(reasoningIndex);
  expect(
    replayItems.some(
      (item) =>
        item.type === "reasoning" &&
        typeof item.encrypted_content === "string" &&
        item.encrypted_content.length > 0,
    ),
  ).toBe(true);
  for (const item of replayItems.filter(
    (candidate) => candidate.type === "message",
  )) {
    expect(item.phase).toBe("commentary");
  }
}

describeProviderIntegration("meta", { env: [{ vars: "META_API_KEY" }] }, () => {
  const config: MetaProviderConfig = {
    name: "meta",
    type: "meta",
    models: [MODEL],
    defaultModel: MODEL.key,
    apiKey: requireEnv("META_API_KEY"),
  };

  it("smoke tests assistant text", async () => {
    const provider = createProvider(config);
    expect(provider).toBeInstanceOf(MetaProvider);
    await expectProviderStreamsAssistantText(provider, {
      model: MODEL.key,
      messages: [{ role: "user", content: "Reply with exactly: OK" }],
    });
  }, 120_000);

  it("streams encrypted reasoning, tools, replay state, and a final response", async () => {
    const provider = createProvider(config);
    const messages: ChatMessage[] = [
      {
        role: "user",
        content:
          "Call the required_value tool with key meta-test. Do not answer before calling it. In the final answer, include the returned value exactly.",
      },
    ];
    const first = await collectMetaStream(provider, {
      model: MODEL.key,
      requestReasoning: true,
      messages,
      tools: [REQUIRED_VALUE_TOOL],
    });

    expect(first.toolEvent?.toolCalls).toHaveLength(1);
    const toolEvent = first.toolEvent as ToolCallsStreamEvent;
    const toolCall = toolEvent.toolCalls[0];
    expect(toolCall.function.name).toBe("required_value");
    expect(first.terminalReason).toBe("tool_use");
    expectValidReplayState(toolEvent);

    const second = await collectMetaStream(provider, {
      model: MODEL.key,
      requestReasoning: true,
      tools: [REQUIRED_VALUE_TOOL],
      messages: [
        ...messages,
        {
          role: "assistant",
          content: first.assistantText,
          reasoningContent: toolEvent.reasoningContent,
          toolCalls: toolEvent.toolCalls,
        },
        {
          role: "tool",
          content: "META_TOOL_ROUND_OK",
          toolCallId: toolCall.id,
        },
      ],
    });

    expect(second.assistantText).toContain("META_TOOL_ROUND_OK");
    expect(second.terminalReason).toBe("end_turn");
  }, 120_000);
});
