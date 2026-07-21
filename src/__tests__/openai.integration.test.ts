import { createProvider } from "../factory.js";
import type { OpenAiProviderConfig } from "../config.js";
import { OpenAiProvider } from "../providers/openai.js";
import type { ChatStreamEvent, ToolCallsStreamEvent } from "../types.js";
import {
  describeProviderIntegration,
  expectProviderStreamsAssistantText,
  requireEnv,
} from "./integrationHarness.js";

const MODELS = [
  { name: "GPT-5.5", key: "gpt-5.5", contextWindowTokens: 1_050_000 },
  { name: "GPT-5.4", key: "gpt-5.4", contextWindowTokens: 1_050_000 },
  {
    name: "GPT-5.4 mini",
    key: "gpt-5.4-mini",
    contextWindowTokens: 400_000,
  },
] as const;

async function collectEvents(
  provider: OpenAiProvider,
  request: Parameters<OpenAiProvider["streamChat"]>[0],
): Promise<ChatStreamEvent[]> {
  const events: ChatStreamEvent[] = [];
  for await (const event of provider.streamChat(request)) {
    events.push(event);
  }
  return events;
}

describeProviderIntegration(
  "openai",
  { env: [{ vars: "OPENAI_API_KEY" }] },
  () => {
    const config: OpenAiProviderConfig = {
      name: "openai",
      type: "openai",
      models: [...MODELS],
      defaultModel: "gpt-5.5",
      apiKey: requireEnv("OPENAI_API_KEY"),
    };

    it.each(MODELS)(
      "smoke tests $key",
      async ({ key }) => {
        const provider = createProvider(config, key);
        expect(provider).toBeInstanceOf(OpenAiProvider);
        await expectProviderStreamsAssistantText(provider, {
          model: key,
          messages: [{ role: "user", content: "Reply with exactly: OK" }],
        });
      },
      60_000,
    );

    it("continues legacy tool-call history without a Responses item id", async () => {
      const provider = createProvider(config, "gpt-5.4-mini");
      expect(provider).toBeInstanceOf(OpenAiProvider);
      const openAiProvider = provider as OpenAiProvider;
      const initialRequest = {
        model: "gpt-5.4-mini",
        messages: [
          {
            role: "user" as const,
            content:
              "Call the lookup tool exactly once with query set to ping. Do not answer without calling it.",
          },
        ],
        tools: [
          {
            type: "function" as const,
            function: {
              name: "lookup",
              description: "Looks up a query.",
              parameters: {
                type: "object",
                properties: { query: { type: "string" } },
                required: ["query"],
              },
            },
          },
        ],
      };
      const initialEvents = await collectEvents(openAiProvider, initialRequest);
      const toolCallsEvent = initialEvents.find(
        (event): event is ToolCallsStreamEvent =>
          "type" in event && event.type === "tool_calls",
      );
      expect(toolCallsEvent?.toolCalls).toHaveLength(1);
      const toolCall = toolCallsEvent?.toolCalls[0];
      expect(toolCall?.id).toBeTruthy();
      if (!toolCall?.id) {
        throw new Error("OpenAI did not return the requested tool call");
      }

      await expectProviderStreamsAssistantText(openAiProvider, {
        model: "gpt-5.4-mini",
        messages: [
          initialRequest.messages[0],
          {
            role: "assistant",
            content: "",
            toolCalls: toolCallsEvent!.toolCalls,
          },
          {
            role: "tool",
            content: '{"result":"pong"}',
            toolCallId: toolCall.id,
          },
        ],
        tools: initialRequest.tools,
      });
    }, 60_000);
  },
);
