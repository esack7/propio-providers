/**
 * Anthropic live integration tests.
 * Requires ANTHROPIC_API_KEY in .env.
 */
import { createProvider } from "../factory.js";
import { AnthropicProviderConfig } from "../config.js";
import { AnthropicProvider } from "../providers/anthropic.js";
import {
  describeProviderIntegration,
  optionalEnv,
  requireEnv,
} from "./integrationHarness.js";

const SONNET_MODEL =
  optionalEnv("ANTHROPIC_SONNET_MODEL") ?? "claude-sonnet-4-6";
const HAIKU_MODEL =
  optionalEnv("ANTHROPIC_HAIKU_MODEL") ?? "claude-haiku-4-5-20251001";

describeProviderIntegration(
  "anthropic",
  {
    env: [{ vars: "ANTHROPIC_API_KEY" }],
  },
  () => {
    const anthropicProviderConfig: AnthropicProviderConfig = {
      name: "anthropic",
      type: "anthropic",
      models: [
        {
          name: "Claude Sonnet 4.6",
          key: SONNET_MODEL,
          contextWindowTokens: 1_000_000,
        },
        {
          name: "Claude Haiku 4.5",
          key: HAIKU_MODEL,
          contextWindowTokens: 200_000,
        },
      ],
      defaultModel: SONNET_MODEL,
      apiKey: requireEnv("ANTHROPIC_API_KEY"),
    };

    async function smokeTestModel(modelKey: string): Promise<void> {
      const provider = createProvider(anthropicProviderConfig, modelKey);
      expect(provider).toBeInstanceOf(AnthropicProvider);
      expect(provider.name).toBe("anthropic");

      const assistantText: string[] = [];
      let terminalStopReason: string | undefined;

      for await (const chunk of provider.streamChat({
        model: modelKey,
        messages: [{ role: "user", content: "Reply with exactly: OK" }],
      })) {
        if (chunk.type === "assistant_text") {
          assistantText.push(chunk.delta);
        }
        if (chunk.type === "terminal") {
          terminalStopReason = chunk.stopReason;
        }
      }

      expect(assistantText.join("")).not.toHaveLength(0);
      expect(terminalStopReason).toBeDefined();
    }

    it("should smoke test Claude Sonnet", async () => {
      await smokeTestModel(SONNET_MODEL);
    }, 30_000);

    it("should smoke test Claude Haiku", async () => {
      await smokeTestModel(HAIKU_MODEL);
    }, 30_000);
  },
);
