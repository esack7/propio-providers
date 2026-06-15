/**
 * Anthropic live integration tests.
 * Requires ANTHROPIC_API_KEY in .env.
 */
import { createProvider } from "../factory.js";
import { AnthropicProviderConfig } from "../config.js";
import { AnthropicProvider } from "../providers/anthropic.js";
import {
  describeProviderIntegration,
  expectProviderStreamsAssistantText,
  optionalEnv,
  requireEnv,
} from "./integrationHarness.js";

const SONNET_MODEL =
  optionalEnv("ANTHROPIC_SONNET_MODEL") ?? "claude-sonnet-4-6";
const HAIKU_MODEL =
  optionalEnv("ANTHROPIC_HAIKU_MODEL") ?? "claude-haiku-4-5-20251001";
const OPUS_MODEL = optionalEnv("ANTHROPIC_OPUS_MODEL") ?? "claude-opus-4-8";
const FABLE_MODEL = optionalEnv("ANTHROPIC_FABLE_MODEL") ?? "claude-fable-5";

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
        {
          name: "Claude Opus 4.8",
          key: OPUS_MODEL,
          contextWindowTokens: 1_000_000,
        },
        {
          name: "Claude Fable 5",
          key: FABLE_MODEL,
          contextWindowTokens: 1_000_000,
        },
      ],
      defaultModel: SONNET_MODEL,
      apiKey: requireEnv("ANTHROPIC_API_KEY"),
    };

    async function smokeTestModel(
      modelKey: string,
      options: { requestReasoning?: boolean } = {},
    ): Promise<void> {
      const provider = createProvider(anthropicProviderConfig, modelKey);
      expect(provider).toBeInstanceOf(AnthropicProvider);
      expect(provider.name).toBe("anthropic");

      await expectProviderStreamsAssistantText(provider, {
        model: modelKey,
        messages: [{ role: "user", content: "Reply with exactly: OK" }],
        ...(options.requestReasoning ? { requestReasoning: true } : {}),
      });
    }

    it("should smoke test Claude Sonnet", async () => {
      await smokeTestModel(SONNET_MODEL);
    }, 30_000);

    it("should smoke test Claude Haiku", async () => {
      await smokeTestModel(HAIKU_MODEL);
    }, 30_000);

    it("should smoke test Claude Opus reasoning", async () => {
      await smokeTestModel(OPUS_MODEL, { requestReasoning: true });
    }, 30_000);

    it.skip("should smoke test Claude Fable", async () => {
      await smokeTestModel(FABLE_MODEL);
    }, 30_000);
  },
);
