import { createProvider } from "../factory.js";
import type { OpenAiProviderConfig } from "../config.js";
import { OpenAiProvider } from "../providers/openai.js";
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
  },
);
