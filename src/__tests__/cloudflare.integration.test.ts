/**
 * Cloudflare Workers AI integration tests (real API).
 * Requires CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in .env.
 */
import { createProvider } from "../factory.js";
import { CloudflareProviderConfig } from "../config.js";
import { CloudflareProvider } from "../providers/cloudflare.js";
import {
  describeProviderIntegration,
  optionalEnv,
  requireEnv,
} from "./integrationHarness.js";

const DEFAULT_MODEL =
  optionalEnv("CLOUDFLARE_MODEL") ?? "cf/moonshotai/kimi-k2.6";

describeProviderIntegration(
  "cloudflare",
  {
    env: [{ vars: "CLOUDFLARE_ACCOUNT_ID" }, { vars: "CLOUDFLARE_API_TOKEN" }],
  },
  () => {
    const cloudflareProviderConfig: CloudflareProviderConfig = {
      name: "cloudflare",
      type: "cloudflare",
      models: [
        {
          name: "Kimi K2.6",
          key: DEFAULT_MODEL,
          contextWindowTokens: 262_144,
        },
      ],
      defaultModel: DEFAULT_MODEL,
      accountId: requireEnv("CLOUDFLARE_ACCOUNT_ID"),
      apiKey: requireEnv("CLOUDFLARE_API_TOKEN"),
    };

    it("should create CloudflareProvider and stream assistant text", async () => {
      const provider = createProvider(cloudflareProviderConfig);
      expect(provider).toBeInstanceOf(CloudflareProvider);
      expect(provider.name).toBe("cloudflare");

      const assistantText: string[] = [];
      let terminalStopReason: string | undefined;

      for await (const chunk of provider.streamChat({
        model: DEFAULT_MODEL,
        messages: [{ role: "user", content: "Reply with exactly: OK" }],
      })) {
        if (chunk.type === "assistant_text") {
          assistantText.push(chunk.delta);
        }
        if (chunk.type === "terminal") {
          terminalStopReason = chunk.stopReason;
        }
      }

      expect(assistantText.join("").length).toBeGreaterThan(0);
      expect(terminalStopReason).toBeDefined();
    }, 30_000);
  },
);
