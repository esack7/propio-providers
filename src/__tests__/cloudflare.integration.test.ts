/**
 * Cloudflare Workers AI integration tests (real API).
 * Run when CLOUDFLARE_ACCOUNT_ID and an API token env var are set, or when a
 * cloudflare provider with credentials exists in .propio/providers.json.
 * Skipped otherwise.
 */
import * as path from "path";
import * as fs from "fs";
import { createProvider } from "../factory.js";
import { CloudflareProvider } from "../providers/cloudflare.js";
import { CloudflareProviderConfig } from "../config.js";

interface CloudflareCredentials {
  accountId?: string;
  apiKey?: string;
}

function resolveCloudflareApiKeyFromEnv(): string | undefined {
  return (
    process.env.CLOUDFLARE_API_TOKEN ??
    process.env.CLOUDFLARE_AUTH_TOKEN ??
    process.env.CLOUDFLARE_API_KEY
  );
}

function getCloudflareCredentialsFromConfig(): CloudflareCredentials {
  const configPath = path.join(process.cwd(), ".propio", "providers.json");
  try {
    if (!fs.existsSync(configPath)) {
      return {};
    }
    const config = JSON.parse(fs.readFileSync(configPath, "utf-8"));
    const cloudflare = config.providers?.find(
      (provider: { type?: string }) => provider.type === "cloudflare",
    );
    if (!cloudflare) {
      return {};
    }
    return {
      accountId: cloudflare.accountId,
      apiKey: cloudflare.apiKey,
    };
  } catch {
    return {};
  }
}

function getCloudflareCredentials(): CloudflareCredentials {
  const fromConfig = getCloudflareCredentialsFromConfig();
  return {
    accountId: process.env.CLOUDFLARE_ACCOUNT_ID ?? fromConfig.accountId,
    apiKey: resolveCloudflareApiKeyFromEnv() ?? fromConfig.apiKey,
  };
}

const credentials = getCloudflareCredentials();
const hasCredentials = Boolean(credentials.accountId && credentials.apiKey);
const itIntegration = hasCredentials ? it : it.skip;

const DEFAULT_MODEL = "cf/moonshotai/kimi-k2.6";

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
  ...(credentials.accountId ? { accountId: credentials.accountId } : {}),
  ...(credentials.apiKey ? { apiKey: credentials.apiKey } : {}),
};

describe("Cloudflare integration (real API)", () => {
  if (!hasCredentials) {
    it("skipped when CLOUDFLARE_ACCOUNT_ID and API token are not set", () => {});
    return;
  }

  describe("factory", () => {
    itIntegration("should create CloudflareProvider via factory", () => {
      const provider = createProvider(cloudflareProviderConfig);
      expect(provider).toBeInstanceOf(CloudflareProvider);
      expect(provider.name).toBe("cloudflare");
    });
  });

  describe("streaming chat", () => {
    itIntegration(
      "should stream chat and yield assistant text",
      async () => {
        const provider = createProvider(cloudflareProviderConfig);
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
      },
      30000,
    );
  });
});
