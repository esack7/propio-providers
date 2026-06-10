import {
  Model,
  ProviderConfig,
  OllamaProviderConfig,
  BedrockProviderConfig,
  OpenRouterProviderConfig,
  GeminiProviderConfig,
  CloudflareProviderConfig,
  AnthropicProviderConfig,
  ProvidersConfig,
} from "../config.js";

describe("Configuration Types (New Structure)", () => {
  describe("Model interface", () => {
    it("should have name and key fields", () => {
      const model: Model = {
        name: "Llama 3.2 3B",
        key: "llama3.2:3b",
        contextWindowTokens: 128_000,
      };
      expect(model.name).toBe("Llama 3.2 3B");
      expect(model.key).toBe("llama3.2:3b");
    });
  });

  describe("OllamaProviderConfig", () => {
    it("should define ollama provider with flat structure", () => {
      const config: OllamaProviderConfig = {
        name: "local-ollama",
        type: "ollama",
        models: [
          {
            name: "Llama 3.2 3B",
            key: "llama3.2:3b",
            contextWindowTokens: 128_000,
          },
          {
            name: "Llama 3.2 90B",
            key: "llama3.2:90b",
            contextWindowTokens: 128_000,
          },
        ],
        defaultModel: "llama3.2:3b",
        host: "http://localhost:11434",
      };
      expect(config.name).toBe("local-ollama");
      expect(config.type).toBe("ollama");
      expect(config.host).toBe("http://localhost:11434");
      expect(config.defaultModel).toBe("llama3.2:3b");
      expect(config.models).toHaveLength(2);
      expect(config.models[0].key).toBe("llama3.2:3b");
    });

    it("should have optional host field", () => {
      const config: OllamaProviderConfig = {
        name: "local-ollama",
        type: "ollama",
        models: [
          { name: "Llama 3.2", key: "llama3.2", contextWindowTokens: 128_000 },
        ],
        defaultModel: "llama3.2",
      };
      expect(config.host).toBeUndefined();
    });
  });

  describe("BedrockProviderConfig", () => {
    it("should define bedrock provider with flat structure", () => {
      const config: BedrockProviderConfig = {
        name: "bedrock-provider",
        type: "bedrock",
        models: [
          {
            name: "Claude 3.5 Sonnet",
            key: "anthropic.claude-3-5-sonnet-20241022-v2:0",
            contextWindowTokens: 128_000,
          },
        ],
        defaultModel: "anthropic.claude-3-5-sonnet-20241022-v2:0",
        region: "us-west-2",
      };
      expect(config.name).toBe("bedrock-provider");
      expect(config.type).toBe("bedrock");
      expect(config.region).toBe("us-west-2");
      expect(config.defaultModel).toBe(
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
      );
      expect(config.models).toHaveLength(1);
    });

    it("should have optional region field", () => {
      const config: BedrockProviderConfig = {
        name: "bedrock-provider",
        type: "bedrock",
        models: [
          {
            name: "Claude 3.5 Sonnet",
            key: "anthropic.claude-3-5-sonnet-20241022-v2:0",
            contextWindowTokens: 128_000,
          },
        ],
        defaultModel: "anthropic.claude-3-5-sonnet-20241022-v2:0",
      };
      expect(config.region).toBeUndefined();
    });
  });

  describe("ProviderConfig union type", () => {
    it("should accept OllamaProviderConfig", () => {
      const config: ProviderConfig = {
        name: "ollama",
        type: "ollama",
        models: [{ name: "Model", key: "model", contextWindowTokens: 128_000 }],
        defaultModel: "model",
      };
      expect(config.type).toBe("ollama");
    });

    it("should accept BedrockProviderConfig", () => {
      const config: ProviderConfig = {
        name: "bedrock",
        type: "bedrock",
        models: [{ name: "Model", key: "model", contextWindowTokens: 128_000 }],
        defaultModel: "model",
      };
      expect(config.type).toBe("bedrock");
    });

    it("should accept OpenRouterProviderConfig", () => {
      const config: ProviderConfig = {
        name: "openrouter",
        type: "openrouter",
        models: [
          {
            name: "GPT-3.5",
            key: "openai/gpt-3.5-turbo",
            contextWindowTokens: 128_000,
          },
        ],
        defaultModel: "openai/gpt-3.5-turbo",
        apiKey: "sk-key",
      };
      expect(config.type).toBe("openrouter");
    });

    it("should accept GeminiProviderConfig", () => {
      const config: ProviderConfig = {
        name: "gemini",
        type: "gemini",
        models: [
          {
            name: "Gemini 3 Flash",
            key: "gemini-3-flash-preview",
            contextWindowTokens: 128_000,
          },
        ],
        defaultModel: "gemini-3-flash-preview",
        apiKey: "gemini-key",
      };
      expect(config.type).toBe("gemini");
    });

    it("should accept CloudflareProviderConfig", () => {
      const config: ProviderConfig = {
        name: "cloudflare",
        type: "cloudflare",
        models: [
          {
            name: "Kimi K2.6",
            key: "cf/moonshotai/kimi-k2.6",
            contextWindowTokens: 262_144,
          },
        ],
        defaultModel: "cf/moonshotai/kimi-k2.6",
        apiKey: "cf-token",
        accountId: "account-id",
      };
      expect(config.type).toBe("cloudflare");
    });

    it("should accept AnthropicProviderConfig", () => {
      const config: ProviderConfig = {
        name: "anthropic",
        type: "anthropic",
        models: [
          {
            name: "Claude Sonnet",
            key: "claude-sonnet-4-6",
            contextWindowTokens: 200_000,
          },
        ],
        defaultModel: "claude-sonnet-4-6",
        apiKey: "anthropic-key",
      } as AnthropicProviderConfig;
      expect(config.type).toBe("anthropic");
    });
  });

  describe("OpenRouterProviderConfig", () => {
    it("should define openrouter provider with optional apiKey, httpReferer, xTitle, and routing fields", () => {
      const config: OpenRouterProviderConfig = {
        name: "openrouter",
        type: "openrouter",
        models: [
          {
            name: "GPT-3.5",
            key: "openai/gpt-3.5-turbo",
            contextWindowTokens: 128_000,
          },
        ],
        defaultModel: "openai/gpt-3.5-turbo",
        apiKey: "sk-key",
        httpReferer: "https://app.com",
        xTitle: "My App",
        provider: {
          allowFallbacks: true,
          order: ["provider-a", "provider-b"],
          requireParameters: false,
        },
        fallbackModels: ["openai/gpt-4o", "openai/gpt-4.1"],
        debugEchoUpstreamBody: true,
      };
      expect(config.type).toBe("openrouter");
      expect(config.apiKey).toBe("sk-key");
      expect(config.httpReferer).toBe("https://app.com");
      expect(config.xTitle).toBe("My App");
      expect(config.provider?.allowFallbacks).toBe(true);
      expect(config.provider?.order).toEqual(["provider-a", "provider-b"]);
      expect(config.fallbackModels).toEqual([
        "openai/gpt-4o",
        "openai/gpt-4.1",
      ]);
      expect(config.debugEchoUpstreamBody).toBe(true);
    });
  });

  describe("GeminiProviderConfig", () => {
    it("should define gemini provider with optional apiKey", () => {
      const config: GeminiProviderConfig = {
        name: "gemini",
        type: "gemini",
        models: [
          {
            name: "Gemini 3.1 Pro Preview",
            key: "gemini-3.1-pro-preview",
            contextWindowTokens: 128_000,
          },
        ],
        defaultModel: "gemini-3.1-pro-preview",
        apiKey: "gemini-key",
      };
      expect(config.type).toBe("gemini");
      expect(config.apiKey).toBe("gemini-key");
    });
  });

  describe("CloudflareProviderConfig", () => {
    it("should define cloudflare provider with optional apiKey and accountId", () => {
      const config: CloudflareProviderConfig = {
        name: "cloudflare",
        type: "cloudflare",
        models: [
          {
            name: "Kimi K2.6",
            key: "cf/moonshotai/kimi-k2.6",
            contextWindowTokens: 262_144,
          },
        ],
        defaultModel: "cf/moonshotai/kimi-k2.6",
        apiKey: "cf-token",
        accountId: "account-id",
      };
      expect(config.type).toBe("cloudflare");
      expect(config.apiKey).toBe("cf-token");
      expect(config.accountId).toBe("account-id");
    });
  });

  describe("ProvidersConfig", () => {
    it("should contain multiple providers and default", () => {
      const config: ProvidersConfig = {
        default: "local-ollama",
        providers: [
          {
            name: "local-ollama",
            type: "ollama",
            models: [
              {
                name: "Llama 3.2",
                key: "llama3.2",
                contextWindowTokens: 128_000,
              },
            ],
            defaultModel: "llama3.2",
            host: "http://localhost:11434",
          },
          {
            name: "bedrock",
            type: "bedrock",
            models: [
              {
                name: "Claude 3.5",
                key: "anthropic.claude-3-5-sonnet-20241022-v2:0",
                contextWindowTokens: 128_000,
              },
            ],
            defaultModel: "anthropic.claude-3-5-sonnet-20241022-v2:0",
            region: "us-west-2",
          },
          {
            name: "anthropic",
            type: "anthropic",
            models: [
              {
                name: "Claude Sonnet",
                key: "claude-sonnet-4-6",
                contextWindowTokens: 200_000,
              },
            ],
            defaultModel: "claude-sonnet-4-6",
            apiKey: "anthropic-key",
          },
        ],
      };
      expect(config.default).toBe("local-ollama");
      expect(config.providers).toHaveLength(3);
      expect(config.providers[0].type).toBe("ollama");
      expect(config.providers[1].type).toBe("bedrock");
      expect(config.providers[2].type).toBe("anthropic");
    });

    it("should support single provider in config", () => {
      const config: ProvidersConfig = {
        default: "ollama",
        providers: [
          {
            name: "ollama",
            type: "ollama",
            models: [
              { name: "Model", key: "model", contextWindowTokens: 128_000 },
            ],
            defaultModel: "model",
          },
        ],
      };
      expect(config.providers).toHaveLength(1);
    });
  });
});
