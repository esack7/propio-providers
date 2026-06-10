import { createProvider, extractModelFromConfig } from "../factory.js";
import { LLMProvider } from "../interface.js";
import {
  ProviderConfig,
  OllamaProviderConfig,
  BedrockProviderConfig,
  OpenRouterProviderConfig,
  GeminiProviderConfig,
  XaiProviderConfig,
  CloudflareProviderConfig,
  AnthropicProviderConfig,
} from "../config.js";
import { OllamaProvider } from "../providers/ollama.js";
import { BedrockProvider } from "../providers/bedrock.js";
import { OpenRouterProvider } from "../providers/openrouter.js";
import { GeminiProvider } from "../providers/gemini.js";
import { XaiProvider } from "../providers/xai.js";
import { CloudflareProvider } from "../providers/cloudflare.js";
import { AnthropicProvider } from "../providers/anthropic.js";

describe("Provider Factory", () => {
  describe("createProvider", () => {
    it("should create OllamaProvider from new config shape", () => {
      const config: OllamaProviderConfig = {
        name: "local-ollama",
        type: "ollama",
        models: [{ name: "Llama", key: "llama3.2", contextWindowTokens: 8192 }],
        defaultModel: "llama3.2",
        host: "http://localhost:11434",
      };

      const provider = createProvider(config);

      expect(provider).toBeInstanceOf(OllamaProvider);
      expect(provider.name).toBe("ollama");
    });

    it("should create BedrockProvider from new config shape", () => {
      const config: BedrockProviderConfig = {
        name: "bedrock",
        type: "bedrock",
        models: [
          {
            name: "Claude",
            key: "anthropic.claude-3-5-sonnet-20241022-v2:0",
            contextWindowTokens: 200_000,
          },
        ],
        defaultModel: "anthropic.claude-3-5-sonnet-20241022-v2:0",
        region: "us-west-2",
      };

      const provider = createProvider(config);

      expect(provider).toBeInstanceOf(BedrockProvider);
      expect(provider.name).toBe("bedrock");
    });

    it("should create OpenRouterProvider from OpenRouter config", () => {
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
        apiKey: "sk-test-key",
      };

      const provider = createProvider(config);

      expect(provider).toBeInstanceOf(OpenRouterProvider);
      expect(provider.name).toBe("openrouter");
    });

    it("should create GeminiProvider from Gemini config", () => {
      const config: GeminiProviderConfig = {
        name: "gemini",
        type: "gemini",
        models: [
          {
            name: "Gemini 3 Flash",
            key: "gemini-3-flash-preview",
            contextWindowTokens: 1_048_576,
          },
        ],
        defaultModel: "gemini-3-flash-preview",
        apiKey: "gemini-test-key",
      };

      const provider = createProvider(config);

      expect(provider).toBeInstanceOf(GeminiProvider);
      expect(provider.name).toBe("gemini");
    });

    it("should accept modelKey parameter and use it instead of defaultModel", () => {
      const config: OllamaProviderConfig = {
        name: "local-ollama",
        type: "ollama",
        models: [
          { name: "Llama 3B", key: "llama3.2:3b", contextWindowTokens: 8192 },
          {
            name: "Llama 90B",
            key: "llama3.2:90b",
            contextWindowTokens: 8192,
          },
        ],
        defaultModel: "llama3.2:3b",
        host: "http://localhost:11434",
      };

      const provider = createProvider(config, "llama3.2:90b");

      expect(provider).toBeInstanceOf(OllamaProvider);
    });

    it("should return LLMProvider interface type", () => {
      const config: OllamaProviderConfig = {
        name: "ollama",
        type: "ollama",
        models: [{ name: "Llama", key: "llama3.2", contextWindowTokens: 8192 }],
        defaultModel: "llama3.2",
      };

      const provider = createProvider(config);

      expect(provider).toBeDefined();
      expect(typeof provider.streamChat).toBe("function");
      expect(provider.name).toBeDefined();
    });

    it("should throw error for unknown provider type", () => {
      const config: any = {
        name: "test",
        type: "unknown",
        models: [{ name: "Test", key: "test", contextWindowTokens: 128_000 }],
        defaultModel: "test",
      };

      expect(() => createProvider(config)).toThrow();
    });

    it("should include valid providers in error message", () => {
      const config: any = {
        name: "test",
        type: "unknown",
        models: [{ name: "Test", key: "test", contextWindowTokens: 128_000 }],
        defaultModel: "test",
      };

      expect(() => createProvider(config)).toThrow(
        /ollama.*bedrock|bedrock.*ollama|openrouter|gemini|xai|cloudflare/,
      );
    });

    it("should create XaiProvider from xai config", () => {
      const config: XaiProviderConfig = {
        name: "xai",
        type: "xai",
        models: [
          {
            name: "Grok Fast",
            key: "grok-4-1-fast-reasoning",
            contextWindowTokens: 2_000_000,
          },
        ],
        defaultModel: "grok-4-1-fast-reasoning",
        apiKey: "xai-test-key",
      };

      const provider = createProvider(config);

      expect(provider).toBeInstanceOf(XaiProvider);
      expect(provider.name).toBe("xai");
    });

    it("should use configured context windows for newly added xAI models", () => {
      const config: XaiProviderConfig = {
        name: "xai",
        type: "xai",
        models: [
          {
            name: "Grok 4.3",
            key: "grok-4.3",
            contextWindowTokens: 1_000_000,
          },
        ],
        defaultModel: "grok-4.3",
        apiKey: "xai-test-key",
      };

      const provider = createProvider(config);

      expect(provider).toBeInstanceOf(XaiProvider);
      expect(provider.getCapabilities().contextWindowTokens).toBe(1_000_000);
    });

    it("should create arbitrary configured Gemini models", () => {
      const config: GeminiProviderConfig = {
        name: "gemini",
        type: "gemini",
        models: [
          {
            name: "Gemini Future Preview",
            key: "gemini-future-preview",
            contextWindowTokens: 2_000_000,
          },
        ],
        defaultModel: "gemini-future-preview",
        apiKey: "gemini-test-key",
      };

      const provider = createProvider(config);

      expect(provider).toBeInstanceOf(GeminiProvider);
      expect(provider.getCapabilities().contextWindowTokens).toBe(2_000_000);
    });

    it("should create CloudflareProvider from cloudflare config", () => {
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
        apiKey: "cf-test-token",
        accountId: "test-account-id",
      };

      const provider = createProvider(config);

      expect(provider).toBeInstanceOf(CloudflareProvider);
      expect(provider.name).toBe("cloudflare");
      expect(provider.getCapabilities().contextWindowTokens).toBe(262_144);
    });

    it("should create AnthropicProvider from anthropic config", () => {
      const config: AnthropicProviderConfig = {
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
        apiKey: "anthropic-test-key",
      };

      const provider = createProvider(config);

      expect(provider).toBeInstanceOf(AnthropicProvider);
      expect(provider.name).toBe("anthropic");
      expect(provider.getCapabilities().contextWindowTokens).toBe(200_000);
    });

    it("should use flat host field for Ollama", () => {
      const config: OllamaProviderConfig = {
        name: "ollama",
        type: "ollama",
        models: [{ name: "Llama", key: "llama3.2", contextWindowTokens: 8192 }],
        defaultModel: "llama3.2",
        host: "http://custom.host:11434",
      };

      const provider = createProvider(config);
      expect(provider).toBeInstanceOf(OllamaProvider);
    });

    it("should use flat region field for Bedrock", () => {
      const config: BedrockProviderConfig = {
        name: "bedrock",
        type: "bedrock",
        models: [
          {
            name: "Claude",
            key: "anthropic.claude-3-5-sonnet-20241022-v2:0",
            contextWindowTokens: 200_000,
          },
        ],
        defaultModel: "anthropic.claude-3-5-sonnet-20241022-v2:0",
        region: "eu-west-1",
      };

      const provider = createProvider(config);
      expect(provider).toBeInstanceOf(BedrockProvider);
    });
  });

  describe("extractModelFromConfig", () => {
    it("should return defaultModel from Ollama config", () => {
      const config: OllamaProviderConfig = {
        name: "ollama",
        type: "ollama",
        models: [{ name: "Llama", key: "llama3.2", contextWindowTokens: 8192 }],
        defaultModel: "llama3.2",
      };

      const model = extractModelFromConfig(config);

      expect(model).toBe("llama3.2");
    });

    it("should return defaultModel from Bedrock config", () => {
      const config: BedrockProviderConfig = {
        name: "bedrock",
        type: "bedrock",
        models: [
          {
            name: "Claude",
            key: "anthropic.claude-3-5-sonnet-20241022-v2:0",
            contextWindowTokens: 200_000,
          },
        ],
        defaultModel: "anthropic.claude-3-5-sonnet-20241022-v2:0",
      };

      const model = extractModelFromConfig(config);

      expect(model).toBe("anthropic.claude-3-5-sonnet-20241022-v2:0");
    });

    it("should work for any provider type", () => {
      const config: ProviderConfig = {
        name: "test",
        type: "ollama",
        models: [
          { name: "Test", key: "test-model", contextWindowTokens: 128_000 },
        ],
        defaultModel: "test-model",
      };

      const model = extractModelFromConfig(config);

      expect(model).toBe("test-model");
    });
  });
});
