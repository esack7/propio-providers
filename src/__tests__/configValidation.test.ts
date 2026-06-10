import {
  getDefaultProviderModelSelection,
  resolveProvider,
  resolveModelKey,
  updateDefaultProviderModelSelection,
  validateProvidersConfig,
} from "../configValidation.js";
import { ProvidersConfig } from "../config.js";

describe("Configuration Validation", () => {
  function expectErrorMessage(
    action: () => void,
    expectedPatterns: RegExp[],
  ): void {
    try {
      action();
      fail("Expected error to be thrown");
    } catch (error) {
      if (!(error instanceof Error)) {
        throw error;
      }
      for (const pattern of expectedPatterns) {
        expect(error.message).toMatch(pattern);
      }
    }
  }

  describe("validateProvidersConfig()", () => {
    it("should reject a non-object configuration root", () => {
      expect(() => validateProvidersConfig(null)).toThrow(
        /must be a JSON object/i,
      );
      expect(() => validateProvidersConfig([])).toThrow(
        /must be a JSON object/i,
      );
    });

    it("should return the config when valid", () => {
      const config: ProvidersConfig = {
        default: "ollama",
        providers: [
          {
            name: "ollama",
            type: "ollama",
            models: [
              { name: "Llama", key: "llama3.2", contextWindowTokens: 128_000 },
            ],
            defaultModel: "llama3.2",
          },
        ],
      };

      expect(validateProvidersConfig(config)).toEqual(config);
    });
  });

  describe("resolveProvider()", () => {
    const testConfig: ProvidersConfig = {
      default: "ollama",
      providers: [
        {
          name: "ollama",
          type: "ollama",
          models: [
            { name: "Llama", key: "llama3.2", contextWindowTokens: 128_000 },
          ],
          defaultModel: "llama3.2",
        },
        {
          name: "bedrock",
          type: "bedrock",
          models: [
            { name: "Claude", key: "claude", contextWindowTokens: 128_000 },
          ],
          defaultModel: "claude",
        },
      ],
    };

    it("should resolve provider by name", () => {
      const provider = resolveProvider(testConfig, "ollama");
      expect(provider.name).toBe("ollama");
      expect(provider.type).toBe("ollama");
    });

    it("should resolve default provider when no name provided", () => {
      const provider = resolveProvider(testConfig);
      expect(provider.name).toBe("ollama");
    });

    it("should throw error for unknown provider name", () => {
      expect(() => resolveProvider(testConfig, "unknown")).toThrow(
        /unknown|not found|available/i,
      );
    });

    it("should list available providers in error message", () => {
      expectErrorMessage(
        () => resolveProvider(testConfig, "unknown"),
        [/ollama/, /bedrock/],
      );
    });
  });

  describe("resolveModelKey()", () => {
    const testProvider = {
      name: "ollama",
      type: "ollama" as const,
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
    };

    it("should return provided model key when valid", () => {
      const key = resolveModelKey(testProvider, "llama3.2:90b");
      expect(key).toBe("llama3.2:90b");
    });

    it("should return default model when no key provided", () => {
      const key = resolveModelKey(testProvider);
      expect(key).toBe("llama3.2:3b");
    });

    it("should throw error for invalid model key", () => {
      expect(() => resolveModelKey(testProvider, "nonexistent")).toThrow(
        /invalid|unknown|not found|available/i,
      );
    });

    it("should list available model keys in error message", () => {
      expectErrorMessage(
        () => resolveModelKey(testProvider, "nonexistent"),
        [/llama3.2:3b/, /llama3.2:90b/],
      );
    });
  });

  describe("default provider/model updates", () => {
    const testConfig: ProvidersConfig = {
      default: "ollama",
      providers: [
        {
          name: "ollama",
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
        },
        {
          name: "bedrock",
          type: "bedrock",
          models: [
            { name: "Claude", key: "claude", contextWindowTokens: 128_000 },
          ],
          defaultModel: "claude",
        },
      ],
    };

    it("should expose the persisted default provider/model selection", () => {
      expect(getDefaultProviderModelSelection(testConfig)).toEqual({
        providerName: "ollama",
        modelKey: "llama3.2:3b",
      });
    });

    it("should update the default provider and selected provider defaultModel", () => {
      const updated = updateDefaultProviderModelSelection(
        testConfig,
        "ollama",
        "llama3.2:90b",
      );

      expect(updated.default).toBe("ollama");
      expect(updated.providers[0].defaultModel).toBe("llama3.2:90b");
      expect(updated.providers[1].defaultModel).toBe("claude");
    });

    it("should switch the root default when selecting a different provider", () => {
      const updated = updateDefaultProviderModelSelection(
        testConfig,
        "bedrock",
      );

      expect(updated.default).toBe("bedrock");
      expect(updated.providers[1].defaultModel).toBe("claude");
    });

    it("should reject invalid default provider/model updates", () => {
      expect(() =>
        updateDefaultProviderModelSelection(
          testConfig,
          "missing-provider",
          "llama3.2:3b",
        ),
      ).toThrow(/unknown|provider/i);

      expect(() =>
        updateDefaultProviderModelSelection(
          testConfig,
          "ollama",
          "missing-model",
        ),
      ).toThrow(/invalid|model/i);
    });
  });
});
