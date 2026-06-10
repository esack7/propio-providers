import * as fs from "fs";
import * as path from "path";
import {
  loadProvidersConfig,
  loadProvidersConfigAsync,
  writeProvidersConfig,
  updateDefaultProviderModelSelectionInFile,
} from "../configFile.js";
import { ProvidersConfig, ProviderConfig } from "../config.js";

describe("Configuration File Helpers", () => {
  const tempDir = "/tmp/config-file-tests";
  const defaultOllamaModel = {
    name: "Llama",
    key: "llama3.2",
    contextWindowTokens: 128_000,
  };

  function createOllamaProvider(
    overrides: Partial<ProviderConfig> = {},
  ): ProviderConfig {
    return {
      name: "ollama",
      type: "ollama",
      models: [defaultOllamaModel],
      defaultModel: "llama3.2",
      ...overrides,
    } as ProviderConfig;
  }

  function createProvidersConfig(
    overrides: Partial<ProvidersConfig> = {},
  ): ProvidersConfig {
    return {
      default: "ollama",
      providers: [createOllamaProvider()],
      ...overrides,
    };
  }

  function writeTempConfig(fileName: string, config: unknown): string {
    const configPath = path.join(tempDir, fileName);
    fs.writeFileSync(configPath, JSON.stringify(config, null, 2));
    return configPath;
  }

  beforeAll(() => {
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir, { recursive: true });
    }
  });

  afterAll(() => {
    if (fs.existsSync(tempDir)) {
      fs.rmSync(tempDir, { recursive: true });
    }
  });

  describe("loadProvidersConfig()", () => {
    it("should load valid JSON file and return ProvidersConfig", () => {
      const configPath = writeTempConfig(
        "valid-config.json",
        createProvidersConfig(),
      );

      const loaded = loadProvidersConfig(configPath);
      expect(loaded.default).toBe("ollama");
      expect(loaded.providers).toHaveLength(1);
      expect(loaded.providers[0].name).toBe("ollama");
    });

    it("should load valid OpenRouter config with routing and debug fields", () => {
      const configPath = path.join(tempDir, "valid-openrouter-config.json");
      const config: ProvidersConfig = {
        default: "openrouter",
        providers: [
          {
            name: "openrouter",
            type: "openrouter",
            models: [
              {
                name: "GPT-4o",
                key: "openai/gpt-4o",
                contextWindowTokens: 128_000,
              },
              {
                name: "DeepSeek",
                key: "deepseek/deepseek-chat",
                contextWindowTokens: 128_000,
              },
            ],
            defaultModel: "openai/gpt-4o",
            apiKey: "sk-or-test",
            provider: {
              allowFallbacks: true,
              order: ["provider-a", "provider-b"],
              requireParameters: false,
            },
            fallbackModels: ["openai/gpt-4o-mini", "openai/gpt-4.1-mini"],
            debugEchoUpstreamBody: true,
          },
        ],
      };
      fs.writeFileSync(configPath, JSON.stringify(config, null, 2));

      const loaded = loadProvidersConfig(configPath);
      expect(loaded.providers[0]).toMatchObject({
        type: "openrouter",
        provider: {
          allowFallbacks: true,
          order: ["provider-a", "provider-b"],
          requireParameters: false,
        },
        fallbackModels: ["openai/gpt-4o-mini", "openai/gpt-4.1-mini"],
        debugEchoUpstreamBody: true,
      });
    });

    it("should throw error for missing file", () => {
      const configPath = path.join(tempDir, "missing-file.json");
      expect(() => loadProvidersConfig(configPath)).toThrow(
        /not found|ENOENT/i,
      );
    });

    it("should use the injected missing-file message when provided", () => {
      const configPath = path.join(tempDir, "missing-with-message.json");
      expect(() =>
        loadProvidersConfig(configPath, {
          missingMessage: "Custom missing config guidance",
        }),
      ).toThrow(/Custom missing config guidance/);
    });

    it("should throw error for invalid JSON", () => {
      const configPath = path.join(tempDir, "invalid-json.json");
      fs.writeFileSync(configPath, "{ invalid json }");

      expect(() => loadProvidersConfig(configPath)).toThrow(
        /JSON|parse|invalid/i,
      );
    });

    it("should throw error when providers array is missing", () => {
      const configPath = path.join(tempDir, "no-providers.json");
      const config = { default: "ollama" };
      fs.writeFileSync(configPath, JSON.stringify(config));

      expect(() => loadProvidersConfig(configPath)).toThrow(
        /providers|required|missing/i,
      );
    });

    it("should throw error when default field is missing", () => {
      const config = {
        providers: [createOllamaProvider()],
      };
      const configPath = writeTempConfig("no-default.json", config);

      expect(() => loadProvidersConfig(configPath)).toThrow(
        /default|required|missing/i,
      );
    });

    it("should validate that default references existing provider", () => {
      const configPath = writeTempConfig("invalid-default-ref.json", {
        default: "nonexistent",
        providers: [createOllamaProvider()],
      } satisfies ProvidersConfig);

      expect(() => loadProvidersConfig(configPath)).toThrow(
        /default.*provider|provider.*not found|unknown provider/i,
      );
    });

    it("should validate defaultModel references valid model key", () => {
      const configPath = writeTempConfig(
        "invalid-model-ref.json",
        createProvidersConfig({
          providers: [
            createOllamaProvider({ defaultModel: "nonexistent-model" } as any),
          ],
        }),
      );

      expect(() => loadProvidersConfig(configPath)).toThrow(
        /defaultModel|model.*not found|unknown model/i,
      );
    });

    it("should validate required provider fields", () => {
      const configPath = path.join(tempDir, "missing-provider-fields.json");
      const config: any = {
        default: "ollama",
        providers: [
          {
            name: "ollama",
            type: "ollama",
            // missing models and defaultModel
          },
        ],
      };
      fs.writeFileSync(configPath, JSON.stringify(config));

      expect(() => loadProvidersConfig(configPath)).toThrow(
        /missing|required|fields/i,
      );
    });

    it("should validate models array structure", () => {
      const configPath = path.join(tempDir, "invalid-model-structure.json");
      const config: any = {
        default: "ollama",
        providers: [
          {
            name: "ollama",
            type: "ollama",
            models: [{ name: "Llama" }], // missing key
            defaultModel: "llama3.2",
          },
        ],
      };
      fs.writeFileSync(configPath, JSON.stringify(config));

      expect(() => loadProvidersConfig(configPath)).toThrow(
        /model|name|key|missing|required/i,
      );
    });

    it("should require contextWindowTokens for each configured model", () => {
      const configPath = path.join(tempDir, "missing-context-window.json");
      const config: any = {
        default: "ollama",
        providers: [
          {
            name: "ollama",
            type: "ollama",
            models: [{ name: "Llama", key: "llama3.2" }],
            defaultModel: "llama3.2",
          },
        ],
      };
      fs.writeFileSync(configPath, JSON.stringify(config));

      expect(() => loadProvidersConfig(configPath)).toThrow(
        /contextWindowTokens|required|missing/i,
      );
    });

    it.each([0, -1, 1.5, "128000", null])(
      "should reject invalid contextWindowTokens value %p",
      (contextWindowTokens) => {
        const configPath = path.join(
          tempDir,
          `invalid-context-window-${String(contextWindowTokens)}.json`,
        );
        const config: any = {
          default: "ollama",
          providers: [
            {
              name: "ollama",
              type: "ollama",
              models: [{ name: "Llama", key: "llama3.2", contextWindowTokens }],
              defaultModel: "llama3.2",
            },
          ],
        };
        fs.writeFileSync(configPath, JSON.stringify(config));

        expect(() => loadProvidersConfig(configPath)).toThrow(
          /contextWindowTokens|positive integer/i,
        );
      },
    );

    it("should validate unique provider names", () => {
      const configPath = path.join(tempDir, "duplicate-names.json");
      const config: any = {
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
            name: "ollama", // duplicate
            type: "bedrock",
            models: [
              { name: "Claude", key: "claude", contextWindowTokens: 128_000 },
            ],
            defaultModel: "claude",
          },
        ],
      };
      fs.writeFileSync(configPath, JSON.stringify(config));

      expect(() => loadProvidersConfig(configPath)).toThrow(
        /duplicate|unique|provider name/i,
      );
    });

    it("should validate unique model keys within provider", () => {
      const duplicateModel = {
        ...defaultOllamaModel,
        name: "Llama Duplicate",
      };
      const configPath = writeTempConfig(
        "duplicate-model-keys.json",
        createProvidersConfig({
          providers: [
            createOllamaProvider({
              models: [defaultOllamaModel, duplicateModel],
            } as any),
          ],
        }),
      );

      expect(() => loadProvidersConfig(configPath)).toThrow(
        /duplicate|unique|model.*key/i,
      );
    });

    it("should reject malformed OpenRouter provider routing fields", () => {
      const configPath = path.join(tempDir, "bad-openrouter-routing.json");
      const config: any = {
        default: "openrouter",
        providers: [
          {
            name: "openrouter",
            type: "openrouter",
            models: [
              {
                name: "GPT-4o",
                key: "openai/gpt-4o",
                contextWindowTokens: 128_000,
              },
            ],
            defaultModel: "openai/gpt-4o",
            provider: {
              allowFallbacks: "yes",
            },
          },
        ],
      };
      fs.writeFileSync(configPath, JSON.stringify(config));

      expect(() => loadProvidersConfig(configPath)).toThrow(
        /provider\.allowFallbacks|boolean/i,
      );
    });

    it("should reject a non-object OpenRouter provider field", () => {
      const configPath = path.join(tempDir, "bad-openrouter-provider.json");
      const config: any = {
        default: "openrouter",
        providers: [
          {
            name: "openrouter",
            type: "openrouter",
            models: [
              {
                name: "GPT-4o",
                key: "openai/gpt-4o",
                contextWindowTokens: 128_000,
              },
            ],
            defaultModel: "openai/gpt-4o",
            provider: [],
          },
        ],
      };
      fs.writeFileSync(configPath, JSON.stringify(config));

      expect(() => loadProvidersConfig(configPath)).toThrow(
        /provider.*must be an object/i,
      );
    });

    it("should reject malformed OpenRouter fallback model lists", () => {
      const configPath = path.join(tempDir, "bad-openrouter-fallbacks.json");
      const config: any = {
        default: "openrouter",
        providers: [
          {
            name: "openrouter",
            type: "openrouter",
            models: [
              {
                name: "GPT-4o",
                key: "openai/gpt-4o",
                contextWindowTokens: 128_000,
              },
            ],
            defaultModel: "openai/gpt-4o",
            fallbackModels: [""],
          },
        ],
      };
      fs.writeFileSync(configPath, JSON.stringify(config));

      expect(() => loadProvidersConfig(configPath)).toThrow(
        /fallbackModels|non-empty array/i,
      );
    });

    it("should reject malformed debugEchoUpstreamBody values", () => {
      const configPath = path.join(tempDir, "bad-openrouter-debug.json");
      const config: any = {
        default: "openrouter",
        providers: [
          {
            name: "openrouter",
            type: "openrouter",
            models: [
              {
                name: "GPT-4o",
                key: "openai/gpt-4o",
                contextWindowTokens: 128_000,
              },
            ],
            defaultModel: "openai/gpt-4o",
            debugEchoUpstreamBody: "true",
          },
        ],
      };
      fs.writeFileSync(configPath, JSON.stringify(config));

      expect(() => loadProvidersConfig(configPath)).toThrow(
        /debugEchoUpstreamBody|boolean/i,
      );
    });

    it("should load config with multiple providers", () => {
      const configPath = path.join(tempDir, "multi-provider.json");
      const config: ProvidersConfig = {
        default: "ollama",
        providers: [
          {
            name: "ollama",
            type: "ollama",
            models: [
              {
                name: "Llama 3.2",
                key: "llama3.2:3b",
                contextWindowTokens: 128_000,
              },
              {
                name: "Llama 3.2 Large",
                key: "llama3.2:90b",
                contextWindowTokens: 128_000,
              },
            ],
            defaultModel: "llama3.2:3b",
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
        ],
      };
      fs.writeFileSync(configPath, JSON.stringify(config, null, 2));

      const loaded = loadProvidersConfig(configPath);
      expect(loaded.default).toBe("ollama");
      expect(loaded.providers).toHaveLength(2);
      expect(loaded.providers[0].type).toBe("ollama");
      expect(loaded.providers[1].type).toBe("bedrock");
    });
  });

  describe("loadProvidersConfigAsync()", () => {
    it("should load valid JSON file and return ProvidersConfig", async () => {
      const config = createProvidersConfig();
      const configPath = writeTempConfig("valid-async-config.json", config);

      await expect(loadProvidersConfigAsync(configPath)).resolves.toEqual(
        config,
      );
    });

    it("should preserve the missing file error message", async () => {
      const configPath = path.join(tempDir, "missing-async-file.json");

      await expect(loadProvidersConfigAsync(configPath)).rejects.toThrow(
        /Configuration file not found/,
      );
    });
  });

  describe("writeProvidersConfig()", () => {
    it("should write a config that round-trips through loadProvidersConfig", () => {
      const configPath = path.join(tempDir, "written-config.json");
      const config = createProvidersConfig();

      writeProvidersConfig(configPath, config);

      expect(loadProvidersConfig(configPath)).toEqual(config);
    });
  });

  describe("updateDefaultProviderModelSelectionInFile()", () => {
    it("should persist updated defaults to disk and keep the file reloadable", () => {
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
        ],
      };
      const configPath = path.join(tempDir, "updated-defaults.json");
      fs.writeFileSync(configPath, JSON.stringify(testConfig, null, 2), "utf8");

      const updated = updateDefaultProviderModelSelectionInFile(
        configPath,
        "ollama",
        "llama3.2:90b",
      );

      expect(updated.default).toBe("ollama");
      expect(updated.providers[0].defaultModel).toBe("llama3.2:90b");

      const reloaded = loadProvidersConfig(configPath);
      expect(reloaded.default).toBe("ollama");
      expect(reloaded.providers[0].defaultModel).toBe("llama3.2:90b");
    });
  });
});
