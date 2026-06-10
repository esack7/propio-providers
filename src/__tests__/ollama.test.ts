// Mock the Ollama package using unstable_mockModule for ESM
const mockChat = jest.fn();
const mockOllamaConstructor = jest.fn().mockImplementation(() => ({
  chat: mockChat,
}));

jest.unstable_mockModule("ollama", () => ({
  Ollama: mockOllamaConstructor,
}));

// Dynamic imports after mocks are set up
let OllamaProvider: any;

beforeAll(async () => {
  const ollamaModule = await import("../providers/ollama.js");
  OllamaProvider = ollamaModule.OllamaProvider;
});

function createTestProvider(options: Record<string, unknown> = {}) {
  return new OllamaProvider({
    model: "test-model",
    contextWindowTokens: 8192,
    ...options,
  });
}

function restoreEnvVar(name: string, value: string | undefined): void {
  if (value === undefined) {
    delete process.env[name];
    return;
  }
  process.env[name] = value;
}

async function collectNonTerminalChunks(
  provider: InstanceType<typeof OllamaProvider>,
  request: {
    model: string;
    messages: unknown[];
    tools?: unknown[];
  } = {
    model: "test-model",
    messages: [{ role: "user", content: "test" }],
  },
): Promise<any[]> {
  const chunks: any[] = [];
  for await (const chunk of provider.streamChat(request)) {
    if (chunk.type !== "terminal") {
      chunks.push(chunk);
    }
  }
  return chunks;
}

function withOllamaEnv(
  env: { host?: string; sandbox?: string },
  testBody: () => void,
): void {
  const originalHost = process.env.OLLAMA_HOST;
  const originalSandbox = process.env.IS_SANDBOX;
  try {
    if (env.host === undefined) {
      delete process.env.OLLAMA_HOST;
    } else {
      process.env.OLLAMA_HOST = env.host;
    }

    if (env.sandbox === undefined) {
      delete process.env.IS_SANDBOX;
    } else {
      process.env.IS_SANDBOX = env.sandbox;
    }

    testBody();
  } finally {
    restoreEnvVar("OLLAMA_HOST", originalHost);
    restoreEnvVar("IS_SANDBOX", originalSandbox);
  }
}

describe("OllamaProvider", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("Initialization", () => {
    it("should initialize with custom host", () => {
      withOllamaEnv({}, () => {
        const host = "http://custom-host:11434";
        createTestProvider({ host });
        expect(mockOllamaConstructor).toHaveBeenCalledWith({ host });
      });
    });

    it("should use localhost default when no host provided", () => {
      withOllamaEnv({ sandbox: process.env.IS_SANDBOX }, () => {
        createTestProvider();
        expect(mockOllamaConstructor).toHaveBeenCalledWith({
          host: "http://localhost:11434",
        });
      });
    });

    it("should use OLLAMA_HOST environment variable if set", () => {
      withOllamaEnv({ host: "http://env-host:11434" }, () => {
        createTestProvider();
        expect(mockOllamaConstructor).toHaveBeenCalledWith({
          host: "http://env-host:11434",
        });
      });
    });

    it("should prioritize environment variable over explicit host", () => {
      withOllamaEnv({ host: "http://env-host:11434" }, () => {
        const host = "http://explicit-host:11434";
        createTestProvider({ host });
        expect(mockOllamaConstructor).toHaveBeenCalledWith({
          host: "http://env-host:11434",
        });
      });
    });
  });

  describe("Backward compatibility", () => {
    it("should support custom host configuration", () => {
      withOllamaEnv({}, () => {
        const host = "http://custom:11434";
        createTestProvider({ model: "llama3.2", host });
        expect(mockOllamaConstructor).toHaveBeenCalledWith({ host });
      });
    });

    it("should handle model parameter in constructor", () => {
      const provider = createTestProvider({ model: "llama3.2" });
      expect(provider).toBeDefined();
    });
  });

  describe("Sandbox mode host resolution", () => {
    it("should use host.docker.internal default in sandbox mode", () => {
      withOllamaEnv({ sandbox: "true" }, () => {
        createTestProvider();
        expect(mockOllamaConstructor).toHaveBeenCalledWith({
          host: "http://host.docker.internal:11434",
        });
      });
    });

    it("should convert localhost to host.docker.internal in sandbox", () => {
      withOllamaEnv({ sandbox: "true" }, () => {
        const host = "http://localhost:12345";
        createTestProvider({ host });
        expect(mockOllamaConstructor).toHaveBeenCalledWith({
          host: "http://host.docker.internal:12345",
        });
      });
    });

    it("should preserve custom hosts in sandbox mode", () => {
      withOllamaEnv({ sandbox: "true" }, () => {
        const host = "http://custom-host:11434";
        createTestProvider({ host });
        expect(mockOllamaConstructor).toHaveBeenCalledWith({
          host: "http://custom-host:11434",
        });
      });
    });

    it("should use localhost default in regular mode", () => {
      withOllamaEnv({}, () => {
        createTestProvider();
        expect(mockOllamaConstructor).toHaveBeenCalledWith({
          host: "http://localhost:11434",
        });
      });
    });

    it("should respect OLLAMA_HOST without conversion in sandbox", () => {
      withOllamaEnv({ host: "http://localhost:9999", sandbox: "true" }, () => {
        createTestProvider();
        expect(mockOllamaConstructor).toHaveBeenCalledWith({
          host: "http://localhost:9999",
        });
      });
    });
  });

  describe("Provider identification", () => {
    it("should have name property set to ollama", () => {
      const provider = createTestProvider();
      expect(provider.name).toBe("ollama");
    });
  });

  describe("streamChat", () => {
    it("should call ollama.chat with correct parameters", async () => {
      const provider = createTestProvider();

      // Mock async generator
      mockChat.mockReturnValue(
        (async function* () {
          yield { message: { content: "Hello", tool_calls: undefined } };
        })(),
      );

      const messages: any[] = [{ role: "user", content: "test" }];

      const chunks: any[] = [];
      for await (const chunk of provider.streamChat({
        model: "test-model",
        messages,
      })) {
        chunks.push(chunk);
      }

      expect(mockChat).toHaveBeenCalledWith(
        expect.objectContaining({
          model: "test-model",
          messages: expect.arrayContaining([
            expect.objectContaining({ role: "user", content: "test" }),
          ]),
          stream: true,
        }),
      );
    });

    it("should yield content deltas from ollama response", async () => {
      const provider = createTestProvider();

      mockChat.mockReturnValue(
        (async function* () {
          yield { message: { content: "Hello " } };
          yield { message: { content: "world" } };
        })(),
      );

      const chunks = await collectNonTerminalChunks(provider);

      expect(chunks).toHaveLength(2);
      expect(chunks[0].delta).toBe("Hello ");
      expect(chunks[1].delta).toBe("world");
    });

    it("should yield thinking deltas from ollama response", async () => {
      const provider = createTestProvider();

      mockChat.mockReturnValue(
        (async function* () {
          yield { message: { content: "", thinking: "Thinking " } };
          yield { message: { content: "Hello" } };
        })(),
      );

      const chunks = await collectNonTerminalChunks(provider);

      expect(chunks).toEqual([
        { type: "thinking_delta", delta: "Thinking " },
        { type: "assistant_text", delta: "Hello" },
      ]);
    });

    it("should handle tool calls in response", async () => {
      const provider = createTestProvider();

      mockChat.mockReturnValue(
        (async function* () {
          yield {
            message: {
              content: "",
              tool_calls: [
                {
                  function: {
                    name: "test_tool",
                    arguments: { arg: "value" },
                  },
                },
              ],
            },
          };
        })(),
      );

      const chunks: any[] = [];
      for await (const chunk of provider.streamChat({
        model: "test-model",
        messages: [{ role: "user", content: "test" }],
        tools: [
          {
            type: "function",
            function: {
              name: "test_tool",
              description: "test",
              parameters: { type: "object", properties: {} },
            },
          },
        ],
      })) {
        chunks.push(chunk);
      }

      const toolCallChunk = chunks.find((c) => c.toolCalls);
      expect(toolCallChunk).toBeDefined();
      expect(toolCallChunk.toolCalls).toHaveLength(1);
      expect(toolCallChunk.toolCalls[0].function.name).toBe("test_tool");
    });

    it("should pass tools to ollama when provided", async () => {
      const provider = createTestProvider();

      mockChat.mockReturnValue(
        (async function* () {
          yield { message: { content: "test" } };
        })(),
      );

      const tools: any[] = [
        {
          type: "function",
          function: {
            name: "test_tool",
            description: "test",
            parameters: { type: "object", properties: {} },
          },
        },
      ];

      for await (const chunk of provider.streamChat({
        model: "test-model",
        messages: [{ role: "user", content: "test" }],
        tools,
      })) {
        // consume stream
      }

      expect(mockChat).toHaveBeenCalledWith(
        expect.objectContaining({
          tools: expect.arrayContaining([
            expect.objectContaining({
              type: "function",
              function: expect.objectContaining({
                name: "test_tool",
              }),
            }),
          ]),
        }),
      );
    });

    it("should expand batched tool results into separate messages", async () => {
      const provider = createTestProvider();

      mockChat.mockReturnValue(
        (async function* () {
          yield { message: { content: "response" } };
        })(),
      );

      const messages: any[] = [
        { role: "user", content: "test" },
        {
          role: "tool",
          content: "",
          toolResults: [
            { toolCallId: "call1", toolName: "tool1", content: "result1" },
            { toolCallId: "call2", toolName: "tool2", content: "result2" },
          ],
        },
      ];

      for await (const chunk of provider.streamChat({
        model: "test-model",
        messages,
      })) {
        // consume stream
      }

      // Should have expanded the batched tool results into 3 messages:
      // 1. user message
      // 2. tool result 1
      // 3. tool result 2
      const callArgs = mockChat.mock.calls[0][0];
      expect(callArgs.messages).toHaveLength(3);
      expect(callArgs.messages[0].role).toBe("user");
      expect(callArgs.messages[1].role).toBe("tool");
      expect(callArgs.messages[1].content).toBe("result1");
      expect(callArgs.messages[2].role).toBe("tool");
      expect(callArgs.messages[2].content).toBe("result2");
    });
  });
});
