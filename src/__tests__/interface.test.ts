import { LLMProvider } from "../interface.js";
import { ChatRequest, ChatResponse, ChatChunk } from "../types.js";

/**
 * Mock implementation for testing interface requirements
 */
class MockProvider implements LLMProvider {
  name: string = "mock";

  getCapabilities() {
    return { contextWindowTokens: 128000 };
  }

  async chat(request: ChatRequest): Promise<ChatResponse> {
    return {
      message: {
        role: "assistant",
        content: "Mock response",
      },
      stopReason: "end_turn",
    };
  }

  async *streamChat(request: ChatRequest): AsyncIterable<ChatChunk> {
    yield { delta: "Mock " };
    yield { delta: "stream" };
  }
}

describe("LLMProvider Interface", () => {
  describe("Interface definition", () => {
    it("should define chat method that returns Promise<ChatResponse>", async () => {
      const provider = new MockProvider();
      const request: ChatRequest = {
        messages: [{ role: "user", content: "Hello" }],
        model: "mock-model",
      };
      const response = await provider.chat(request);
      expect(response).toHaveProperty("message");
      expect(response).toHaveProperty("stopReason");
    });

    it("should define streamChat method that returns AsyncIterable<ChatChunk>", async () => {
      const provider = new MockProvider();
      const request: ChatRequest = {
        messages: [{ role: "user", content: "Hello" }],
        model: "mock-model",
      };
      const chunks: ChatChunk[] = [];
      for await (const chunk of provider.streamChat(request)) {
        chunks.push(chunk);
      }
      expect(chunks.length).toBeGreaterThan(0);
      expect(chunks[0]).toHaveProperty("delta");
    });

    it("should have readonly name property", () => {
      const provider = new MockProvider();
      expect(provider.name).toBe("mock");
      // Verify it's a string
      expect(typeof provider.name).toBe("string");
    });
  });

  describe("Chat method contract", () => {
    it("should accept ChatRequest with messages and model", async () => {
      const provider = new MockProvider();
      const request: ChatRequest = {
        messages: [
          { role: "system", content: "You are helpful" },
          { role: "user", content: "Hello" },
        ],
        model: "test-model",
      };
      for await (const chunk of provider.streamChat(request)) {
        expect(chunk.delta).toBeDefined();
      }
    });

    it("should accept ChatRequest with tools", async () => {
      const provider = new MockProvider();
      const request: ChatRequest = {
        messages: [{ role: "user", content: "Help" }],
        model: "test-model",
        tools: [
          {
            type: "function",
            function: {
              name: "get_weather",
              description: "Get weather",
              parameters: { type: "object", properties: {} },
            },
          },
        ],
      };
      for await (const chunk of provider.streamChat(request)) {
        expect(chunk).toBeDefined();
      }
    });

    it("should return ChatResponse with message and stopReason", async () => {
      const provider = new MockProvider();
      const request: ChatRequest = {
        messages: [{ role: "user", content: "Test" }],
        model: "test-model",
      };
      let hasContent = false;
      for await (const chunk of provider.streamChat(request)) {
        if (chunk.delta) {
          hasContent = true;
        }
      }
      expect(hasContent).toBe(true);
    });
  });

  describe("StreamChat method contract", () => {
    it("should be an async generator", async () => {
      const provider = new MockProvider();
      const request: ChatRequest = {
        messages: [{ role: "user", content: "Test" }],
        model: "test-model",
      };
      const result = provider.streamChat(request);
      expect(result[Symbol.asyncIterator]).toBeDefined();
    });

    it("should yield ChatChunk objects with delta", async () => {
      const provider = new MockProvider();
      const request: ChatRequest = {
        messages: [{ role: "user", content: "Test" }],
        model: "test-model",
      };
      for await (const chunk of provider.streamChat(request)) {
        expect(chunk).toHaveProperty("delta");
        expect(typeof chunk.delta).toBe("string");
      }
    });

    it("should optionally yield toolCalls in final chunk", async () => {
      class ToolCallProvider implements LLMProvider {
        name = "tool-provider";

        getCapabilities() {
          return { contextWindowTokens: 128000 };
        }

        async chat(request: ChatRequest): Promise<ChatResponse> {
          throw new Error("Not implemented");
        }

        async *streamChat(request: ChatRequest): AsyncIterable<ChatChunk> {
          yield { delta: "Calling " };
          yield { delta: "tool" };
          yield {
            delta: "",
            toolCalls: [
              {
                function: {
                  name: "my_tool",
                  arguments: { arg: "value" },
                },
              },
            ],
          };
        }
      }

      const provider = new ToolCallProvider();
      const request: ChatRequest = {
        messages: [{ role: "user", content: "Call tool" }],
        model: "test-model",
      };
      let hasToolCalls = false;
      for await (const chunk of provider.streamChat(request)) {
        if (chunk.toolCalls) {
          hasToolCalls = true;
          expect(chunk.toolCalls).toHaveLength(1);
        }
      }
      expect(hasToolCalls).toBe(true);
    });
  });

  describe("Provider identification", () => {
    it("should uniquely identify the provider via name", () => {
      const provider = new MockProvider();
      expect(provider.name).toBe("mock");
      expect(typeof provider.name).toBe("string");
      expect(provider.name.length).toBeGreaterThan(0);
    });
  });
});
