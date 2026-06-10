import {
  ChatMessage,
  ChatTool,
  ChatToolCall,
  ChatRequest,
  ChatResponse,
  ChatChunk,
  ChatStreamEvent,
  ProviderError,
  ProviderAuthenticationError,
  ProviderRateLimitError,
  ProviderModelNotFoundError,
} from "../types.js";

describe("Provider Types", () => {
  describe("ChatMessage", () => {
    it("should create a user message with role and content", () => {
      const msg: ChatMessage = {
        role: "user",
        content: "Hello",
      };
      expect(msg.role).toBe("user");
      expect(msg.content).toBe("Hello");
    });

    it("should create an assistant message", () => {
      const msg: ChatMessage = {
        role: "assistant",
        content: "Hi there",
      };
      expect(msg.role).toBe("assistant");
    });

    it("should create a system message", () => {
      const msg: ChatMessage = {
        role: "system",
        content: "System prompt",
      };
      expect(msg.role).toBe("system");
    });

    it("should create a tool message", () => {
      const msg: ChatMessage = {
        role: "tool",
        content: "Tool result",
      };
      expect(msg.role).toBe("tool");
    });

    it("should include optional toolCalls", () => {
      const toolCall: ChatToolCall = {
        function: {
          name: "test_func",
          arguments: { arg1: "value" },
        },
      };
      const msg: ChatMessage = {
        role: "assistant",
        content: "Calling tool",
        toolCalls: [toolCall],
      };
      expect(msg.toolCalls).toHaveLength(1);
      expect(msg.toolCalls![0].function.name).toBe("test_func");
    });

    it("should include optional images as Uint8Array", () => {
      const imageData = new Uint8Array([1, 2, 3]);
      const msg: ChatMessage = {
        role: "user",
        content: "Look at this image",
        images: [imageData],
      };
      expect(msg.images).toHaveLength(1);
      expect(msg.images![0]).toEqual(imageData);
    });

    it("should include optional images as base64 string", () => {
      const msg: ChatMessage = {
        role: "user",
        content: "Look at this image",
        images: ["data:image/png;base64,iVBORw0KGgo="],
      };
      expect(msg.images).toHaveLength(1);
      expect(typeof msg.images![0]).toBe("string");
    });
  });

  describe("ChatTool", () => {
    it("should define a tool with type function", () => {
      const tool: ChatTool = {
        type: "function",
        function: {
          name: "get_weather",
          description: "Get weather for a location",
          parameters: {
            type: "object",
            properties: {
              location: {
                type: "string",
                description: "The city name",
              },
            },
            required: ["location"],
          },
        },
      };
      expect(tool.type).toBe("function");
      expect(tool.function.name).toBe("get_weather");
    });

    it("should support JSON Schema parameters", () => {
      const tool: ChatTool = {
        type: "function",
        function: {
          name: "calculate",
          description: "Perform calculation",
          parameters: {
            type: "object",
            properties: {
              operation: {
                type: "string",
                enum: ["add", "subtract", "multiply"],
              },
              numbers: {
                type: "array",
                items: { type: "number" },
              },
            },
            required: ["operation", "numbers"],
          },
        },
      };
      expect(tool.function.parameters.properties.operation.enum).toContain(
        "add",
      );
    });
  });

  describe("ChatToolCall", () => {
    it("should represent a function tool call", () => {
      const toolCall: ChatToolCall = {
        thoughtSignature: "sig-123",
        function: {
          name: "get_weather",
          arguments: { location: "New York" },
        },
      };
      expect(toolCall.function.name).toBe("get_weather");
      expect(toolCall.function.arguments.location).toBe("New York");
      expect(toolCall.thoughtSignature).toBe("sig-123");
    });

    it("should support complex arguments", () => {
      const toolCall: ChatToolCall = {
        function: {
          name: "process_data",
          arguments: {
            items: [1, 2, 3],
            metadata: { key: "value" },
          },
        },
      };
      expect(Array.isArray(toolCall.function.arguments.items)).toBe(true);
      expect(toolCall.function.arguments.metadata.key).toBe("value");
    });
  });

  describe("ChatRequest", () => {
    it("should include messages and model", () => {
      const request: ChatRequest = {
        messages: [{ role: "user", content: "Hello" }],
        model: "gpt-3.5",
      };
      expect(request.messages).toHaveLength(1);
      expect(request.model).toBe("gpt-3.5");
    });

    it("should optionally include tools", () => {
      const request: ChatRequest = {
        messages: [{ role: "user", content: "Get weather" }],
        model: "gpt-3.5",
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
      expect(request.tools).toHaveLength(1);
    });

    it("should optionally include stream flag", () => {
      const request: ChatRequest = {
        messages: [{ role: "user", content: "Hello" }],
        model: "gpt-3.5",
        stream: true,
      };
      expect(request.stream).toBe(true);
    });
  });

  describe("ChatResponse", () => {
    it("should include message and stop reason", () => {
      const response: ChatResponse = {
        message: {
          role: "assistant",
          content: "Response text",
        },
        stopReason: "end_turn",
      };
      expect(response.message.content).toBe("Response text");
      expect(response.stopReason).toBe("end_turn");
    });

    it("should handle tool_use stop reason", () => {
      const response: ChatResponse = {
        message: {
          role: "assistant",
          content: "Calling tool",
          toolCalls: [
            {
              function: { name: "func", arguments: {} },
            },
          ],
        },
        stopReason: "tool_use",
      };
      expect(response.stopReason).toBe("tool_use");
    });

    it("should handle max_tokens stop reason", () => {
      const response: ChatResponse = {
        message: { role: "assistant", content: "Partial" },
        stopReason: "max_tokens",
      };
      expect(response.stopReason).toBe("max_tokens");
    });

    it("should handle stop_sequence stop reason", () => {
      const response: ChatResponse = {
        message: { role: "assistant", content: "Text" },
        stopReason: "stop_sequence",
      };
      expect(response.stopReason).toBe("stop_sequence");
    });
  });

  describe("ChatStreamEvent", () => {
    it("should support thinking deltas", () => {
      const event: ChatStreamEvent = {
        type: "thinking_delta",
        delta: "Reasoning",
      };

      expect(event.type).toBe("thinking_delta");
      if (event.type === "thinking_delta") {
        expect(event.delta).toBe("Reasoning");
      }
    });
  });

  describe("ChatChunk", () => {
    it("should contain delta content", () => {
      const chunk: ChatChunk = {
        delta: "Hello ",
      };
      expect(chunk.delta).toBe("Hello ");
    });

    it("should optionally include tool calls", () => {
      const chunk: ChatChunk = {
        delta: "",
        toolCalls: [
          {
            function: { name: "tool", arguments: {} },
          },
        ],
      };
      expect(chunk.toolCalls).toHaveLength(1);
    });
  });

  describe("ChatStreamEvent", () => {
    it("should support typed assistant_text events", () => {
      const event: ChatStreamEvent = {
        type: "assistant_text",
        delta: "Hello",
      };
      expect(event.type).toBe("assistant_text");
    });

    it("should support typed reasoning_summary events", () => {
      const event: ChatStreamEvent = {
        type: "reasoning_summary",
        summary: "Checked tools and answered from results.",
        source: "provider",
      };
      expect(event.type).toBe("reasoning_summary");
    });
  });

  describe("ProviderError", () => {
    it("should be constructable with message", () => {
      const error = new ProviderError("Something went wrong");
      expect(error.message).toBe("Something went wrong");
      expect(error).toBeInstanceOf(Error);
    });

    it("should include original error", () => {
      const originalError = new Error("Original");
      const error = new ProviderError("Wrapped", originalError);
      expect(error.originalError).toBe(originalError);
    });
  });

  describe("ProviderAuthenticationError", () => {
    it("should extend ProviderError", () => {
      const error = new ProviderAuthenticationError("Auth failed");
      expect(error).toBeInstanceOf(ProviderError);
      expect(error).toBeInstanceOf(Error);
    });
  });

  describe("ProviderRateLimitError", () => {
    it("should extend ProviderError and include retry info", () => {
      const error = new ProviderRateLimitError("Rate limited", 60);
      expect(error).toBeInstanceOf(ProviderError);
      expect(error.retryAfterSeconds).toBe(60);
    });
  });

  describe("ProviderModelNotFoundError", () => {
    it("should extend ProviderError and include model name", () => {
      const error = new ProviderModelNotFoundError("gpt-5", "invalid model");
      expect(error).toBeInstanceOf(ProviderError);
      expect(error.modelName).toBe("gpt-5");
      expect(error.message).toContain("invalid model");
    });
  });
});
