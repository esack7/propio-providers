import {
  ChatMessage,
  ChatRequest,
  ChatStreamEvent,
  ProviderAuthenticationError,
  ProviderError,
} from "../types.js";
import type { ProviderDiagnosticListener } from "../diagnostics.js";
import {
  applyOpenAIMessageCore,
  buildOpenAIChatCompletionRequestBody,
  createOpenAIMessageWithImages,
  type OpenAIMessageContentPart,
  type OpenAIStreamToolCallAccumulator,
} from "../internal/shared.js";
import {
  consumeOpenAiChatCompletionsStream,
  fetchOpenAiCompatibleStreamReader,
} from "../internal/openAiStream.js";
import {
  OpenAiCompatibleProvider,
  type OpenAiCompatibleProviderOptions,
} from "../internal/openAiCompatibleProvider.js";

const CLOUDFLARE_CHAT_COMPLETIONS_URL =
  "https://api.cloudflare.com/client/v4/accounts";

interface OpenAIMessage {
  role: "user" | "assistant" | "system" | "tool";
  content: string | OpenAIMessageContentPart[];
  tool_calls?: Array<{
    id: string;
    type: "function";
    function: { name: string; arguments: string };
  }>;
  tool_call_id?: string;
}

export interface CloudflareProviderOptions extends OpenAiCompatibleProviderOptions {
  accountId?: string;
}

export function normalizeCloudflareModelId(modelId: string): string {
  if (modelId.startsWith("@")) {
    return modelId;
  }
  if (modelId.startsWith("cf/")) {
    return `@${modelId}`;
  }
  return modelId;
}

function resolveCloudflareApiKey(apiKey?: string): string {
  return (
    apiKey ??
    process.env.CLOUDFLARE_API_TOKEN ??
    process.env.CLOUDFLARE_AUTH_TOKEN ??
    process.env.CLOUDFLARE_API_KEY ??
    ""
  );
}

function resolveCloudflareAccountId(accountId?: string): string {
  return accountId ?? process.env.CLOUDFLARE_ACCOUNT_ID ?? "";
}

/**
 * Cloudflare Workers AI implementation using the OpenAI-compatible chat-completions endpoint.
 */
export class CloudflareProvider extends OpenAiCompatibleProvider {
  readonly name = "cloudflare";
  private readonly model: string;
  private readonly apiKey: string;
  private readonly accountId: string;
  private readonly retryConfig?: {
    maxRetries: number;
    consecutive529Limit: number;
  };
  private readonly onDiagnosticEvent?: ProviderDiagnosticListener;

  constructor(options: CloudflareProviderOptions) {
    super();
    const apiKey = resolveCloudflareApiKey(options.apiKey);
    const accountId = resolveCloudflareAccountId(options.accountId);

    if (!accountId || accountId.trim() === "") {
      throw new ProviderAuthenticationError(
        "Cloudflare account ID is required. Set CLOUDFLARE_ACCOUNT_ID or pass accountId in options.",
      );
    }
    if (!apiKey || apiKey.trim() === "") {
      throw new ProviderAuthenticationError(
        "Cloudflare API token is required. Set CLOUDFLARE_API_TOKEN, CLOUDFLARE_AUTH_TOKEN, or CLOUDFLARE_API_KEY, or pass apiKey in options.",
      );
    }

    this.model = options.model;
    this.configureCapabilities(options.contextWindowTokens);
    this.apiKey = apiKey;
    this.accountId = accountId;
    this.retryConfig = options.retryConfig;
    this.onDiagnosticEvent = options.onDiagnosticEvent;
  }

  private getChatCompletionsUrl(): string {
    return `${CLOUDFLARE_CHAT_COMPLETIONS_URL}/${this.accountId}/ai/v1/chat/completions`;
  }

  protected chatMessageToOpenAIMessage(msg: ChatMessage): OpenAIMessage {
    return applyOpenAIMessageCore(createOpenAIMessageWithImages(msg), msg);
  }

  private createChatCompletionRequestBody(
    request: ChatRequest,
  ): Record<string, unknown> {
    const effectiveModel = normalizeCloudflareModelId(
      request.model || this.model,
    );
    return buildOpenAIChatCompletionRequestBody({
      request: { ...request, model: effectiveModel },
      model: effectiveModel,
      mapMessage: (msg) => this.chatMessageToOpenAIMessage(msg),
      mapTool: (tool) => this.chatToolToOpenAITool(tool),
    });
  }

  private async fetchStream(
    body: Record<string, unknown>,
    signal: AbortSignal | undefined,
  ): Promise<Response> {
    const response = await fetch(this.getChatCompletionsUrl(), {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        Accept: "text/event-stream, application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
      signal,
    });

    if (!response.ok) {
      let errorBody = "";
      try {
        errorBody = await response.text();
      } catch {
        // ignore read failures
      }
      throw this.translateError(
        new Error(errorBody || `HTTP ${response.status}`),
        response,
      );
    }

    return response;
  }

  async *streamChat(request: ChatRequest): AsyncIterable<ChatStreamEvent> {
    try {
      const body = this.createChatCompletionRequestBody(request);
      const reader = await fetchOpenAiCompatibleStreamReader({
        body,
        signal: request.signal,
        fetchStream: (streamBody, signal) =>
          this.fetchStream(streamBody, signal),
        retryOptions: this.buildRetryOptions(
          request,
          this.model,
          this.retryConfig,
          this.onDiagnosticEvent,
        ),
        translateError: (error) => this.translateError(error),
      });

      const toolCallsByIndex = new Map<
        number,
        OpenAIStreamToolCallAccumulator
      >();
      yield* consumeOpenAiChatCompletionsStream(reader, toolCallsByIndex);
    } catch (error) {
      if (error instanceof ProviderError) throw error;
      throw this.translateError(error);
    }
  }

  protected translateError(
    error: unknown,
    response?: Response,
    _responseBody?: string,
  ): ProviderError {
    return this.translateStandardOpenAiError(error, response, {
      model: this.model,
      authenticationMessage: "Invalid Cloudflare API token",
      rateLimitMessage: "Cloudflare rate limit exceeded",
      serviceErrorMessage: "Cloudflare service error",
      connectionErrorMessage: "Failed to connect to Cloudflare API",
      requestFailedMessage: "Cloudflare request failed",
    });
  }
}
