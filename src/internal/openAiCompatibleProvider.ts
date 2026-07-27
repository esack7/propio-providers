import type { ProviderDiagnosticListener } from "../diagnostics.js";
import type { LLMProvider, ProviderCapabilities } from "../interface.js";
import type {
  ChatMessage,
  ChatRequest,
  ChatStreamEvent,
  ChatTool,
} from "../types.js";
import { createProviderCapabilities } from "./capabilities.js";
import type { WithRetryOptions } from "./withRetry.js";
import {
  ProviderError,
  ProviderAuthenticationError,
  ProviderContextLengthError,
  ProviderInvalidRequestError,
  ProviderModelNotFoundError,
  ProviderRateLimitError,
} from "../types.js";
import {
  OpenAIMessageCore,
  OpenAIToolDefinition,
  applyOpenAIMessageCore,
  createOpenAIToolDefinition,
  createProviderRetryOptions,
  isAbortOrTransportError,
  isContextLengthError,
  normalizeErrorMessage,
  parseRetryAfterSeconds,
} from "./shared.js";

export interface OpenAiCompatibleRetryConfig {
  readonly maxRetries: number;
  readonly consecutive529Limit: number;
  readonly baseDelayMs?: number;
}

export interface OpenAiCompatibleProviderOptions {
  readonly model: string;
  readonly contextWindowTokens: number;
  readonly apiKey?: string;
  readonly retryConfig?: OpenAiCompatibleRetryConfig;
  readonly onDiagnosticEvent?: ProviderDiagnosticListener;
}

interface StandardOpenAiErrorOptions {
  readonly model: string;
  readonly authenticationMessage: string;
  readonly rateLimitMessage: string;
  readonly serviceErrorMessage: string;
  readonly connectionErrorMessage: string;
  readonly requestFailedMessage: string;
}

export abstract class OpenAiCompatibleProvider implements LLMProvider {
  abstract readonly name: string;
  private capabilities?: ProviderCapabilities;

  abstract streamChat(request: ChatRequest): AsyncIterable<ChatStreamEvent>;
  protected abstract translateError(
    error: unknown,
    response?: Response,
    responseBody?: string,
  ): ProviderError;

  protected chatMessageToOpenAIMessage(msg: ChatMessage): OpenAIMessageCore {
    const role = msg.role as OpenAIMessageCore["role"];
    const out: OpenAIMessageCore = { role, content: msg.content ?? "" };
    return applyOpenAIMessageCore(out, msg);
  }

  protected chatToolToOpenAITool(tool: ChatTool): OpenAIToolDefinition {
    return createOpenAIToolDefinition(tool);
  }

  getCapabilities(): ProviderCapabilities {
    if (!this.capabilities) {
      throw new ProviderError("Provider capabilities were not configured");
    }
    return this.capabilities;
  }

  protected configureCapabilities(contextWindowTokens: number): void {
    this.capabilities = createProviderCapabilities(contextWindowTokens);
  }

  protected createOriginalError(error: unknown): Error {
    return error instanceof Error ? error : new Error(String(error));
  }

  protected buildRetryOptions(
    request: ChatRequest,
    model: string,
    retryConfig: OpenAiCompatibleRetryConfig | undefined,
    onDiagnosticEvent?: ProviderDiagnosticListener,
  ): WithRetryOptions {
    return createProviderRetryOptions({
      request,
      model,
      provider: this.name,
      retryConfig,
      isRetryable: (err) => this.isRetryableError(err),
      onDiagnosticEvent,
    });
  }

  protected translateCommonMessageError(
    msg: string | undefined,
    originalError: Error,
    connectionErrorMessage: string,
  ): ProviderError | null {
    if (!msg) {
      return null;
    }

    if (isContextLengthError(msg)) {
      return new ProviderContextLengthError(
        `Context length exceeded: ${msg}`,
        originalError,
      );
    }

    if (msg.includes("AbortError") || msg.includes("aborted")) {
      return new ProviderError("Request cancelled", originalError);
    }

    if (isAbortOrTransportError(msg)) {
      return new ProviderError(connectionErrorMessage, originalError);
    }

    return null;
  }

  protected translateStandardOpenAiError(
    error: unknown,
    response: Response | undefined,
    options: StandardOpenAiErrorOptions,
  ): ProviderError {
    const originalError = this.createOriginalError(error);
    const normalizedMessage = normalizeErrorMessage(originalError.message);

    if (response) {
      const responseError = this.translateStandardOpenAiResponseError(
        response,
        normalizedMessage,
        originalError,
        options,
      );
      if (responseError) {
        return responseError;
      }
    }

    return (
      this.translateCommonMessageError(
        normalizedMessage || originalError.message,
        originalError,
        options.connectionErrorMessage,
      ) ??
      new ProviderError(
        originalError.message || options.requestFailedMessage,
        originalError,
      )
    );
  }

  private translateStandardOpenAiResponseError(
    response: Response,
    normalizedMessage: string,
    originalError: Error,
    options: StandardOpenAiErrorOptions,
  ): ProviderError | null {
    return (
      this.translateClientResponseError(
        response,
        normalizedMessage,
        originalError,
        options,
      ) ??
      this.translateServerResponseError(
        response,
        normalizedMessage,
        originalError,
        options,
      )
    );
  }

  private translateClientResponseError(
    response: Response,
    normalizedMessage: string,
    originalError: Error,
    options: StandardOpenAiErrorOptions,
  ): ProviderError | null {
    switch (response.status) {
      case 400:
        return this.translateBadRequestError(normalizedMessage, originalError);
      case 401:
        return new ProviderAuthenticationError(
          options.authenticationMessage,
          originalError,
        );
      case 429:
        return new ProviderRateLimitError(
          options.rateLimitMessage,
          parseRetryAfterSeconds(response.headers.get("retry-after")),
          originalError,
        );
      case 404:
        return new ProviderModelNotFoundError(
          options.model,
          `Model not found: ${options.model}`,
          originalError,
        );
      default:
        return null;
    }
  }

  private translateBadRequestError(
    normalizedMessage: string,
    originalError: Error,
  ): ProviderError {
    if (isContextLengthError(normalizedMessage)) {
      return new ProviderContextLengthError(
        `Context length exceeded: ${normalizedMessage}`,
        originalError,
      );
    }

    return new ProviderInvalidRequestError(
      normalizedMessage || "Invalid request",
      originalError,
    );
  }

  private translateServerResponseError(
    response: Response,
    normalizedMessage: string,
    originalError: Error,
    options: StandardOpenAiErrorOptions,
  ): ProviderError | null {
    if (response.status < 500 || response.status >= 600) {
      return null;
    }

    const message = normalizedMessage
      ? `${options.serviceErrorMessage}: ${normalizedMessage}`
      : options.serviceErrorMessage;
    return new ProviderError(message, originalError);
  }

  protected isRetryableError(err: unknown): boolean {
    if (err instanceof ProviderAuthenticationError) return false;
    if (err instanceof ProviderContextLengthError) return false;
    if (err instanceof ProviderInvalidRequestError) return false;
    if (err instanceof ProviderModelNotFoundError) return false;
    return err instanceof ProviderError;
  }
}
