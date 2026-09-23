import { LLMProvider, ProviderCapabilities } from "../interface.js";
import { createProviderCapabilities } from "./capabilities.js";
import { ChatRequest, ChatStreamEvent } from "../types.js";
import type {
  ProviderDiagnosticListener,
  ProviderRetryConfig,
} from "../diagnostics.js";
import { createProviderRetryOptions } from "./shared.js";

export interface BaseProviderOptions {
  model: string;
  contextWindowTokens: number;
  retryConfig?: ProviderRetryConfig;
  onDiagnosticEvent?: ProviderDiagnosticListener;
}

export abstract class BaseProvider implements LLMProvider {
  abstract readonly name: string;
  protected model: string;
  protected capabilities: ProviderCapabilities;
  protected retryConfig?: ProviderRetryConfig;
  protected onDiagnosticEvent?: ProviderDiagnosticListener;

  constructor(options: BaseProviderOptions) {
    this.model = options.model;
    this.capabilities = createProviderCapabilities(options.contextWindowTokens);
    this.retryConfig = options.retryConfig;
    this.onDiagnosticEvent = options.onDiagnosticEvent;
  }

  // Public LLMProvider contract; callers receive providers through the factory interface.
  getCapabilities(): ProviderCapabilities {
    return this.capabilities;
  }

  protected buildBaseRetryOptions(
    request: ChatRequest,
    isRetryable: (error: unknown) => boolean,
    baseDelayMs = 500,
    endpointClass?: string | ((attemptNumber: number) => string),
    requestPayload?: (attemptNumber: number) => {
      readonly transport: "http_json" | "sdk_input";
      readonly requestBody: unknown;
    },
  ) {
    return createProviderRetryOptions({
      request,
      model: this.model,
      provider: this.name,
      retryConfig: this.retryConfig
        ? { ...this.retryConfig, baseDelayMs }
        : undefined,
      isRetryable,
      onDiagnosticEvent: this.onDiagnosticEvent,
      endpointClass,
      requestPayload,
    });
  }

  abstract streamChat(request: ChatRequest): AsyncIterable<ChatStreamEvent>;
}
