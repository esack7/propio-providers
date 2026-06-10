import { LLMProvider, ProviderCapabilities } from "../interface.js";
import { createProviderCapabilities } from "./capabilities.js";
import { ChatRequest, ChatStreamEvent } from "../types.js";
import type {
  ProviderDiagnosticListener,
  ProviderRetryConfig,
} from "../diagnostics.js";

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

  getCapabilities(): ProviderCapabilities {
    return this.capabilities;
  }

  abstract streamChat(request: ChatRequest): AsyncIterable<ChatStreamEvent>;
}
