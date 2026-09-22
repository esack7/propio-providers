import type { ChatRequest } from "../types.js";
import {
  emitProviderResponseMetadata,
  emitProviderUsage,
  type ProviderReportedCost,
  type ProviderTokenUsage,
} from "../trace.js";

export interface ProviderTokenUsageInput {
  readonly inputTokens?: unknown;
  readonly outputTokens?: unknown;
  readonly cacheReadInputTokens?: unknown;
  readonly cacheWriteInputTokens?: unknown;
  readonly reasoningTokens?: unknown;
  readonly totalTokens?: unknown;
}

function normalizeProviderTokenUsage(
  input: ProviderTokenUsageInput,
): ProviderTokenUsage {
  const usage: Record<string, number> = {};
  for (const [key, value] of Object.entries(input)) {
    if (typeof value === "number" && Number.isFinite(value) && value >= 0) {
      usage[key] = value;
    }
  }
  return usage;
}

export function emitNormalizedProviderUsage(options: {
  readonly provider: string;
  readonly request: ChatRequest;
  readonly endpointClass: string;
  readonly usage: ProviderTokenUsageInput;
  readonly reportKind?: "delta" | "cumulative";
  readonly providerReportedCost?: ProviderReportedCost;
}): void {
  const usage = normalizeProviderTokenUsage(options.usage);
  if (Object.keys(usage).length === 0) return;
  emitProviderUsage({
    provider: options.provider,
    request: options.request,
    endpointClass: options.endpointClass,
    availability:
      usage.inputTokens !== undefined && usage.outputTokens !== undefined
        ? "reported"
        : "partial",
    reportKind: options.reportKind ?? "cumulative",
    usage,
    ...(options.providerReportedCost
      ? { providerReportedCost: options.providerReportedCost }
      : {}),
  });
}

interface OpenAiUsagePayload {
  readonly prompt_tokens?: number;
  readonly completion_tokens?: number;
  readonly input_tokens?: number;
  readonly output_tokens?: number;
  readonly total_tokens?: number;
  readonly prompt_tokens_details?: { readonly cached_tokens?: number };
  readonly completion_tokens_details?: { readonly reasoning_tokens?: number };
  readonly input_tokens_details?: { readonly cached_tokens?: number };
  readonly output_tokens_details?: { readonly reasoning_tokens?: number };
}

interface OpenAiTraceChunk {
  readonly id?: string;
  readonly model?: string;
  readonly usage?: OpenAiUsagePayload & { readonly cost?: number };
  readonly usage_metadata?: {
    readonly prompt_token_count?: number;
    readonly candidates_token_count?: number;
    readonly total_token_count?: number;
    readonly cached_content_token_count?: number;
    readonly thoughts_token_count?: number;
  };
}

export function emitOpenAiCompatibleSseTrace(options: {
  readonly data: string;
  readonly provider: string;
  readonly request: ChatRequest;
  readonly endpointClass: string;
  readonly costCurrency?: string;
  readonly state?: { responseMetadataObserved: boolean };
}): OpenAiTraceChunk | undefined {
  if (options.data === "[DONE]") return undefined;
  let chunk: OpenAiTraceChunk;
  try {
    chunk = JSON.parse(options.data) as OpenAiTraceChunk;
  } catch {
    return undefined;
  }
  emitOpenAiCompatibleTrace({
    provider: options.provider,
    request: options.request,
    endpointClass: options.endpointClass,
    responseId: chunk.id,
    actualModel: chunk.model,
    usage: chunk.usage,
    includeResponseMetadata: !options.state?.responseMetadataObserved,
    ...(typeof chunk.usage?.cost === "number"
      ? {
          providerReportedCost: {
            amount: chunk.usage.cost,
            ...(options.costCurrency ? { currency: options.costCurrency } : {}),
          },
        }
      : {}),
  });
  if (options.state && (chunk.id || chunk.model)) {
    options.state.responseMetadataObserved = true;
  }
  return chunk;
}

export function emitOpenAiCompatibleTrace(options: {
  readonly provider: string;
  readonly request: ChatRequest;
  readonly endpointClass: string;
  readonly responseId?: string;
  readonly actualModel?: string;
  readonly usage?: OpenAiUsagePayload;
  readonly providerReportedCost?: ProviderReportedCost;
  readonly includeResponseMetadata?: boolean;
}): void {
  if (
    options.includeResponseMetadata !== false &&
    (options.responseId || options.actualModel)
  ) {
    emitProviderResponseMetadata({
      provider: options.provider,
      request: options.request,
      endpointClass: options.endpointClass,
      ...(options.responseId ? { upstreamRequestId: options.responseId } : {}),
      ...(options.actualModel ? { actualModel: options.actualModel } : {}),
    });
  }
  if (!options.usage) return;
  emitNormalizedProviderUsage({
    provider: options.provider,
    request: options.request,
    endpointClass: options.endpointClass,
    usage: {
      inputTokens: options.usage.prompt_tokens ?? options.usage.input_tokens,
      outputTokens:
        options.usage.completion_tokens ?? options.usage.output_tokens,
      totalTokens: options.usage.total_tokens,
      cacheReadInputTokens:
        options.usage.prompt_tokens_details?.cached_tokens ??
        options.usage.input_tokens_details?.cached_tokens,
      reasoningTokens:
        options.usage.completion_tokens_details?.reasoning_tokens ??
        options.usage.output_tokens_details?.reasoning_tokens,
    },
    ...(options.providerReportedCost
      ? { providerReportedCost: options.providerReportedCost }
      : {}),
  });
}
