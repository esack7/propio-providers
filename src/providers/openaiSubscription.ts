import type { ChatRequest } from "../types.js";
import { ProviderError, ProviderContextLengthError } from "../types.js";
import {
  OpenAiResponsesProvider,
  type OpenAiResponsesProviderOptions,
} from "../internal/openAiResponsesProvider.js";

function record(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null
    ? (value as Record<string, unknown>)
    : {};
}

function stringField(
  value: Record<string, unknown>,
  key: string,
): string | undefined {
  const candidate = value[key];
  return typeof candidate === "string" &&
    /^[a-zA-Z0-9_.:-]{1,160}$/.test(candidate)
    ? candidate
    : undefined;
}

function planErrorDetails(body?: string) {
  let parsed: Record<string, unknown> = {};
  try {
    parsed = record(JSON.parse(body ?? "{}"));
  } catch {
    /* Status identifies non-JSON failures. */
  }
  const error = record(parsed.error);
  return {
    code: stringField(error, "code"),
    param: stringField(error, "param"),
  };
}

export interface OpenAiSubscriptionOptions extends Omit<
  OpenAiResponsesProviderOptions,
  "apiKey" | "getBearerToken"
> {
  readonly getAccessToken: (signal?: AbortSignal) => Promise<string>;
}

/** Safe subscription metadata; upstream message/body may echo credentials. */
export class ChatGptPlanError extends ProviderError {
  constructor(
    message: string,
    readonly code?: string,
    readonly status?: number,
    readonly requestId?: string,
    readonly param?: string,
    readonly retryable = false,
    readonly retryAfterSeconds?: number,
  ) {
    super(message);
    this.name = "ChatGptPlanError";
  }
}

/** Direct plan access. Authentication, storage and billing selection belong to the caller. */
export class OpenAiSubscriptionProvider extends OpenAiResponsesProvider {
  constructor(options: OpenAiSubscriptionOptions) {
    super(
      { ...options, getBearerToken: options.getAccessToken },
      {
        name: "openai-chatgpt",
        apiUrl: "https://api.openai.com/v1/responses",
        apiKeyEnv: "PROPIO_UNUSED_SUBSCRIPTION_API_KEY",
        missingApiKeyMessage: "Continue with ChatGPT before using plan access.",
        missingBodyMessage: "Subscription response had no body",
        unauthorizedMessage: "ChatGPT plan permission was denied",
        capacityMessage: "ChatGPT plan usage is temporarily unavailable",
        authenticationMessage: "ChatGPT sign-in is required",
        rateLimitMessage: "ChatGPT plan usage limit reached",
        serviceErrorMessage: "ChatGPT plan service error",
        connectionErrorMessage: "Failed to connect to ChatGPT plan inference",
        requestFailedMessage: "ChatGPT plan inference failed",
        preserveAssistantCommentary: true,
        routeCommentaryToThinking: true,
        requireCompletedResponse: true,
      },
    );
  }

  getCapabilities() {
    return { ...super.getCapabilities(), supportsImages: false };
  }

  protected createRequestBody(request: ChatRequest): Record<string, unknown> {
    if (request.messages.some((message) => message.images?.length))
      throw new ChatGptPlanError(
        "Images have not been verified for ChatGPT plan access. Use text or explicitly select an API provider.",
        "unsupported_input",
      );
    if (request.tools?.some((tool) => tool.type !== "function"))
      throw new ChatGptPlanError(
        "ChatGPT plan access supports local function tools only.",
        "unsupported_tool",
      );
    const body = super.createRequestBody(request);
    body.input = (body.input as Record<string, unknown>[]).map((item) =>
      item.role === "system" ? { ...item, role: "developer" } : item,
    );
    if (body.tools) {
      body.tools = [
        {
          type: "namespace",
          name: "functions",
          description: "Tools executed locally by Propio.",
          tools: body.tools,
        },
      ];
    }
    return body;
  }

  protected translateError(
    error: unknown,
    response?: Response,
    responseBody?: string,
  ): ProviderError {
    if (error instanceof ProviderError) return error;
    if (error instanceof Error && error.name === "AbortError") throw error;
    if (!responseBody && !response)
      return new ChatGptPlanError(
        "ChatGPT transport or credential acquisition failed; no automatic replay.",
        "transport_error",
      );
    const details = planErrorDetails(responseBody);
    const metadata = planResponseMetadata(response);
    if (details.code === "context_length_exceeded")
      return Object.assign(
        new ProviderContextLengthError(
          "ChatGPT prompt exceeds the model context window.",
        ),
        {
          code: details.code,
          ...metadata,
        },
      );
    const retryable = retryableAdmission(details.code, metadata.status);
    const message = planFailureMessage(details.code);
    return new ChatGptPlanError(
      message,
      details.code,
      metadata.status,
      metadata.requestId,
      details.param,
      retryable,
      metadata.retryAfterSeconds,
    );
  }

  protected isRetryableError(error: unknown): boolean {
    if (error instanceof ChatGptPlanError) {
      return error.retryable;
    }
    return false;
  }
}

function retryableAdmission(
  code: string | undefined,
  status: number | undefined,
): boolean {
  return (
    [
      "usage_temporarily_unavailable",
      "user_temporarily_unavailable",
      "routing_temporarily_unavailable",
      "server_error",
    ].includes((code ?? "").replace(/^subscription_sharing_/, "")) &&
    [502, 503, 504].includes(status ?? 0)
  );
}
function planFailureMessage(code?: string): string {
  return code?.includes("usage_limit_exceeded")
    ? "ChatGPT plan usage limit reached. Manage usage: https://chatgpt.com/settings/usage. Select API billing explicitly to switch."
    : "ChatGPT plan request failed; check the selected account, permission and supported inputs.";
}

function planResponseMetadata(response?: Response) {
  if (!response)
    return {
      status: undefined,
      requestId: undefined,
      retryAfterSeconds: undefined,
    };
  return {
    status: response.status,
    requestId: response.headers.get("x-request-id") ?? undefined,
    retryAfterSeconds: Number(response.headers.get("retry-after")) || undefined,
  };
}
