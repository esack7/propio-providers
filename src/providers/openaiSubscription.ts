import type { ChatRequest } from "../types.js";
import {
  ProviderError,
  ProviderContextLengthError,
  ProviderRateLimitError,
  ProviderAuthenticationError,
} from "../types.js";
import {
  OpenAiResponsesProvider,
  type OpenAiResponsesProviderOptions,
} from "../internal/openAiResponsesProvider.js";
import { parseRetryAfterSeconds } from "../internal/shared.js";

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
    if (typeof options.getAccessToken !== "function")
      throw new ProviderAuthenticationError(
        "ChatGPT access requires an application-owned getAccessToken function.",
      );
    if (
      options.idleTimeoutMs !== undefined &&
      (!Number.isSafeInteger(options.idleTimeoutMs) ||
        options.idleTimeoutMs <= 0 ||
        options.idleTimeoutMs > 2147483647)
    )
      throw new Error(
        "Subscription idleTimeoutMs must be a positive integer no greater than 2147483647.",
      );
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
        authenticationMessage:
          "ChatGPT sign-in is required. Check the selected account and granted scopes.",
        rateLimitMessage: "ChatGPT request rate limit reached",
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
    if (record(error).name === "AbortError")
      return new ProviderError("Request cancelled");
    if (!responseBody && !response)
      return new ChatGptPlanError(
        "ChatGPT transport or credential acquisition failed; no automatic replay.",
        "transport_error",
      );
    const details = planErrorDetails(responseBody);
    const metadata = planResponseMetadata(response);
    const typed = typedPlanError(details.code);
    if (typed) return Object.assign(typed, details, metadata);
    if (isStandardHttpFailure(details.code, response))
      return Object.assign(
        super.translateError(
          new Error("ChatGPT plan request failed."),
          response,
        ),
        details,
        metadata,
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
    // A generic HTTP 429 is safe to retry before inference starts. Plan usage
    // limits and ambiguous transport failures must never become replay retries.
    return error instanceof ProviderRateLimitError;
  }
}

function typedPlanError(code?: string): ProviderError | undefined {
  if (code === "subscription_sharing_invalid_user")
    return new ProviderAuthenticationError(
      "ChatGPT subscriber context was not accepted. Check the selected account and granted scopes.",
    );
  if (code === "context_length_exceeded")
    return new ProviderContextLengthError(
      "ChatGPT prompt exceeds the model context window.",
    );
  return undefined;
}

const TEMPORARY_ADMISSION_CODES = new Set([
  "usage_temporarily_unavailable",
  "user_temporarily_unavailable",
  "usage_unavailable",
  "user_unavailable",
  "routing_temporarily_unavailable",
  "server_error",
]);
function isTemporaryAdmissionCode(code?: string): boolean {
  return TEMPORARY_ADMISSION_CODES.has(
    (code ?? "").replace(/^subscription_sharing_/, ""),
  );
}
function retryableAdmission(
  code: string | undefined,
  status: number | undefined,
): boolean {
  return (
    isTemporaryAdmissionCode(code) && [502, 503, 504].includes(status ?? 0)
  );
}
function isPlanSpecificCode(code?: string): boolean {
  return (
    code?.startsWith("subscription_sharing_") === true ||
    code?.startsWith("chatpass_v2_") === true ||
    isTemporaryAdmissionCode(code)
  );
}
function isStandardHttpFailure(
  code: string | undefined,
  response?: Response,
): response is Response {
  return Boolean(
    response && response.status >= 400 && !isPlanSpecificCode(code),
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
    retryAfterSeconds: parseRetryAfterSeconds(
      response.headers.get("retry-after"),
    ),
  };
}
