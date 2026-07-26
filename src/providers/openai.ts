import type { OpenAiCompatibleProviderOptions } from "../internal/openAiCompatibleProvider.js";
import {
  OpenAiResponsesProvider,
  type OpenAiResponsesProviderProfile,
} from "../internal/openAiResponsesProvider.js";

const OPENAI_PROFILE: OpenAiResponsesProviderProfile = {
  name: "openai",
  apiUrl: "https://api.openai.com/v1/responses",
  apiKeyEnv: "OPENAI_API_KEY",
  missingApiKeyMessage:
    "OpenAI API key is required. Set OPENAI_API_KEY or pass apiKey in options.",
  missingBodyMessage: "OpenAI response had no body",
  unauthorizedMessage: "OpenAI API key is not authorized for this request",
  capacityMessage: "OpenAI capacity is temporarily exhausted",
  authenticationMessage: "Invalid OpenAI API key",
  rateLimitMessage: "OpenAI rate limit exceeded",
  serviceErrorMessage: "OpenAI service error",
  connectionErrorMessage: "Failed to connect to OpenAI API",
  requestFailedMessage: "OpenAI request failed",
};

/** OpenAI implementation using the Responses API. */
export class OpenAiProvider extends OpenAiResponsesProvider {
  constructor(options: OpenAiCompatibleProviderOptions) {
    super(options, OPENAI_PROFILE);
  }
}
