import type { OpenAiCompatibleProviderOptions } from "../internal/openAiCompatibleProvider.js";
import {
  OpenAiResponsesProvider,
  type OpenAiResponsesProviderProfile,
} from "../internal/openAiResponsesProvider.js";

const META_PROFILE: OpenAiResponsesProviderProfile = {
  name: "meta",
  apiUrl: "https://api.meta.ai/v1/responses",
  apiKeyEnv: "MODEL_API_KEY",
  missingApiKeyMessage:
    "Meta Model API key is required. Set MODEL_API_KEY or pass apiKey in options.",
  missingBodyMessage: "Meta Model API response had no body",
  unauthorizedMessage: "Meta Model API key is not authorized for this request",
  capacityMessage: "Meta Model API capacity is temporarily exhausted",
  authenticationMessage: "Invalid Meta Model API key",
  rateLimitMessage: "Meta Model API rate limit exceeded",
  serviceErrorMessage: "Meta Model API service error",
  connectionErrorMessage: "Failed to connect to Meta Model API",
  requestFailedMessage: "Meta Model API request failed",
  preserveAssistantOutputItems: true,
  useCommentaryPhaseForLegacyToolCalls: true,
};

/** Meta Model API implementation using the OpenAI-compatible Responses API. */
export class MetaProvider extends OpenAiResponsesProvider {
  constructor(options: OpenAiCompatibleProviderOptions) {
    super(options, META_PROFILE);
  }
}
