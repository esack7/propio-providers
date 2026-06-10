import { ProvidersConfig, ProviderConfig } from "./config.js";
import { validateContextWindowTokens } from "./internal/capabilities.js";

export interface ProviderModelSelection {
  readonly providerName: string;
  readonly modelKey: string;
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function isNonEmptyStringArray(value: unknown): value is string[] {
  return (
    Array.isArray(value) &&
    value.length > 0 &&
    value.every((item) => isNonEmptyString(item))
  );
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Validate an arbitrary value as a ProvidersConfig
 *
 * @param config - The parsed configuration value to validate
 * @returns ProvidersConfig object with all validation completed
 * @throws Error if validation fails or references are invalid
 */
export function validateProvidersConfig(config: unknown): ProvidersConfig {
  if (!isPlainObject(config)) {
    throw new Error("Configuration root must be a JSON object");
  }
  const candidate = config as any;

  // Validate required fields exist
  if (!candidate.providers) {
    throw new Error('Configuration must include a "providers" array');
  }
  if (candidate.default === undefined) {
    throw new Error(
      'Configuration must include a "default" field specifying default provider',
    );
  }

  // Validate that default references an existing provider
  const defaultProviderExists = candidate.providers.some(
    (p: any) => p.name === candidate.default,
  );
  if (!defaultProviderExists) {
    const availableProviders = candidate.providers
      .map((p: any) => p.name)
      .join(", ");
    throw new Error(
      `Default provider "${candidate.default}" not found in providers list. Available: ${availableProviders}`,
    );
  }

  // Validate each provider
  const seenNames = new Set<string>();
  for (const provider of candidate.providers) {
    validateProviderConfig(provider, seenNames);
  }

  return candidate as unknown as ProvidersConfig;
}

/**
 * Validate a single provider configuration
 */
function validateProviderConfig(provider: any, seenNames: Set<string>): void {
  validateProviderRequiredFields(provider);
  validateUniqueProviderName(provider.name, seenNames);
  validateProviderModels(provider);

  if (provider.type === "openrouter") {
    validateOpenRouterProviderConfig(provider);
  }

  validateDefaultModel(provider);
}

function validateProviderRequiredFields(provider: any): void {
  const requiredFields = ["name", "type", "models", "defaultModel"];
  const missingFields = requiredFields.filter((field) => !provider[field]);
  if (missingFields.length > 0) {
    throw new Error(
      `Provider is missing required fields: ${missingFields.join(", ")}`,
    );
  }
}

function validateUniqueProviderName(
  providerName: string,
  seenNames: Set<string>,
): void {
  if (seenNames.has(providerName)) {
    throw new Error(
      `Duplicate provider name: "${providerName}". Provider names must be unique.`,
    );
  }
  seenNames.add(providerName);
}

function validateProviderModels(provider: any): void {
  if (!Array.isArray(provider.models) || provider.models.length === 0) {
    throw new Error(
      `Provider "${provider.name}" must have at least one model in the models array`,
    );
  }

  const seenModelKeys = new Set<string>();
  for (const model of provider.models) {
    validateProviderModel(provider.name, model, seenModelKeys);
  }
}

function validateProviderModel(
  providerName: string,
  model: any,
  seenModelKeys: Set<string>,
): void {
  validateModelRequiredFields(providerName, model);
  validateContextWindowTokens(
    model.contextWindowTokens,
    `Provider "${providerName}" model "${model.key}" contextWindowTokens`,
  );
  validateUniqueModelKey(providerName, model.key, seenModelKeys);
}

function validateModelRequiredFields(providerName: string, model: any): void {
  if (!model.name || !model.key) {
    throw new Error(
      `Provider "${providerName}" has model missing required fields: each model must have "name", "key", and "contextWindowTokens"`,
    );
  }

  if (model.contextWindowTokens === undefined) {
    throw new Error(
      `Provider "${providerName}" model "${model.key}" is missing required field "contextWindowTokens"`,
    );
  }
}

function validateUniqueModelKey(
  providerName: string,
  modelKey: string,
  seenModelKeys: Set<string>,
): void {
  if (seenModelKeys.has(modelKey)) {
    throw new Error(
      `Provider "${providerName}" has duplicate model key: "${modelKey}". Model keys must be unique within a provider.`,
    );
  }
  seenModelKeys.add(modelKey);
}

function validateDefaultModel(provider: any): void {
  const defaultModelExists = provider.models.some(
    (m: any) => m.key === provider.defaultModel,
  );
  if (!defaultModelExists) {
    const availableModels = provider.models.map((m: any) => m.key).join(", ");
    throw new Error(
      `Provider "${provider.name}" defaultModel "${provider.defaultModel}" not found in models list. Available: ${availableModels}`,
    );
  }
}

function validateOpenRouterProviderConfig(provider: any): void {
  if (provider.provider !== undefined) {
    validateOpenRouterRoutingConfig(provider.name, provider.provider);
  }

  validateOptionalStringArray(
    provider.name,
    "fallbackModels",
    provider.fallbackModels,
  );
  validateOptionalBoolean(
    provider.name,
    "debugEchoUpstreamBody",
    provider.debugEchoUpstreamBody,
  );
}

function validateOpenRouterRoutingConfig(
  providerName: string,
  routing: unknown,
): void {
  if (!isPlainObject(routing)) {
    throw new Error(
      `Provider "${providerName}" OpenRouter "provider" field must be an object`,
    );
  }

  validateOptionalBoolean(
    providerName,
    "provider.allowFallbacks",
    routing.allowFallbacks,
  );
  validateOptionalBoolean(
    providerName,
    "provider.requireParameters",
    routing.requireParameters,
  );
  validateOptionalStringArray(providerName, "provider.order", routing.order);
}

function validateOptionalBoolean(
  providerName: string,
  fieldName: string,
  value: unknown,
): void {
  if (value !== undefined && typeof value !== "boolean") {
    throw new Error(
      `Provider "${providerName}" OpenRouter "${fieldName}" must be a boolean`,
    );
  }
}

function validateOptionalStringArray(
  providerName: string,
  fieldName: string,
  value: unknown,
): void {
  if (value !== undefined && !isNonEmptyStringArray(value)) {
    throw new Error(
      `Provider "${providerName}" OpenRouter "${fieldName}" must be a non-empty array of non-empty strings`,
    );
  }
}

/**
 * Resolve a provider from ProvidersConfig by name
 *
 * @param config - The providers configuration
 * @param providerName - Optional provider name. If not provided, uses config.default
 * @returns The resolved ProviderConfig
 * @throws Error if provider not found
 */
export function resolveProvider(
  config: ProvidersConfig,
  providerName?: string,
): ProviderConfig {
  const name = providerName || config.default;

  const provider = config.providers.find((p) => p.name === name);
  if (!provider) {
    const availableProviders = config.providers.map((p) => p.name).join(", ");
    throw new Error(
      `Unknown provider: "${name}". Available providers: ${availableProviders}`,
    );
  }

  return provider;
}

/**
 * Resolve a model key from a ProviderConfig
 *
 * @param provider - The provider configuration
 * @param modelKey - Optional model key. If not provided, uses provider.defaultModel
 * @returns The resolved model key
 * @throws Error if model key not found
 */
export function resolveModelKey(
  provider: ProviderConfig,
  modelKey?: string,
): string {
  const key = modelKey || provider.defaultModel;

  const model = provider.models.find((m) => m.key === key);
  if (!model) {
    const availableModels = provider.models.map((m) => m.key).join(", ");
    throw new Error(
      `Invalid model key: "${key}" for provider "${provider.name}". Available models: ${availableModels}`,
    );
  }

  return key;
}

export function getDefaultProviderModelSelection(
  config: ProvidersConfig,
): ProviderModelSelection {
  const provider = resolveProvider(config);
  return {
    providerName: provider.name,
    modelKey: provider.defaultModel,
  };
}

export function updateDefaultProviderModelSelection(
  config: ProvidersConfig,
  providerName: string,
  modelKey?: string,
): ProvidersConfig {
  const provider = resolveProvider(config, providerName);
  const resolvedModelKey = resolveModelKey(provider, modelKey);

  const updatedConfig = {
    ...config,
    default: provider.name,
    providers: config.providers.map((entry) =>
      entry.name === provider.name
        ? { ...entry, defaultModel: resolvedModelKey }
        : entry,
    ),
  };

  return validateProvidersConfig(updatedConfig);
}
