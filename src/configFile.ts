import {
  readJsonFile,
  readJsonFileAsync,
  writeJsonFileAtomic,
} from "./internal/jsonFile.js";
import { ProvidersConfig } from "./config.js";
import {
  validateProvidersConfig,
  updateDefaultProviderModelSelection,
} from "./configValidation.js";

export interface LoadProvidersConfigOptions {
  /**
   * Message thrown when the config file does not exist. Defaults to a
   * generic "Configuration file not found" message; consumers can inject
   * application-specific guidance here.
   */
  readonly missingMessage?: string;
}

function readOptions(filePath: string, options?: LoadProvidersConfigOptions) {
  return {
    invalidJsonPrefix: "Invalid JSON in configuration file",
    missingMessage:
      options?.missingMessage ?? `Configuration file not found: ${filePath}`,
    readErrorPrefix: "Failed to read configuration file",
  };
}

/**
 * Load and validate a ProvidersConfig from a JSON file
 *
 * @param filePath - Path to the providers config JSON file
 * @param options - Optional overrides such as the missing-file message
 * @returns ProvidersConfig object with all validation completed
 * @throws Error if file not found, invalid JSON, validation fails, or references are invalid
 */
export function loadProvidersConfig(
  filePath: string,
  options?: LoadProvidersConfigOptions,
): ProvidersConfig {
  return validateProvidersConfig(
    readJsonFile(filePath, readOptions(filePath, options)),
  );
}

export async function loadProvidersConfigAsync(
  filePath: string,
  options?: LoadProvidersConfigOptions,
): Promise<ProvidersConfig> {
  return validateProvidersConfig(
    await readJsonFileAsync(filePath, readOptions(filePath, options)),
  );
}

export function writeProvidersConfig(
  filePath: string,
  config: ProvidersConfig,
): void {
  writeJsonFileAtomic(filePath, "providers", config);
}

export function updateDefaultProviderModelSelectionInFile(
  filePath: string,
  providerName: string,
  modelKey?: string,
): ProvidersConfig {
  const config = loadProvidersConfig(filePath);
  const updatedConfig = updateDefaultProviderModelSelection(
    config,
    providerName,
    modelKey,
  );
  writeProvidersConfig(filePath, updatedConfig);
  return updatedConfig;
}
