import type { ProviderCapabilities } from "../interface.js";

export function validateContextWindowTokens(
  value: unknown,
  label: string,
): number {
  if (!Number.isInteger(value) || (value as number) <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }

  return value as number;
}

export function createProviderCapabilities(
  contextWindowTokens: number,
): ProviderCapabilities {
  return {
    contextWindowTokens: validateContextWindowTokens(
      contextWindowTokens,
      "contextWindowTokens",
    ),
  };
}
