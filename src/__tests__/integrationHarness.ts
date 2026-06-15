import type { LLMProvider } from "../interface.js";
import type { ChatRequest } from "../types.js";

export const KNOWN_INTEGRATION_PROVIDERS = [
  "cloudflare",
  "anthropic",
  "bedrock",
  "ollama",
  "openrouter",
  "gemini",
  "xai",
] as const;

export type IntegrationProvider = (typeof KNOWN_INTEGRATION_PROVIDERS)[number];

export interface EnvRequirement {
  /** One env var name, or alternatives where any one satisfies the requirement. */
  vars: string | string[];
  /** Optional label for error messages (defaults to vars joined with " or "). */
  label?: string;
}

export interface ProviderIntegrationRequirements {
  env: EnvRequirement[];
}

function requirementLabel(requirement: EnvRequirement): string {
  if (requirement.label) {
    return requirement.label;
  }
  return Array.isArray(requirement.vars)
    ? requirement.vars.join(" or ")
    : requirement.vars;
}

function isRequirementSatisfied(requirement: EnvRequirement): boolean {
  const vars = Array.isArray(requirement.vars)
    ? requirement.vars
    : [requirement.vars];
  return vars.some((name) => Boolean(process.env[name]?.trim()));
}

function getMissingEnvRequirements(
  requirements: ProviderIntegrationRequirements,
): EnvRequirement[] {
  return requirements.env.filter(
    (requirement) => !isRequirementSatisfied(requirement),
  );
}

function formatMissingEnvMessage(
  providerName: IntegrationProvider,
  missing: EnvRequirement[],
): string {
  const vars = missing.map((requirement) => requirementLabel(requirement));
  return (
    `Integration test for "${providerName}" is missing required environment variable(s): ` +
    `${vars.join(", ")}. ` +
    `Copy .env.example to .env and set the values for this provider.`
  );
}

export function getSelectedIntegrationProvider():
  | IntegrationProvider
  | undefined {
  const provider = process.env.PROVIDER?.trim().toLowerCase();
  if (!provider) {
    return undefined;
  }
  return provider as IntegrationProvider;
}

function shouldRunProviderIntegration(
  providerName: IntegrationProvider,
): boolean {
  const selected = getSelectedIntegrationProvider();
  return !selected || selected === providerName;
}

export function requireEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return value;
}

export function optionalEnv(name: string): string | undefined {
  const value = process.env[name]?.trim();
  return value || undefined;
}

export async function expectProviderStreamsAssistantText(
  provider: Pick<LLMProvider, "streamChat">,
  request: ChatRequest,
): Promise<void> {
  const assistantText: string[] = [];
  let terminalStopReason: string | undefined;

  for await (const chunk of provider.streamChat(request)) {
    if ("type" in chunk && chunk.type === "assistant_text") {
      assistantText.push(chunk.delta);
    }
    if ("type" in chunk && chunk.type === "terminal") {
      terminalStopReason = chunk.stopReason;
    }
  }

  expect(assistantText.join("")).not.toHaveLength(0);
  expect(terminalStopReason).toBeDefined();
}

export function describeProviderIntegration(
  providerName: IntegrationProvider,
  requirements: ProviderIntegrationRequirements,
  fn: () => void,
): void {
  if (!shouldRunProviderIntegration(providerName)) {
    describe.skip(`${providerName} integration (real API)`, fn);
    return;
  }

  const missing = getMissingEnvRequirements(requirements);
  if (missing.length > 0) {
    describe(`${providerName} integration (real API)`, () => {
      it("requires environment variables", () => {
        throw new Error(formatMissingEnvMessage(providerName, missing));
      });
    });
    return;
  }

  describe(`${providerName} integration (real API)`, fn);
}
