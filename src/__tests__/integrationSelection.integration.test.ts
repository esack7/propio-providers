import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

import {
  KNOWN_INTEGRATION_PROVIDERS,
  getSelectedIntegrationProvider,
} from "./integrationHarness.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const implementedIntegrationProviders = new Set(
  fs
    .readdirSync(__dirname)
    .filter((file) => file.endsWith(".integration.test.ts"))
    .map((file) => file.replace(/\.integration\.test\.ts$/, ""))
    .filter(
      (provider): provider is (typeof KNOWN_INTEGRATION_PROVIDERS)[number] =>
        KNOWN_INTEGRATION_PROVIDERS.includes(
          provider as (typeof KNOWN_INTEGRATION_PROVIDERS)[number],
        ),
    ),
);

describe("integration provider selection", () => {
  it("has a matching integration file for PROVIDER", () => {
    const selectedProvider = getSelectedIntegrationProvider();
    if (!selectedProvider) {
      return;
    }

    if (!implementedIntegrationProviders.has(selectedProvider)) {
      throw new Error(
        `No integration test file has been implemented for provider "${selectedProvider}". ` +
          `Available provider integration files: ${
            Array.from(implementedIntegrationProviders).join(", ") || "none"
          }.`,
      );
    }
  });
});
