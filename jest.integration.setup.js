import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.join(__dirname, ".env"), quiet: true });

const KNOWN_PROVIDERS = [
  "cloudflare",
  "anthropic",
  "bedrock",
  "ollama",
  "openrouter",
  "gemini",
  "xai",
];

const provider = process.env.PROVIDER?.trim().toLowerCase();
if (provider && !KNOWN_PROVIDERS.includes(provider)) {
  throw new Error(
    `Unknown integration test provider "${process.env.PROVIDER}". ` +
      `Known providers: ${KNOWN_PROVIDERS.join(", ")}`,
  );
}
