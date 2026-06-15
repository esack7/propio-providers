# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project Overview

`@propio-ai/providers` is an ESM-only TypeScript package that exposes provider adapters for LLM APIs behind a unified streaming chat interface. It supports providers such as Anthropic, AWS Bedrock, Ollama, OpenRouter, Gemini, xAI, and Cloudflare Workers AI.

The public surface is exported from `src/index.ts` and built to `dist/`.

## Repository Layout

- `src/interface.ts`, `src/types.ts`, and `src/diagnostics.ts` define shared contracts and events.
- `src/factory.ts` creates provider instances from config.
- `src/config*.ts` and `src/internal/jsonFile.ts` handle provider config validation and file operations.
- `src/internal/` contains reusable provider machinery such as retries, OpenAI-compatible streaming, capabilities, and base provider behavior.
- `src/providers/` contains concrete provider implementations.
- `src/__tests__/` contains Jest unit tests and integration tests.
- `jest.integration.config.cjs` and `jest.integration.setup.js` configure live-provider integration tests.

## Development Guidelines

- Keep the package ESM-compatible and Node.js >= 20 compatible.
- Prefer small, provider-local changes unless a shared behavior genuinely belongs in `src/internal/`.
- Preserve the async iterable `streamChat` contract and existing `ChatStreamEvent` shapes.
- When adding provider behavior, update capabilities and tests so callers can reason about differences between providers.
- Avoid leaking provider SDK-specific types through the public API unless that is an intentional API change.
- Do not commit generated `dist/` output unless the maintainer explicitly asks for release artifacts.
- Keep secrets out of the repo. Integration tests that require live credentials should read them from environment variables.

## Testing and Validation

Run relevant checks after changes:

- `npm test` for unit/integration coverage.
- `npm run build` to verify TypeScript compilation.
- `npm run format:check` for formatting compliance when touching multiple files.
- `npx fallow audit` after substantial TypeScript/JavaScript changes, refactors, or agent-generated edits to catch dead code, duplication, and complexity issues.

Use Fallow as a structural codebase-quality check, not as a replacement for tests, type-checking, or linting.

If a check is skipped, state that clearly in your final summary.

## Integration Tests

- Use `npm run test:integration` only when live provider behavior needs validation and the required credentials are available.
- Prefer targeted integration runs when investigating a single provider.
- Keep integration tests isolated from unit tests when they require network access or paid provider calls.

## Style

- Use Prettier for formatting.
- Follow existing TypeScript patterns in nearby files.
- Keep comments focused on non-obvious behavior, provider quirks, or contract decisions.
- Add or update README documentation when a user-facing API, provider option, or behavior changes.
