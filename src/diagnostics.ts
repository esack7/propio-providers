/**
 * Diagnostic events emitted by providers.
 *
 * Providers report operational events (currently only retries) through an
 * optional listener callback. Consumers can forward these into their own
 * diagnostics pipelines; the event shapes are stable, provider-owned types.
 */

/**
 * Emitted when a provider retries a failed request.
 */
export interface ProviderRetryDiagnosticEvent {
  type: "provider_retry";
  provider: string;
  model: string;
  iteration: number;
  reason: string;
  attemptNumber: number;
  delayMs: number;
}

export type ProviderDiagnosticEvent = ProviderRetryDiagnosticEvent;

export type ProviderDiagnosticListener = (
  event: ProviderDiagnosticEvent,
) => void;

/**
 * Retry behavior configuration accepted by providers and the factory.
 */
export interface ProviderRetryConfig {
  maxRetries: number;
  consecutive529Limit: number;
}
