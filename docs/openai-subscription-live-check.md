# Live subscription tool round trip

Passed on October 12, 2026 at 03:32 UTC (October 11 in America/Los_Angeles), using provider commit `711b4eeaaa73fc625a435bd1976a26645f89491c` and agent commit `6798a2f174203d948cc51f31eeea04c123e5eb36`. Runtime: Node 22.23.3 on macOS with the production application-owned Keychain credential backend. Model: `gpt-6.1-sol`. Account label: `personal`.

This check made actual OpenAI requests using the production provider factory and `OpenAiAuth` renewable source. It used the user's existing signed-in production registration; its expired credentials were renewed through the application before model discovery. The catalog returned seven visible models including the selected model. No POC tokens were copied or imported, and no defaults or billing selections were changed. This was not a new interactive browser authorization; logout/relogin remains a separate acceptance gate.

| Stage | Actual result |
| --- | --- |
| Existing registration | Connected, plan permission enabled, OS credential storage |
| Renewal/discovery | Expired credentials renewed; selected model visible among seven models |
| First inference | Real `POST https://api.openai.com/v1/responses`, HTTP 200, stateless SSE, developer instruction and `functions` namespace containing `echo_marker` |
| Tool admission | Exactly one completed local `echo_marker` call; arguments matched a synthetic challenge; `response.completed` observed before execution |
| Local execution | Echoed the synthetic marker entirely in memory; no file, network or shell access |
| Result replay | Second real request contained `function_call` and `function_call_output`; HTTP 200 and `response.completed` |
| Continuation assertion | Final text contained the local tool result; no additional tool calls |

The [sanitized observation](openai-subscription-live-check.json) records actual request shape, SSE event counts and pass/fail assertions. A temporary harness observed real requests and streams while exercising the production provider; it did not mock fetch or replace responses with simulated data. Retry count was zero and there were exactly two inference requests. The renewable source was acquired for each inference round. The harness executed a synthetic local echo function rather than the ordinary Agent built-in/MCP tool runner.

Only safe test metadata is committed. Authorization headers, tokens, account/client/subject IDs, prompt/answer content, tool arguments/results and encrypted reasoning are excluded. No request ID was recorded from the response headers. The recorded response items did not contain a `namespace` field, so `observedFunctionNamespaces` is empty; the outgoing requests did contain the `functions` namespace wrapper and were accepted. The JSON is a sanitized live observation, not a raw wire response or a fixture replayed by unit tests.

This establishes one production registration/model's real endpoint admission, namespace-wrapped local tool call and result continuation after credential renewal. No plan limit/admission failure was intentionally provoked; error-code handling remains covered by simulated regressions. It does not establish other models/accounts, fresh browser sign-in, ordinary Agent MCP/session/summary acceptance, long reasoning gaps, logout/relogin or sandbox expiry/platform support.

To repeat, sign in through the consuming application's normal account flow, bind its renewable source to the immutable registration, discover an eligible model, then use `createProvider` with explicit ChatGPT auth and the caller-owned source. Give it one harmless echo function under normal `ChatRequest.tools`, collect a confirmed tool event, execute that function locally and replay the assistant tool history plus its result. Check HTTP 200 and `response.completed` in both rounds and verify the final answer incorporates the result. Keep retries bounded, use a cancellation deadline, and record only the same safe metadata. Never substitute an API key or import another application's tokens.
