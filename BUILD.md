# The best free AI model aggregator — entry notes

One server (`main.ts`, Node 22, standard library only) that fronts ten free-tier providers on
three wires (OpenAI, Anthropic, Gemini) behind the two client APIs (OpenAI Chat Completions and
Anthropic Messages), each in JSON and in SSE streaming.

## Layout

- `main.ts` — the server: startup, registration, routing policy, the `/work` poll.
- `lib/brief.ts` — the brief at `$MAKEFUN_PROMPT_FILE`, parsed into the provider directory, the
  quality table and the workload manifest (documents found by their `schema` member).
- `lib/providers.ts` — catalog probes (one GET per provider, in its own wire and auth), the
  catalog document, and the inference-call plumbing.
- `lib/wire.ts` — all wire translation: client request → normalized form → provider request;
  provider answer → normalized answer → client JSON or SSE.

No kit SDK is vendored: this competition waives model calls (`model_calls: optional`), and the
entry makes none — it is a scripted server, which the rules expressly allow.

## How a trial runs

1. Read the brief; probe all ten providers' catalog routes concurrently (catalog calls spend
   none of a provider's inference budget, so probing is free and parallel).
2. Build one catalog document — providers in brief order, models sorted by id, `base_url` the
   provider's own path from the brief, the free flag and context length settled by the quality
   table where it has a row, else the listing's word. The same object is printed by
   `node main.ts --list` and posted as the register body's `catalog`, byte for byte. (Run with
   no brief and no env, `--list` prints `{"providers":[]}`.)
3. Listen on an ephemeral port, `POST /register {protocol_version: 1, port, catalog}`, then poll
   `GET /work` every 100 ms and exit 0 when it says done.
4. `GET /v1/models` answers in the caller's wire (an `x-api-key` header gets the Anthropic list,
   anything else the OpenAI list) with exactly the catalog's free model ids, each once.

## Model choice and failover

Requirements come from the workload manifest, by arrival order (the client sends the manifest in
order, one request at a time; the manifest's row also settles what a body cannot say, such as
`min_context`). Candidates are the free models on probed providers that meet the requirements,
ordered by the quality table's `rating_milli`, ties broken by provider id then model id. Two
extra routing rules:

- a JSON-mode request (an OpenAI `response_format`) never routes to the anthropic wire, which
  cannot carry JSON mode and so could not echo it;
- providers are called one at a time (a serialized call chain), so no provider ever sees a
  second concurrent call.

On a refusal (402, 429, 5xx, or no answer) the same call is retried once — a refusal is often a
single call on the provider's failure schedule, and the retry keeps the request on its best
model — then the request fails over to the next candidate. A provider that refuses on three
separate requests is treated as spent (its budget is gone, or its outage has begun) and skipped
while anything else can serve. A paid model is never a candidate.

## Wire translation

Requests are normalized once (system prompt as the one joined string every wire computes, the
messages part by part — text, base64 and URL images, tool calls, tool results — the tool names in
order, `max_tokens`, stop sequences, JSON mode, tool choice), then written in the provider's
wire. The echo-critical fields survive every pairing: system text hashes equal because both sides
join text blocks the same way; the last user text is preserved message-for-message (tool results
fold into tool messages / `tool_result` blocks / `functionResponse` parts without disturbing it);
images keep their bytes (data URL ↔ base64 source ↔ `inlineData`) or their URL (`image_url` ↔
URL source ↔ `fileData`), so their hashes match; `max_tokens` and stops map field-for-field;
JSON mode maps to `response_format` (OpenAI) or `responseMimeType: application/json` (Gemini).

Providers are always called non-streaming: an answer's text, usage and stop reason are identical
either way, and the client's stream is written from the whole answer in the client's own event
order (OpenAI: role chunk, 13-character text chunks, finish chunk, a usage chunk, `[DONE]`;
Anthropic: `message_start`, `content_block_start`, `content_block_delta`s, `content_block_stop`,
`message_delta`, `message_stop`). Stop reasons are mapped by family (end_turn ↔ stop, max_tokens
↔ length, tool_use ↔ tool_calls, refusal ↔ content_filter), and usage is reported with the
provider's own counts in the client's field names. All output is deterministic — fixed ids and
timestamps, no wall clock, no randomness — so a replayed trial leaves a byte-identical export.

## Verification (all with the kit's own env and scorer, plain node processes)

Practice trials run manually per the kit's protocol (env seeded from `practice/trial-N/input/env`,
this server as the agent, state export scored by `program/scorer/score.mjs`):

| trial | score | quality | served | wire | failover | catalog | valid | upstream calls |
|-------|-------|---------|--------|------|----------|---------|-------|----------------|
| 0     | 99    | 99      | 100    | 100  | 100      | 100     | 20/20 | 32             |
| 1     | 99    | 99      | 100    | 100  | 100      | 100     | 20/20 | 35             |
| 2     | 99    | 99      | 100    | 100  | 100      | 100     | 20/20 | 39             |

Quality 99 is the practical ceiling on these draws: the top model's provider budget (e.g. 12
calls on trial-0, two of them eaten by scheduled failures) cannot cover every request that wants
it, and the remainder land on the 1470-rated models, 5 milli-points below the 1475 best — the
scorer's weighted floor turns any nonzero loss into 99. All 233 response checks passed on every
trial.

Also verified:

- **Replay determinism**: trial-0 run twice from scratch; the two `state.json` exports are
  byte-identical (`cmp` clean).
- **`--list`**: with no env vars it prints `{"providers":[]}` and exits 0; inside a trial setup
  its output deep-equals the catalog in the state export (the registered document).
- **Echo invariants**: a unit battery of 9 synthetic client bodies (multi-turn tool calls and
  results, block-array system prompts, string and array stops, `max_completion_tokens`,
  `json_schema`, base64 and URL images) × all 3 provider wires, checked with the env's own
  parsing rules — every echo field matches in all 27 combinations (the one exception by design:
  an anthropic-wire provider cannot echo JSON mode, and routing never sends it one).
- **Probe failure fallback**: with the env unreachable, `--list` still names all ten providers
  with the quality table's free models, so the catalog stays accurate where a listing cannot be
  read.
