// The best free AI model aggregator: one server, two client APIs, ten free-tier providers.
//
// A trial hands this server a brief (the provider directory, the sealed quality table and the
// workload manifest, at $MAKEFUN_PROMPT_FILE) and an env ($MAKEFUN_ENV_URL) that serves the
// providers and sends the workload. The server:
//   1. probes every provider's catalog route in its own wire and builds one catalog document;
//   2. listens on 0.0.0.0 at a port it picks and registers it with the env, catalog attached;
//   3. answers POST /v1/chat/completions and POST /v1/messages, in JSON and in SSE, routing each
//      request to the best free model that meets the request's requirements, in that provider's
//      own wire, and translating the answer back into the client's wire;
//   4. answers GET /v1/models in the shape of the caller's auth header, listing the free models;
//   5. polls GET /work and exits 0 when the env says the workload is done.
//
// Routing: candidates are the free models meeting the request's requirements, best rating first.
// A provider that refuses (402, 429, 5xx, or no answer) is retried once on the spot — a refusal is
// often one call on its failure schedule, and the retry keeps the request on its best model — then
// the request fails over to the next candidate; a provider that refuses on three requests is
// treated as spent and skipped
// while anything else can serve. A JSON-mode request never routes to the anthropic wire, which
// cannot carry it. Every choice derives from the brief and the providers' answers alone: no wall
// clock, no randomness, so a replayed trial makes the same calls and leaves the same export.
//
// `node main.ts --list` prints the catalog document and exits 0 without starting the server; run
// with no brief and no env it prints {"providers":[]}.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { isRecord, readBrief, type Brief, type Json, type ProviderEntry, type Requirement } from './lib/brief.ts';
import {
  buildCatalog,
  callProvider,
  capabilitiesOf,
  isFree,
  meets,
  probeProvider,
  type Catalog,
  type ListedModel,
  type Probe,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicMessage,
  anthropicModelList,
  anthropicStream,
  buildAnthropicUpstream,
  buildGeminiUpstream,
  buildOpenAIUpstream,
  openAICompletion,
  openAIError,
  openAIModelList,
  openAIStream,
  parseAnthropicClient,
  parseOpenAIClient,
  parseUpstreamAnswer,
  type Answer,
  type NormRequest,
  type UpstreamCall,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 10000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Provider refusals after which a provider is treated as spent for the rest of the trial. */
const STRIKE_LIMIT = 3;
/** The most provider calls one client request may make, retries and failovers included. */
const MAX_ATTEMPTS = 14;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ---------------------------------------------------------------------------
// Startup: the brief, the probes, the one catalog.

const brief: Brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  catalog = buildCatalog(brief, probes);
}

// ---------------------------------------------------------------------------
// Routing state: what the trial has taught the server about the providers.

/** Refusals seen per provider; at STRIKE_LIMIT the provider is treated as spent. */
const strikes = new Map<string, number>();
/** Providers whose catalog probe failed: they get no traffic. */
const unreachable = new Set<string>();

interface Candidate {
  provider: ProviderEntry;
  model: ListedModel;
  rating: number;
}

/** Every routable free model: probed providers only, with the quality table's rating and powers. */
function allCandidates(): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      if (!isFree(brief, probe.provider.id, model)) continue;
      const caps = capabilitiesOf(brief, probe.provider.id, model);
      out.push({ provider: probe.provider, model, rating: caps.rating_milli });
    }
  }
  return out;
}

function candidatesFor(norm: NormRequest, requires: Requirement): Candidate[] {
  const eligible = allCandidates().filter((c) => {
    const caps = capabilitiesOf(brief, c.provider.id, c.model);
    if (!meets(requires, caps)) return false;
    // A JSON-mode request must go to a wire that can carry it; the anthropic wire cannot.
    if (norm.jsonMode && c.provider.wire === 'anthropic') return false;
    return true;
  });
  eligible.sort((a, b) => {
    if (a.rating !== b.rating) return b.rating - a.rating;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return eligible;
}

/** Inference calls run one at a time: a provider serves one call at a time, and the client is serial. */
let callChain: Promise<unknown> = Promise.resolve();
function serialized<T>(fn: () => Promise<T>): Promise<T> {
  const run = callChain.then(fn, fn);
  callChain = run.then(
    () => undefined,
    () => undefined,
  );
  return run;
}

interface Served {
  answer: Answer;
  modelId: string;
}

/** Whether a refusal is worth one immediate retry of the same call: a budget or rate refusal
 * (402/429), a provider error (5xx) or no answer at all may be one call on a failure schedule,
 * where the next call succeeds; a 4xx refusal of the request itself would simply repeat. */
function retryable(status: number): boolean {
  return status === 0 || status === 402 || status === 429 || status >= 500;
}

/** Serve one request: the best candidate first, failing over down the list on every refusal. */
async function serve(norm: NormRequest, requires: Requirement): Promise<Served | null> {
  const candidates = candidatesFor(norm, requires);
  const tried = new Set<string>();
  let attempts = 0;
  // Two passes: live providers first; spent providers only when nothing else can serve.
  for (const includeSpent of [false, true]) {
    for (const candidate of candidates) {
      if (attempts >= MAX_ATTEMPTS) return null;
      if (tried.has(candidate.provider.id)) continue;
      const spent = (strikes.get(candidate.provider.id) ?? 0) >= STRIKE_LIMIT;
      if (spent && !includeSpent) continue;
      tried.add(candidate.provider.id);
      const call: UpstreamCall =
        candidate.provider.wire === 'openai'
          ? buildOpenAIUpstream(norm, candidate.model.id)
          : candidate.provider.wire === 'anthropic'
            ? buildAnthropicUpstream(norm, candidate.model.id)
            : buildGeminiUpstream(norm, candidate.model.id);
      // One retry on a retryable refusal: a refusal on the provider's failure schedule costs the
      // request its best model if it fails over at once, while a spent provider merely refuses
      // again. A provider earns at most one strike per request, however its calls went.
      let servedHere: Served | null = null;
      for (let attempt = 0; attempt < 2; attempt++) {
        if (attempts >= MAX_ATTEMPTS) break;
        attempts += 1;
        const res = await callProvider(ENV_URL, candidate.provider, call.path, call.body, CALL_TIMEOUT_MS);
        const answer = res.status >= 200 && res.status < 300 ? parseUpstreamAnswer(candidate.provider.wire, res.body) : null;
        if (answer !== null) {
          servedHere = { answer, modelId: candidate.model.id };
          break;
        }
        if (attempt === 0 && !retryable(res.status)) break;
      }
      if (servedHere !== null) return servedHere;
      strikes.set(candidate.provider.id, (strikes.get(candidate.provider.id) ?? 0) + 1);
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// The workload's requirements, by arrival order.

let manifestIndex = 0;

/**
 * The requirements of the request now arriving: the manifest's next row when it names this API
 * (the client sends the manifest in order, one request at a time), else read off the body itself.
 */
function requirementsFor(api: 'openai' | 'anthropic', body: Json): Requirement {
  const entry = brief.manifest[manifestIndex];
  if (entry !== undefined && entry.api === api) {
    manifestIndex += 1;
    return entry.requires;
  }
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

// ---------------------------------------------------------------------------
// HTTP plumbing.

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendStream(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(text);
}

/** The free model ids GET /v1/models lists: the catalog's free models, each id once, in order. */
function freeModelList(): { id: string; provider: string }[] {
  const seen = new Set<string>();
  const out: { id: string; provider: string }[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (!model.free || seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: provider.id });
    }
  }
  return out;
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = freeModelList();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

async function chatCompletions(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendJson(res, 400, openAIError('the request body is not one JSON object', 'invalid_request_error', null));
  const norm = parseOpenAIClient(body);
  const served = await serialized(() => serve(norm, requirementsFor('openai', body)));
  if (served === null) return sendJson(res, 502, openAIError('no free model could serve this request', 'api_error', 'no_free_model'));
  if (norm.stream) return sendStream(res, openAIStream(served.answer, served.modelId));
  return sendJson(res, 200, openAICompletion(served.answer, served.modelId));
}

async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendJson(res, 400, anthropicError('invalid_request_error', 'the request body is not one JSON object'));
  const norm = parseAnthropicClient(body);
  const served = await serialized(() => serve(norm, requirementsFor('anthropic', body)));
  if (served === null) return sendJson(res, 502, anthropicError('api_error', 'no free model could serve this request'));
  if (norm.stream) return sendStream(res, anthropicStream(served.answer, served.modelId));
  return sendJson(res, 200, anthropicMessage(served.answer, served.modelId));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') return models(req, res);
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') return chatCompletions(req, res);
  if (req.method === 'POST' && url.pathname === '/v1/messages') return messages(req, res);
  if (url.pathname === '/v1/messages') return sendJson(res, 404, anthropicError('not_found_error', `no route ${req.method ?? ''} ${url.pathname}`));
  return sendJson(res, 404, openAIError(`no route ${req.method ?? ''} ${url.pathname}`, 'invalid_request_error', 'not_found'));
}

// ---------------------------------------------------------------------------
// Registration and the work poll.

async function register(port: number): Promise<void> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < 40; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = err;
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
  }
  throw lastError instanceof Error ? lastError : new Error('register failed');
}

/** Poll GET /work; exit 0 the moment the env says the workload is done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The one catalog object, printed byte for byte as the register body carries it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  for (const probe of probes) if (!probe.ok) unreachable.add(probe.provider.id);
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        if (req.url?.startsWith('/v1/messages') === true) sendJson(res, 500, anthropicError('api_error', 'the server failed to answer'));
        else sendJson(res, 500, openAIError('the server failed to answer', 'api_error', 'internal_error'));
      }
    });
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  log(`listening on 0.0.0.0:${address.port}; ${catalog.providers.length} providers in the catalog`);
  await register(address.port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
