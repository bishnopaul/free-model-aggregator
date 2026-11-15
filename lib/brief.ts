// The brief: the three JSON documents a trial hands the entry in $MAKEFUN_PROMPT_FILE.
//
// The file is prose with three embedded JSON documents, found by their `schema` member:
//   - free-model-aggregator-01/providers/v1  the provider directory (id, wire, auth, key, base path)
//   - free-model-aggregator-01/ratings/v1    the quality table (rating and capabilities per model)
//   - free-model-aggregator-01/workload/v1   the workload manifest (the requests, in sending order)
//
// Everything read here is a pure function of the file's bytes: no clock, no randomness, so a replay
// of the trial reads the same brief and makes the same choices.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export type Json = Record<string, unknown>;

export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

/** One provider as the brief's directory lists it. `base` is its base path (or an absolute URL). */
export interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

/** One row of the quality table: a model's rating and what it can do, on one provider. */
export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

/** What one request needs from its model; a model without it cannot serve the request. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

/** One workload request, in the order the client sends them. */
export interface WorkloadEntry {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

export interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  manifest: WorkloadEntry[];
  error: string | null;
}

const EMPTY: Brief = { providers: [], ratings: [], manifest: [], error: null };

function str(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** The index of the `}` that closes the object opening at `start`, skipping string contents. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` carrying a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let at = text.indexOf('{'); at !== -1; at = text.indexOf('{', at + 1)) {
    const end = objectEnd(text, at);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(at, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not a standalone JSON document: keep looking.
    }
  }
  return out;
}

function providerOf(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = str(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  // The directory spells the base as `path` (/p/<id>, joined to $MAKEFUN_ENV_URL to dial); accept the
  // absolute spellings too, first non-empty wins.
  const base = [v.path, v.base_url, v.base_path, v.base].map(str).find((s) => s !== '') ?? '';
  return { id, name: str(v.name) !== '' ? str(v.name) : id, wire, auth, base, key_env: str(v.key_env), key: str(v.key) };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = str(v.model);
  const provider = str(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: numOrNull(v.rating_milli) ?? 0,
    context_length: numOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function requirementOf(v: unknown): Requirement {
  if (!isRecord(v)) return {};
  const out: Requirement = {};
  if (v.tools === true) out.tools = true;
  if (v.vision === true) out.vision = true;
  if (v.json === true) out.json = true;
  const min = numOrNull(v.min_context);
  if (min !== null) out.min_context = min;
  return out;
}

function workloadOf(v: unknown): WorkloadEntry | null {
  if (!isRecord(v)) return null;
  const id = str(v.id);
  if (id === '') return null;
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: requirementOf(v.requires),
  };
}

/** Read and parse the brief at `path`; an unreadable or empty path gives an empty brief with an error. */
export function readBrief(path: string): Brief {
  if (path === '') return { ...EMPTY, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...EMPTY, error: `cannot read the brief at ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directory = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers));
  if (directory === undefined) return { ...EMPTY, error: `the brief holds no ${PROVIDERS_SCHEMA} document` };
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings));
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest));
  return {
    providers: (directory.providers as unknown[]).flatMap((v) => {
      const p = providerOf(v);
      return p === null ? [] : [p];
    }),
    ratings:
      ratingsDoc === undefined
        ? []
        : (ratingsDoc.ratings as unknown[]).flatMap((v) => {
            const r = ratingOf(v);
            return r === null ? [] : [r];
          }),
    manifest:
      workloadDoc === undefined
        ? []
        : (workloadDoc.manifest as unknown[]).flatMap((v) => {
            const w = workloadOf(v);
            return w === null ? [] : [w];
          }),
    error: null,
  };
}

/** The quality table's row for one provider's model, or null when the table has none. */
export function ratingRow(brief: Brief, providerId: string, modelId: string): RatingRow | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}
