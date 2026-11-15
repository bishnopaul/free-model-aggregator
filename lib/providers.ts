// The providers: catalog probes, the catalog document, and inference calls.
//
// Each provider speaks one wire (openai, anthropic or gemini) and takes its key in its own header.
// Its catalog route lists every model it serves, free and paid, with capability fields; a probe is a
// plain GET on that route. Catalog calls are the trial's free observation: they spend none of a
// provider's inference budget, so all ten are probed at once at startup.
//
// The catalog built here is one object used twice, byte for byte: printed by `node main.ts --list`
// and posted as the register body's `catalog`. Its base_url is the provider's own path from the
// brief, never joined to the env's URL: the env's origin differs between a trial's live pass and its
// replay, while the path names the provider in both.

import {
  isRecord,
  ratingRow,
  type AuthScheme,
  type Brief,
  type Json,
  type ProviderEntry,
  type Wire,
} from './brief.ts';

/** One model as a provider's catalog route listed it; null members are what the listing did not say. */
export interface ListedModel {
  id: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** The outcome of one provider's catalog probe. */
export interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  status: number;
  models: ListedModel[];
}

/** One model in the catalog document. */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** One provider in the catalog document. */
export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

/** The catalog document: what --list prints and what /register carries. */
export interface Catalog {
  providers: CatalogProvider[];
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

function numOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** The env's origin joined to a provider's base path; an absolute base is used as it stands. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The header a provider's auth scheme carries its key in; no header for `none` or an empty key. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** The headers one provider call carries: its auth, plus anthropic-version on the anthropic wire. */
export function providerHeaders(provider: ProviderEntry): Record<string, string> {
  const headers: Record<string, string> = { ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  return headers;
}

/** The catalog route's URL for one provider, in its own wire. */
export function catalogUrl(envUrl: string, provider: ProviderEntry): string {
  const base = joinBase(envUrl, provider.base);
  return provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
}

/** One listed model read out of a catalog answer, in whichever wire's spelling it came in. */
function listedModel(wire: Wire, v: unknown): ListedModel | null {
  if (!isRecord(v)) return null;
  const rawId = wire === 'gemini' ? (typeof v.name === 'string' ? v.name.replace(/^models\//, '') : '') : typeof v.id === 'string' ? v.id : '';
  if (rawId === '') return null;
  return {
    id: rawId,
    free: boolOrNull(v.free),
    context_length: numOrNull(v.context_length) ?? numOrNull(v.context_window) ?? numOrNull(v.inputTokenLimit),
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}

/** Probe one provider's catalog route; every failure is a probe that lists nothing. */
export async function probeProvider(envUrl: string, provider: ProviderEntry, timeoutMs = 5000): Promise<Probe> {
  const url = catalogUrl(envUrl, provider);
  try {
    const res = await fetch(url, {
      headers: { accept: 'application/json', ...providerHeaders(provider) },
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!res.ok) return { provider, ok: false, status: res.status, models: [] };
    const body: unknown = await res.json();
    if (!isRecord(body)) return { provider, ok: false, status: res.status, models: [] };
    const list = provider.wire === 'gemini' ? body.models : body.data;
    if (!Array.isArray(list)) return { provider, ok: false, status: res.status, models: [] };
    const models = list.flatMap((v) => {
      const m = listedModel(provider.wire, v);
      return m === null ? [] : [m];
    });
    return { provider, ok: true, status: res.status, models };
  } catch {
    return { provider, ok: false, status: 0, models: [] };
  }
}

const byModelId = (a: { id: string }, b: { id: string }): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/**
 * The catalog from the probes and the quality table. A provider that answered lists what it listed;
 * the quality table's row, where one exists, settles the free flag and the context length. A provider
 * whose probe failed is still named, with the models the quality table knows for it: the table holds
 * the free models, so the entry stays accurate where the listing could not be read.
 */
export function buildCatalog(brief: Brief, probes: readonly Probe[]): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    const { provider } = probe;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    const push = (id: string, free: boolean, context: number | null): void => {
      if (seen.has(id)) return;
      seen.add(id);
      models.push({ id, free, context_length: context });
    };
    if (probe.ok) {
      for (const model of [...probe.models].sort(byModelId)) {
        const row = ratingRow(brief, provider.id, model.id);
        push(model.id, row?.free ?? model.free ?? false, row?.context_length ?? model.context_length ?? null);
      }
    } else {
      for (const row of brief.ratings.filter((r) => r.provider === provider.id).sort((a, b) => byModelId({ id: a.model }, { id: b.model }))) {
        push(row.model, row.free, row.context_length);
      }
    }
    providers.push({ id: provider.id, base_url: provider.base, key_env: provider.key_env, wire: provider.wire, models });
  }
  return { providers };
}

/** A model's capabilities for routing: the quality table's row first, the listing's fields second. */
export interface Capabilities {
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

export function capabilitiesOf(brief: Brief, providerId: string, model: ListedModel): Capabilities {
  const row = ratingRow(brief, providerId, model.id);
  return {
    rating_milli: row?.rating_milli ?? 0,
    context_length: row?.context_length ?? model.context_length ?? null,
    tools: row?.tools ?? model.tools ?? false,
    vision: row?.vision ?? model.vision ?? false,
    json: row?.json ?? model.json ?? false,
  };
}

/** Whether a model with these capabilities may serve a request with these requirements. */
export function meets(req: { tools?: boolean; vision?: boolean; json?: boolean; min_context?: number }, caps: Capabilities): boolean {
  if (req.tools === true && !caps.tools) return false;
  if (req.vision === true && !caps.vision) return false;
  if (req.json === true && !caps.json) return false;
  if (req.min_context !== undefined && (caps.context_length === null || caps.context_length < req.min_context)) return false;
  return true;
}

/** Whether a listed model is free: the quality table's row where it has one, else the listing's word. */
export function isFree(brief: Brief, providerId: string, model: ListedModel): boolean {
  const row = ratingRow(brief, providerId, model.id);
  return row?.free ?? model.free ?? false;
}

/** One inference call's answer: an HTTP status and its parsed JSON body, or null on a network failure. */
export interface ProviderAnswer {
  status: number;
  body: Json | null;
}

/**
 * POST one inference call to a provider. `path` is the wire's route under the provider's base.
 * A transport failure answers status 0 with a null body: the router treats it as a refusal.
 */
export async function callProvider(
  envUrl: string,
  provider: ProviderEntry,
  path: string,
  body: Json,
  timeoutMs = 15000,
): Promise<ProviderAnswer> {
  const url = `${joinBase(envUrl, provider.base)}${path}`;
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json', ...providerHeaders(provider) },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutMs),
    });
    const text = await res.text();
    let parsed: Json | null = null;
    try {
      const value: unknown = JSON.parse(text);
      parsed = isRecord(value) ? value : null;
    } catch {
      parsed = null;
    }
    return { status: res.status, body: parsed };
  } catch {
    return { status: 0, body: null };
  }
}
