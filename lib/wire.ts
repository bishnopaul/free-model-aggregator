// The wires: translation between the client's API and each provider's API.
//
// Every request is parsed once into a normalized form, then written out in the routed provider's
// wire; every provider answer is parsed back into one normalized answer, then written in the
// client's wire, as JSON or as an event stream. The fields a provider echoes back about a request
// (its system prompt, its last user text, its images, its tool names, its max_tokens, its stop
// sequences, its JSON mode) are preserved exactly across every translation: the normalized form
// carries the system prompt as the one joined string both sides compute, the messages part by part,
// and the tools by name in order, so any wire's reading of the request agrees with any other's.
//
// Providers are always called without streaming: an answer's text, usage and stop reason are the
// same either way, and the client's stream is then written from the whole answer in the client's
// own event order. All output is a pure function of the request and the provider's answer: fixed
// ids and timestamps, no clock, no randomness.

import { isRecord, type Json } from './brief.ts';

/** A fixed epoch second for `created` fields: nothing the entry writes reads the clock. */
export const FIXED_CREATED = 1767225600;
/** A fixed instant for Anthropic-shaped `created_at` fields. */
export const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
/** Characters of answer text per streamed delta, the size the simulated providers use. */
const DELTA_CHARS = 13;

// ---------------------------------------------------------------------------
// The normalized request

export type Part =
  | { kind: 'text'; text: string }
  | { kind: 'image'; mediaType: string; data: string }
  | { kind: 'imageUrl'; url: string }
  | { kind: 'toolUse'; id: string; name: string; input: Json }
  | { kind: 'toolResult'; id: string; text: string; parts: Part[] };

export interface NormMessage {
  role: 'user' | 'assistant' | 'tool';
  parts: Part[];
}

export interface NormTool {
  name: string;
  description: string;
  schema: Json;
}

export type ToolChoice = { kind: 'auto' | 'none' | 'required' } | { kind: 'tool'; name: string };

export interface NormRequest {
  systemText: string;
  messages: NormMessage[];
  tools: NormTool[];
  /** Tool call ids to their names, for wires that name a call's result by the tool. */
  toolNames: Map<string, string>;
  maxTokens: number | null;
  stops: string[];
  /** The client asked for JSON mode (an OpenAI response_format); only some wires can carry it. */
  jsonMode: boolean;
  /** The client's response_format object, passed through whole on the OpenAI wire. */
  responseFormat: Json | null;
  temperature: number | null;
  topP: number | null;
  toolChoice: ToolChoice | null;
  stream: boolean;
}

function numOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** The text of a string body or of a block array's text blocks, joined with newlines. */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

/** Split a data URL into its media type and base64 payload; null when it is not one. */
function dataUrl(url: string): { mediaType: string; data: string } | null {
  const m = /^data:([^;,]+)?(;base64)?,(.*)$/s.exec(url);
  if (m === null || m[2] === undefined) return null;
  return { mediaType: m[1] ?? 'image/png', data: m[3] };
}

function emptyNorm(): NormRequest {
  return {
    systemText: '',
    messages: [],
    tools: [],
    toolNames: new Map(),
    maxTokens: null,
    stops: [],
    jsonMode: false,
    responseFormat: null,
    temperature: null,
    topP: null,
    toolChoice: null,
    stream: false,
  };
}

// ---------------------------------------------------------------------------
// Reading the client's request

/** An OpenAI Chat Completions body in normalized form. */
export function parseOpenAIClient(body: Json): NormRequest {
  const out = emptyNorm();
  const systemParts: string[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'user') {
      out.messages.push({ role: 'user', parts: openAIUserParts(raw.content) });
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Part[] = [];
      const text = blocksText(raw.content);
      if (text !== '') parts.push({ kind: 'text', text });
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function) || typeof call.function.name !== 'string') continue;
          const id = typeof call.id === 'string' ? call.id : '';
          out.toolNames.set(id, call.function.name);
          parts.push({ kind: 'toolUse', id, name: call.function.name, input: argumentsObject(call.function.arguments) });
        }
      }
      out.messages.push({ role: 'assistant', parts });
      continue;
    }
    if (raw.role === 'tool' || raw.role === 'function') {
      const id = typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '';
      out.messages.push({ role: 'tool', parts: [{ kind: 'toolResult', id, text: blocksText(raw.content), parts: [] }] });
    }
  }
  out.systemText = systemParts.join('\n');
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || !isRecord(tool.function) || typeof tool.function.name !== 'string') continue;
      out.tools.push({
        name: tool.function.name,
        description: typeof tool.function.description === 'string' ? tool.function.description : '',
        schema: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  out.maxTokens = numOrNull(body.max_tokens) ?? numOrNull(body.max_completion_tokens);
  out.stops = stringList(body.stop);
  const format = isRecord(body.response_format) ? body.response_format : null;
  if (format !== null && (format.type === 'json_object' || format.type === 'json_schema')) {
    out.jsonMode = true;
    out.responseFormat = format;
  }
  out.temperature = numOrNull(body.temperature);
  out.topP = numOrNull(body.top_p);
  out.toolChoice = openAIToolChoice(body.tool_choice);
  out.stream = body.stream === true;
  return out;
}

function openAIUserParts(content: unknown): Part[] {
  if (typeof content === 'string') return content === '' ? [] : [{ kind: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Part[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') {
      parts.push({ kind: 'text', text: block.text });
    } else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
      const parsed = dataUrl(block.image_url.url);
      if (parsed !== null) parts.push({ kind: 'image', mediaType: parsed.mediaType, data: parsed.data });
      else parts.push({ kind: 'imageUrl', url: block.image_url.url });
    }
  }
  return parts;
}

function argumentsObject(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input object.
    }
  }
  return {};
}

function openAIToolChoice(v: unknown): ToolChoice | null {
  if (v === 'auto') return { kind: 'auto' };
  if (v === 'none') return { kind: 'none' };
  if (v === 'required') return { kind: 'required' };
  if (isRecord(v) && isRecord(v.function) && typeof v.function.name === 'string') return { kind: 'tool', name: v.function.name };
  if (isRecord(v) && typeof v.name === 'string') return { kind: 'tool', name: v.name };
  return null;
}

/** An Anthropic Messages body in normalized form. */
export function parseAnthropicClient(body: Json): NormRequest {
  const out = emptyNorm();
  out.systemText = blocksText(body.system);
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'assistant' ? 'assistant' : 'user';
    out.messages.push({ role, parts: anthropicParts(raw.content, out) });
  }
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || typeof tool.name !== 'string') continue;
      out.tools.push({
        name: tool.name,
        description: typeof tool.description === 'string' ? tool.description : '',
        schema: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  out.maxTokens = numOrNull(body.max_tokens);
  out.stops = stringList(body.stop_sequences);
  out.temperature = numOrNull(body.temperature);
  out.topP = numOrNull(body.top_p);
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.toolChoice = { kind: 'auto' };
    else if (choice.type === 'none') out.toolChoice = { kind: 'none' };
    else if (choice.type === 'any') out.toolChoice = { kind: 'required' };
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.toolChoice = { kind: 'tool', name: choice.name };
  }
  out.stream = body.stream === true;
  return out;
}

function anthropicParts(content: unknown, norm: NormRequest): Part[] {
  if (typeof content === 'string') return content === '' ? [] : [{ kind: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Part[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') {
      parts.push({ kind: 'text', text: block.text });
    } else if (block.type === 'image' && isRecord(block.source)) {
      const source = block.source;
      if (source.type === 'base64' && typeof source.data === 'string') {
        parts.push({ kind: 'image', mediaType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data });
      } else if (source.type === 'url' && typeof source.url === 'string') {
        parts.push({ kind: 'imageUrl', url: source.url });
      }
    } else if (block.type === 'tool_use' && typeof block.name === 'string') {
      const id = typeof block.id === 'string' ? block.id : '';
      norm.toolNames.set(id, block.name);
      parts.push({ kind: 'toolUse', id, name: block.name, input: isRecord(block.input) ? block.input : {} });
    } else if (block.type === 'tool_result') {
      const id = typeof block.tool_use_id === 'string' ? block.tool_use_id : '';
      const inner = anthropicParts(block.content, norm).filter((p) => p.kind === 'text' || p.kind === 'image' || p.kind === 'imageUrl');
      const text = inner
        .filter((p): p is Extract<Part, { kind: 'text' }> => p.kind === 'text')
        .map((p) => p.text)
        .join('\n');
      parts.push({ kind: 'toolResult', id, text, parts: inner });
    }
  }
  return parts;
}

// ---------------------------------------------------------------------------
// Writing the provider's request

export interface UpstreamCall {
  path: string;
  body: Json;
}

/** The normalized request as an OpenAI Chat Completions call naming `model` (never streamed). */
export function buildOpenAIUpstream(norm: NormRequest, model: string): UpstreamCall {
  const messages: Json[] = [];
  if (norm.systemText !== '') messages.push({ role: 'system', content: norm.systemText });
  for (const message of norm.messages) {
    if (message.role === 'user') {
      for (const part of message.parts) {
        if (part.kind === 'toolResult') messages.push({ role: 'tool', tool_call_id: part.id, content: part.text });
      }
      const parts = message.parts.filter((p) => p.kind !== 'toolResult');
      messages.push({ role: 'user', content: openAIContent(parts) });
    } else if (message.role === 'assistant') {
      const text = message.parts
        .filter((p): p is Extract<Part, { kind: 'text' }> => p.kind === 'text')
        .map((p) => p.text)
        .join('\n');
      const calls = message.parts.filter((p): p is Extract<Part, { kind: 'toolUse' }> => p.kind === 'toolUse');
      messages.push({
        role: 'assistant',
        content: text === '' ? null : text,
        ...(calls.length > 0
          ? { tool_calls: calls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: JSON.stringify(c.input) } })) }
          : {}),
      });
    } else {
      for (const part of message.parts) {
        if (part.kind === 'toolResult') messages.push({ role: 'tool', tool_call_id: part.id, content: part.text });
      }
    }
  }
  const body: Json = { model, messages, stream: false };
  if (norm.tools.length > 0) {
    body.tools = norm.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.schema } }));
  }
  if (norm.toolChoice !== null) body.tool_choice = openAIToolChoiceOut(norm.toolChoice);
  if (norm.maxTokens !== null) body.max_tokens = norm.maxTokens;
  if (norm.stops.length > 0) body.stop = norm.stops;
  if (norm.jsonMode) body.response_format = norm.responseFormat ?? { type: 'json_object' };
  if (norm.temperature !== null) body.temperature = norm.temperature;
  if (norm.topP !== null) body.top_p = norm.topP;
  return { path: '/v1/chat/completions', body };
}

function openAIContent(parts: Part[]): unknown {
  if (parts.length === 0) return '';
  if (parts.length === 1 && parts[0].kind === 'text') return parts[0].text;
  return parts.flatMap((part): Json[] => {
    if (part.kind === 'text') return [{ type: 'text', text: part.text }];
    if (part.kind === 'image') return [{ type: 'image_url', image_url: { url: `data:${part.mediaType};base64,${part.data}` } }];
    if (part.kind === 'imageUrl') return [{ type: 'image_url', image_url: { url: part.url } }];
    return [];
  });
}

function openAIToolChoiceOut(choice: ToolChoice): unknown {
  if (choice.kind === 'tool') return { type: 'function', function: { name: choice.name } };
  return choice.kind;
}

/** The normalized request as an Anthropic Messages call naming `model` (never streamed). */
export function buildAnthropicUpstream(norm: NormRequest, model: string): UpstreamCall {
  const messages: Json[] = [];
  const pushUser = (blocks: Json[]): void => {
    messages.push({ role: 'user', content: blocks.length > 0 ? blocks : [{ type: 'text', text: '' }] });
  };
  for (const message of norm.messages) {
    if (message.role === 'user') {
      pushUser(message.parts.flatMap((p) => anthropicBlock(p)));
    } else if (message.role === 'assistant') {
      const blocks = message.parts.flatMap((p) => anthropicBlock(p));
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : [{ type: 'text', text: '' }] });
    } else {
      // A tool message becomes tool_result blocks, folded into the last user message when there is one
      // so the message order (and so the last user text) survives the move.
      const blocks = message.parts.flatMap((p) => anthropicBlock(p));
      const last = messages[messages.length - 1];
      if (last !== undefined && last.role === 'user' && Array.isArray(last.content)) {
        (last.content as Json[]).push(...blocks);
      } else {
        pushUser(blocks);
      }
    }
  }
  const body: Json = { model, messages, stream: false };
  if (norm.systemText !== '') body.system = norm.systemText;
  if (norm.maxTokens !== null) body.max_tokens = norm.maxTokens;
  if (norm.stops.length > 0) body.stop_sequences = norm.stops;
  if (norm.tools.length > 0) {
    body.tools = norm.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.schema }));
  }
  if (norm.toolChoice !== null) {
    body.tool_choice =
      norm.toolChoice.kind === 'tool'
        ? { type: 'tool', name: norm.toolChoice.name }
        : norm.toolChoice.kind === 'required'
          ? { type: 'any' }
          : { type: norm.toolChoice.kind };
  }
  if (norm.temperature !== null) body.temperature = norm.temperature;
  if (norm.topP !== null) body.top_p = norm.topP;
  return { path: '/v1/messages', body };
}

function anthropicBlock(part: Part): Json[] {
  if (part.kind === 'text') return [{ type: 'text', text: part.text }];
  if (part.kind === 'image') return [{ type: 'image', source: { type: 'base64', media_type: part.mediaType, data: part.data } }];
  if (part.kind === 'imageUrl') return [{ type: 'image', source: { type: 'url', url: part.url } }];
  if (part.kind === 'toolUse') return [{ type: 'tool_use', id: part.id, name: part.name, input: part.input }];
  // toolResult
  const content: unknown =
    part.parts.length === 0
      ? part.text
      : part.parts.flatMap((p) => anthropicBlock(p));
  return [{ type: 'tool_result', tool_use_id: part.id, content }];
}

/** The normalized request as a Gemini generateContent call naming `model` (never streamed). */
export function buildGeminiUpstream(norm: NormRequest, model: string): UpstreamCall {
  const contents: Json[] = [];
  for (const message of norm.messages) {
    if (message.role === 'assistant') {
      contents.push({ role: 'model', parts: message.parts.flatMap((p) => geminiPart(p, norm)) });
    } else if (message.role === 'user') {
      contents.push({ role: 'user', parts: message.parts.flatMap((p) => geminiPart(p, norm)) });
    } else {
      const parts = message.parts.flatMap((p) => geminiPart(p, norm));
      const last = contents[contents.length - 1];
      if (last !== undefined && last.role === 'user' && Array.isArray(last.parts)) {
        (last.parts as Json[]).push(...parts);
      } else {
        contents.push({ role: 'user', parts });
      }
    }
  }
  for (const content of contents) {
    if (Array.isArray(content.parts) && content.parts.length === 0) content.parts = [{ text: '' }];
  }
  const body: Json = { contents };
  if (norm.systemText !== '') body.systemInstruction = { parts: [{ text: norm.systemText }] };
  if (norm.tools.length > 0) {
    body.tools = [
      { functionDeclarations: norm.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.schema })) },
    ];
  }
  if (norm.toolChoice !== null) {
    body.toolConfig = {
      functionCallingConfig:
        norm.toolChoice.kind === 'tool'
          ? { mode: 'ANY', allowedFunctionNames: [norm.toolChoice.name] }
          : { mode: norm.toolChoice.kind === 'auto' ? 'AUTO' : norm.toolChoice.kind === 'none' ? 'NONE' : 'ANY' },
    };
  }
  const config: Json = {};
  if (norm.maxTokens !== null) config.maxOutputTokens = norm.maxTokens;
  if (norm.stops.length > 0) config.stopSequences = norm.stops;
  if (norm.temperature !== null) config.temperature = norm.temperature;
  if (norm.topP !== null) config.topP = norm.topP;
  if (norm.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return { path: `/v1beta/models/${encodeURIComponent(model)}:generateContent`, body };
}

function geminiPart(part: Part, norm: NormRequest): Json[] {
  if (part.kind === 'text') return [{ text: part.text }];
  if (part.kind === 'image') return [{ inlineData: { mimeType: part.mediaType, data: part.data } }];
  if (part.kind === 'imageUrl') return [{ fileData: { fileUri: part.url } }];
  if (part.kind === 'toolUse') return [{ functionCall: { name: part.name, args: part.input } }];
  return [{ functionResponse: { name: norm.toolNames.get(part.id) ?? part.id, response: { result: part.text } } }];
}

// ---------------------------------------------------------------------------
// Reading the provider's answer

/** One provider answer, normalized: the whole text, the usage counts, the wire's own stop reason. */
export interface Answer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stop: string | null;
}

function numOrZero(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

/** The text of a response content value: a string, or its text blocks joined with no separator. */
function answerText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('');
}

/** Parse one provider's non-streamed answer; null when the body is not a usable answer of its wire. */
export function parseUpstreamAnswer(wire: 'openai' | 'anthropic' | 'gemini', body: Json | null): Answer | null {
  if (body === null) return null;
  if (wire === 'openai') {
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null || !isRecord(choice.message)) return null;
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text: answerText(choice.message.content),
      inputTokens: numOrZero(usage.prompt_tokens),
      outputTokens: numOrZero(usage.completion_tokens),
      stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : null,
    };
  }
  if (wire === 'anthropic') {
    if (body.type !== 'message' || !Array.isArray(body.content)) return null;
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text: answerText(body.content),
      inputTokens: numOrZero(usage.input_tokens),
      outputTokens: numOrZero(usage.output_tokens),
      stop: typeof body.stop_reason === 'string' ? body.stop_reason : null,
    };
  }
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null || !isRecord(candidate.content) || !Array.isArray(candidate.content.parts)) return null;
  const texts: string[] = [];
  for (const part of candidate.content.parts) {
    if (isRecord(part) && typeof part.text === 'string') texts.push(part.text);
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: texts.join(''),
    inputTokens: numOrZero(usage.promptTokenCount),
    outputTokens: numOrZero(usage.candidatesTokenCount),
    stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : null,
  };
}

// ---------------------------------------------------------------------------
// Writing the client's answer

const STOP_FAMILIES = new Map<string, number>([
  ['stop', 0],
  ['end_turn', 0],
  ['STOP', 0],
  ['length', 1],
  ['max_tokens', 1],
  ['MAX_TOKENS', 1],
  ['tool_calls', 2],
  ['function_call', 2],
  ['tool_use', 2],
  ['content_filter', 3],
  ['refusal', 3],
  ['SAFETY', 3],
]);

const OPENAI_STOPS = ['stop', 'length', 'tool_calls', 'content_filter'];
const ANTHROPIC_STOPS = ['end_turn', 'max_tokens', 'tool_use', 'refusal'];

function stopFamily(stop: string | null): number {
  if (stop === null) return 0;
  return STOP_FAMILIES.get(stop) ?? 0;
}

/** The answer as an OpenAI Chat Completions JSON body. */
export function openAICompletion(answer: Answer, model: string): Json {
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      { index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: OPENAI_STOPS[stopFamily(answer.stop)], logprobs: null },
    ],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

/** The answer as an Anthropic Messages JSON body. */
export function anthropicMessage(answer: Answer, model: string): Json {
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: ANTHROPIC_STOPS[stopFamily(answer.stop)],
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += DELTA_CHARS) pieces.push(text.slice(at, at + DELTA_CHARS));
  return pieces.length === 0 ? [''] : pieces;
}

function sseData(data: unknown): string {
  return `data: ${JSON.stringify(data)}\n\n`;
}

function sseEvent(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

/** The answer as an OpenAI SSE stream: role chunk, text chunks, finish chunk, usage chunk, [DONE]. */
export function openAIStream(answer: Answer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseData(chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) frames.push(sseData(chunk({ content: piece }, null)));
  frames.push(sseData(chunk({}, OPENAI_STOPS[stopFamily(answer.stop)])));
  frames.push(
    sseData({
      id: 'chatcmpl-agg',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.inputTokens,
        completion_tokens: answer.outputTokens,
        total_tokens: answer.inputTokens + answer.outputTokens,
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** The answer as an Anthropic SSE stream, in the Messages event order. */
export function anthropicStream(answer: Answer, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: ANTHROPIC_STOPS[stopFamily(answer.stop)], stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

// ---------------------------------------------------------------------------
// Model lists and errors

export interface ListedEntry {
  id: string;
  provider: string;
}

/** GET /v1/models in the OpenAI list shape. */
export function openAIModelList(models: readonly ListedEntry[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models in the Anthropic list shape. */
export function anthropicModelList(models: readonly ListedEntry[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.id, created_at: FIXED_CREATED_AT })),
    has_more: false,
  };
}

/** An error body in OpenAI's envelope. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, param: null, code } };
}

/** An error body in Anthropic's envelope. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}
