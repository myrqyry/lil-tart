import {
  InferenceError,
  type Backend,
  type ModelManifest,
  type Pipeline,
  type PipelineProgress,
  type PipelineStatus,
  type RuntimeContext,
  type TextInferenceResult,
} from '@litert-playground/inference-core';
import type {
  ConversationConfig,
  SamplerParameters,
  SessionConfig,
  Tool,
  ToolCall,
} from '@litert-lm/core';
import {
  type TextGenerationConfig,
  type TextGenerationInput,
  type TextMessage,
  TEXT_GENERATION_DEFAULTS,
} from './types';
import {
  litertLmManifest,
  lfm2_5InstructManifest,
  lfm2_5InstructInt8Manifest,
  lfm2_5ThinkingManifest,
  lfm2_5ThinkingInt8Manifest,
  gemma4E2bManifest,
  gemma4E4bManifest,
} from './manifest';

interface LiteRtLmModule {
  Engine: {
    create: (settings: {
      model: string | Blob | ReadableStream<Uint8Array>;
      backend?: 'webgpu' | 'wasm' | 'cpu';
      mainExecutorSettings?: { maxNumTokens?: number };
    }) => Promise<LiteRtLmEngine>;
  };
}

interface LiteRtLmEngine {
  createConversation(config?: ConversationConfig): Promise<LiteRtLmConversation>;
  delete(): Promise<void>;
}

interface LiteRtLmConversation {
  sendMessage(
    message: { role: string; content: string } | Array<{ role: string; content: string }>,
  ): Promise<{ text?: string }>;
  sendMessageStreaming(
    message: { role: string; content: string } | Array<{ role: string; content: string }>,
  ): ReadableStream<StreamChunk>;
  cancel(): void;
  delete(): Promise<void>;
}

interface StreamChunk {
  text?: string;
  content?: string | Array<{ type?: string; text?: string }>;
  channels?: Record<string, string>;
  tool_calls?: ToolCall[];
}

export interface LiteRtLmToolCall {
  name: string;
  arguments: Record<string, unknown>;
}

export interface LiteRtLmTextConfig extends TextGenerationConfig {
  model?: string | Blob | ReadableStream<Uint8Array>;
  maxContextTokens?: number;
  maxOutputTokens?: number;
  seed?: number;
  history?: TextMessage[];
  tools?: Tool[];
  onToken?: (text: string) => void;
  onReasoning?: (text: string) => void;
  onToolCall?: (calls: LiteRtLmToolCall[]) => void;
}

const DEFAULTS = {
  model: 'litert-community/Qwen3-0.6B/resolve/main/Qwen3-0.6B.litertlm',
  maxContextTokens: 4096,
  maxOutputTokens: TEXT_GENERATION_DEFAULTS.maxTokens,
} satisfies Pick<LiteRtLmTextConfig, 'model' | 'maxContextTokens' | 'maxOutputTokens'>;

const KNOWN_MANIFESTS: ModelManifest[] = [
  litertLmManifest,
  lfm2_5InstructManifest,
  lfm2_5InstructInt8Manifest,
  lfm2_5ThinkingManifest,
  lfm2_5ThinkingInt8Manifest,
  gemma4E2bManifest,
  gemma4E4bManifest,
];

export function resolveTextGenerationManifest(modelId: string): ModelManifest | undefined {
  return KNOWN_MANIFESTS.find((manifest) => manifest.modelId === modelId);
}

function toLiteRtMessages(input: TextGenerationInput, history: TextMessage[] = []): Array<{ role: string; content: string }> {
  const messages: Array<{ role: string; content: string }> = [];
  for (const msg of history) {
    messages.push({ role: msg.role === 'model' ? 'assistant' : msg.role, content: msg.content });
  }
  for (const msg of input.messages) {
    const role = msg.role === 'model' ? 'assistant' : msg.role;
    messages.push({ role, content: msg.content });
  }
  return messages;
}

function extractText(message: StreamChunk): string {
  if (typeof message.text === 'string') return message.text;
  if (typeof message.content === 'string') return message.content;
  if (Array.isArray(message.content)) {
    return message.content
      .filter((part) => part.type === 'text' && typeof part.text === 'string')
      .map((part) => part.text)
      .join('');
  }
  return '';
}

function extractReasoning(message: StreamChunk): string {
  return message.channels?.reasoning ?? message.channels?.thought ?? message.channels?.think ?? '';
}

function extractToolCalls(message: StreamChunk): LiteRtLmToolCall[] {
  return (message.tool_calls ?? [])
    .map((call) => ({
      name: call.function?.name ?? '',
      arguments: (call.function?.arguments ?? {}) as Record<string, unknown>,
    }))
    .filter((call) => call.name.length > 0);
}

function buildConversationConfig(input: TextGenerationInput, config: LiteRtLmTextConfig): ConversationConfig | undefined {
  const conversationConfig: ConversationConfig = {};
  const hasSampler =
    config.temperature !== undefined ||
    config.topK !== undefined ||
    config.topP !== undefined ||
    config.seed !== undefined;
  if (hasSampler || config.maxOutputTokens !== undefined || config.maxTokens !== undefined) {
    const sessionConfig: SessionConfig = {};
    if (hasSampler) {
      const samplerParams: SamplerParameters = {};
      if (config.temperature !== undefined) samplerParams.temperature = config.temperature;
      if (config.topK !== undefined) samplerParams.k = config.topK;
      if (config.topP !== undefined) samplerParams.p = config.topP;
      if (config.seed !== undefined) samplerParams.seed = config.seed;
      sessionConfig.samplerParams = samplerParams;
    }
    sessionConfig.maxOutputTokens = config.maxOutputTokens ?? config.maxTokens;
    conversationConfig.sessionConfig = sessionConfig;
  }
  if (input.systemPrompt?.trim() || config.tools?.length) {
    conversationConfig.preface = {};
    if (input.systemPrompt?.trim()) {
      conversationConfig.preface.messages = [{ role: 'system', content: input.systemPrompt }];
    }
    if (config.tools?.length) conversationConfig.preface.tools = config.tools;
  }
  return Object.keys(conversationConfig).length > 0 ? conversationConfig : undefined;
}

async function readStream(
  stream: ReadableStream<StreamChunk>,
  signal: AbortSignal | undefined,
  onToken: ((text: string) => void) | undefined,
  onReasoning: ((text: string) => void) | undefined,
  onToolCall: ((calls: LiteRtLmToolCall[]) => void) | undefined,
): Promise<{ text: string; reasoning?: string }> {
  const reader = stream.getReader();
  let full = '';
  let reasoning = '';
  try {
    while (true) {
      if (signal?.aborted) throw new Error('CANCELLED');
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        const text = extractText(value);
        if (text) {
          full += text;
          onToken?.(text);
        }
        const reason = extractReasoning(value);
        if (reason) {
          reasoning += reason;
          onReasoning?.(reason);
        }
        const toolCalls = extractToolCalls(value);
        if (toolCalls.length > 0) onToolCall?.(toolCalls);
      }
    }
  } finally {
    reader.releaseLock();
  }
  return { text: full, ...(reasoning ? { reasoning } : {}) };
}

// ponytail: manifest asset paths are repo-relative. Handing one to the engine raw lets the
// browser resolve it against the app origin, which 404s, so the caller supplies the model
// base and the path is made absolute. The pipeline fetches with an abort signal
// and passes the response body to the engine. Deliberately not routed through
// context.assets.stream(): resolvers in this repo
// materialize the whole checkpoint into an ArrayBuffer before wrapping it in a
// ReadableStream, and a full buffered read is the wrong memory trade for a
// multi-billion-parameter model in a browser.
function resolveModelReference(path: string, base?: string): string {
  if (!base) return path
  const pageBase = (globalThis as { location?: { href: string } }).location?.href ?? 'http://localhost/'
  try {
    return new URL(path, new URL(base, pageBase)).href
  } catch {
    return path
  }
}

function modelFetchError(path: string, error: unknown, signal: AbortSignal): Error {
  if (signal.aborted) {
    return new InferenceError('CANCELLED', `Model download cancelled for ${path}`, { asset: path, cause: error })
  }
  return new InferenceError('ASSET_FETCH_FAILED', `Failed to fetch model ${path}: ${String(error)}`, {
    asset: path,
    cause: error,
  })
}

export interface LiteRtLmTextPipelineOptions {
  // Base the manifest's repo-relative asset path is resolved against. Omit to
  // pass the path through unchanged.
  modelBase?: string;
  // Downstream consumers may already own/persist model bytes. Supplying a model
  // keeps that cache authoritative instead of forcing a second network fetch.
  model?: string | Blob | ReadableStream<Uint8Array>;
  maxContextTokens?: number;
}

export class LiteRtLmTextPipeline
  implements Pipeline<TextGenerationInput, TextInferenceResult, LiteRtLmTextConfig>
{
  readonly manifest: ModelManifest;
  status: PipelineStatus = 'idle';
  onProgress?: (progress: PipelineProgress) => void;

  private engine: LiteRtLmEngine | null = null;
  private conversation: LiteRtLmConversation | null = null;
  private loadMs = 0;
  private loadAbort: AbortController | null = null;
  private loadToken = 0;
  private disposed = false;
  private readonly options: LiteRtLmTextPipelineOptions;

  constructor(
    manifestOrModelId: ModelManifest | string = litertLmManifest,
    options: LiteRtLmTextPipelineOptions = {}
  ) {
    this.options = options;
    this.manifest =
      typeof manifestOrModelId === 'string'
        ? resolveTextGenerationManifest(manifestOrModelId) ?? {
            ...litertLmManifest,
            modelId: manifestOrModelId,
            name: manifestOrModelId,
          }
        : manifestOrModelId;
  }

  async load(context: RuntimeContext): Promise<void> {
    return this.loadInternal(context.backend, context.signal);
  }

  /**
   * LiteRT-LM only needs a backend and optional cancellation signal. This
   * entrypoint lets downstream consumers use the text package without
   * fabricating unrelated LiteRT tensor/runtime services.
   */
  async loadForBackend(backend: Backend, signal?: AbortSignal): Promise<void> {
    return this.loadInternal(backend, signal);
  }

  private async loadInternal(backend: Backend, signal?: AbortSignal): Promise<void> {
    if (this.status === 'ready') return;
    if (this.disposed) {
      throw new InferenceError('CANCELLED', 'Pipeline was disposed and cannot load again');
    }
    this.status = 'loading';
    const loadStart = performance.now();
    const controller = new AbortController();
    this.loadAbort = controller;
    const token = ++this.loadToken;
    const onExternalAbort = () => controller.abort();
    if (signal?.aborted) controller.abort();
    else signal?.addEventListener('abort', onExternalAbort, { once: true });
    const cancelled = () => token !== this.loadToken || controller.signal.aborted;
    try {
      this.report({ phase: 'loading', step: 1, total: 2 });
      const module = (await import('@litert-lm/core')) as unknown as LiteRtLmModule;
      this.report({ phase: 'loading', step: 2, total: 2 });
      const modelPath =
        this.manifest.assets.find((a) => a.id === 'model')?.path ??
        this.manifest.assets[0]?.path ??
        DEFAULTS.model;
      const model =
        this.options.model ?? await this.resolveModelInput(modelPath, controller.signal);
      if (cancelled()) {
        throw new InferenceError('CANCELLED', 'Model load was cancelled', { asset: modelPath });
      }
      const resolvedBackend = backend === 'webnn' ? undefined : backend;
      const engine = await module.Engine.create({
        model,
        backend: resolvedBackend,
        mainExecutorSettings: {
          maxNumTokens: this.options.maxContextTokens ?? DEFAULTS.maxContextTokens,
        },
      });
      if (cancelled()) {
        try {
          await engine.delete();
        } catch (cleanupError) {
          throw new InferenceError('CANCELLED', 'Model load was cancelled', {
            asset: modelPath,
            cause: cleanupError,
          });
        }
        throw new InferenceError('CANCELLED', 'Model load was cancelled', { asset: modelPath });
      }
      this.engine = engine;
      this.loadMs = performance.now() - loadStart;
      this.status = 'ready';
    } catch (e) {
      this.status = this.disposed ? 'disposed' : controller.signal.aborted ? 'idle' : 'error';
      throw e instanceof Error ? e : new Error(String(e));
    } finally {
      signal?.removeEventListener('abort', onExternalAbort);
    }
  }

  // With a model base, the fetch happens here so it can be aborted. The engine's own
  // model fetch takes no signal, so a multi-gigabyte checkpoint cannot otherwise be
  // cancelled when the caller switches models mid-download. Handing the engine the
  // response body keeps the memory and streaming behaviour identical, because
  // Engine.create treats a string and a ReadableStream the same way from there on.
  private async resolveModelInput(
    path: string,
    signal: AbortSignal,
  ): Promise<string | ReadableStream<Uint8Array>> {
    const base = this.options.modelBase
    // No base means nothing to resolve against, so keep the engine's own fetch.
    if (!base) return path

    const url = resolveModelReference(path, base)
    let response: Response
    try {
      // credentials mirrors the engine's model fetch so a same-origin model server
      // still works.
      response = await fetch(url, { signal, credentials: 'same-origin' })
    } catch (error) {
      throw modelFetchError(path, error, signal)
    }
    if (!response.ok || !response.body) {
      throw new InferenceError('ASSET_FETCH_FAILED', `HTTP ${response.status} fetching model ${path}`, {
        asset: path,
      })
    }
    return response.body
  }

  async run(
    input: TextGenerationInput,
    config?: LiteRtLmTextConfig,
    signal?: AbortSignal,
  ): Promise<TextInferenceResult> {
    if (this.status !== 'ready') throw new Error('Pipeline not ready');
    if (!this.engine) throw new Error('LiteRT-LM pipeline not loaded');
    this.status = 'running';
    const cfg = { ...DEFAULTS, ...config };
    let conversation: LiteRtLmConversation | null = null;
    try {
      if (signal?.aborted) throw new Error('CANCELLED');
      const messages = toLiteRtMessages(input, cfg.history);
      conversation = await this.engine.createConversation(buildConversationConfig(input, cfg));
      this.conversation = conversation;
      const prompt = messages.pop()!;
      for (const msg of messages) {
        if (signal?.aborted) throw new Error('CANCELLED');
        await conversation.sendMessage(msg);
      }
      if (signal?.aborted) throw new Error('CANCELLED');
      const stream = conversation.sendMessageStreaming(prompt);
      const { text, reasoning } = await readStream(
        stream,
        signal,
        cfg.onToken,
        cfg.onReasoning,
        cfg.onToolCall,
      );
      return {
        kind: 'text',
        text,
        ...(reasoning ? { reasoning } : {}),
      } satisfies TextInferenceResult;
    } catch (e) {
      conversation?.cancel();
      throw e instanceof Error ? e : new Error(String(e));
    } finally {
      this.status = this.disposed ? 'disposed' : 'ready';
      // dispose() takes ownership by clearing this.conversation before deleting it.
      // Only the run that still owns the slot should release the conversation here;
      // this avoids both native-resource leaks across runs and double-deletes when
      // disposal races an in-flight generation.
      if (conversation && this.conversation === conversation) {
        this.conversation = null;
        await conversation.delete();
      }
    }
  }

  async dispose(): Promise<void> {
    // Record disposal before anything else so an in-flight load cannot overwrite it.
    this.disposed = true;
    // Stop an in-flight model download even when the caller wired no signal.
    this.loadAbort?.abort();
    this.loadAbort = null;
    // Invalidate any load still awaiting a result.
    this.loadToken += 1;
    const conversation = this.conversation;
    const engine = this.engine;
    this.conversation = null;
    this.engine = null;
    this.status = 'disposed';
    // Each cleanup gets an attempt even if the other rejects. Keep the terminal
    // state truthful and propagate cleanup errors to the caller.
    const errors: unknown[] = [];
    try { await conversation?.delete(); } catch (error) { errors.push(error); }
    try { await engine?.delete(); } catch (error) { errors.push(error); }
    if (errors.length === 1) throw errors[0];
    if (errors.length > 1) throw Object.assign(new Error('Pipeline cleanup failed'), { errors });
  }

  private report(progress: PipelineProgress): void {
    this.onProgress?.(progress);
  }
}

export type { TextGenerationConfig, TextGenerationInput, TextMessage };
