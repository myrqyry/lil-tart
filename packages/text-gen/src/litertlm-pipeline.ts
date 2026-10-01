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
  // Treat tool payloads as untrusted at this package boundary. The current
  // @litert-lm/core type is stricter than values observed across integrations,
  // so normalization below owns validation instead of a cast.
  tool_calls?: unknown;
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
  /**
   * Called once, after a tool-bearing stream completes successfully.
   * Buffered calls are intentionally not emitted from an aborted/failed turn.
   */
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

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function normalizeToolArguments(value: unknown): Record<string, unknown> {
  if (value === undefined || value === null) return {};
  if (isRecord(value)) return value;
  if (typeof value === 'string') {
    try {
      const parsed = JSON.parse(value) as unknown;
      if (isRecord(parsed)) return parsed;
    } catch (error) {
      throw new InferenceError('OUTPUT_INVALID', 'LiteRT-LM returned invalid JSON tool arguments', {
        cause: error,
      });
    }
  }
  throw new InferenceError('OUTPUT_INVALID', 'LiteRT-LM returned unsupported tool arguments');
}

function normalizeToolCall(call: unknown): LiteRtLmToolCall {
  if (!isRecord(call) || !isRecord(call.function)) {
    throw new InferenceError('OUTPUT_INVALID', 'LiteRT-LM returned a malformed tool call');
  }
  const name = typeof call.function.name === 'string' ? call.function.name.trim() : '';
  if (!name) {
    throw new InferenceError('OUTPUT_INVALID', 'LiteRT-LM returned a tool call without a function name');
  }
  return {
    name,
    arguments: normalizeToolArguments(call.function.arguments),
  };
}

function attachCleanupFailure(primaryError: Error, cleanupError: unknown): void {
  const errorWithCause = primaryError as Error & { cause?: unknown; cleanupError?: unknown };
  const key = errorWithCause.cause === undefined ? 'cause' : 'cleanupError';
  Object.defineProperty(errorWithCause, key, {
    value: cleanupError,
    configurable: true,
  });
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
  const streamedToolCalls: unknown[] = [];
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
        if (value.tool_calls !== undefined && value.tool_calls !== null) {
          if (!Array.isArray(value.tool_calls)) {
            throw new InferenceError(
              'OUTPUT_INVALID',
              'LiteRT-LM returned a non-array tool_calls payload',
            );
          }
          streamedToolCalls.push(...value.tool_calls);
        }
      }
    }
    // @litert-lm/core's conversation contract emits complete ToolCall objects,
    // and AutoToolChat likewise concatenates chunk.tool_calls. Batch those complete
    // calls until the stream closes so consumers never receive a half-finished turn.
    // We intentionally do not invent OpenAI-style id/index delta merging that this
    // runtime does not expose.
    if (streamedToolCalls.length > 0) {
      onToolCall?.(streamedToolCalls.map(normalizeToolCall));
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

export type LiteRtLmBackend = Exclude<Backend, 'webnn'>;
export type LiteRtLmModelSource =
  | string
  | Blob
  | (() => string | Blob | ReadableStream<Uint8Array>);

export interface LiteRtLmTextPipelineOptions {
  // Base the manifest's repo-relative asset path is resolved against. Omit to
  // pass the path through unchanged.
  modelBase?: string;
  // Downstream consumers may already own/persist model bytes. A stream is
  // intentionally supplied through a factory because ReadableStream is one-shot
  // and a cancelled/failed load must be able to request a fresh source on retry.
  model?: LiteRtLmModelSource;
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
  private conversationCleanup: Promise<void> | null = null;
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
    // RuntimeContext can select WebNN before this package gets a vote. LiteRT-LM
    // does not expose a WebNN backend, so preserve the runtime path's historical
    // degradation behavior and let the engine choose a supported backend.
    const backend = context.backend === 'webnn' ? undefined : context.backend;
    return this.loadInternal(backend, context.signal);
  }

  /**
   * LiteRT-LM only needs a backend and optional cancellation signal. This
   * entrypoint lets downstream consumers use the text package without
   * fabricating unrelated LiteRT tensor/runtime services.
   */
  async loadForBackend(backend: LiteRtLmBackend, signal?: AbortSignal): Promise<void> {
    // Keep a runtime check for untyped/JavaScript callers even though TypeScript
    // excludes WebNN from LiteRtLmBackend.
    if ((backend as Backend) === 'webnn') {
      throw new InferenceError(
        'BACKEND_UNAVAILABLE',
        'LiteRT-LM text generation does not support the WebNN backend',
      );
    }
    return this.loadInternal(backend, signal);
  }

  private async loadInternal(
    backend: LiteRtLmBackend | undefined,
    signal?: AbortSignal,
  ): Promise<void> {
    if (this.status === 'ready') return;
    if (this.status === 'loading') {
      throw new InferenceError(
        'INFERENCE_FAILED',
        'LiteRT-LM model loading is already in progress for this pipeline',
      );
    }
    if (this.disposed) {
      throw new InferenceError('CANCELLED', 'Pipeline was disposed and cannot load again');
    }
    if (this.engine) {
      throw new InferenceError(
        'INFERENCE_FAILED',
        'LiteRT-LM still owns a native engine after a failed conversation cleanup; dispose the pipeline before loading again',
      );
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
      const configuredModel = this.options.model;
      const model =
        configuredModel !== undefined
          ? typeof configuredModel === 'function'
            ? configuredModel()
            : configuredModel
          : await this.resolveModelInput(modelPath, controller.signal);
      if (cancelled()) {
        throw new InferenceError('CANCELLED', 'Model load was cancelled', { asset: modelPath });
      }
      const engine = await module.Engine.create({
        model,
        backend,
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
    let result: TextInferenceResult | undefined;
    let primaryError: Error | undefined;
    let cleanupError: unknown;

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
      result = {
        kind: 'text',
        text,
        ...(reasoning ? { reasoning } : {}),
      } satisfies TextInferenceResult;
    } catch (error) {
      conversation?.cancel();
      primaryError = error instanceof Error ? error : new Error(String(error));
    }

    // Keep the pipeline busy until native conversation cleanup has settled. This
    // prevents a new run or dispose() from racing engine teardown ahead of the
    // conversation that still owns native state.
    if (conversation && this.conversation === conversation) {
      const cleanup = conversation.delete();
      this.conversationCleanup = cleanup;
      try {
        await cleanup;
        if (this.conversation === conversation) this.conversation = null;
      } catch (error) {
        cleanupError = error;
        // Keep the conversation reference so dispose() can retry cleanup.
      } finally {
        if (this.conversationCleanup === cleanup) this.conversationCleanup = null;
      }
    }

    this.status = this.disposed ? 'disposed' : cleanupError ? 'error' : 'ready';

    if (primaryError) {
      if (cleanupError !== undefined) attachCleanupFailure(primaryError, cleanupError);
      throw primaryError;
    }

    if (cleanupError !== undefined) {
      // The model result is already complete; a teardown failure must not replace
      // valid inference output. Mark the pipeline unusable until dispose() retries.
      console.warn('[text-gen] LiteRT-LM conversation cleanup failed', cleanupError);
    }
    return result!;
  }

  async dispose(): Promise<void> {
    // Record disposal before anything else so an in-flight load cannot overwrite it.
    this.disposed = true;
    this.loadAbort?.abort();
    this.loadAbort = null;
    this.loadToken += 1;

    const conversation = this.conversation;
    const pendingConversationCleanup = this.conversationCleanup;
    const engine = this.engine;
    this.conversation = null;
    this.engine = null;
    this.status = 'disposed';

    const errors: unknown[] = [];
    let pendingCleanupError: unknown;
    if (pendingConversationCleanup) {
      try {
        await pendingConversationCleanup;
      } catch (error) {
        pendingCleanupError = error;
      }
    }

    // If run() was already deleting this conversation, only retry when that
    // cleanup failed. Otherwise cancel active generation before deleting.
    if (conversation && (!pendingConversationCleanup || pendingCleanupError !== undefined)) {
      if (!pendingConversationCleanup) {
        try { conversation.cancel(); } catch { /* best-effort cancellation */ }
      }
      try {
        await conversation.delete();
        pendingCleanupError = undefined;
      } catch (error) {
        if (pendingCleanupError !== undefined) errors.push(pendingCleanupError);
        errors.push(error);
      }
    } else if (pendingCleanupError !== undefined) {
      errors.push(pendingCleanupError);
    }

    try { await engine?.delete(); } catch (error) { errors.push(error); }
    if (errors.length === 1) throw errors[0];
    if (errors.length > 1) throw Object.assign(new Error('Pipeline cleanup failed'), { errors });
  }

  private report(progress: PipelineProgress): void {
    this.onProgress?.(progress);
  }
}

export type { TextGenerationConfig, TextGenerationInput, TextMessage };
