import {
  InferenceError,
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
}

export interface LiteRtLmTextConfig extends TextGenerationConfig {
  model?: string | Blob | ReadableStream<Uint8Array>;
  maxContextTokens?: number;
  maxOutputTokens?: number;
  seed?: number;
  history?: TextMessage[];
  onToken?: (text: string) => void;
  onReasoning?: (text: string) => void;
}

const DEFAULTS: Pick<LiteRtLmTextConfig, 'model' | 'maxContextTokens' | 'maxOutputTokens'> = {
  model: 'litert-community/Qwen3-0.6B/resolve/main/Qwen3-0.6B.litertlm',
  maxContextTokens: 4096,
  maxOutputTokens: TEXT_GENERATION_DEFAULTS.maxTokens,
};

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
  if (input.systemPrompt?.trim()) {
    conversationConfig.preface = { messages: [{ role: 'system', content: input.systemPrompt }] };
  }
  return Object.keys(conversationConfig).length > 0 ? conversationConfig : undefined;
}

async function readStream(
  stream: ReadableStream<StreamChunk>,
  signal: AbortSignal | undefined,
  onToken: ((text: string) => void) | undefined,
  onReasoning: ((text: string) => void) | undefined,
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
      }
    }
  } finally {
    reader.releaseLock();
  }
  return { text: full, ...(reasoning ? { reasoning } : {}) };
}

// ponytail: manifest asset paths are repo-relative. Handing one to the engine raw lets the
// browser resolve it against the app origin, which 404s, so the caller supplies the model
// base and the path is made absolute. The engine then fetches and streams the checkpoint
// itself. Deliberately NOT routed through context.assets.stream(): resolvers in this repo
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
}

export class LiteRtLmTextPipeline
  implements Pipeline<TextGenerationInput, TextInferenceResult, LiteRtLmTextConfig>
{
  readonly manifest: ModelManifest;
  status: PipelineStatus = 'idle';
  onProgress?: (progress: PipelineProgress) => void;

  private context: RuntimeContext | null = null;
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
    if (this.status === 'ready') return;
    // A disposed pipeline stays disposed; nothing may load through it again.
    if (this.disposed) return;
    this.status = 'loading';
    this.context = context;
    const loadStart = performance.now();
    const controller = new AbortController();
    this.loadAbort = controller;
    const token = ++this.loadToken;
    // Cancelling from either side has to reach the transfer: the caller's signal,
    // or dispose() on this pipeline. An already-aborted signal has already dispatched
    // its event and will never dispatch again, so it has to be checked directly --
    // the same thing the runtime does before subscribing.
    const onExternalAbort = () => controller.abort();
    if (context.signal?.aborted) controller.abort();
    else context.signal?.addEventListener('abort', onExternalAbort, { once: true });
    try {
      this.report({ phase: 'loading', step: 1, total: 2 });
      const module = (await import('@litert-lm/core')) as unknown as LiteRtLmModule;
      this.report({ phase: 'loading', step: 2, total: 2 });
      const modelPath =
        this.manifest.assets.find((a) => a.id === 'model')?.path ??
        this.manifest.assets[0]?.path ??
        DEFAULTS.model;
      const model = await this.resolveModelInput(modelPath, controller.signal);
      // Do not start a multi-hundred-megabyte compile for a pipeline that was
      // disposed while the model was still arriving. Reject rather than resolve:
      // the caller asked for a model and did not get one.
      if (token !== this.loadToken) {
        throw new InferenceError('CANCELLED', 'Pipeline was disposed during model load', {
          asset: modelPath,
        });
      }
      const backend = context.backend === 'webnn' ? undefined : context.backend;
      const engine = await module.Engine.create({
        model,
        backend,
        mainExecutorSettings: { maxNumTokens: DEFAULTS.maxContextTokens },
      });
      // dispose() can land while the engine is still compiling, and aborting the
      // fetch cannot stop that. The late result must be released, not published,
      // or a disposed pipeline resurrects itself and leaks the engine.
      if (token !== this.loadToken) {
        await engine.delete();
        throw new InferenceError('CANCELLED', 'Pipeline was disposed during model load', {
          asset: modelPath,
        });
      }
      this.engine = engine;
      this.loadMs = performance.now() - loadStart;
      this.status = 'ready';
    } catch (e) {
      // A cancelled load is not a failure of the model; leave the pipeline
      // retryable rather than latched to 'error'. Disposal wins the race: a
      // cancelled load must not stamp 'idle' over an already disposed pipeline.
      this.status = this.disposed ? 'disposed' : controller.signal.aborted ? 'idle' : 'error';
      throw e instanceof Error ? e : new Error(String(e));
    } finally {
      context.signal?.removeEventListener('abort', onExternalAbort);
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
    try {
      if (signal?.aborted) throw new Error('CANCELLED');
      const messages = toLiteRtMessages(input, cfg.history);
      this.conversation = await this.engine.createConversation(buildConversationConfig(input, cfg));
      const prompt = messages.pop()!;
      for (const msg of messages) {
        if (signal?.aborted) throw new Error('CANCELLED');
        await this.conversation.sendMessage(msg);
      }
      if (signal?.aborted) throw new Error('CANCELLED');
      const stream = this.conversation.sendMessageStreaming(prompt);
      const { text, reasoning } = await readStream(stream, signal, cfg.onToken, cfg.onReasoning);
      this.status = 'ready';
      return {
        kind: 'text',
        text,
        ...(reasoning ? { reasoning } : {}),
      } satisfies TextInferenceResult;
    } catch (e) {
      this.conversation?.cancel();
      this.status = 'ready';
      throw e instanceof Error ? e : new Error(String(e));
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
    await this.conversation?.delete();
    this.conversation = null;
    await this.engine?.delete();
    this.engine = null;
    this.context = null;
    this.status = 'disposed';
  }

  private report(progress: PipelineProgress): void {
    this.onProgress?.(progress);
  }
}

export type { TextGenerationConfig, TextGenerationInput, TextMessage };
