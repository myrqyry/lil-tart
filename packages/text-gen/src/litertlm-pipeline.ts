import {
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

// ponytail: manifest asset paths are repo-relative, so they must travel through the
// runtime's asset resolver. Handing the raw path to the engine would let the browser
// resolve it against the app origin instead of the configured model base.
async function resolveModelInput(
  context: RuntimeContext,
  path: string,
): Promise<string | Blob | ReadableStream<Uint8Array>> {
  const assets = context?.assets
  if (typeof assets?.resolve !== 'function') return path
  const asset = { id: path, path }
  if (typeof assets.stream === 'function') {
    return assets.stream(asset, { signal: context.signal })
  }
  return new Blob([await assets.resolve(asset, { signal: context.signal })])
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

  constructor(manifestOrModelId: ModelManifest | string = litertLmManifest) {
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
    this.status = 'loading';
    this.context = context;
    const loadStart = performance.now();
    try {
      this.report({ phase: 'loading', step: 1, total: 2 });
      const module = (await import('@litert-lm/core')) as unknown as LiteRtLmModule;
      this.report({ phase: 'loading', step: 2, total: 2 });
      const modelPath =
        this.manifest.assets.find((a) => a.id === 'model')?.path ??
        this.manifest.assets[0]?.path ??
        DEFAULTS.model;
      const model = await resolveModelInput(context, modelPath);
      const backend = context.backend === 'webnn' ? undefined : context.backend;
      this.engine = await module.Engine.create({
        model,
        backend,
        mainExecutorSettings: { maxNumTokens: DEFAULTS.maxContextTokens },
      });
      this.loadMs = performance.now() - loadStart;
      this.status = 'ready';
    } catch (e) {
      this.status = 'error';
      throw e instanceof Error ? e : new Error(String(e));
    }
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
