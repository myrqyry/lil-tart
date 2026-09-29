import type {
  ConversationConfig,
  Message,
  SamplerParameters,
  SessionConfig,
} from '@litert-lm/core';
import { MediaPipeMultimodalEngine } from './mediapipe-engine';
import type {
  LiteRtLmWorkerEngine,
  LiteRtLmWorkerGenerationConfig,
  LiteRtLmWorkerMessage,
  LiteRtLmWorkerPrompt,
  LiteRtLmWorkerRequest,
  LiteRtLmWorkerResponse,
} from './protocol';

interface WorkerScope {
  onmessage: ((event: MessageEvent) => void) | null;
  postMessage(message: unknown): void;
}

interface LiteRtLmModule {
  Engine: {
    create: (settings: {
      model: string | Blob | ReadableStream<Uint8Array>;
      mainExecutorSettings?: { maxNumTokens?: number };
    }) => Promise<LiteRtLmEngine>;
  };
}

interface LiteRtLmEngine {
  createConversation(config?: ConversationConfig): Promise<LiteRtLmConversation>;
  delete(): Promise<void>;
}

interface LiteRtLmConversation {
  sendMessage(message: Message): Promise<{ text?: string }>;
  sendMessageStreaming(message: Message): ReadableStream<Message>;
  cancel(): void;
  delete(): Promise<void>;
}

type Conversation = Awaited<ReturnType<LiteRtLmEngine['createConversation']>>;

const worker = self as unknown as WorkerScope;

let liteRtLmEngine: LiteRtLmEngine | undefined;
let mediaPipeEngine: MediaPipeMultimodalEngine | undefined;
let activeEngine: LiteRtLmWorkerEngine = 'litert-lm';
let mediaPipeQueue: Promise<void> = Promise.resolve();
const activeConversations = new Map<string, Conversation>();
const mediaPipeGenerationIds = new Set<string>();
const cancelledGenerations = new Set<string>();

function extractText(message: Message): string {
  if (typeof message.content === 'string') return message.content;
  if (Array.isArray(message.content)) {
    return message.content
      .filter((part) => part.type === 'text' && typeof part.text === 'string')
      .map((part) => (part as { text: string }).text)
      .join('');
  }
  return '';
}

function extractReasoning(message: Message): string {
  return message.channels?.reasoning ?? message.channels?.thought ?? message.channels?.think ?? '';
}

function emit(response: LiteRtLmWorkerResponse): void {
  worker.postMessage(response);
}

function buildConversationConfig(config?: LiteRtLmWorkerGenerationConfig): ConversationConfig | undefined {
  const conversationConfig: ConversationConfig = {};
  const hasSampler =
    config?.temperature !== undefined ||
    config?.topK !== undefined ||
    config?.topP !== undefined ||
    config?.seed !== undefined;
  if (hasSampler || config?.maxOutputTokens !== undefined) {
    const sessionConfig: SessionConfig = {};
    if (hasSampler) {
      const samplerParams: SamplerParameters = {};
      if (config?.temperature !== undefined) samplerParams.temperature = config.temperature;
      if (config?.topK !== undefined) samplerParams.k = config.topK;
      if (config?.topP !== undefined) samplerParams.p = config.topP;
      if (config?.seed !== undefined) samplerParams.seed = config.seed;
      sessionConfig.samplerParams = samplerParams;
    }
    if (config?.maxOutputTokens !== undefined) sessionConfig.maxOutputTokens = config.maxOutputTokens;
    conversationConfig.sessionConfig = sessionConfig;
  }
  if (config?.systemPrompt?.trim()) {
    conversationConfig.preface = { messages: [{ role: 'system', content: config.systemPrompt }] };
  }
  return Object.keys(conversationConfig).length > 0 ? conversationConfig : undefined;
}

async function replayHistory(conversation: Conversation, history?: LiteRtLmWorkerMessage[]): Promise<void> {
  if (!history?.length) return;
  for (const message of history) {
    await conversation.sendMessage({
      role: message.role === 'assistant' ? 'model' : message.role,
      content: message.content,
    });
  }
}

async function streamResponse(
  stream: ReadableStream<Message>,
  onMessage: (message: Message) => void,
): Promise<void> {
  if (Symbol.asyncIterator in stream) {
    for await (const chunk of stream) onMessage(chunk);
  } else {
    const reader = (stream as ReadableStream<Message>).getReader();
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        onMessage(value);
      }
    } finally {
      reader.releaseLock();
    }
  }
}

async function disposeConversation(id: string, conversation: Conversation): Promise<void> {
  if (activeConversations.get(id) !== conversation) return;
  activeConversations.delete(id);
  await conversation.delete();
}

function toLiteRtLmText(prompt: LiteRtLmWorkerPrompt): string {
  if (typeof prompt === 'string') return prompt;
  if (prompt.some((part) => part.type !== 'text')) {
    throw new Error(
      'LiteRT-LM JavaScript currently supports text-only prompts; load the worker with engine="mediapipe" for image/audio input',
    );
  }
  return prompt
    .map((part) => (part.type === 'text' ? part.text : ''))
    .join('');
}

async function generateLiteRtLm(
  id: string,
  prompt: LiteRtLmWorkerPrompt,
  config?: LiteRtLmWorkerGenerationConfig,
): Promise<void> {
  if (!liteRtLmEngine) throw new Error('LiteRT-LM is not loaded');
  const conversation = await liteRtLmEngine.createConversation(buildConversationConfig(config));
  activeConversations.set(id, conversation);
  try {
    await replayHistory(conversation, config?.history);
    await streamResponse(
      conversation.sendMessageStreaming({ role: 'user', content: toLiteRtLmText(prompt) }),
      (message) => {
        const reasoning = extractReasoning(message);
        if (reasoning) emit({ type: 'reasoning', id, text: reasoning });
        const text = extractText(message);
        if (text) emit({ type: 'token', id, text });
      },
    );
    emit({ type: 'complete', id });
  } finally {
    await disposeConversation(id, conversation);
  }
}

async function generateMediaPipe(id: string, prompt: LiteRtLmWorkerPrompt): Promise<void> {
  if (!mediaPipeEngine) throw new Error('MediaPipe LLM Inference is not loaded');
  if (cancelledGenerations.has(id)) {
    cancelledGenerations.delete(id);
    return;
  }

  mediaPipeGenerationIds.add(id);
  try {
    await mediaPipeEngine.generate(prompt, (text) => {
      if (!cancelledGenerations.has(id) && text) {
        emit({ type: 'token', id, text });
      }
    });
    if (!cancelledGenerations.has(id)) {
      emit({ type: 'complete', id });
    }
  } finally {
    mediaPipeGenerationIds.delete(id);
    cancelledGenerations.delete(id);
  }
}

async function enqueueMediaPipeGeneration(
  id: string,
  prompt: LiteRtLmWorkerPrompt,
): Promise<void> {
  const task = mediaPipeQueue
    .catch(() => undefined)
    .then(() => generateMediaPipe(id, prompt));
  mediaPipeQueue = task.then(
    () => undefined,
    () => undefined,
  );
  await task;
}

async function disposeAllConversations(): Promise<void> {
  for (const [id, conversation] of activeConversations) {
    conversation.cancel();
    await conversation.delete();
    activeConversations.delete(id);
  }
}

async function disposeLoadedEngine(): Promise<void> {
  await disposeAllConversations();

  if (liteRtLmEngine) {
    await liteRtLmEngine.delete();
    liteRtLmEngine = undefined;
  }

  if (mediaPipeEngine) {
    for (const id of mediaPipeGenerationIds) cancelledGenerations.add(id);
    await mediaPipeQueue.catch(() => undefined);
    mediaPipeEngine.dispose();
    mediaPipeEngine = undefined;
  }
}

async function loadEngine(data: Extract<LiteRtLmWorkerRequest, { type: 'load' }>): Promise<void> {
  await disposeLoadedEngine();

  activeEngine = data.options?.engine ?? 'litert-lm';
  if (activeEngine === 'mediapipe') {
    mediaPipeEngine = new MediaPipeMultimodalEngine();
    await mediaPipeEngine.load(data.model, data.options?.mediaPipe);
    return;
  }

  const module = (await import('@litert-lm/core')) as unknown as LiteRtLmModule;
  liteRtLmEngine = await module.Engine.create({
    model: data.model,
    mainExecutorSettings: { maxNumTokens: 8192 },
  });
}

worker.onmessage = async (event: MessageEvent<LiteRtLmWorkerRequest>) => {
  const data = event.data;
  try {
    switch (data.type) {
      case 'load':
        await loadEngine(data);
        emit({ type: 'ready' });
        break;
      case 'generate':
        if (activeEngine === 'mediapipe') {
          await enqueueMediaPipeGeneration(data.id, data.prompt);
        } else {
          await generateLiteRtLm(data.id, data.prompt, data.config);
        }
        break;
      case 'cancel':
        if (activeEngine === 'mediapipe') {
          cancelledGenerations.add(data.id);
        } else {
          activeConversations.get(data.id)?.cancel();
        }
        break;
      case 'dispose':
        for (const id of mediaPipeGenerationIds) cancelledGenerations.add(id);
        await disposeLoadedEngine();
        emit({ type: 'disposed' });
        break;
    }
  } catch (error) {
    emit({
      type: 'error',
      id: data.type === 'generate' ? data.id : undefined,
      message: error instanceof Error ? error.message : String(error),
    });
  }
};
