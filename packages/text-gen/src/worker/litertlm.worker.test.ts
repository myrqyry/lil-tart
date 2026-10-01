import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { LiteRtLmWorkerRequest, LiteRtLmWorkerResponse } from './protocol';

const mediaPipe = vi.hoisted(() => ({
  load: vi.fn(),
  generate: vi.fn(),
  cancel: vi.fn(),
  dispose: vi.fn(),
}));

const liteRt = vi.hoisted(() => ({
  createEngine: vi.fn(),
  createConversation: vi.fn(),
  engineDelete: vi.fn(),
  conversationCancel: vi.fn(),
  conversationDelete: vi.fn(),
  sendMessage: vi.fn(),
  sendMessageStreaming: vi.fn(),
}));

vi.mock('@litert-lm/core', () => ({
  Engine: { create: liteRt.createEngine },
}));

vi.mock('./mediapipe-engine', () => ({
  MediaPipeMultimodalEngine: class {
    load(model: string | Blob, options?: unknown) {
      return mediaPipe.load(model, options);
    }
    generate(prompt: unknown, onToken: unknown) {
      return mediaPipe.generate(prompt, onToken);
    }
    cancel() {
      return mediaPipe.cancel();
    }
    dispose() {
      return mediaPipe.dispose();
    }
  },
}));

interface FakeWorkerScope {
  onmessage: ((event: MessageEvent<LiteRtLmWorkerRequest>) => void) | null;
  postMessage: ReturnType<typeof vi.fn>;
}
async function flushWorker(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
  await new Promise((resolve) => setTimeout(resolve, 0));
}

async function bootWorker(): Promise<{
  scope: FakeWorkerScope;
  messages: LiteRtLmWorkerResponse[];
}> {
  vi.resetModules();
  const messages: LiteRtLmWorkerResponse[] = [];
  const scope: FakeWorkerScope = {
    onmessage: null,
    postMessage: vi.fn((message: LiteRtLmWorkerResponse) => messages.push(message)),
  };
  vi.stubGlobal('self', scope);
  await import('./litertlm.worker');
  if (!scope.onmessage) throw new Error('worker did not install onmessage');
  return { scope, messages };
}

function send(scope: FakeWorkerScope, data: LiteRtLmWorkerRequest): void {
  scope.onmessage?.({ data } as MessageEvent<LiteRtLmWorkerRequest>);
}

describe('LiteRT-LM worker dispatch', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mediaPipe.load.mockResolvedValue(undefined);
    mediaPipe.generate.mockResolvedValue('');

    liteRt.sendMessage.mockResolvedValue({ text: '' });
    liteRt.sendMessageStreaming.mockReturnValue(
      new ReadableStream({
        start(controller) {
          controller.close();
        },
      }),
    );
    liteRt.createConversation.mockResolvedValue({
      sendMessage: liteRt.sendMessage,
      sendMessageStreaming: liteRt.sendMessageStreaming,
      cancel: liteRt.conversationCancel,
      delete: liteRt.conversationDelete,
    });
    liteRt.createEngine.mockResolvedValue({
      createConversation: liteRt.createConversation,
      delete: liteRt.engineDelete,
    });
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('rejects MediaPipe per-generation config instead of silently dropping it', async () => {
    const { scope, messages } = await bootWorker();
    send(scope, { type: 'load', model: 'model.task', options: { engine: 'mediapipe' } });
    await flushWorker();

    send(scope, {
      type: 'generate',
      id: 'configured',
      prompt: 'hello',
      config: { temperature: 0.2 },
    });
    await flushWorker();

    expect(mediaPipe.generate).not.toHaveBeenCalled();
    expect(messages).toContainEqual({
      type: 'error',
      id: 'configured',
      message: expect.stringContaining('Per-generation config is not supported'),
    });
  });

  it('serializes reload behind active generation while keeping cancel immediate', async () => {
    const { scope, messages } = await bootWorker();
    send(scope, { type: 'load', model: 'first.task', options: { engine: 'mediapipe' } });
    await flushWorker();
    let releaseGeneration!: () => void;
    mediaPipe.generate.mockImplementationOnce(
      () => new Promise<string>((resolve) => {
        releaseGeneration = () => resolve('');
      }),
    );

    send(scope, { type: 'generate', id: 'active', prompt: 'hello' });
    await flushWorker();
    expect(mediaPipe.generate).toHaveBeenCalledTimes(1);

    send(scope, { type: 'load', model: 'second.task', options: { engine: 'mediapipe' } });
    send(scope, { type: 'cancel', id: 'active' });

    expect(mediaPipe.cancel).toHaveBeenCalledTimes(1);
    expect(mediaPipe.load).toHaveBeenCalledTimes(1);

    releaseGeneration();
    await flushWorker();

    expect(mediaPipe.dispose).toHaveBeenCalledTimes(1);
    expect(mediaPipe.load).toHaveBeenCalledTimes(2);
    expect(messages).not.toContainEqual({ type: 'complete', id: 'active' });
    expect(messages).toContainEqual({ type: 'cancelled', id: 'active' });
    expect(messages.filter((message) => message.type === 'ready')).toHaveLength(2);
  });

  it('cancels LiteRT-LM generation that is waiting for createConversation', async () => {
    const { scope, messages } = await bootWorker();
    send(scope, { type: 'load', model: 'model.litertlm' });
    await flushWorker();

    let releaseConversation!: () => void;
    liteRt.createConversation.mockImplementationOnce(
      () => new Promise((resolve) => {
        releaseConversation = () => resolve({
          sendMessage: liteRt.sendMessage,
          sendMessageStreaming: liteRt.sendMessageStreaming,
          cancel: liteRt.conversationCancel,
          delete: liteRt.conversationDelete,
        });
      }),
    );

    send(scope, { type: 'generate', id: 'litert-pending', prompt: 'hello' });
    await flushWorker();
    expect(liteRt.createConversation).toHaveBeenCalledTimes(1);

    send(scope, { type: 'cancel', id: 'litert-pending' });
    expect(liteRt.conversationCancel).not.toHaveBeenCalled();

    releaseConversation();
    await flushWorker();

    expect(liteRt.conversationDelete).toHaveBeenCalledTimes(1);
    expect(liteRt.sendMessageStreaming).not.toHaveBeenCalled();
    expect(messages).not.toContainEqual({ type: 'complete', id: 'litert-pending' });
    expect(messages).toContainEqual({ type: 'cancelled', id: 'litert-pending' });

    // The generation's finally block must clear bookkeeping: a late cancel for
    // the same id is ignored rather than resurrecting cancellation state.
    send(scope, { type: 'cancel', id: 'litert-pending' });
    expect(liteRt.conversationCancel).not.toHaveBeenCalled();
  });

  it('emits only cancelled when cancellation lands during LiteRT conversation deletion', async () => {
    const { scope, messages } = await bootWorker();
    send(scope, { type: 'load', model: 'model.litertlm' });
    await flushWorker();

    let releaseDelete!: () => void;
    liteRt.conversationDelete.mockImplementationOnce(
      () => new Promise<void>((resolve) => {
        releaseDelete = resolve;
      }),
    );

    send(scope, { type: 'generate', id: 'cleanup-race', prompt: 'hello' });
    await flushWorker();

    expect(liteRt.conversationDelete).toHaveBeenCalledTimes(1);
    expect(messages).not.toContainEqual({ type: 'complete', id: 'cleanup-race' });

    send(scope, { type: 'cancel', id: 'cleanup-race' });
    releaseDelete();
    await flushWorker();

    const terminal = messages.filter(
      (message) =>
        'id' in message &&
        message.id === 'cleanup-race' &&
        (message.type === 'complete' || message.type === 'cancelled'),
    );
    expect(terminal).toEqual([{ type: 'cancelled', id: 'cleanup-race' }]);
  });

  it('cancels active LiteRT-LM generation before serialized dispose tears down the engine', async () => {
    const { scope, messages } = await bootWorker();
    send(scope, { type: 'load', model: 'model.litertlm' });
    await flushWorker();

    let streamController!: ReadableStreamDefaultController<unknown>;
    liteRt.sendMessageStreaming.mockReturnValueOnce(
      new ReadableStream({
        start(controller) {
          streamController = controller;
        },
      }),
    );

    send(scope, { type: 'generate', id: 'litert-active', prompt: 'hello' });
    await flushWorker();
    expect(liteRt.sendMessageStreaming).toHaveBeenCalledTimes(1);

    send(scope, { type: 'dispose' });
    expect(liteRt.conversationCancel).toHaveBeenCalledTimes(1);

    streamController.close();
    await flushWorker();

    expect(liteRt.conversationDelete).toHaveBeenCalledTimes(1);
    expect(liteRt.engineDelete).toHaveBeenCalledTimes(1);
    expect(messages).toContainEqual({ type: 'cancelled', id: 'litert-active' });
    expect(messages).toContainEqual({ type: 'disposed' });
    expect(messages).not.toContainEqual({ type: 'complete', id: 'litert-active' });

    send(scope, { type: 'cancel', id: 'litert-active' });
    expect(liteRt.conversationCancel).toHaveBeenCalledTimes(1);
  });

  it('ignores late cancellation for an unknown generation id', async () => {
    const { scope } = await bootWorker();
    send(scope, { type: 'load', model: 'model.task', options: { engine: 'mediapipe' } });
    await flushWorker();

    send(scope, { type: 'cancel', id: 'already-finished' });

    expect(mediaPipe.cancel).not.toHaveBeenCalled();
  });
});
