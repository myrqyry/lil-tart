import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { LiteRtLmWorkerRequest, LiteRtLmWorkerResponse } from './protocol';

const mediaPipe = vi.hoisted(() => ({
  load: vi.fn(),
  generate: vi.fn(),
  cancel: vi.fn(),
  dispose: vi.fn(),
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
    expect(messages.filter((message) => message.type === 'ready')).toHaveLength(2);
  });

  it('ignores late cancellation for an unknown generation id', async () => {
    const { scope } = await bootWorker();
    send(scope, { type: 'load', model: 'model.task', options: { engine: 'mediapipe' } });
    await flushWorker();

    send(scope, { type: 'cancel', id: 'already-finished' });

    expect(mediaPipe.cancel).not.toHaveBeenCalled();
  });
});
