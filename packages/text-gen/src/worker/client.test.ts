import { describe, expect, it, vi } from 'vitest';
import { LiteRtLmWorkerClient } from './client';
import type {
  LiteRtLmWorkerRequest,
  LiteRtLmWorkerResponse,
} from './protocol';

class FakeWorker {
  onmessage: ((event: MessageEvent<LiteRtLmWorkerResponse>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  messages: LiteRtLmWorkerRequest[] = [];
  terminate = vi.fn();

  postMessage(message: LiteRtLmWorkerRequest): void {
    this.messages.push(message);
  }

  emit(message: LiteRtLmWorkerResponse): void {
    this.onmessage?.({ data: message } as MessageEvent<LiteRtLmWorkerResponse>);
  }
}

describe('LiteRtLmWorkerClient', () => {
  it('advertises the additive multimodal capability explicitly', () => {
    expect(LiteRtLmWorkerClient.capabilities.mediaPipeMultimodal).toBe(true);
  });

  it('passes explicit MediaPipe load options through without changing text callers', async () => {
    const worker = new FakeWorker();
    const client = new LiteRtLmWorkerClient(() => worker);
    const loading = client.load('model.task', {
      engine: 'mediapipe',
      mediaPipe: { maxNumImages: 1, supportAudio: true },
    });

    expect(worker.messages).toEqual([
      {
        type: 'load',
        model: 'model.task',
        options: {
          engine: 'mediapipe',
          mediaPipe: { maxNumImages: 1, supportAudio: true },
        },
      },
    ]);

    worker.emit({ type: 'ready' });
    await loading;
  });

  it('structured-clones multimodal prompt parts through the worker protocol', async () => {
    const worker = new FakeWorker();
    const client = new LiteRtLmWorkerClient(() => worker);
    const image = new Blob(['image']);
    const tokens: string[] = [];

    const generating = client.generate(
      [
        { type: 'text', text: 'Look.' },
        { type: 'image', data: image },
      ],
      (text) => tokens.push(text),
    );

    const request = worker.messages[0];
    expect(request).toMatchObject({
      type: 'generate',
      id: '1',
      prompt: [
        { type: 'text', text: 'Look.' },
        { type: 'image', data: image },
      ],
    });

    worker.emit({ type: 'token', id: '1', text: 'nice' });
    worker.emit({ type: 'complete', id: '1' });

    await expect(generating).resolves.toBe('nice');
    expect(tokens).toEqual(['nice']);
  });

  it('keeps abort semantics for multimodal generations', async () => {
    const worker = new FakeWorker();
    const client = new LiteRtLmWorkerClient(() => worker);
    const controller = new AbortController();

    const generating = client.generate(
      [{ type: 'image', data: new Blob(['image']) }],
      vi.fn(),
      controller.signal,
    );

    controller.abort();

    await expect(generating).rejects.toMatchObject({ name: 'AbortError' });
    expect(worker.messages[worker.messages.length - 1]).toEqual({
      type: 'cancel',
      id: '1',
    });
  });
});
