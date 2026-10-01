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
  it('advertises compiled worker engines without claiming runtime availability', () => {
    expect(LiteRtLmWorkerClient.advertisedEngines).toEqual(['litert-lm', 'mediapipe']);
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

  it('allows a later load to switch models or worker engines', async () => {
    const worker = new FakeWorker();
    const client = new LiteRtLmWorkerClient(() => worker);

    const first = client.load('first.litertlm');
    worker.emit({ type: 'ready' });
    await first;

    const second = client.load('second.task', {
      engine: 'mediapipe',
      mediaPipe: { maxNumImages: 1 },
    });
    expect(worker.messages[worker.messages.length - 1]).toEqual({
      type: 'load',
      model: 'second.task',
      options: {
        engine: 'mediapipe',
        mediaPipe: { maxNumImages: 1 },
      },
    });

    worker.emit({ type: 'ready' });
    await second;
  });

  it('deduplicates only identical concurrent load callers', async () => {
    const worker = new FakeWorker();
    const client = new LiteRtLmWorkerClient(() => worker);

    const first = client.load('model.task', { engine: 'mediapipe' });
    const second = client.load('model.task', { engine: 'mediapipe' });

    expect(worker.messages).toHaveLength(1);
    worker.emit({ type: 'ready' });
    await expect(Promise.all([first, second])).resolves.toEqual([undefined, undefined]);
  });

  it('rejects a conflicting concurrent load instead of pretending the first model satisfies it', async () => {
    const worker = new FakeWorker();
    const client = new LiteRtLmWorkerClient(() => worker);

    const first = client.load('first.litertlm');
    await expect(
      client.load('second.task', { engine: 'mediapipe' }),
    ).rejects.toThrow('different worker model or engine is already loading');

    expect(worker.messages).toHaveLength(1);
    worker.emit({ type: 'ready' });
    await first;
  });

  it('rejects operations after disposal instead of posting to a terminated worker', async () => {
    const worker = new FakeWorker();
    const client = new LiteRtLmWorkerClient(() => worker);
    client.dispose();

    await expect(client.load('model.task')).rejects.toThrow('worker client is disposed');
    await expect(client.generate('hello', vi.fn())).rejects.toThrow('worker client is disposed');
    expect(worker.terminate).toHaveBeenCalledTimes(1);
    expect(worker.messages).toEqual([{ type: 'dispose' }]);
  });

  it('uses distinct disposal errors for independent pending generations', async () => {
    const worker = new FakeWorker();
    const client = new LiteRtLmWorkerClient(() => worker);
    const first = client.generate('one', vi.fn());
    const second = client.generate('two', vi.fn());

    client.dispose();

    const [firstResult, secondResult] = await Promise.allSettled([first, second]);
    expect(firstResult.status).toBe('rejected');
    expect(secondResult.status).toBe('rejected');
    if (firstResult.status === 'rejected' && secondResult.status === 'rejected') {
      expect(firstResult.reason).not.toBe(secondResult.reason);
      expect(firstResult.reason.message).toContain('generation 1');
      expect(secondResult.reason.message).toContain('generation 2');
    }
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
