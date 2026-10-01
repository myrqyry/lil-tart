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

  it('normalizes omitted defaults and undefined MediaPipe fields for load dedupe', async () => {
    const worker = new FakeWorker();
    const client = new LiteRtLmWorkerClient(() => worker);

    const first = client.load('model.litertlm');
    const second = client.load('model.litertlm', { engine: 'litert-lm' });

    expect(worker.messages).toHaveLength(1);
    worker.emit({ type: 'ready' });
    await expect(Promise.all([first, second])).resolves.toEqual([undefined, undefined]);

    const third = client.load('model.task', {
      engine: 'mediapipe',
      mediaPipe: {},
    });
    const fourth = client.load('model.task', {
      engine: 'mediapipe',
      mediaPipe: { temperature: undefined },
    });

    expect(worker.messages).toHaveLength(2);
    worker.emit({ type: 'ready' });
    await expect(Promise.all([third, fourth])).resolves.toEqual([undefined, undefined]);
  });

  it('deduplicates distinct Blob objects only when the caller supplies the same loadKey', async () => {
    const worker = new FakeWorker();
    const client = new LiteRtLmWorkerClient(() => worker);
    const firstBlob = new Blob([new Uint8Array([1, 2, 3])], { type: 'application/octet-stream' });
    const secondBlob = new Blob([new Uint8Array([1, 2, 3])], { type: 'application/octet-stream' });

    const first = client.load(firstBlob, { engine: 'mediapipe', loadKey: 'cached:model:v1' });
    const second = client.load(secondBlob, { engine: 'mediapipe', loadKey: 'cached:model:v1' });

    expect(worker.messages).toHaveLength(1);
    worker.emit({ type: 'ready' });
    await expect(Promise.all([first, second])).resolves.toEqual([undefined, undefined]);
  });

  it('normalizes caller-owned Blob loadKey whitespace for dedupe identity', async () => {
    const worker = new FakeWorker();
    const client = new LiteRtLmWorkerClient(() => worker);

    const first = client.load(new Blob(['model']), {
      engine: 'mediapipe',
      loadKey: 'cached:model:v1',
    });
    const second = client.load(new Blob(['model']), {
      engine: 'mediapipe',
      loadKey: ' cached:model:v1 ',
    });

    expect(worker.messages).toHaveLength(1);
    worker.emit({ type: 'ready' });
    await expect(Promise.all([first, second])).resolves.toEqual([undefined, undefined]);
  });

  it('does not guess Blob identity from size and MIME type alone', async () => {
    const worker = new FakeWorker();
    const client = new LiteRtLmWorkerClient(() => worker);
    const firstBlob = new Blob(['aaa'], { type: 'application/octet-stream' });
    const secondBlob = new Blob(['bbb'], { type: 'application/octet-stream' });

    const first = client.load(firstBlob, { engine: 'mediapipe' });
    await expect(
      client.load(secondBlob, { engine: 'mediapipe' }),
    ).rejects.toThrow('different worker model or engine is already loading');

    worker.emit({ type: 'ready' });
    await first;
  });

  it('makes the complete load-options structure participate in dedupe identity', async () => {
    const worker = new FakeWorker();
    const client = new LiteRtLmWorkerClient(() => worker);
    const model = new Blob(['model']);

    const first = client.load(model, {
      engine: 'mediapipe',
      mediaPipe: { temperature: 0.5 },
      loadKey: 'model',
    });

    await expect(
      client.load(model, {
        engine: 'mediapipe',
        mediaPipe: { temperature: 0.6 },
        loadKey: 'model',
      }),
    ).rejects.toThrow('different worker model or engine is already loading');

    await expect(
      client.load(model, {
        engine: 'mediapipe',
        mediaPipe: {},
        loadKey: 'model',
      }),
    ).rejects.toThrow('different worker model or engine is already loading');

    worker.emit({ type: 'ready' });
    await first;
  });

  it('degrades exotic untyped load options to a normal conflict instead of throwing from keying', async () => {
    const worker = new FakeWorker();
    const client = new LiteRtLmWorkerClient(() => worker);
    const model = new Blob(['model']);

    const first = client.load(model, {
      engine: 'mediapipe',
      mediaPipe: { temperature: (() => 0.5) as unknown as number },
    });
    await expect(
      client.load(model, {
        engine: 'mediapipe',
        mediaPipe: { temperature: (() => 0.5) as unknown as number },
      }),
    ).rejects.toThrow('different worker model or engine is already loading');

    worker.emit({ type: 'ready' });
    await first;
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
    expect(worker.terminate).not.toHaveBeenCalled();
    expect(worker.messages).toEqual([{ type: 'dispose' }]);

    worker.emit({ type: 'disposed' });
    expect(worker.terminate).toHaveBeenCalledTimes(1);
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

    worker.emit({ type: 'disposed' });
    expect(worker.terminate).toHaveBeenCalledTimes(1);
  });

  it('maps a worker terminal cancellation to AbortError for direct protocol cancellation', async () => {
    const worker = new FakeWorker();
    const client = new LiteRtLmWorkerClient(() => worker);
    const generating = client.generate('hello', vi.fn());

    worker.emit({ type: 'cancelled', id: '1' });

    await expect(generating).rejects.toMatchObject({ name: 'AbortError' });
  });

  it('structured-clones multimodal prompt parts through the worker protocol', async () => {
    const worker = new FakeWorker();
    const client = new LiteRtLmWorkerClient(() => worker);
    const image = new Blob(['image']);
    const audioSamples = new Float32Array([0.1, -0.1]);
    const tokens: string[] = [];

    const generating = client.generate(
      [
        { type: 'text', text: 'Look.' },
        { type: 'image', data: image },
        { type: 'audio', audioSamples, audioSampleRateHz: 16_000 },
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
        { type: 'audio', audioSamples, audioSampleRateHz: 16_000 },
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
