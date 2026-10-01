import { describe, expect, it, vi } from 'vitest';
import { MediaPipeMultimodalEngine } from './mediapipe-engine';

function createFakeMediaPipe() {
  const generateResponse = vi.fn(
    async (
      _query: unknown,
      listener?: (partialResult: string, done: boolean) => unknown,
    ) => {
      listener?.('hel', false);
      listener?.('lo', true);
      return 'hello';
    },
  );
  const cancelProcessing = vi.fn();
  const close = vi.fn();
  const inference = { generateResponse, cancelProcessing, close };
  const forGenAiTasks = vi.fn().mockResolvedValue({ wasmLoaderPath: 'fake' });
  const createFromOptions = vi.fn().mockResolvedValue(inference);

  const module = {
    FilesetResolver: { forGenAiTasks },
    LlmInference: { createFromOptions },
  } as unknown as typeof import('@mediapipe/tasks-genai');

  return {
    module,
    generateResponse,
    cancelProcessing,
    close,
    forGenAiTasks,
    createFromOptions,
  };
}

describe('MediaPipeMultimodalEngine', () => {
  it('loads a remote model directly and preserves multimodal options', async () => {
    const fake = createFakeMediaPipe();
    const engine = new MediaPipeMultimodalEngine({
      loadModule: async () => fake.module,
    });

    await engine.load('https://models.example/gemma.task', {
      wasmBaseUrl: 'https://cdn.example/wasm',
      maxTokens: 1024,
      topK: 40,
      temperature: 0.7,
      randomSeed: 42,
      maxNumImages: 2,
      supportAudio: true,
      disableRewinding: true,
    });

    expect(fake.forGenAiTasks).toHaveBeenCalledWith('https://cdn.example/wasm');
    expect(fake.createFromOptions).toHaveBeenCalledWith(
      expect.anything(),
      {
        baseOptions: { modelAssetPath: 'https://models.example/gemma.task' },
        maxTokens: 1024,
        topK: 40,
        temperature: 0.7,
        randomSeed: 42,
        maxNumImages: 2,
        supportAudio: true,
        disableRewinding: true,
      },
    );
  });

  it('streams a Blob model instead of materializing it into an ArrayBuffer', async () => {
    const fake = createFakeMediaPipe();
    const engine = new MediaPipeMultimodalEngine({
      loadModule: async () => fake.module,
    });
    const model = new Blob([new Uint8Array([1, 2, 3])]);

    await engine.load(model);

    const options = fake.createFromOptions.mock.calls[0]?.[1];
    const reader = options?.baseOptions?.modelAssetBuffer;
    expect(reader).toBeDefined();
    expect(typeof reader?.read).toBe('function');
    expect(options?.baseOptions?.modelAssetPath).toBeUndefined();
  });

  it('decodes image Blobs and forwards worker-safe PCM audio chunks', async () => {
    const fake = createFakeMediaPipe();
    const closeBitmap = vi.fn();
    const bitmap = { width: 1, height: 1, close: closeBitmap } as unknown as ImageBitmap;
    const decodeImage = vi.fn().mockResolvedValue(bitmap);
    const engine = new MediaPipeMultimodalEngine({
      loadModule: async () => fake.module,
      decodeImage,
    });
    await engine.load('model.task', { maxNumImages: 1, supportAudio: true });

    const chunks: string[] = [];
    const audioSamples = new Float32Array([0.25, -0.25]);
    const result = await engine.generate(
      [
        { type: 'text', text: 'Describe this.' },
        { type: 'image', data: new Blob(['image']) },
        { type: 'audio', audioSamples, audioSampleRateHz: 16_000 },
      ],
      (text) => chunks.push(text),
    );

    expect(result).toBe('hello');
    expect(chunks).toEqual(['hel', 'lo']);
    expect(decodeImage).toHaveBeenCalledTimes(1);
    expect(fake.generateResponse).toHaveBeenCalledWith(
      [
        'Describe this.',
        { imageSource: bitmap },
        { audioSource: { audioSamples, audioSampleRateHz: 16_000 } },
      ],
      expect.any(Function),
    );
    expect(closeBitmap).toHaveBeenCalledTimes(1);

    engine.dispose();
    expect(fake.close).toHaveBeenCalledTimes(1);
  });

  it('does not start native inference when cancellation arrives during image decoding', async () => {
    const fake = createFakeMediaPipe()
    const closeBitmap = vi.fn()
    const bitmap = { width: 1, height: 1, close: closeBitmap } as unknown as ImageBitmap
    let releaseDecode!: () => void
    const decodeImage = vi.fn(
      () => new Promise<ImageBitmap>((resolve) => {
        releaseDecode = () => resolve(bitmap)
      }),
    )
    const engine = new MediaPipeMultimodalEngine({
      loadModule: async () => fake.module,
      decodeImage,
    })
    await engine.load('model.task', { maxNumImages: 2 })

    let cancelled = false
    const running = engine.generate(
      [
        { type: 'image', data: new Blob(['image-1']) },
        { type: 'image', data: new Blob(['image-2']) },
      ],
      vi.fn(),
      () => cancelled,
    )

    cancelled = true
    releaseDecode()

    await expect(running).resolves.toBe('')
    expect(fake.generateResponse).not.toHaveBeenCalled()
    expect(decodeImage).toHaveBeenCalledTimes(1)
    expect(closeBitmap).toHaveBeenCalledTimes(1)
  })

  it('delegates active decoding cancellation to MediaPipe', async () => {
    const fake = createFakeMediaPipe();
    const engine = new MediaPipeMultimodalEngine({
      loadModule: async () => fake.module,
    });
    await engine.load('model.task');

    engine.cancel();

    expect(fake.cancelProcessing).toHaveBeenCalledTimes(1);
  });

  it('closes decoded image resources when generation fails', async () => {
    const fake = createFakeMediaPipe();
    fake.generateResponse.mockRejectedValueOnce(new Error('generation failed'));
    const closeBitmap = vi.fn();
    const bitmap = { width: 1, height: 1, close: closeBitmap } as unknown as ImageBitmap;
    const engine = new MediaPipeMultimodalEngine({
      loadModule: async () => fake.module,
      decodeImage: async () => bitmap,
    });
    await engine.load('model.task', { maxNumImages: 1 });

    await expect(
      engine.generate([{ type: 'image', data: new Blob(['image']) }], vi.fn()),
    ).rejects.toThrow('generation failed');
    expect(closeBitmap).toHaveBeenCalledTimes(1);
  });
});
