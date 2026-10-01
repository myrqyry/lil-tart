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

  it('converts image/audio Blobs to scoped object URLs and revokes them', async () => {
    const fake = createFakeMediaPipe();
    const createObjectURL = vi
      .fn<(blob: Blob) => string>()
      .mockReturnValueOnce('blob:image')
      .mockReturnValueOnce('blob:audio');
    const revokeObjectURL = vi.fn();
    const engine = new MediaPipeMultimodalEngine({
      loadModule: async () => fake.module,
      createObjectURL,
      revokeObjectURL,
    });
    await engine.load('model.task', { maxNumImages: 1, supportAudio: true });

    const chunks: string[] = [];
    const result = await engine.generate(
      [
        { type: 'text', text: 'Describe this.' },
        { type: 'image', data: new Blob(['image']) },
        { type: 'audio', data: new Blob(['audio']) },
      ],
      (text) => chunks.push(text),
    );

    expect(result).toBe('hello');
    expect(chunks).toEqual(['hel', 'lo']);
    expect(fake.generateResponse).toHaveBeenCalledWith(
      [
        'Describe this.',
        { imageSource: 'blob:image' },
        { audioSource: 'blob:audio' },
      ],
      expect.any(Function),
    );
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:image');
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:audio');

    engine.dispose();
    expect(fake.close).toHaveBeenCalledTimes(1);
  });

  it('delegates active decoding cancellation to MediaPipe', async () => {
    const fake = createFakeMediaPipe();
    const engine = new MediaPipeMultimodalEngine({
      loadModule: async () => fake.module,
    });
    await engine.load('model.task');

    engine.cancel();

    expect(fake.cancelProcessing).toHaveBeenCalledTimes(1);
  });

  it('revokes media URLs when generation fails', async () => {
    const fake = createFakeMediaPipe();
    fake.generateResponse.mockRejectedValueOnce(new Error('generation failed'));
    const revokeObjectURL = vi.fn();
    const engine = new MediaPipeMultimodalEngine({
      loadModule: async () => fake.module,
      createObjectURL: () => 'blob:image',
      revokeObjectURL,
    });
    await engine.load('model.task', { maxNumImages: 1 });

    await expect(
      engine.generate([{ type: 'image', data: new Blob(['image']) }], vi.fn()),
    ).rejects.toThrow('generation failed');
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:image');
  });
});
