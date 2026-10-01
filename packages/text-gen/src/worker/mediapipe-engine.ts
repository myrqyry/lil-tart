import type {
  LlmInference,
  LlmInferenceOptions,
  Prompt,
} from '@mediapipe/tasks-genai';
import type {
  LiteRtLmWorkerMediaPipeLoadOptions,
  LiteRtLmWorkerPrompt,
} from './protocol';

export const DEFAULT_MEDIAPIPE_GENAI_WASM_BASE =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai@0.10.29/wasm';

type MediaPipeModule = typeof import('@mediapipe/tasks-genai');

interface MediaPipeInferenceLike {
  generateResponse(
    query: Prompt,
    progressListener?: (partialResult: string, done: boolean) => unknown,
  ): Promise<string>;
  cancelProcessing(): void;
  close(): void;
}

interface MediaPipeEngineDeps {
  loadModule?: () => Promise<MediaPipeModule>;
  decodeImage?: (blob: Blob) => Promise<ImageBitmap>;
}

interface PreparedPrompt {
  query: Prompt;
  cleanup(): void;
}

export class MediaPipeMultimodalEngine {
  private inference: MediaPipeInferenceLike | null = null;
  private readonly loadModule: () => Promise<MediaPipeModule>;
  private readonly decodeImage: (blob: Blob) => Promise<ImageBitmap>;

  constructor(deps: MediaPipeEngineDeps = {}) {
    this.loadModule = deps.loadModule ?? (() => import('@mediapipe/tasks-genai'));
    this.decodeImage = deps.decodeImage ?? ((blob) => createImageBitmap(blob));
  }

  async load(
    model: string | Blob,
    options: LiteRtLmWorkerMediaPipeLoadOptions = {},
  ): Promise<void> {
    this.dispose();

    const mediaPipe = await this.loadModule();
    const fileset = await mediaPipe.FilesetResolver.forGenAiTasks(
      options.wasmBaseUrl ?? DEFAULT_MEDIAPIPE_GENAI_WASM_BASE,
    );

    const baseOptions: NonNullable<LlmInferenceOptions['baseOptions']> =
      typeof model === 'string'
        ? { modelAssetPath: model }
        : { modelAssetBuffer: model.stream().getReader() };

    const inferenceOptions: LlmInferenceOptions = {
      baseOptions,
      ...(options.maxTokens !== undefined ? { maxTokens: options.maxTokens } : {}),
      ...(options.topK !== undefined ? { topK: options.topK } : {}),
      ...(options.temperature !== undefined ? { temperature: options.temperature } : {}),
      ...(options.randomSeed !== undefined ? { randomSeed: options.randomSeed } : {}),
      ...(options.maxNumImages !== undefined ? { maxNumImages: options.maxNumImages } : {}),
      ...(options.supportAudio !== undefined ? { supportAudio: options.supportAudio } : {}),
      ...(options.disableRewinding !== undefined
        ? { disableRewinding: options.disableRewinding }
        : {}),
    };

    const inference = await mediaPipe.LlmInference.createFromOptions(
      fileset,
      inferenceOptions,
    );
    this.inference = inference as LlmInference;
  }

  async generate(
    prompt: LiteRtLmWorkerPrompt,
    onToken: (text: string, done: boolean) => void,
    shouldCancel?: () => boolean,
  ): Promise<string> {
    if (!this.inference) {
      throw new Error('MediaPipe LLM Inference is not loaded');
    }

    const prepared = await this.preparePrompt(prompt);
    let streamed = '';

    try {
      // Image decoding is asynchronous. Cancellation can arrive while
      // createImageBitmap() is still running, before MediaPipe itself has any
      // native work to cancel.
      if (shouldCancel?.()) return '';

      const response = await this.inference.generateResponse(
        prepared.query,
        (partialResult, done) => {
          streamed += partialResult;
          onToken(partialResult, done);
        },
      );
      return streamed || response;
    } finally {
      prepared.cleanup();
    }
  }

  /**
   * Ask MediaPipe to cancel the active decoding pass.
   *
   * Upstream 0.10.29 does not cancel initialization or prefilling yet, so callers
   * must still treat cancellation as best-effort until decoding has started.
   */
  cancel(): void {
    this.inference?.cancelProcessing();
  }

  dispose(): void {
    this.inference?.close();
    this.inference = null;
  }

  private async preparePrompt(prompt: LiteRtLmWorkerPrompt): Promise<PreparedPrompt> {
    if (typeof prompt === 'string') {
      return { query: prompt, cleanup: () => undefined };
    }

    const imageBitmaps: ImageBitmap[] = [];
    const query: Array<string | { imageSource: ImageBitmap } | {
      audioSource: { audioSamples: Float32Array; audioSampleRateHz: number };
    }> = [];

    try {
      for (const part of prompt) {
        if (part.type === 'text') {
          query.push(part.text);
          continue;
        }

        if (part.type === 'image') {
          const bitmap = await this.decodeImage(part.data);
          imageBitmaps.push(bitmap);
          query.push({ imageSource: bitmap });
          continue;
        }

        query.push({
          audioSource: {
            audioSamples: part.audioSamples,
            audioSampleRateHz: part.audioSampleRateHz,
          },
        });
      }
    } catch (error) {
      for (const bitmap of imageBitmaps) bitmap.close();
      throw error;
    }

    return {
      query: query as Prompt,
      cleanup: () => {
        for (const bitmap of imageBitmaps) bitmap.close();
      },
    };
  }
}
