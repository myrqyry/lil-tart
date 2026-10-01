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
  createObjectURL?: (blob: Blob) => string;
  revokeObjectURL?: (url: string) => void;
}

interface PreparedPrompt {
  query: Prompt;
  cleanup(): void;
}

export class MediaPipeMultimodalEngine {
  private inference: MediaPipeInferenceLike | null = null;
  private readonly loadModule: () => Promise<MediaPipeModule>;
  private readonly createObjectURL: (blob: Blob) => string;
  private readonly revokeObjectURL: (url: string) => void;

  constructor(deps: MediaPipeEngineDeps = {}) {
    this.loadModule = deps.loadModule ?? (() => import('@mediapipe/tasks-genai'));
    this.createObjectURL = deps.createObjectURL ?? ((blob) => URL.createObjectURL(blob));
    this.revokeObjectURL = deps.revokeObjectURL ?? ((url) => URL.revokeObjectURL(url));
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
  ): Promise<string> {
    if (!this.inference) {
      throw new Error('MediaPipe LLM Inference is not loaded');
    }

    const prepared = this.preparePrompt(prompt);
    let streamed = '';

    try {
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

  private preparePrompt(prompt: LiteRtLmWorkerPrompt): PreparedPrompt {
    if (typeof prompt === 'string') {
      return { query: prompt, cleanup: () => undefined };
    }

    const objectUrls: string[] = [];
    const query = prompt.map((part) => {
      if (part.type === 'text') return part.text;

      const url = this.createObjectURL(part.data);
      objectUrls.push(url);

      if (part.type === 'image') {
        return { imageSource: url };
      }

      return { audioSource: url };
    }) as Prompt;

    return {
      query,
      cleanup: () => {
        for (const url of objectUrls) this.revokeObjectURL(url);
      },
    };
  }
}
