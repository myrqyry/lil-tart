import type {
  LiteRtLmWorkerGenerationConfig,
  LiteRtLmWorkerLoadOptions,
  LiteRtLmWorkerPrompt,
  LiteRtLmWorkerRequest,
  LiteRtLmWorkerResponse,
} from './protocol';

export interface LiteRtLmGenerationOptions extends LiteRtLmWorkerGenerationConfig {
  onReasoning?: (text: string) => void;
}

interface WorkerLike {
  postMessage(message: LiteRtLmWorkerRequest): void;
  terminate(): void;
  onmessage: ((event: MessageEvent<LiteRtLmWorkerResponse>) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
}

interface PendingGeneration {
  resolve: (value: string) => void;
  reject: (reason: unknown) => void;
  onToken: ((text: string) => void) | undefined;
  onReasoning?: (text: string) => void;
  result: string;
  cleanup: () => void;
}

interface ActiveLoadRequest {
  model: string | Blob;
  options?: LiteRtLmWorkerLoadOptions;
}

function loadOptionsKey(options?: LiteRtLmWorkerLoadOptions): string {
  const mediaPipe = options?.mediaPipe;
  return JSON.stringify({
    engine: options?.engine ?? 'litert-lm',
    mediaPipe: mediaPipe
      ? {
          wasmBaseUrl: mediaPipe.wasmBaseUrl ?? null,
          maxTokens: mediaPipe.maxTokens ?? null,
          topK: mediaPipe.topK ?? null,
          temperature: mediaPipe.temperature ?? null,
          randomSeed: mediaPipe.randomSeed ?? null,
          maxNumImages: mediaPipe.maxNumImages ?? null,
          supportAudio: mediaPipe.supportAudio ?? null,
          disableRewinding: mediaPipe.disableRewinding ?? null,
        }
      : null,
  });
}

function sameLoadRequest(
  left: ActiveLoadRequest,
  model: string | Blob,
  options?: LiteRtLmWorkerLoadOptions,
): boolean {
  return left.model === model && loadOptionsKey(left.options) === loadOptionsKey(options);
}

function createAbortError(): DOMException {
  return new DOMException('Inference was aborted', 'AbortError');
}

export class LiteRtLmWorkerClient {
  /**
   * Engines compiled into this worker surface. This is intentionally not a
   * runtime capability probe: actual availability is established by load().
   */
  static readonly advertisedEngines = ['litert-lm', 'mediapipe'] as const;

  private worker: WorkerLike;
  private loadPromise: Promise<void> | null = null;
  private loadRequest: ActiveLoadRequest | null = null;
  private loadResolve: (() => void) | null = null;
  private loadReject: ((reason: unknown) => void) | null = null;
  private pending = new Map<string, PendingGeneration>();
  private nextId = 0;
  private disposed = false;

  constructor(createWorker: () => WorkerLike = () => new Worker(new URL('./litertlm.worker.ts', import.meta.url), { type: 'module' })) {
    this.worker = createWorker();
    this.worker.onmessage = (event) => this.handleMessage(event);
    this.worker.onerror = (event) => this.handleError(event);
  }

  async load(model: string | Blob, options?: LiteRtLmWorkerLoadOptions): Promise<void> {
    if (this.disposed) {
      throw new Error('LiteRT-LM worker client is disposed');
    }
    if (this.loadPromise) {
      if (this.loadRequest && sameLoadRequest(this.loadRequest, model, options)) {
        return this.loadPromise;
      }
      throw new Error('A different worker model or engine is already loading');
    }

    this.loadRequest = { model, options };
    const loading = new Promise<void>((resolve, reject) => {
      this.loadResolve = resolve;
      this.loadReject = reject;
      this.worker.postMessage({ type: 'load', model, ...(options ? { options } : {}) });
    });
    this.loadPromise = loading;
    try {
      await loading;
    } finally {
      if (this.loadPromise === loading) {
        this.loadPromise = null;
        this.loadRequest = null;
      }
    }
  }

  generate(
    prompt: LiteRtLmWorkerPrompt,
    onToken: (text: string) => void,
    signal?: AbortSignal,
    options: LiteRtLmGenerationOptions = {},
  ): Promise<string> {
    const id = String(++this.nextId);
    if (this.disposed) return Promise.reject(new Error('LiteRT-LM worker client is disposed'));
    if (signal?.aborted) return Promise.reject(createAbortError());
    const { onReasoning, ...config } = options;
    const hasConfig = Object.keys(config).length > 0;
    return new Promise<string>((resolve, reject) => {
      const abort = (): void => {
        this.worker.postMessage({ type: 'cancel', id });
        this.pending.delete(id);
        cleanup();
        reject(createAbortError());
      };
      const cleanup = (): void => {
        signal?.removeEventListener('abort', abort);
      };
      const pending: PendingGeneration = { resolve, reject, onToken, onReasoning, result: '', cleanup };
      this.pending.set(id, pending);
      signal?.addEventListener('abort', abort, { once: true });
      this.worker.postMessage({ type: 'generate', id, prompt, ...(hasConfig ? { config } : {}) });
    });
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;

    this.loadReject?.(new Error('LiteRT-LM worker disposed during model load'));
    this.loadResolve = null;
    this.loadReject = null;
    this.loadPromise = null;
    this.loadRequest = null;

    this.worker.postMessage({ type: 'dispose' });
    this.worker.terminate();

    for (const [id, pending] of this.pending) {
      pending.cleanup();
      pending.reject(new Error(`LiteRT-LM worker disposed during generation ${id}`));
    }
    this.pending.clear();
  }

  private handleMessage(event: MessageEvent<LiteRtLmWorkerResponse>): void {
    const message = event.data;
    switch (message.type) {
      case 'ready':
        this.loadResolve?.();
        this.loadResolve = null;
        this.loadReject = null;
        break;
      case 'error':
        if (message.id) {
          const pending = this.pending.get(message.id);
          if (pending) {
            this.pending.delete(message.id);
            pending.cleanup();
            pending.reject(new Error(message.message));
          }
        } else {
          this.loadReject?.(new Error(message.message));
          this.loadReject = null;
          this.loadResolve = null;
          this.loadPromise = null;
          this.loadRequest = null;
        }
        break;
      case 'token':
        if (message.text) {
          const pending = this.pending.get(message.id);
          if (pending) {
            pending.result += message.text;
            pending.onToken?.(message.text);
          }
        }
        break;
      case 'reasoning':
        if (message.text) {
          const pending = this.pending.get(message.id);
          pending?.onReasoning?.(message.text);
        }
        break;
      case 'complete':
        {
          const pending = this.pending.get(message.id);
          if (pending) {
            this.pending.delete(message.id);
            pending.cleanup();
            pending.resolve(pending.result);
          }
        }
        break;
      case 'disposed':
        break;
    }
  }

  private handleError(event: ErrorEvent): void {
    this.loadReject?.(new Error(event.message));
    this.loadReject = null;
    this.loadResolve = null;
    this.loadPromise = null;
    this.loadRequest = null;
    for (const pending of this.pending.values()) {
      pending.cleanup();
      pending.reject(new Error(event.message));
    }
    this.pending.clear();
  }
}
