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

function createAbortError(): DOMException {
  return new DOMException('Inference was aborted', 'AbortError');
}

export class LiteRtLmWorkerClient {
  static readonly capabilities = {
    mediaPipeMultimodal: true,
  } as const;

  private worker: WorkerLike;
  private loadPromise: Promise<void> | null = null;
  private loadResolve: (() => void) | null = null;
  private loadReject: ((reason: unknown) => void) | null = null;
  private pending = new Map<string, PendingGeneration>();
  private nextId = 0;

  constructor(createWorker: () => WorkerLike = () => new Worker(new URL('./litertlm.worker.ts', import.meta.url), { type: 'module' })) {
    this.worker = createWorker();
    this.worker.onmessage = (event) => this.handleMessage(event);
    this.worker.onerror = (event) => this.handleError(event);
  }

  async load(model: string | Blob, options?: LiteRtLmWorkerLoadOptions): Promise<void> {
    if (this.loadPromise) return this.loadPromise;
    this.loadPromise = new Promise<void>((resolve, reject) => {
      this.loadResolve = resolve;
      this.loadReject = reject;
      this.worker.postMessage({ type: 'load', model, ...(options ? { options } : {}) });
    });
    try {
      await this.loadPromise;
    } catch (error) {
      this.loadPromise = null;
      throw error;
    }
    return this.loadPromise;
  }

  generate(
    prompt: LiteRtLmWorkerPrompt,
    onToken: (text: string) => void,
    signal?: AbortSignal,
    options: LiteRtLmGenerationOptions = {},
  ): Promise<string> {
    const id = String(++this.nextId);
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
    this.worker.postMessage({ type: 'dispose' });
    this.worker.terminate();
    for (const pending of this.pending.values()) {
      pending.cleanup();
      pending.reject(new Error('LiteRT-LM worker disposed'));
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
    for (const pending of this.pending.values()) {
      pending.cleanup();
      pending.reject(new Error(event.message));
    }
    this.pending.clear();
  }
}
