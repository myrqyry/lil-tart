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

interface CanonicalResult {
  ok: boolean;
  value?: unknown;
}

function canonicalValue(value: unknown): CanonicalResult {
  if (value === undefined) return { ok: true, value: undefined };
  if (value === null) return { ok: true, value: null };

  switch (typeof value) {
    case 'string':
    case 'number':
    case 'boolean':
      return { ok: true, value };
    case 'object': {
      if (Array.isArray(value)) {
        const items: unknown[] = [];
        for (const item of value) {
          const normalized = canonicalValue(item);
          if (!normalized.ok) return { ok: false };
          items.push(normalized.value);
        }
        return { ok: true, value: items };
      }

      const entries: Record<string, unknown> = {};
      for (const key of Object.keys(value as Record<string, unknown>).sort()) {
        const normalized = canonicalValue((value as Record<string, unknown>)[key]);
        if (!normalized.ok) return { ok: false };
        if (normalized.value !== undefined) entries[key] = normalized.value;
      }
      return { ok: true, value: entries };
    }
    default:
      // Exotic values are not part of the typed public contract. Under an
      // untyped JS caller, degrade to "cannot prove identical" rather than
      // throwing a validation error only when a second load happens to race.
      return { ok: false };
  }
}

function loadOptionsKey(options?: LiteRtLmWorkerLoadOptions): string | null {
  const engine = options?.engine ?? 'litert-lm';
  const semanticOptions: Record<string, unknown> = {
    ...(options as Record<string, unknown> | undefined),
    engine,
  };

  // loadKey identifies model bytes and is handled separately.
  delete semanticOptions.loadKey;

  // The worker ignores MediaPipe options unless that engine is selected.
  if (engine !== 'mediapipe') {
    delete semanticOptions.mediaPipe;
  } else if (options?.mediaPipe) {
    const hasMeaningfulMediaPipeOption = Object.values(options.mediaPipe)
      .some((value) => value !== undefined);
    if (!hasMeaningfulMediaPipeOption) delete semanticOptions.mediaPipe;
  }

  // Canonicalization drops undefined object values recursively, so omitted
  // defaults and explicit undefined describe the same worker behavior while
  // future public fields automatically participate in identity.
  const canonical = canonicalValue(semanticOptions);
  return canonical.ok ? JSON.stringify(canonical.value) : null;
}

function sameModelIdentity(
  left: ActiveLoadRequest,
  model: string | Blob,
  options?: LiteRtLmWorkerLoadOptions,
): boolean {
  if (left.model === model) return true;

  if (left.model instanceof Blob && model instanceof Blob) {
    const leftKey = left.options?.loadKey?.trim();
    const rightKey = options?.loadKey?.trim();
    return Boolean(leftKey && rightKey && leftKey === rightKey);
  }

  return false;
}

function sameLoadRequest(
  left: ActiveLoadRequest,
  model: string | Blob,
  options?: LiteRtLmWorkerLoadOptions,
): boolean {
  if (!sameModelIdentity(left, model, options)) return false;

  const leftKey = loadOptionsKey(left.options);
  const rightKey = loadOptionsKey(options);
  return leftKey !== null && rightKey !== null && leftKey === rightKey;
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
  private terminationTimer: ReturnType<typeof setTimeout> | null = null;
  private terminated = false;

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
    // Give the worker a bounded opportunity to cancel/delete native resources
    // before termination. The public API stays synchronous; local callers are
    // rejected immediately while teardown completes in the worker.
    this.terminationTimer = setTimeout(() => this.finishTermination(), 1_000);

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
      case 'cancelled':
        // The raw worker protocol exposes a terminal cancellation event for
        // direct consumers. This client settles AbortSignal cancellation
        // immediately in generate(), so the worker acknowledgement is only
        // informational here.
        break;
      case 'disposed':
        this.finishTermination();
        break;
    }
  }

  private finishTermination(): void {
    if (this.terminated) return;
    this.terminated = true;
    if (this.terminationTimer) {
      clearTimeout(this.terminationTimer);
      this.terminationTimer = null;
    }
    this.worker.terminate();
    this.worker.onmessage = null;
    this.worker.onerror = null;
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
    if (this.disposed) this.finishTermination();
  }
}
