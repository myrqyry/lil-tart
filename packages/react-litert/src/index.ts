import { useCallback, useEffect, useMemo, useSyncExternalStore } from 'react'
import type {
  LiteRtModelInfo,
  LiteRtModelInput,
  LiteRtModelOptions,
  LiteRtModelOutput,
  ManagedLiteRtRuntime,
} from '@litert-playground/runtime-litert'

export type ManagedModelStatus = 'idle' | 'loading' | 'ready' | 'error'

export interface ManagedModelSnapshot {
  status: ManagedModelStatus
  info: LiteRtModelInfo | null
  error: Error | null
}

export interface ManagedModelOptions {
  /** Existing shared runtime; ownership stays with the calling application. */
  runtime: ManagedLiteRtRuntime | null
  modelPath: string | null
  accelerator?: LiteRtModelOptions['accelerator']
}

export interface RunModelOptions {
  /** Named signature for multi-signature LiteRT models. */
  signature?: string
  /** Caller retains ownership of input tensors and successful output tensors. */
  label?: string
  signal?: AbortSignal
}

const EMPTY: ManagedModelSnapshot = { status: 'idle', info: null, error: null }

export class ModelRequestCancelledError extends Error {
  constructor() {
    super('The model request was cancelled or superseded')
    this.name = 'AbortError'
  }
}

function asError(cause: unknown): Error {
  return cause instanceof Error ? cause : new Error(String(cause))
}

function discardOutput(output: LiteRtModelOutput): void {
  const tensors = Array.isArray(output) ? output : Object.values(output)
  for (const tensor of new Set(tensors)) {
    try { tensor.delete() } catch { /* completed cancellation must release what it can */ }
  }
}

/**
 * Lifecycle owner for one React model selection, not for the shared inference
 * runtime. No implicit downloads, compilation or runtime disposal.
 *
 * A new controller is created when the model path, runtime or accelerator
 * changes; abort only our own subscribers so other consumers sharing the
 * runtime can continue loading the same model.
 */
export class ManagedLiteRtModelController {
  private snapshot: ManagedModelSnapshot = EMPTY
  private readonly listeners = new Set<() => void>()
  private pending: {
    controller: AbortController
    ticket: number
    promise: Promise<LiteRtModelInfo | null>
  } | null = null
  private readonly activeRuns = new Set<AbortController>()
  private ticket = 0
  private disposed = false
  private mounts = 0

  constructor(private readonly options: ManagedModelOptions) {}

  /**
   * React 19 Strict Mode may run effect setup/cleanup/setup again during mount.
   * Release aborts current requests immediately, but defers permanent disposal
   * until after a same-controller remount has had a chance to retain it.
   */
  readonly retain = (): void => {
    if (this.disposed) throw new ModelRequestCancelledError()
    this.mounts += 1
  }

  readonly release = (): void => {
    if (this.mounts === 0) return
    this.mounts -= 1
    if (this.mounts > 0) return
    this.ticket += 1
    this.pending?.controller.abort()
    this.pending = null
    for (const controller of this.activeRuns) controller.abort()
    queueMicrotask(() => {
      if (this.mounts === 0) this.dispose()
    })
  }

  readonly subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => { this.listeners.delete(listener) }
  }

  readonly getSnapshot = (): ManagedModelSnapshot => this.snapshot

  private update(snapshot: ManagedModelSnapshot): void {
    if (this.disposed) return
    this.snapshot = snapshot
    for (const listener of [...this.listeners]) listener()
  }

  readonly load = async (): Promise<LiteRtModelInfo | null> => {
    if (this.disposed) throw new ModelRequestCancelledError()
    const { runtime, modelPath, accelerator } = this.options
    if (!runtime || !modelPath) throw new Error('Choose a runtime and model path before loading')
    if (this.snapshot.status === 'ready') return this.snapshot.info
    if (this.pending) return this.pending.promise

    const controller = new AbortController()
    const ticket = ++this.ticket
    this.update({ status: 'loading', info: null, error: null })

    const promise = (async () => {
      try {
        await runtime.loadModel(modelPath, { accelerator, signal: controller.signal })
        if (this.disposed || controller.signal.aborted || ticket !== this.ticket) {
          throw new ModelRequestCancelledError()
        }
        const info = runtime.getModelInfo(modelPath, { accelerator }) ?? null
        this.update({ status: 'ready', info, error: null })
        return info
      } catch (cause) {
        if (!this.disposed && ticket === this.ticket) {
          this.update(controller.signal.aborted
            ? EMPTY
            : { status: 'error', info: null, error: asError(cause) })
        }
        throw cause
      } finally {
        if (this.pending?.ticket === ticket) this.pending = null
      }
    })()
    this.pending = { controller, ticket, promise }
    return promise
  }

  readonly run = async (
    input: LiteRtModelInput,
    { signature, label, signal }: RunModelOptions = {},
  ): Promise<LiteRtModelOutput> => {
    if (this.disposed) throw new ModelRequestCancelledError()
    if (this.snapshot.status !== 'ready') throw new Error('Load the model before inference')
    const { runtime, modelPath, accelerator } = this.options
    if (!runtime || !modelPath) throw new Error('Model runtime is unavailable')
    if (signal?.aborted) throw new ModelRequestCancelledError()
    const ticket = this.ticket
    const controller = new AbortController()
    const onAbort = () => controller.abort()
    signal?.addEventListener('abort', onAbort, { once: true })
    if (signal?.aborted) controller.abort()
    this.activeRuns.add(controller)
    try {
      const options = { accelerator, label, signal: controller.signal }
      const output = signature
        ? await runtime.predictWithSignature(modelPath, signature, input, options)
        : await runtime.predict(modelPath, input, options)
      // An already resolved native/browser inference may race an AbortSignal.
      // Never hand its orphaned tensors back to a disposed/superseded component.
      if (this.disposed || ticket !== this.ticket || controller.signal.aborted) {
        discardOutput(output)
        throw new ModelRequestCancelledError()
      }
      return output
    } finally {
      signal?.removeEventListener('abort', onAbort)
      this.activeRuns.delete(controller)
    }
  }

  /** Cancel only this adapter's outstanding requests; never dispose a shared runtime. */
  readonly cancel = (): void => {
    this.ticket += 1
    this.pending?.controller.abort()
    this.pending = null
    for (const controller of this.activeRuns) controller.abort()
    this.update(EMPTY)
  }

  readonly dispose = (): void => {
    if (this.disposed) return
    // Unmount cleanup must never fire a new external-store notification.
    this.disposed = true
    this.ticket += 1
    this.pending?.controller.abort()
    this.pending = null
    for (const controller of this.activeRuns) controller.abort()
    this.listeners.clear()
  }
}

/** Thin React adapter over the existing managed LiteRT runtime. */
export function useManagedLiteRtModel(options: ManagedModelOptions & { autoLoad?: boolean }) {
  const { runtime, modelPath, accelerator, autoLoad = false } = options
  const controller = useMemo(
    () => new ManagedLiteRtModelController({ runtime, modelPath, accelerator }),
    [runtime, modelPath, accelerator],
  )
  useEffect(() => {
    controller.retain()
    return () => controller.release()
  }, [controller])
  useEffect(() => {
    if (autoLoad) void controller.load().catch(() => {
      // The controller publishes real failures to the snapshot; do not produce
      // an unhandled rejection for an abandoned auto-load.
    })
  }, [autoLoad, controller])
  const snapshot = useSyncExternalStore(
    controller.subscribe,
    controller.getSnapshot,
    controller.getSnapshot,
  )
  const load = useCallback(() => controller.load(), [controller])
  const run = useCallback(
    (input: LiteRtModelInput, runOptions?: RunModelOptions) => controller.run(input, runOptions),
    [controller],
  )
  const cancel = useCallback(() => controller.cancel(), [controller])
  return { ...snapshot, load, run, cancel }
}
