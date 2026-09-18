import { useCallback, useEffect, useRef, useState } from 'react'
import {
  createLiteRtRuntime,
  type BackendPreference,
  type LiteRtModelInfo,
  type LiteRtPreflightResult,
  type LiteRtTelemetryRecord,
  type ManagedLiteRtRuntimeContext,
} from '@litert-playground/runtime-litert'
import type { Tensor } from '@litertjs/core'
import type { InferenceContext, ModelAdapter, TensorSpec } from '../adapters/types'
import { createModelLibraryAssetResolver, registerModelAssets } from '../modelStorage'

export type Accelerator = BackendPreference

export interface RawTensor {
  data: Float32Array
  shape: number[]
}

interface UseModelRunnerReturn {
  loadModel: (adapter: ModelAdapter, accelerator?: Accelerator) => Promise<void>
  unloadModel: () => Promise<void>
  runInference: (values: Record<string, unknown>) => Promise<void>
  preflightModel: () => Promise<void>
  outputs: Record<string, unknown> | null
  outputTensors: Record<string, RawTensor> | null
  outputSpecs: TensorSpec[]
  accelerator: Accelerator
  setAccelerator: (accelerator: Accelerator) => void
  resolvedAccelerator: string | null
  modelInfo: LiteRtModelInfo | null
  preflight: LiteRtPreflightResult | null
  telemetry: readonly LiteRtTelemetryRecord[]
  error: string | null
  loading: boolean
  loaded: boolean
  downloadProgress: { loadedBytes: number; totalBytes?: number } | null
  modelBase: string
  setModelBase: (base: string) => void
}

function pageBase(): string {
  if (typeof document !== 'undefined' && document.baseURI) return document.baseURI
  if (typeof location !== 'undefined') return location.href
  return 'http://localhost/'
}

function normalizeOutputs(
  result: Tensor[] | Record<string, Tensor>,
  specs: TensorSpec[],
): Record<string, Tensor> {
  if (!Array.isArray(result)) return result
  return Object.fromEntries(
    result
      .map((tensor, index) => [specs[index]?.name ?? `output_${index}`, tensor] as const),
  )
}

function typedInput(value: unknown): Float32Array | Int32Array | Int8Array | Uint8Array | null {
  if (value instanceof Float32Array) return value
  if (value instanceof Int32Array) return value
  if (value instanceof Int8Array) return value
  if (value instanceof Uint8Array) return value
  return null
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

export function useModelRunner(): UseModelRunnerReturn {
  const [outputs, setOutputs] = useState<Record<string, unknown> | null>(null)
  const [outputTensors, setOutputTensors] = useState<Record<string, RawTensor> | null>(null)
  const [outputSpecs, setOutputSpecs] = useState<TensorSpec[]>([])
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [loaded, setLoaded] = useState(false)
  const [accelerator, setAccelerator] = useState<Accelerator>('auto')
  const [resolvedAccelerator, setResolvedAccelerator] = useState<string | null>(null)
  const [modelInfo, setModelInfo] = useState<LiteRtModelInfo | null>(null)
  const [preflight, setPreflight] = useState<LiteRtPreflightResult | null>(null)
  const [telemetry, setTelemetry] = useState<readonly LiteRtTelemetryRecord[]>([])
  const [downloadProgress, setDownloadProgress] = useState<{ loadedBytes: number; totalBytes?: number } | null>(null)
  const [modelBase, setModelBase] = useState(pageBase())

  const runtimePromiseRef = useRef<Promise<ManagedLiteRtRuntimeContext> | null>(null)
  const adapterRef = useRef<ModelAdapter | null>(null)
  const loadControllerRef = useRef<AbortController | null>(null)
  const requestIdRef = useRef(0)

  const ensureRuntime = useCallback(() => {
    if (!runtimePromiseRef.current) {
      runtimePromiseRef.current = createLiteRtRuntime({
        backend: 'auto',
        assets: createModelLibraryAssetResolver(modelBase),
        telemetryLimit: 256,
      }).catch((cause) => {
        runtimePromiseRef.current = null
        throw cause
      })
    }
    return runtimePromiseRef.current
  }, [modelBase])

  const refreshRuntimeState = useCallback((runtime: ManagedLiteRtRuntimeContext, adapter: ModelAdapter, target: Accelerator) => {
    const info = runtime.liteRt.getModelInfo(adapter.metadata.modelPath, { accelerator: target }) ?? null
    setModelInfo(info)
    setResolvedAccelerator(info?.resolvedBackend ?? runtime.backend)
    setTelemetry(runtime.liteRt.getTelemetry())
  }, [])

  const loadModel = useCallback(async (adapter: ModelAdapter, acc?: Accelerator) => {
    const target = acc ?? accelerator
    const requestId = ++requestIdRef.current
    loadControllerRef.current?.abort()
    const controller = new AbortController()
    loadControllerRef.current = controller

    setLoading(true)
    setError(null)
    setLoaded(false)
    setPreflight(null)
    setOutputs(null)
    setOutputTensors(null)
    setDownloadProgress(null)

    try {
      const graphPaths = [adapter.metadata.modelPath, ...(adapter.graphs ?? []).map((graph) => graph.modelPath)]
      registerModelAssets(adapter.modelId, graphPaths)
      const runtime = await ensureRuntime()
      const previous = adapterRef.current
      if (previous) {
        runtime.liteRt.disposeModel(previous.metadata.modelPath)
        previous.graphs?.forEach((graph) => runtime.liteRt.disposeModel(graph.modelPath))
      }

      for (const modelPath of graphPaths) {
        await runtime.liteRt.loadModel(modelPath, {
          accelerator: target,
          signal: controller.signal,
          webNNOptions: target === 'webnn' || target === 'auto'
            ? { devicePreference: 'npu', powerPreference: 'high-performance' }
            : undefined,
          onProgress: (progress) => setDownloadProgress(progress),
        })
      }

      if (requestId !== requestIdRef.current || controller.signal.aborted) return
      adapterRef.current = adapter
      setAccelerator(target)
      setLoaded(true)
      refreshRuntimeState(runtime, adapter, target)
    } catch (cause) {
      if (requestId !== requestIdRef.current) return
      if (controller.signal.aborted) return
      adapterRef.current = null
      setModelInfo(null)
      setResolvedAccelerator(null)
      setError(errorMessage(cause))
    } finally {
      if (requestId === requestIdRef.current) setLoading(false)
    }
  }, [accelerator, ensureRuntime, refreshRuntimeState])

  const unloadModel = useCallback(async () => {
    requestIdRef.current += 1
    loadControllerRef.current?.abort()
    loadControllerRef.current = null

    const adapter = adapterRef.current
    adapterRef.current = null

    setLoading(false)
    setLoaded(false)
    setModelInfo(null)
    setResolvedAccelerator(null)
    setPreflight(null)
    setOutputs(null)
    setOutputTensors(null)
    setOutputSpecs([])
    setError(null)
    setDownloadProgress(null)

    const runtimePromise = runtimePromiseRef.current
    if (!runtimePromise || !adapter) return

    try {
      const runtime = await runtimePromise
      runtime.liteRt.disposeModel(adapter.metadata.modelPath)
      adapter.graphs?.forEach((graph) => runtime.liteRt.disposeModel(graph.modelPath))
      setTelemetry(runtime.liteRt.getTelemetry())
    } catch {
      // The runtime may have failed before a model became disposable.
    }
  }, [])

  const runInference = useCallback(async (values: Record<string, unknown>) => {
    const adapter = adapterRef.current
    if (!adapter) {
      setError('No model loaded')
      return
    }

    setLoading(true)
    setError(null)
    try {
      const runtime = await ensureRuntime()
      const webNNOptions = accelerator === 'webnn' || accelerator === 'auto'
        ? { devicePreference: 'npu' as const, powerPreference: 'high-performance' as const }
        : undefined

      let parsed: Record<string, unknown>
      if (adapter.run) {
        const ctx: InferenceContext = {
          predict: async (graph, inputs, signature) => {
            const graphPath = graph === 'main'
              ? adapter.metadata.modelPath
              : adapter.graphs?.find((entry) => entry.name === graph)?.modelPath
            if (!graphPath) throw new Error(`Unknown graph '${graph}' for ${adapter.modelId}`)
            const options = { accelerator, label: `playground:${adapter.modelId}:${graph}`, webNNOptions }
            const result = signature
              ? await runtime.liteRt.predictWithSignature(graphPath, signature, inputs, options)
              : await runtime.liteRt.predict(graphPath, inputs, options)
            return normalizeOutputs(result, [])
          },
          createTensor: (data, shape) => runtime.liteRt.createTensor(data, shape),
        }
        parsed = await adapter.run(values, ctx)
      } else {
        let inputs = adapter.prepareInputs(values)
        if (!Object.keys(inputs).length && Object.keys(values).length) {
          inputs = {}
          for (const spec of adapter.inputSpecs) {
            const data = typedInput(values[spec.name])
            if (data) inputs[spec.name] = runtime.liteRt.createTensor(data, spec.shape)
          }
        }

        const result = await runtime.liteRt.predict(adapter.metadata.modelPath, inputs, {
          accelerator,
          label: `playground:${adapter.modelId}`,
          webNNOptions,
        })
        const outputRecord = normalizeOutputs(result, adapter.outputSpecs)
        parsed = await adapter.parseOutputs(outputRecord)

        const raw: Record<string, RawTensor> = {}
        for (const spec of adapter.outputSpecs) {
          const tensor = outputRecord[spec.name]
          if (!tensor) continue
          const data = await tensor.data()
          raw[spec.name] = { data: new Float32Array(data), shape: spec.shape }
        }
        setOutputTensors(raw)
        setOutputSpecs(adapter.outputSpecs)

        Object.values(inputs).forEach((tensor) => {
          const disposable = tensor as Tensor & { delete?: () => void }
          disposable.delete?.()
        })
      }

      setOutputs(parsed)
      refreshRuntimeState(runtime, adapter, accelerator)
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setLoading(false)
    }
  }, [accelerator, ensureRuntime, refreshRuntimeState])

  const preflightModel = useCallback(async () => {
    const adapter = adapterRef.current
    if (!adapter) {
      setError('No model loaded')
      return
    }

    setLoading(true)
    setError(null)
    try {
      const runtime = await ensureRuntime()
      const result = await runtime.liteRt.preflight(adapter.metadata.modelPath, {
        accelerator,
        webNNOptions: accelerator === 'webnn' || accelerator === 'auto'
          ? { devicePreference: 'npu', powerPreference: 'high-performance' }
          : undefined,
      })
      setPreflight(result)
      refreshRuntimeState(runtime, adapter, accelerator)
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setLoading(false)
    }
  }, [accelerator, ensureRuntime, refreshRuntimeState])

  useEffect(() => {
    return () => {
      requestIdRef.current += 1
      loadControllerRef.current?.abort()
      const runtime = runtimePromiseRef.current
      runtimePromiseRef.current = null
      adapterRef.current = null
      setLoaded(false)
      setModelInfo(null)
      setPreflight(null)
      setOutputs(null)
      setOutputTensors(null)
      setError(null)
      setDownloadProgress(null)
      if (runtime) void runtime.then((context) => context.liteRt.dispose()).catch(() => undefined)
    }
  }, [modelBase])

  return {
    loadModel,
    unloadModel,
    runInference,
    preflightModel,
    outputs,
    outputTensors,
    outputSpecs,
    accelerator,
    setAccelerator,
    resolvedAccelerator,
    modelInfo,
    preflight,
    telemetry,
    error,
    loading,
    loaded,
    downloadProgress,
    modelBase,
    setModelBase,
  }
}
