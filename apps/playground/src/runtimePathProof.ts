import type { LiteRtModelInfo, LiteRtTelemetryRecord } from '@litert-playground/runtime-litert'
import type { ModelAdapter, ModelVerification } from './adapters/types'

export type RuntimeOperation = 'idle' | 'model-load' | 'preflight' | 'inference'

export interface RuntimePathInferenceEvent {
  graph: string
  modelPath: string
  requestedBackend: string
  resolvedBackend: string
  inferenceDurationMs: number
  outputCount?: number
  fallbackCount: number
  timestamp: string
}

export interface RuntimePathGraphBackend {
  graph: string
  modelPath: string
  requestedBackend: string
  resolvedBackend: string
  fallbackCount: number
}

export interface RuntimePathProof {
  modelId: string
  modelName: string
  modelPath: string
  selectedBackend: string
  mainGraphRequestedBackend: string
  mainGraphResolvedBackend: string
  compileDurationMs: number
  fallbackCount: number
  outputCount: number
  graphBackends: readonly RuntimePathGraphBackend[]
  inferenceEvents: readonly RuntimePathInferenceEvent[]
  durableVerification: ModelVerification | null
  capturedAt: string
}

interface CreateRuntimePathProofOptions {
  adapter: Pick<ModelAdapter, 'modelId' | 'metadata' | 'verification' | 'graphs'>
  selectedBackend: string
  modelInfo: LiteRtModelInfo | null
  telemetry: readonly LiteRtTelemetryRecord[]
  telemetryStart: number
  outputCount: number
  capturedAt?: string
}

function copyVerification(verification: ModelVerification | undefined): ModelVerification | null {
  if (!verification) return null
  return {
    ...verification,
    backends: verification.backends ? [...verification.backends] : undefined,
  }
}

function graphName(adapter: CreateRuntimePathProofOptions['adapter'], modelPath: string): string {
  if (modelPath === adapter.metadata.modelPath) return 'main'
  return adapter.graphs?.find((graph) => graph.modelPath === modelPath)?.name ?? modelPath
}

/**
 * Build a session proof only from inference telemetry emitted by this run.
 *
 * This is deliberately separate from durable adapter verification. A successful
 * browser session can produce a recipe input without silently promoting the
 * adapter's persisted evidence level.
 */
export function createRuntimePathProof(options: CreateRuntimePathProofOptions): RuntimePathProof | null {
  const inferenceEvents = options.telemetry
    .slice(options.telemetryStart)
    .filter((entry) => entry.event === 'inference' && entry.inferenceDurationMs !== undefined)
    .map((entry): RuntimePathInferenceEvent => ({
      graph: graphName(options.adapter, entry.modelPath),
      modelPath: entry.modelPath,
      requestedBackend: entry.requestedBackend,
      resolvedBackend: entry.resolvedBackend,
      inferenceDurationMs: entry.inferenceDurationMs!,
      outputCount: entry.outputCount,
      fallbackCount: entry.fallbackCount,
      timestamp: entry.timestamp,
    }))

  if (inferenceEvents.length === 0) return null

  const graphBackendMap = new Map<string, RuntimePathGraphBackend>()
  for (const event of inferenceEvents) {
    graphBackendMap.set(event.graph, {
      graph: event.graph,
      modelPath: event.modelPath,
      requestedBackend: event.requestedBackend,
      resolvedBackend: event.resolvedBackend,
      fallbackCount: event.fallbackCount,
    })
  }
  const graphBackends = [...graphBackendMap.values()]
  const fallbackCount = graphBackends.reduce((total, graph) => total + graph.fallbackCount, 0)

  const mainEvent = [...inferenceEvents]
    .reverse()
    .find((entry) => entry.modelPath === options.adapter.metadata.modelPath)
  return {
    modelId: options.adapter.modelId,
    modelName: options.adapter.metadata.name,
    modelPath: options.adapter.metadata.modelPath,
    selectedBackend: options.selectedBackend,
    mainGraphRequestedBackend: options.modelInfo?.requestedBackend ?? mainEvent?.requestedBackend ?? 'unknown',
    mainGraphResolvedBackend: options.modelInfo?.resolvedBackend ?? mainEvent?.resolvedBackend ?? 'unknown',
    compileDurationMs: options.modelInfo?.compileDurationMs ?? 0,
    fallbackCount,
    outputCount: options.outputCount,
    graphBackends,
    inferenceEvents,
    durableVerification: copyVerification(options.adapter.verification),
    capturedAt: options.capturedAt ?? new Date().toISOString(),
  }
}
