import type { LiteRtModelInfo, LiteRtTelemetryRecord } from '@litert-playground/runtime-litert'
import type { ModelAdapter, ModelVerification } from './adapters/types'

export type RuntimeOperation = 'idle' | 'model-load' | 'preflight' | 'inference'

export interface RuntimePathInferenceEvent {
  modelPath: string
  resolvedBackend: string
  inferenceDurationMs: number
  outputCount?: number
  fallbackCount: number
  timestamp: string
}

export interface RuntimePathProof {
  modelId: string
  modelName: string
  modelPath: string
  requestedBackend: string
  resolvedBackend: string
  compileDurationMs: number
  fallbackCount: number
  outputCount: number
  inferenceEvents: readonly RuntimePathInferenceEvent[]
  durableVerification: ModelVerification | null
  capturedAt: string
}

interface CreateRuntimePathProofOptions {
  adapter: Pick<ModelAdapter, 'modelId' | 'metadata' | 'verification'>
  requestedBackend: string
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
      modelPath: entry.modelPath,
      resolvedBackend: entry.resolvedBackend,
      inferenceDurationMs: entry.inferenceDurationMs!,
      outputCount: entry.outputCount,
      fallbackCount: entry.fallbackCount,
      timestamp: entry.timestamp,
    }))

  if (inferenceEvents.length === 0) return null

  const mainEvent = [...inferenceEvents]
    .reverse()
    .find((entry) => entry.modelPath === options.adapter.metadata.modelPath)
  const receiptEvent = mainEvent ?? inferenceEvents[inferenceEvents.length - 1]!

  return {
    modelId: options.adapter.modelId,
    modelName: options.adapter.metadata.name,
    modelPath: options.adapter.metadata.modelPath,
    requestedBackend: options.requestedBackend,
    resolvedBackend: options.modelInfo?.resolvedBackend ?? receiptEvent.resolvedBackend,
    compileDurationMs: options.modelInfo?.compileDurationMs ?? 0,
    fallbackCount: options.modelInfo?.fallbackCount ?? receiptEvent.fallbackCount,
    outputCount: options.outputCount,
    inferenceEvents,
    durableVerification: copyVerification(options.adapter.verification),
    capturedAt: options.capturedAt ?? new Date().toISOString(),
  }
}
