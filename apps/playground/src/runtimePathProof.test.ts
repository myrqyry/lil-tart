import { describe, expect, it } from 'vitest'
import type { LiteRtTelemetryRecord } from '@litert-playground/runtime-litert'
import { createRuntimePathProof } from './runtimePathProof'

const adapter = {
  modelId: 'tiny-model',
  metadata: {
    name: 'Tiny Model',
    description: 'test model',
    modelPath: 'models/tiny.tflite',
    tags: ['test'],
  },
  verification: {
    status: 'compile-verified' as const,
    backends: ['webgpu' as const],
    evidence: 'docs/verification/tiny.md',
  },
}

function event(overrides: Partial<LiteRtTelemetryRecord> = {}): LiteRtTelemetryRecord {
  return {
    event: 'inference',
    timestamp: '2026-09-17T20:00:00.000Z',
    modelPath: 'models/tiny.tflite',
    requestedBackend: 'auto',
    resolvedBackend: 'webgpu',
    compileDurationMs: 12,
    fallbackCount: 0,
    tensorCopyCount: 0,
    inferenceDurationMs: 34,
    outputCount: 1,
    ...overrides,
  }
}

describe('createRuntimePathProof', () => {
  it('refuses to manufacture proof without inference telemetry from this run', () => {
    const proof = createRuntimePathProof({
      adapter,
      requestedBackend: 'auto',
      modelInfo: null,
      telemetry: [event({ event: 'compile', inferenceDurationMs: undefined })],
      telemetryStart: 1,
      outputCount: 1,
      capturedAt: '2026-09-17T20:01:00.000Z',
    })

    expect(proof).toBeNull()
  })

  it('captures the resolved runtime path without promoting durable verification', () => {
    const proof = createRuntimePathProof({
      adapter,
      requestedBackend: 'auto',
      modelInfo: {
        modelPath: 'models/tiny.tflite',
        requestedBackend: 'auto',
        resolvedBackend: 'webgpu',
        compileDurationMs: 12,
        fallbackCount: 0,
      },
      telemetry: [
        event({ event: 'compile', inferenceDurationMs: undefined }),
        event(),
      ],
      telemetryStart: 1,
      outputCount: 2,
      capturedAt: '2026-09-17T20:01:00.000Z',
    })

    expect(proof).toMatchObject({
      modelId: 'tiny-model',
      requestedBackend: 'auto',
      resolvedBackend: 'webgpu',
      compileDurationMs: 12,
      fallbackCount: 0,
      outputCount: 2,
      durableVerification: {
        status: 'compile-verified',
        backends: ['webgpu'],
      },
    })
    expect(proof?.inferenceEvents).toHaveLength(1)
  })

  it('preserves every graph event instead of flattening a multi-graph run', () => {
    const proof = createRuntimePathProof({
      adapter,
      requestedBackend: 'auto',
      modelInfo: null,
      telemetry: [
        event({ modelPath: 'models/encoder.tflite', resolvedBackend: 'webgpu', inferenceDurationMs: 8 }),
        event({ modelPath: 'models/decoder.tflite', resolvedBackend: 'wasm', inferenceDurationMs: 15, fallbackCount: 1 }),
      ],
      telemetryStart: 0,
      outputCount: 1,
      capturedAt: '2026-09-17T20:01:00.000Z',
    })

    expect(proof?.inferenceEvents.map((entry) => [entry.modelPath, entry.resolvedBackend])).toEqual([
      ['models/encoder.tflite', 'webgpu'],
      ['models/decoder.tflite', 'wasm'],
    ])
  })
})
