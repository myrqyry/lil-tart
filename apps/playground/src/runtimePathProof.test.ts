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
  graphs: [
    { name: 'encoder', modelPath: 'models/encoder.tflite' },
    { name: 'decoder', modelPath: 'models/decoder.tflite' },
  ],
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
      selectedBackend: 'auto',
      modelInfo: null,
      telemetry: [event({ event: 'compile', inferenceDurationMs: undefined })],
      telemetryStart: 1,
      outputCount: 1,
      capturedAt: '2026-09-17T20:01:00.000Z',
    })

    expect(proof).toBeNull()
  })

  it('captures the main graph path without promoting durable verification', () => {
    const proof = createRuntimePathProof({
      adapter,
      selectedBackend: 'auto',
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
      selectedBackend: 'auto',
      mainGraphRequestedBackend: 'auto',
      mainGraphResolvedBackend: 'webgpu',
      compileDurationMs: 12,
      fallbackCount: 0,
      outputCount: 2,
      durableVerification: {
        status: 'compile-verified',
        backends: ['webgpu'],
      },
    })
    expect(proof?.graphBackends).toEqual([
      {
        graph: 'main',
        modelPath: 'models/tiny.tflite',
        requestedBackend: 'auto',
        resolvedBackend: 'webgpu',
        fallbackCount: 0,
      },
    ])
    expect(proof?.inferenceEvents).toHaveLength(1)
  })

  it('records a correctness override as selection → runtime request → resolution, not as fallback', () => {
    const proof = createRuntimePathProof({
      adapter,
      selectedBackend: 'webgpu',
      modelInfo: {
        modelPath: 'models/tiny.tflite',
        requestedBackend: 'wasm',
        resolvedBackend: 'wasm',
        compileDurationMs: 10,
        fallbackCount: 0,
      },
      telemetry: [
        event({ requestedBackend: 'wasm', resolvedBackend: 'wasm', fallbackCount: 0 }),
      ],
      telemetryStart: 0,
      outputCount: 1,
      capturedAt: '2026-09-17T20:01:00.000Z',
    })

    expect(proof).toMatchObject({
      selectedBackend: 'webgpu',
      mainGraphRequestedBackend: 'wasm',
      mainGraphResolvedBackend: 'wasm',
      fallbackCount: 0,
    })
    expect(proof?.graphBackends[0]).toMatchObject({
      graph: 'main',
      requestedBackend: 'wasm',
      resolvedBackend: 'wasm',
      fallbackCount: 0,
    })
  })

  it('does not borrow a subgraph backend for missing main-graph proof', () => {
    const proof = createRuntimePathProof({
      adapter,
      selectedBackend: 'webgpu',
      modelInfo: null,
      telemetry: [
        event({
          modelPath: 'models/encoder.tflite',
          requestedBackend: 'wasm',
          resolvedBackend: 'wasm',
          inferenceDurationMs: 10,
        }),
      ],
      telemetryStart: 0,
      outputCount: 1,
      capturedAt: '2026-09-17T20:01:00.000Z',
    })

    expect(proof).toMatchObject({
      mainGraphRequestedBackend: 'unknown',
      mainGraphResolvedBackend: 'unknown',
    })
    expect(proof?.graphBackends[0]).toMatchObject({
      graph: 'encoder',
      requestedBackend: 'wasm',
      resolvedBackend: 'wasm',
    })
  })

  it('preserves each graph path and aggregates one fallback count per graph', () => {
    const proof = createRuntimePathProof({
      adapter,
      selectedBackend: 'webgpu',
      modelInfo: null,
      telemetry: [
        event({
          modelPath: 'models/tiny.tflite',
          requestedBackend: 'webgpu',
          resolvedBackend: 'webgpu',
          inferenceDurationMs: 8,
        }),
        event({
          modelPath: 'models/encoder.tflite',
          requestedBackend: 'wasm',
          resolvedBackend: 'wasm',
          inferenceDurationMs: 10,
          fallbackCount: 0,
        }),
        event({
          modelPath: 'models/decoder.tflite',
          requestedBackend: 'wasm',
          resolvedBackend: 'webgpu',
          inferenceDurationMs: 15,
          fallbackCount: 1,
        }),
        // A repeated decoder invocation must not double-count its compile fallback.
        event({
          modelPath: 'models/decoder.tflite',
          requestedBackend: 'wasm',
          resolvedBackend: 'webgpu',
          inferenceDurationMs: 14,
          fallbackCount: 1,
        }),
      ],
      telemetryStart: 0,
      outputCount: 1,
      capturedAt: '2026-09-17T20:01:00.000Z',
    })

    expect(proof?.graphBackends).toEqual([
      {
        graph: 'main',
        modelPath: 'models/tiny.tflite',
        requestedBackend: 'webgpu',
        resolvedBackend: 'webgpu',
        fallbackCount: 0,
      },
      {
        graph: 'encoder',
        modelPath: 'models/encoder.tflite',
        requestedBackend: 'wasm',
        resolvedBackend: 'wasm',
        fallbackCount: 0,
      },
      {
        graph: 'decoder',
        modelPath: 'models/decoder.tflite',
        requestedBackend: 'wasm',
        resolvedBackend: 'webgpu',
        fallbackCount: 1,
      },
    ])
    expect(proof?.fallbackCount).toBe(1)
    expect(proof?.inferenceEvents.map((entry) => [
      entry.graph,
      entry.requestedBackend,
      entry.resolvedBackend,
    ])).toEqual([
      ['main', 'webgpu', 'webgpu'],
      ['encoder', 'wasm', 'wasm'],
      ['decoder', 'wasm', 'webgpu'],
      ['decoder', 'wasm', 'webgpu'],
    ])
  })
})
