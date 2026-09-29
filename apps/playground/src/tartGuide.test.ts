import { describe, expect, it } from 'vitest'
import { getTartGuideMessage, type TartGuideSnapshot } from './tartGuide'

function snapshot(overrides: Partial<TartGuideSnapshot> = {}): TartGuideSnapshot {
  return {
    selectedModelName: null,
    operation: 'idle',
    loaded: false,
    progressPercent: null,
    error: null,
    requestedBackend: 'auto',
    resolvedBackend: null,
    backendOverrides: [],
    runtimeGraphBackends: [],
    fallbackCount: 0,
    preflightComplete: false,
    pathProofAvailable: false,
    ...overrides,
  }
}

describe('getTartGuideMessage', () => {
  it('starts as a model-selection guide', () => {
    const message = getTartGuideMessage(snapshot())

    expect(message.tone).toBe('idle')
    expect(message.title).toContain('Pick')
  })

  it('reports model download progress', () => {
    const message = getTartGuideMessage(snapshot({
      selectedModelName: 'Tiny Model',
      operation: 'model-load',
      progressPercent: 42,
    }))

    expect(message.tone).toBe('working')
    expect(message.title).toContain('Tiny Model')
    expect(message.message).toContain('42%')
  })

  it('describes preflight as a main-graph check', () => {
    const message = getTartGuideMessage(snapshot({
      selectedModelName: 'Tiny Model',
      operation: 'preflight',
      loaded: true,
      resolvedBackend: 'webgpu',
      progressPercent: 100,
    }))

    expect(message.tone).toBe('working')
    expect(message.title).toContain('preflight')
    expect(message.message).toContain('main graph')
    expect(message.message).not.toContain('Fetching')
  })

  it('describes constrained multi-graph inference without pretending there is one execution backend', () => {
    const message = getTartGuideMessage(snapshot({
      selectedModelName: 'Tiny Pipeline',
      operation: 'inference',
      loaded: true,
      backendOverrides: [
        { graph: 'enc_tx', backend: 'wasm' },
        { graph: 'dec_tx', backend: 'wasm' },
      ],
    }))

    expect(message.tone).toBe('working')
    expect(message.kicker).toBe('Real inference')
    expect(message.message).toContain('enc_tx → WASM')
    expect(message.message).toContain('dec_tx → WASM')
    expect(message.message).toContain('each graph')
  })

  it('offers preflight after a clean load', () => {
    const message = getTartGuideMessage(snapshot({
      selectedModelName: 'Tiny Model',
      loaded: true,
      resolvedBackend: 'webgpu',
    }))

    expect(message.tone).toBe('success')
    expect(message.action).toBe('preflight')
    expect(message.message).toContain('main graph')
    expect(message.message).toContain('WEBGPU')
  })

  it('surfaces explicit graph correctness overrides without calling them fallbacks', () => {
    const message = getTartGuideMessage(snapshot({
      selectedModelName: 'Tiny Pipeline',
      loaded: true,
      requestedBackend: 'webgpu',
      backendOverrides: [
        { graph: 'enc_tx', backend: 'wasm' },
        { graph: 'dec_tx', backend: 'wasm' },
      ],
      fallbackCount: 0,
    }))

    expect(message.tone).toBe('warning')
    expect(message.kicker).toBe('Correctness override')
    expect(message.message).toContain('WEBGPU')
    expect(message.message).toContain('enc_tx → WASM')
    expect(message.message).toContain('dec_tx → WASM')
    expect(message.message).toContain('not runtime fallbacks')
  })

  it('does not claim per-graph preservation before a session proof exists', () => {
    const message = getTartGuideMessage(snapshot({
      selectedModelName: 'Tiny Pipeline',
      loaded: true,
      requestedBackend: 'webgpu',
      backendOverrides: [{ graph: 'decoder', backend: 'wasm' }],
      fallbackCount: 1,
      pathProofAvailable: false,
    }))

    expect(message.tone).toBe('warning')
    expect(message.message).toContain('main graph')
    expect(message.message).toContain('per-graph split will not exist')
    expect(message.message).not.toContain('preserved per graph')
  })

  it('reports both correctness overrides and runtime fallbacks when both occurred', () => {
    const message = getTartGuideMessage(snapshot({
      selectedModelName: 'Tiny Pipeline',
      loaded: true,
      requestedBackend: 'webgpu',
      backendOverrides: [{ graph: 'decoder', backend: 'wasm' }],
      runtimeGraphBackends: [
        {
          graph: 'main',
          modelPath: 'main.tflite',
          requestedBackend: 'webgpu',
          resolvedBackend: 'webgpu',
          fallbackCount: 0,
        },
        {
          graph: 'decoder',
          modelPath: 'decoder.tflite',
          requestedBackend: 'wasm',
          resolvedBackend: 'webgpu',
          fallbackCount: 1,
        },
      ],
      fallbackCount: 1,
      pathProofAvailable: true,
    }))

    expect(message.tone).toBe('warning')
    expect(message.kicker).toBe('Correctness override')
    expect(message.message).toContain('decoder → WASM')
    expect(message.message).toContain('1 fallback')
    expect(message.message).toContain('preserved per graph')
  })

  it('surfaces runtime fallback without flattening a multi-graph proof to one resolved backend', () => {
    const message = getTartGuideMessage(snapshot({
      selectedModelName: 'Tiny Pipeline',
      loaded: true,
      requestedBackend: 'webgpu',
      resolvedBackend: 'webgpu',
      runtimeGraphBackends: [
        {
          graph: 'main',
          modelPath: 'main.tflite',
          requestedBackend: 'webgpu',
          resolvedBackend: 'webgpu',
          fallbackCount: 0,
        },
        {
          graph: 'decoder',
          modelPath: 'decoder.tflite',
          requestedBackend: 'webgpu',
          resolvedBackend: 'wasm',
          fallbackCount: 1,
        },
      ],
      fallbackCount: 1,
      pathProofAvailable: true,
    }))

    expect(message.tone).toBe('warning')
    expect(message.message).toContain('2 graph paths')
    expect(message.message).toContain('1 fallback')
    expect(message.message).toContain('each graph')
    expect(message.message).not.toContain('landed on WEBGPU')
  })

  it('treats completed multi-graph inference as a proven per-graph local path', () => {
    const message = getTartGuideMessage(snapshot({
      selectedModelName: 'Tiny Pipeline',
      loaded: true,
      runtimeGraphBackends: [
        {
          graph: 'main',
          modelPath: 'main.tflite',
          requestedBackend: 'webgpu',
          resolvedBackend: 'webgpu',
          fallbackCount: 0,
        },
        {
          graph: 'decoder',
          modelPath: 'decoder.tflite',
          requestedBackend: 'wasm',
          resolvedBackend: 'wasm',
          fallbackCount: 0,
        },
      ],
      preflightComplete: true,
      pathProofAvailable: true,
    }))

    expect(message.tone).toBe('success')
    expect(message.kicker).toBe('Inference complete')
    expect(message.message).toContain('2 graph paths')
    expect(message.message).toContain('every inference graph')
  })

  it('gives runtime errors top priority', () => {
    const message = getTartGuideMessage(snapshot({
      selectedModelName: 'Tiny Model',
      operation: 'model-load',
      error: 'Graph compile failed',
    }))

    expect(message.tone).toBe('error')
    expect(message.message).toBe('Graph compile failed')
  })
})
