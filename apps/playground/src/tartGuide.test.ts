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

  it('describes preflight without pretending the model is downloading', () => {
    const message = getTartGuideMessage(snapshot({
      selectedModelName: 'Tiny Model',
      operation: 'preflight',
      loaded: true,
      resolvedBackend: 'webgpu',
      progressPercent: 100,
    }))

    expect(message.tone).toBe('working')
    expect(message.title).toContain('preflight')
    expect(message.message).not.toContain('Fetching')
  })

  it('describes real inference without pretending the model is compiling', () => {
    const message = getTartGuideMessage(snapshot({
      selectedModelName: 'Tiny Model',
      operation: 'inference',
      loaded: true,
      resolvedBackend: 'webgpu',
      progressPercent: 100,
    }))

    expect(message.tone).toBe('working')
    expect(message.kicker).toBe('Real inference')
    expect(message.message).not.toContain('Fetching')
  })

  it('offers preflight after a clean load', () => {
    const message = getTartGuideMessage(snapshot({
      selectedModelName: 'Tiny Model',
      loaded: true,
      resolvedBackend: 'webgpu',
    }))

    expect(message.tone).toBe('success')
    expect(message.action).toBe('preflight')
    expect(message.message).toContain('WEBGPU')
  })

  it('surfaces backend fallback before celebrating success', () => {
    const message = getTartGuideMessage(snapshot({
      selectedModelName: 'Tiny Model',
      loaded: true,
      requestedBackend: 'webgpu',
      resolvedBackend: 'wasm',
      fallbackCount: 1,
      pathProofAvailable: true,
    }))

    expect(message.tone).toBe('warning')
    expect(message.message).toContain('WEBGPU')
    expect(message.message).toContain('WASM')
    expect(message.message).toContain('1 fallback')
  })

  it('treats completed inference as a proven local path', () => {
    const message = getTartGuideMessage(snapshot({
      selectedModelName: 'Tiny Model',
      loaded: true,
      resolvedBackend: 'webgpu',
      preflightComplete: true,
      pathProofAvailable: true,
    }))

    expect(message.tone).toBe('success')
    expect(message.kicker).toBe('Inference complete')
    expect(message.message).toContain('session path')
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
