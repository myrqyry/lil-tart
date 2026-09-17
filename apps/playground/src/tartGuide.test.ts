import { describe, expect, it } from 'vitest'
import { getTartGuideMessage, type TartGuideSnapshot } from './tartGuide'

function snapshot(overrides: Partial<TartGuideSnapshot> = {}): TartGuideSnapshot {
  return {
    selectedModelName: null,
    loading: false,
    loaded: false,
    progressPercent: null,
    error: null,
    requestedBackend: 'auto',
    resolvedBackend: null,
    fallbackCount: 0,
    preflightComplete: false,
    inferenceComplete: false,
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
      loading: true,
      progressPercent: 42,
    }))

    expect(message.tone).toBe('working')
    expect(message.title).toContain('Tiny Model')
    expect(message.message).toContain('42%')
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
      inferenceComplete: true,
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
      inferenceComplete: true,
    }))

    expect(message.tone).toBe('success')
    expect(message.kicker).toBe('Inference complete')
    expect(message.message).toContain('working path')
  })

  it('gives runtime errors top priority', () => {
    const message = getTartGuideMessage(snapshot({
      selectedModelName: 'Tiny Model',
      loading: true,
      error: 'Graph compile failed',
    }))

    expect(message.tone).toBe('error')
    expect(message.message).toBe('Graph compile failed')
  })
})
