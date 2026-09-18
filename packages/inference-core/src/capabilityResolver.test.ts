import { describe, expect, it } from 'vitest'
import {
  getCapabilityVerificationLevel,
  resolveCapabilityProvider,
  type Backend,
  type CapabilityProvider,
  type ModelManifest,
  type ModelVerification,
} from './index'

function verification(level: 'registered' | 'assets' | 'compile' | 'inference' | 'output'): ModelVerification | undefined {
  if (level === 'registered') return undefined
  return {
    assets: 'pass',
    compile: level === 'assets' ? 'untested' : 'pass',
    inference: level === 'assets' || level === 'compile' ? 'untested' : 'pass',
    output: level === 'output' ? 'pass' : 'untested',
  }
}

function provider(
  id: string,
  opts: {
    modelId?: string
    capabilities?: ModelManifest['capabilities']
    backends?: ModelManifest['backends']
    verification?: ModelVerification
    downloadBytes?: number
    residentBytes?: number
    priority?: number
  } = {},
): CapabilityProvider {
  return {
    id,
    priority: opts.priority,
    manifest: {
      modelId: opts.modelId ?? id,
      name: id,
      version: '1.0.0',
      capabilities: opts.capabilities ?? ['text-generation'],
      backends: opts.backends ?? { webgpu: true, wasm: true },
      memory: {
        downloadBytes: opts.downloadBytes ?? 100,
        residentBytes: opts.residentBytes ?? 200,
      },
      assets: [],
      verification: opts.verification,
    },
  }
}

describe('capability provider resolution', () => {
  it('selects a provider by capability and ordered backend preference', () => {
    const resolution = resolveCapabilityProvider(
      { capability: 'text-generation', backends: ['wasm', 'webgpu'] },
      [provider('gemma', { backends: { webgpu: true, wasm: true } })],
    )

    expect(resolution.selection?.provider.id).toBe('gemma')
    expect(resolution.selection?.backend).toBe('wasm')
    expect(resolution.rejected).toEqual([])
  })

  it('reports every failed constraint for rejected providers', () => {
    const resolution = resolveCapabilityProvider(
      {
        capability: 'text-to-speech',
        backends: ['webgpu'],
        minimumVerification: 'inference',
        maxDownloadBytes: 50,
        maxResidentBytes: 100,
      },
      [
        provider('wrong', {
          capabilities: ['text-generation'],
          backends: { wasm: true },
          verification: verification('assets'),
          downloadBytes: 75,
          residentBytes: 125,
        }),
      ],
    )

    expect(resolution.selection).toBeNull()
    expect(resolution.rejected).toHaveLength(1)
    expect(resolution.rejected[0]?.reasons.map((reason) => reason.code)).toEqual([
      'capability-mismatch',
      'backend-mismatch',
      'verification-too-low',
      'download-budget-exceeded',
      'resident-budget-exceeded',
    ])
  })

  it('scopes verification evidence to the selected backend', () => {
    const webGpuOnlyVerification = verification('output')!
    webGpuOnlyVerification.environments = [
      { browser: 'Chrome', backend: 'webgpu', runtime: 'LiteRT.js' },
    ]

    const wasm = resolveCapabilityProvider(
      {
        capability: 'text-generation',
        backends: ['wasm'],
        minimumVerification: 'output',
      },
      [
        provider('dual-backend', {
          backends: { webgpu: true, wasm: true },
          verification: webGpuOnlyVerification,
        }),
      ],
    )

    expect(wasm.selection).toBeNull()
    expect(wasm.rejected[0]?.reasons).toEqual([
      {
        code: 'verification-too-low',
        detail: 'model is registered; request requires output',
      },
    ])

    const webgpu = resolveCapabilityProvider(
      {
        capability: 'text-generation',
        backends: ['webgpu'],
        minimumVerification: 'output',
      },
      [
        provider('dual-backend', {
          backends: { webgpu: true, wasm: true },
          verification: webGpuOnlyVerification,
        }),
      ],
    )

    expect(webgpu.selection?.backend).toBe('webgpu')
    expect(webgpu.selection?.verification).toBe('output')
  })

  it('keeps manifest-wide verification for legacy evidence without environments', () => {
    const resolution = resolveCapabilityProvider(
      {
        capability: 'text-generation',
        backends: ['wasm'],
        minimumVerification: 'output',
      },
      [provider('legacy', { verification: verification('output') })],
    )

    expect(resolution.selection?.backend).toBe('wasm')
    expect(resolution.selection?.verification).toBe('output')
  })

  it('honors explicit model preference before provider priority', () => {
    const resolution = resolveCapabilityProvider(
      {
        capability: 'text-generation',
        preferredModelIds: ['small-local'],
      },
      [
        provider('large', { modelId: 'large-fast', priority: 100 }),
        provider('small', { modelId: 'small-local', priority: 0 }),
      ],
    )

    expect(resolution.selection?.provider.id).toBe('small')
  })

  it('uses stable identifiers as the final deterministic tie-breaker', () => {
    const a = provider('provider-z', { modelId: 'model-b' })
    const b = provider('provider-a', { modelId: 'model-a' })
    const request = { capability: 'text-generation' as const }

    const forward = resolveCapabilityProvider(request, [a, b])
    const reverse = resolveCapabilityProvider(request, [b, a])

    expect(forward.selection?.provider.id).toBe('provider-a')
    expect(reverse.selection?.provider.id).toBe('provider-a')
  })

  it('uses locale-independent code-unit ordering for deterministic identifiers', () => {
    const request = { capability: 'text-generation' as const }
    const resolution = resolveCapabilityProvider(request, [
      provider('provider-umlaut', { modelId: 'ä-model' }),
      provider('provider-z', { modelId: 'z-model' }),
    ])

    // UTF-16 code-unit order is stable across hosts: "z" sorts before "ä".
    expect(resolution.selection?.provider.id).toBe('provider-z')

    const rejected = resolveCapabilityProvider(
      { capability: 'text-to-speech' },
      [
        provider('ä-rejected'),
        provider('z-rejected'),
      ],
    )

    expect(rejected.rejected.map((candidate) => candidate.providerId)).toEqual([
      'z-rejected',
      'ä-rejected',
    ])
  })

  it('can reject experimental-only backends', () => {
    const backends: Partial<Record<Backend, boolean | 'experimental'>> = { webgpu: 'experimental' }
    const resolution = resolveCapabilityProvider(
      {
        capability: 'text-generation',
        backends: ['webgpu'],
        allowExperimentalBackends: false,
      },
      [provider('experimental', { backends })],
    )

    expect(resolution.selection).toBeNull()
    expect(resolution.rejected[0]?.reasons.map((reason) => reason.code)).toContain('backend-mismatch')
  })

  it('derives monotonic verification levels from manifest evidence', () => {
    expect(getCapabilityVerificationLevel(provider('registered').manifest)).toBe('registered')
    expect(getCapabilityVerificationLevel(provider('assets', { verification: verification('assets') }).manifest)).toBe('assets')
    expect(getCapabilityVerificationLevel(provider('compile', { verification: verification('compile') }).manifest)).toBe('compile')
    expect(getCapabilityVerificationLevel(provider('inference', { verification: verification('inference') }).manifest)).toBe('inference')
    expect(getCapabilityVerificationLevel(provider('output', { verification: verification('output') }).manifest)).toBe('output')
  })
})
