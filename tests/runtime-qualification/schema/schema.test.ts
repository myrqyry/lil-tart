import { describe, expect, it } from 'vitest'
import type { QualificationResult } from './types'
import schema from './qualification-result.schema.json'

describe('qualification result schema', () => {
  it('round-trips a serializable qualification result', () => {
    const result: QualificationResult = {
      schemaVersion: 1,
      caseId: 'tiny-litert-wasm-baseline',
      evidenceKind: 'browser-observation',
      timestamp: '2026-08-13T00:00:00.000Z',
      playgroundRevision: '2fa9aeb',
      runtimePackage: '@litertjs/core',
      runtimeVersion: '2.5.3',
      environment: {
        runtimePackage: '@litertjs/core',
        runtimeVersion: '2.5.3',
        requestedBackend: 'wasm',
      },
      expected: { status: 'pass' },
      observed: { status: 'pass', resolvedBackend: 'wasm' },
      matchesExpectation: true,
    }

    expect(JSON.parse(JSON.stringify(result))).toEqual(result)
  })

  it('accepts an explicit unsupported backend observation', () => {
    expect(schema.$defs.observation.properties.status.enum).toContain('unsupported')
  })

  it('classifies the Qwen prefill ceiling as resource exhaustion', () => {
    expect(schema.$defs.observation.properties.limitation.enum).toContain('resource-exhausted')
  })

  it('records numeric parity evidence for silent output corruption', () => {
    expect(schema.$defs.observation.properties.numericComparison).toEqual({
      $ref: '#/$defs/numericComparison',
    })
    expect(schema.$defs.numericComparison.required).toContain('nonFiniteCount')
    expect(schema.$defs.numericComparison.required).toContain('maxAbsDiff')
    expect(schema.$defs.numericComparison.required).toContain('passed')
  })

  it('requires schema version one and a case identifier', () => {
    expect(schema.properties.schemaVersion).toEqual({ const: 1 })
    expect(schema.required).toContain('caseId')
    expect(schema.required).toContain('evidenceKind')
    expect(schema.required).toContain('matchesExpectation')
  })
})
