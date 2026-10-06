import { describe, expect, it } from 'vitest'
import { mapQualificationStatus } from './schema/types'
import type { QualificationResult } from './schema/types'
import { compareNumericOutputs } from './shared/numericParity'

describe('qualification manifest mapping', () => {
  it('does not promote failed numeric parity to a qualified manifest', () => {
    const numericComparison = compareNumericOutputs([1], [3], {
      referenceLabel: 'reference', candidateLabel: 'candidate', atol: 0, rtol: 0,
    })
    expect(mapQualificationStatus({
      evidenceKind: 'browser-observation',
      observed: { status: 'pass', numericComparison },
    })).toBe('limited')
    expect(mapQualificationStatus({
      evidenceKind: 'browser-observation',
      observed: { status: 'pass', numericComparison: compareNumericOutputs([1], [1], {
        referenceLabel: 'reference', candidateLabel: 'candidate', atol: 0, rtol: 0,
      }) },
    })).toBe('qualified')
  })
  it('maps passing and limited observations to manifest status', () => {
    expect(mapQualificationStatus({
      evidenceKind: 'browser-observation',
      observed: { status: 'known-limitation' },
    })).toBe('limited')
    expect(mapQualificationStatus({
      evidenceKind: 'browser-observation',
      observed: { status: 'pass' },
    })).toBe('qualified')
  })

  it('requires runtime and model evidence before claiming qualification', () => {
    const result: QualificationResult = {
      schemaVersion: 1,
      caseId: 'qwen-browsermemory-generator',
      evidenceKind: 'contract',
      timestamp: '2026-08-13T00:00:00.000Z',
      playgroundRevision: '2fa9aeb',
      runtimePackage: '@litertjs/core',
      runtimeVersion: '2.5.3',
      environment: {
        runtimePackage: '@litertjs/core',
        runtimeVersion: '2.5.3',
        requestedBackend: 'wasm',
      },
      model: {
        id: 'qwen3-tts',
        revision: 'model-revision',
        assets: [],
      },
      expected: { status: 'known-limitation', error: { stage: 'talker-prefill' } },
      observed: { status: 'fail' },
      matchesExpectation: true,
    }

    expect(result.environment.runtimeVersion).toBe('2.5.3')
    expect(result.model?.revision).toBeTruthy()
    expect(() => mapQualificationStatus(result)).toThrow('browser observation')
  })
})
