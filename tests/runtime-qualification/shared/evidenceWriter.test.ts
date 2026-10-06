import { describe, expect, it } from 'vitest'
import {
  createQualificationResult,
  matchQualificationExpectation,
  normalizeQualificationError,
  writeQualificationResult,
} from './evidenceWriter'
import type {
  QualificationCase,
  QualificationObservation,
} from '../schema/types'
import { mkdtemp, readFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { compareNumericOutputs } from './numericParity'
import { matchQualificationExpectation as matchSchemaExpectation } from '../schema/types'

const failedParity = compareNumericOutputs([1, 2], [1, 5], {
  referenceLabel: 'reference', candidateLabel: 'candidate', atol: 0, rtol: 0,
})

describe('qualification evidence', () => {
  it.each([matchQualificationExpectation, matchSchemaExpectation])(
    'rejects a passing status with failed numeric parity at either matching entrypoint',
    (match) => {
      expect(match({ status: 'pass' }, {
        status: 'pass', numericComparison: failedParity,
      })).toBe(false)
      expect(match({ status: 'pass' }, {
        status: 'pass', numericComparison: compareNumericOutputs([1, 2], [1, 2], {
          referenceLabel: 'reference', candidateLabel: 'candidate', atol: 0, rtol: 0,
        }),
      })).toBe(true)
      expect(match({ status: 'known-limitation' }, {
        status: 'fail', numericComparison: failedParity,
      })).toBe(false)
    },
  )

  it('keeps failed numeric evidence while reporting a qualification mismatch', () => {
    const result = createQualificationResult({
      caseId: 'numeric-parity', evidenceKind: 'browser-observation',
      timestamp: '2026-10-05T00:00:00.000Z', playgroundRevision: 'test',
      runtimePackage: '@litertjs/core', runtimeVersion: '2.5.3',
      environment: {
        runtimePackage: '@litertjs/core', runtimeVersion: '2.5.3', requestedBackend: 'wasm',
      },
      expected: { status: 'pass' },
      observed: { status: 'pass', numericComparison: failedParity },
    })
    expect(result.matchesExpectation).toBe(false)
    expect(result.observed.numericComparison).toEqual(failedParity)
  })

  it('matches a passing observation', () => {
    expect(matchQualificationExpectation(
      { status: 'pass' },
      { status: 'pass' },
    )).toBe(true)
  })

  it('matches a confirmed known limitation', () => {
    const expected: QualificationCase['expected'] = {
      status: 'known-limitation',
      error: {
        code: 'ASSET_FETCH_FAILED',
        stage: 'prefill',
        messagePattern: 'XNNPACK',
      },
    }
    const observed: QualificationObservation = {
      status: 'fail',
      stage: 'prefill',
      error: {
        code: 'ASSET_FETCH_FAILED',
        stage: 'prefill',
        message: 'XNNPACK runtime creation failed',
      },
    }

    expect(matchQualificationExpectation(expected, observed)).toBe(true)
  })

  it('matches a promoted known-limitation observation', () => {
    const expected: QualificationCase['expected'] = {
      status: 'known-limitation',
      error: {
        code: 'RESOURCE_EXHAUSTED',
        stage: 'talker-prefill',
        messagePattern: 'TensorBuffer',
      },
    }
    const observed: QualificationObservation = {
      status: 'known-limitation',
      stage: 'talker-prefill',
      limitation: 'resource-exhausted',
      error: {
        code: 'RESOURCE_EXHAUSTED',
        stage: 'talker-prefill',
        message: 'TensorBuffer allocation failed',
      },
    }

    expect(matchQualificationExpectation(expected, observed)).toBe(true)
    expect(matchQualificationExpectation({ status: 'fail' }, observed)).toBe(false)
  })

  it('does not treat an upstream fix as a known limitation match', () => {
    expect(matchQualificationExpectation(
      { status: 'known-limitation' },
      { status: 'pass' },
    )).toBe(false)
  })

  it('normalizes errors with a fallback stage', () => {
    expect(normalizeQualificationError(new Error('broken'), 'load')).toEqual({
      message: 'broken',
      stage: 'load',
    })
  })

  it('writes JSON evidence without runtime-only values', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'qualification-'))
    const result = createQualificationResult({
      caseId: 'case',
      evidenceKind: 'browser-observation',
      timestamp: '2026-08-13T00:00:00.000Z',
      playgroundRevision: 'abc',
      runtimePackage: '@litertjs/core',
      runtimeVersion: '2.5.3',
      environment: {
        runtimePackage: '@litertjs/core',
        runtimeVersion: '2.5.3',
        requestedBackend: 'wasm',
      },
      expected: { status: 'pass' },
      observed: { status: 'pass' },
    })

    const path = await writeQualificationResult(directory, result)
    expect(JSON.parse(await readFile(path, 'utf8'))).toEqual(result)
  })
})
