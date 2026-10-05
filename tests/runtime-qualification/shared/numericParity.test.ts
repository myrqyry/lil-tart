import { describe, expect, it } from 'vitest'

import { compareNumericOutputs } from './numericParity'

const options = {
  referenceLabel: 'wasm',
  candidateLabel: 'webgpu',
  atol: 1e-4,
  rtol: 1e-3,
}

describe('numeric parity evidence', () => {
  it('passes outputs inside absolute and relative tolerance', () => {
    const result = compareNumericOutputs(
      new Float32Array([0, 1, 100]),
      new Float32Array([0.00005, 1.0005, 100.05]),
      options,
    )

    expect(result.passed).toBe(true)
    expect(result.failingElements).toBe(0)
    expect(result.nonFiniteCount).toBe(0)
  })

  it('records silent numerical corruption as failed evidence', () => {
    const result = compareNumericOutputs(
      new Float32Array([1, 2, 3]),
      new Float32Array([1, Number.NaN, 5]),
      options,
    )

    expect(result.passed).toBe(false)
    expect(result.nonFiniteCount).toBe(1)
    expect(result.failingElements).toBe(2)
    expect(result.maxAbsDiff).toBe(2)
  })

  it('rejects shape-length mismatches instead of comparing partial outputs', () => {
    expect(() => compareNumericOutputs(
      new Float32Array([1, 2]),
      new Float32Array([1]),
      options,
    )).toThrow('Numeric parity requires equal element counts')
  })
})
