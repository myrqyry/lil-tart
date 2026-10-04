import type { QualificationNumericComparison } from '../schema/types'

export interface NumericParityOptions {
  referenceLabel: string
  candidateLabel: string
  atol: number
  rtol: number
}

export function compareNumericOutputs(
  reference: ArrayLike<number>,
  candidate: ArrayLike<number>,
  options: NumericParityOptions,
): QualificationNumericComparison {
  if (reference.length !== candidate.length) {
    throw new Error(
      `Numeric parity requires equal element counts: ${reference.length} !== ${candidate.length}`,
    )
  }

  let maxAbsDiff = 0
  let maxRelDiff = 0
  let nonFiniteCount = 0
  let failingElements = 0

  for (let index = 0; index < reference.length; index += 1) {
    const expected = reference[index]
    const actual = candidate[index]
    if (!Number.isFinite(expected) || !Number.isFinite(actual)) {
      nonFiniteCount += 1
      failingElements += 1
      continue
    }

    const absDiff = Math.abs(actual - expected)
    const relDiff = absDiff / Math.max(Math.abs(expected), Number.EPSILON)
    maxAbsDiff = Math.max(maxAbsDiff, absDiff)
    maxRelDiff = Math.max(maxRelDiff, relDiff)

    if (absDiff > options.atol + options.rtol * Math.abs(expected)) {
      failingElements += 1
    }
  }

  return {
    reference: options.referenceLabel,
    candidate: options.candidateLabel,
    elementCount: reference.length,
    nonFiniteCount,
    failingElements,
    maxAbsDiff,
    maxRelDiff,
    atol: options.atol,
    rtol: options.rtol,
    passed: nonFiniteCount === 0 && failingElements === 0,
  }
}
