import { mkdir, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import type {
  QualificationCase,
  QualificationError,
  QualificationObservation,
  QualificationResult,
} from '../schema/types'

export function normalizeQualificationError(
  error: unknown,
  fallbackStage: string,
): QualificationError {
  if (error && typeof error === 'object') {
    const value = error as Record<string, unknown>
    return {
      code: typeof value.code === 'string' ? value.code : undefined,
      stage: typeof value.stage === 'string' ? value.stage : fallbackStage,
      message: typeof value.message === 'string'
        ? value.message
        : String(error),
    }
  }

  return { message: String(error), stage: fallbackStage }
}

export function matchQualificationExpectation(
  expected: QualificationCase['expected'],
  observed: QualificationObservation,
): boolean {
  if (observed.numericComparison?.passed === false) return false
  if (expected.status === 'pass') return observed.status === 'pass'
  // ponytail: expected known-limitation accepts both the raw failure and a
  // promoted observation; everything else must match exactly.
  const statusMatches = observed.status === 'fail'
    || (expected.status === 'known-limitation' && observed.status === 'known-limitation')
  if (!statusMatches) return false

  const expectedError = expected.error
  if (!expectedError) return true

  const actual = observed.error
  if (!actual) return false
  if (expectedError.code && expectedError.code !== actual.code) return false
  if (expectedError.stage && expectedError.stage !== (observed.stage ?? actual.stage)) {
    return false
  }
  if (expectedError.messagePattern && !new RegExp(expectedError.messagePattern).test(actual.message)) {
    return false
  }
  return true
}

export function createQualificationResult(
  input: Omit<QualificationResult, 'schemaVersion' | 'matchesExpectation'>,
): QualificationResult {
  return {
    ...input,
    schemaVersion: 1,
    matchesExpectation: matchQualificationExpectation(input.expected, input.observed),
  }
}

export async function writeQualificationResult(
  directory: string,
  result: QualificationResult,
): Promise<string> {
  const serialized = JSON.stringify(result, null, 2)
  await mkdir(directory, { recursive: true })
  const filePath = join(directory, `${result.caseId}-${result.timestamp.replace(/:/g, '-')}.json`)
  await writeFile(filePath, `${serialized}\n`, 'utf8')
  return filePath
}
