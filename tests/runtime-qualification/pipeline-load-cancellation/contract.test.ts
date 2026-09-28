import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { pipelineLoadCancellationCase } from './case'
import { ABORT_PROBE_BYTES, ABORT_PROBE_PATH } from './probeAsset.meta'

describe('pipeline load cancellation contract', () => {
  it('expects a passing browser observation', () => {
    expect(pipelineLoadCancellationCase.expected).toEqual({ status: 'pass' })
    expect(pipelineLoadCancellationCase.evidenceKind).toBe('browser-observation')
  })

  it('declares a single WASM environment', () => {
    expect(pipelineLoadCancellationCase.environments).toEqual([
      expect.objectContaining({ requestedBackend: 'wasm' }),
    ])
  })

  // The probe asserts "fewer bytes arrived than the asset holds". If the asset were
  // small enough to arrive in a single chunk, that comparison would be meaningless
  // and the case would pass for the wrong reason.
  it('uses a probe asset large enough to observe a partial transfer', () => {
    expect(ABORT_PROBE_BYTES).toBeGreaterThan(4 * 1024 * 1024)
  })

  // Reads .gitignore rather than asserting the constant matches itself: the previous
  // version compared the path to a prefix it had just been written with, so deleting
  // the ignore rule would have left it green while a 24 MiB binary sat one `git add`
  // away from a permanent commit.
  it('keeps the generated probe asset out of version control', () => {
    const gitignore = readFileSync(new URL('../../../.gitignore', import.meta.url), 'utf8')
    const directory = ABORT_PROBE_PATH.split('/')[0]
    const rules = gitignore
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith('#'))
    const ignored = rules.some((rule) => {
      const normalized = rule.replace(/^\/+|\/+$/g, '')
      return normalized === directory || normalized === `/${directory}`
    })
    expect(ignored, `${directory} must be ignored so the generated asset cannot be committed`).toBe(true)
  })
})
