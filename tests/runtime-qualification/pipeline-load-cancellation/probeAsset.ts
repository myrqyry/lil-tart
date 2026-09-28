import { mkdirSync, statSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { ABORT_PROBE_BYTES, ABORT_PROBE_PATH } from './probeAsset.meta'

// A deliberately large, deterministic local asset used to observe whether an abort
// actually stops a real in-flight transfer. It is generated rather than committed so
// no multi-megabyte binary lands in the repository, and it lives under a gitignored
// directory. The size is large enough that the first chunk is nowhere near the end,
// which is what makes "stopped short of the full asset" a meaningful observation.
export { ABORT_PROBE_BYTES, ABORT_PROBE_PATH }

export function ensureAbortProbeAsset(): string {
  const target = join(process.cwd(), ABORT_PROBE_PATH)
  try {
    if (statSync(target).size === ABORT_PROBE_BYTES) return target
  } catch {
    // Not present yet; generate it below.
  }
  mkdirSync(join(process.cwd(), ABORT_PROBE_PATH.split('/')[0]), { recursive: true })
  // A repeating pattern keeps the payload compressible on disk without relying on
  // anything random, so repeated runs produce identical bytes.
  const block = new Uint8Array(1024 * 1024)
  for (let index = 0; index < block.length; index += 1) block[index] = index % 251
  for (let index = 0; index < ABORT_PROBE_BYTES / block.length; index += 1) {
    writeFileSync(target, block, { flag: index === 0 ? 'w' : 'a' })
  }
  return target
}
