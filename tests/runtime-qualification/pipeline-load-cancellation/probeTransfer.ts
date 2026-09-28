import { createHttpAssetResolver } from '../../../packages/inference-core/src/assets/http-resolver'
import type { QualificationObservation } from '../schema/types'
import { ABORT_PROBE_BYTES } from './probeAsset.meta'

// This is the buffered resolve() path used by runtime-litert, not the resolver's
// unused stream() entrypoint. Progress lets us abort before the full buffer exists.
export async function probeAbortStopsTransfer(origin: string): Promise<QualificationObservation> {
  const resolver = createHttpAssetResolver(origin)
  const controller = new AbortController()
  let loadedBytes = 0
  let cancellationCode: string | undefined
  let resolved = false
  try {
    await resolver.resolve({ id: 'abort-probe.bin', path: '/asset' }, {
      signal: controller.signal,
      onProgress: (progress) => {
        loadedBytes = progress.loadedBytes
        if (loadedBytes > 0) controller.abort()
      },
    })
    resolved = true
  } catch (error) {
    cancellationCode = (error as { code?: string } | null)?.code
  }
  if (resolved || !controller.signal.aborted || loadedBytes <= 0
    || loadedBytes >= ABORT_PROBE_BYTES || cancellationCode !== 'CANCELLED') {
    return {
      status: 'fail', stage: 'probe', error: {
        message: `resolve() cancellation failed: ${loadedBytes} of ${ABORT_PROBE_BYTES} bytes, `
          + `resolved ${resolved}, cancellation code ${String(cancellationCode)}`,
      },
    }
  }
  return { status: 'pass' }
}
