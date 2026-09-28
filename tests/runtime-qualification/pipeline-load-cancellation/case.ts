import type {
  QualificationCase,
  QualificationContext,
  QualificationObservation,
} from '../schema/types'
import { pipelineLoadCancellationExpected } from './expected'

export async function runPipelineLoadCancellation(
  context: QualificationContext,
): Promise<QualificationObservation> {
  const probe = context.runtime.probeAbortStopsTransfer
  if (!probe) {
    throw new Error('Abort probe browser adapter is unavailable')
  }
  return probe()
}

export const pipelineLoadCancellationCase: QualificationCase = {
  id: 'pipeline-load-cancellation',
  description:
    'Proves that aborting a real in-flight model transfer stops the bytes, using the shared asset resolver.',
  evidenceKind: 'browser-observation',
  requiresProbeAsset: true,
  environments: [
    {
      runtimePackage: '@litertjs/core',
      runtimeVersion: '2.5.3',
      requestedBackend: 'wasm',
    },
  ],
  expected: pipelineLoadCancellationExpected,
  run: runPipelineLoadCancellation,
}
