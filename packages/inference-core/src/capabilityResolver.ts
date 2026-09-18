import type {
  CapabilityCandidateRejection,
  CapabilityProvider,
  CapabilityRequest,
  CapabilityResolution,
  CapabilitySelection,
  CapabilityVerificationLevel,
} from './capabilities'
import type { Backend, ModelManifest } from './types'

const DEFAULT_BACKEND_ORDER: readonly Backend[] = ['webgpu', 'webnn', 'wasm']

const VERIFICATION_RANK: Record<CapabilityVerificationLevel, number> = {
  registered: 0,
  assets: 1,
  compile: 2,
  inference: 3,
  output: 4,
}

interface RankedCandidate {
  selection: CapabilitySelection
  preferredModelRank: number
  backendRank: number
  experimentalBackend: boolean
  priority: number
}

export function getCapabilityVerificationLevel(
  manifest: ModelManifest,
  backend?: Backend,
): CapabilityVerificationLevel {
  const verification = manifest.verification
  if (!verification || verification.assets !== 'pass') return 'registered'

  // When qualification environments are recorded, the evidence only applies to
  // those backends. Manifests without environment metadata keep the legacy
  // manifest-wide behavior.
  if (
    backend &&
    verification.environments &&
    verification.environments.length > 0 &&
    !verification.environments.some((environment) => environment.backend === backend)
  ) {
    return 'registered'
  }

  if (verification.compile !== 'pass') return 'assets'
  if (verification.inference !== 'pass') return 'compile'
  if (verification.output !== 'pass') return 'inference'
  return 'output'
}

export function resolveCapabilityProvider(
  request: CapabilityRequest,
  providers: readonly CapabilityProvider[],
): CapabilityResolution {
  const rejected: CapabilityCandidateRejection[] = []
  const candidates: RankedCandidate[] = []
  const backendOrder = request.backends ?? DEFAULT_BACKEND_ORDER
  const minimumVerification = request.minimumVerification ?? 'registered'
  const allowExperimental = request.allowExperimentalBackends ?? true

  for (const provider of providers) {
    const reasons: CapabilityCandidateRejection['reasons'] = []
    const backend = findBackend(provider.manifest, backendOrder, allowExperimental)
    const verification = getCapabilityVerificationLevel(provider.manifest, backend?.backend)

    if (!provider.manifest.capabilities.includes(request.capability)) {
      reasons.push({
        code: 'capability-mismatch',
        detail: `model does not declare ${request.capability}`,
      })
    }

    if (!backend) {
      reasons.push({
        code: 'backend-mismatch',
        detail: 'model has no backend compatible with the request',
      })
    }

    if (VERIFICATION_RANK[verification] < VERIFICATION_RANK[minimumVerification]) {
      reasons.push({
        code: 'verification-too-low',
        detail: `model is ${verification}; request requires ${minimumVerification}`,
      })
    }

    if (
      request.maxDownloadBytes !== undefined &&
      provider.manifest.memory.downloadBytes > request.maxDownloadBytes
    ) {
      reasons.push({
        code: 'download-budget-exceeded',
        detail: `${provider.manifest.memory.downloadBytes} > ${request.maxDownloadBytes} bytes`,
      })
    }

    if (
      request.maxResidentBytes !== undefined &&
      provider.manifest.memory.residentBytes > request.maxResidentBytes
    ) {
      reasons.push({
        code: 'resident-budget-exceeded',
        detail: `${provider.manifest.memory.residentBytes} > ${request.maxResidentBytes} bytes`,
      })
    }

    if (reasons.length > 0 || !backend) {
      rejected.push({
        providerId: provider.id,
        modelId: provider.manifest.modelId,
        reasons,
      })
      continue
    }

    candidates.push({
      selection: {
        provider,
        backend: backend.backend,
        verification,
      },
      preferredModelRank: preferredModelRank(provider.manifest.modelId, request.preferredModelIds),
      backendRank: backend.rank,
      experimentalBackend: backend.experimental,
      priority: provider.priority ?? 0,
    })
  }

  candidates.sort(compareCandidates)
  rejected.sort((a, b) =>
    compareIdentifiers(a.providerId, b.providerId) || compareIdentifiers(a.modelId, b.modelId),
  )

  return {
    selection: candidates[0]?.selection ?? null,
    rejected,
  }
}

function findBackend(
  manifest: ModelManifest,
  backendOrder: readonly Backend[],
  allowExperimental: boolean,
): { backend: Backend; rank: number; experimental: boolean } | null {
  for (let rank = 0; rank < backendOrder.length; rank += 1) {
    const backend = backendOrder[rank]
    if (!backend) continue
    const support = manifest.backends[backend]
    if (support === true) return { backend, rank, experimental: false }
    if (support === 'experimental' && allowExperimental) {
      return { backend, rank, experimental: true }
    }
  }
  return null
}

function preferredModelRank(modelId: string, preferredModelIds?: readonly string[]): number {
  if (!preferredModelIds) return Number.MAX_SAFE_INTEGER
  const rank = preferredModelIds.indexOf(modelId)
  return rank === -1 ? Number.MAX_SAFE_INTEGER : rank
}

function compareIdentifiers(a: string, b: string): number {
  if (a === b) return 0
  return a < b ? -1 : 1
}

function compareCandidates(a: RankedCandidate, b: RankedCandidate): number {
  if (a.preferredModelRank !== b.preferredModelRank) {
    return a.preferredModelRank - b.preferredModelRank
  }
  if (a.backendRank !== b.backendRank) return a.backendRank - b.backendRank
  if (a.experimentalBackend !== b.experimentalBackend) return a.experimentalBackend ? 1 : -1
  if (a.priority !== b.priority) return b.priority - a.priority

  return (
    compareIdentifiers(a.selection.provider.manifest.modelId, b.selection.provider.manifest.modelId) ||
    compareIdentifiers(a.selection.provider.id, b.selection.provider.id)
  )
}
