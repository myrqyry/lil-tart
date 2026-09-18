import type { Backend, Capability, ModelManifest } from './types'

export type CapabilityVerificationLevel =
  | 'registered'
  | 'assets'
  | 'compile'
  | 'inference'
  | 'output'

/**
 * A product-level request for an inference capability. Backend order is both a
 * restriction and a preference order when supplied.
 */
export interface CapabilityRequest {
  capability: Capability
  backends?: readonly Backend[]
  minimumVerification?: CapabilityVerificationLevel
  maxDownloadBytes?: number
  maxResidentBytes?: number
  preferredModelIds?: readonly string[]
  allowExperimentalBackends?: boolean
}

/**
 * A model-backed implementation that can satisfy one or more capabilities.
 * Higher priority wins only after explicit request preferences have been
 * considered. Equal candidates are resolved by stable identifiers.
 */
export interface CapabilityProvider {
  id: string
  manifest: ModelManifest
  priority?: number
}

export type CapabilityRejectionReasonCode =
  | 'capability-mismatch'
  | 'backend-mismatch'
  | 'verification-too-low'
  | 'download-budget-exceeded'
  | 'resident-budget-exceeded'

export interface CapabilityRejectionReason {
  code: CapabilityRejectionReasonCode
  detail: string
}

export interface CapabilityCandidateRejection {
  providerId: string
  modelId: string
  reasons: CapabilityRejectionReason[]
}

export interface CapabilitySelection {
  provider: CapabilityProvider
  backend: Backend
  verification: CapabilityVerificationLevel
}

export interface CapabilityResolution {
  selection: CapabilitySelection | null
  rejected: CapabilityCandidateRejection[]
}
