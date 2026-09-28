import type { InferenceDiagnostics } from '../../../packages/inference-core/src/types'
import type { Qwen3TtsVariant } from '../../../packages/qwen3-tts/src/manifest'
import type { QwenTtsConfig } from '../../../packages/qwen3-tts/src/types'
import type { GeneratorTraceEvent } from '../../../packages/qwen3-tts/src/generator-trace'

export type QualificationStatus = 'pass' | 'known-limitation' | 'fail' | 'unsupported'
export type QualificationBackend = 'wasm' | 'webgpu'
export type QualificationEvidenceKind = 'contract' | 'browser-observation'
export type QualificationDType = 'float32' | 'int32' | 'uint8'
export type QualificationLimitation = 'resource-exhausted'

export interface QualificationEnvironment {
  browser?: string
  browserVersion?: string
  operatingSystem?: string
  device?: string
  gpu?: string
  runtimePackage: string
  runtimeVersion: string
  requestedBackend: QualificationBackend
  resolvedBackend?: QualificationBackend
  webgpuAvailable?: boolean
}

export interface ModelAssetDescriptor {
  id: string
  url: string
  bytes: number
  sha256: string
  mimeType?: string
}

export interface QualificationModel {
  id: string
  variant?: string
  revision?: string
  revisions?: Record<string, string>
  assets: ModelAssetDescriptor[]
}

export interface QualificationError {
  code?: string
  stage?: string
  message: string
}

export interface QualificationObservation {
  status: QualificationStatus
  limitation?: QualificationLimitation
  stage?: string
  resolvedBackend?: QualificationBackend
  diagnostics?: InferenceDiagnostics
  error?: QualificationError
  receipts?: GeneratorTraceEvent[]
}

export interface QualificationContext {
  requestedBackend: QualificationBackend
  environment: QualificationEnvironment
  fetchAsset(asset: ModelAssetDescriptor): Promise<ArrayBuffer>
  runtime: QualificationRuntime
}

export interface QualificationRuntime {
  initialize?(): Promise<void>
  runModuleWorkerLoader?(): Promise<QualificationObservation>
  probeAbortStopsTransfer?(): Promise<QualificationObservation>
  loadAndCompile(
    model: Uint8Array,
    options: { accelerator: QualificationBackend },
  ): Promise<QualificationCompiledModel>
  loadAndCompileAsset?(
    asset: ModelAssetDescriptor,
    options: { accelerator: QualificationBackend },
  ): Promise<QualificationCompiledModel>
  runQwenGenerator?(request: QwenGeneratorRequest): Promise<QwenGeneratorRunResult>
}

export interface QualificationCompiledModel {
  getInputDetails(): readonly QualificationTensorDetails[]
  getOutputDetails(): readonly QualificationTensorDetails[]
  getModelDetails?(): Promise<{
    inputs: readonly QualificationTensorDetails[]
    outputs: readonly QualificationTensorDetails[]
  }>
  getSignatureDetails(signature: string): Promise<{
    inputs: readonly QualificationTensorDetails[]
    outputs: readonly QualificationTensorDetails[]
  }>
  runSignatureWithZeros?(signature: string): Promise<void>
  runWithZeros?(): Promise<void>
  run(
    input: QualificationTensorInput,
    signature?: string,
  ): Promise<QualificationTensorOutput>
  delete(): Promise<void> | void
}

export interface QwenGeneratorAssetDescriptor extends ModelAssetDescriptor {
  path: string
}

export interface QwenGeneratorRequest {
  variant: Qwen3TtsVariant
  assets: QwenGeneratorAssetDescriptor[]
  backend: QualificationBackend
  text: string
  config: QwenTtsConfig
}

export interface QwenGeneratorRunResult {
  observation: QualificationObservation
  receipts: GeneratorTraceEvent[]
}

export interface QualificationTensorDetails {
  name: string
  shape: readonly number[]
  dtype: QualificationDType
}

export interface QualificationTensor {
  data: unknown
  shape?: readonly number[]
  dtype?: QualificationDType
}

export type QualificationTensorInput =
  | QualificationTensor[]
  | Record<string, QualificationTensor>

export type QualificationTensorOutput =
  | QualificationTensor[]
  | Record<string, QualificationTensor>

export interface QualificationCase {
  id: string
  description: string
  evidenceKind: QualificationEvidenceKind
  model?: QualificationModel
  environments: QualificationEnvironment[]
  expected: {
    status: 'pass' | 'known-limitation'
    error?: {
      code?: string
      stage?: string
      messagePattern?: string
    }
  }
  run(context: QualificationContext): Promise<QualificationObservation>
}

export interface QualificationResult {
  schemaVersion: 1
  caseId: string
  evidenceKind: QualificationEvidenceKind
  timestamp: string
  playgroundRevision: string
  runtimePackage: string
  runtimeVersion: string
  environment: QualificationEnvironment
  model?: QualificationModel
  expected: QualificationCase['expected']
  observed: QualificationObservation
  matchesExpectation: boolean
}

export interface QualificationSelection {
  caseIds?: string[]
  backends?: QualificationBackend[]
}

export interface QualificationRunOptions {
  playgroundRevision: string
  runtimePackage: string
  runtimeVersion: string
  resultsDirectory?: string
  now?: () => Date
}

export function matchQualificationExpectation(
  expected: QualificationCase['expected'],
  observed: QualificationObservation,
): boolean {
  if (expected.status === 'pass') return observed.status === 'pass'
  return observed.status === 'fail'
}

export function mapQualificationStatus(
  result: Pick<QualificationResult, 'evidenceKind' | 'observed'>,
): 'limited' | 'qualified' {
  if (result.evidenceKind !== 'browser-observation') {
    throw new Error('Manifest qualification requires browser observation evidence')
  }
  return result.observed.status === 'pass' ? 'qualified' : 'limited'
}
