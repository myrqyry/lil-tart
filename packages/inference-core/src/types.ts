export type Capability =
  | 'text-generation'
  | 'text-to-speech'
  | 'speech-to-text'
  | 'image-generation'
  | 'image-edit'
  | 'image-classification'
  | 'image-embedding'
  | 'image-segmentation'
  | 'image-super-resolution'
  | 'image-depth'
  | 'image-matting'
  | 'image-detection'
  | 'image-pose'
  | 'audio-generation'
  | 'text-embedding'
  | 'audio-embedding'
  | 'image-text-to-text'
  | 'token-classification'
  | 'text-classification'
  | 'reranking'
  | 'multi-vector-embedding'
  | 'policy-classification'
  | 'reasoning'

export type Backend = 'webgpu' | 'wasm' | 'webnn'
export type VerificationState = 'pass' | 'fail' | 'untested'
export type QualificationStatus = 'unverified' | 'qualified' | 'limited'

export interface ModelAsset {
  id: string
  path: string
  bytes?: number
  sha256?: string
  mimeType?: string
  role?: string
  optional?: boolean
}

export interface AssetProgress {
  asset: string
  loadedBytes: number
  totalBytes?: number
}

export interface AssetRequestOptions {
  signal?: AbortSignal
  onProgress?: (progress: AssetProgress) => void
}

export interface AssetResolver {
  resolve(asset: ModelAsset, options?: AssetRequestOptions): Promise<ArrayBuffer>
  stream?(asset: ModelAsset, options?: AssetRequestOptions): Promise<ReadableStream<Uint8Array>>
}

export interface RuntimeContext {
  backend: Backend
  assets: AssetResolver
  signal?: AbortSignal
  liteRt: LiteRtRuntime
}

export interface LiteRtLoadOptions {
  signal?: AbortSignal
}

export interface LiteRtRuntime {
  loadModel(path: string, options?: LiteRtLoadOptions): Promise<any>
  loadNpy(path: string, signal?: AbortSignal): Promise<Float32Array>
  fetchBuffer(path: string, signal?: AbortSignal): Promise<ArrayBuffer>
}

export type PipelineStatus = 'idle' | 'loading' | 'ready' | 'running' | 'error' | 'disposed'

export interface Pipeline<I = unknown, O = unknown, C = unknown> {
  readonly manifest: ModelManifest
  readonly status: PipelineStatus
  load(context: RuntimeContext): Promise<void>
  run(input: I, config?: C, signal?: AbortSignal): Promise<O>
  dispose(): Promise<void>
}

export interface PipelineProgress {
  phase: string
  step: number
  total: number
}

export interface PipelineProgressReporter {
  report(progress: PipelineProgress): void
}

export type InferenceResult =
  | AudioInferenceResult
  | TextInferenceResult
  | ImageInferenceResult
  | EmbeddingInferenceResult
  | MultiVectorEmbeddingResult
  | TensorInferenceResult

export interface AudioInferenceResult {
  kind: 'audio'
  samples: Float32Array
  sampleRate: number
  channels: number
  durationSeconds: number
  receipt: InferenceReceipt
}

export interface TextInferenceResult {
  kind: 'text'
  text: string
  reasoning?: string
  receipt?: InferenceReceipt
}

export interface ImageInferenceResult {
  kind: 'image'
  width: number
  height: number
  pixels: Uint8ClampedArray
}

export interface EmbeddingInferenceResult {
  kind: 'embedding'
  values: Float32Array
  dimensions: number
}

export interface MultiVectorEmbeddingResult {
  kind: 'multi-vector-embedding'
  values: Float32Array
  tokens: number
  dimensions: number
}

export interface RetrievalResult<T = unknown> {
  id: string
  score: number
  payload?: T
}

export interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
  label?: string
  score?: number
}

export interface Point {
  x: number
  y: number
}

export interface ImageInput {
  width: number
  height: number
  data: Uint8Array
  mimeType?: string
}

export interface VisionLanguageInput {
  image: ImageInput
  prompt: string
}

export interface VisionLanguageResult {
  text: string
  boxes?: BoundingBox[]
  points?: Point[]
  receipt: InferenceReceipt
}

export interface TensorInferenceResult {
  kind: 'tensor'
  tensors: Record<string, any>
}

export interface ModelManifest {
  modelId: string
  name: string
  version: string
  capabilities: Capability[]
  backends: Partial<Record<Backend, boolean | 'experimental'>>
  memory: { downloadBytes: number; residentBytes: number }
  assets: ModelAsset[]
  verification?: ModelVerification
}

export interface ModelVerification {
  assets: VerificationState
  compile: VerificationState
  inference: VerificationState
  output: VerificationState
  qualification?: QualificationStatus
  upstreamRevision?: string
  environments?: ModelVerificationEnvironment[]
  expectedOutput?: ExpectedOutputCharacteristics
  lastVerifiedAt?: string
  environment?: string
}

export interface ModelVerificationEnvironment {
  browser: string
  backend: Backend
  runtime: string
  device?: string
}

export interface ExpectedOutputCharacteristics {
  preprocessing?: string[]
  outputShape?: number[]
  outputDimension?: number
  labels?: {
    assetId: string
    count: number
    mapping: string
  }
  behavior?: string[]
}

export type InferenceErrorCode =
  | 'BACKEND_UNAVAILABLE'
  | 'ASSET_FETCH_FAILED'
  | 'ASSET_INTEGRITY_FAILED'
  | 'MODEL_COMPILE_FAILED'
  | 'INVALID_INPUT'
  | 'OUT_OF_MEMORY'
  | 'INFERENCE_FAILED'
  | 'OUTPUT_INVALID'
  | 'CANCELLED'

export class InferenceError extends Error {
  readonly code: InferenceErrorCode
  readonly stage?: string
  readonly asset?: string
  readonly cause?: unknown

  constructor(code: InferenceErrorCode, message: string, opts?: { stage?: string; asset?: string; cause?: unknown }) {
    super(message)
    this.name = 'InferenceError'
    this.code = code
    this.stage = opts?.stage
    this.asset = opts?.asset
    if (opts?.cause !== undefined) this.cause = opts.cause
  }
}

export interface InferencePhaseReceipt {
  name: string
  backend: Backend
  loadMs?: number
  compileMs?: number
  inferenceMs?: number
  warnings?: string[]
}

export interface InferenceReceipt {
  modelId: string
  pipelineVersion: string
  backend: Backend
  timestamp: string
  loadMs: number
  compileMs: number
  inferenceMs: number
  inputSummary: string
  outputSummary: string
  warnings: string[]
  environment?: string
  phases?: InferencePhaseReceipt[]
  diagnostics?: InferenceDiagnostics
}

export interface InferenceDiagnostics {
  packageName: string
  modelId: string
  requestedBackend: Backend | 'auto'
  resolvedBackend?: Backend
  cacheHit: boolean
  compileMs?: number
  inferenceMs?: number
  fallbackCount: number
  queueMs?: number
  error?: {
    code: string
    message: string
    stage?: string
    asset?: string
  }
}

export interface RuntimeCapabilities {
  webgpu: { available: boolean; adapter?: Record<string, unknown> }
  wasm: { available: boolean; simd: boolean; threads: boolean; jspi: boolean }
  webnn: { available: boolean; reason?: string }
}
