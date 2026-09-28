import {
  InferenceError,
  type Pipeline,
  type PipelineProgress,
  type PipelineStatus,
  type RuntimeContext,
} from '@litert-playground/inference-core'
import type { MultiVectorEmbeddingResult } from '@litert-playground/inference-core'
import { colbertManifest } from './manifest'

export interface ColBertInput {
  text: string
}

export interface ColBertConfig {
  maxTokens?: number
  maxSeqLen?: number
  progressCallback?: (p: PipelineProgress) => void
}

export interface ColBertPipelineOptions {
  manifest?: typeof colbertManifest
}

interface TransformersTokenizer {
  encode: (
    text: string,
    options?: { padding?: boolean | string; truncation?: boolean | string; max_length?: number },
  ) => Promise<{ input_ids: { data: Int32Array; dims: number[] } | { tolist(): number[][] } }>
}

interface TransformersModule {
  AutoTokenizer: {
    from_pretrained: (
      repo: string,
      options?: { subfolder?: string; dtype?: string },
    ) => Promise<TransformersTokenizer>
  }
}

interface CompiledModel {
  run: (input: unknown) => Promise<Array<{ data: () => Promise<Float32Array | Int32Array> }>>
}

const DEFAULTS = { maxTokens: 512 }

export class ColBertPipeline
  implements Pipeline<ColBertInput, MultiVectorEmbeddingResult, ColBertConfig>
{
  readonly manifest = colbertManifest
  status: PipelineStatus = 'idle'
  onProgress?: (progress: PipelineProgress) => void

  private model: CompiledModel | null = null
  private tokenizer: TransformersTokenizer | null = null
  private loadAbort: AbortController | null = null
  private loadToken = 0
  private disposed = false

  constructor(options: ColBertPipelineOptions = {}) {
    this.manifest = options.manifest ?? colbertManifest
  }

  async load(context: RuntimeContext): Promise<void> {
    if (this.disposed) throw new InferenceError('CANCELLED', 'Pipeline is disposed')
    if (this.status === 'ready') return
    this.status = 'loading'
    // The model load is the cancellable part; the tokenizer fetch is not, because
    // transformers.js exposes no signal for from_pretrained.
    const controller = new AbortController()
    this.loadAbort = controller
    const token = ++this.loadToken
    // An already-aborted signal has already dispatched its event and will never
    // dispatch again, so it must be checked directly before subscribing. This
    // mirrors what the runtime does in startPendingLoad.
    const onExternalAbort = () => controller.abort()
    if (context.signal?.aborted) controller.abort()
    else context.signal?.addEventListener('abort', onExternalAbort, { once: true })
    // Both cancellation routes have to be checked wherever a result would be published:
    // disposal or a superseding load bumps loadToken, while a caller aborting
    // context.signal only flips controller.signal.
    const cancelled = () => token !== this.loadToken || controller.signal.aborted
    try {
      this.report({ phase: 'loading-tokenizer', step: 1, total: 3 })
      const transformers = (await import('@huggingface/transformers')) as unknown as TransformersModule
      // ponytail: repo id is the first two path segments of any asset URL
      const repoId = this.manifest.assets[0].path.split('/').slice(0, 2).join('/')
      const tokenizer = await transformers.AutoTokenizer.from_pretrained(repoId)
      // A dispose landing during the tokenizer fetch must not repopulate state.
      // Reject rather than resolve: the caller asked for a model and did not get one.
      if (cancelled()) {
        throw new InferenceError('CANCELLED', 'Model load was cancelled')
      }
      this.report({ phase: 'loading-model', step: 2, total: 3 })
      const modelPath = this.manifest.assets[0].path
      const model = (await context.liteRt.loadModel(modelPath, { signal: controller.signal })) as CompiledModel
      // A cancellation landing during the load must not resurrect the pipeline. The
      // runtime still owns the compiled model and disposes it with itself.
      if (cancelled()) {
        throw new InferenceError('CANCELLED', 'Model load was cancelled')
      }
      this.tokenizer = tokenizer
      this.model = model
      this.report({ phase: 'ready', step: 3, total: 3 })
      this.status = 'ready'
    } catch (e) {
      // A cancelled load is retryable, not a model failure. Disposal wins the
      // race: a cancelled load must not stamp 'idle' over a disposed pipeline.
      this.status = this.disposed ? 'disposed' : controller.signal.aborted ? 'idle' : 'error'
      throw e instanceof Error ? e : new Error(String(e))
    } finally {
      context.signal?.removeEventListener('abort', onExternalAbort)
    }
  }

  async run(input: ColBertInput, config?: ColBertConfig): Promise<MultiVectorEmbeddingResult> {
    if (this.status !== 'ready' || !this.model || !this.tokenizer) {
      throw new Error('ColBERT pipeline not ready')
    }
    const maxTokens = config?.maxTokens ?? DEFAULTS.maxTokens
    this.status = 'running'
    try {
      const encoded = await this.tokenizer.encode(input.text, {
        padding: 'max_length',
        truncation: true,
        max_length: maxTokens,
      })
      const inputIds = toInt32Array(encoded.input_ids)
      const outputs = await this.model.run(createInputTensor(inputIds, maxTokens))
      const first = outputs[0]
      const data = await first.data()
      const dim = data.length / maxTokens
      return {
        kind: 'multi-vector-embedding',
        values: data instanceof Float32Array ? data : new Float32Array(data),
        tokens: maxTokens,
        dimensions: dim,
      }
    } finally {
      this.status = this.disposed ? 'disposed' : 'ready'
    }
  }

  async dispose(): Promise<void> {
    // Record disposal before anything else so an in-flight load cannot overwrite it.
    this.disposed = true
    // Stop an in-flight model download even when the caller wired no signal.
    this.loadAbort?.abort()
    this.loadAbort = null
    // Invalidate any load still awaiting a result.
    this.loadToken += 1
    this.model = null
    this.tokenizer = null
    this.status = 'disposed'
  }

  private report(progress: PipelineProgress): void {
    this.onProgress?.(progress)
  }
}

type EncodedInputIds = { data: Int32Array; dims: number[] } | { tolist(): number[][] }

function toInt32Array(inputIds: EncodedInputIds): Int32Array {
  if ('data' in inputIds) {
    return inputIds.data
  }
  const rows = inputIds.tolist()
  return new Int32Array(rows.length > 0 ? rows[0] : [])
}

function createInputTensor(ids: Int32Array, seqLen: number): unknown {
  return {
    data: ids.length === seqLen ? ids : pad(ids, seqLen),
    shape: [1, seqLen],
  }
}

function pad(ids: Int32Array, len: number): Int32Array {
  const out = new Int32Array(len)
  out.set(ids.subarray(0, Math.min(ids.length, len)))
  return out
}
