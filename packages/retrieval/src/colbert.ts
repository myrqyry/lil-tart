import {
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

  constructor(options: ColBertPipelineOptions = {}) {
    this.manifest = options.manifest ?? colbertManifest
  }

  async load(context: RuntimeContext): Promise<void> {
    if (this.status === 'ready') return
    this.status = 'loading'
    // The model load is the cancellable part; the tokenizer fetch is not, because
    // transformers.js exposes no signal for from_pretrained.
    const controller = new AbortController()
    this.loadAbort = controller
    const onExternalAbort = () => controller.abort()
    context.signal?.addEventListener('abort', onExternalAbort, { once: true })
    try {
      this.report({ phase: 'loading-tokenizer', step: 1, total: 3 })
      const transformers = (await import('@huggingface/transformers')) as unknown as TransformersModule
      // ponytail: repo id is the first two path segments of any asset URL
      const repoId = this.manifest.assets[0].path.split('/').slice(0, 2).join('/')
      this.tokenizer = await transformers.AutoTokenizer.from_pretrained(repoId)
      this.report({ phase: 'loading-model', step: 2, total: 3 })
      const modelPath = this.manifest.assets[0].path
      this.model = (await context.liteRt.loadModel(modelPath, { signal: controller.signal })) as CompiledModel
      this.report({ phase: 'ready', step: 3, total: 3 })
      this.status = 'ready'
    } catch (e) {
      // A cancelled load is retryable, not a model failure.
      this.status = controller.signal.aborted ? 'idle' : 'error'
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
      this.status = 'ready'
    }
  }

  async dispose(): Promise<void> {
    // Stop an in-flight model download even when the caller wired no signal.
    this.loadAbort?.abort()
    this.loadAbort = null
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
