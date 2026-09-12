import type { InferenceContext, ModelAdapter } from './types'
import { loadHfTokenizer } from '../hfTokenizer'

// ColBERT skiplist: 32 ASCII punctuation tokens contribute no document signal.
const SKIPLIST = new Set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'.split(''))

interface LateInteractionConfig {
  modelId: string
  name: string
  description: string
  tags: string[]
  base: string
  file: string
  signatures: number[]
  dim: number
  padId: number
  queryMarker: number
  documentMarker: number
  queryMaxLength: number
  documentMaxLength: number
  lowercase?: boolean
  skiplist?: boolean
  /** Force every query through this signature (LFM2.5-ColBERT uses 32 for query expansion). */
  querySize?: number
  /** Keep all query vectors of the signature, including padded expansion positions. */
  queryKeepAll?: boolean
}

function smallestSignature(length: number, signatures: number[]): number {
  return signatures.find((size) => size >= length) ?? signatures[signatures.length - 1]
}

/** Runs one ColBERT encode signature, returning the L2-normalized per-token vectors to score. */
async function encodeVectors(
  text: string,
  marker: number,
  maxLength: number,
  ctx: InferenceContext,
  config: LateInteractionConfig,
  isQuery: boolean,
): Promise<Float32Array[]> {
  const tokenizer = await loadHfTokenizer(`${config.base}/tokenizer.json`)
  const base = tokenizer.encode(config.lowercase ? text.toLowerCase() : text, { maxLength })
  const ids = [base[0], marker, ...base.slice(1)]
  const size = isQuery && config.querySize ? config.querySize : smallestSignature(ids.length, config.signatures)
  const inputIds = new Int32Array(size).fill(config.padId)
  inputIds.set(ids)
  const mask = new Int32Array(size)
  for (let i = 0; i < ids.length; i++) mask[i] = 1

  const result = await ctx.predict(
    'main',
    { input_ids: ctx.createTensor(inputIds, [1, size]), attention_mask: ctx.createTensor(mask, [1, size]) },
    `encode_${size}`,
  )
  const data = (await Object.values(result)[0].data()) as Float32Array
  const rows = isQuery && config.queryKeepAll ? size : ids.length
  const vectors: Float32Array[] = []
  for (let row = 0; row < rows; row++) {
    if (config.skiplist && row < ids.length) {
      const token = tokenizer.tokenOf(ids[row])
      if (token !== undefined && token.length === 1 && SKIPLIST.has(token)) continue
    }
    vectors.push(data.subarray(row * config.dim, (row + 1) * config.dim))
  }
  return vectors
}

function maxSim(query: Float32Array[], document: Float32Array[], dim: number): number {
  let total = 0
  for (const q of query) {
    let best = -Infinity
    for (const d of document) {
      let dot = 0
      for (let k = 0; k < dim; k++) dot += q[k] * d[k]
      if (dot > best) best = dot
    }
    total += best
  }
  return total
}

function makeLateInteractionAdapter(config: LateInteractionConfig): ModelAdapter {
  return {
    modelId: config.modelId,
    metadata: { name: config.name, description: config.description, modelPath: `${config.base}/${config.file}`, tags: config.tags },
    inputSpecs: [
      { name: 'query', dtype: 'string', shape: [], description: 'Search query', constraints: { text: true } },
      { name: 'document', dtype: 'string', shape: [], description: 'Passage to score against the query', constraints: { text: true } },
    ],
    outputSpecs: [{ name: 'score', dtype: 'float32', shape: [], description: 'MaxSim relevance score (higher is more relevant)' }],
    prepareInputs: () => ({}),
    parseOutputs: async () => ({}),
    async run(values, ctx) {
      const query = String(values.query ?? '').trim()
      const document = String(values.document ?? '').trim()
      if (!query || !document) throw new Error('Provide both a query and a document')
      const queryVectors = await encodeVectors(query, config.queryMarker, config.queryMaxLength, ctx, config, true)
      const documentVectors = await encodeVectors(document, config.documentMarker, config.documentMaxLength, ctx, config, false)
      return { score: Number(maxSim(queryVectors, documentVectors, config.dim).toFixed(4)) }
    },
  }
}

export const mxbaiColbertAdapter = makeLateInteractionAdapter({
  modelId: 'mxbai-colbert',
  name: 'mxbai-edge-colbert-v0-32m',
  description: 'Late-interaction (ColBERT) text reranker. Scores how relevant a document is to a query via MaxSim over per-token embeddings.',
  tags: ['text', 'retrieval', 'colbert'],
  base: 'https://huggingface.co/litert-community/mxbai-edge-colbert-v0-32m/resolve/main',
  file: 'mxbai-edge-colbert-v0-32m_fp16.tflite',
  signatures: [48, 128, 256, 512],
  dim: 64,
  padId: 50284,
  queryMarker: 50368,
  documentMarker: 50369,
  queryMaxLength: 47,
  documentMaxLength: 511,
  lowercase: true,
  skiplist: true,
})

export const mlateonAdapter = makeLateInteractionAdapter({
  modelId: 'mlateon',
  name: 'mLateOn',
  description: 'Multilingual late-interaction (ColBERT) retriever. 128-dim per-token vectors scored with MaxSim; no query expansion or skiplist.',
  tags: ['text', 'retrieval', 'colbert'],
  base: 'https://huggingface.co/litert-community/mLateOn/resolve/main',
  file: 'mLateOn_wi8fc.tflite',
  signatures: [32, 128, 256, 512],
  dim: 128,
  padId: 4,
  queryMarker: 256000,
  documentMarker: 256001,
  queryMaxLength: 511,
  documentMaxLength: 511,
})

export const lfmColbertAdapter = makeLateInteractionAdapter({
  modelId: 'lfm2.5-colbert',
  name: 'LFM2.5-ColBERT-350M',
  description: 'Multilingual late-interaction retriever with query expansion. Queries keep all 32 vectors; documents drop punctuation tokens before MaxSim.',
  tags: ['text', 'retrieval', 'colbert'],
  base: 'https://huggingface.co/litert-community/LFM2.5-ColBERT-350M/resolve/main',
  file: 'LFM2.5-ColBERT-350M_wi8fc.tflite',
  signatures: [32, 128, 256, 512],
  dim: 128,
  padId: 7,
  queryMarker: 64400,
  documentMarker: 64401,
  queryMaxLength: 31,
  documentMaxLength: 511,
  skiplist: true,
  querySize: 32,
  queryKeepAll: true,
})

interface EmbeddingConfig {
  modelId: string
  name: string
  description: string
  tags: string[]
  base: string
  file: string
  signatures: number[]
  dim: number
  pool: 'mean' | 'cls'
  signaturePrefix: string
  queryPrefix?: string
  documentPrefix?: string
}

function l2Normalize(vector: Float32Array): Float32Array {
  let norm = 0
  for (const value of vector) norm += value * value
  norm = Math.sqrt(norm) || 1
  for (let i = 0; i < vector.length; i++) vector[i] /= norm
  return vector
}

function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i]
  return dot
}

/** Encodes one text through a signature encoder into a single sentence vector. */
async function embedText(text: string, config: EmbeddingConfig, ctx: InferenceContext, prefix = ''): Promise<Float32Array> {
  const tokenizer = await loadHfTokenizer(`${config.base}/tokenizer.json`)
  const ids = tokenizer.encode(prefix + text, { maxLength: config.signatures[config.signatures.length - 1] })
  const size = config.signatures.find((signature) => signature >= ids.length) ?? config.signatures[config.signatures.length - 1]
  const inputIds = new Int32Array(size)
  inputIds.set(ids)
  const mask = new Int32Array(size)
  for (let i = 0; i < ids.length; i++) mask[i] = 1

  const result = await ctx.predict(
    'main',
    { input_ids: ctx.createTensor(inputIds, [1, size]), attention_mask: ctx.createTensor(mask, [1, size]) },
    `${config.signaturePrefix}_${size}`,
  )
  const data = (await Object.values(result)[0].data()) as Float32Array
  if (config.pool === 'cls') return data.subarray(0, config.dim)

  const pooled = new Float32Array(config.dim)
  for (let row = 0; row < ids.length; row++) for (let k = 0; k < config.dim; k++) pooled[k] += data[row * config.dim + k]
  for (let k = 0; k < config.dim; k++) pooled[k] /= ids.length
  return l2Normalize(pooled)
}

function makeEmbeddingAdapter(config: EmbeddingConfig): ModelAdapter {
  return {
    modelId: config.modelId,
    metadata: { name: config.name, description: config.description, modelPath: `${config.base}/${config.file}`, tags: config.tags },
    inputSpecs: [
      { name: 'text_a', dtype: 'string', shape: [], description: 'First text', constraints: { text: true } },
      { name: 'text_b', dtype: 'string', shape: [], description: 'Second text', constraints: { text: true } },
    ],
    outputSpecs: [{ name: 'similarity', dtype: 'float32', shape: [], description: 'Cosine similarity of the two sentence embeddings (-1 to 1)' }],
    prepareInputs: () => ({}),
    parseOutputs: async () => ({}),
    async run(values, ctx) {
      const textA = String(values.text_a ?? '').trim()
      const textB = String(values.text_b ?? '').trim()
      if (!textA || !textB) throw new Error('Provide both texts')
      const vectorA = await embedText(textA, config, ctx, config.queryPrefix)
      const vectorB = await embedText(textB, config, ctx, config.documentPrefix)
      return { similarity: Number(cosine(vectorA, vectorB).toFixed(4)), dims: config.dim }
    },
  }
}

const LFM2_ENCODER_BASE = 'https://huggingface.co/litert-community/LFM2.5-Encoder-230M/resolve/main'
const GRANITE_EMBED_BASE = 'https://huggingface.co/litert-community/granite-embedding-311m-multilingual-r2/resolve/main'

export const lfm2EncoderAdapter = makeEmbeddingAdapter({
  modelId: 'lfm2.5-encoder',
  name: 'LFM2.5-Encoder-230M',
  description: 'Multilingual bidirectional sentence encoder. Mean-pools the token states into a 1024-dim vector and reports the cosine similarity of two texts.',
  tags: ['text', 'embedding'],
  base: LFM2_ENCODER_BASE,
  file: 'LFM2.5-Encoder-230M_wi8fc.tflite',
  signatures: [64, 128, 256, 512],
  dim: 1024,
  pool: 'mean',
  signaturePrefix: 'encode',
})

export const graniteEmbedAdapter = makeEmbeddingAdapter({
  modelId: 'granite-embedding',
  name: 'granite-embedding-311m-multilingual-r2',
  description: 'Multilingual ModernBERT bi-encoder. CLS pooling and L2 normalization are inside the graph; reports the cosine similarity of two texts.',
  tags: ['text', 'embedding'],
  base: GRANITE_EMBED_BASE,
  file: 'granite-embedding-311m-r2_wi8fc.tflite',
  signatures: [64, 128, 256, 512],
  dim: 768,
  pool: 'cls',
  signaturePrefix: 'embed',
})

export const lfm2EmbeddingAdapter = makeEmbeddingAdapter({
  modelId: 'lfm2.5-embedding',
  name: 'LFM2.5-Embedding-350M',
  description: 'Multilingual sentence-embedding model. CLS pooling and L2 normalization are inside the graph; reports the cosine similarity of two texts.',
  tags: ['text', 'embedding'],
  base: 'https://huggingface.co/litert-community/LFM2.5-Embedding-350M/resolve/main',
  file: 'LFM2.5-Embedding-350M_wi8fc.tflite',
  signatures: [64, 128, 256, 512],
  dim: 1024,
  pool: 'cls',
  signaturePrefix: 'embed',
  queryPrefix: 'query: ',
  documentPrefix: 'document: ',
})

export const voyageEmbedAdapter = makeEmbeddingAdapter({
  modelId: 'voyage-4-nano',
  name: 'voyage-4-nano',
  description: 'Multilingual 2048-dim embedding model. Projection, mean pooling and L2 normalization are inside the graph; reports the cosine similarity of two texts.',
  tags: ['text', 'embedding'],
  base: 'https://huggingface.co/litert-community/voyage-4-nano/resolve/main',
  file: 'voyage-4-nano_wi8fc.tflite',
  signatures: [64, 128, 256, 512],
  dim: 2048,
  pool: 'cls',
  signaturePrefix: 'embed',
  queryPrefix: 'Represent the query for retrieving supporting documents: ',
  documentPrefix: 'Represent the document for retrieval: ',
})

export const nemotronEmbedAdapter = makeEmbeddingAdapter({
  modelId: 'nemotron-3-embed',
  name: 'Nemotron-3-Embed-1B',
  description: 'Multilingual 2048-dim embedding model. Mean pooling and L2 normalization are inside the graph; reports the cosine similarity of two texts.',
  tags: ['text', 'embedding'],
  base: 'https://huggingface.co/litert-community/Nemotron-3-Embed-1B/resolve/main',
  file: 'Nemotron-3-Embed-1B_wi8fc.tflite',
  signatures: [64, 128, 256, 512],
  dim: 2048,
  pool: 'cls',
  signaturePrefix: 'embed',
  queryPrefix: 'query: ',
  documentPrefix: 'passage: ',
})

export const harrierEmbedAdapter = makeEmbeddingAdapter({
  modelId: 'harrier-oss',
  name: 'harrier-oss-v1-0.6b',
  description: 'Multilingual 1024-dim embedding model. Last-token pooling and L2 normalization are inside the graph; reports the cosine similarity of two texts.',
  tags: ['text', 'embedding'],
  base: 'https://huggingface.co/litert-community/harrier-oss-v1-0.6b/resolve/main',
  file: 'harrier-oss-v1-0.6b_wi8fc.tflite',
  signatures: [64, 128, 256, 512],
  dim: 1024,
  pool: 'cls',
  signaturePrefix: 'embed',
  queryPrefix: 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ',
})

export const textAdapters: ModelAdapter[] = [
  mxbaiColbertAdapter,
  mlateonAdapter,
  lfmColbertAdapter,
  lfm2EncoderAdapter,
  graniteEmbedAdapter,
  lfm2EmbeddingAdapter,
  voyageEmbedAdapter,
  nemotronEmbedAdapter,
  harrierEmbedAdapter,
]
