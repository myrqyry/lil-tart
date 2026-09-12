import type { InferenceContext, ModelAdapter } from './types'
import { loadHfTokenizer } from '../hfTokenizer'

const MXBAI_BASE = 'https://huggingface.co/litert-community/mxbai-edge-colbert-v0-32m/resolve/main'
const MXBAI_MODEL = `${MXBAI_BASE}/mxbai-edge-colbert-v0-32m_fp16.tflite`
const MXBAI_TOKENIZER = `${MXBAI_BASE}/tokenizer.json`

const Q_ID = 50368
const D_ID = 50369
const PAD_ID = 50284
const SIGNATURES = [48, 128, 256, 512]
const QUERY_MAX = 47
const DOC_MAX = 511
const DIM = 64

// ColBERT skiplist: 32 ASCII punctuation tokens contribute no document signal.
const SKIPLIST = new Set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'.split(''))

function smallestSignature(length: number): number {
  return SIGNATURES.find((size) => size >= length) ?? SIGNATURES[SIGNATURES.length - 1]
}

/** Runs one ColBERT encode signature, returning the L2-normalized vectors for the real tokens. */
async function encodeVectors(text: string, marker: number, maxLength: number, ctx: InferenceContext): Promise<Float32Array[]> {
  const tokenizer = await loadHfTokenizer(MXBAI_TOKENIZER)
  const base = tokenizer.encode(text.toLowerCase(), { maxLength })
  const ids = [base[0], marker, ...base.slice(1)]
  const size = smallestSignature(ids.length)
  const inputIds = new Int32Array(size).fill(PAD_ID)
  inputIds.set(ids)
  const mask = new Int32Array(size)
  for (let i = 0; i < ids.length; i++) mask[i] = 1

  const result = await ctx.predict(
    'main',
    { input_ids: ctx.createTensor(inputIds, [1, size]), attention_mask: ctx.createTensor(mask, [1, size]) },
    `encode_${size}`,
  )
  const data = (await Object.values(result)[0].data()) as Float32Array
  const vectors: Float32Array[] = []
  for (let row = 0; row < ids.length; row++) {
    const vector = data.subarray(row * DIM, (row + 1) * DIM)
    const token = tokenizer.tokenOf(ids[row])
    if (token !== undefined && token.length === 1 && SKIPLIST.has(token)) continue
    vectors.push(vector)
  }
  return vectors
}

function maxSim(query: Float32Array[], document: Float32Array[]): number {
  let total = 0
  for (const q of query) {
    let best = -Infinity
    for (const d of document) {
      let dot = 0
      for (let k = 0; k < DIM; k++) dot += q[k] * d[k]
      if (dot > best) best = dot
    }
    total += best
  }
  return total
}

export const mxbaiColbertAdapter: ModelAdapter = {
  modelId: 'mxbai-colbert',
  metadata: {
    name: 'mxbai-edge-colbert-v0-32m',
    description: 'Late-interaction (ColBERT) text reranker. Scores how relevant a document is to a query via MaxSim over per-token embeddings.',
    modelPath: MXBAI_MODEL,
    tags: ['text', 'retrieval', 'colbert'],
  },
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
    const queryVectors = await encodeVectors(query, Q_ID, QUERY_MAX, ctx)
    const documentVectors = await encodeVectors(document, D_ID, DOC_MAX, ctx)
    return { score: Number(maxSim(queryVectors, documentVectors).toFixed(4)) }
  },
}

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
async function embedText(text: string, config: EmbeddingConfig, ctx: InferenceContext): Promise<Float32Array> {
  const tokenizer = await loadHfTokenizer(`${config.base}/tokenizer.json`)
  const ids = tokenizer.encode(text, { maxLength: config.signatures[config.signatures.length - 1] })
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
      const vectorA = await embedText(textA, config, ctx)
      const vectorB = await embedText(textB, config, ctx)
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

export const textAdapters: ModelAdapter[] = [mxbaiColbertAdapter, lfm2EncoderAdapter, graniteEmbedAdapter]
