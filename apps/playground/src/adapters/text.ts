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

export const textAdapters: ModelAdapter[] = [mxbaiColbertAdapter]
