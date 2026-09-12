import type { InferenceContext, ModelAdapter } from './types'
import { loadHfTokenizer, type HfTokenizer } from '../hfTokenizer'

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
  padId?: number
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
  const inputIds = new Int32Array(size).fill(config.padId ?? 0)
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

export const lfm2Encoder350Adapter = makeEmbeddingAdapter({
  modelId: 'lfm2.5-encoder-350m',
  name: 'LFM2.5-Encoder-350M',
  description: 'Multilingual bidirectional sentence encoder. Mean-pools the token states into a 1024-dim vector and reports the cosine similarity of two texts.',
  tags: ['text', 'embedding'],
  base: 'https://huggingface.co/litert-community/LFM2.5-Encoder-350M/resolve/main',
  file: 'LFM2.5-Encoder-350M_wi8fc.tflite',
  signatures: [64, 128, 256, 512],
  dim: 1024,
  pool: 'mean',
  signaturePrefix: 'encode',
})

export const gigaEmbedAdapter = makeEmbeddingAdapter({
  modelId: 'giga-embeddings',
  name: 'Giga-Embeddings-instruct-480M-0826',
  description: 'Russian + English 1024-dim embedding model. Mean pooling and L2 normalization are inside the graph; reports the cosine similarity of two texts.',
  tags: ['text', 'embedding'],
  base: 'https://huggingface.co/litert-community/Giga-Embeddings-instruct-480M-0826/resolve/main',
  file: 'Giga-Embeddings-instruct-480M-0826_wi8fc.tflite',
  signatures: [64, 128, 256, 512],
  dim: 1024,
  pool: 'cls',
  signaturePrefix: 'embed',
  padId: 2,
  queryPrefix: 'Instruct: Given a query, retrieve relevant passages\nQuery: ',
})

const ETTIN_BASE = 'https://huggingface.co/litert-community/ettin-reranker-400m-v1/resolve/main'
const ETTIN_SIGNATURES = [128, 256, 512]
const ETTIN_PAD_ID = 50283

/** Scores one (query, passage) pair through the cross-encoder; returns the raw relevance logit. */
async function rerankerScore(query: string, passage: string, ctx: InferenceContext): Promise<number> {
  const tokenizer = await loadHfTokenizer(`${ETTIN_BASE}/tokenizer.json`)
  const size = smallestSignature(tokenizer.encodePair(query, passage).length, ETTIN_SIGNATURES)
  const ids = tokenizer.encodePair(query, passage, { maxLength: size, length: size, padId: ETTIN_PAD_ID })
  const inputIds = Int32Array.from(ids)
  const attentionMask = Int32Array.from(ids, (id) => (id === ETTIN_PAD_ID ? 0 : 1))
  const result = await ctx.predict(
    'main',
    { input_ids: ctx.createTensor(inputIds, [1, size]), attention_mask: ctx.createTensor(attentionMask, [1, size]) },
    `score_${size}`,
  )
  const data = await Object.values(result)[0].data()
  return data[0]
}

export const ettinRerankerAdapter: ModelAdapter = {
  modelId: 'ettin-reranker',
  metadata: {
    name: 'Ettin Reranker 400M',
    description: 'Cross-encoder reranker. Scores each candidate passage against the query with a raw relevance logit (do not sigmoid).',
    modelPath: `${ETTIN_BASE}/ettin-reranker-400m-v1_wi8fc.tflite`,
    tags: ['text', 'reranker'],
  },
  inputSpecs: [
    { name: 'query', dtype: 'string', shape: [], description: 'Search query', constraints: { text: true } },
    { name: 'passages', dtype: 'string', shape: [], description: 'Candidate passages, one per line', constraints: { text: true } },
  ],
  outputSpecs: [{ name: 'scores', dtype: 'float32', shape: [], description: 'Raw relevance logits, highest first' }],
  prepareInputs() {
    return {}
  },
  async parseOutputs() {
    return {}
  },
  async run(values, ctx) {
    const query = String(values['query'] ?? '').trim()
    const passages = String(values['passages'] ?? '')
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean)
      .slice(0, 20)
    if (!query) throw new Error('Provide a query')
    if (!passages.length) throw new Error('Provide at least one passage')
    const scored = await Promise.all(passages.map(async (passage) => ({ passage, score: await rerankerScore(query, passage, ctx) })))
    scored.sort((a, b) => b.score - a.score)
    return { scores: scored.map(({ passage, score }) => `${score.toFixed(4)}  ${passage}`).join('\n') }
  },
}

const PII_BASE = 'https://huggingface.co/litert-community/LFM2.5-Encoder-350M-PII-Detector/resolve/main'
const PII_SIGNATURES = [128, 512]
const PII_LOGIT_WIDTH = 161
const PII_NUM_LABELS = 109
const PII_SCHEME = ['B', 'I', 'E', 'S']

// label_schema.json types_in_order; id 1 + 4*i + {B:0,I:1,E:2,S:3}, id 0 = O.
const PII_TYPES = [
  'contact.address', 'contact.email', 'contact.ip_address', 'contact.phone', 'contact.postal_code',
  'credential.api_key', 'device.mac_address',
  'financial.bank_account', 'financial.credit_card', 'financial.iban', 'financial.swift_bic',
  'healthcare.condition', 'healthcare.medical_record', 'healthcare.medication',
  'identity.date_of_birth', 'identity.drivers_license', 'identity.national_id', 'identity.passport',
  'identity.person_name', 'identity.ssn',
  'legal.case_number', 'location.gps_coordinates',
  'online.url', 'online.username', 'org.company_name',
  'special.political', 'special.religion',
]

function piiLabel(id: number): string {
  if (id === 0) return 'O'
  return `${PII_SCHEME[(id - 1) & 3]}-${PII_TYPES[(id - 1) >> 2]}`
}

function readableToken(token: string): string {
  return token.replace(/Ċ/g, '\n').replace(/[Ġ▁]/g, ' ')
}

/** Groups a per-token BIOES tag sequence into `<type>  <text>` lines. */
function decodePiiSpans(tags: number[], tokens: string[]): string {
  const spans: { type: string; text: string }[] = []
  let current: { type: string; text: string } | null = null
  const flush = () => {
    if (current) spans.push(current)
    current = null
  }
  for (let i = 0; i < tags.length; i++) {
    const label = piiLabel(tags[i])
    if (label === 'O') {
      flush()
      continue
    }
    const type = label.slice(2)
    const piece = readableToken(tokens[i] ?? '')
    if (label[0] === 'S') {
      flush()
      spans.push({ type, text: piece })
    } else if (label[0] === 'B' || !current || current.type !== type) {
      flush()
      current = { type, text: piece }
    } else if (label[0] === 'E') {
      current.text += piece
      flush()
    } else {
      current.text += piece
    }
  }
  flush()
  return spans.map((span) => `${span.type}  ${span.text.trim()}`).join('\n')
}

export const piiDetectorAdapter: ModelAdapter = {
  modelId: 'lfm2.5-pii',
  metadata: {
    name: 'LFM2.5-Encoder-350M-PII-Detector',
    description: 'Detects ~40 kinds of personal information (emails, phone numbers, names, IDs…) across 16 languages and returns the tagged spans.',
    modelPath: `${PII_BASE}/LFM2.5-Encoder-350M-PII-Detector_wi8fc.tflite`,
    tags: ['text', 'ner', 'privacy'],
  },
  inputSpecs: [{ name: 'text', dtype: 'string', shape: [], description: 'Text to scan for personal information', constraints: { text: true } }],
  outputSpecs: [{ name: 'entities', dtype: 'float32', shape: [], description: 'Detected entities as "<type>  <text>" lines' }],
  prepareInputs() {
    return {}
  },
  async parseOutputs() {
    return {}
  },
  async run(values, ctx) {
    const text = String(values['text'] ?? '').trim()
    if (!text) throw new Error('Provide text to scan')
    const tokenizer = await loadHfTokenizer(`${PII_BASE}/tokenizer.json`)
    const ids = tokenizer.encode(text, { maxLength: PII_SIGNATURES[PII_SIGNATURES.length - 1] })
    const size = smallestSignature(ids.length, PII_SIGNATURES)
    const inputIds = new Int32Array(size)
    inputIds.set(ids)
    const mask = new Int32Array(size)
    for (let i = 0; i < ids.length; i++) mask[i] = 1

    const result = await ctx.predict(
      'main',
      { input_ids: ctx.createTensor(inputIds, [1, size]), attention_mask: ctx.createTensor(mask, [1, size]) },
      `pii_${size}`,
    )
    const logits = (await Object.values(result)[0].data()) as Float32Array
    const tags: number[] = []
    const tokens: string[] = []
    for (let t = 0; t < ids.length; t++) {
      let best = 0
      let bestScore = logits[t * PII_LOGIT_WIDTH]
      for (let c = 1; c < PII_NUM_LABELS; c++) {
        const score = logits[t * PII_LOGIT_WIDTH + c]
        if (score > bestScore) {
          bestScore = score
          best = c
        }
      }
      tags.push(best)
      tokens.push(tokenizer.tokenOf(ids[t]) ?? '')
    }
    return { entities: decodePiiSpans(tags, tokens) || 'No personal information detected' }
  },
}

const LINT_BASE = 'https://huggingface.co/litert-community/LFM2.5-Encoder-350M-Policy-Linter/resolve/main'
const LINT_SIGNATURES = [128, 512]
const LINT_MAX_RULES = 8

/**
 * Token spans of each list item inside the prompt. ponytail: derived from cumulative
 * prefix encodings rather than char offsets — the tokenizer isolates `- ` and
 * `\n` into their own chunks, so no BPE merge crosses an item boundary.
 */
function itemTokenSpans(tokenizer: HfTokenizer, prefix: string, items: string[], header: string): [number, number][] {
  const spans: [number, number][] = []
  let pos = header.length
  for (const item of items) {
    const start = pos + 2
    const end = start + item.length
    spans.push([tokenizer.encode(prefix.slice(0, start)).length, tokenizer.encode(prefix.slice(0, end)).length])
    pos = end + 1
  }
  return spans
}

export const policyLinterAdapter: ModelAdapter = {
  modelId: 'lfm2.5-policy-linter',
  metadata: {
    name: 'LFM2.5-Encoder-350M-Policy-Linter',
    description: 'Zero-shot policy linting: scores every document token against up to 8 plain-language rules in one pass.',
    modelPath: `${LINT_BASE}/LFM2.5-Encoder-350M-Policy-Linter_wi8fc.tflite`,
    tags: ['text', 'policy', 'compliance'],
  },
  inputSpecs: [
    { name: 'text', dtype: 'string', shape: [], description: 'Document to lint', constraints: { text: true } },
    { name: 'rules', dtype: 'string', shape: [], description: 'Policy rules, one per line (up to 8)', constraints: { text: true } },
  ],
  outputSpecs: [{ name: 'violations', dtype: 'float32', shape: [], description: 'Flagged spans per rule' }],
  prepareInputs() {
    return {}
  },
  async parseOutputs() {
    return {}
  },
  async run(values, ctx) {
    const text = String(values['text'] ?? '').trim()
    const rules = String(values['rules'] ?? '')
      .split('\n')
      .map((rule) => rule.trim())
      .filter(Boolean)
      .slice(0, LINT_MAX_RULES)
    if (!text) throw new Error('Provide a document to lint')
    if (!rules.length) throw new Error('Provide at least one rule')

    const tokenizer = await loadHfTokenizer(`${LINT_BASE}/tokenizer.json`)
    const prefix = `Policy:\n${rules.map((rule) => `- ${rule}`).join('\n')}\n\nText:\n`
    const ids = tokenizer.encode(prefix + text)
    if (ids.length > LINT_SIGNATURES[LINT_SIGNATURES.length - 1]) throw new Error('Document and rules exceed 512 tokens')
    const size = smallestSignature(ids.length, LINT_SIGNATURES)

    const rulePool = new Float32Array(LINT_MAX_RULES * size)
    for (const [r, [start, end]] of itemTokenSpans(tokenizer, prefix, rules, 'Policy:\n').entries()) {
      const n = end - start
      if (n <= 0) continue
      for (let i = start; i < end; i++) rulePool[r * size + i] = 1 / n
    }

    const inputIds = new Int32Array(size)
    inputIds.set(ids)
    const mask = new Int32Array(size)
    for (let i = 0; i < ids.length; i++) mask[i] = 1

    const result = await ctx.predict(
      'main',
      {
        input_ids: ctx.createTensor(inputIds, [1, size]),
        attention_mask: ctx.createTensor(mask, [1, size]),
        rule_pool: ctx.createTensor(rulePool, [1, LINT_MAX_RULES, size]),
      },
      `lint_${size}`,
    )
    const scores = (await Object.values(result)[0].data()) as Float32Array

    const docStart = tokenizer.encode(prefix).length
    const lines = rules.map((rule, r) => {
      const flagged: string[] = []
      for (let t = docStart; t < ids.length; t++) {
        if (1 / (1 + Math.exp(-scores[t * LINT_MAX_RULES + r])) > 0.5) flagged.push(readableToken(tokenizer.tokenOf(ids[t]) ?? ''))
      }
      return `rule: ${rule}\n  flagged: ${flagged.join('').trim() || '(none)'}`
    })
    return { violations: lines.join('\n') }
  },
}

const ROUTE_BASE = 'https://huggingface.co/litert-community/LFM2.5-Encoder-350M-Prompt-Router/resolve/main'
const ROUTE_SIGNATURES = [128, 512]
const ROUTE_MAX_LANES = 8

export const promptRouterAdapter: ModelAdapter = {
  modelId: 'lfm2.5-prompt-router',
  metadata: {
    name: 'LFM2.5-Encoder-350M-Prompt-Router',
    description: 'Routes a prompt to one of up to 8 named lanes in a single pass, returning per-lane confidence.',
    modelPath: `${ROUTE_BASE}/LFM2.5-Encoder-350M-Prompt-Router_wi8fc.tflite`,
    tags: ['text', 'router', 'classification'],
  },
  inputSpecs: [
    { name: 'text', dtype: 'string', shape: [], description: 'Prompt to route', constraints: { text: true } },
    { name: 'lanes', dtype: 'string', shape: [], description: 'Routing lanes, one per line (up to 8)', constraints: { text: true } },
  ],
  outputSpecs: [{ name: 'routes', dtype: 'float32', shape: [], description: 'Per-lane confidence, ranked' }],
  prepareInputs() {
    return {}
  },
  async parseOutputs() {
    return {}
  },
  async run(values, ctx) {
    const text = String(values['text'] ?? '').trim()
    const lanes = String(values['lanes'] ?? '')
      .split('\n')
      .map((lane) => lane.trim())
      .filter(Boolean)
      .slice(0, ROUTE_MAX_LANES)
    if (!text) throw new Error('Provide a prompt to route')
    if (!lanes.length) throw new Error('Provide at least one lane')

    const tokenizer = await loadHfTokenizer(`${ROUTE_BASE}/tokenizer.json`)
    const prefix = `Categories:\n${lanes.map((lane) => `- ${lane}`).join('\n')}\n\nText:\n`
    const ids = tokenizer.encode(prefix + text)
    if (ids.length > ROUTE_SIGNATURES[ROUTE_SIGNATURES.length - 1]) throw new Error('Prompt and lanes exceed 512 tokens')
    const size = smallestSignature(ids.length, ROUTE_SIGNATURES)

    const categoryPool = new Float32Array(ROUTE_MAX_LANES * size)
    for (const [r, [start, end]] of itemTokenSpans(tokenizer, prefix, lanes, 'Categories:\n').entries()) {
      const n = end - start
      if (n <= 0) continue
      for (let i = start; i < end; i++) categoryPool[r * size + i] = 1 / n
    }

    const textStart = tokenizer.encode(prefix).length
    const textPool = new Float32Array(size)
    const textCount = ids.length - textStart
    for (let i = textStart; i < ids.length; i++) textPool[i] = 1 / textCount

    const inputIds = new Int32Array(size)
    inputIds.set(ids)
    const mask = new Int32Array(size)
    for (let i = 0; i < ids.length; i++) mask[i] = 1

    const result = await ctx.predict(
      'main',
      {
        input_ids: ctx.createTensor(inputIds, [1, size]),
        attention_mask: ctx.createTensor(mask, [1, size]),
        text_pool: ctx.createTensor(textPool, [1, 1, size]),
        category_pool: ctx.createTensor(categoryPool, [1, ROUTE_MAX_LANES, size]),
      },
      `route_${size}`,
    )
    const logits = (await Object.values(result)[0].data()) as Float32Array

    let max = -Infinity
    for (let r = 0; r < lanes.length; r++) if (logits[r] > max) max = logits[r]
    let sum = 0
    for (let r = 0; r < lanes.length; r++) sum += Math.exp(logits[r] - max)
    const ranked = lanes
      .map((lane, r) => ({ lane, p: Math.exp(logits[r] - max) / sum }))
      .sort((a, b) => b.p - a.p)
    return { routes: ranked.map(({ lane, p }) => `${(p * 100).toFixed(1)}%  ${lane}`).join('\n') }
  },
}

const GEC_BASE = 'https://huggingface.co/litert-community/LFM2.5-Encoder-350M-Spellchecker/resolve/main'
const GEC_SEQ_LEN = 128
const GEC_VOCAB = 64400
const GEC_TAG_WIDTH = 128802
const GEC_MIN_ERROR_PROB = 0.5
const GEC_MAX_PASSES = 3

export const spellcheckerAdapter: ModelAdapter = {
  modelId: 'lfm2.5-spellchecker',
  metadata: {
    name: 'LFM2.5-Encoder-350M-Spellchecker',
    description: 'GECToR-style tagger that corrects spelling and grammar token by token, iterating up to 3 passes.',
    modelPath: `${GEC_BASE}/LFM2.5-Encoder-350M-Spellchecker_wi8fc.tflite`,
    tags: ['text', 'grammar', 'spellcheck'],
  },
  inputSpecs: [{ name: 'text', dtype: 'string', shape: [], description: 'Text to correct', constraints: { text: true } }],
  outputSpecs: [{ name: 'corrected', dtype: 'float32', shape: [], description: 'Corrected text' }],
  prepareInputs() {
    return {}
  },
  async parseOutputs() {
    return {}
  },
  async run(values, ctx) {
    const text = String(values['text'] ?? '').trim()
    if (!text) throw new Error('Provide text to correct')

    const tokenizer = await loadHfTokenizer(`${GEC_BASE}/tokenizer.json`)
    let ids = tokenizer.encode(text)
    if (ids.length > GEC_SEQ_LEN) throw new Error('Text exceeds the 128-token window')

    for (let pass = 0; pass < GEC_MAX_PASSES; pass++) {
      const inputIds = new Int32Array(GEC_SEQ_LEN)
      inputIds.set(ids)
      const mask = new Int32Array(GEC_SEQ_LEN)
      for (let i = 0; i < ids.length; i++) mask[i] = 1

      const result = await ctx.predict(
        'main',
        {
          input_ids: ctx.createTensor(inputIds, [1, GEC_SEQ_LEN]),
          attention_mask: ctx.createTensor(mask, [1, GEC_SEQ_LEN]),
        },
        'gec_128',
      )
      const labelLogits = (await Object.values(result)[0].data()) as Float32Array
      const detectLogits = (await Object.values(result)[1].data()) as Float32Array

      const edits: [number, number][] = []
      for (let t = 0; t < ids.length; t++) {
        const d0 = detectLogits[t * 2]
        const d1 = detectLogits[t * 2 + 1]
        const max = Math.max(d0, d1)
        const errorProb = Math.exp(d1 - max) / (Math.exp(d0 - max) + Math.exp(d1 - max))
        if (errorProb < GEC_MIN_ERROR_PROB) continue

        let tag = 0
        let best = -Infinity
        for (let k = 0; k < GEC_TAG_WIDTH; k++) {
          const v = labelLogits[t * GEC_TAG_WIDTH + k]
          if (v > best) {
            best = v
            tag = k
          }
        }
        if (tag === 0) continue
        edits.push([t, tag])
      }
      if (!edits.length) break

      for (let e = edits.length - 1; e >= 0; e--) {
        const [t, tag] = edits[e]
        if (tag === 1) ids.splice(t, 1)
        else if (tag < 2 + GEC_VOCAB) ids[t] = tag - 2
        else ids.splice(t + 1, 0, tag - 2 - GEC_VOCAB)
      }
    }
    return { corrected: tokenizer.decode(ids).trim() }
  },
}

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
  lfm2Encoder350Adapter,
  gigaEmbedAdapter,
  ettinRerankerAdapter,
  piiDetectorAdapter,
  policyLinterAdapter,
  promptRouterAdapter,
  spellcheckerAdapter,
]
