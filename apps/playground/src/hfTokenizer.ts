// ponytail: minimal HuggingFace tokenizer.json reader covering the ByteLevel-BPE + TemplateProcessing
// subset used by the playground's text encoders (mxbai ColBERT first). Extend the switch when a new
// normalizer/pre-tokenizer node type lands; unknown nodes throw so failures are loud, not silent.
import { BYTE_TO_UNICODE } from './clipTokenizer'

const BYTE_LEVEL_REGEX = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu

const UNICODE_TO_BYTE = new Map<string, number>()
BYTE_TO_UNICODE.forEach((ch, byte) => UNICODE_TO_BYTE.set(ch, byte))

export interface HfTokenizerJson {
  normalizer?: HfNode | null
  pre_tokenizer?: HfNode | null
  model: { type: string; vocab: Record<string, number>; merges?: (string | [string, string])[]; unk_token?: string | null }
  post_processor?: HfNode | null
  added_tokens?: { content: string; id: number }[]
}

type HfNode = { type: string; [key: string]: any }

export interface HfEncodeOptions {
  /** Total token budget including template specials (HF `truncation max_length`). */
  maxLength?: number
  /** If set, pad the result to exactly this many tokens. */
  length?: number
  padId?: number
}

export class HfTokenizer {
  private vocab: Record<string, number>
  private inverse = new Map<number, string>()
  private merges = new Map<string, number>()
  private added = new Map<string, number>()
  private cache = new Map<string, number[]>()
  private normalizer: HfNode | null
  private preTokenizer: HfNode | null
  private byteLevel: boolean
  private prefixSpace: boolean
  private template: HfNode | null
  private pairTemplate: HfNode | null
  private unkId: number

  constructor(json: HfTokenizerJson) {
    this.vocab = json.model.vocab
    for (const [token, id] of Object.entries(this.vocab)) if (!this.inverse.has(id)) this.inverse.set(id, token)
    ;(json.model.merges ?? []).forEach((merge, rank) => {
      const key = Array.isArray(merge) ? `${merge[0]} ${merge[1]}` : merge
      if (!this.merges.has(key)) this.merges.set(key, rank)
    })
    for (const token of json.added_tokens ?? []) this.added.set(token.content, token.id)
    this.normalizer = json.normalizer ?? null
    this.preTokenizer = json.pre_tokenizer ?? null
    this.byteLevel = usesByteLevel(this.preTokenizer)
    const info = postProcessorInfo(json.post_processor ?? null)
    this.prefixSpace = info.prefixSpace
    this.template = info.template
    this.pairTemplate = info.pairTemplate
    const unk =
      json.post_processor?.special_tokens?.['[UNK]']?.ids?.[0] ??
      (json.model.unk_token ? this.vocab[json.model.unk_token] : undefined) ??
      this.added.get('<unk>')
    this.unkId = unk ?? 0
  }

  /** Inverse lookup for skiplist filtering; falls back to the added-token content. */
  tokenOf(id: number): string | undefined {
    return this.inverse.get(id) ?? [...this.added.entries()].find(([, value]) => value === id)?.[0]
  }

  /** Decode ids back to text (inverse of the ByteLevel mapping when active). Specials are dropped. */
  decode(ids: number[]): string {
    const addedIds = new Set(this.added.values())
    let out = ''
    const bytes: number[] = []
    const flush = () => {
      if (!bytes.length) return
      out += new TextDecoder().decode(Uint8Array.from(bytes))
      bytes.length = 0
    }
    for (const id of ids) {
      if (addedIds.has(id)) continue
      const token = this.tokenOf(id)
      if (!token) continue
      if (this.byteLevel) {
        for (const ch of token) {
          const byte = UNICODE_TO_BYTE.get(ch)
          if (byte !== undefined) bytes.push(byte)
        }
      } else {
        out += token.replace(/[Ġ▁]/g, ' ')
      }
    }
    flush()
    return out
  }

  private normalize(text: string): string {
    return applyNormalizer(this.normalizer, text)
  }

  private preTokenize(text: string): string[] {
    return applyPreTokenizer(this.preTokenizer, text)
  }

  /** ponytail: O(n²) per piece; fine for prompt-length text. */
  private bpe(piece: string): number[] {
    const cached = this.cache.get(piece)
    if (cached) return cached
    let word = piece.split('')
    while (word.length > 1) {
      let bestRank = Infinity
      let bestIdx = -1
      for (let i = 0; i < word.length - 1; i++) {
        const rank = this.merges.get(`${word[i]} ${word[i + 1]}`)
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank
          bestIdx = i
        }
      }
      if (bestIdx === -1) break
      const [a, b] = [word[bestIdx], word[bestIdx + 1]]
      const merged: string[] = []
      for (let i = 0; i < word.length; i++) {
        if (i < word.length - 1 && word[i] === a && word[i + 1] === b) {
          merged.push(a + b)
          i++
        } else {
          merged.push(word[i])
        }
      }
      word = merged
    }
    const ids = word.map((token) => this.vocab[token] ?? this.unkId)
    this.cache.set(piece, ids)
    return ids
  }

  /** Body token ids with no template specials. */
  private bodyIds(text: string): number[] {
    const ids: number[] = []
    for (const piece of this.preTokenize(this.normalize(text))) {
      const special = this.added.get(piece)
      if (special !== undefined) {
        ids.push(special)
        continue
      }
      let word = piece
      if (this.byteLevel) {
        word = ''
        for (const byte of new TextEncoder().encode(piece)) word += BYTE_TO_UNICODE[byte]
      }
      ids.push(...this.bpe(word))
    }
    return ids
  }

  encode(text: string, options: HfEncodeOptions = {}): number[] {
    let ids = this.bodyIds(this.prefixSpace ? ` ${text}` : text)
    if (options.maxLength !== undefined) {
      const specials = countSpecials(this.template?.single)
      ids = ids.slice(0, Math.max(0, options.maxLength - specials))
    }
    const out = applyPostProcessor(this.template, ids)
    if (options.length !== undefined) {
      while (out.length < options.length) out.push(options.padId ?? 0)
    }
    return out
  }

  /** Encodes a text pair through the tokenizer's `pair` template (e.g. `[CLS] A [SEP] B [SEP]`). */
  encodePair(a: string, b: string, options: HfEncodeOptions = {}): number[] {
    let aIds = this.bodyIds(this.prefixSpace ? ` ${a}` : a)
    let bIds = this.bodyIds(this.prefixSpace ? ` ${b}` : b)
    if (options.maxLength !== undefined) {
      const entries = this.pairTemplate ? this.pairTemplate.pair : this.template?.single
      const budget = Math.max(0, options.maxLength - countSpecials(entries))
      while (aIds.length + bIds.length > budget) {
        if (aIds.length >= bIds.length) aIds = aIds.slice(0, -1)
        else bIds = bIds.slice(0, -1)
      }
    }
    const out = this.pairTemplate ? applyPairTemplate(this.pairTemplate, aIds, bIds) : applyPostProcessor(this.template, [...aIds, ...bIds])
    if (options.length !== undefined) {
      while (out.length < options.length) out.push(options.padId ?? 0)
    }
    return out
  }
}

function applyNormalizer(node: HfNode | null, text: string): string {
  if (!node) return text
  switch (node.type) {
    case 'NFC':
      return text.normalize('NFC')
    case 'NFD':
      return text.normalize('NFD')
    case 'Lowercase':
      return text.toLowerCase()
    case 'Replace': {
      const pattern = node.pattern
      if (typeof pattern === 'string') return text.split(pattern).join(node.content)
      if (pattern?.String !== undefined) return text.split(pattern.String).join(node.content)
      return text.replace(new RegExp(pattern.Regex, 'gu'), node.content)
    }
    case 'Sequence':
      return (node.normalizers as HfNode[]).reduce((acc, n) => applyNormalizer(n, acc), text)
    default:
      throw new Error(`hfTokenizer: unsupported normalizer "${node.type}"`)
  }
}

function applyPreTokenizer(node: HfNode | null, text: string): string[] {
  if (!node) return [text]
  switch (node.type) {
    case 'ByteLevel': {
      // Byte mapping happens once in `bodyIds`; this node only splits (and optionally prefixes a space).
      const source = node.add_prefix_space ? ` ${text}` : text
      return node.use_regex ? (source.match(BYTE_LEVEL_REGEX) ?? []) : [source]
    }
    case 'Sequence':
      return (node.pretokenizers as HfNode[]).reduce((acc, n) => acc.flatMap((piece) => applyPreTokenizer(n, piece)), [text])
    case 'Split': {
      const behavior = node.behavior ?? 'Removed'
      const pattern = node.pattern
      if (pattern?.String !== undefined) {
        const parts = text.split(pattern.String)
        if (node.invert) return [pattern.String]
        if (behavior === 'Removed') return parts.filter((part) => part.length > 0)
        if (behavior === 'MergedWithPrevious') return parts.map((part, i) => (i < parts.length - 1 ? part + pattern.String : part)).filter((part) => part.length > 0)
        if (behavior === 'Isolated') {
          const out: string[] = []
          parts.forEach((part, i) => {
            if (part) out.push(part)
            if (i < parts.length - 1) out.push(pattern.String)
          })
          return out
        }
        throw new Error(`hfTokenizer: unsupported Split behavior "${behavior}"`)
      }
      // ponytail: JS has no inline (?i:...) groups — hoist the flag globally; safe because only the
      // contraction alternative is case-sensitive.
      const flags = /\(\?i:/.test(pattern.Regex) ? 'giu' : 'gu'
      const regex = new RegExp(pattern.Regex.replace(/\(\?i:/g, '(?:'), flags)
      const matches = [...text.matchAll(regex)]
      if (node.invert) return matches.map((m) => m[0])
      const out: string[] = []
      let last = 0
      for (const match of matches) {
        const gap = text.slice(last, match.index)
        if (gap) out.push(gap)
        if (behavior === 'Isolated') out.push(match[0])
        else if (behavior === 'MergedWithPrevious') {
          if (out.length) out[out.length - 1] += match[0]
          else out.push(match[0])
        } else if (behavior !== 'Removed') throw new Error(`hfTokenizer: unsupported Split behavior "${behavior}"`)
        last = match.index + match[0].length
      }
      const tail = text.slice(last)
      if (tail) out.push(tail)
      return out
    }
    default:
      throw new Error(`hfTokenizer: unsupported pre_tokenizer "${node.type}"`)
  }
}

function usesByteLevel(node: HfNode | null): boolean {
  if (!node) return false
  if (node.type === 'ByteLevel') return true
  if (node.type === 'Sequence') return (node.pretokenizers as HfNode[]).some(usesByteLevel)
  return false
}

/** Flattens a post_processor into the TemplateProcessing node plus any ByteLevel prefix-space request. */
function postProcessorInfo(node: HfNode | null): { prefixSpace: boolean; template: HfNode | null; pairTemplate: HfNode | null } {
  if (!node) return { prefixSpace: false, template: null, pairTemplate: null }
  if (node.type === 'ByteLevel') return { prefixSpace: Boolean(node.add_prefix_space), template: null, pairTemplate: null }
  if (node.type === 'TemplateProcessing') return { prefixSpace: false, template: node, pairTemplate: node.pair ? node : null }
  if (node.type === 'Sequence') {
    let prefixSpace = false
    let template: HfNode | null = null
    let pairTemplate: HfNode | null = null
    for (const child of node.processors as HfNode[]) {
      const info = postProcessorInfo(child)
      prefixSpace ||= info.prefixSpace
      template = info.template ?? template
      pairTemplate = info.pairTemplate ?? pairTemplate
    }
    return { prefixSpace, template, pairTemplate }
  }
  throw new Error(`hfTokenizer: unsupported post_processor "${node.type}"`)
}

function countSpecials(entries: HfNode[] | undefined): number {
  return (entries ?? []).filter((entry) => entry.SpecialToken).length
}

function applyPostProcessor(node: HfNode | null, ids: number[]): number[] {
  if (!node) return ids
  if (node.type !== 'TemplateProcessing') throw new Error(`hfTokenizer: unsupported post_processor "${node.type}"`)
  const specials = (node.special_tokens ?? {}) as Record<string, { ids: number[] }>
  const out: number[] = []
  for (const entry of node.single as HfNode[]) {
    if (entry.SpecialToken) out.push(...(specials[entry.SpecialToken.id]?.ids ?? []))
    else if (entry.Sequence) out.push(...ids)
  }
  return out
}

function applyPairTemplate(node: HfNode, aIds: number[], bIds: number[]): number[] {
  const specials = (node.special_tokens ?? {}) as Record<string, { ids: number[] }>
  const out: number[] = []
  for (const entry of node.pair as HfNode[]) {
    if (entry.SpecialToken) out.push(...(specials[entry.SpecialToken.id]?.ids ?? []))
    else if (entry.Sequence) out.push(...(entry.Sequence.id === 'B' ? bIds : aIds))
  }
  return out
}

let tokenizerPromises = new Map<string, Promise<HfTokenizer>>()

export function loadHfTokenizer(url: string): Promise<HfTokenizer> {
  let promise = tokenizerPromises.get(url)
  if (!promise) {
    promise = fetch(url)
      .then((response) => {
        if (!response.ok) throw new Error(`tokenizer fetch failed (${response.status}): ${url}`)
        return response.json()
      })
      .then((json) => new HfTokenizer(json as HfTokenizerJson))
      .catch((cause) => {
        tokenizerPromises.delete(url)
        throw cause
      })
    tokenizerPromises.set(url, promise)
  }
  return promise
}
