// ponytail: minimal HuggingFace tokenizer.json reader covering the ByteLevel-BPE + TemplateProcessing
// subset used by the playground's text encoders (mxbai ColBERT first). Extend the switch when a new
// normalizer/pre-tokenizer node type lands; unknown nodes throw so failures are loud, not silent.
import { BYTE_TO_UNICODE } from './clipTokenizer'

const BYTE_LEVEL_REGEX = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu

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
  private postProcessor: HfNode | null
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
    this.postProcessor = json.post_processor ?? null
    const unk = json.post_processor?.special_tokens?.['[UNK]']?.ids?.[0] ?? (json.model.unk_token ? this.vocab[json.model.unk_token] : undefined)
    this.unkId = unk ?? 0
  }

  /** Inverse lookup for skiplist filtering; falls back to the added-token content. */
  tokenOf(id: number): string | undefined {
    return this.inverse.get(id) ?? [...this.added.entries()].find(([, value]) => value === id)?.[0]
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
      let word = ''
      for (const byte of new TextEncoder().encode(piece)) word += BYTE_TO_UNICODE[byte]
      ids.push(...this.bpe(word))
    }
    return ids
  }

  encode(text: string, options: HfEncodeOptions = {}): number[] {
    let ids = this.bodyIds(text)
    if (options.maxLength !== undefined) {
      const specials = countTemplateSpecials(this.postProcessor)
      ids = ids.slice(0, Math.max(0, options.maxLength - specials))
    }
    const out = applyPostProcessor(this.postProcessor, ids)
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
    case 'Replace':
      return text.replace(new RegExp(node.pattern, 'gu'), node.content)
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
    default:
      throw new Error(`hfTokenizer: unsupported pre_tokenizer "${node.type}"`)
  }
}

function countTemplateSpecials(node: HfNode | null): number {
  if (!node || node.type !== 'TemplateProcessing') return 0
  return (node.single as HfNode[]).filter((entry) => entry.SpecialToken).length
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
