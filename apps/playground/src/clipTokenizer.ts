// ponytail: CLIP BPE only (SOT/EOT hardcoded). Generalize post-processing if a second tokenizer family lands.
const SOT = 49406
const EOT = 49407

export const BYTE_TO_UNICODE: string[] = (() => {
  const bs: number[] = []
  for (let i = 33; i <= 126; i++) bs.push(i)
  for (let i = 161; i <= 172; i++) bs.push(i)
  for (let i = 174; i <= 255; i++) bs.push(i)
  const cs = bs.slice()
  let n = 0
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b)
      cs.push(256 + n++)
    }
  }
  const out: string[] = []
  for (let i = 0; i < bs.length; i++) out[bs[i]] = String.fromCharCode(cs[i])
  return out
})()

const CLIP_REGEX = /<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+/gu

export interface ClipTokenizerJson {
  model: { vocab: Record<string, number>; merges?: (string | [string, string])[] }
  added_tokens?: { content: string; id: number }[]
}

export class ClipTokenizer {
  private vocab: Record<string, number>
  private merges = new Map<string, number>()
  private special = new Map<string, number>()
  private cache = new Map<string, number[]>()

  constructor(json: ClipTokenizerJson) {
    this.vocab = json.model.vocab
    ;(json.model.merges ?? []).forEach((merge, rank) => {
      const key = Array.isArray(merge) ? `${merge[0]} ${merge[1]}` : merge
      if (!this.merges.has(key)) this.merges.set(key, rank)
    })
    for (const token of json.added_tokens ?? []) this.special.set(token.content, token.id)
    if (!this.special.has('<|startoftext|>')) this.special.set('<|startoftext|>', SOT)
    if (!this.special.has('<|endoftext|>')) this.special.set('<|endoftext|>', EOT)
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
    const ids = word.map((token) => this.vocab[token] ?? EOT)
    this.cache.set(piece, ids)
    return ids
  }

  encode(text: string, length = 77): number[] {
    const normalized = text.normalize('NFC').replace(/\s+/g, ' ').toLowerCase()
    const ids: number[] = []
    for (const piece of normalized.match(CLIP_REGEX) ?? []) {
      const special = this.special.get(piece)
      if (special !== undefined) {
        ids.push(special)
        continue
      }
      let word = ''
      for (const byte of new TextEncoder().encode(piece)) word += BYTE_TO_UNICODE[byte]
      ids.push(...this.bpe(word))
    }
    const out = [SOT, ...ids.slice(0, length - 2), EOT]
    while (out.length < length) out.push(EOT)
    return out
  }
}

let clipTokenizerPromise: Promise<ClipTokenizer> | null = null

export function loadClipTokenizer(
  url = 'https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/tokenizer.json',
): Promise<ClipTokenizer> {
  if (!clipTokenizerPromise) {
    clipTokenizerPromise = fetch(url)
      .then((response) => {
        if (!response.ok) throw new Error(`CLIP tokenizer fetch failed: ${response.status}`)
        return response.json()
      })
      .then((json) => new ClipTokenizer(json as ClipTokenizerJson))
      .catch((cause) => {
        clipTokenizerPromise = null
        throw cause
      })
  }
  return clipTokenizerPromise
}
