const TARGET_RATE = 16000
const MASK_NEG = -1e9

/** Linear-interp resample of per-channel PCM to 16 kHz mono (matches the reference np.interp). */
export function resampleToMono16k(channels: Float32Array[], sampleRate: number): Float32Array {
  const frames = channels[0]?.length ?? 0
  if (!frames) return new Float32Array(0)
  const out = new Float32Array(sampleRate === TARGET_RATE ? frames : Math.max(1, Math.round(frames / (sampleRate / TARGET_RATE))))
  const ratio = sampleRate / TARGET_RATE
  for (let i = 0; i < out.length; i++) {
    const src = i * ratio
    const i0 = Math.min(Math.floor(src), frames - 1)
    const i1 = Math.min(i0 + 1, frames - 1)
    const frac = src - i0
    let sum = 0
    for (const ch of channels) sum += ch[i0] * (1 - frac) + ch[i1] * frac
    out[i] = sum / channels.length
  }
  return out
}

/** Additive causal attention mask [size,size]: 0 on/below diagonal, -1e9 above. */
export function makeCausalMask(size: number): Float32Array {
  const mask = new Float32Array(size * size)
  for (let row = 0; row < size; row++) {
    for (let col = row + 1; col < size; col++) mask[row * size + col] = MASK_NEG
  }
  return mask
}

/** SentencePiece decode: `<0xXX>` byte-fallback + `▁`→space, then UTF-8 decode and strip leading space. */
export function decodeSentencePiece(ids: number[], vocab: string[]): string {
  const bytes: number[] = []
  for (const id of ids) {
    const token = vocab[id]
    if (token === undefined) continue
    const hex = /^<0x([0-9a-fA-F]{2})>$/.exec(token)
    if (hex) {
      bytes.push(parseInt(hex[1], 16))
      continue
    }
    for (const ch of token.replace(/▁/g, ' ')) {
      const cp = ch.codePointAt(0) as number
      if (cp < 0x80) bytes.push(cp)
      else if (cp < 0x800) bytes.push(0xc0 | (cp >> 6), 0x80 | (cp & 0x3f))
      else if (cp < 0x10000) bytes.push(0xe0 | (cp >> 12), 0x80 | ((cp >> 6) & 0x3f), 0x80 | (cp & 0x3f))
      else bytes.push(0xf0 | (cp >> 18), 0x80 | ((cp >> 12) & 0x3f), 0x80 | ((cp >> 6) & 0x3f), 0x80 | (cp & 0x3f))
    }
  }
  return new TextDecoder().decode(new Uint8Array(bytes)).replace(/^\s+/, '')
}

/** Split audio into fixed-size windows (last one may be shorter; caller zero-pads). */
export function windowsOf(audio: Float32Array, size: number): Float32Array[] {
  if (!audio.length) return [new Float32Array(0)]
  const windows: Float32Array[] = []
  for (let i = 0; i < audio.length; i += size) windows.push(audio.subarray(i, i + size))
  return windows
}
