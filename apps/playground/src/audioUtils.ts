const MASK_NEG = -1e9

/** Linear-interp resample of per-channel PCM to mono at `targetRate` (matches the reference np.interp). */
export function resampleToMono(channels: Float32Array[], sampleRate: number, targetRate = 16000): Float32Array {
  const frames = channels[0]?.length ?? 0
  if (!frames) return new Float32Array(0)
  const out = new Float32Array(sampleRate === targetRate ? frames : Math.max(1, Math.round(frames / (sampleRate / targetRate))))
  const ratio = sampleRate / targetRate
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

/** In-place iterative radix-2 FFT; `re`/`im` must share a power-of-two length. */
export function fft(re: Float32Array, im: Float32Array): void {
  const n = re.length
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) {
      const tr = re[i]; re[i] = re[j]; re[j] = tr
      const ti = im[i]; im[i] = im[j]; im[j] = ti
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const angle = (-2 * Math.PI) / len
    const wr = Math.cos(angle)
    const wi = Math.sin(angle)
    for (let i = 0; i < n; i += len) {
      let cr = 1
      let ci = 0
      for (let k = 0; k < len / 2; k++) {
        const a = i + k
        const b = a + len / 2
        const vr = re[b] * cr - im[b] * ci
        const vi = re[b] * ci + im[b] * cr
        re[b] = re[a] - vr; im[b] = im[a] - vi
        re[a] += vr; im[a] += vi
        const nr = cr * wr - ci * wi
        ci = cr * wi + ci * wr
        cr = nr
      }
    }
  }
}

/**
 * torchlibrosa-exact log-mel for PANNs: reflect-pad centre, periodic Hann, 1024-pt rFFT,
 * power, mel matmul, `10*log10(max(mel,1e-10))`. Returns a frame-major [frames, nMels] buffer.
 */
export function logMelSpectrogram(
  audio: Float32Array,
  melBasis: Float32Array,
  nMels: number,
  nFft = 1024,
  hop = 320,
  sampleRate = 32000,
): Float32Array {
  const clip = Math.floor(sampleRate * 10)
  const pad = nFft / 2
  const padded = new Float32Array(clip + nFft)
  for (let i = 0; i < padded.length; i++) {
    let src = i - pad
    if (src < 0) src = -src
    else if (src >= clip) src = 2 * clip - src - 2
    padded[i] = src < audio.length ? audio[src] : 0
  }

  const window = new Float32Array(nFft)
  for (let n = 0; n < nFft; n++) window[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / nFft)

  const bins = nFft / 2 + 1
  const frames = 1 + Math.floor(clip / hop)
  const out = new Float32Array(frames * nMels)
  const re = new Float32Array(nFft)
  const im = new Float32Array(nFft)

  for (let t = 0; t < frames; t++) {
    for (let n = 0; n < nFft; n++) {
      re[n] = padded[t * hop + n] * window[n]
      im[n] = 0
    }
    fft(re, im)
    for (let m = 0; m < nMels; m++) {
      let sum = 0
      const row = m * bins
      for (let k = 0; k < bins; k++) sum += (re[k] * re[k] + im[k] * im[k]) * melBasis[row + k]
      out[t * nMels + m] = 10 * Math.log10(Math.max(sum, 1e-10))
    }
  }
  return out
}
