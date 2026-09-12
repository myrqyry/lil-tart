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

/** SentencePiece Unigram decode: join pieces, `▁`→space, drop `<...>` specials, strip. */
export function decodeUnigram(ids: number[], pieces: string[]): string {
  let text = ''
  for (const id of ids) {
    const piece = pieces[id]
    if (!piece || (piece.startsWith('<') && piece.endsWith('>'))) continue
    text += piece.replace(/▁/g, ' ')
  }
  return text.trim()
}

/** Split audio into fixed-size windows (last one may be shorter; caller zero-pads). */
export function windowsOf(audio: Float32Array, size: number): Float32Array[] {
  if (!audio.length) return [new Float32Array(0)]
  const windows: Float32Array[] = []
  for (let i = 0; i < audio.length; i += size) windows.push(audio.subarray(i, i + size))
  return windows
}

/** 16-bit mono PCM WAV container for playback of generated audio. */
export function encodeWav(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const bitsPerSample = 16
  const byteRate = (sampleRate * bitsPerSample) / 8
  const dataSize = samples.length * 2
  const buf = new ArrayBuffer(44 + dataSize)
  const v = new DataView(buf)
  const w = (off: number, str: string) => {
    for (let i = 0; i < str.length; i++) v.setUint8(off + i, str.charCodeAt(i))
  }
  w(0, 'RIFF'); v.setUint32(4, 36 + dataSize, true); w(8, 'WAVE')
  w(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true)
  v.setUint16(22, 1, true); v.setUint32(24, sampleRate, true)
  v.setUint32(28, byteRate, true); v.setUint16(32, 2, true)
  v.setUint16(34, bitsPerSample, true)
  w(36, 'data'); v.setUint32(40, dataSize, true)
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }
  return buf
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
 * Raw power mel spectrogram (no dB): reflect-pad centre, periodic Hann, rFFT, power, mel matmul.
 * `winLength` shorter than `nFft` is zero-padded centred (torch.stft); `sampleCount` is the
 * reflect-padded clip length. Frame-major [frames, nMels].
 */
export function melSpectrogram(
  audio: Float32Array,
  melBasis: Float32Array,
  nMels: number,
  nFft = 1024,
  hop = 320,
  sampleRate = 32000,
  winLength = nFft,
  sampleCount = Math.floor(sampleRate * 10),
): Float32Array {
  const clip = sampleCount
  const pad = nFft / 2
  const padded = new Float32Array(clip + nFft)
  for (let i = 0; i < padded.length; i++) {
    let src = i - pad
    if (src < 0) src = -src
    else if (src >= clip) src = 2 * clip - src - 2
    padded[i] = src < audio.length ? audio[src] : 0
  }

  const window = new Float32Array(nFft)
  const offset = (nFft - winLength) >> 1
  for (let n = 0; n < winLength; n++) window[offset + n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / winLength)

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
      out[t * nMels + m] = sum
    }
  }
  return out
}

/** torchlibrosa-exact log-mel for PANNs: `10*log10(max(mel,1e-10))`. Frame-major [frames, nMels]. */
export function logMelSpectrogram(
  audio: Float32Array,
  melBasis: Float32Array,
  nMels: number,
  nFft = 1024,
  hop = 320,
  sampleRate = 32000,
): Float32Array {
  const mel = melSpectrogram(audio, melBasis, nMels, nFft, hop, sampleRate)
  for (let i = 0; i < mel.length; i++) mel[i] = 10 * Math.log10(Math.max(mel[i], 1e-10))
  return mel
}

/** torchaudio `melscale_fbanks` HTK triangular filterbank, norm=None. Returns [nMels, nFft/2+1]. */
export function melFilterbank(
  sampleRate: number,
  nFft: number,
  nMels: number,
  fMin = 0,
  fMax = sampleRate / 2,
): Float32Array {
  const toMel = (f: number) => 2595 * Math.log10(1 + f / 700)
  const toHz = (m: number) => 700 * (10 ** (m / 2595) - 1)
  const melMin = toMel(fMin)
  const melMax = toMel(fMax)
  const points = new Float32Array(nMels + 2)
  for (let i = 0; i < points.length; i++) points[i] = toHz(melMin + ((melMax - melMin) * i) / (nMels + 1))

  const bins = nFft / 2 + 1
  const basis = new Float32Array(nMels * bins)
  for (let m = 0; m < nMels; m++) {
    const left = points[m]
    const center = points[m + 1]
    const right = points[m + 2]
    for (let k = 0; k < bins; k++) {
      const f = (k * sampleRate) / nFft
      let weight = 0
      if (f >= left && f <= center) weight = (f - left) / (center - left)
      else if (f > center && f <= right) weight = (right - f) / (right - center)
      basis[m * bins + k] = weight
    }
  }
  return basis
}

/** librosa default (slaney) mel filterbank — used by the OpenAI Whisper front-end. */
export function melFilterbankSlaney(
  sampleRate: number,
  nFft: number,
  nMels: number,
  fMin = 0,
  fMax = sampleRate / 2,
): Float32Array {
  const fSp = 200
  const minLogHz = 1000
  const minLogMel = (minLogHz - 0) / fSp
  const logStep = Math.log(6.4) / 27
  const toMel = (f: number) => (f < minLogHz ? (f - 0) / fSp : minLogMel + Math.log(f / minLogHz) / logStep)
  const toHz = (m: number) => (m < minLogMel ? fSp * m : minLogHz * Math.exp(logStep * (m - minLogMel)))

  const melMin = toMel(fMin)
  const melMax = toMel(fMax)
  const hz = new Float64Array(nMels + 2)
  for (let i = 0; i < hz.length; i++) hz[i] = toHz(melMin + ((melMax - melMin) * i) / (nMels + 1))
  const fdiff = new Float64Array(nMels + 1)
  for (let i = 0; i < fdiff.length; i++) fdiff[i] = hz[i + 1] - hz[i]

  const bins = nFft / 2 + 1
  const basis = new Float32Array(nMels * bins)
  for (let m = 0; m < nMels; m++) {
    const norm = 2 / (hz[m + 2] - hz[m])
    for (let k = 0; k < bins; k++) {
      const f = (k * sampleRate) / nFft
      const lower = (f - hz[m]) / fdiff[m]
      const upper = (hz[m + 2] - f) / fdiff[m + 1]
      basis[m * bins + k] = Math.max(0, Math.min(lower, upper)) * norm
    }
  }
  return basis
}

/** torchaudio `compute_deltas` over a [rows, cols] buffer (time along cols), replicate padding. */
export function computeDeltas(input: Float32Array, rows: number, cols: number, winLength = 3): Float32Array {
  const k = (winLength - 1) >> 1
  let denominator = 0
  for (let i = 1; i <= k; i++) denominator += 2 * i * i
  const out = new Float32Array(rows * cols)
  for (let r = 0; r < rows; r++) {
    const row = r * cols
    for (let t = 0; t < cols; t++) {
      let sum = 0
      for (let i = 1; i <= k; i++) {
        const before = input[row + Math.max(0, t - i)]
        const after = input[row + Math.min(cols - 1, t + i)]
        sum += i * (after - before)
      }
      out[row + t] = sum / denominator
    }
  }
  return out
}
