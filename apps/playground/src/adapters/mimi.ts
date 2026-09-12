import type { InferenceContext, ModelAdapter } from './types'
import { flatten } from './util'

const MIMI_BASE = 'https://huggingface.co/litert-community/Mimi/resolve/main'

// ponytail: the shipped graphs are built for one 2-second clip. Other durations need a re-conversion.
const MIMI_RATE = 24000
const MIMI_SAMPLES = 48000
const MIMI_HIDDEN = 512
const MIMI_DIM = 256
const MIMI_SIZE = 2048
const MIMI_ACOUSTIC = 31
const MIMI_FEAT_FRAMES = 50
const MIMI_EMB_FRAMES = 25

export interface RvqTables {
  dim: number
  size: number
  hidden: number
  semWin: Float32Array
  acoWin: Float32Array
  semWout: Float32Array
  acoWout: Float32Array
  semCb: Float32Array
  acoCb: Float32Array[]
}

/** Splits `mimi_rvq.bin` (float32 LE, contiguous) into its quantizer matrices. */
export function parseRvq(
  data: Float32Array,
  acoustic: number,
  dim = MIMI_DIM,
  size = MIMI_SIZE,
  hidden = MIMI_HIDDEN,
): RvqTables {
  let offset = 0
  const take = (length: number) => {
    const out = data.subarray(offset, offset + length)
    offset += length
    return out
  }
  const semWin = take(dim * hidden)
  const acoWin = take(dim * hidden)
  const semWout = take(hidden * dim)
  const acoWout = take(hidden * dim)
  const semCb = take(size * dim)
  const acoCb = Array.from({ length: acoustic }, () => take(size * dim))
  return { dim, size, hidden, semWin, acoWin, semWout, acoWout, semCb, acoCb }
}

function squaredNorms(codebook: Float32Array, size: number, dim: number): Float32Array {
  const norms = new Float32Array(size)
  for (let s = 0; s < size; s++) {
    let sum = 0
    const base = s * dim
    for (let k = 0; k < dim; k++) sum += codebook[base + k] * codebook[base + k]
    norms[s] = sum
  }
  return norms
}

/** Euclidean argmin of one residual row against a codebook. */
function nearestRow(
  residual: Float32Array,
  offset: number,
  codebook: Float32Array,
  norms: Float32Array,
  size: number,
  dim: number,
): number {
  let best = 0
  let bestDistance = Infinity
  for (let s = 0; s < size; s++) {
    let dot = 0
    const base = s * dim
    for (let k = 0; k < dim; k++) dot += residual[offset + k] * codebook[base + k]
    const distance = norms[s] - 2 * dot
    if (distance < bestDistance) {
      bestDistance = distance
      best = s
    }
  }
  return best
}

/** Split RVQ encode: `[channels, frames]` embedding -> `(1 + acoustic) * frames` codes. */
export function rvqEncode(emb: Float32Array, channels: number, frames: number, t: RvqTables): Int32Array {
  const { dim, size, hidden } = t
  const residual = new Float32Array(frames * dim)
  const semNorms = squaredNorms(t.semCb, size, dim)
  for (let f = 0; f < frames; f++) {
    for (let d = 0; d < dim; d++) {
      let sum = 0
      for (let h = 0; h < hidden; h++) sum += t.semWin[d * hidden + h] * emb[h * frames + f]
      residual[f * dim + d] = sum
    }
  }
  const codes = new Int32Array((1 + t.acoCb.length) * frames)
  for (let f = 0; f < frames; f++) codes[f] = nearestRow(residual, f * dim, t.semCb, semNorms, size, dim)

  for (let d = 0; d < dim; d++) {
    for (let f = 0; f < frames; f++) {
      let sum = 0
      for (let h = 0; h < hidden; h++) sum += t.acoWin[d * hidden + h] * emb[h * frames + f]
      residual[f * dim + d] = sum
    }
  }
  for (let i = 0; i < t.acoCb.length; i++) {
    const cb = t.acoCb[i]
    const norms = squaredNorms(cb, size, dim)
    for (let f = 0; f < frames; f++) {
      const code = nearestRow(residual, f * dim, cb, norms, size, dim)
      codes[(1 + i) * frames + f] = code
      const base = code * dim
      for (let d = 0; d < dim; d++) residual[f * dim + d] -= cb[base + d]
    }
  }
  return codes
}

/** Split RVQ decode: codes -> `[hidden, frames]` embedding. */
export function rvqDecode(codes: Int32Array, frames: number, t: RvqTables): Float32Array {
  const { dim, hidden } = t
  const acoustic = new Float32Array(frames * dim)
  for (let i = 0; i < t.acoCb.length; i++) {
    const cb = t.acoCb[i]
    for (let f = 0; f < frames; f++) {
      const base = codes[(1 + i) * frames + f] * dim
      for (let d = 0; d < dim; d++) acoustic[f * dim + d] += cb[base + d]
    }
  }
  const out = new Float32Array(hidden * frames)
  for (let h = 0; h < hidden; h++) {
    for (let f = 0; f < frames; f++) {
      let sum = 0
      for (let d = 0; d < dim; d++) {
        sum += t.semWout[h * dim + d] * t.semCb[codes[f] * dim + d]
        sum += t.acoWout[h * dim + d] * acoustic[f * dim + d]
      }
      out[h * frames + f] = sum
    }
  }
  return out
}

/** `[1, channels, frames]` -> `[1, frames, channels]` (enc_conv feat feeds enc_tx time-major). */
export function transposeFeature(feat: Float32Array, channels: number, frames: number): Float32Array {
  const out = new Float32Array(channels * frames)
  for (let c = 0; c < channels; c++) {
    for (let f = 0; f < frames; f++) out[f * channels + c] = feat[c * frames + f]
  }
  return out
}

let rvqPromise: Promise<RvqTables> | null = null

function loadRvq(): Promise<RvqTables> {
  rvqPromise ??= fetch(`${MIMI_BASE}/mimi_rvq.bin`)
    .then((response) => {
      if (!response.ok) throw new Error(`Failed to fetch mimi_rvq.bin (${response.status})`)
      return response.arrayBuffer()
    })
    .then((buffer) => parseRvq(new Float32Array(buffer), MIMI_ACOUSTIC))
    .catch((error) => {
      rvqPromise = null
      throw error
    })
  return rvqPromise
}

async function firstData(outputs: Record<string, import('@litertjs/core').Tensor>): Promise<Float32Array> {
  return (await Object.values(outputs)[0].data()) as Float32Array
}

export const mimiAdapter: ModelAdapter = {
  modelId: 'mimi-codec',
  metadata: {
    name: 'Mimi Codec',
    description:
      'Kyutai/Moshi neural audio codec (24 kHz). Encodes to 32 RVQ codebooks and decodes back. Fixed 2-second clips; the transformer graphs are most accurate on the CPU/wasm accelerator.',
    modelPath: `${MIMI_BASE}/mimi_enc_conv_fp16.tflite`,
    tags: ['audio', 'codec', 'mimi'],
  },
  graphs: [
    { name: 'enc_tx', modelPath: `${MIMI_BASE}/mimi_enc_tx_fp16.tflite` },
    { name: 'dec_tx', modelPath: `${MIMI_BASE}/mimi_dec_tx_fp16.tflite` },
    { name: 'deconly', modelPath: `${MIMI_BASE}/mimi_deconly_fp16.tflite` },
  ],
  inputSpecs: [
    {
      name: 'audio',
      dtype: 'float32',
      shape: [1, MIMI_SAMPLES],
      description: 'Mono PCM at 24 kHz in [-1, 1]; clips are truncated or padded to 2 seconds',
      constraints: { sampleRate: MIMI_RATE },
    },
  ],
  outputSpecs: [{ name: 'audio', dtype: 'float32', shape: [], description: 'Round-tripped 24 kHz audio' }],
  prepareInputs() {
    return {}
  },
  async parseOutputs() {
    return {}
  },
  async run(values: Record<string, any>, ctx: InferenceContext) {
    const audio = values.audio instanceof Float32Array ? values.audio : flatten(values.audio ?? [])
    if (!audio.length) throw new Error('No audio provided')

    const input = new Float32Array(MIMI_SAMPLES)
    input.set(audio.subarray(0, MIMI_SAMPLES))

    const [rvq, convOut] = await Promise.all([
      loadRvq(),
      ctx.predict('main', [ctx.createTensor(input, [1, 1, MIMI_SAMPLES])]),
    ])
    const feat = await firstData(convOut)
    const featTimeMajor = transposeFeature(feat, MIMI_HIDDEN, MIMI_FEAT_FRAMES)

    const txOut = await ctx.predict('enc_tx', [
      ctx.createTensor(featTimeMajor, [1, MIMI_FEAT_FRAMES, MIMI_HIDDEN]),
    ])
    const emb = await firstData(txOut)

    const codes = rvqEncode(emb, MIMI_HIDDEN, MIMI_EMB_FRAMES, rvq)
    const quantized = rvqDecode(codes, MIMI_EMB_FRAMES, rvq)

    const decOut = await ctx.predict('dec_tx', [ctx.createTensor(quantized, [1, MIMI_HIDDEN, MIMI_EMB_FRAMES])])
    const convIn = await firstData(decOut)

    const waveOut = await ctx.predict('deconly', [ctx.createTensor(convIn, [1, MIMI_HIDDEN, MIMI_FEAT_FRAMES])])
    const wave = await firstData(waveOut)

    const samples = new Float32Array(wave.length)
    for (let i = 0; i < wave.length; i++) samples[i] = Math.max(-1, Math.min(1, wave[i]))

    return { audio: { samples, sampleRate: MIMI_RATE } }
  },
}

export const mimiAdapters: ModelAdapter[] = [mimiAdapter]
