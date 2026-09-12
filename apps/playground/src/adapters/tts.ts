import type { InferenceContext, ModelAdapter } from './types'

const MATCHA_BASE = 'https://huggingface.co/litert-community/Matcha-TTS/resolve/main'
const MAX_TEXT = 256
const MAX_MEL = 512
const N_FEATS = 80
const N_CHANNELS = 192
const HOP = 256
const SAMPLE_RATE = 22050
const LENGTH_SCALE = 0.95
const MEL_MEAN = -5.536622047424316
const MEL_STD = 2.116101026535034
const N_TIMESTEPS = 10

interface MatchaConfig {
  symbols: string[]
}

async function fetchBuffer(path: string): Promise<ArrayBuffer> {
  const res = await fetch(path)
  if (!res.ok) throw new Error(`Failed to fetch ${path} (${res.status})`)
  return res.arrayBuffer()
}

let configPromise: Promise<MatchaConfig> | null = null
function loadConfig(): Promise<MatchaConfig> {
  configPromise ??= fetchBuffer(`${MATCHA_BASE}/config.json`)
    .then(buf => JSON.parse(new TextDecoder().decode(buf)) as MatchaConfig)
    .catch(err => {
      configPromise = null
      throw err
    })
  return configPromise
}

let embPromise: Promise<Float32Array> | null = null
function loadEmb(): Promise<Float32Array> {
  embPromise ??= fetchBuffer(`${MATCHA_BASE}/emb.bin`)
    .then(buf => new Float32Array(buf))
    .catch(err => {
      embPromise = null
      throw err
    })
  return embPromise
}

let dictPromise: Promise<Map<string, string>> | null = null
function loadDict(): Promise<Map<string, string>> {
  dictPromise ??= (async () => {
    const res = await fetch(`${MATCHA_BASE}/g2p_dict.txt.gz`)
    if (!res.ok || !res.body) throw new Error(`Failed to fetch G2P dictionary (${res.status})`)
    const text = await new Response(res.body.pipeThrough(new DecompressionStream('gzip'))).text()
    const dict = new Map<string, string>()
    for (const line of text.split('\n')) {
      const tab = line.indexOf('\t')
      if (tab > 0) dict.set(line.slice(0, tab), line.slice(tab + 1).trim())
    }
    return dict
  })().catch(err => {
    dictPromise = null
    throw err
  })
  return dictPromise
}

/** Text → IPA → Matcha phoneme ids (blank-interspersed), plus the encoder validity mask. */
function phonemize(text: string, dict: Map<string, string>, symbolOf: (s: string) => number | undefined) {
  const words = text.toLowerCase().match(/[a-z']+/g) ?? []
  const ipa = words.map(w => dict.get(w) ?? dict.get(w.replace(/'/g, ''))).filter(Boolean).join(' ')
  if (!ipa) throw new Error('No pronounceable words found')
  const ids = new Int32Array(MAX_TEXT)
  let count = 0
  for (const ch of `${ipa} .`) {
    const id = symbolOf(ch)
    if (id !== undefined && count < (MAX_TEXT - 1) / 2) {
      ids[1 + 2 * count] = id
      count++
    }
  }
  const mask = new Float32Array(MAX_TEXT)
  mask.fill(1, 0, 2 * count + 1)
  return { ids, mask }
}

function timeSin(t: number): Float32Array {
  const half = 80
  const out = new Float32Array(2 * half)
  for (let i = 0; i < half; i++) {
    const e = 1000 * t * Math.exp((-i * Math.log(10000)) / (half - 1))
    out[i] = Math.sin(e)
    out[half + i] = Math.cos(e)
  }
  return out
}

function randn(): number {
  return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random())
}

async function synthesize(text: string, ctx: InferenceContext): Promise<Float32Array> {
  const [{ symbols }, emb, dict] = await Promise.all([loadConfig(), loadEmb(), loadDict()])
  const symbolIndex = new Map(symbols.map((s, i) => [s, i]))
  const { ids, mask } = phonemize(text, dict, s => symbolIndex.get(s))

  const embInput = new Float32Array(MAX_TEXT * N_CHANNELS)
  for (let t = 0; t < MAX_TEXT; t++) {
    embInput.set(emb.subarray(ids[t] * N_CHANNELS, (ids[t] + 1) * N_CHANNELS), t * N_CHANNELS)
  }

  const enc = await ctx.predict('main', [
    ctx.createTensor(embInput, [1, MAX_TEXT, N_CHANNELS]),
    ctx.createTensor(mask, [1, 1, MAX_TEXT]),
  ])
  const encOut = await Promise.all(Object.values(enc).map(t => t.data() as Promise<Float32Array>))
  const [mu, logw] = encOut.sort((a, b) => b.length - a.length) as [Float32Array, Float32Array]

  const durations = new Float32Array(MAX_TEXT)
  let total = 0
  for (let t = 0; t < MAX_TEXT; t++) {
    durations[t] = Math.ceil(Math.exp(logw[t]) * mask[t]) * LENGTH_SCALE
    total += durations[t]
  }
  const ylen = Math.min(Math.max(Math.floor(total), 1), MAX_MEL)

  const muY = new Float32Array(N_FEATS * MAX_MEL)
  let source = 0
  for (let j = 0; j < ylen; j++) {
    while (source < MAX_TEXT - 1 && durations[source] <= j) source++
    for (let c = 0; c < N_FEATS; c++) muY[c * MAX_MEL + j] = mu[c * MAX_TEXT + source]
  }
  const ymask = new Float32Array(MAX_MEL)
  ymask.fill(1, 0, ylen)

  const x = new Float32Array(N_FEATS * MAX_MEL)
  for (let c = 0; c < N_FEATS; c++) for (let j = 0; j < ylen; j++) x[c * MAX_MEL + j] = randn()

  for (let step = 0; step < N_TIMESTEPS; step++) {
    const out = await ctx.predict('decoder', [
      ctx.createTensor(x, [1, N_FEATS, MAX_MEL]),
      ctx.createTensor(muY, [1, N_FEATS, MAX_MEL]),
      ctx.createTensor(timeSin(step / N_TIMESTEPS), [1, 2 * 80]),
      ctx.createTensor(ymask, [1, 1, MAX_MEL]),
    ])
    const velocity = (await Object.values(out)[0].data()) as Float32Array
    for (let c = 0; c < N_FEATS; c++) for (let j = 0; j < ylen; j++) x[c * MAX_MEL + j] += velocity[c * MAX_MEL + j] / N_TIMESTEPS
  }

  const mel = new Float32Array(N_FEATS * MAX_MEL)
  for (let c = 0; c < N_FEATS; c++) for (let j = 0; j < ylen; j++) mel[c * MAX_MEL + j] = x[c * MAX_MEL + j] * MEL_STD + MEL_MEAN

  const voc = await ctx.predict('vocoder', [ctx.createTensor(mel, [1, N_FEATS, MAX_MEL])])
  const wave = (await Object.values(voc)[0].data()) as Float32Array
  const samples = wave.slice(0, ylen * HOP)
  for (let i = 0; i < samples.length; i++) samples[i] = Math.max(-1, Math.min(1, samples[i]))
  return samples
}

export const matchaTtsAdapter: ModelAdapter = {
  modelId: 'matcha-tts',
  metadata: {
    name: 'Matcha-TTS',
    description: 'Fast non-autoregressive text-to-speech (22.05 kHz)',
    modelPath: `${MATCHA_BASE}/matcha_textenc_fp16.tflite`,
    tags: ['audio', 'tts', 'matcha'],
  },
  graphs: [
    { name: 'decoder', modelPath: `${MATCHA_BASE}/matcha_decoder_fp16.tflite` },
    { name: 'vocoder', modelPath: `${MATCHA_BASE}/matcha_vocoder_fp16.tflite` },
  ],
  inputSpecs: [
    { name: 'text', dtype: 'string', shape: [], description: 'Text to speak (English)', constraints: { text: true } },
  ],
  outputSpecs: [{ name: 'audio', dtype: 'float32', shape: [], description: 'Synthesized 22.05 kHz audio' }],
  prepareInputs: () => ({}),
  parseOutputs: async () => ({}),
  async run(values, ctx) {
    const text = String(values['text'] ?? '').trim()
    if (!text) throw new Error('Provide text to speak')
    return { audio: { samples: await synthesize(text, ctx), sampleRate: SAMPLE_RATE } }
  },
}

export const ttsAdapters: ModelAdapter[] = [matchaTtsAdapter]
