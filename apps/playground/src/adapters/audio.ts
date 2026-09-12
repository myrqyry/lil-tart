import type { InferenceContext, ModelAdapter } from './types'
import { computeDeltas, decodeSentencePiece, logMelSpectrogram, makeCausalMask, melFilterbank, melFilterbankSlaney, melSpectrogram, windowsOf } from '../audioUtils'
import { loadHfTokenizer } from '../hfTokenizer'
import { flatten } from './util'

const MODEL_PATH = 'https://huggingface.co/litert-community/moonshine-tiny/resolve/main/moonshine_tiny_5s_f32.tflite'
const TOKENIZER_PATH = 'https://huggingface.co/UsefulSensors/moonshine-tiny/resolve/main/tokenizer.json'
const WINDOW_SAMPLES = 80000 // 5 s @ 16 kHz
const MAX_TOKENS = 64
const START_TOKEN = 1
const EOS_TOKEN = 2

let vocabPromise: Promise<string[]> | null = null

function loadVocab(): Promise<string[]> {
  if (!vocabPromise) {
    vocabPromise = fetch(TOKENIZER_PATH)
      .then(response => {
        if (!response.ok) throw new Error(`tokenizer.json ${response.status}`)
        return response.json()
      })
      .then(json => {
        const vocab: string[] = []
        for (const [token, id] of Object.entries(json.model.vocab as Record<string, number>)) vocab[id] = token
        return vocab
      })
      .catch(cause => {
        vocabPromise = null
        throw cause
      })
  }
  return vocabPromise
}

function argmax(values: Float32Array, offset: number, length: number): number {
  let best = 0
  let bestValue = -Infinity
  for (let i = 0; i < length; i++) {
    const value = values[offset + i]
    if (value > bestValue) {
      bestValue = value
      best = i
    }
  }
  return best
}

async function transcribeWindow(audio: Float32Array, ctx: InferenceContext): Promise<string> {
  const padded = new Float32Array(WINDOW_SAMPLES)
  padded.set(audio.subarray(0, Math.min(audio.length, WINDOW_SAMPLES)))

  const encoded = await ctx.predict('main', [ctx.createTensor(padded, [1, WINDOW_SAMPLES])], 'encode')
  const states = Object.values(encoded)[0]

  const mask = makeCausalMask(MAX_TOKENS)
  const tokens = new Int32Array(MAX_TOKENS).fill(EOS_TOKEN)
  tokens[0] = START_TOKEN
  const decoded: number[] = []

  for (let position = 1; position < MAX_TOKENS; position++) {
    const out = await ctx.predict('main', [
      states,
      ctx.createTensor(tokens, [1, MAX_TOKENS]),
      ctx.createTensor(mask, [1, 1, MAX_TOKENS, MAX_TOKENS]),
    ], 'decode')
    const logits = Object.values(out)[0]
    const data = (await logits.data()) as Float32Array
    const next = argmax(data, (position - 1) * 32768, 32768)
    if (next === EOS_TOKEN) break
    tokens[position] = next
    decoded.push(next)
  }

  return decodeSentencePiece(decoded, await loadVocab())
}

export const moonshineAdapter: ModelAdapter = {
  modelId: 'moonshine-tiny',
  metadata: {
    name: 'Moonshine Tiny — Speech Recognition',
    description: 'Greedy ASR over 5-second windows. Raw 16 kHz audio in, transcript out (no external frontend).',
    modelPath: MODEL_PATH,
    tags: ['audio', 'asr', 'moonshine'],
  },
  inputSpecs: [{
    name: 'audio',
    dtype: 'float32',
    shape: [1, WINDOW_SAMPLES],
    description: 'Mono PCM at 16 kHz in [-1, 1]; longer clips are split into 5-second windows',
  }],
  outputSpecs: [{
    name: 'text',
    dtype: 'float32',
    shape: [],
    description: 'Transcribed text',
  }],
  prepareInputs() {
    return {}
  },
  async parseOutputs() {
    return {}
  },
  async run(values, ctx) {
    const raw = values['audio']
    const audio = raw instanceof Float32Array ? raw : flatten(raw ?? [])
    if (!audio.length) throw new Error('No audio provided')
    const parts: string[] = []
    for (const window of windowsOf(audio, WINDOW_SAMPLES)) {
      const text = await transcribeWindow(window, ctx)
      if (text) parts.push(text)
    }
    return { text: parts.join(' ').trim() }
  },
}

const WHISPER_RATE = 16000
const WHISPER_SAMPLES = 480000 // 30 s @ 16 kHz
const WHISPER_FRAMES = 3000
const WHISPER_FFT = 400
const WHISPER_HOP = 160
const WHISPER_NMELS = 80
const WHISPER_MAX_TOKENS = 128
const WHISPER_VOCAB = 51865
const WHISPER_EOT = 50257
// <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
const WHISPER_PROMPT = [50258, 50259, 50359, 50363]
const WHISPER_TOKENIZER = 'https://huggingface.co/openai/whisper-base/resolve/main/tokenizer.json'
const WHISPER_MEL = melFilterbankSlaney(WHISPER_RATE, WHISPER_FFT, WHISPER_NMELS, 0, 8000)

// OpenAI Whisper front-end: 80-mel log-spectrogram, global max-8 clamp, (x+4)/4 scaling.
function whisperFeatures(audio: Float32Array): Float32Array {
  const padded = new Float32Array(WHISPER_SAMPLES)
  padded.set(audio.subarray(0, Math.min(audio.length, WHISPER_SAMPLES)))
  const power = melSpectrogram(padded, WHISPER_MEL, WHISPER_NMELS, WHISPER_FFT, WHISPER_HOP, WHISPER_RATE, WHISPER_FFT, WHISPER_SAMPLES)
  const log = new Float32Array(WHISPER_NMELS * WHISPER_FRAMES)
  let max = -Infinity
  for (let m = 0; m < WHISPER_NMELS; m++) {
    for (let t = 0; t < WHISPER_FRAMES; t++) {
      const value = Math.log10(Math.max(power[t * WHISPER_NMELS + m], 1e-10))
      log[m * WHISPER_FRAMES + t] = value
      if (value > max) max = value
    }
  }
  const floor = max - 8
  for (let i = 0; i < log.length; i++) log[i] = (Math.max(log[i], floor) + 4) / 4
  return log
}

async function transcribeWhisper(audio: Float32Array, ctx: InferenceContext): Promise<string> {
  const features = whisperFeatures(audio)
  const encoded = await ctx.predict('main', [ctx.createTensor(features, [1, WHISPER_NMELS, WHISPER_FRAMES])], 'encode')
  const states = Object.values(encoded)[0]

  const mask = makeCausalMask(WHISPER_MAX_TOKENS)
  const tokens = new Int32Array(WHISPER_MAX_TOKENS).fill(WHISPER_EOT)
  WHISPER_PROMPT.forEach((id, i) => { tokens[i] = id })
  let length = WHISPER_PROMPT.length

  while (length < WHISPER_MAX_TOKENS) {
    const out = await ctx.predict('main', [
      states,
      ctx.createTensor(tokens, [1, WHISPER_MAX_TOKENS]),
      ctx.createTensor(mask, [1, 1, WHISPER_MAX_TOKENS, WHISPER_MAX_TOKENS]),
    ], 'decode')
    const data = (await Object.values(out)[0].data()) as Float32Array
    const next = argmax(data, (length - 1) * WHISPER_VOCAB, WHISPER_VOCAB)
    if (next === WHISPER_EOT) break
    tokens[length] = next
    length++
  }

  const tokenizer = await loadHfTokenizer(WHISPER_TOKENIZER)
  return tokenizer.decode(Array.from(tokens.subarray(0, length))).trim()
}

function makeWhisperAdapter(repo: string, file: string, modelId: string, name: string, description: string): ModelAdapter {
  return {
    modelId,
    metadata: {
      name,
      description,
      modelPath: `https://huggingface.co/litert-community/${repo}/resolve/main/${file}`,
      tags: ['audio', 'asr', 'whisper'],
    },
    inputSpecs: [{
      name: 'audio',
      dtype: 'float32',
      shape: [1, WHISPER_SAMPLES],
      description: 'Mono PCM at 16 kHz in [-1, 1]; clips are split into 30-second windows',
    }],
    outputSpecs: [{
      name: 'text',
      dtype: 'float32',
      shape: [],
      description: 'Transcribed text',
    }],
    prepareInputs() {
      return {}
    },
    async parseOutputs() {
      return {}
    },
    async run(values, ctx) {
      const raw = values['audio']
      const audio = raw instanceof Float32Array ? raw : flatten(raw ?? [])
      if (!audio.length) throw new Error('No audio provided')
      const parts: string[] = []
      for (const window of windowsOf(audio, WHISPER_SAMPLES)) {
        const text = await transcribeWhisper(window, ctx)
        if (text) parts.push(text)
      }
      return { text: parts.join(' ').trim() }
    },
  }
}

export const whisperTinyAdapter = makeWhisperAdapter(
  'whisper-tiny',
  'whisper_tiny_30s_f32.tflite',
  'whisper-tiny',
  'Whisper Tiny — Speech Recognition',
  'Multilingual ASR over 30-second windows with a host-side 80-mel front-end.',
)

export const whisperBaseAdapter = makeWhisperAdapter(
  'whisper-base',
  'whisper_base_30s_f32.tflite',
  'whisper-base',
  'Whisper Base — Speech Recognition',
  'Multilingual ASR over 30-second windows with a host-side 80-mel front-end.',
)

const CREPE_PATH = 'https://huggingface.co/litert-community/CREPE-pitch-LiteRT/resolve/main/crepe_full_fp16.tflite'
const CREPE_FRAME = 1024
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

// torchcrepe weighted_argmax decode: peak bin ± 4, activation-weighted average.
function decodePitch(activation: Float32Array): { hz: number; midi: number; confidence: number } {
  let peak = 0
  let best = -Infinity
  for (let i = 0; i < activation.length; i++) {
    if (activation[i] > best) { best = activation[i]; peak = i }
  }
  const start = Math.max(0, peak - 4)
  const end = Math.min(activation.length, peak + 5)
  let weight = 0
  let weighted = 0
  for (let i = start; i < end; i++) { weight += activation[i]; weighted += activation[i] * i }
  const cents = 20 * (weighted / weight) + 1997.3794084376191
  const hz = 10 * 2 ** (cents / 1200)
  return { hz, midi: 69 + 12 * Math.log2(hz / 440), confidence: activation[peak] }
}

function formatPitch(hz: number, midi: number): string {
  const rounded = Math.round(midi)
  return `${hz.toFixed(1)} Hz · ${NOTE_NAMES[((rounded % 12) + 12) % 12]}${Math.floor(rounded / 12) - 1} ${Math.round((midi - rounded) * 100) >= 0 ? '+' : ''}${Math.round((midi - rounded) * 100)} cents`
}

export const crepeAdapter: ModelAdapter = {
  modelId: 'crepe-pitch',
  metadata: {
    name: 'CREPE — Pitch Detection',
    description: 'Monophonic f0 over 1024-sample (16 kHz) frames → 360 pitch-bin activations, decoded host-side to Hz and nearest note.',
    modelPath: CREPE_PATH,
    tags: ['audio', 'pitch', 'tuner', 'crepe'],
  },
  inputSpecs: [{
    name: 'audio',
    dtype: 'float32',
    shape: [1, CREPE_FRAME],
    description: 'Mono PCM at 16 kHz in [-1, 1]; split into 1024-sample frames',
  }],
  outputSpecs: [{
    name: 'pitch',
    dtype: 'float32',
    shape: [],
    description: 'Median detected pitch (Hz + nearest note)',
  }],
  prepareInputs() {
    return {}
  },
  async parseOutputs() {
    return {}
  },
  async run(values, ctx) {
    const raw = values['audio']
    const audio = raw instanceof Float32Array ? raw : flatten(raw ?? [])
    if (!audio.length) throw new Error('No audio provided')
    const frames = windowsOf(audio, CREPE_FRAME).slice(0, 256) // ponytail: cap at 256 frames (~16 s); add overlap when pitch tracking needs smoothing
    const hz: number[] = []
    const candidates: { hz: number; midi: number; confidence: number }[] = []
    for (const frame of frames) {
      const input = new Float32Array(CREPE_FRAME)
      input.set(frame.subarray(0, CREPE_FRAME))
      let mean = 0
      for (const value of input) mean += value
      mean /= CREPE_FRAME
      let variance = 0
      for (const value of input) variance += (value - mean) ** 2
      const std = Math.max(Math.sqrt(variance / CREPE_FRAME), 1e-10)
      for (let i = 0; i < CREPE_FRAME; i++) input[i] = (input[i] - mean) / std
      const out = await ctx.predict('main', { input: ctx.createTensor(input, [1, CREPE_FRAME]) })
      const activation = (await Object.values(out)[0].data()) as Float32Array
      const decoded = decodePitch(activation)
      hz.push(Number(decoded.hz.toFixed(1)))
      if (decoded.confidence > 0.5) candidates.push(decoded)
    }
    if (!candidates.length) return { pitch: 'No clear pitch detected', hz }
    candidates.sort((a, b) => a.hz - b.hz)
    const median = candidates[Math.floor(candidates.length / 2)]
    return { pitch: formatPitch(median.hz, median.midi), hz }
  },
}

const W2V2_FRONTEND_PATH = 'https://huggingface.co/litert-community/wav2vec2-base-960h-LiteRT/resolve/main/w2v2_asr_frontend_fp16.tflite'
const W2V2_HEAD_PATH = 'https://huggingface.co/litert-community/wav2vec2-base-960h-LiteRT/resolve/main/w2v2_asr_head_fp16.tflite'
const W2V2_TOKENS_PATH = 'https://huggingface.co/litert-community/wav2vec2-base-960h-LiteRT/resolve/main/tokens.txt'
const W2V2_SAMPLES = 256000 // 16 s @ 16 kHz

let w2v2TokensPromise: Promise<string[]> | null = null

function loadW2V2Tokens(): Promise<string[]> {
  if (!w2v2TokensPromise) {
    w2v2TokensPromise = fetch(W2V2_TOKENS_PATH)
      .then(response => {
        if (!response.ok) throw new Error(`tokens.txt ${response.status}`)
        return response.text()
      })
      .then(text => text.split('\n').map(line => line.replace(/\r$/, '')))
      .catch(cause => {
        w2v2TokensPromise = null
        throw cause
      })
  }
  return w2v2TokensPromise
}

export const wav2vec2Adapter: ModelAdapter = {
  modelId: 'wav2vec2-960h',
  metadata: {
    name: 'wav2vec2-base-960h — Speech Recognition',
    description: 'Character-level CTC ASR, raw 16 kHz waveform straight into the 1D-conv frontend (no FFT). Two GPU graphs: frontend → head.',
    modelPath: W2V2_FRONTEND_PATH,
    tags: ['audio', 'asr', 'ctc', 'wav2vec2'],
  },
  graphs: [{ name: 'head', modelPath: W2V2_HEAD_PATH }],
  inputSpecs: [{
    name: 'audio',
    dtype: 'float32',
    shape: [1, W2V2_SAMPLES],
    description: 'Mono PCM at 16 kHz in [-1, 1]; padded/truncated to 16 seconds',
  }],
  outputSpecs: [{
    name: 'text',
    dtype: 'float32',
    shape: [],
    description: 'Transcribed text',
  }],
  prepareInputs() {
    return {}
  },
  async parseOutputs() {
    return {}
  },
  async run(values, ctx) {
    const raw = values['audio']
    const audio = raw instanceof Float32Array ? raw : flatten(raw ?? [])
    if (!audio.length) throw new Error('No audio provided')
    const samples = Math.min(audio.length, W2V2_SAMPLES)
    const input = new Float32Array(W2V2_SAMPLES)
    input.set(audio.subarray(0, samples))

    const frontend = await ctx.predict('main', { input: ctx.createTensor(input, [1, W2V2_SAMPLES]) })
    const features = Object.values(frontend)[0]
    const head = await ctx.predict('head', { input: features })
    const logits = (await Object.values(head)[0].data()) as Float32Array

    // Conv feature-extractor length formula from the reference (kernels/strides).
    let length = samples
    for (const [kernel, stride] of [[10, 5], [3, 2], [3, 2], [3, 2], [3, 2], [2, 2], [2, 2]]) {
      length = Math.floor((length - kernel) / stride) + 1
    }

    const tokens = await loadW2V2Tokens()
    const out: string[] = []
    let previous = -1
    for (let t = 0; t < length; t++) {
      let best = 0
      let bestValue = -Infinity
      for (let c = 0; c < 32; c++) {
        const value = logits[t * 32 + c]
        if (value > bestValue) { bestValue = value; best = c }
      }
      if (best !== previous && best !== 0) out.push(tokens[best] ?? '')
      previous = best
    }
    return { text: out.join('').replace(/\|/g, ' ').trim() }
  },
}

const W2V2_KWS_FRONTEND_PATH = 'https://huggingface.co/litert-community/wav2vec2-keyword-spotting/resolve/main/w2v2_frontend_fp16.tflite'
const W2V2_KWS_HEAD_PATH = 'https://huggingface.co/litert-community/wav2vec2-keyword-spotting/resolve/main/w2v2_head_fp16.tflite'
const W2V2_KWS_SAMPLES = 16000 // 1 s @ 16 kHz
const W2V2_KWS_LABELS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', '_unknown_', '_silence_']

export const wav2vec2KeywordAdapter: ModelAdapter = {
  modelId: 'wav2vec2-kws',
  metadata: {
    name: 'wav2vec2 Keyword Spotting',
    description: 'Speech-Commands keyword spotting, raw 16 kHz waveform straight into the 1D-conv frontend (no FFT). Two GPU graphs: frontend → head.',
    modelPath: W2V2_KWS_FRONTEND_PATH,
    tags: ['audio', 'classification', 'keyword-spotting', 'wav2vec2'],
  },
  graphs: [{ name: 'head', modelPath: W2V2_KWS_HEAD_PATH }],
  inputSpecs: [{
    name: 'audio',
    dtype: 'float32',
    shape: [1, W2V2_KWS_SAMPLES],
    description: 'Mono PCM at 16 kHz in [-1, 1]; padded/truncated to 1 second',
  }],
  outputSpecs: [{
    name: 'keyword',
    dtype: 'float32',
    shape: [],
    description: 'Keyword class scores',
  }],
  prepareInputs() {
    return {}
  },
  async parseOutputs() {
    return {}
  },
  async run(values, ctx) {
    const raw = values['audio']
    const audio = raw instanceof Float32Array ? raw : flatten(raw ?? [])
    if (!audio.length) throw new Error('No audio provided')
    const input = new Float32Array(W2V2_KWS_SAMPLES)
    input.set(audio.subarray(0, Math.min(audio.length, W2V2_KWS_SAMPLES)))

    const frontend = await ctx.predict('main', { input: ctx.createTensor(input, [1, W2V2_KWS_SAMPLES]) })
    const features = Object.values(frontend)[0]
    const head = await ctx.predict('head', { input: features })
    const logits = (await Object.values(head)[0].data()) as Float32Array

    let max = -Infinity
    for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i]
    const probs = Array.from(logits, value => Math.exp(value - max))
    const sum = probs.reduce((total, value) => total + value, 0)
    const ranked = probs
      .map((value, index) => ({ label: W2V2_KWS_LABELS[index] ?? `#${index}`, p: value / sum }))
      .sort((a, b) => b.p - a.p)

    return { keyword: ranked.map(entry => `${(entry.p * 100).toFixed(1)}%  ${entry.label}`).join('\n') }
  },
}

const PANNS_PATH = 'https://huggingface.co/litert-community/PANNs-CNN14-AudioSet-LiteRT/resolve/main/cnn14_audioset_fp16.tflite'
const PANNS_MEL_PATH = 'https://huggingface.co/litert-community/PANNs-CNN14-AudioSet-LiteRT/resolve/main/mel_basis.bin'
const PANNS_LABELS_PATH = 'https://huggingface.co/litert-community/PANNs-CNN14-AudioSet-LiteRT/resolve/main/audioset_labels.txt'
const PANNS_RATE = 32000
const PANNS_SAMPLES = 320000 // 10 s @ 32 kHz
const PANNS_NMELS = 64
const PANNS_FRAMES = 1001

let melBasisPromise: Promise<Float32Array> | null = null
let labelsPromise: Promise<string[]> | null = null

function loadMelBasis(): Promise<Float32Array> {
  if (!melBasisPromise) {
    melBasisPromise = fetch(PANNS_MEL_PATH)
      .then(response => {
        if (!response.ok) throw new Error(`mel_basis.bin ${response.status}`)
        return response.arrayBuffer()
      })
      .then(buffer => new Float32Array(buffer))
      .catch(cause => {
        melBasisPromise = null
        throw cause
      })
  }
  return melBasisPromise
}

function loadLabels(): Promise<string[]> {
  if (!labelsPromise) {
    labelsPromise = fetch(PANNS_LABELS_PATH)
      .then(response => {
        if (!response.ok) throw new Error(`audioset_labels.txt ${response.status}`)
        return response.text()
      })
      .then(text => text.split('\n').map(line => line.replace(/\r$/, '')))
      .catch(cause => {
        labelsPromise = null
        throw cause
      })
  }
  return labelsPromise
}

export const pannsAdapter: ModelAdapter = {
  modelId: 'panns-cnn14',
  metadata: {
    name: 'PANNs CNN14 — AudioSet Tagging',
    description: 'Multi-label sound-event tagging over 527 AudioSet classes. Host log-mel (1024-pt FFT), CNN14 on GPU.',
    modelPath: PANNS_PATH,
    tags: ['audio', 'tagging', 'audioset', 'panns'],
  },
  inputSpecs: [{
    name: 'audio',
    dtype: 'float32',
    shape: [1, PANNS_SAMPLES],
    description: 'Mono PCM at 32 kHz in [-1, 1]; padded/truncated to 10 seconds',
    constraints: { sampleRate: PANNS_RATE },
  }],
  outputSpecs: [{
    name: 'tags',
    dtype: 'float32',
    shape: [],
    description: 'Top-5 AudioSet tags with probabilities',
  }],
  prepareInputs() {
    return {}
  },
  async parseOutputs() {
    return {}
  },
  async run(values, ctx) {
    const raw = values['audio']
    const audio = raw instanceof Float32Array ? raw : flatten(raw ?? [])
    if (!audio.length) throw new Error('No audio provided')
    const [basis, labels] = await Promise.all([loadMelBasis(), loadLabels()])
    const logmel = logMelSpectrogram(audio, basis, PANNS_NMELS, 1024, 320, PANNS_RATE)
    const out = await ctx.predict('main', [ctx.createTensor(logmel, [1, 1, PANNS_FRAMES, PANNS_NMELS])])
    const probs = (await Object.values(out)[0].data()) as Float32Array

    const ranked = Array.from(probs, (score, id) => ({ score, id }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 5)
    return { tags: ranked.map(({ score, id }) => `${score.toFixed(3)}  ${labels[id] ?? `class ${id}`}`).join('\n') }
  },
}

const BASIC_PITCH_PATH = 'https://huggingface.co/litert-community/Basic-Pitch-LiteRT/resolve/main/basicpitch.tflite'
const BASIC_PITCH_RATE = 22050
const BASIC_PITCH_SAMPLES = 43844 // 2 s @ 22.05 kHz
const BASIC_PITCH_FRAMES = 172
const BASIC_PITCH_BINS = 88 // MIDI 21..108
const BASIC_PITCH_MIDI_BASE = 21
const BASIC_PITCH_HOP = 256

function midiName(midi: number): string {
  const rounded = Math.round(midi)
  return `${NOTE_NAMES[((rounded % 12) + 12) % 12]}${Math.floor(rounded / 12) - 1}`
}

export const basicPitchAdapter: ModelAdapter = {
  modelId: 'basic-pitch',
  metadata: {
    name: 'Basic Pitch — Music Transcription',
    description: 'Polyphonic note detection (MIDI 21–108) from 2-second windows of 22.05 kHz audio. Single graph, note posteriorgram decoded host-side.',
    modelPath: BASIC_PITCH_PATH,
    tags: ['audio', 'music', 'transcription', 'basic-pitch'],
  },
  inputSpecs: [{
    name: 'audio',
    dtype: 'float32',
    shape: [1, BASIC_PITCH_SAMPLES],
    description: 'Mono PCM at 22.05 kHz in [-1, 1]; split into 2-second windows',
    constraints: { sampleRate: BASIC_PITCH_RATE },
  }],
  outputSpecs: [{
    name: 'notes',
    dtype: 'float32',
    shape: [],
    description: 'Detected notes (start time · name · duration)',
  }],
  prepareInputs() {
    return {}
  },
  async parseOutputs() {
    return {}
  },
  async run(values, ctx) {
    const raw = values['audio']
    const audio = raw instanceof Float32Array ? raw : flatten(raw ?? [])
    if (!audio.length) throw new Error('No audio provided')

    const events: { midi: number; start: number; end: number }[] = []
    const active = new Map<number, number>() // key -> global start frame
    let globalFrame = 0

    for (const window of windowsOf(audio, BASIC_PITCH_SAMPLES)) {
      const input = new Float32Array(BASIC_PITCH_SAMPLES)
      input.set(window.subarray(0, BASIC_PITCH_SAMPLES))
      const out = await ctx.predict('main', [ctx.createTensor(input, [1, BASIC_PITCH_SAMPLES])])
      const note = (await Object.values(out)[1].data()) as Float32Array // outputs: contour, note, onset

      for (let frame = 0; frame < BASIC_PITCH_FRAMES; frame++) {
        for (let key = 0; key < BASIC_PITCH_BINS; key++) {
          if (note[frame * BASIC_PITCH_BINS + key] > 0.5) {
            if (!active.has(key)) active.set(key, globalFrame + frame)
          } else if (active.has(key)) {
            events.push({ midi: key + BASIC_PITCH_MIDI_BASE, start: active.get(key)!, end: globalFrame + frame })
            active.delete(key)
          }
        }
      }
      globalFrame += BASIC_PITCH_FRAMES
    }
    for (const [key, start] of active) events.push({ midi: key + BASIC_PITCH_MIDI_BASE, start, end: globalFrame })

    // ponytail: frame-threshold decoding; onset-triggered decoding would sharpen attacks
    events.sort((a, b) => a.start - b.start)
    const seconds = (frame: number) => (frame * BASIC_PITCH_HOP) / BASIC_PITCH_RATE
    const lines = events.slice(0, 200).map(e =>
      `${seconds(e.start).toFixed(2)}s  ${midiName(e.midi)} (midi ${e.midi})  ${seconds(e.end - e.start).toFixed(2)}s`)
    return { notes: lines.length ? lines.join('\n') : 'No notes detected' }
  },
}

const GRANITE_SPEECH_BASE = 'https://huggingface.co/litert-community/granite-speech-5.0-470m-turboctc/resolve/main'
const GRANITE_SPEECH_PATH = `${GRANITE_SPEECH_BASE}/granite_speech_ctc_wi8fc.tflite`
const GRANITE_WINDOWS = [5, 10, 30]
const GRANITE_RATE = 16000
const GRANITE_NMELS = 80
const GRANITE_MEL = melFilterbank(GRANITE_RATE, 512, GRANITE_NMELS, 0, 8000)

// 5/10/30 s window → [nFrames/2, 320] features (80 log-mel + 80 deltas, two frames interleaved).
function graniteFeatures(audio: Float32Array, seconds: number): Float32Array {
  const samples = seconds * GRANITE_RATE
  const padded = new Float32Array(samples)
  padded.set(audio.subarray(0, Math.min(audio.length, samples)))
  const power = melSpectrogram(padded, GRANITE_MEL, GRANITE_NMELS, 512, 160, GRANITE_RATE, 400, samples)
  const nFrames = Math.floor(samples / 160)

  const mel = new Float32Array(GRANITE_NMELS * nFrames)
  let max = -Infinity
  for (let t = 0; t < nFrames; t++) {
    for (let m = 0; m < GRANITE_NMELS; m++) {
      const value = Math.log10(Math.max(power[t * GRANITE_NMELS + m], 1e-10))
      mel[m * nFrames + t] = value
      if (value > max) max = value
    }
  }
  const floor = max - 8
  for (let i = 0; i < mel.length; i++) mel[i] = Math.max(mel[i], floor) / 4 + 1
  const deltas = computeDeltas(mel, GRANITE_NMELS, nFrames, 3)

  const rows = nFrames >> 1
  const features = new Float32Array(rows * 320)
  for (let i = 0; i < rows; i++) {
    for (let m = 0; m < GRANITE_NMELS; m++) {
      features[i * 320 + m] = mel[m * nFrames + 2 * i]
      features[i * 320 + 80 + m] = deltas[m * nFrames + 2 * i]
      features[i * 320 + 160 + m] = mel[m * nFrames + 2 * i + 1]
      features[i * 320 + 240 + m] = deltas[m * nFrames + 2 * i + 1]
    }
  }
  return features
}

async function transcribeGranite(audio: Float32Array, seconds: number, ctx: InferenceContext): Promise<string> {
  const features = graniteFeatures(audio, seconds)
  const out = await ctx.predict(
    'main',
    { input_features: ctx.createTensor(features, [1, features.length / 320, 320]) },
    `transcribe_${seconds}s`,
  )
  let ids: Int32Array | null = null
  for (const tensor of Object.values(out)) {
    const data = await tensor.data()
    if (data instanceof Int32Array) { ids = data; break }
  }
  if (!ids) return ''
  const collapsed: number[] = []
  let previous = -1
  for (const id of ids) {
    if (id !== previous && id !== 0) collapsed.push(id)
    previous = id
  }
  const tokenizer = await loadHfTokenizer(`${GRANITE_SPEECH_BASE}/tokenizer.json`)
  return tokenizer.decode(collapsed).trim()
}

export const graniteSpeechAdapter: ModelAdapter = {
  modelId: 'granite-speech',
  metadata: {
    name: 'Granite Speech 470M — Speech Recognition',
    description: 'CTC ASR with a host log-mel frontend (no external assets). Single graph; CTC argmax is decoded in-graph.',
    modelPath: GRANITE_SPEECH_PATH,
    tags: ['audio', 'asr', 'ctc', 'granite'],
  },
  inputSpecs: [{
    name: 'audio',
    dtype: 'float32',
    shape: [1, 30 * GRANITE_RATE],
    description: 'Mono PCM at 16 kHz in [-1, 1]; processed in up to 30-second windows',
  }],
  outputSpecs: [{
    name: 'text',
    dtype: 'float32',
    shape: [],
    description: 'Transcribed text',
  }],
  prepareInputs() {
    return {}
  },
  async parseOutputs() {
    return {}
  },
  async run(values, ctx) {
    const raw = values['audio']
    const audio = raw instanceof Float32Array ? raw : flatten(raw ?? [])
    if (!audio.length) throw new Error('No audio provided')
    const parts: string[] = []
    for (const window of windowsOf(audio, 30 * GRANITE_RATE)) {
      const seconds = GRANITE_WINDOWS.find(s => window.length <= s * GRANITE_RATE) ?? 30
      const text = await transcribeGranite(window, seconds, ctx)
      if (text) parts.push(text)
    }
    return { text: parts.join(' ').trim() }
  },
}

export const audioAdapters: ModelAdapter[] = [moonshineAdapter, whisperTinyAdapter, whisperBaseAdapter, crepeAdapter, wav2vec2Adapter, wav2vec2KeywordAdapter, pannsAdapter, basicPitchAdapter, graniteSpeechAdapter]
