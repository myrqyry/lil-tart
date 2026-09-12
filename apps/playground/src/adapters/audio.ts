import type { InferenceContext, ModelAdapter } from './types'
import { decodeSentencePiece, makeCausalMask, windowsOf } from '../audioUtils'
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

export const audioAdapters: ModelAdapter[] = [moonshineAdapter]
