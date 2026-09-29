import { Tensor } from '@litertjs/core'
import type { ModelAdapter } from './types'
import { flatten, toNestedArray } from './util'

export const audioPreprocessor: ModelAdapter = {
  modelId: 'audio_preprocessor',
  metadata: {
    name: 'Audio Preprocessor',
    description: 'Converts raw audio waveform (10s @ 16kHz) into mel-spectrogram features [1, 992, 128]',
    modelPath: '/models/audio_preprocessor.tflite',
    tags: ['musiccoca', 'audio'],
  },
  disabled: true, // Official google/magenta-realtime-2 HF repo contains MusicCoCa TFLites; keep disabled until this adapter contract is verified.
  inputSpecs: [{
    name: 'waveform',
    dtype: 'float32',
    shape: [1, 160000],
    description: 'Raw audio samples (10 seconds at 16kHz, mean-centered)',
    constraints: { min: -1, max: 1 },
  }],
  outputSpecs: [{
    name: 'features',
    dtype: 'float32',
    shape: [1, 992, 128],
    description: 'Mel-spectrogram features',
  }],
  prepareInputs(values: Record<string, any>) {
    let arr = flatten(values['waveform'] ?? [])
    if (arr.length !== 160000) { const tmp = new Float32Array(160000); tmp.set(arr.slice(0, 160000)); arr = tmp }
    return { waveform: new Tensor(arr, [1, 160000]) }
  },
  async parseOutputs(outputs) {
    const t = outputs['Identity']
    const data = await t.data()
    return { features: toNestedArray(data as Float32Array, [1, 992, 128]) }
  },
}

export const musicEncoder: ModelAdapter = {
  modelId: 'music_encoder',
  metadata: {
    name: 'Music Encoder',
    description: 'Encodes audio features [1, 992, 128] into a 768-dim music embedding',
    modelPath: '/models/music_encoder.tflite',
    tags: ['musiccoca', 'audio', 'embedding'],
  },
  disabled: true, // Official google/magenta-realtime-2 HF repo contains MusicCoCa TFLites; keep disabled until this adapter contract is verified.
  inputSpecs: [{
    name: 'features',
    dtype: 'float32',
    shape: [1, 992, 128],
    description: 'Mel-spectrogram features from Audio Preprocessor',
  }],
  outputSpecs: [{
    name: 'embedding',
    dtype: 'float32',
    shape: [1, 768],
    description: 'Music embedding vector',
  }],
  prepareInputs(values: Record<string, any>) {
    let arr = flatten(values['features'] ?? [[]])
    if (arr.length !== 992 * 128) { const tmp = new Float32Array(992 * 128); tmp.set(arr.slice(0, 992 * 128)); arr = tmp }
    return { 'serving_default_args_0:0': new Tensor(arr, [1, 992, 128]) }
  },
  async parseOutputs(outputs) {
    const t = outputs['StatefulPartitionedCall:0']
    const data = await t.data()
    return { embedding: toNestedArray(data as Float32Array, [1, 768]) }
  },
}

export const textEncoder: ModelAdapter = {
  modelId: 'text_encoder',
  metadata: {
    name: 'Text Encoder',
    description: 'Encodes tokenized text (max 128 tokens) into a 768-dim text embedding',
    modelPath: '/models/text_encoder.tflite',
    tags: ['musiccoca', 'text', 'embedding'],
  },
  disabled: true, // Official google/magenta-realtime-2 HF repo contains MusicCoCa TFLites; keep disabled until this adapter contract is verified.
  inputSpecs: [
    {
      name: 'ids',
      dtype: 'int32',
      shape: [1, 128],
      description: 'Token IDs (pad to 128)',
    },
    {
      name: 'paddings',
      dtype: 'float32',
      shape: [1, 128],
      description: 'Attention mask (1.0 for real tokens, 0.0 for padding)',
      constraints: { min: 0, max: 1 },
    },
  ],
  outputSpecs: [{
    name: 'embedding',
    dtype: 'float32',
    shape: [1, 768],
    description: 'Text embedding vector',
  }],
  prepareInputs(values: Record<string, any>) {
    let ids = flatten(values['ids'] ?? [])
    ids = ids.map(Math.round) as any
    if (ids.length !== 128) { const tmp = new Float32Array(128); tmp.set(ids.slice(0, 128)); ids = tmp }
    let paddings = flatten(values['paddings'] ?? [])
    if (paddings.length !== 128) { const tmp = new Float32Array(128); tmp.set(paddings.slice(0, 128)); paddings = tmp }
    return {
      'serving_default_ids:0': new Tensor(new Int32Array(Array.from(ids)), [1, 128]),
      'serving_default_paddings:0': new Tensor(paddings, [1, 128]),
    }
  },
  async parseOutputs(outputs) {
    const t = outputs['StatefulPartitionedCall:0']
    const data = await t.data()
    return { embedding: toNestedArray(data as Float32Array, [1, 768]) }
  },
}

export const mapper: ModelAdapter = {
  modelId: 'mapper',
  metadata: {
    name: 'Mapper',
    description: 'Projects between music and text embedding spaces [1, 768] → [1, 768]',
    modelPath: '/models/mapper.tflite',
    tags: ['musiccoca', 'projection'],
  },
  disabled: true, // Official google/magenta-realtime-2 HF repo contains MusicCoCa TFLites; keep disabled until this adapter contract is verified.
  inputSpecs: [
    {
      name: 'input_a',
      dtype: 'float32',
      shape: [1, 768],
      description: 'First embedding (e.g. text embedding)',
    },
    {
      name: 'input_b',
      dtype: 'float32',
      shape: [1, 768],
      description: 'Second embedding (e.g. music embedding)',
    },
  ],
  outputSpecs: [{
    name: 'output',
    dtype: 'float32',
    shape: [1, 768],
    description: 'Projected embedding',
  }],
  prepareInputs(values: Record<string, any>) {
    let a = flatten(values['input_a'] ?? [])
    if (a.length !== 768) { const tmp = new Float32Array(768); tmp.set(a.slice(0, 768)); a = tmp }
    let b = flatten(values['input_b'] ?? [])
    if (b.length !== 768) { const tmp = new Float32Array(768); tmp.set(b.slice(0, 768)); b = tmp }
    return {
      'serving_default_args_0:0': new Tensor(a, [1, 768]),
      'serving_default_args_1:0': new Tensor(b, [1, 768]),
    }
  },
  async parseOutputs(outputs) {
    const t = outputs['StatefulPartitionedCall:0']
    const data = await t.data()
    return { output: toNestedArray(data as Float32Array, [1, 768]) }
  },
}

export const quantizer: ModelAdapter = {
  modelId: 'quantizer',
  metadata: {
    name: 'Pretrained Vector Quantizer',
    description: 'Quantizes a 768-dim embedding into 12 discrete tokens (int32)',
    modelPath: '/models/pretrained_vector_quantizer.tflite',
    tags: ['musiccoca', 'quantization'],
  },
  disabled: true, // Official google/magenta-realtime-2 HF repo contains MusicCoCa TFLites; keep disabled until this adapter contract is verified.
  inputSpecs: [{
    name: 'embedding',
    dtype: 'float32',
    shape: [1, 768],
    description: 'Embedding vector to quantize',
  }],
  outputSpecs: [{
    name: 'tokens',
    dtype: 'int32',
    shape: [12, 1],
    description: '12 discrete quantized tokens',
  }],
  prepareInputs(values: Record<string, any>) {
    let arr = flatten(values['embedding'] ?? [])
    if (arr.length !== 768) { const tmp = new Float32Array(768); tmp.set(arr.slice(0, 768)); arr = tmp }
    return { 'quantize_inputs:0': new Tensor(arr, [1, 768]) }
  },
  async parseOutputs(outputs) {
    const t = outputs['StatefulPartitionedCall_1:0']
    const data = await t.data()
    return { tokens: Array.from(data as Int32Array) }
  },
}

export const musicCocaAdapters: ModelAdapter[] = [
  audioPreprocessor,
  musicEncoder,
  textEncoder,
  mapper,
  quantizer,
]
