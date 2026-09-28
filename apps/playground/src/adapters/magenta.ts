import { Tensor } from '@litertjs/core'
import type { ModelAdapter } from './types'
import { flatten, toNestedArray } from './util'

const INPUT_SPECS = [
  {
    name: 'input',
    dtype: 'float32' as const,
    shape: [1, 256, 1],
    description: 'Audio input frame (mean-centered normalized samples)'
  },
  {
    name: 'length',
    dtype: 'float32' as const,
    shape: [1],
    description: 'Generation length in seconds',
    constraints: { min: 0.5, max: 30.0 }
  },
  {
    name: 'temperature_harmonic',
    dtype: 'float32' as const,
    shape: [1],
    description: 'Harmonic temperature (higher = more random)',
    constraints: { min: 0.1, max: 5.0 }
  }
]

const OUTPUT_SPECS = [
  {
    name: 'output',
    dtype: 'float32' as const,
    shape: [1, 256, 1],
    description: 'Generated audio frame'
  },
  {
    name: 'state',
    dtype: 'float32' as const,
    shape: [1, 256],
    description: 'Recurrent state (feed back as input for continuation)'
  }
]

export const magentaAdapter: ModelAdapter = {
  modelId: 'magenta-realtime-2',
  metadata: {
    name: 'Magenta RealTime 2',
    description: 'Real-time music generation model',
    modelPath: '/models/magenta.tflite',
    tags: ['audio', 'music', 'magenta']
  },
  disabled: true, // Official HF repo exists, but Magenta RealTime 2 is a multi-component pipeline; this single-graph placeholder is not a verified contract.
  inputSpecs: INPUT_SPECS,
  outputSpecs: OUTPUT_SPECS,

  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const inputs: Record<string, Tensor> = {}
    for (const spec of INPUT_SPECS) {
      const raw = values[spec.name]
      if (raw === undefined) throw new Error(`Missing input: ${spec.name}`)
      inputs[spec.name] = new Tensor(flatten(raw), spec.shape)
    }
    return inputs
  },

  async parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
    const result: Record<string, any> = {}
    for (const spec of OUTPUT_SPECS) {
      const tensor = outputs[spec.name]
      if (!tensor) throw new Error(`Missing output: ${spec.name}`)
      const data = await tensor.data()
      if (spec.shape.length > 1) {
        const dims = spec.shape.slice(1)
        result[spec.name] = toNestedArray(data as Float32Array, dims)
      } else {
        result[spec.name] = Array.from(data as Float32Array)
      }
    }
    return result
  }
}

export const magentaAdapters: ModelAdapter[] = [magentaAdapter]
