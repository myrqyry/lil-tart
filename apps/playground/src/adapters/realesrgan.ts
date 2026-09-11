import { Tensor } from '@litertjs/core'
import type { ModelAdapter, TensorSpec } from './types'
import { normalizeAndFormatImageData, resizeImageData } from '../imageUtils'

const INPUT_SPECS: TensorSpec[] = [
  {
    name: 'args_0',
    dtype: 'float32',
    shape: [1, 128, 128, 3],
    description: 'Input image (NHWC, 128x128, RGB, 0-1)',
  },
]

const OUTPUT_SPECS: TensorSpec[] = [
  {
    name: 'output',
    dtype: 'float32',
    shape: [1, 3, 512, 512],
    description: 'Upscaled image (NCHW, 512x512, 4x upscale)',
  },
]

export const realesrganAdapter: ModelAdapter = {
  modelId: 'realesrgan',
  metadata: {
    name: 'Real-ESRGAN x4v3',
    description: 'General image super-resolution (4x upscale)',
    modelPath: 'https://huggingface.co/litert-community/real-esrgan-x4v3-litert/resolve/main/realesr_general_x4v3.tflite',
    tags: ['vision', 'super-resolution'],
  },
  inputSpecs: INPUT_SPECS,
  outputSpecs: OUTPUT_SPECS,

  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const spec = INPUT_SPECS[0]
    const img = values['image'] as ImageData
    if (!img) throw new Error('Image data not provided for realesrgan')
    const [, H, W] = spec.shape
    const resized = resizeImageData(img, W, H)
    return { [spec.name]: normalizeAndFormatImageData(resized, spec.shape, { dataFormat: 'NHWC', colorOrder: 'RGB', normalization: '0-1' }) }
  },

  async parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
    const entries = await Promise.all(
      Object.entries(outputs).map(async ([name, tensor]) => {
        const data = await tensor.data()
        return [name, Array.from(new Float32Array(data))]
      })
    )
    return Object.fromEntries(entries)
  },
}
