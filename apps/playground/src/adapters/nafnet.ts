import { Tensor } from '@litertjs/core'
import type { ModelAdapter, TensorSpec } from './types'
import { normalizeAndFormatImageData, resizeImageData } from '../imageUtils'

const INPUT_SPECS: TensorSpec[] = [
  {
    name: 'args_0',
    dtype: 'float32',
    shape: [1, 3, 256, 256],
    description: 'Noisy image (NCHW, 256x256, RGB, 0-1)',
  },
]

const OUTPUT_SPECS: TensorSpec[] = [
  {
    name: 'output',
    dtype: 'float32',
    shape: [1, 3, 256, 256],
    description: 'Denoised image (NCHW, 256x256)',
  },
]

export const nafnetAdapter: ModelAdapter = {
  modelId: 'nafnet',
  metadata: {
    name: 'NAFNet SIDD width32',
    description: 'Image denoising (SIDD benchmark)',
    modelPath: 'https://huggingface.co/litert-community/NAFNet-SIDD-width32-LiteRT/resolve/main/nafnet_sidd_width32_fp16.tflite',
    tags: ['vision', 'denoising'],
  },
  inputSpecs: INPUT_SPECS,
  outputSpecs: OUTPUT_SPECS,

  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const spec = INPUT_SPECS[0]
    const img = values['image'] as ImageData
    if (!img) throw new Error('Image data not provided for nafnet')
    const [, C, H, W] = spec.shape
    const resized = resizeImageData(img, W, H)
    return { [spec.name]: normalizeAndFormatImageData(resized, [1, C, H, W], { dataFormat: 'NCHW', colorOrder: 'RGB', normalization: '0-1' }) }
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
