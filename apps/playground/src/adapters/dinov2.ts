import { Tensor } from '@litertjs/core'
import type { ModelAdapter, TensorSpec } from './types'
import { normalizeAndFormatImageData, resizeImageData } from '../imageUtils'

const INPUT_SPECS: TensorSpec[] = [
  {
    name: 'args_0',
    dtype: 'float32',
    shape: [1, 3, 448, 448],
    description: 'Input image (NCHW, 448x448, RGB, ImageNet-normalized)',
  },
]

const OUTPUT_SPECS: TensorSpec[] = [
  {
    name: 'output',
    dtype: 'float32',
    shape: [1, 1024, 384],
    description: 'DINOv2 patch embeddings (1024 patches, 384-dim each)',
  },
]

export const dinov2Adapter: ModelAdapter = {
  modelId: 'dinov2',
  metadata: {
    name: 'DINOv2 ViT-S/14',
    description: 'Vision Transformer patch embeddings (384-dim)',
    modelPath: 'https://huggingface.co/litert-community/DINOv2-ViT-S14-LiteRT/resolve/main/dinov2_s_fp16.tflite',
    tags: ['vision', 'embeddings'],
  },
  inputSpecs: INPUT_SPECS,
  outputSpecs: OUTPUT_SPECS,

  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const spec = INPUT_SPECS[0]
    const img = values['image'] as ImageData
    if (!img) throw new Error('Image data not provided for dinov2')
    const [, C, H, W] = spec.shape
    const resized = resizeImageData(img, W, H)
    return { [spec.name]: normalizeAndFormatImageData(resized, [1, C, H, W], { dataFormat: 'NCHW', colorOrder: 'RGB', normalization: 'imagenet' }) }
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
