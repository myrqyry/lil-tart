import type { ModelAdapter } from './types'
import { Tensor } from '@litertjs/core'
import { normalizeAndFormatImageData, resizeImageData, tensorToImageData } from '../imageUtils'

export const sam2EncoderAdapter: ModelAdapter = {
  modelId: 'sam2-encoder',
  metadata: { name: 'SAM2 — Image Encoder', description: 'SAM2.1 Hiera-Tiny image encoder (1024×1024)', modelPath: 'https://huggingface.co/litert-community/SAM2.1-Hiera-Tiny-Image-Encoder/resolve/main/sam2_tiny_image_encoder_v2_fp16.tflite', tags: ['vision', 'segmentation'] },
  inputSpecs: [
    { name: 'image', dtype: 'float32', shape: [1, 3, 1024, 1024], description: 'RGB ImageNet-normalized NCHW' },
  ],
  outputSpecs: [
    { name: 'image_embeddings', dtype: 'float32', shape: [1, 256, 64, 64], description: 'Image feature embeddings' },
    { name: 'feat_s1', dtype: 'float32', shape: [1, 64, 128, 128], description: 'FPN feature map s1' },
    { name: 'feat_s0', dtype: 'float32', shape: [1, 32, 256, 256], description: 'FPN feature map s0' },
  ],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const imageData = values['image'] as ImageData
    if (!imageData) throw new Error('Image data not provided for sam2-encoder')
    const [, C, H, W] = this.inputSpecs[0].shape
    const resized = resizeImageData(imageData, W, H)
    return { image: normalizeAndFormatImageData(resized, [1, C, H, W], { dataFormat: 'NCHW', normalization: 'imagenet' }) }
  },
  async parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
    const result: Record<string, any> = {}
    for (const spec of this.outputSpecs) {
      const t = outputs[spec.name]
      if (!t) throw new Error(`Missing sam2-encoder output: ${spec.name}`)
      result[spec.name] = await t.data()
    }
    return result
  },
}

export const sam2DecoderAdapter: ModelAdapter = {
  modelId: 'sam2-decoder',
  metadata: { name: 'SAM2 — Mask Decoder', description: 'SAM2.1 Hiera-Tiny mask decoder (promptable segmentation)', modelPath: 'https://huggingface.co/litert-community/SAM2.1-Hiera-Tiny-Mask-Decoder/resolve/main/sam2_tiny_mask_decoder_v2_fp16.tflite', tags: ['vision', 'segmentation'] },
  // The graph contract is verified, but this component is not a usable standalone playground model:
  // it needs encoder features plus host-side point encoding. Keep it unavailable until those steps are one pipeline.
  disabled: true,
  requiredBackend: 'wasm', // LiteRT.js 2.5.3 browser sweep: WebGPU runs but does not match WASM for this graph.
  inputSpecs: [
    { name: 'image_embeddings', dtype: 'float32', shape: [1, 256, 64, 64], description: 'From SAM2 image encoder v2' },
    { name: 'feat_s1', dtype: 'float32', shape: [1, 64, 128, 128], description: 'FPN feature s1 from encoder' },
    { name: 'feat_s0', dtype: 'float32', shape: [1, 32, 256, 256], description: 'FPN feature s0 from encoder' },
    { name: 'sparse_prompt', dtype: 'float32', shape: [1, 2, 256], description: 'Host-side prompt encoding (point coords + labels)' },
  ],
  outputSpecs: [
    { name: 'pred_masks', dtype: 'float32', shape: [1, 3, 256, 256], description: '3 mask-logit predictions' },
    { name: 'iou_scores', dtype: 'float32', shape: [1, 3], description: 'IoU scores per mask' },
  ],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const out: Record<string, Tensor> = {}
    for (const spec of this.inputSpecs) {
      const val = values[spec.name]
      if (!val) throw new Error(`Missing input for sam2-decoder: ${spec.name}`)
      if (val instanceof Tensor) {
        out[spec.name] = val
      } else if (val instanceof Float32Array) {
        out[spec.name] = new Tensor(val, spec.shape)
      } else if (Array.isArray(val)) {
        out[spec.name] = new Tensor(new Float32Array(val.flat(Infinity) as number[]), spec.shape)
      } else {
        throw new Error(`Invalid input type for ${spec.name}: expected Tensor or Float32Array`)
      }
    }
    return out
  },
  async parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
    const masks = outputs['pred_masks']
    const iou = outputs['iou_scores']
    if (!masks || !iou) throw new Error('Missing sam2-decoder outputs')
    const masksData = await masks.data() as Float32Array
    const iouData = await iou.data() as Float32Array
    // masks is [1,3,256,256] — split into 3 separate ImageData masks
    const plane = 256 * 256
    const masksOut: ImageData[] = []
    for (let m = 0; m < 3; m++) {
      const slice = masksData.slice(m * plane, (m + 1) * plane)
      const tensor = new Tensor(slice, [1, 1, 256, 256])
      masksOut.push(await tensorToImageData(tensor, 256, 256))
    }
    return { pred_masks: masksOut, iou_scores: Array.from(iouData) }
  },
}

export const sam2Adapters: ModelAdapter[] = [sam2EncoderAdapter, sam2DecoderAdapter]
