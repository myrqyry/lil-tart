import type { ModelAdapter } from './types'
import { Tensor } from '@litertjs/core'
import { normalizeAndFormatImageData, resizeImageData, tensorToImageData } from '../imageUtils'

const inpSpec = (n: string, s: number[], d: 'float32' | 'int32', desc: string) =>
  ({ name: n, dtype: d, shape: s, description: desc })
const outSpec = (n: string, s: number[], d: 'float32' | 'int32', desc: string) =>
  ({ name: n, dtype: d, shape: s, description: desc })

function nchwToImageData(data: Float32Array, w: number, h: number): ImageData {
  const out = new ImageData(w, h)
  const plane = w * h
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x
      out.data[idx * 4 + 0] = Math.max(0, Math.min(255, Math.round(data[0 * plane + idx] * 255)))
      out.data[idx * 4 + 1] = Math.max(0, Math.min(255, Math.round(data[1 * plane + idx] * 255)))
      out.data[idx * 4 + 2] = Math.max(0, Math.min(255, Math.round(data[2 * plane + idx] * 255)))
      out.data[idx * 4 + 3] = 255
    }
  }
  return out
}

function nchwToImageDataMinusOneToOne(data: Float32Array, w: number, h: number): ImageData {
  const out = new ImageData(w, h)
  const plane = w * h
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x
      out.data[idx * 4 + 0] = Math.max(0, Math.min(255, Math.round((data[0 * plane + idx] * 0.5 + 0.5) * 255)))
      out.data[idx * 4 + 1] = Math.max(0, Math.min(255, Math.round((data[1 * plane + idx] * 0.5 + 0.5) * 255)))
      out.data[idx * 4 + 2] = Math.max(0, Math.min(255, Math.round((data[2 * plane + idx] * 0.5 + 0.5) * 255)))
      out.data[idx * 4 + 3] = 255
    }
  }
  return out
}

function buildNhwcRawTensor(imageData: ImageData, w: number, h: number, bgr = false): Tensor {
  const data = new Float32Array(1 * h * w * 3)
  const srcW = imageData.width
  // resize already done by caller; just interleave
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const srcIdx = (y * srcW + x) * 4
      const r = imageData.data[srcIdx]
      const g = imageData.data[srcIdx + 1]
      const b = imageData.data[srcIdx + 2]
      const base = (y * w + x) * 3
      if (bgr) {
        data[base + 0] = b
        data[base + 1] = g
        data[base + 2] = r
      } else {
        data[base + 0] = r
        data[base + 1] = g
        data[base + 2] = b
      }
    }
  }
  return new Tensor(data, [1, h, w, 3])
}

function letterbox(imageData: ImageData, size: number, pad = 114): ImageData {
  const scale = Math.min(size / imageData.width, size / imageData.height)
  const w = Math.max(1, Math.round(imageData.width * scale))
  const h = Math.max(1, Math.round(imageData.height * scale))
  const src = new OffscreenCanvas(imageData.width, imageData.height)
  src.getContext('2d')!.putImageData(imageData, 0, 0)
  const dst = new OffscreenCanvas(size, size)
  const ctx = dst.getContext('2d')!
  ctx.fillStyle = `rgb(${pad},${pad},${pad})`
  ctx.fillRect(0, 0, size, size)
  ctx.drawImage(src, Math.round((size - w) / 2), Math.round((size - h) / 2), w, h)
  return ctx.getImageData(0, 0, size, size)
}

export const headpose6drepnetAdapter: ModelAdapter = {
  modelId: '6drepnet',
  metadata: { name: '6DRepNet — Head Pose', description: '6D head pose estimation (Euler angles)', modelPath: 'https://huggingface.co/litert-community/6DRepNet-HeadPose-LiteRT/resolve/main/6drepnet.tflite', tags: ['vision', 'pose'] },
  inputSpecs: [inpSpec('input', [1, 3, 224, 224], 'float32', 'RGB ImageNet-normalized NCHW')],
  outputSpecs: [outSpec('output', [6], 'float32', '6D rotation vector → Gram-Schmidt → Euler')],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const imageData = values['image'] as ImageData
    if (!imageData) throw new Error('Image data not provided for 6drepnet')
    const [, C, H, W] = this.inputSpecs[0].shape
    const resized = resizeImageData(imageData, W, H)
    return { input: normalizeAndFormatImageData(resized, [1, C, H, W], { dataFormat: 'NCHW', normalization: 'imagenet' }) }
  },
  async parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
    const t = outputs['output']
    if (!t) throw new Error('Missing output tensor')
    const data = await t.data() as Float32Array
    return { output: Array.from(data) }
  },
}

export const blazeFaceAdapter: ModelAdapter = {
  modelId: 'blaze-face',
  disabled: true,
  metadata: { name: 'BlazeFace — Face Detection', description: 'MediaPipe face detection (full-range) — needs locating', modelPath: '/models/blaze-face/blaze_face_full_range.tflite', tags: ['vision', 'face'] },
  inputSpecs: [inpSpec('input', [1, 128, 128, 3], 'float32', 'RGB 0-255 NHWC')],
  outputSpecs: [
    outSpec('regressors', [1, 896, 16], 'float32', 'Bounding box + 6 keypoint regressors'),
    outSpec('classificators', [1, 896, 1], 'float32', 'Face presence scores'),
  ],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const imageData = values['image'] as ImageData
    if (!imageData) throw new Error('Image data not provided for blaze-face')
    const resized = resizeImageData(imageData, 128, 128)
    return { input: buildNhwcRawTensor(resized, 128, 128) }
  },
  async parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
    const regressors = outputs['regressors']
    const classificators = outputs['classificators']
    if (!regressors || !classificators) throw new Error('Missing blaze-face outputs')
    return {
      regressors: Array.from(await regressors.data() as Float32Array),
      classificators: Array.from(await classificators.data() as Float32Array),
    }
  },
}

export const yoloxAdapter: ModelAdapter = {
  modelId: 'yolox',
  metadata: { name: 'YOLOX-M — Object Detection', description: 'YOLOX-M COCO detection (640×640)', modelPath: 'https://huggingface.co/litert-community/yolox-m-litert/resolve/main/yolox_m.tflite', tags: ['vision', 'detection'] },
  inputSpecs: [inpSpec('images', [1, 640, 640, 3], 'float32', 'BGR 0-255 NHWC, letterbox pad 114')],
  outputSpecs: [outSpec('output', [1, 8400, 85], 'float32', 'Raw heads: 4 box + 1 obj + 80 class')],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const imageData = values['image'] as ImageData
    if (!imageData) throw new Error('Image data not provided for yolox')
    const padded = letterbox(imageData, 640)
    return { images: buildNhwcRawTensor(padded, 640, 640, true) }
  },
  async parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
    const t = outputs['output']
    if (!t) throw new Error('Missing yolox output')
    return { output: Array.from(await t.data() as Float32Array) }
  },
}

export const u2netAdapter: ModelAdapter = {
  modelId: 'u2net',
  metadata: { name: 'U2-Net — Portrait Sketch', description: 'Photo to pencil line drawing', modelPath: 'https://huggingface.co/litert-community/U2Net-Portrait-Sketch-LiteRT/resolve/main/portrait.tflite', tags: ['vision', 'creative'] },
  inputSpecs: [inpSpec('input', [1, 3, 512, 512], 'float32', 'RGB ImageNet-normalized NCHW')],
  outputSpecs: [outSpec('output', [1, 1, 512, 512], 'float32', 'Sketch map [0,1], invert for dark-on-white')],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const imageData = values['image'] as ImageData
    if (!imageData) throw new Error('Image data not provided for u2net')
    const [, C, H, W] = this.inputSpecs[0].shape
    const resized = resizeImageData(imageData, W, H)
    return { input: normalizeAndFormatImageData(resized, [1, C, H, W], { dataFormat: 'NCHW', normalization: 'imagenet' }) }
  },
  async parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
    const t = outputs['output']
    if (!t) throw new Error('Missing u2net output')
    const [, , H, W] = this.outputSpecs[0].shape
    return { output: await tensorToImageData(t, W, H) }
  },
}

export const edsrAdapter: ModelAdapter = {
  modelId: 'edsr',
  metadata: { name: 'EDSR ×4 — Super Resolution', description: '4× super resolution (128→512)', modelPath: 'https://huggingface.co/litert-community/EDSR-x4-LiteRT/resolve/main/edsr.tflite', tags: ['vision', 'enhancement'] },
  inputSpecs: [inpSpec('input', [1, 3, 128, 128], 'float32', 'RGB x/255 NCHW')],
  outputSpecs: [outSpec('output', [1, 3, 512, 512], 'float32', 'RGB 0-1 NCHW, clamp ×255')],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const imageData = values['image'] as ImageData
    if (!imageData) throw new Error('Image data not provided for edsr')
    const [, C, H, W] = this.inputSpecs[0].shape
    const resized = resizeImageData(imageData, W, H)
    return { input: normalizeAndFormatImageData(resized, [1, C, H, W], { dataFormat: 'NCHW', normalization: '0-1' }) }
  },
  async parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
    const t = outputs['output']
    if (!t) throw new Error('Missing edsr output')
    const data = await t.data() as Float32Array
    const [, , H, W] = this.outputSpecs[0].shape
    return { output: nchwToImageData(data, W, H) }
  },
}

export const miganAdapter: ModelAdapter = {
  modelId: 'migan',
  metadata: { name: 'MI-GAN — Image Inpainting', description: 'Object removal / image inpainting (512×512)', modelPath: 'https://huggingface.co/litert-community/MI-GAN-512-Places2-LiteRT/resolve/main/migan_fp16.tflite', tags: ['vision', 'inpainting'] },
  inputSpecs: [inpSpec('input', [1, 4, 512, 512], 'float32', 'concat(mask-0.5, rgb·mask) NCHW')],
  outputSpecs: [outSpec('output', [1, 3, 512, 512], 'float32', 'Inpainted RGB [-1,1] NCHW')],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const imageData = values['image'] as ImageData
    if (!imageData) throw new Error('Image data not provided for migan')
    const maskData = values['mask'] as ImageData | undefined
    const resized = resizeImageData(imageData, 512, 512)
    const maskResized = maskData ? resizeImageData(maskData, 512, 512) : undefined
    const data = new Float32Array(1 * 4 * 512 * 512)
    const plane = 512 * 512
    for (let y = 0; y < 512; y++) {
      for (let x = 0; x < 512; x++) {
        const idx = (y * 512 + x) * 4
        const r = resized.data[idx] / 255
        const g = resized.data[idx + 1] / 255
        const b = resized.data[idx + 2] / 255
        const maskVal = maskResized ? maskResized.data[idx] / 255 : 1
        // channel 0: mask - 0.5
        data[0 * plane + y * 512 + x] = maskVal - 0.5
        // channels 1-3: rgb * mask
        data[1 * plane + y * 512 + x] = r * maskVal
        data[2 * plane + y * 512 + x] = g * maskVal
        data[3 * plane + y * 512 + x] = b * maskVal
      }
    }
    return { input: new Tensor(data, [1, 4, 512, 512]) }
  },
  async parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
    const t = outputs['output']
    if (!t) throw new Error('Missing migan output')
    const data = await t.data() as Float32Array
    return { output: nchwToImageDataMinusOneToOne(data, 512, 512) }
  },
}

const sharedStyleSpecs = {
  inputSpecs: [inpSpec('input', [1, 3, 256, 256], 'float32', 'RGB 0-255 NCHW (no normalization)')],
  outputSpecs: [outSpec('output', [1, 3, 256, 256], 'float32', 'RGB 0-255 NCHW (clamp)')],
}

function makeStyleAdapter(modelId: string, name: string, modelPath: string): ModelAdapter {
  return {
    modelId,
    metadata: { name, description: `Fast Neural Style Transfer (${name.split('—')[1]?.trim() ?? modelId})`, modelPath, tags: ['vision', 'creative'] },
    ...sharedStyleSpecs,
    prepareInputs(values: Record<string, any>): Record<string, Tensor> {
      const imageData = values['image'] as ImageData
      if (!imageData) throw new Error(`Image data not provided for ${modelId}`)
      const resized = resizeImageData(imageData, 256, 256)
      // raw 0-255 NCHW — build planar without normalization
      const data = new Float32Array(1 * 3 * 256 * 256)
      const plane = 256 * 256
      for (let y = 0; y < 256; y++) {
        for (let x = 0; x < 256; x++) {
          const idx = (y * 256 + x) * 4
          data[0 * plane + y * 256 + x] = resized.data[idx]
          data[1 * plane + y * 256 + x] = resized.data[idx + 1]
          data[2 * plane + y * 256 + x] = resized.data[idx + 2]
        }
      }
      return { input: new Tensor(data, [1, 3, 256, 256]) }
    },
    async parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
      const t = outputs['output']
      if (!t) throw new Error(`Missing output for ${modelId}`)
      const data = await t.data() as Float32Array
      // output is 0-255 NCHW — convert to ImageData
      const out = new ImageData(256, 256)
      const plane = 256 * 256
      for (let y = 0; y < 256; y++) {
        for (let x = 0; x < 256; x++) {
          const idx = y * 256 + x
          out.data[idx * 4 + 0] = Math.max(0, Math.min(255, Math.round(data[0 * plane + idx])))
          out.data[idx * 4 + 1] = Math.max(0, Math.min(255, Math.round(data[1 * plane + idx])))
          out.data[idx * 4 + 2] = Math.max(0, Math.min(255, Math.round(data[2 * plane + idx])))
          out.data[idx * 4 + 3] = 255
        }
      }
      return { output: out }
    },
  }
}

export const styleAdapters: ModelAdapter[] = [
  makeStyleAdapter('style-candy', 'Neural Style — Candy', 'https://huggingface.co/litert-community/Fast-Neural-Style-LiteRT/resolve/main/style_candy_fp16.tflite'),
  makeStyleAdapter('style-mosaic', 'Neural Style — Mosaic', 'https://huggingface.co/litert-community/Fast-Neural-Style-LiteRT/resolve/main/style_mosaic_fp16.tflite'),
  makeStyleAdapter('style-rain-princess', 'Neural Style — Rain Princess', 'https://huggingface.co/litert-community/Fast-Neural-Style-LiteRT/resolve/main/style_rain_princess_fp16.tflite'),
  makeStyleAdapter('style-udnie', 'Neural Style — Udnie', 'https://huggingface.co/litert-community/Fast-Neural-Style-LiteRT/resolve/main/style_udnie_fp16.tflite'),
]

export const visionAdapters: ModelAdapter[] = [
  headpose6drepnetAdapter,
  blazeFaceAdapter,
  yoloxAdapter,
  u2netAdapter,
  edsrAdapter,
  miganAdapter,
  ...styleAdapters,
]
