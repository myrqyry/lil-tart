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

function firstOutput(outputs: Record<string, Tensor>, name: string): Tensor {
  const t = outputs[name] ?? Object.values(outputs)[0]
  if (!t) throw new Error('Missing output tensor')
  return t
}

function buildNchwBgrRawTensor(imageData: ImageData, w: number, h: number): Tensor {
  const data = new Float32Array(3 * h * w)
  const plane = h * w
  const srcW = imageData.width
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const s = (y * srcW + x) * 4
      const idx = y * w + x
      data[0 * plane + idx] = imageData.data[s + 2]
      data[1 * plane + idx] = imageData.data[s + 1]
      data[2 * plane + idx] = imageData.data[s + 0]
    }
  }
  return new Tensor(data, [1, 3, h, w])
}

function maskToImageData(data: Float32Array, w: number, h: number): ImageData {
  const out = new ImageData(w, h)
  for (let i = 0; i < w * h; i++) {
    const v = Math.max(0, Math.min(255, Math.round(data[i] * 255)))
    out.data[i * 4] = v
    out.data[i * 4 + 1] = v
    out.data[i * 4 + 2] = v
    out.data[i * 4 + 3] = 255
  }
  return out
}

const SEG_PALETTE = [
  [0, 0, 0], [220, 20, 60], [0, 128, 0], [30, 144, 255], [255, 215, 0],
  [148, 0, 211], [255, 140, 0], [0, 206, 209], [255, 105, 180], [128, 128, 0],
  [70, 130, 180], [210, 105, 30], [46, 139, 87], [199, 21, 133], [72, 61, 139],
  [0, 100, 0], [255, 99, 71], [95, 158, 160], [255, 20, 147],
]

function argmaxColorize(data: Float32Array, c: number, h: number, w: number): ImageData {
  const out = new ImageData(w, h)
  const plane = h * w
  for (let i = 0; i < plane; i++) {
    let best = 0
    let bestV = data[i]
    for (let k = 1; k < c; k++) {
      const v = data[k * plane + i]
      if (v > bestV) { bestV = v; best = k }
    }
    const col = SEG_PALETTE[best % SEG_PALETTE.length]
    out.data[i * 4] = col[0]
    out.data[i * 4 + 1] = col[1]
    out.data[i * 4 + 2] = col[2]
    out.data[i * 4 + 3] = 255
  }
  return out
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

export const yunetAdapter: ModelAdapter = {
  modelId: 'yunet-face',
  metadata: { name: 'YuNet — Face Detection', description: 'YuNet face detection with 5-point landmarks (640×640)', modelPath: 'https://huggingface.co/litert-community/YuNet-Face-LiteRT/resolve/main/yunet_fp16.tflite', tags: ['vision', 'face', 'detection'] },
  inputSpecs: [inpSpec('image', [1, 3, 640, 640], 'float32', 'BGR 0-255 NCHW (no normalization)')],
  outputSpecs: [
    outSpec('cls_8', [1, 6400, 1], 'float32', 'Class score, stride 8'),
    outSpec('cls_16', [1, 1600, 1], 'float32', 'Class score, stride 16'),
    outSpec('cls_32', [1, 400, 1], 'float32', 'Class score, stride 32'),
    outSpec('obj_8', [1, 6400, 1], 'float32', 'Objectness, stride 8'),
    outSpec('obj_16', [1, 1600, 1], 'float32', 'Objectness, stride 16'),
    outSpec('obj_32', [1, 400, 1], 'float32', 'Objectness, stride 32'),
    outSpec('bbox_8', [1, 6400, 4], 'float32', 'Box deltas, stride 8'),
    outSpec('bbox_16', [1, 1600, 4], 'float32', 'Box deltas, stride 16'),
    outSpec('bbox_32', [1, 400, 4], 'float32', 'Box deltas, stride 32'),
    outSpec('kps_8', [1, 6400, 10], 'float32', '5 landmarks, stride 8'),
    outSpec('kps_16', [1, 1600, 10], 'float32', '5 landmarks, stride 16'),
    outSpec('kps_32', [1, 400, 10], 'float32', '5 landmarks, stride 32'),
  ],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const imageData = values['image'] as ImageData
    if (!imageData) throw new Error('Image data not provided for yunet-face')
    const resized = resizeImageData(imageData, 640, 640)
    return { image: buildNchwBgrRawTensor(resized, 640, 640) }
  },
  async parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
    const result: Record<string, any> = {}
    for (const [k, t] of Object.entries(outputs)) result[k] = Array.from(await t.data() as Float32Array)
    return result
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

function makeYoloxAdapter(modelId: string, label: string, modelPath: string, size: number, rows: number): ModelAdapter {
  return {
    modelId,
    metadata: { name: `YOLOX-${label} — Object Detection`, description: `YOLOX-${label} COCO detection (${size}×${size})`, modelPath, tags: ['vision', 'detection'] },
    inputSpecs: [inpSpec('images', [1, size, size, 3], 'float32', 'BGR 0-255 NHWC, letterbox pad 114')],
    outputSpecs: [outSpec('output', [1, rows, 85], 'float32', 'Raw heads: 4 box + 1 obj + 80 class')],
    prepareInputs(values: Record<string, any>): Record<string, Tensor> {
      const imageData = values['image'] as ImageData
      if (!imageData) throw new Error(`Image data not provided for ${modelId}`)
      const padded = letterbox(imageData, size)
      return { images: buildNhwcRawTensor(padded, size, size, true) }
    },
    async parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
      const t = firstOutput(outputs, 'output')
      return { output: Array.from(await t.data() as Float32Array) }
    },
  }
}

export const yoloxNanoAdapter = makeYoloxAdapter('yolox-nano', 'Nano', 'https://huggingface.co/litert-community/yolox-nano-litert/resolve/main/yolox_nano.tflite', 416, 3549)
export const yoloxTinyAdapter = makeYoloxAdapter('yolox-tiny', 'Tiny', 'https://huggingface.co/litert-community/yolox-tiny-litert/resolve/main/yolox_tiny.tflite', 416, 3549)
export const yoloxSAdapter = makeYoloxAdapter('yolox-s', 'S', 'https://huggingface.co/litert-community/yolox-s-litert/resolve/main/yolox_s.tflite', 640, 8400)

export const sinetAdapter: ModelAdapter = {
  modelId: 'sinet-v2',
  metadata: { name: 'SINet-V2 — Camouflage Detection', description: 'Camouflaged object segmentation', modelPath: 'https://huggingface.co/litert-community/SINet-V2-Camouflage-LiteRT/resolve/main/sinet.tflite', tags: ['vision', 'segmentation'] },
  inputSpecs: [inpSpec('input', [1, 3, 352, 352], 'float32', 'RGB ImageNet-normalized NCHW')],
  outputSpecs: [outSpec('output', [1, 1, 352, 352], 'float32', 'Camouflaged-object probability')],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const imageData = values['image'] as ImageData
    if (!imageData) throw new Error('Image data not provided for sinet-v2')
    const resized = resizeImageData(imageData, 352, 352)
    return { input: normalizeAndFormatImageData(resized, [1, 3, 352, 352], { dataFormat: 'NCHW', normalization: 'imagenet' }) }
  },
  async parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
    const data = await firstOutput(outputs, 'output').data() as Float32Array
    return { output: maskToImageData(data, 352, 352) }
  },
}

export const disAdapter: ModelAdapter = {
  modelId: 'dis-isnet',
  metadata: { name: 'DIS-ISNet — Dichotomous Segmentation', description: 'High-accuracy dichotomous image segmentation', modelPath: 'https://huggingface.co/litert-community/DIS-ISNet-LiteRT/resolve/main/dis.tflite', tags: ['vision', 'segmentation'] },
  inputSpecs: [inpSpec('input', [1, 3, 1024, 1024], 'float32', 'RGB (x/255 - 0.5) NCHW')],
  outputSpecs: [outSpec('output', [1, 1, 1024, 1024], 'float32', 'Foreground probability')],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const imageData = values['image'] as ImageData
    if (!imageData) throw new Error('Image data not provided for dis-isnet')
    const resized = resizeImageData(imageData, 1024, 1024)
    return { input: normalizeAndFormatImageData(resized, [1, 3, 1024, 1024], { dataFormat: 'NCHW', normalization: 'imagenet', mean: [0.5, 0.5, 0.5], std: [1, 1, 1] }) }
  },
  async parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
    const data = await firstOutput(outputs, 'output').data() as Float32Array
    return { output: maskToImageData(data, 1024, 1024) }
  },
}

export const unisalAdapter: ModelAdapter = {
  modelId: 'unisal',
  metadata: { name: 'UniSal — Saliency Detection', description: 'Unified saliency / visual attention map', modelPath: 'https://huggingface.co/litert-community/UniSal-Saliency-LiteRT/resolve/main/unisal_fp16.tflite', tags: ['vision', 'saliency'] },
  inputSpecs: [inpSpec('input', [1, 3, 256, 256], 'float32', 'RGB ImageNet-normalized NCHW')],
  outputSpecs: [outSpec('output', [1, 1, 256, 256], 'float32', 'Saliency map (higher = attended)')],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const imageData = values['image'] as ImageData
    if (!imageData) throw new Error('Image data not provided for unisal')
    const resized = resizeImageData(imageData, 256, 256)
    return { input: normalizeAndFormatImageData(resized, [1, 3, 256, 256], { dataFormat: 'NCHW', normalization: 'imagenet' }) }
  },
  async parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
    const data = await firstOutput(outputs, 'output').data() as Float32Array
    return { output: maskToImageData(data, 256, 256) }
  },
}

export const bisenetFaceAdapter: ModelAdapter = {
  modelId: 'bisenet-face-parsing',
  metadata: { name: 'BiSeNet — Face Parsing', description: '19-class face parsing', modelPath: 'https://huggingface.co/litert-community/BiSeNet-Face-Parsing-LiteRT/resolve/main/faceparsing.tflite', tags: ['vision', 'segmentation', 'face'] },
  inputSpecs: [inpSpec('input', [1, 3, 512, 512], 'float32', 'RGB ImageNet-normalized NCHW')],
  outputSpecs: [outSpec('output', [1, 19, 512, 512], 'float32', '19-class logits')],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const imageData = values['image'] as ImageData
    if (!imageData) throw new Error('Image data not provided for bisenet-face-parsing')
    const resized = resizeImageData(imageData, 512, 512)
    return { input: normalizeAndFormatImageData(resized, [1, 3, 512, 512], { dataFormat: 'NCHW', normalization: 'imagenet' }) }
  },
  async parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
    const data = await firstOutput(outputs, 'output').data() as Float32Array
    return { output: argmaxColorize(data, 19, 512, 512) }
  },
}

export const pidnetAdapter: ModelAdapter = {
  modelId: 'pidnet-s-cityscapes',
  metadata: { name: 'PIDNet-S — Cityscapes Segmentation', description: '19-class urban scene segmentation (1/8 resolution)', modelPath: 'https://huggingface.co/litert-community/PIDNet-S-Cityscapes-LiteRT/resolve/main/pidnet_s.tflite', tags: ['vision', 'segmentation'] },
  inputSpecs: [inpSpec('input', [1, 3, 1024, 1024], 'float32', 'RGB ImageNet-normalized NCHW')],
  outputSpecs: [outSpec('output', [1, 19, 128, 128], 'float32', '19-class logits at 1/8 resolution')],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const imageData = values['image'] as ImageData
    if (!imageData) throw new Error('Image data not provided for pidnet-s-cityscapes')
    const resized = resizeImageData(imageData, 1024, 1024)
    return { input: normalizeAndFormatImageData(resized, [1, 3, 1024, 1024], { dataFormat: 'NCHW', normalization: 'imagenet' }) }
  },
  async parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
    const data = await firstOutput(outputs, 'output').data() as Float32Array
    return { output: argmaxColorize(data, 19, 128, 128) }
  },
}

export const twinliteAdapter: ModelAdapter = {
  modelId: 'twinlitenet',
  metadata: { name: 'TwinLiteNet — Drivable Area & Lanes', description: 'Drivable-area and lane-line segmentation (360×640)', modelPath: 'https://huggingface.co/litert-community/TwinLiteNet-LiteRT/resolve/main/twinlite.tflite', tags: ['vision', 'segmentation'] },
  inputSpecs: [inpSpec('input', [1, 3, 360, 640], 'float32', 'RGB x/255 NCHW')],
  outputSpecs: [
    outSpec('drivable_area', [1, 2, 360, 640], 'float32', 'Drivable-area logits'),
    outSpec('lane_line', [1, 2, 360, 640], 'float32', 'Lane-line logits'),
  ],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const imageData = values['image'] as ImageData
    if (!imageData) throw new Error('Image data not provided for twinlitenet')
    const resized = resizeImageData(imageData, 640, 360)
    return { input: normalizeAndFormatImageData(resized, [1, 3, 360, 640], { dataFormat: 'NCHW', normalization: '0-1' }) }
  },
  async parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
    const [a, b] = Object.values(outputs)
    if (!a || !b) throw new Error('Missing twinlitenet outputs')
    return {
      drivable_area: argmaxColorize(await a.data() as Float32Array, 2, 360, 640),
      lane_line: argmaxColorize(await b.data() as Float32Array, 2, 360, 640),
    }
  },
}

export const midasAdapter: ModelAdapter = {
  modelId: 'midas-small',
  metadata: { name: 'MiDaS-Small — Depth', description: 'Monocular relative inverse depth (256×256)', modelPath: 'https://huggingface.co/litert-community/MiDaS-small/resolve/main/midas_small_256_fp16.tflite', tags: ['vision', 'depth'] },
  inputSpecs: [inpSpec('input', [1, 256, 256, 3], 'float32', 'RGB ImageNet-normalized NHWC')],
  outputSpecs: [outSpec('output', [1, 256, 256], 'float32', 'Relative inverse depth')],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const imageData = values['image'] as ImageData
    if (!imageData) throw new Error('Image data not provided for midas-small')
    const resized = resizeImageData(imageData, 256, 256)
    return { input: normalizeAndFormatImageData(resized, [1, 256, 256, 3], { dataFormat: 'NHWC', normalization: 'imagenet' }) }
  },
  async parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
    return { output: await tensorToImageData(firstOutput(outputs, 'output'), 256, 256) }
  },
}

export const mogeAdapter: ModelAdapter = {
  modelId: 'moge-2',
  metadata: { name: 'MoGe-2 — Geometry', description: 'Monocular point map, normals and valid mask (448×448)', modelPath: 'https://huggingface.co/litert-community/MoGe-2-LiteRT/resolve/main/moge_fp16.tflite', tags: ['vision', 'depth', 'geometry'] },
  inputSpecs: [inpSpec('input', [1, 3, 448, 448], 'float32', 'RGB [0,1] NCHW')],
  outputSpecs: [
    outSpec('points', [1, 448, 448, 3], 'float32', 'Affine point map (exp remap)'),
    outSpec('normal', [1, 448, 448, 3], 'float32', 'L2-normalized normals'),
    outSpec('mask', [1, 448, 448, 1], 'float32', 'Valid mask (sigmoid > 0.5)'),
    outSpec('scale', [1, 1, 1, 1], 'float32', 'Scale'),
  ],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const imageData = values['image'] as ImageData
    if (!imageData) throw new Error('Image data not provided for moge-2')
    const resized = resizeImageData(imageData, 448, 448)
    return { input: normalizeAndFormatImageData(resized, [1, 3, 448, 448], { dataFormat: 'NCHW', normalization: '0-1' }) }
  },
  async parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
    const vals = Object.values(outputs)
    const normal = vals[1]
    const mask = vals[2]
    const scale = vals[3]
    return {
      normal: normal ? await tensorToImageData(normal, 448, 448) : undefined,
      mask: mask ? await tensorToImageData(mask, 448, 448) : undefined,
      scale: scale ? (await scale.data() as Float32Array)[0] : undefined,
    }
  },
}

export const tipsv2Adapter: ModelAdapter = {
  modelId: 'tipsv2-b14-dpt',
  metadata: { name: 'TIPSv2 — Depth, Normals & Seg', description: 'Depth (metres), surface normals and 150-class segmentation (448×448)', modelPath: 'https://huggingface.co/litert-community/TIPSv2-B14-DPT-LiteRT/resolve/main/tipsv2_b14_dpt_fp16.tflite', tags: ['vision', 'depth'] },
  inputSpecs: [inpSpec('input', [1, 3, 448, 448], 'float32', 'RGB [0,1] NCHW (no ImageNet)')],
  outputSpecs: [
    outSpec('depth', [1, 1, 448, 448], 'float32', 'Depth in metres'),
    outSpec('normals', [1, 3, 448, 448], 'float32', 'Unit surface normals'),
    outSpec('seg', [1, 150, 256, 256], 'float32', '150-class segmentation logits'),
  ],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const imageData = values['image'] as ImageData
    if (!imageData) throw new Error('Image data not provided for tipsv2-b14-dpt')
    const resized = resizeImageData(imageData, 448, 448)
    return { input: normalizeAndFormatImageData(resized, [1, 3, 448, 448], { dataFormat: 'NCHW', normalization: '0-1' }) }
  },
  async parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
    const [d, n, s] = Object.values(outputs)
    return {
      depth: d ? await tensorToImageData(d, 448, 448) : undefined,
      normals: n ? await tensorToImageData(n, 448, 448) : undefined,
      seg: s ? argmaxColorize(await s.data() as Float32Array, 150, 256, 256) : undefined,
    }
  },
}

export const nafnetGoproAdapter: ModelAdapter = {
  modelId: 'nafnet-gopro',
  metadata: { name: 'NAFNet — Image Deblurring (GoPro)', description: 'Motion deblurring (256×256)', modelPath: 'https://huggingface.co/litert-community/NAFNet-GoPro-width32-LiteRT/resolve/main/nafnet_fp16.tflite', tags: ['vision', 'restoration'] },
  inputSpecs: [inpSpec('input', [1, 3, 256, 256], 'float32', 'RGB [0,1] NCHW')],
  outputSpecs: [outSpec('output', [1, 3, 256, 256], 'float32', 'Deblurred RGB [0,1] NCHW')],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const imageData = values['image'] as ImageData
    if (!imageData) throw new Error('Image data not provided for nafnet-gopro')
    const resized = resizeImageData(imageData, 256, 256)
    return { input: normalizeAndFormatImageData(resized, [1, 3, 256, 256], { dataFormat: 'NCHW', normalization: '0-1' }) }
  },
  async parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
    const data = await firstOutput(outputs, 'output').data() as Float32Array
    return { output: nchwToImageData(data, 256, 256) }
  },
}

export const gfpganAdapter: ModelAdapter = {
  modelId: 'gfpgan-v1.4',
  metadata: { name: 'GFPGAN — Face Restoration', description: 'Blind face restoration (512×512, ~431 MB)', modelPath: 'https://huggingface.co/litert-community/GFPGAN-v1.4-LiteRT/resolve/main/gfpgan_fp16.tflite', tags: ['vision', 'face', 'restoration'] },
  inputSpecs: [inpSpec('input', [1, 3, 512, 512], 'float32', 'RGB [-1,1] NCHW')],
  outputSpecs: [outSpec('output', [1, 3, 512, 512], 'float32', 'Restored RGB [-1,1] NCHW')],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const imageData = values['image'] as ImageData
    if (!imageData) throw new Error('Image data not provided for gfpgan-v1.4')
    const resized = resizeImageData(imageData, 512, 512)
    return { input: normalizeAndFormatImageData(resized, [1, 3, 512, 512], { dataFormat: 'NCHW', normalization: '-1-1' }) }
  },
  async parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
    const data = await firstOutput(outputs, 'output').data() as Float32Array
    return { output: nchwToImageDataMinusOneToOne(data, 512, 512) }
  },
}

export const siglip2Adapter: ModelAdapter = {
  modelId: 'siglip2-base',
  metadata: { name: 'SigLIP2 — Image Embedding', description: 'L2-normalized 768-d image embedding', modelPath: 'https://huggingface.co/litert-community/SigLIP2-base-patch16-224/resolve/main/siglip2_base_224_fp16.tflite', tags: ['vision', 'embedding'] },
  inputSpecs: [inpSpec('input', [1, 3, 224, 224], 'float32', 'RGB [-1,1] NCHW')],
  outputSpecs: [outSpec('output', [1, 768], 'float32', 'L2-normalized embedding')],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const imageData = values['image'] as ImageData
    if (!imageData) throw new Error('Image data not provided for siglip2-base')
    const resized = resizeImageData(imageData, 224, 224)
    return { input: normalizeAndFormatImageData(resized, [1, 3, 224, 224], { dataFormat: 'NCHW', normalization: '-1-1' }) }
  },
  async parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
    const data = await firstOutput(outputs, 'output').data() as Float32Array
    return { output: Array.from(data) }
  },
}

function buildNchwRgbOnesTensor(imageData: ImageData, w: number, h: number): Tensor {
  const { data } = imageData
  const plane = w * h
  const out = new Float32Array(4 * plane)
  for (let i = 0; i < plane; i++) {
    const o = i * 4
    out[i] = data[o] / 127.5 - 1
    out[plane + i] = data[o + 1] / 127.5 - 1
    out[2 * plane + i] = data[o + 2] / 127.5 - 1
    out[3 * plane + i] = 1
  }
  return new Tensor(out, [1, 4, h, w])
}

export const cpgaAdapter: ModelAdapter = {
  modelId: 'cpga-net-lowlight',
  metadata: { name: 'CPGA-Net — Low-Light Enhancement', description: 'Low-light image enhancement, RGB [0,1]', modelPath: 'https://huggingface.co/litert-community/CPGA-Net-LowLight-LiteRT/resolve/main/cpga_fp16.tflite', tags: ['vision', 'restoration'] },
  inputSpecs: [inpSpec('image', [1, 3, 256, 256], 'float32', 'RGB [0,1] NCHW')],
  outputSpecs: [outSpec('output', [1, 3, 256, 256], 'float32', 'Enhanced RGB [0,1]')],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const imageData = values['image'] as ImageData
    if (!imageData) throw new Error('Image data not provided for cpga-net-lowlight')
    const resized = resizeImageData(imageData, 256, 256)
    return { image: normalizeAndFormatImageData(resized, [1, 3, 256, 256], { dataFormat: 'NCHW', normalization: '0-1' }) }
  },
  async parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
    const data = await firstOutput(outputs, 'output').data() as Float32Array
    return { output: await nchwToImageData(data, 256, 256) }
  },
}

export const peCoreAdapter: ModelAdapter = {
  modelId: 'pe-core-base',
  metadata: { name: 'PE-Core — Image Embedding', description: 'L2-normalized 1024-d image embedding', modelPath: 'https://huggingface.co/litert-community/PE-Core-base-patch16-224/resolve/main/pe_core_base_224_fp16.tflite', tags: ['vision', 'embedding'] },
  inputSpecs: [inpSpec('image', [1, 3, 224, 224], 'float32', 'RGB [-1,1] NCHW')],
  outputSpecs: [outSpec('output', [1, 1024], 'float32', 'L2-normalized embedding')],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const imageData = values['image'] as ImageData
    if (!imageData) throw new Error('Image data not provided for pe-core-base')
    const resized = resizeImageData(imageData, 224, 224)
    return { image: normalizeAndFormatImageData(resized, [1, 3, 224, 224], { dataFormat: 'NCHW', normalization: '-1-1' }) }
  },
  async parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
    const data = await firstOutput(outputs, 'output').data() as Float32Array
    return { output: Array.from(data) }
  },
}

export const clothSegAdapter: ModelAdapter = {
  modelId: 'cloth-segmentation',
  metadata: { name: 'Cloth Segmentation (U²-Net)', description: '4-class clothing parsing', modelPath: 'https://huggingface.co/litert-community/Cloth-Segmentation-U2Net-LiteRT/resolve/main/clothseg.tflite', tags: ['vision', 'segmentation'] },
  inputSpecs: [inpSpec('image', [1, 3, 768, 768], 'float32', 'RGB [-1,1] NCHW')],
  outputSpecs: [outSpec('output', [1, 4, 768, 768], 'float32', '4-class logits')],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const imageData = values['image'] as ImageData
    if (!imageData) throw new Error('Image data not provided for cloth-segmentation')
    const resized = resizeImageData(imageData, 768, 768)
    return { image: normalizeAndFormatImageData(resized, [1, 3, 768, 768], { dataFormat: 'NCHW', normalization: '-1-1' }) }
  },
  async parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
    const data = await firstOutput(outputs, 'output').data() as Float32Array
    return { output: argmaxColorize(data, 4, 768, 768) }
  },
}

export const mlsdAdapter: ModelAdapter = {
  modelId: 'm-lsd-tiny',
  metadata: { name: 'M-LSD-tiny — Line Detection', description: 'Line-segment center heatmap', modelPath: 'https://huggingface.co/litert-community/M-LSD-tiny-LiteRT/resolve/main/mlsd_fp16.tflite', tags: ['vision', 'line-detection'] },
  inputSpecs: [inpSpec('image', [1, 4, 512, 512], 'float32', 'RGB + ones channel, x/127.5-1, NCHW')],
  outputSpecs: [outSpec('tpMap', [1, 9, 256, 256], 'float32', 'ch0 center, ch1-4 displacement')],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const imageData = values['image'] as ImageData
    if (!imageData) throw new Error('Image data not provided for m-lsd-tiny')
    const resized = resizeImageData(imageData, 512, 512)
    return { image: buildNchwRgbOnesTensor(resized, 512, 512) }
  },
  async parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
    const data = await firstOutput(outputs, 'tpMap').data() as Float32Array
    const plane = 256 * 256
    const center = new Float32Array(plane)
    for (let i = 0; i < plane; i++) center[i] = 1 / (1 + Math.exp(-data[i]))
    return { heatmap: maskToImageData(center, 256, 256) }
  },
}

export const visionAdapters: ModelAdapter[] = [
  headpose6drepnetAdapter,
  yunetAdapter,
  yoloxAdapter,
  yoloxNanoAdapter,
  yoloxTinyAdapter,
  yoloxSAdapter,
  u2netAdapter,
  edsrAdapter,
  miganAdapter,
  sinetAdapter,
  disAdapter,
  unisalAdapter,
  bisenetFaceAdapter,
  pidnetAdapter,
  twinliteAdapter,
  midasAdapter,
  mogeAdapter,
  tipsv2Adapter,
  nafnetGoproAdapter,
  gfpganAdapter,
  siglip2Adapter,
  cpgaAdapter,
  peCoreAdapter,
  clothSegAdapter,
  mlsdAdapter,
  ...styleAdapters,
]
