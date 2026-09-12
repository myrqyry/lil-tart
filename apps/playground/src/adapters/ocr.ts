import type { InferenceContext, ModelAdapter } from './types'
import { normalizeAndFormatImageData, resizeImageData } from '../imageUtils'

const OCR_BASE = 'https://huggingface.co/litert-community/PP-OCRv5-LiteRT/resolve/main'
const DET_PATH = `${OCR_BASE}/ppocr_det_fp16.tflite`
const REC_PATH = `${OCR_BASE}/ppocr_rec_fp32.tflite`
const DICT_PATH = `${OCR_BASE}/ppocrv5_dict.txt`

const DET_SIZE = 640
const REC_HEIGHT = 48
const REC_WIDTH = 320
const BOX_THRESHOLD = 0.3
const BOX_MEAN_THRESHOLD = 0.5
const MIN_BOX_SIZE = 6

let dictPromise: Promise<string[]> | null = null
function loadDict(): Promise<string[]> {
  if (!dictPromise) {
    dictPromise = fetch(DICT_PATH)
      .then((response) => response.text())
      .then((text) => {
        const lines = text.split('\n').map((line) => line.replace(/\r$/, ''))
        if (lines.length > 0 && lines[lines.length - 1] === '') lines.pop()
        // ponytail: CTC class 0 is blank, classes 1..n are the dict, the last class is space.
        return ['', ...lines, ' ']
      })
      .catch((error) => {
        dictPromise = null
        throw error
      })
  }
  return dictPromise
}

interface Box {
  x: number
  y: number
  w: number
  h: number
}

function findBoxes(prob: Float32Array): Box[] {
  const size = DET_SIZE
  const visited = new Uint8Array(size * size)
  const stack: number[] = []
  const boxes: Box[] = []

  for (let start = 0; start < size * size; start++) {
    if (visited[start] || prob[start] <= BOX_THRESHOLD) continue
    stack.length = 0
    stack.push(start)
    visited[start] = 1
    let minX = size
    let minY = size
    let maxX = 0
    let maxY = 0
    let count = 0
    let sum = 0
    while (stack.length > 0) {
      const p = stack.pop() as number
      const x = p % size
      const y = (p / size) | 0
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y
      count++
      sum += prob[p]
      if (x > 0 && !visited[p - 1] && prob[p - 1] > BOX_THRESHOLD) {
        visited[p - 1] = 1
        stack.push(p - 1)
      }
      if (x < size - 1 && !visited[p + 1] && prob[p + 1] > BOX_THRESHOLD) {
        visited[p + 1] = 1
        stack.push(p + 1)
      }
      if (y > 0 && !visited[p - size] && prob[p - size] > BOX_THRESHOLD) {
        visited[p - size] = 1
        stack.push(p - size)
      }
      if (y < size - 1 && !visited[p + size] && prob[p + size] > BOX_THRESHOLD) {
        visited[p + size] = 1
        stack.push(p + size)
      }
    }
    const w = maxX - minX + 1
    const h = maxY - minY + 1
    if (w < MIN_BOX_SIZE || h < MIN_BOX_SIZE) continue
    if (sum / count < BOX_MEAN_THRESHOLD) continue
    // DB postprocess approximation: expand the box by a size-scaled pad.
    const pad = Math.min(Math.max(0.35 * Math.min(w, h), 2), 24)
    boxes.push({ x: minX - pad, y: minY - pad, w: w + 2 * pad, h: h + 2 * pad })
  }
  return boxes
}

function cropImageData(source: ImageData, box: Box): ImageData {
  const x0 = Math.max(0, box.x)
  const y0 = Math.max(0, box.y)
  const x1 = Math.min(source.width, box.x + box.w)
  const y1 = Math.min(source.height, box.y + box.h)
  const width = Math.max(1, x1 - x0)
  const height = Math.max(1, y1 - y0)
  const out = new ImageData(width, height)
  for (let row = 0; row < height; row++) {
    const from = ((y0 + row) * source.width + x0) * 4
    out.data.set(source.data.subarray(from, from + width * 4), row * width * 4)
  }
  return out
}

function buildLineImage(source: ImageData): ImageData {
  const line = new ImageData(REC_WIDTH, REC_HEIGHT)
  for (let i = 3; i < line.data.length; i += 4) line.data[i] = 255
  const width = Math.min(REC_WIDTH, source.width)
  for (let row = 0; row < REC_HEIGHT && row < source.height; row++) {
    line.data.set(source.data.subarray(row * source.width * 4, row * source.width * 4 + width * 4), row * REC_WIDTH * 4)
  }
  return line
}

function ctcDecode(logits: Float32Array, classes: number, chars: string[]): string {
  const steps = Math.floor(logits.length / classes)
  let text = ''
  let previous = -1
  for (let t = 0; t < steps; t++) {
    const offset = t * classes
    let best = 0
    let bestValue = -Infinity
    for (let c = 0; c < classes; c++) {
      const value = logits[offset + c]
      if (value > bestValue) {
        bestValue = value
        best = c
      }
    }
    if (best !== 0 && best !== previous) text += chars[best] ?? ''
    previous = best
  }
  return text
}

export const ppOcrAdapter: ModelAdapter = {
  modelId: 'pp-ocrv5',
  metadata: {
    name: 'PP-OCRv5',
    description: 'Text detection and recognition (Chinese + English)',
    modelPath: DET_PATH,
    tags: ['vision', 'ocr', 'text'],
  },
  graphs: [{ name: 'rec', modelPath: REC_PATH }],
  inputSpecs: [
    { name: 'image', dtype: 'float32', shape: [1, 3, DET_SIZE, DET_SIZE], description: 'Image containing text' },
  ],
  outputSpecs: [{ name: 'text', dtype: 'string', shape: [], description: 'Recognized text lines' }],
  prepareInputs() {
    throw new Error('Image data not provided for pp-ocrv5')
  },
  async parseOutputs() {
    return {}
  },
  async run(values, ctx: InferenceContext) {
    const image = values.image
    if (!(image instanceof ImageData)) throw new Error('Image data not provided for pp-ocrv5')
    const chars = await loadDict()

    const resized = resizeImageData(image, DET_SIZE, DET_SIZE)
    const detInput = normalizeAndFormatImageData(resized, [1, 3, DET_SIZE, DET_SIZE], {
      colorOrder: 'RGB',
      dataFormat: 'NCHW',
      normalization: 'imagenet',
    })
    const detOutputs = await ctx.predict('main', { input: detInput })
    const prob = (await Object.values(detOutputs)[0].data()) as Float32Array

    const boxes = findBoxes(prob).sort((a, b) => a.y - b.y || a.x - b.x)
    if (boxes.length === 0) return { text: 'No text detected' }

    const lines: string[] = []
    for (const box of boxes) {
      const crop = cropImageData(resized, box)
      const width = Math.min(REC_WIDTH, Math.max(1, Math.round((REC_HEIGHT * crop.width) / crop.height)))
      const line = buildLineImage(resizeImageData(crop, width, REC_HEIGHT))
      const recInput = normalizeAndFormatImageData(line, [1, 3, REC_HEIGHT, REC_WIDTH], {
        colorOrder: 'RGB',
        dataFormat: 'NCHW',
        normalization: 'imagenet',
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.5, 0.5],
      })
      const recOutputs = await ctx.predict('rec', { input: recInput })
      const logits = (await Object.values(recOutputs)[0].data()) as Float32Array
      const text = ctcDecode(logits, chars.length, chars).trim()
      if (text) lines.push(text)
    }
    return { text: lines.length > 0 ? lines.join('\n') : 'No text recognized' }
  },
}

export const ocrAdapters: ModelAdapter[] = [ppOcrAdapter]
