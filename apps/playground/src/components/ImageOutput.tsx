import { useRef, useEffect } from 'react'

interface ImageOutputProps {
  data: Float32Array | Uint8Array | Uint8ClampedArray | Int32Array
  shape: number[]
  label: string
}

function tensorToImageData(data: Float32Array | Uint8Array | Uint8ClampedArray | Int32Array, shape: number[]): ImageData | null {
  if (shape.length === 4) {
    const [, c, h, w] = shape
    // NCHW format
    const pixels = new Uint8ClampedArray(w * h * 4)
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = y * w + x
        if (c >= 3) {
          pixels[idx * 4]     = clampByte(data[idx])
          pixels[idx * 4 + 1] = clampByte(data[h * w + idx])
          pixels[idx * 4 + 2] = clampByte(data[2 * h * w + idx])
        } else {
          const v = clampByte(data[idx])
          pixels[idx * 4] = v; pixels[idx * 4 + 1] = v; pixels[idx * 4 + 2] = v
        }
        pixels[idx * 4 + 3] = 255
      }
    }
    return new ImageData(pixels, w, h)
  }

  if (shape.length === 3) {
    const [c, h, w] = shape
    const pixels = new Uint8ClampedArray(w * h * 4)
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = y * w + x
        if (c >= 3) {
          pixels[idx * 4]     = clampByte(data[idx])
          pixels[idx * 4 + 1] = clampByte(data[h * w + idx])
          pixels[idx * 4 + 2] = clampByte(data[2 * h * w + idx])
        } else {
          pixels[idx * 4] = clampByte(data[idx])
          pixels[idx * 4 + 1] = pixels[idx * 4]
          pixels[idx * 4 + 2] = pixels[idx * 4]
        }
        pixels[idx * 4 + 3] = 255
      }
    }
    return new ImageData(pixels, w, h)
  }

  if (shape.length === 2) {
    const [h, w] = shape
    const pixels = new Uint8ClampedArray(w * h * 4)
    for (let i = 0; i < h * w; i++) {
      const v = clampByte(data[i])
      pixels[i * 4] = v; pixels[i * 4 + 1] = v; pixels[i * 4 + 2] = v; pixels[i * 4 + 3] = 255
    }
    return new ImageData(pixels, w, h)
  }

  return null
}

function clampByte(v: number): number {
  if (v >= 0 && v <= 1) return Math.round(v * 255)
  if (v >= 0 && v <= 255) return Math.round(v)
  if (v < 0) return Math.round(v + 256)
  return Math.round(Math.min(255, Math.max(0, v)))
}

function isImageShape(shape: number[]): boolean {
  if (shape.length === 4) {
    const [, c, h, w] = shape
    return c <= 4 && h > 1 && w > 1 && h <= 2048 && w <= 2048
  }
  if (shape.length === 3) {
    const [c, h, w] = shape
    return c <= 4 && h > 1 && w > 1 && h <= 2048 && w <= 2048
  }
  if (shape.length === 2) {
    const [h, w] = shape
    return h > 1 && w > 1 && h <= 4096 && w <= 4096
  }
  return false
}

export default function ImageOutput({ data, shape, label }: ImageOutputProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !data || !shape || !ArrayBuffer.isView(data)) return

    const imageData = tensorToImageData(data as any, shape)
    if (!imageData) return

    canvas.width = imageData.width
    canvas.height = imageData.height
    const ctx = canvas.getContext('2d')
    if (ctx) ctx.putImageData(imageData, 0, 0)
  }, [data, shape])

  return (
    <div className="mb-4">
      <div className="text-sm font-medium text-[var(--color-on-surface-variant)] mb-2">{label}</div>
      {data && shape && isImageShape(shape) ? (
        <canvas
          ref={canvasRef}
          className="max-w-full h-auto rounded-xl border border-[var(--color-outline)]"
          style={{ maxHeight: 400 }}
        />
      ) : (
        <div className="text-xs text-[var(--color-on-surface-variant)] italic">
          Not renderable as image (shape: [{shape?.join(', ')}])
        </div>
      )}
    </div>
  )
}

export { isImageShape }
