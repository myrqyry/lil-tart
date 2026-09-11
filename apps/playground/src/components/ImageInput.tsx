import { useRef, useState, useCallback } from 'react'
import type { TensorSpec } from '../adapters/types'

interface ImageInputProps {
  specs: TensorSpec[]
  onChange: (values: Record<string, any>) => void
}

export default function ImageInput({ specs, onChange }: ImageInputProps) {
  const spec = specs.find(s => s.shape.length === 4)
  const [preview, setPreview] = useState<string | null>(null)
  const [imgDims, setImgDims] = useState<string>('')
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const processImage = useCallback((file: File) => {
    if (!spec) return
    const url = URL.createObjectURL(file)
    const img = new Image()
    img.onload = () => {
      processImageData(img, setPreview, setImgDims, onChange, canvasRef)
      URL.revokeObjectURL(url)
    }
    img.src = url
  }, [spec, onChange])

  const handleFile = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]; if (f) processImage(f)
  }, [processImage])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const f = e.dataTransfer.files[0]; if (f?.type.startsWith('image/')) processImage(f)
  }, [processImage])

  const handlePaste = useCallback((e: React.ClipboardEvent) => {
    const f = e.clipboardData.files[0]; if (f?.type.startsWith('image/')) processImage(f)
  }, [processImage])

  return (
    <div className="space-y-3">
      <h2 className="text-sm font-semibold text-on-surface">Image Input</h2>
      {spec && <p className="text-xs text-on-surface-variant">
        Expected: {spec.shape.join('×')} {spec.dtype} — {spec.description}
      </p>}

      <div
        onDrop={handleDrop}
        onPaste={handlePaste}
        onDragOver={e => e.preventDefault()}
        onClick={() => inputRef.current?.click()}
        className="border-2 border-dashed border-outline-variant rounded-xl p-8 text-center cursor-pointer hover:border-primary transition-colors"
      >
        {preview ? (
          <div className="space-y-2">
            <img src={preview} className="max-h-56 mx-auto rounded-lg shadow" alt="Preview" />
            <p className="text-xs text-m3-onSurfaceVariant">{imgDims}</p>
              <button
                onClick={e => { e.stopPropagation(); setPreview(null); onChange({}) }}
                className="text-xs text-error hover:underline"
            >
              Remove
            </button>
          </div>
        ) : (
          <div className="text-on-surface-variant space-y-1">
            <svg className="w-10 h-10 mx-auto opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            <p className="text-sm font-medium">Click to upload / Drop image / Paste</p>
            <p className="text-xs">PNG, JPG, WebP</p>
          </div>
        )}
        <input ref={inputRef} type="file" accept="image/*" onChange={handleFile} className="hidden" />
      </div>
      <canvas ref={canvasRef} className="hidden" />
    </div>
  )
}

// ponytail: emits the raw ImageData under `image`; each adapter owns its resize/normalize.
function processImageData(
  img: HTMLImageElement,
  setPreview: (url: string) => void,
  setImgDims: (d: string) => void,
  onChange: (v: Record<string, any>) => void,
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
) {
  const canvas = canvasRef.current!
  canvas.width = img.width
  canvas.height = img.height
  const ctx = canvas.getContext('2d')!
  ctx.drawImage(img, 0, 0)
  setPreview(canvas.toDataURL('image/webp', 0.7))
  setImgDims(`${img.width}×${img.height}`)
  onChange({ image: ctx.getImageData(0, 0, img.width, img.height) })
}
