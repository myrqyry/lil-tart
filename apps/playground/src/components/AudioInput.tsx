import { useState } from 'react'
import type { TensorSpec } from '../adapters/types'
import { resampleToMono16k } from '../audioUtils'

interface AudioInputProps {
  specs: TensorSpec[]
  onChange: (values: Record<string, any>) => void
}

export default function AudioInput({ specs, onChange }: AudioInputProps) {
  const [status, setStatus] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const spec = specs.find(s => s.dtype === 'float32') ?? specs[0]
  if (!spec) return null

  const handleFile = async (file: File) => {
    setError(null)
    setStatus('Decoding…')
    try {
      const context = new AudioContext()
      const buffer = await context.decodeAudioData(await file.arrayBuffer())
      void context.close()
      const channels = Array.from({ length: buffer.numberOfChannels }, (_, c) => buffer.getChannelData(c))
      const audio = resampleToMono16k(channels, buffer.sampleRate)
      onChange({ [spec.name]: audio })
      setStatus(`${(audio.length / 16000).toFixed(2)} s · 16 kHz mono`)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
      setStatus(null)
    }
  }

  return (
    <div>
      <h2 className="mb-3 text-sm font-semibold text-on-surface-variant uppercase tracking-wide">Audio Input</h2>
      <div className="mb-4">
        <label className="mb-1 block text-sm font-semibold text-on-surface">
          {spec.name}
          <span className="ml-2 text-xs font-normal text-on-surface-variant">
            {spec.dtype} {JSON.stringify(spec.shape)}
          </span>
        </label>
        <p className="mb-1 text-xs text-on-surface-variant">{spec.description}</p>
        <input
          type="file"
          accept="audio/*"
          onChange={e => { const file = e.target.files?.[0]; if (file) void handleFile(file) }}
          className="w-full rounded-lg border border-outline bg-surface-container px-3 py-2 text-sm text-on-surface file:mr-3 file:rounded-full file:border-0 file:bg-primary file:px-3 file:py-1 file:text-xs file:font-medium file:text-on-primary"
        />
        {status && <p className="mt-1 text-xs text-on-surface-variant">{status}</p>}
        {error && <p className="mt-1 text-xs text-error">{error}</p>}
      </div>
    </div>
  )
}
