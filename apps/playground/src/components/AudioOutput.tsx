import { useEffect, useState } from 'react'
import { encodeWav } from '../audioUtils'

interface AudioOutputProps {
  samples: Float32Array
  sampleRate: number
  label?: string
}

export default function AudioOutput({ samples, sampleRate, label }: AudioOutputProps) {
  const [url, setUrl] = useState<string | null>(null)

  useEffect(() => {
    const objectUrl = URL.createObjectURL(new Blob([encodeWav(samples, sampleRate)], { type: 'audio/wav' }))
    setUrl(objectUrl)
    return () => URL.revokeObjectURL(objectUrl)
  }, [samples, sampleRate])

  return (
    <div className="space-y-2">
      {label && <div className="text-xs font-medium text-on-surface-variant">{label}</div>}
      {url && <audio controls src={url} className="w-full" />}
      <div className="text-xs text-on-surface-variant">
        {(samples.length / sampleRate).toFixed(2)}s · {sampleRate} Hz
      </div>
    </div>
  )
}
