import { useEffect, useRef, useState } from 'react'
import { Qwen3TtsPipeline, qwen3TtsManifest, type QwenTtsConfig } from '@litert-playground/qwen3-tts'
import { createLiteRtRuntime, type ManagedLiteRtRuntimeContext } from '@litert-playground/runtime-litert'
import { createModelLibraryAssetResolver, registerModelAssets } from '../modelStorage'
import type { PipelineProgress } from '@litert-playground/inference-core'

let pipeline: Qwen3TtsPipeline | null = null

export function Qwen3TtsPanel() {
  const [text, setText] = useState('Hello, welcome to my world.')
  const [generating, setGenerating] = useState(false)
  const [progress, setProgress] = useState('')
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [status, setStatus] = useState('Not loaded')
  const [error, setError] = useState<string | null>(null)
  const [cfg, setCfg] = useState<QwenTtsConfig>({
    temperature: 0.85,
    topK: 25,
    repetitionPenalty: 1.05,
    voice: 'demo_speaker',
    maxFrames: 512,
    language: 'english'
  })

  const updateCfg = (updater: (prev: QwenTtsConfig) => QwenTtsConfig) => setCfg(updater)
  const ctxRef = useRef<ManagedLiteRtRuntimeContext | null>(null)

  const disposePipeline = async () => {
    const current = pipeline
    const ctx = ctxRef.current
    pipeline = null
    ctxRef.current = null
    if (current) {
      try { await current.dispose() } catch { /* best-effort cleanup */ }
    }
    ctx?.liteRt.dispose()
  }

  useEffect(() => {
    return () => {
      void disposePipeline()
    }
  }, [])

  const handleLoad = async () => {
    if (status === 'Loading...' || status === 'Ready') return
    setStatus('Loading...')
    setError(null)
    setProgress('Preparing runtime…')

    try {
      await disposePipeline()
      registerModelAssets(qwen3TtsManifest.modelId, qwen3TtsManifest.assets.map((asset) => asset.path))

      const p = new Qwen3TtsPipeline()
      p.onProgress = (pr: PipelineProgress) => {
        setProgress(`${pr.phase} ${pr.step}/${pr.total}`)
      }

      const ctx = await createLiteRtRuntime({
        assetBase: '/models/qwen3-tts',
        assets: createModelLibraryAssetResolver('/models/qwen3-tts/'),
      })
      ctxRef.current = ctx
      await p.load(ctx)
      pipeline = p
      setStatus('Ready')
      setProgress('')
    } catch (e: unknown) {
      await disposePipeline()
      setStatus('Load failed')
      setError(String(e))
    }
  }

  const handleUnload = async () => {
    await disposePipeline()
    setStatus('Not loaded')
    setProgress('')
  }

  const handleGenerate = async () => {
    if (!pipeline || generating) return
    setGenerating(true)
    setAudioUrl(null)
    try {
      setProgress('Generating...')
      const result = await pipeline.run({ text }, cfg)
      const wav = encodeWav(result.samples, result.sampleRate)
      const url = URL.createObjectURL(new Blob([wav], { type: 'audio/wav' }))
      setAudioUrl(url)
      setProgress(`Done (${result.durationSeconds.toFixed(1)}s)`)
    } catch (e: unknown) {
      setProgress(`Error: ${e}`)
    }
    setGenerating(false)
  }

  const handleRetry = () => {
    setStatus('Not loaded')
    setError(null)
    setProgress('')
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="text-sm text-on-surface-variant">Status: {status}</div>
        <div className="flex gap-2">
          <select
            className="rounded-lg border border-outline bg-surface-container px-2 py-1 text-xs text-on-surface"
            value={cfg.language}
            onChange={e => updateCfg(c => ({ ...c, language: e.target.value }))}
          >
            <option value="english">English</option>
            <option value="chinese">Chinese</option>
            <option value="japanese">Japanese</option>
            <option value="korean">Korean</option>
            <option value="german">German</option>
            <option value="french">French</option>
            <option value="spanish">Spanish</option>
            <option value="italian">Italian</option>
            <option value="portuguese">Portuguese</option>
            <option value="russian">Russian</option>
          </select>
          <select
            className="rounded-lg border border-outline bg-surface-container px-2 py-1 text-xs text-on-surface"
            value={cfg.voice}
            onChange={e => updateCfg(c => ({ ...c, voice: e.target.value }))}
          >
            <option value="demo_speaker">Demo Speaker</option>
          </select>
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        {status === 'Ready' ? (
          <button
            type="button"
            onClick={() => void handleUnload()}
            className="rounded-full border border-outline px-5 py-2 text-xs font-medium text-on-surface hover:bg-surface-container-high"
          >
            Unload pipeline
          </button>
        ) : (
          <button
            type="button"
            onClick={() => void handleLoad()}
            disabled={status === 'Loading...'}
            className="rounded-full bg-primary px-5 py-2 text-xs font-semibold text-on-primary disabled:opacity-50"
          >
            {status === 'Loading...' ? 'Preparing…' : 'Prepare pipeline'}
          </button>
        )}
        <p className="self-center text-[11px] text-on-surface-variant">
          Model assets are only fetched when inference actually needs them.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 rounded-xl bg-surface-container-low p-3 text-xs">
        <div>
          <label className="mb-1 block text-on-surface-variant">Temperature ({cfg.temperature})</label>
          <input
            type="range" min="0.1" max="2.0" step="0.05"
            value={cfg.temperature}
            onChange={e => updateCfg(c => ({ ...c, temperature: parseFloat(e.target.value) }))}
            className="w-full accent-primary"
          />
        </div>
        <div>
          <label className="mb-1 block text-on-surface-variant">Top-K ({cfg.topK})</label>
          <input
            type="number" min="1" max="100"
            value={cfg.topK}
            onChange={e => updateCfg(c => ({ ...c, topK: parseInt(e.target.value) || 25 }))}
            className="w-full rounded border border-outline bg-surface px-2 py-1 text-on-surface"
          />
        </div>
        <div>
          <label className="mb-1 block text-on-surface-variant">Repetition Penalty ({cfg.repetitionPenalty})</label>
          <input
            type="range" min="1.0" max="2.0" step="0.01"
            value={cfg.repetitionPenalty}
            onChange={e => updateCfg(c => ({ ...c, repetitionPenalty: parseFloat(e.target.value) }))}
            className="w-full accent-primary"
          />
        </div>
        <div>
          <label className="mb-1 block text-on-surface-variant">Max Frames ({cfg.maxFrames})</label>
          <input
            type="number" min="64" max="2048" step="64"
            value={cfg.maxFrames}
            onChange={e => updateCfg(c => ({ ...c, maxFrames: parseInt(e.target.value) || 512 }))}
            className="w-full rounded border border-outline bg-surface px-2 py-1 text-on-surface"
          />
        </div>
      </div>

      {error && (
        <div className="rounded-lg bg-error-container p-3 text-sm text-on-error-container">
          {error}
          <button onClick={handleRetry} className="ml-3 underline">Retry</button>
        </div>
      )}
      <textarea
        className="h-24 w-full rounded-lg border border-outline bg-surface-container p-3 font-mono text-sm text-on-surface transition-colors focus:border-primary focus:ring-2 focus:ring-primary/30 focus:outline-none"
        value={text}
        onChange={e => setText(e.target.value)}
        placeholder="Enter text to synthesize..."
      />
      <button
        className="inline-flex items-center justify-center rounded-full bg-primary px-8 py-3 text-sm font-medium text-on-primary shadow-md transition-all duration-300 hover:scale-[1.02] hover:shadow-lg active:scale-[0.97] disabled:opacity-50 disabled:shadow-none"
        style={{ transitionTimingFunction: 'var(--ease-spring)' }}
        onClick={handleGenerate}
        disabled={generating || status !== 'Ready'}
      >
        {generating ? 'Generating...' : 'Generate Speech'}
      </button>
      {progress && <div className="text-xs text-on-surface-variant">{progress}</div>}
      {audioUrl && (
        <audio controls src={audioUrl} className="w-full" />
      )}
    </div>
  )
}

function encodeWav(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const numChannels = 1
  const bitsPerSample = 16
  const byteRate = sampleRate * numChannels * bitsPerSample / 8
  const blockAlign = numChannels * bitsPerSample / 8
  const dataSize = samples.length * blockAlign
  const buf = new ArrayBuffer(44 + dataSize)
  const v = new DataView(buf)

  const w = (off: number, str: string) => {
    for (let i = 0; i < str.length; i++) v.setUint8(off + i, str.charCodeAt(i))
  }
  w(0, 'RIFF'); v.setUint32(4, 36 + dataSize, true); w(8, 'WAVE')
  w(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true)
  v.setUint16(22, numChannels, true); v.setUint32(24, sampleRate, true)
  v.setUint32(28, byteRate, true); v.setUint16(32, blockAlign, true)
  v.setUint16(34, bitsPerSample, true)
  w(36, 'data'); v.setUint32(40, dataSize, true)

  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }
  return buf
}
