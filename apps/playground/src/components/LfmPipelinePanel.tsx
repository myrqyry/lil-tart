import { useState, useEffect, useRef } from 'react'
import type { ModelManifest } from '@litert-playground/inference-core'
import { createHttpAssetResolver } from '@litert-playground/inference-core'
import { createLiteRtRuntime, type ManagedLiteRtRuntimeContext } from '@litert-playground/runtime-litert'
import {
  LiteRtLmTextPipeline,
  lfm2_5InstructManifest,
  lfm2_5ThinkingManifest,
  gemma4E2bManifest,
  gemma4E4bManifest,
  type TextGenerationInput,
} from '@litert-playground/text-gen'
import { ColBertPipeline, colbertManifest, rankColBert, type ColBertConfig } from '@litert-playground/retrieval'
import {
  EncoderPipeline,
  encoder230mManifest,
  encoderSpellcheckerManifest,
  encoderPolicyLinterManifest,
  type EncoderConfig,
  type EncoderResult,
} from '@litert-playground/encoder'

interface LfmModelEntry {
  id: string
  label: string
  manifest: ModelManifest
  kind: 'text' | 'colbert' | 'encoder'
}

const MODELS: LfmModelEntry[] = [
  { id: 'gemma-4-e2b-it', label: 'Gemma 4 E2B Instruct', manifest: gemma4E2bManifest, kind: 'text' },
  { id: 'gemma-4-e4b-it', label: 'Gemma 4 E4B Instruct', manifest: gemma4E4bManifest, kind: 'text' },
  { id: 'lfm2.5-1.2b-instruct', label: 'LFM2.5 1.2B Instruct', manifest: lfm2_5InstructManifest, kind: 'text' },
  { id: 'lfm2.5-1.2b-thinking', label: 'LFM2.5 1.2B Thinking', manifest: lfm2_5ThinkingManifest, kind: 'text' },
  { id: 'lfm2.5-colbert-350m', label: 'LFM2.5 ColBERT-350M', manifest: colbertManifest, kind: 'colbert' },
  { id: 'lfm2.5-encoder-230m', label: 'LFM2.5 Encoder-230M', manifest: encoder230mManifest, kind: 'encoder' },
  { id: 'lfm2.5-encoder-350m-spellchecker', label: 'LFM2.5 Spellchecker-350M', manifest: encoderSpellcheckerManifest, kind: 'encoder' },
  { id: 'lfm2.5-encoder-350m-policy-linter', label: 'LFM2.5 Policy-Linter-350M', manifest: encoderPolicyLinterManifest, kind: 'encoder' },
]

const CANDIDATES = [
  { id: 'doc-a', text: 'The knight must capture the rook to break the defensive line.' },
  { id: 'doc-b', text: 'Preserving material leads to a winning endgame in most lines.' },
  { id: 'doc-c', text: 'A quick castle develops the king and activates the rook.' },
]

type AnyPipeline = LiteRtLmTextPipeline | ColBertPipeline | EncoderPipeline

export function LfmPipelinePanel() {
  const [modelId, setModelId] = useState(MODELS[0].id)
  const [text, setText] = useState('Return the captured pawn to its original square.')
  const [maxTokens, setMaxTokens] = useState(128)
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState('')
  const [output, setOutput] = useState('')
  const [status, setStatus] = useState('Not loaded')
  const [error, setError] = useState<string | null>(null)

  const pipelineRef = useRef<AnyPipeline | null>(null)
  const ctxRef = useRef<ManagedLiteRtRuntimeContext | null>(null)
  const loadGenerationRef = useRef(0)

  const entry = MODELS.find(m => m.id === modelId) ?? MODELS[0]

  const disposePipeline = async () => {
    const pipeline = pipelineRef.current
    const ctx = ctxRef.current
    pipelineRef.current = null
    ctxRef.current = null

    if (pipeline) {
      try { await pipeline.dispose() } catch { /* cleanup should be best-effort */ }
    }
    ctx?.liteRt.dispose()
  }

  const load = async (id: string, generation: number) => {
    const m = MODELS.find(x => x.id === id) ?? MODELS[0]
    setStatus('Loading...')
    setError(null)

    let ctx: ManagedLiteRtRuntimeContext | null = null
    let nextPipeline: AnyPipeline | null = null

    try {
      await disposePipeline()
      if (generation !== loadGenerationRef.current) return

      ctx = await createLiteRtRuntime({
        assetBase: '/',
        assets: createHttpAssetResolver('/'),
        supportedBackends: { webgpu: true, wasm: true },
      })
      if (generation !== loadGenerationRef.current) {
        ctx.liteRt.dispose()
        return
      }

      ctxRef.current = ctx
      nextPipeline =
        m.kind === 'text' ? new LiteRtLmTextPipeline(m.manifest)
        : m.kind === 'colbert' ? new ColBertPipeline({ manifest: m.manifest })
        : new EncoderPipeline({ manifest: m.manifest })

      await nextPipeline.load(ctx)
      if (generation !== loadGenerationRef.current) {
        try { await nextPipeline.dispose() } catch { /* cleanup should be best-effort */ }
        ctx.liteRt.dispose()
        if (ctxRef.current === ctx) ctxRef.current = null
        return
      }

      pipelineRef.current = nextPipeline
      setStatus('Ready')
      setProgress('')
    } catch (e: unknown) {
      if (generation !== loadGenerationRef.current) return
      if (nextPipeline) {
        try { await nextPipeline.dispose() } catch { /* cleanup should be best-effort */ }
      }
      ctx?.liteRt.dispose()
      if (ctxRef.current === ctx) ctxRef.current = null
      setStatus('Load failed')
      setError(String(e))
    }
  }

  useEffect(() => {
    const generation = ++loadGenerationRef.current
    void load(modelId, generation)

    return () => {
      loadGenerationRef.current += 1
      void disposePipeline()
    }
  }, [modelId])

  const handleRun = async () => {
    const p = pipelineRef.current
    if (!p || running || status !== 'Ready') return
    setRunning(true)
    setOutput('')
    try {
      if (entry.kind === 'text') {
        setProgress('Generating...')
        const input: TextGenerationInput = { messages: [{ role: 'user', content: text }] }
        const result = await (p as LiteRtLmTextPipeline).run(input, { model: entry.manifest.assets[0]?.path ?? '', maxTokens })
        setOutput(result.text)
        setProgress(`Done (${result.text.length} chars)`)
      } else if (entry.kind === 'colbert') {
        setProgress('Embedding query + candidates...')
        const cp = p as ColBertPipeline
        const cfg: ColBertConfig = { maxTokens }
        const query = await cp.run({ text }, cfg)
        const docs = await Promise.all(
          CANDIDATES.map(async d => ({ id: d.id, embedding: await cp.run({ text: d.text }, cfg) })),
        )
        const ranked = rankColBert(query, docs)
        setOutput(ranked.map((r, i) => `${i + 1}. ${r.id}  ${r.score.toFixed(4)}`).join('\n'))
        setProgress(`Done (${query.tokens} tokens x ${query.dimensions} dims)`)
      } else {
        setProgress('Encoding...')
        const ep = p as EncoderPipeline
        const cfg: EncoderConfig = { maxTokens }
        const result: EncoderResult = await ep.run({ text }, cfg)
        if (result.kind === 'embedding') {
          setOutput(`embedding: ${result.dimensions} dims\n${Array.from(result.values.slice(0, 8)).map(v => v.toFixed(4)).join(', ')}...`)
        } else {
          setOutput(`token scores: ${result.tokens} tokens x ${result.dimensions} dims\n${Array.from(result.scores.slice(0, 12)).map(v => v.toFixed(4)).join(', ')}...`)
        }
        setProgress('Done')
      }
    } catch (e: unknown) {
      setProgress(`Error: ${e}`)
    }
    setRunning(false)
  }

  const handleRetry = () => {
    const generation = ++loadGenerationRef.current
    setStatus('Not loaded')
    setError(null)
    setProgress('')
    void load(modelId, generation)
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-2">
        <div className="text-sm text-on-surface-variant">Status: {status}</div>
        <select
          className="rounded-lg border border-outline bg-surface-container px-2 py-1 text-xs text-on-surface"
          value={modelId}
          onChange={e => setModelId(e.target.value)}
        >
          {MODELS.map(m => <option key={m.id} value={m.id}>{m.label}</option>)}
        </select>
      </div>

      <div className="grid grid-cols-2 gap-3 rounded-xl bg-surface-container-low p-3 text-xs">
        <div>
          <label className="mb-1 block text-on-surface-variant">Max tokens ({maxTokens})</label>
          <input
            type="range" min="16" max="2048" step="16"
            value={maxTokens}
            onChange={e => setMaxTokens(parseInt(e.target.value))}
            className="w-full accent-primary"
          />
        </div>
        <div className="text-on-surface-variant">
          {entry.manifest.memory.downloadBytes > 0
            ? `Model: ${(entry.manifest.memory.downloadBytes / 1e6).toFixed(0)} MB download`
            : 'Model: local asset'}
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
        placeholder="Enter text..."
      />
      <button
        className="inline-flex items-center justify-center rounded-full bg-primary px-8 py-3 text-sm font-medium text-on-primary shadow-md transition-all duration-300 hover:scale-[1.02] hover:shadow-lg active:scale-[0.97] disabled:opacity-50 disabled:shadow-none"
        style={{ transitionTimingFunction: 'var(--ease-spring)' }}
        onClick={handleRun}
        disabled={running || status !== 'Ready'}
      >
        {running ? 'Running...' : 'Run'}
      </button>
      {progress && <div className="text-xs text-on-surface-variant">{progress}</div>}
      {output && (
        <pre className="whitespace-pre-wrap rounded-lg border border-outline bg-surface-container p-3 font-mono text-sm text-on-surface">
          {output}
        </pre>
      )}
    </div>
  )
}
