import { useCallback, useEffect, useMemo, useState } from 'react'
import { useModelRunner } from '../hooks/useModelRunner'
import type { Accelerator } from '../hooks/useModelRunner'
import type { ModelAdapter, TensorSpec } from '../adapters/types'
import { getTartGuideMessage } from '../tartGuide'
import {
  clearStoredModels,
  listStoredModels,
  removeStoredModel,
  type StoredModelInfo,
} from '../modelStorage'
import ModelList from './ModelList'
import InputEditor from './InputEditor'
import ImageInput from './ImageInput'
import AudioInput from './AudioInput'
import OutputViewer from './OutputViewer'
import TartGuide from './TartGuide'

function isVisionSpec(spec: TensorSpec): boolean {
  const s = spec.shape
  if (s.length !== 4) return false
  if (s[2] > 4 && s[3] > 4) return true
  if (s[1] > 4 && s[2] > 4 && s[3] >= 1 && s[3] <= 4) return true
  return false
}

function isAudioSpec(spec: TensorSpec): boolean {
  const s = spec.shape
  return s.length === 2 && s[0] === 1 && s[1] > 256
}

interface ModelRunnerProps {
  adapters: ModelAdapter[]
  selectedId?: string | null
  onSelect?: (id: string | null) => void
}

const ACCEL_OPTIONS: { value: Accelerator; label: string }[] = [
  { value: 'auto', label: 'Auto (GPU → NPU → CPU)' },
  { value: 'webgpu', label: 'WebGPU' },
  { value: 'webnn', label: 'WebNN' },
  { value: 'wasm', label: 'WASM (CPU)' },
]

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`
}

function progressPercent(progress: { loadedBytes: number; totalBytes?: number } | null): number {
  if (!progress || !progress.totalBytes) return 0
  return Math.min(100, Math.round((progress.loadedBytes / progress.totalBytes) * 100))
}

function metric(value: number | undefined): string {
  return value === undefined ? '—' : `${Math.round(value)} ms`
}

export default function ModelRunner({ adapters, onSelect }: ModelRunnerProps) {
  const {
    loadModel,
    unloadModel,
    runInference,
    preflightModel,
    outputs,
    outputTensors,
    outputSpecs,
    accelerator,
    setAccelerator,
    resolvedAccelerator,
    modelInfo,
    preflight,
    telemetry,
    error,
    loading,
    loaded,
    downloadProgress,
    modelBase,
    setModelBase,
  } = useModelRunner()
  const [selectedAdapter, setSelectedAdapter] = useState<ModelAdapter | null>(null)
  const [inputValues, setInputValues] = useState<Record<string, unknown>>({})
  const [search, setSearch] = useState('')
  const [downloadingId, setDownloadingId] = useState<string | null>(null)
  const [loadedModelId, setLoadedModelId] = useState<string | null>(null)
  const [storedModels, setStoredModels] = useState<StoredModelInfo[]>([])
  const [storageBusy, setStorageBusy] = useState(false)
  const [modelBaseInput, setModelBaseInput] = useState(modelBase)

  // ponytail: merge so image + text + scalar widgets can coexist on one adapter (e.g. CLIPSeg).
  const mergeInput = useCallback((values: Record<string, unknown>) => {
    setInputValues(prev => ({ ...prev, ...values }))
  }, [])

  const refreshStoredModels = useCallback(async () => {
    setStoredModels(await listStoredModels())
  }, [])

  useEffect(() => {
    void refreshStoredModels()
  }, [refreshStoredModels])

  const handleSelect = (adapter: ModelAdapter) => {
    setSelectedAdapter(adapter)
    setInputValues({})
  }

  const handleLoadSelected = async () => {
    if (!selectedAdapter || selectedAdapter.isPipeline || selectedAdapter.disabled) return
    setDownloadingId(selectedAdapter.modelId)
    setLoadedModelId(selectedAdapter.modelId)
    try {
      await loadModel(selectedAdapter)
      await refreshStoredModels()
    } finally {
      setDownloadingId(null)
    }
  }

  const handleUnload = async () => {
    await unloadModel()
    setLoadedModelId(null)
  }

  const handleRemoveStored = async (modelId: string) => {
    setStorageBusy(true)
    try {
      if (loadedModelId === modelId) await handleUnload()
      await removeStoredModel(modelId)
      await refreshStoredModels()
    } finally {
      setStorageBusy(false)
    }
  }

  const handleClearStored = async () => {
    setStorageBusy(true)
    try {
      if (loadedModelId) await handleUnload()
      await clearStoredModels()
      await refreshStoredModels()
    } finally {
      setStorageBusy(false)
    }
  }

  const handleAcceleratorChange = (next: Accelerator) => {
    if (loadedModelId) void handleUnload()
    setAccelerator(next)
  }

  const commitModelBase = () => {
    if (loadedModelId) void handleUnload()
    setModelBase(modelBaseInput)
  }

  const filtered = search
    ? adapters.filter(a =>
        a.metadata.name.toLowerCase().includes(search.toLowerCase()) ||
        a.metadata.tags.some(t => t.toLowerCase().includes(search.toLowerCase())) ||
        a.modelId.toLowerCase().includes(search.toLowerCase())
      )
    : adapters

  const storedModelMap = useMemo(
    () => new Map(storedModels.map((model) => [model.modelId, model])),
    [storedModels],
  )
  const storedModelIds = useMemo(() => new Set(storedModelMap.keys()), [storedModelMap])
  const selectedLoaded = !!selectedAdapter && !selectedAdapter.isPipeline && loaded && loadedModelId === selectedAdapter.modelId
  const selectedStored = selectedAdapter ? storedModelMap.get(selectedAdapter.modelId) : undefined
  const storedBytes = storedModels.reduce((total, model) => total + model.bytes, 0)
  const recentTelemetry = telemetry.slice(-4).reverse()
  const tartGuide = getTartGuideMessage({
    selectedModelName: selectedAdapter?.metadata.name ?? null,
    loading,
    loaded: selectedLoaded,
    progressPercent: downloadProgress?.totalBytes ? progressPercent(downloadProgress) : null,
    error,
    requestedBackend: accelerator,
    resolvedBackend: selectedLoaded ? resolvedAccelerator : null,
    fallbackCount: selectedLoaded ? modelInfo?.fallbackCount ?? 0 : 0,
    preflightComplete: selectedLoaded && preflight !== null,
    inferenceComplete: selectedLoaded && outputs !== null,
  })

  return (
    <div className="min-h-screen bg-surface-dim">
      <div className="mx-auto p-6" style={{ maxWidth: 900 }}>
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-3xl font-bold text-on-surface">Lil Tart</h1>
            <p className="mt-1 text-sm text-on-surface-variant">
              Browse first. Download only what you choose, then prove the runtime path locally.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-xs text-on-surface-variant">Accelerator:</label>
            <select
              value={accelerator}
              onChange={e => handleAcceleratorChange(e.target.value as Accelerator)}
              className="rounded-lg border border-outline bg-surface-container px-2 py-1 text-xs text-on-surface"
            >
              {ACCEL_OPTIONS.map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>
        </div>

        <input
          type="text"
          placeholder="Search models..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="mb-3 w-full rounded-lg border border-outline bg-surface-container px-4 py-2 text-sm text-on-surface transition-colors focus:border-primary focus:ring-2 focus:ring-primary/30 focus:outline-none"
        />

        <div className="mb-4 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-outline/40 bg-surface-container-low p-3">
          <div>
            <p className="text-xs font-semibold text-on-surface">Model library</p>
            <p className="mt-0.5 text-[11px] text-on-surface-variant">
              {storedModels.length} stored · {formatBytes(storedBytes)}
            </p>
          </div>
          {storedModels.length > 0 && (
            <button
              type="button"
              onClick={() => void handleClearStored()}
              disabled={storageBusy || loading}
              className="rounded-full border border-outline px-3 py-1.5 text-xs text-on-surface-variant transition-colors hover:bg-surface-container-high hover:text-on-surface disabled:opacity-50"
            >
              Clear all downloads
            </button>
          )}
        </div>

        <details className="mb-4 rounded-xl border border-outline/40 bg-surface-container-low px-3 py-2">
          <summary className="cursor-pointer text-xs font-medium text-on-surface-variant">Runtime settings</summary>
          <div className="mt-3">
            <label className="mb-1 block text-xs font-medium text-on-surface-variant">Model server base URL</label>
            <input
              type="url"
              placeholder="https://your-model-server.com/"
              value={modelBaseInput}
              onChange={e => setModelBaseInput(e.target.value)}
              onBlur={commitModelBase}
              onKeyDown={e => { if (e.key === 'Enter') commitModelBase() }}
              className="w-full rounded-lg border border-outline bg-surface-container px-4 py-2 text-sm text-on-surface transition-colors focus:border-primary focus:ring-2 focus:ring-primary/30 focus:outline-none"
            />
          </div>
        </details>

        <ModelList
          adapters={filtered}
          onSelect={handleSelect}
          disabled={loading}
          loadingModelId={downloadingId}
          downloadProgress={downloadingId === selectedAdapter?.modelId ? downloadProgress : null}
          selectedModelId={selectedAdapter?.modelId ?? null}
          loadedModelId={loaded && loadedModelId ? loadedModelId : null}
          storedModelIds={storedModelIds}
        />

        {selectedAdapter && (
          <section className="mt-5 rounded-2xl border border-outline/50 bg-surface-container p-4">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div className="min-w-0 flex-1">
                <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-on-surface-variant">
                  Selected
                </p>
                <h2 className="mt-1 text-lg font-semibold text-on-surface">{selectedAdapter.metadata.name}</h2>
                <p className="mt-1 text-sm text-on-surface-variant">{selectedAdapter.metadata.description}</p>
                {selectedStored && (
                  <p className="mt-2 text-xs text-on-surface-variant">
                    Stored locally · {formatBytes(selectedStored.bytes)} across {selectedStored.assets} {selectedStored.assets === 1 ? 'asset' : 'assets'}
                  </p>
                )}
              </div>

              <div className="flex flex-wrap justify-end gap-2">
                {selectedAdapter.isPipeline ? (
                  <button
                    type="button"
                    onClick={() => onSelect?.(selectedAdapter.modelId)}
                    className="rounded-full bg-primary px-4 py-2 text-xs font-semibold text-on-primary"
                  >
                    Open pipeline
                  </button>
                ) : selectedLoaded ? (
                  <button
                    type="button"
                    onClick={() => void handleUnload()}
                    disabled={loading}
                    className="rounded-full border border-outline px-4 py-2 text-xs font-medium text-on-surface hover:bg-surface-container-high disabled:opacity-50"
                  >
                    Unload from memory
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={() => void handleLoadSelected()}
                    disabled={loading || !!selectedAdapter.disabled}
                    className="rounded-full bg-primary px-4 py-2 text-xs font-semibold text-on-primary disabled:opacity-50"
                  >
                    {selectedStored ? 'Load from device' : 'Download & load'}
                  </button>
                )}

                {selectedStored && !selectedAdapter.isPipeline && (
                  <button
                    type="button"
                    onClick={() => void handleRemoveStored(selectedAdapter.modelId)}
                    disabled={storageBusy || loading}
                    className="rounded-full border border-error/50 px-4 py-2 text-xs font-medium text-error hover:bg-error-container/30 disabled:opacity-50"
                  >
                    Remove download
                  </button>
                )}
              </div>
            </div>
          </section>
        )}

        {error && (
          <div className="mt-3 rounded-lg bg-error-container p-3 text-sm text-on-error-container">
            {error}
          </div>
        )}

        {selectedAdapter && selectedLoaded && (
          <div className="mt-6 space-y-6">
            <section className="rounded-2xl border border-outline/40 bg-surface-container p-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <p className="text-xs font-medium uppercase tracking-wide text-on-surface-variant">Runtime receipt</p>
                  <p className="mt-1 text-sm font-semibold text-on-surface">
                    requested {accelerator.toUpperCase()} → resolved {(resolvedAccelerator ?? 'unknown').toUpperCase()}
                  </p>
                  {selectedAdapter.metadata.tags.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1">
                      {selectedAdapter.metadata.tags.map(t => (
                        <span key={t} className="rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary">{t}</span>
                      ))}
                    </div>
                  )}
                </div>
                <button
                  type="button"
                  onClick={() => void preflightModel()}
                  disabled={loading}
                  className="rounded-full border border-outline px-4 py-2 text-xs font-medium text-on-surface transition-colors hover:bg-surface-container-high disabled:opacity-50"
                >
                  {loading ? 'Working…' : 'Run preflight'}
                </button>
              </div>

              <div className="mt-3 grid grid-cols-2 gap-3 text-xs sm:grid-cols-4">
                <div>
                  <p className="text-on-surface-variant">Compile</p>
                  <p className="font-medium text-on-surface">{metric(modelInfo?.compileDurationMs)}</p>
                </div>
                <div>
                  <p className="text-on-surface-variant">Fallbacks</p>
                  <p className="font-medium text-on-surface">{modelInfo?.fallbackCount ?? 0}</p>
                </div>
                <div>
                  <p className="text-on-surface-variant">Preflight inference</p>
                  <p className="font-medium text-on-surface">{metric(preflight?.inferenceDurationMs)}</p>
                </div>
                <div>
                  <p className="text-on-surface-variant">Preflight outputs</p>
                  <p className="font-medium text-on-surface">{preflight?.outputCount ?? '—'}</p>
                </div>
              </div>

              {recentTelemetry.length > 0 && (
                <div className="mt-4 border-t border-outline/30 pt-3">
                  <p className="mb-2 text-xs font-medium text-on-surface-variant">Recent runtime events</p>
                  <div className="space-y-1 font-mono text-[11px] text-on-surface-variant">
                    {recentTelemetry.map((entry, index) => (
                      <div key={`${entry.timestamp}-${entry.event}-${index}`} className="flex flex-wrap justify-between gap-2">
                        <span>{entry.event} · {entry.resolvedBackend}</span>
                        <span>
                          {entry.inferenceDurationMs !== undefined
                            ? `${Math.round(entry.inferenceDurationMs)} ms`
                            : `${Math.round(entry.compileDurationMs)} ms`}
                          {entry.fallbackCount > 0 ? ` · ${entry.fallbackCount} fallback` : ''}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </section>

            {selectedAdapter.inputSpecs.some(isVisionSpec) && (
              <ImageInput specs={selectedAdapter.inputSpecs} onChange={mergeInput} />
            )}
            {selectedAdapter.inputSpecs.some(isAudioSpec) && (
              <AudioInput specs={selectedAdapter.inputSpecs} onChange={mergeInput} />
            )}
            {selectedAdapter.inputSpecs.some(spec => !isVisionSpec(spec) && !isAudioSpec(spec)) && (
              <InputEditor
                specs={selectedAdapter.inputSpecs.filter(spec => !isVisionSpec(spec) && !isAudioSpec(spec))}
                onChange={mergeInput}
              />
            )}

            <button
              onClick={() => void runInference(inputValues)}
              disabled={loading}
              className="inline-flex items-center justify-center rounded-full bg-primary px-8 py-3 text-sm font-medium text-on-primary shadow-md transition-all duration-300 hover:scale-[1.02] hover:shadow-lg active:scale-[0.97] disabled:opacity-50 disabled:shadow-none"
              style={{ transitionTimingFunction: 'var(--ease-spring)' }}
            >
              {loading ? 'Running...' : `Run Inference (${(resolvedAccelerator ?? accelerator).toUpperCase()})`}
            </button>

            <OutputViewer outputs={outputs} outputTensors={outputTensors} outputSpecs={outputSpecs} />
          </div>
        )}

        {selectedAdapter && downloadingId === selectedAdapter.modelId && loading && (
          <div className="mt-3">
            <p className="text-on-surface-variant">Loading model...</p>
            {downloadProgress && (
              <div className="mt-2">
                <div className="h-2 w-full overflow-hidden rounded-full bg-outline/30">
                  <div
                    className="h-full rounded-full bg-primary transition-all duration-300"
                    style={{ width: `${progressPercent(downloadProgress)}%` }}
                  />
                </div>
                <p className="mt-1 text-xs text-on-surface-variant">
                  {formatBytes(downloadProgress.loadedBytes)}
                  {downloadProgress.totalBytes ? ` / ${formatBytes(downloadProgress.totalBytes)}` : ''}
                  {downloadProgress.totalBytes ? ` (${progressPercent(downloadProgress)}%)` : ''}
                </p>
              </div>
            )}
          </div>
        )}
      </div>

      <TartGuide
        guide={tartGuide}
        onRunPreflight={selectedLoaded ? () => void preflightModel() : undefined}
      />
    </div>
  )
}
