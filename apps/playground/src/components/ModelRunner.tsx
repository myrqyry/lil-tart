import { useCallback, useEffect, useMemo, useState } from 'react'
import { Delete2Regular, StorageRegular } from '@mingcute/react/core-regular'
import { useModelRunner } from '../hooks/useModelRunner'
import type { Accelerator } from '../hooks/useModelRunner'
import type { ModelAdapter, TensorSpec } from '../adapters/types'
import { getTartGuideMessage } from '../tartGuide'
import {
  clearStoredModels,
  listOrphanedModels,
  listStoredModels,
  removeOrphanedModel,
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
    operation,
    runtimePathProof,
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
  const [orphanedModels, setOrphanedModels] = useState<StoredModelInfo[]>([])
  const [storageBusy, setStorageBusy] = useState(false)
  const [modelBaseInput, setModelBaseInput] = useState(modelBase)

  // ponytail: merge so image + text + scalar widgets can coexist on one adapter (e.g. CLIPSeg).
  const mergeInput = useCallback((values: Record<string, unknown>) => {
    setInputValues(prev => ({ ...prev, ...values }))
  }, [])

  const refreshStoredModels = useCallback(async () => {
    setStoredModels(await listStoredModels(modelBase))
    setOrphanedModels(await listOrphanedModels(modelBase))
  }, [modelBase])

  useEffect(() => {
    void refreshStoredModels()
  }, [refreshStoredModels])

  const handleSelect = (adapter: ModelAdapter) => {
    setSelectedAdapter((current) => current?.modelId === adapter.modelId ? null : adapter)
    setInputValues({})
  }

  const handleLoad = async (adapter: ModelAdapter) => {
    if (adapter.isPipeline || adapter.disabled) return
    setSelectedAdapter(adapter)
    setInputValues({})
    setDownloadingId(adapter.modelId)
    setLoadedModelId(adapter.modelId)
    try {
      await loadModel(adapter)
      await refreshStoredModels()
    } finally {
      setDownloadingId(null)
    }
  }

  const handleUnload = async (adapter?: ModelAdapter) => {
    if (adapter && loadedModelId !== adapter.modelId) return
    await unloadModel()
    setLoadedModelId(null)
  }

  const handleRemoveStored = async (modelId: string) => {
    setStorageBusy(true)
    try {
      if (loadedModelId === modelId) await handleUnload()
      await removeStoredModel(modelId, modelBase)
      await refreshStoredModels()
    } finally {
      setStorageBusy(false)
    }
  }

  // Reclaims only the bytes written under a previous base. The live copy is a
  // different cache entry and must survive, so this does not unload the model.
  const handleRemoveOrphaned = async (modelId: string) => {
    setStorageBusy(true)
    try {
      await removeOrphanedModel(modelId, modelBase)
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
  const selectedLoaded = !!selectedAdapter && !selectedAdapter.isPipeline && loaded && loadedModelId === selectedAdapter.modelId
  const backendConstraints = selectedAdapter && !selectedAdapter.isPipeline
    ? [
        ...(selectedAdapter.requiredBackend ? [{ graph: 'main', backend: selectedAdapter.requiredBackend }] : []),
        ...(selectedAdapter.graphs ?? [])
          .filter((graph) => graph.requiredBackend)
          .map((graph) => ({ graph: graph.name, backend: graph.requiredBackend! })),
      ]
    : []
  const activeBackendOverrides = accelerator === 'auto'
    ? []
    : backendConstraints.filter((constraint) => constraint.backend !== accelerator)
  const runtimeFallbackCount = selectedLoaded
    ? runtimePathProof?.fallbackCount ?? modelInfo?.fallbackCount ?? 0
    : 0
  const storedBytes = storedModels.reduce((total, model) => total + model.bytes, 0)
  const orphanedBytes = orphanedModels.reduce((total, model) => total + model.bytes, 0)
  const recentTelemetry = telemetry.slice(-4).reverse()
  const tartGuide = getTartGuideMessage({
    selectedModelName: selectedAdapter?.metadata.name ?? null,
    operation,
    loaded: selectedLoaded,
    progressPercent: downloadProgress?.totalBytes ? progressPercent(downloadProgress) : null,
    error,
    requestedBackend: accelerator,
    resolvedBackend: selectedLoaded ? resolvedAccelerator : null,
    backendOverrides: activeBackendOverrides,
    runtimeGraphBackends: runtimePathProof?.graphBackends ?? [],
    fallbackCount: runtimeFallbackCount,
    preflightComplete: selectedLoaded && preflight !== null,
    pathProofAvailable: selectedLoaded && runtimePathProof !== null,
  })


  return (
    <div className="app-shell min-h-screen bg-surface-dim">
      <div
        className="workspace-grid mx-auto grid w-full gap-4 px-3 py-2 md:px-4 lg:grid-cols-[minmax(32rem,42%)_minmax(0,1fr)] xl:px-5"
        style={{ maxWidth: 1800 }}
      >
        <aside className="lg:sticky lg:top-2 lg:flex lg:h-[calc(100vh-1rem)] lg:min-h-0 lg:flex-col">
          <TartGuide
            guide={tartGuide}
            onRunPreflight={selectedLoaded ? () => void preflightModel() : undefined}
          />

          <section className="neo-pod inference-pod mt-3 flex min-h-0 flex-1 flex-col overflow-hidden">
            <div className="pod-header px-4 py-3">
              <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-on-surface-muted">Inference lab</p>
              <div className="mt-0.5 flex items-center justify-between gap-2">
                <h2 className="text-[15px] font-semibold text-on-surface">
                  {selectedAdapter ? selectedAdapter.metadata.name : 'No model selected'}
                </h2>
                {selectedLoaded && (
                  <span className="rounded-full bg-tertiary-container px-2 py-0.5 text-[11px] font-semibold text-on-tertiary-container">
                    loaded
                  </span>
                )}
              </div>
            </div>

            <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3 [scrollbar-width:thin]">
              {selectedAdapter ? (
                <>
                  <section>
                    <p className="text-[13px] leading-relaxed text-on-surface-variant">
                      {selectedAdapter.metadata.description}
                    </p>
                    <div className="mt-2 flex flex-wrap gap-1">
                      {selectedAdapter.metadata.tags.map((tag) => (
                        <span key={tag} className="tag-chip px-2 py-1 text-[11px] text-on-surface-variant">
                          {tag}
                        </span>
                      ))}
                      {selectedAdapter.verification?.status && (
                        <span className="tag-chip tag-chip--verified px-2 py-1 text-[11px] text-on-tertiary-container">
                          {selectedAdapter.verification.status.replace(/-/g, ' ')}
                        </span>
                      )}
                    </div>

                    <details className="neo-well mt-2 px-3 py-2.5 text-xs text-on-surface-variant">
                      <summary className="cursor-pointer select-none font-medium text-on-surface">Model details</summary>
                      <div className="mt-2 space-y-2">
                        <div>
                          <p className="uppercase tracking-wide text-on-surface-muted">Model id</p>
                          <p className="mt-0.5 break-all font-mono">{selectedAdapter.modelId}</p>
                        </div>
                        <div>
                          <p className="uppercase tracking-wide text-on-surface-muted">Asset</p>
                          <p className="mt-0.5 break-all font-mono">{selectedAdapter.metadata.modelPath}</p>
                        </div>
                      </div>
                    </details>
                  </section>

                  <section className="border-t border-outline-variant/60 pt-3">
                    <div className="flex items-center justify-between gap-2">
                      <div>
                        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-on-surface-muted">Test backend</p>
                        <p className="mt-0.5 text-[11px] text-on-surface-variant">Selection stays yours; verified graph pins stay visible.</p>
                      </div>
                    </div>
                    <select
                      value={accelerator}
                      onChange={e => handleAcceleratorChange(e.target.value as Accelerator)}
                      className="neo-select mt-2 w-full px-3 py-2.5 text-sm text-on-surface"
                    >
                      {ACCEL_OPTIONS.map(o => (
                        <option key={o.value} value={o.value}>{o.label}</option>
                      ))}
                    </select>

                    {activeBackendOverrides.length > 0 && (
                      <div className="neo-alert mt-2 p-2.5 text-xs text-on-tertiary-container">
                        <p className="font-semibold">Correctness override</p>
                        <p className="mt-1 leading-snug">
                          {activeBackendOverrides
                            .map(({ graph, backend }) => graph + ' → ' + backend.toUpperCase())
                            .join(', ')}
                        </p>
                      </div>
                    )}
                  </section>

                  {error && (
                    <div className="rounded-lg bg-error-container p-2.5 text-xs leading-relaxed text-on-error-container">
                      {error}
                    </div>
                  )}

                  {selectedLoaded ? (
                    <>
                      <section className="border-t border-outline-variant/60 pt-3">
                        <div className="mb-2 flex items-center justify-between gap-2">
                          <div>
                            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-on-surface-muted">Inference workspace</p>
                            <p className="mt-0.5 text-[11px] text-on-surface-variant">
                              Real inputs and outputs live here in the inference panel.
                            </p>
                          </div>
                          <span className="rounded-full bg-surface-container-high px-2 py-0.5 text-[10px] font-medium text-on-surface-variant">
                            {activeBackendOverrides.length > 0 ? 'mixed' : (resolvedAccelerator ?? accelerator).toUpperCase()}
                          </span>
                        </div>

                        <div className="neo-well space-y-3 p-3">
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
                            className="expressive-primary-button inline-flex w-full items-center justify-center px-4 py-3 text-sm font-semibold text-on-primary disabled:opacity-50"
                            style={{ transitionTimingFunction: 'var(--ease-spring)' }}
                          >
                            {operation === 'inference'
                              ? 'Running inference…'
                              : activeBackendOverrides.length > 0
                                ? 'Run Inference · mixed backends'
                                : 'Run Inference · ' + (resolvedAccelerator ?? accelerator).toUpperCase()}
                          </button>
                        </div>

                        <div className="neo-well mt-2 min-w-0 p-3">
                          <OutputViewer outputs={outputs} outputTensors={outputTensors} outputSpecs={outputSpecs} />
                        </div>
                      </section>

                    <section className="border-t border-outline-variant/60 pt-3">
                      <div className="flex items-center justify-between gap-2">
                        <div>
                          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-on-surface-muted">Runtime test</p>
                          <p className="mt-0.5 text-[11px] text-on-surface-variant">
                            Main graph first; complete graph paths appear after real inference.
                          </p>
                        </div>
                        <button
                          type="button"
                          onClick={() => void preflightModel()}
                          disabled={loading}
                          className="expressive-secondary-button px-3 py-2 text-xs font-semibold text-on-surface disabled:opacity-50"
                        >
                          {operation === 'preflight' ? 'Checking…' : 'Preflight'}
                        </button>
                      </div>

                      <div className="neo-well mt-2 p-3">
                        <p className="text-[11px] uppercase tracking-wide text-on-surface-muted">Main graph</p>
                        <p className="mt-1 text-[13px] font-semibold text-on-surface">
                          {(modelInfo?.requestedBackend ?? accelerator).toUpperCase()} → {(resolvedAccelerator ?? 'unknown').toUpperCase()}
                        </p>
                        <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
                          <div>
                            <p className="text-on-surface-muted">Compile</p>
                            <p className="font-medium text-on-surface">{metric(modelInfo?.compileDurationMs)}</p>
                          </div>
                          <div>
                            <p className="text-on-surface-muted">{runtimePathProof ? 'Session fallbacks' : 'Main fallbacks'}</p>
                            <p className="font-medium text-on-surface">{runtimeFallbackCount}</p>
                          </div>
                          <div>
                            <p className="text-on-surface-muted">Preflight</p>
                            <p className="font-medium text-on-surface">{metric(preflight?.inferenceDurationMs)}</p>
                          </div>
                          <div>
                            <p className="text-on-surface-muted">Outputs</p>
                            <p className="font-medium text-on-surface">{preflight?.outputCount ?? '—'}</p>
                          </div>
                        </div>
                      </div>

                      {runtimePathProof && (
                        <div className="neo-proof mt-2 p-3">
                          <div className="flex items-center justify-between gap-2">
                            <div>
                              <p className="text-xs font-semibold text-on-surface">Session proof</p>
                              <p className="mt-0.5 text-[11px] text-on-surface-variant">
                                selected {runtimePathProof.selectedBackend.toUpperCase()} · {runtimePathProof.graphBackends.length} graph {runtimePathProof.graphBackends.length === 1 ? 'path' : 'paths'}
                              </p>
                            </div>
                            <span className="rounded-md bg-primary/10 px-1.5 py-0.5 text-[10px] font-medium text-primary">
                              {runtimePathProof.inferenceEvents.length} events
                            </span>
                          </div>
                          <div className="mt-2 space-y-1.5 font-mono text-[11px] text-on-surface-variant">
                            {runtimePathProof.graphBackends.map((graph) => (
                              <div key={graph.graph} className="rounded-md bg-surface/45 px-2 py-1.5">
                                <div className="flex items-center justify-between gap-2">
                                  <span className="font-semibold text-on-surface">{graph.graph}</span>
                                  <span>{graph.requestedBackend.toUpperCase()} → {graph.resolvedBackend.toUpperCase()}</span>
                                </div>
                                {graph.fallbackCount > 0 && (
                                  <p className="mt-0.5 text-error">{graph.fallbackCount} runtime fallback{graph.fallbackCount === 1 ? '' : 's'}</p>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {recentTelemetry.length > 0 && (
                        <details className="neo-well mt-2 px-3 py-2.5 text-xs">
                          <summary className="cursor-pointer select-none font-medium text-on-surface">Recent runtime events</summary>
                          <div className="mt-2 space-y-1 font-mono text-[10px] text-on-surface-variant">
                            {recentTelemetry.map((entry, index) => (
                              <div key={[entry.timestamp, entry.event, index].join('-')}>
                                {entry.event} · {entry.requestedBackend} → {entry.resolvedBackend}
                                {entry.fallbackCount > 0 ? ' · ' + entry.fallbackCount + ' fallback' : ''}
                              </div>
                            ))}
                          </div>
                        </details>
                      )}
                    </section>
                    </>
                  ) : (
                    <div className="rounded-lg border border-dashed border-outline-variant p-3 text-xs leading-relaxed text-on-surface-variant">
                      Download and load this model from its card to unlock preflight, inference, and runtime receipts.
                    </div>
                  )}

                  <details className="border-t border-outline-variant/60 pt-3">
                    <summary className="cursor-pointer select-none text-xs font-semibold uppercase tracking-[0.14em] text-on-surface-muted">
                      Runtime settings
                    </summary>
                    <div className="mt-2">
                      <label className="block text-[11px] font-medium text-on-surface-muted">Model server base URL</label>
                      <input
                        type="url"
                        placeholder="https://your-model-server.com/"
                        value={modelBaseInput}
                        onChange={e => setModelBaseInput(e.target.value)}
                        onBlur={commitModelBase}
                        onKeyDown={e => { if (e.key === 'Enter') commitModelBase() }}
                        className="neo-input mt-1 w-full px-3 py-2 text-xs text-on-surface focus:outline-none"
                      />
                    </div>
                  </details>
                </>
              ) : (
                <div className="neo-empty-state p-5 text-center">
                  <p className="text-sm font-semibold text-on-surface">Pick a model from the library.</p>
                  <p className="mt-1 text-xs leading-relaxed text-on-surface-variant">
                    Its description, backend constraints, preflight tools, and runtime proof will live here.
                  </p>
                </div>
              )}
            </div>
          </section>
        </aside>

        <main className="min-w-0">

          <div className="mb-3 flex items-center gap-2">
            <input
              type="text"
              placeholder="Search name, task, or model id…"
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="neo-search min-w-0 flex-1 px-4 py-3 text-[15px] text-on-surface placeholder:text-on-surface-muted focus:outline-none"
            />

            {(storedModels.length > 0 || orphanedModels.length > 0) && (
              <div className="neo-badge flex shrink-0 items-center gap-1.5 px-2 py-1.5 text-xs text-on-surface-variant">
                <StorageRegular className="h-4 w-4 text-secondary" />
                <span className="font-medium text-on-surface">{storedModels.length}</span>
                <span className="hidden text-on-surface-muted xl:inline">{formatBytes(storedBytes)}</span>
                <button
                  type="button"
                  onClick={() => void handleClearStored()}
                  disabled={storageBusy || loading}
                  className="ml-0.5 inline-flex h-7 w-7 items-center justify-center rounded-full text-error transition-transform hover:scale-110 hover:bg-error-container/30 active:scale-95 disabled:opacity-40"
                  title="Clear downloaded models"
                  aria-label="Clear downloaded models"
                >
                  <Delete2Regular className="h-4 w-4" />
                </button>
              </div>
            )}
          </div>

          {orphanedModels.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-1.5 text-[11px] text-on-surface-variant">
              {orphanedModels.map((model) => (
                <button
                  key={model.modelId}
                  type="button"
                  title={model.unverified ? 'Load this model to verify its older cache metadata, or clear all downloads.' : 'Remove ' + model.modelId}
                  onClick={() => void handleRemoveOrphaned(model.modelId)}
                  disabled={storageBusy || loading || model.unverified}
                  className="tag-chip px-2 py-1 text-error disabled:opacity-45"
                >
                  {model.unverified ? 'Unverified' : 'Remove'} {model.modelId}
                </button>
              ))}
            </div>
          )}

          <ModelList
            adapters={filtered}
            onSelect={handleSelect}
            onLoad={(adapter) => void handleLoad(adapter)}
            onUnload={(adapter) => void handleUnload(adapter)}
            onOpenPipeline={(modelId) => onSelect?.(modelId)}
            onRemoveStored={(modelId) => void handleRemoveStored(modelId)}
            disabled={loading}
            storageBusy={storageBusy}
            loadingModelId={downloadingId}
            downloadProgress={downloadingId ? downloadProgress : null}
            selectedModelId={selectedAdapter?.modelId ?? null}
            loadedModelId={loaded && loadedModelId ? loadedModelId : null}
            storedModels={storedModelMap}
            searching={search.trim().length > 0}
          />

          {selectedAdapter && downloadingId === selectedAdapter.modelId && operation === 'model-load' && (
            <div className="neo-pod mt-3 p-4">
              <div className="flex items-center justify-between gap-2 text-xs text-on-surface-variant">
                <span>Loading {selectedAdapter.metadata.name}</span>
                {downloadProgress?.totalBytes && <span>{progressPercent(downloadProgress)}%</span>}
              </div>
              {downloadProgress && (
                <>
                  <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-outline/30">
                    <div
                      className="h-full rounded-full bg-primary transition-all duration-300"
                      style={{ width: progressPercent(downloadProgress) + '%' }}
                    />
                  </div>
                  <p className="mt-1 text-[11px] text-on-surface-muted">
                    {formatBytes(downloadProgress.loadedBytes)}
                    {downloadProgress.totalBytes ? ' / ' + formatBytes(downloadProgress.totalBytes) : ''}
                  </p>
                </>
              )}
            </div>
          )}

        </main>
      </div>
    </div>
  )
}
