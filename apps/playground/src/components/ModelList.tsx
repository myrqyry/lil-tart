import { useEffect, useMemo, useState } from 'react'
import type { ModelAdapter, VerificationStatus } from '../adapters/types'

interface ModelListProps {
  adapters: ModelAdapter[]
  onSelect: (adapter: ModelAdapter) => void
  disabled?: boolean
  loadingModelId?: string | null
  downloadProgress?: { loadedBytes: number; totalBytes?: number } | null
  selectedModelId?: string | null
  loadedModelId?: string | null
  storedModelIds?: ReadonlySet<string>
  searching?: boolean
}

const GROUP_ORDER = [
  'On this device',
  'Pipelines',
  'Language',
  'Speech & audio',
  'Vision & image',
  'Other',
] as const

type ModelGroup = typeof GROUP_ORDER[number]

const GROUP_LABELS: Record<ModelGroup, string> = {
  'On this device': 'Downloaded',
  Pipelines: 'Pipelines',
  Language: 'Language',
  'Speech & audio': 'Audio',
  'Vision & image': 'Vision',
  Other: 'Other',
}

const VERIFICATION_LABELS: Partial<Record<VerificationStatus, string>> = {
  'compile-verified': 'Compile ✓',
  'inference-verified': 'Runs ✓',
  'output-verified': 'Output ✓',
  'manually-verified': 'Checked ✓',
}

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

function categoryFor(adapter: ModelAdapter, stored: boolean): ModelGroup {
  if (stored) return 'On this device'
  if (adapter.isPipeline) return 'Pipelines'

  const tags = new Set(adapter.metadata.tags.map((tag) => tag.toLowerCase()))
  if (
    tags.has('llm') ||
    tags.has('text') ||
    tags.has('retrieval') ||
    tags.has('encoder') ||
    tags.has('embedding') ||
    tags.has('classification')
  ) return 'Language'

  if (
    tags.has('audio') ||
    tags.has('speech') ||
    tags.has('tts') ||
    tags.has('asr') ||
    tags.has('music') ||
    tags.has('codec')
  ) return 'Speech & audio'

  if (
    tags.has('vision') ||
    tags.has('image') ||
    tags.has('ocr') ||
    tags.has('segmentation') ||
    tags.has('detection') ||
    tags.has('depth') ||
    tags.has('pose')
  ) return 'Vision & image'

  return 'Other'
}

function defaultGroup(grouped: Map<ModelGroup, ModelAdapter[]>): ModelGroup {
  if (grouped.get('Language')?.length) return 'Language'
  return GROUP_ORDER.find((group) => grouped.get(group)?.length) ?? 'Other'
}

export default function ModelList({
  adapters,
  onSelect,
  disabled,
  loadingModelId,
  downloadProgress,
  selectedModelId,
  loadedModelId,
  storedModelIds = new Set(),
  searching = false,
}: ModelListProps) {
  const grouped = useMemo(() => {
    const next = new Map<ModelGroup, ModelAdapter[]>()
    for (const adapter of adapters) {
      const group = categoryFor(adapter, storedModelIds.has(adapter.modelId))
      const values = next.get(group) ?? []
      values.push(adapter)
      next.set(group, values)
    }

    for (const values of next.values()) {
      values.sort((a, b) => a.metadata.name < b.metadata.name ? -1 : a.metadata.name > b.metadata.name ? 1 : 0)
    }
    return next
  }, [adapters, storedModelIds])

  const [activeGroup, setActiveGroup] = useState<ModelGroup>(() => defaultGroup(grouped))

  useEffect(() => {
    if (searching || !selectedModelId) return
    const selected = adapters.find((adapter) => adapter.modelId === selectedModelId)
    if (selected) setActiveGroup(categoryFor(selected, storedModelIds.has(selected.modelId)))
  }, [adapters, searching, selectedModelId, storedModelIds])

  useEffect(() => {
    if (!grouped.get(activeGroup)?.length) setActiveGroup(defaultGroup(grouped))
  }, [activeGroup, grouped])

  const visibleItems = searching
    ? adapters
    : grouped.get(activeGroup) ?? []

  return (
    <div>
      {!searching && (
        <div className="mb-3 flex gap-1 overflow-x-auto pb-1">
          {GROUP_ORDER.map((group) => {
            const count = grouped.get(group)?.length ?? 0
            if (count === 0) return null
            const active = activeGroup === group
            return (
              <button
                key={group}
                type="button"
                onClick={() => setActiveGroup(group)}
                className={`shrink-0 rounded-full px-3 py-1.5 text-xs font-medium transition-colors ${
                  active
                    ? 'bg-primary text-on-primary'
                    : 'bg-surface-container text-on-surface-variant hover:bg-surface-container-high hover:text-on-surface'
                }`}
              >
                {GROUP_LABELS[group]} <span className="opacity-70">{count}</span>
              </button>
            )
          })}
        </div>
      )}

      {searching && (
        <p className="mb-2 text-xs text-on-surface-variant">
          {visibleItems.length} {visibleItems.length === 1 ? 'match' : 'matches'}
        </p>
      )}

      <div className="grid grid-cols-1 gap-1.5 md:grid-cols-2">
        {visibleItems.map((adapter) => {
          const isLoading = loadingModelId === adapter.modelId && !!disabled
          const isSelected = selectedModelId === adapter.modelId
          const isLoaded = loadedModelId === adapter.modelId
          const isStored = storedModelIds.has(adapter.modelId)
          const isUnavailable = !!adapter.disabled
          const verificationStatus = adapter.verification?.status ?? 'registered'
          const verificationLabel = VERIFICATION_LABELS[verificationStatus]

          return (
            <button
              key={adapter.modelId}
              type="button"
              onClick={() => !isUnavailable && onSelect(adapter)}
              disabled={isUnavailable || !!disabled}
              title={isUnavailable ? 'No browser-fetchable .tflite yet — needs locating' : adapter.metadata.description}
              className={`group rounded-lg border px-3 py-2 text-left transition-colors disabled:opacity-55 ${
                isSelected
                  ? 'border-primary bg-primary/7'
                  : 'border-outline/40 bg-surface-container/60 hover:border-primary/40 hover:bg-surface-container'
              }`}
            >
              <div className="flex min-w-0 items-center gap-2">
                <div className="min-w-0 flex-1">
                  <p className="truncate text-xs font-semibold text-on-surface">{adapter.metadata.name}</p>
                  <p className="mt-0.5 truncate text-[10px] text-on-surface-variant">
                    {adapter.metadata.description}
                  </p>
                </div>

                <div className="flex shrink-0 items-center gap-1.5 text-[9px] font-medium">
                  {verificationLabel && !isLoaded && (
                    <span className="text-on-surface-variant">{verificationLabel}</span>
                  )}
                  {isUnavailable ? (
                    <span className="text-on-surface-variant">Unavailable</span>
                  ) : isLoading ? (
                    <span className="text-primary">Downloading</span>
                  ) : isLoaded ? (
                    <span className="text-primary">● Loaded</span>
                  ) : isStored ? (
                    <span className="text-secondary">● Stored</span>
                  ) : null}
                </div>
              </div>

              {isLoading && downloadProgress && (
                <div className="mt-1.5 flex items-center gap-2">
                  <div className="h-1 flex-1 overflow-hidden rounded-full bg-outline/30">
                    <div
                      className="h-full rounded-full bg-primary transition-all duration-300"
                      style={{ width: `${progressPercent(downloadProgress)}%` }}
                    />
                  </div>
                  <span className="shrink-0 text-[9px] text-on-surface-variant">
                    {downloadProgress.totalBytes
                      ? `${progressPercent(downloadProgress)}% · ${formatBytes(downloadProgress.loadedBytes)}`
                      : formatBytes(downloadProgress.loadedBytes)}
                  </span>
                </div>
              )}
            </button>
          )
        })}
      </div>
    </div>
  )
}
