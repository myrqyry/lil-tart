import { useEffect, useMemo, useState } from 'react'
import type { ModelAdapter, VerificationStatus } from '../adapters/types'

interface StoredModelSummary {
  bytes: number
  assets: number
}

interface ModelListProps {
  adapters: ModelAdapter[]
  onSelect: (adapter: ModelAdapter) => void
  onLoadSelected: () => void
  onUnloadSelected: () => void
  onOpenPipeline?: (modelId: string) => void
  onRemoveStored: (modelId: string) => void
  disabled?: boolean
  storageBusy?: boolean
  loadingModelId?: string | null
  downloadProgress?: { loadedBytes: number; totalBytes?: number } | null
  selectedModelId?: string | null
  loadedModelId?: string | null
  storedModels?: ReadonlyMap<string, StoredModelSummary>
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
type ModelFamily = Exclude<ModelGroup, 'On this device'>

const GROUP_LABELS: Record<ModelGroup, string> = {
  'On this device': 'Downloaded',
  Pipelines: 'Pipelines',
  Language: 'Language',
  'Speech & audio': 'Audio',
  'Vision & image': 'Vision',
  Other: 'Other',
}

const FAMILY_LABELS: Record<ModelFamily, string> = {
  Pipelines: 'Pipeline',
  Language: 'Language',
  'Speech & audio': 'Audio',
  'Vision & image': 'Vision',
  Other: 'Other',
}

const FAMILY_CARD_CLASS: Record<ModelFamily, string> = {
  Pipelines: 'border-type-pipeline/30 hover:border-type-pipeline/55',
  Language: 'border-type-language/30 hover:border-type-language/55',
  'Speech & audio': 'border-type-audio/30 hover:border-type-audio/55',
  'Vision & image': 'border-type-vision/30 hover:border-type-vision/55',
  Other: 'border-type-other/30 hover:border-type-other/55',
}

const FAMILY_SELECTED_CLASS: Record<ModelFamily, string> = {
  Pipelines: 'border-type-pipeline/70 bg-type-pipeline/8',
  Language: 'border-type-language/70 bg-type-language/8',
  'Speech & audio': 'border-type-audio/70 bg-type-audio/8',
  'Vision & image': 'border-type-vision/70 bg-type-vision/8',
  Other: 'border-type-other/70 bg-type-other/8',
}

const FAMILY_TEXT_CLASS: Record<ModelFamily, string> = {
  Pipelines: 'text-type-pipeline',
  Language: 'text-type-language',
  'Speech & audio': 'text-type-audio',
  'Vision & image': 'text-type-vision',
  Other: 'text-type-other',
}

const GROUP_TAB_CLASS: Record<ModelGroup, string> = {
  'On this device': 'text-secondary hover:bg-secondary-container/35',
  Pipelines: 'text-type-pipeline hover:bg-type-pipeline/10',
  Language: 'text-type-language hover:bg-type-language/10',
  'Speech & audio': 'text-type-audio hover:bg-type-audio/10',
  'Vision & image': 'text-type-vision hover:bg-type-vision/10',
  Other: 'text-type-other hover:bg-type-other/10',
}

const GROUP_ACTIVE_CLASS: Record<ModelGroup, string> = {
  'On this device': 'bg-secondary-container text-on-secondary-container',
  Pipelines: 'bg-type-pipeline/18 text-type-pipeline',
  Language: 'bg-type-language/18 text-type-language',
  'Speech & audio': 'bg-type-audio/18 text-type-audio',
  'Vision & image': 'bg-type-vision/18 text-type-vision',
  Other: 'bg-type-other/18 text-type-other',
}

const VERIFICATION_LABELS: Partial<Record<VerificationStatus, string>> = {
  'compile-verified': 'Compile verified',
  'inference-verified': 'Inference verified',
  'output-verified': 'Output verified',
  'manually-verified': 'Manually verified',
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

function familyFor(adapter: ModelAdapter): ModelFamily {
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

function groupFor(adapter: ModelAdapter, stored: boolean): ModelGroup {
  return stored ? 'On this device' : familyFor(adapter)
}

function defaultGroup(grouped: Map<ModelGroup, ModelAdapter[]>): ModelGroup {
  if (grouped.get('Language')?.length) return 'Language'
  return GROUP_ORDER.find((group) => grouped.get(group)?.length) ?? 'Other'
}

export default function ModelList({
  adapters,
  onSelect,
  onLoadSelected,
  onUnloadSelected,
  onOpenPipeline,
  onRemoveStored,
  disabled,
  storageBusy,
  loadingModelId,
  downloadProgress,
  selectedModelId,
  loadedModelId,
  storedModels = new Map(),
  searching = false,
}: ModelListProps) {
  const grouped = useMemo(() => {
    const next = new Map<ModelGroup, ModelAdapter[]>()
    for (const adapter of adapters) {
      const group = groupFor(adapter, storedModels.has(adapter.modelId))
      const values = next.get(group) ?? []
      values.push(adapter)
      next.set(group, values)
    }

    for (const values of next.values()) {
      values.sort((a, b) => a.metadata.name < b.metadata.name ? -1 : a.metadata.name > b.metadata.name ? 1 : 0)
    }
    return next
  }, [adapters, storedModels])

  const [activeGroup, setActiveGroup] = useState<ModelGroup>(() => defaultGroup(grouped))

  useEffect(() => {
    if (searching || !selectedModelId) return
    const selected = adapters.find((adapter) => adapter.modelId === selectedModelId)
    if (selected) setActiveGroup(groupFor(selected, storedModels.has(selected.modelId)))
  }, [adapters, searching, selectedModelId, storedModels])

  useEffect(() => {
    if (!grouped.get(activeGroup)?.length) setActiveGroup(defaultGroup(grouped))
  }, [activeGroup, grouped])

  const visibleItems = searching ? adapters : grouped.get(activeGroup) ?? []

  return (
    <div>
      {!searching && (
        <div className="mb-2 flex gap-1 overflow-x-auto pb-1">
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
                  active ? GROUP_ACTIVE_CLASS[group] : GROUP_TAB_CLASS[group]
                }`}
              >
                {GROUP_LABELS[group]} <span className="opacity-65">{count}</span>
              </button>
            )
          })}
        </div>
      )}

      {searching && (
        <p className="mb-2 text-xs text-on-surface-muted">
          {visibleItems.length} {visibleItems.length === 1 ? 'match' : 'matches'}
        </p>
      )}

      <div className="grid grid-cols-1 gap-1.5 md:grid-cols-2">
        {visibleItems.map((adapter) => {
          const family = familyFor(adapter)
          const isLoading = loadingModelId === adapter.modelId
          const isSelected = selectedModelId === adapter.modelId
          const isLoaded = loadedModelId === adapter.modelId
          const stored = storedModels.get(adapter.modelId)
          const isUnavailable = !!adapter.disabled
          const verificationStatus = adapter.verification?.status ?? 'registered'
          const verificationLabel = VERIFICATION_LABELS[verificationStatus]

          return (
            <article
              key={adapter.modelId}
              className={`overflow-hidden rounded-lg border bg-surface-container-low/70 transition-colors ${
                isSelected
                  ? `md:col-span-2 ${FAMILY_SELECTED_CLASS[family]}`
                  : FAMILY_CARD_CLASS[family]
              }`}
            >
              <button
                type="button"
                onClick={() => !isUnavailable && onSelect(adapter)}
                disabled={isUnavailable || (!!disabled && !isSelected)}
                title={isUnavailable ? 'No browser-fetchable .tflite yet — needs locating' : undefined}
                className="flex w-full items-center gap-2 px-3 py-2 text-left disabled:opacity-55"
              >
                <span className={`h-2 w-2 shrink-0 rounded-full bg-current ${FAMILY_TEXT_CLASS[family]}`} />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-xs font-semibold text-on-surface">{adapter.metadata.name}</p>
                </div>
                <span className={`shrink-0 text-[9px] font-semibold uppercase tracking-wide ${FAMILY_TEXT_CLASS[family]}`}>
                  {FAMILY_LABELS[family]}
                </span>
                {isLoading ? (
                  <span className="shrink-0 text-[9px] font-medium text-primary">Downloading</span>
                ) : isLoaded ? (
                  <span className="shrink-0 text-[9px] font-medium text-tertiary">● Loaded</span>
                ) : stored ? (
                  <span className="shrink-0 text-[9px] font-medium text-secondary">● Stored</span>
                ) : isUnavailable ? (
                  <span className="shrink-0 text-[9px] text-on-surface-muted">Unavailable</span>
                ) : null}
                <span className="shrink-0 text-[10px] text-on-surface-muted">{isSelected ? '▴' : '▾'}</span>
              </button>

              {isSelected && (
                <div className="border-t border-outline-variant/60 px-3 pb-3 pt-2.5">
                  <p className="max-w-3xl text-xs leading-relaxed text-on-surface-variant">
                    {adapter.metadata.description}
                  </p>

                  <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[10px]">
                    {adapter.metadata.tags.map((tag) => (
                      <span key={tag} className="rounded-md bg-surface-container-high px-1.5 py-0.5 text-on-surface-variant">
                        {tag}
                      </span>
                    ))}
                    {verificationLabel && (
                      <span className="rounded-md bg-tertiary-container/70 px-1.5 py-0.5 text-on-tertiary-container">
                        {verificationLabel}
                      </span>
                    )}
                    {adapter.verification?.backends?.map((backend) => (
                      <span key={backend} className="rounded-md bg-surface-container-highest px-1.5 py-0.5 text-on-surface-variant">
                        {backend.toUpperCase()}
                      </span>
                    ))}
                    {stored && (
                      <span className="rounded-md bg-secondary-container/70 px-1.5 py-0.5 text-on-secondary-container">
                        {formatBytes(stored.bytes)} · {stored.assets} {stored.assets === 1 ? 'asset' : 'assets'}
                      </span>
                    )}
                  </div>

                  {isLoading && downloadProgress && (
                    <div className="mt-2.5 flex items-center gap-2">
                      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-outline-variant">
                        <div
                          className="h-full rounded-full bg-primary transition-all duration-300"
                          style={{ width: `${progressPercent(downloadProgress)}%` }}
                        />
                      </div>
                      <span className="shrink-0 text-[10px] text-on-surface-variant">
                        {downloadProgress.totalBytes
                          ? `${progressPercent(downloadProgress)}% · ${formatBytes(downloadProgress.loadedBytes)} / ${formatBytes(downloadProgress.totalBytes)}`
                          : formatBytes(downloadProgress.loadedBytes)}
                      </span>
                    </div>
                  )}

                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {adapter.isPipeline ? (
                      <button
                        type="button"
                        onClick={() => onOpenPipeline?.(adapter.modelId)}
                        className="rounded-lg bg-primary px-3 py-1.5 text-xs font-semibold text-on-primary"
                      >
                        Open pipeline
                      </button>
                    ) : isLoaded ? (
                      <button
                        type="button"
                        onClick={onUnloadSelected}
                        disabled={disabled}
                        className="rounded-lg bg-surface-container-highest px-3 py-1.5 text-xs font-medium text-on-surface hover:bg-surface-bright disabled:opacity-50"
                      >
                        Unload from memory
                      </button>
                    ) : (
                      <button
                        type="button"
                        onClick={onLoadSelected}
                        disabled={disabled || isUnavailable}
                        className="rounded-lg bg-primary px-3 py-1.5 text-xs font-semibold text-on-primary disabled:opacity-50"
                      >
                        {stored ? 'Load from device' : 'Download & load'}
                      </button>
                    )}

                    {stored && !adapter.isPipeline && (
                      <button
                        type="button"
                        onClick={() => onRemoveStored(adapter.modelId)}
                        disabled={storageBusy || disabled}
                        className="rounded-lg px-3 py-1.5 text-xs font-medium text-error transition-colors hover:bg-error-container/35 disabled:opacity-50"
                      >
                        Remove download
                      </button>
                    )}
                  </div>
                </div>
              )}
            </article>
          )
        })}
      </div>
    </div>
  )
}
