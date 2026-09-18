import { useMemo, useState } from 'react'
import type { ModelAdapter, VerificationStatus } from '../adapters/types'

interface StoredModelSummary {
  bytes: number
  assets: number
}

interface ModelListProps {
  adapters: ModelAdapter[]
  onSelect: (adapter: ModelAdapter) => void
  onLoad: (adapter: ModelAdapter) => void
  onUnload: (adapter: ModelAdapter) => void
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

const FAMILY_ORDER = ['Pipelines', 'Language', 'Speech & audio', 'Vision & image', 'Other'] as const
type ModelFamily = typeof FAMILY_ORDER[number]
type ModelFilter = 'All' | 'Downloaded' | ModelFamily

const FILTER_ORDER: ModelFilter[] = ['All', 'Downloaded', ...FAMILY_ORDER]

const FILTER_LABELS: Record<ModelFilter, string> = {
  All: 'All models',
  Downloaded: 'Downloaded',
  Pipelines: 'Pipelines',
  Language: 'Language',
  'Speech & audio': 'Audio',
  'Vision & image': 'Vision',
  Other: 'Other',
}

const FAMILY_CLASS: Record<ModelFamily, string> = {
  Pipelines: 'model-card--pipeline',
  Language: 'model-card--language',
  'Speech & audio': 'model-card--audio',
  'Vision & image': 'model-card--vision',
  Other: 'model-card--other',
}

const FILTER_COLOR_CLASS: Record<ModelFamily, string> = {
  Pipelines: 'text-type-pipeline',
  Language: 'text-type-language',
  'Speech & audio': 'text-type-audio',
  'Vision & image': 'text-type-vision',
  Other: 'text-type-other',
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

function FamilyGlyph({ family, className = '' }: { family: ModelFamily; className?: string }) {
  const common = {
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: 1.8,
    strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const,
    className,
    'aria-hidden': true,
  }

  if (family === 'Speech & audio') {
    return (
      <svg {...common}>
        <path d="M4 13v-2M8 16V8m4 11V5m4 11V8m4 5v-2" />
      </svg>
    )
  }

  if (family === 'Vision & image') {
    return (
      <svg {...common}>
        <rect x="3.5" y="4" width="17" height="16" rx="3" />
        <circle cx="9" cy="9" r="1.5" />
        <path d="m6 17 4-4 3 3 2.5-2.5L18 16" />
      </svg>
    )
  }

  if (family === 'Language') {
    return (
      <svg {...common}>
        <path d="M5 6h14M8 6v12m8-12v12M5 18h14" />
      </svg>
    )
  }

  if (family === 'Pipelines') {
    return (
      <svg {...common}>
        <circle cx="6" cy="6" r="2" />
        <circle cx="18" cy="6" r="2" />
        <circle cx="12" cy="18" r="2" />
        <path d="M7.7 7.1 10.8 16M16.3 7.1 13.2 16M8 6h8" />
      </svg>
    )
  }

  return (
    <svg {...common}>
      <rect x="4" y="4" width="6" height="6" rx="1.5" />
      <rect x="14" y="4" width="6" height="6" rx="1.5" />
      <rect x="4" y="14" width="6" height="6" rx="1.5" />
      <path d="M17 14v6m-3-3h6" />
    </svg>
  )
}

function FilterGlyph({ filter, className = '' }: { filter: ModelFilter; className?: string }) {
  if (FAMILY_ORDER.includes(filter as ModelFamily)) {
    return <FamilyGlyph family={filter as ModelFamily} className={className} />
  }

  if (filter === 'Downloaded') {
    return (
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
        className={className}
        aria-hidden
      >
        <path d="M12 4v10m-4-4 4 4 4-4M5 19h14" />
      </svg>
    )
  }

  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden
    >
      <rect x="4" y="4" width="6" height="6" rx="1.5" />
      <rect x="14" y="4" width="6" height="6" rx="1.5" />
      <rect x="4" y="14" width="6" height="6" rx="1.5" />
      <rect x="14" y="14" width="6" height="6" rx="1.5" />
    </svg>
  )
}

function hasAny(tags: Set<string>, values: readonly string[]): boolean {
  return values.some((value) => tags.has(value))
}

function familyFor(adapter: ModelAdapter): ModelFamily {
  if (adapter.isPipeline) return 'Pipelines'

  const tags = new Set(adapter.metadata.tags.map((tag) => tag.toLowerCase()))
  const searchable = `${adapter.metadata.name} ${adapter.metadata.description}`.toLowerCase()

  if (
    hasAny(tags, ['audio', 'speech', 'tts', 'asr', 'music', 'codec', 'voice']) ||
    /wav2vec|whisper|speech|audio|music|voice|tts|asr/.test(searchable)
  ) return 'Speech & audio'

  if (
    hasAny(tags, ['vision', 'image', 'ocr', 'segmentation', 'detection', 'depth', 'pose', 'restoration']) ||
    /image|vision|ocr|segment|detect|depth|pose|clipseg|sam|yolo/.test(searchable)
  ) return 'Vision & image'

  if (
    hasAny(tags, ['llm', 'text', 'retrieval', 'encoder', 'embedding', 'classification', 'reranker']) ||
    /text|language|embed|encoder|rerank|colbert|llm/.test(searchable)
  ) return 'Language'

  return 'Other'
}

function modelActionLabel(
  adapter: ModelAdapter,
  isLoaded: boolean,
  stored: StoredModelSummary | undefined,
  isLoading: boolean,
): string {
  if (isLoading) return 'Downloading…'
  if (adapter.disabled) return 'Unavailable'
  if (adapter.isPipeline) return 'Open'
  if (isLoaded) return 'Unload'
  if (stored) return 'Load'
  return 'Download'
}

export default function ModelList({
  adapters,
  onSelect,
  onLoad,
  onUnload,
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
  const [activeFilter, setActiveFilter] = useState<ModelFilter>('All')

  const families = useMemo(
    () => new Map(adapters.map((adapter) => [adapter.modelId, familyFor(adapter)])),
    [adapters],
  )

  const counts = useMemo(() => {
    const next = new Map<ModelFilter, number>()
    next.set('All', adapters.length)
    next.set('Downloaded', adapters.filter((adapter) => storedModels.has(adapter.modelId)).length)
    for (const family of FAMILY_ORDER) {
      next.set(family, adapters.filter((adapter) => families.get(adapter.modelId) === family).length)
    }
    return next
  }, [adapters, families, storedModels])

  const visibleItems = useMemo(() => {
    const source = searching || activeFilter === 'All'
      ? adapters
      : activeFilter === 'Downloaded'
        ? adapters.filter((adapter) => storedModels.has(adapter.modelId))
        : adapters.filter((adapter) => families.get(adapter.modelId) === activeFilter)

    return [...source].sort((a, b) =>
      a.metadata.name < b.metadata.name ? -1 : a.metadata.name > b.metadata.name ? 1 : 0
    )
  }, [activeFilter, adapters, families, searching, storedModels])

  return (
    <div>
      <div className="mb-3 flex items-center gap-1.5 overflow-x-auto pb-1">
        {FILTER_ORDER.map((filter) => {
          const count = counts.get(filter) ?? 0
          if (filter === 'Downloaded' && count === 0) return null

          const active = !searching && activeFilter === filter
          const family = FAMILY_ORDER.includes(filter as ModelFamily) ? filter as ModelFamily : null

          return (
            <button
              key={filter}
              type="button"
              onClick={() => setActiveFilter(filter)}
              aria-label={FILTER_LABELS[filter]}
              title={FILTER_LABELS[filter]}
              className={`flex shrink-0 items-center gap-1.5 rounded-full border px-2.5 py-1.5 text-xs font-medium transition-all ${
                active
                  ? family
                    ? `border-current bg-surface-container-high ${FILTER_COLOR_CLASS[family]}`
                    : 'border-outline bg-surface-container-high text-on-surface'
                  : family
                    ? `border-transparent bg-surface-container-low/60 ${FILTER_COLOR_CLASS[family]} hover:bg-surface-container`
                    : 'border-transparent bg-surface-container-low/60 text-on-surface-variant hover:bg-surface-container hover:text-on-surface'
              }`}
            >
              <FilterGlyph filter={filter} className="h-4 w-4" />
              <span className="text-[10px] opacity-60">{count}</span>
            </button>
          )
        })}
      </div>

      {searching && (
        <p className="mb-2 text-xs text-on-surface-muted">
          {visibleItems.length} {visibleItems.length === 1 ? 'match' : 'matches'} across all types
        </p>
      )}

      <div className="model-grid">
        {visibleItems.map((adapter) => {
          const family = families.get(adapter.modelId) ?? 'Other'
          const isLoading = loadingModelId === adapter.modelId
          const isSelected = selectedModelId === adapter.modelId
          const isLoaded = loadedModelId === adapter.modelId
          const stored = storedModels.get(adapter.modelId)
          const isUnavailable = !!adapter.disabled
          const verificationStatus = adapter.verification?.status ?? 'registered'
          const verificationLabel = VERIFICATION_LABELS[verificationStatus]
          const actionLabel = modelActionLabel(adapter, isLoaded, stored, isLoading)

          const handleAction = () => {
            if (isUnavailable || isLoading) return
            if (adapter.isPipeline) {
              onOpenPipeline?.(adapter.modelId)
              return
            }
            if (isLoaded) {
              onUnload(adapter)
              return
            }
            onLoad(adapter)
          }

          return (
            <article
              key={adapter.modelId}
              className={`model-card ${FAMILY_CLASS[family]} ${isSelected ? 'model-card--selected' : ''}`}
            >
              <div className="model-card__rail" />

              <button
                type="button"
                onClick={() => !isUnavailable && onSelect(adapter)}
                disabled={isUnavailable && !isSelected}
                className="flex w-full flex-1 flex-col items-start px-3 pt-3 text-left disabled:opacity-55"
              >
                <span className="model-card__icon mb-2 inline-flex h-8 w-8 items-center justify-center rounded-xl">
                  <FamilyGlyph family={family} className="h-5 w-5" />
                </span>
                <p className="break-words text-sm font-semibold leading-snug text-on-surface">
                  {adapter.metadata.name}
                </p>
              </button>

              {isSelected && (
                <div className="px-3 pb-2 pt-2">
                  <p className="max-w-2xl text-xs leading-relaxed text-on-surface-variant">
                    {adapter.metadata.description}
                  </p>

                  <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[10px]">
                    {adapter.metadata.tags.slice(0, 5).map((tag) => (
                      <span key={tag} className="rounded-md bg-surface-container-high px-1.5 py-0.5 text-on-surface-variant">
                        {tag}
                      </span>
                    ))}
                    {stored && (
                      <span className="rounded-md bg-secondary-container/70 px-1.5 py-0.5 text-on-secondary-container">
                        {formatBytes(stored.bytes)} stored
                      </span>
                    )}
                  </div>

                  <details className="mt-2 rounded-lg bg-surface/45 px-2.5 py-2 text-[10px] text-on-surface-variant">
                    <summary className="cursor-pointer select-none font-medium text-on-surface-muted">
                      Details
                    </summary>
                    <div className="mt-2 space-y-2">
                      <div>
                        <p className="uppercase tracking-wide text-on-surface-muted">Model id</p>
                        <p className="mt-0.5 break-all font-mono">{adapter.modelId}</p>
                      </div>
                      <div>
                        <p className="uppercase tracking-wide text-on-surface-muted">Asset</p>
                        <p className="mt-0.5 break-all font-mono">{adapter.metadata.modelPath}</p>
                      </div>
                      {(verificationLabel || adapter.verification?.backends?.length) && (
                        <div className="flex flex-wrap gap-1.5 pt-1">
                          {verificationLabel && (
                            <span className="rounded-md bg-tertiary-container/70 px-1.5 py-0.5 text-on-tertiary-container">
                              {verificationLabel}
                            </span>
                          )}
                          {adapter.verification?.backends?.map((backend) => (
                            <span key={backend} className="rounded-md bg-surface-container-highest px-1.5 py-0.5">
                              {backend.toUpperCase()}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </details>
                </div>
              )}

              {isLoading && downloadProgress && (
                <div className="px-3 pb-2">
                  <div className="h-1.5 overflow-hidden rounded-full bg-outline-variant">
                    <div
                      className="h-full rounded-full bg-primary transition-all duration-300"
                      style={{ width: `${progressPercent(downloadProgress)}%` }}
                    />
                  </div>
                  <p className="mt-1 text-[10px] text-on-surface-variant">
                    {downloadProgress.totalBytes
                      ? `${progressPercent(downloadProgress)}% · ${formatBytes(downloadProgress.loadedBytes)} / ${formatBytes(downloadProgress.totalBytes)}`
                      : formatBytes(downloadProgress.loadedBytes)}
                  </p>
                </div>
              )}

              <div className="mt-auto flex items-center justify-end px-2.5 pb-2 pt-1.5">
                {isLoaded && (
                  <span className="mr-auto text-[10px] font-medium text-tertiary">● Loaded</span>
                )}
                {stored && !isLoaded && (
                  <span className="mr-auto text-[10px] font-medium text-secondary">● Stored</span>
                )}
                {isUnavailable && (
                  <span className="mr-auto text-[10px] text-on-surface-muted">Unavailable</span>
                )}
                <button
                  type="button"
                  onClick={handleAction}
                  disabled={isUnavailable || isLoading || (!!disabled && !isLoaded)}
                  className="model-card__action shrink-0 rounded-lg px-2.5 py-1.5 text-[10px] font-semibold transition-all disabled:cursor-not-allowed disabled:opacity-45"
                >
                  {actionLabel}
                </button>
              </div>

              {isSelected && stored && !adapter.isPipeline && (
                <button
                  type="button"
                  onClick={() => onRemoveStored(adapter.modelId)}
                  disabled={storageBusy || disabled}
                  className="mx-3 mb-2 text-left text-[10px] font-medium text-error transition-opacity hover:opacity-80 disabled:opacity-40"
                >
                  Remove downloaded files
                </button>
              )}
            </article>
          )
        })}
      </div>
    </div>
  )
}
