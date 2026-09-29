import { useMemo, useState } from 'react'
import {
  ArrowRightUpRegular,
  Camera2Regular,
  Delete2Regular,
  Download2Regular,
  Eye2Regular,
  Grid2Regular,
  Loading3Regular,
  AudioTapeRegular,
  HeadphoneRegular,
  Message1Regular,
  MicRegular,
  Music2Regular,
  PlayRegular,
  PowerRegular,
  RouteRegular,
  SoundLineRegular,
  SpeakerRegular,
  SubtitleRegular,
} from '@mingcute/react/core-regular'
import type { ModelAdapter } from '../adapters/types'

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

function producesImage(adapter: ModelAdapter): boolean {
  const tags = new Set(adapter.metadata.tags.map((tag) => tag.toLowerCase()))
  const searchable = `${adapter.metadata.name} ${adapter.metadata.description}`.toLowerCase()

  return (
    hasAny(tags, [
      'generative',
      'generation',
      'image-generation',
      'creative',
      'restoration',
      'inpainting',
      'denoising',
      'super-resolution',
      'style-transfer',
      'stylization',
    ]) ||
    /generat|restor|inpaint|denois|super[- ]?resolution|style transfer|styliz|esrgan|gfpgan|nafnet|swinir/.test(searchable)
  )
}

function AudioGlyph({ adapter, className = '' }: { adapter?: ModelAdapter; className?: string }) {
  if (!adapter) return <HeadphoneRegular className={className} />

  const tags = new Set(adapter.metadata.tags.map((tag) => tag.toLowerCase()))
  const searchable = `${adapter.metadata.name} ${adapter.metadata.description}`.toLowerCase()

  if (tags.has('music') || /music transcription|music generation|musiccoca/.test(searchable)) {
    return <Music2Regular className={className} />
  }
  if (tags.has('tts') || /text[- ]?to[- ]?speech|speech synthesis|synthesized audio/.test(searchable)) {
    return <SpeakerRegular className={className} />
  }
  if (tags.has('codec') || /codec|mimi/.test(searchable)) {
    return <AudioTapeRegular className={className} />
  }
  if (tags.has('asr') || /speech recognition|speech[- ]?to[- ]?text|whisper|moonshine|parakeet|granite speech/.test(searchable)) {
    return <SubtitleRegular className={className} />
  }
  if (tags.has('keyword-spotting') || /keyword spotting|voice activity/.test(searchable)) {
    return <MicRegular className={className} />
  }
  if (tags.has('pitch') || tags.has('tuner') || tags.has('tagging') || /pitch|audioset|audio tagging|audio classification/.test(searchable)) {
    return <SoundLineRegular className={className} />
  }
  return <HeadphoneRegular className={className} />
}

function FamilyGlyph({
  adapter,
  family,
  className = '',
}: {
  adapter?: ModelAdapter
  family: ModelFamily
  className?: string
}) {
  if (family === 'Pipelines') return <RouteRegular className={className} />
  if (family === 'Language') return <Message1Regular className={className} />
  if (family === 'Speech & audio') return <AudioGlyph adapter={adapter} className={className} />
  if (family === 'Vision & image') {
    return adapter && producesImage(adapter)
      ? <Camera2Regular className={className} />
      : <Eye2Regular className={className} />
  }
  return <Grid2Regular className={className} />
}

function FilterGlyph({ filter, className = '' }: { filter: ModelFilter; className?: string }) {
  if (filter === 'All') return <Grid2Regular className={className} />
  if (filter === 'Downloaded') return <Download2Regular className={className} />
  return <FamilyGlyph family={filter as ModelFamily} className={className} />
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

    const sourceIndex = new Map(source.map((adapter, index) => [adapter.modelId, index]))
    const priority = (adapter: ModelAdapter) => {
      if (adapter.modelId === loadedModelId) return 0
      if (storedModels.has(adapter.modelId)) return 1
      if (adapter.isPipeline) return 2
      if (!adapter.disabled) return 3
      return 4
    }

    return [...source].sort((a, b) =>
      priority(a) - priority(b) ||
      (sourceIndex.get(a.modelId) ?? 0) - (sourceIndex.get(b.modelId) ?? 0)
    )
  }, [activeFilter, adapters, families, loadedModelId, searching, storedModels])

  return (
    <div>
      <div className="mb-2 flex items-center gap-1 overflow-x-auto pb-0.5">
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
              className={`filter-chip flex shrink-0 items-center gap-1.5 px-2.5 py-1.5 text-[13px] font-medium ${
                active
                  ? family
                    ? `border-current bg-surface-container-high ${FILTER_COLOR_CLASS[family]}`
                    : 'border-outline bg-surface-container-high text-on-surface'
                  : family
                    ? `border-transparent bg-surface-container-low/60 ${FILTER_COLOR_CLASS[family]} hover:bg-surface-container`
                    : 'border-transparent bg-surface-container-low/60 text-on-surface-variant hover:bg-surface-container hover:text-on-surface'
              }`}
            >
              <FilterGlyph filter={filter} className="h-[17px] w-[17px]" />
              <span className="text-xs opacity-60">{count}</span>
            </button>
          )
        })}
      </div>

      {searching && (
        <p className="mb-2 text-[13px] text-on-surface-muted">
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
          const verificationStatus = adapter.verification?.status?.replace(/-/g, ' ')

          const handleCornerAction = () => {
            if (isUnavailable || isLoading) return
            if (adapter.isPipeline) {
              onOpenPipeline?.(adapter.modelId)
              return
            }
            if (!stored) {
              onLoad(adapter)
              return
            }
            if (isLoaded) onUnload(adapter)
            else onLoad(adapter)
          }

          const cornerActionLabel = adapter.isPipeline
            ? `Open ${adapter.metadata.name}`
            : isLoading
              ? `Downloading ${adapter.metadata.name}`
              : !stored
                ? `Download ${adapter.metadata.name}`
                : isLoaded
                  ? `Unload ${adapter.metadata.name}`
                  : `Load ${adapter.metadata.name}`

          const status = isUnavailable
            ? 'Unavailable'
            : isLoaded
              ? 'Loaded'
              : stored
                ? `${formatBytes(stored.bytes)} local`
                : isSelected
                  ? 'Selected'
                  : null

          return (
            <article
              key={adapter.modelId}
              className={`model-card ${FAMILY_CLASS[family]} ${isSelected ? 'model-card--selected' : ''}`}
            >
              <div className="model-card__rail" />

              <span
                className="model-card__notch model-card__notch--type"
                title={FILTER_LABELS[family]}
                aria-label={FILTER_LABELS[family]}
              >
                <FamilyGlyph adapter={adapter} family={family} className="h-3.5 w-3.5" />
              </span>

              <button
                type="button"
                onClick={handleCornerAction}
                disabled={isUnavailable || isLoading || (!!disabled && !adapter.isPipeline)}
                aria-label={cornerActionLabel}
                title={cornerActionLabel}
                className="model-card__notch model-card__notch--action"
              >
                {isLoading ? (
                  <Loading3Regular className="h-3.5 w-3.5 animate-spin" />
                ) : adapter.isPipeline ? (
                  <ArrowRightUpRegular className="h-3.5 w-3.5" />
                ) : !stored ? (
                  <Download2Regular className="h-3.5 w-3.5" />
                ) : isLoaded ? (
                  <PowerRegular className="h-3.5 w-3.5" />
                ) : (
                  <PlayRegular className="h-3.5 w-3.5" />
                )}
              </button>

              <button
                type="button"
                onClick={() => onSelect(adapter)}
                aria-label={`${adapter.metadata.name}. ${isUnavailable ? 'Unavailable for execution; ' : ''}${verificationStatus ? 'Verification: ' + verificationStatus + '. ' : ''}Select to inspect details.`}
                title={`${isUnavailable ? 'Unavailable for execution · ' : ''}${verificationStatus ? 'Verification: ' + verificationStatus : 'Inspect model details'}`}
                className={`block w-full flex-1 px-2.5 pt-11 text-left ${isLoading ? 'pb-10' : 'pb-2.5'}`}
              >
                <p className="model-card__title break-words text-[15px] font-semibold leading-[1.2] text-on-surface">
                  {adapter.metadata.name}
                </p>
                <p className="model-card__description mt-1 text-[13px] leading-[1.28] text-on-surface-variant">
                  {adapter.metadata.description}
                </p>
                <div className="mt-2 flex min-h-4 items-center gap-1.5 pr-7 text-[11px]">
                  {status && (
                    <span className={isLoaded ? 'font-medium text-tertiary' : isUnavailable ? 'text-on-surface-muted' : 'text-on-surface-variant'}>
                      {status}
                    </span>
                  )}
                  {verificationStatus && (
                    <span className="truncate text-[10px] text-on-surface-muted">· {verificationStatus}</span>
                  )}
                </div>
              </button>

              {stored && !isLoaded && !isLoading && (
                <button
                  type="button"
                  onClick={() => onRemoveStored(adapter.modelId)}
                  disabled={storageBusy || disabled}
                  aria-label={`Remove downloaded files for ${adapter.metadata.name}`}
                  title="Remove downloaded files"
                  className="absolute bottom-2 right-2 z-10 inline-flex h-6 w-6 items-center justify-center rounded-full text-on-surface-muted transition-transform hover:scale-110 hover:bg-error-container/25 hover:text-error active:scale-90 disabled:opacity-40"
                >
                  <Delete2Regular className="h-3.5 w-3.5" />
                </button>
              )}

              {isLoading && downloadProgress && (
                <div className="absolute inset-x-2.5 bottom-1.5">
                  <div className="mb-1 flex items-center justify-between gap-2 text-[10px] text-on-surface-muted">
                    <span>{progressPercent(downloadProgress)}%</span>
                    <span>
                      {formatBytes(downloadProgress.loadedBytes)}
                      {downloadProgress.totalBytes ? ' / ' + formatBytes(downloadProgress.totalBytes) : ''}
                    </span>
                  </div>
                  <div className="h-1 overflow-hidden rounded-full bg-outline-variant">
                    <div
                      className="h-full rounded-full bg-primary transition-all duration-300"
                      style={{ width: `${progressPercent(downloadProgress)}%` }}
                    />
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
