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
}

const VERIFICATION_LABELS: Record<VerificationStatus, string> = {
  registered: 'Registered',
  'compile-verified': 'Compile verified',
  'inference-verified': 'Inference verified',
  'output-verified': 'Output verified',
  'manually-verified': 'Manually verified',
}

const GROUP_ORDER = [
  'On this device',
  'Pipelines',
  'Language',
  'Speech & audio',
  'Vision & image',
  'Other',
] as const

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

function categoryFor(adapter: ModelAdapter, stored: boolean): typeof GROUP_ORDER[number] {
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

export default function ModelList({
  adapters,
  onSelect,
  disabled,
  loadingModelId,
  downloadProgress,
  selectedModelId,
  loadedModelId,
  storedModelIds = new Set(),
}: ModelListProps) {
  const grouped = new Map<typeof GROUP_ORDER[number], ModelAdapter[]>()

  for (const adapter of adapters) {
    const group = categoryFor(adapter, storedModelIds.has(adapter.modelId))
    const values = grouped.get(group) ?? []
    values.push(adapter)
    grouped.set(group, values)
  }

  for (const values of grouped.values()) {
    values.sort((a, b) => a.metadata.name < b.metadata.name ? -1 : a.metadata.name > b.metadata.name ? 1 : 0)
  }

  return (
    <div className="space-y-6">
      {GROUP_ORDER.map((group) => {
        const items = grouped.get(group)
        if (!items?.length) return null

        return (
          <section key={group}>
            <div className="mb-2 flex items-center gap-2">
              <h2 className="text-xs font-semibold uppercase tracking-[0.16em] text-on-surface-variant">
                {group}
              </h2>
              <span className="rounded-full bg-surface-container-high px-2 py-0.5 text-[10px] text-on-surface-variant">
                {items.length}
              </span>
            </div>

            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {items.map((adapter) => {
                const isLoading = loadingModelId === adapter.modelId && !!disabled
                const isSelected = selectedModelId === adapter.modelId
                const isLoaded = loadedModelId === adapter.modelId
                const isStored = storedModelIds.has(adapter.modelId)
                const isUnavailable = !!adapter.disabled
                const verificationStatus = adapter.verification?.status ?? 'registered'

                return (
                  <button
                    key={adapter.modelId}
                    type="button"
                    onClick={() => !isUnavailable && onSelect(adapter)}
                    disabled={isUnavailable || !!disabled}
                    title={isUnavailable ? 'No browser-fetchable .tflite yet — needs locating' : undefined}
                    className={`rounded-xl border p-4 text-left transition-all disabled:opacity-60 ${
                      isSelected
                        ? 'border-primary bg-primary/5 shadow-md'
                        : 'border-outline/60 bg-surface-container hover:border-primary/50 hover:shadow-md'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-semibold text-on-surface">{adapter.metadata.name}</p>
                        <p className="mt-1 line-clamp-2 text-xs text-on-surface-variant">{adapter.metadata.description}</p>
                      </div>

                      <div className="shrink-0">
                        {isUnavailable ? (
                          <span className="rounded-full bg-outline/30 px-2 py-0.5 text-[10px] font-medium text-on-surface-variant">
                            Needs locating
                          </span>
                        ) : isLoading ? (
                          <span className="text-[10px] font-medium text-primary">Downloading…</span>
                        ) : isLoaded ? (
                          <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary">
                            Loaded
                          </span>
                        ) : isStored ? (
                          <span className="rounded-full bg-secondary-container px-2 py-0.5 text-[10px] font-medium text-on-secondary-container">
                            Stored
                          </span>
                        ) : adapter.isPipeline ? (
                          <span className="rounded-full bg-surface-container-high px-2 py-0.5 text-[10px] font-medium text-on-surface-variant">
                            Pipeline
                          </span>
                        ) : (
                          <span className="text-[10px] font-medium text-on-surface-variant">Available</span>
                        )}
                      </div>
                    </div>

                    <div className="mt-3 flex flex-wrap items-center gap-1">
                      <span className="rounded-full bg-surface-container-high px-2 py-0.5 text-[10px] font-medium text-on-surface-variant">
                        {VERIFICATION_LABELS[verificationStatus]}
                      </span>
                      {adapter.verification?.backends?.map((backend) => (
                        <span key={backend} className="rounded-full bg-secondary-container px-2 py-0.5 text-[10px] font-medium text-on-secondary-container">
                          {backend.toUpperCase()}
                        </span>
                      ))}
                    </div>

                    {adapter.metadata.tags.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1">
                        {adapter.metadata.tags.slice(0, 4).map((tag) => (
                          <span key={tag} className="rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary">
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}

                    {isLoading && downloadProgress && (
                      <div className="mt-3">
                        <div className="h-1.5 w-full overflow-hidden rounded-full bg-outline/30">
                          <div
                            className="h-full rounded-full bg-primary transition-all duration-300"
                            style={{ width: `${progressPercent(downloadProgress)}%` }}
                          />
                        </div>
                        <p className="mt-1 text-[10px] text-on-surface-variant">
                          {formatBytes(downloadProgress.loadedBytes)}
                          {downloadProgress.totalBytes ? ` / ${formatBytes(downloadProgress.totalBytes)}` : ''}
                          {downloadProgress.totalBytes ? ` (${progressPercent(downloadProgress)}%)` : ''}
                        </p>
                      </div>
                    )}
                  </button>
                )
              })}
            </div>
          </section>
        )
      })}
    </div>
  )
}
