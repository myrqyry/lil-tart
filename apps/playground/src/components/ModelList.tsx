import type { ModelAdapter, VerificationStatus } from '../adapters/types'

interface ModelListProps {
  adapters: ModelAdapter[]
  onSelect: (adapter: ModelAdapter) => void
  disabled?: boolean
  loadingModelId?: string | null
  downloadProgress?: { loadedBytes: number; totalBytes?: number } | null
  selectedModelId?: string | null
  isModelLoaded?: boolean
}

const VERIFICATION_LABELS: Record<VerificationStatus, string> = {
  registered: 'Registered',
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

export default function ModelList({ adapters, onSelect, disabled, loadingModelId, downloadProgress, selectedModelId, isModelLoaded }: ModelListProps) {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {adapters.map(a => {
        const isLoading = loadingModelId === a.modelId && !!disabled
        const isSelectedLoaded = selectedModelId === a.modelId && isModelLoaded
        const isUnavailable = !!a.disabled
        const verificationStatus = a.verification?.status ?? 'registered'
        return (
          <button
            key={a.modelId}
            onClick={() => !isLoading && !isUnavailable && onSelect(a)}
            disabled={isUnavailable || (!!disabled && !isLoading)}
            title={isUnavailable ? 'No browser-fetchable .tflite yet — needs locating' : undefined}
            className="rounded-xl border border-outline/60 bg-surface-container p-4 text-left transition-all hover:border-primary/50 hover:shadow-md disabled:opacity-60"
          >
            <div className="flex items-start justify-between gap-2">
              <div className="flex-1">
                <p className="text-sm font-semibold text-on-surface">{a.metadata.name}</p>
                <p className="mt-1 line-clamp-2 text-xs text-on-surface-variant">{a.metadata.description}</p>
                <div className="mt-2 flex flex-wrap items-center gap-1">
                  <span className="rounded-full bg-surface-container-high px-2 py-0.5 text-[10px] font-medium text-on-surface-variant">
                    Evidence: {VERIFICATION_LABELS[verificationStatus]}
                  </span>
                  {a.verification?.backends?.map(backend => (
                    <span key={backend} className="rounded-full bg-secondary-container px-2 py-0.5 text-[10px] font-medium text-on-secondary-container">
                      {backend.toUpperCase()}
                    </span>
                  ))}
                </div>
                {a.metadata.tags.length > 0 && (
                  <p className="mt-2 text-[10px] font-medium uppercase tracking-wide text-on-surface-variant">Task types</p>
                )}
                <div className="mt-1 flex flex-wrap gap-1">
                  {a.metadata.tags.map(t => (
                    <span key={t} className="rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary">{t}</span>
                  ))}
                </div>
              </div>
              <div className="shrink-0">
                {isUnavailable ? (
                  <span className="rounded-full bg-outline/30 px-2 py-0.5 text-[10px] font-medium text-on-surface-variant">Needs locating</span>
                ) : isSelectedLoaded ? (
                  <span className="text-[10px] font-medium text-green-600">Ready</span>
                ) : isLoading && downloadProgress ? (
                  <span className="text-[10px] font-medium text-primary">Downloading…</span>
                ) : (
                  <span className="text-[10px] font-medium text-on-surface-variant">Download</span>
                )}
              </div>
            </div>
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
  )
}
