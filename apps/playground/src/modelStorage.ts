import {
  createHttpAssetResolver,
  type AssetRequestOptions,
  type AssetResolver,
  type ModelAsset,
} from '@litert-playground/inference-core'

const CACHE_NAME = 'lil-tart-model-library-v1'
const CACHE_PATH = '/__lil_tart_model_cache__/'
const assetOwners = new Map<string, string>()

export interface StoredModelInfo {
  modelId: string
  bytes: number
  assets: number
}

function pageBase(): string {
  if (typeof document !== 'undefined' && document.baseURI) return document.baseURI
  if (typeof location !== 'undefined') return location.href
  return 'http://localhost/'
}

function cacheStorage(): CacheStorage | null {
  return typeof caches === 'undefined' ? null : caches
}

function cacheKey(modelId: string, assetUrl: string): Request {
  const origin = new URL(pageBase()).origin
  const url = new URL(`${CACHE_PATH}${encodeURIComponent(modelId)}/${encodeURIComponent(assetUrl)}`, origin)
  return new Request(url.href)
}

function modelIdFromRequest(request: Request): string | null {
  const pathname = new URL(request.url).pathname
  if (!pathname.startsWith(CACHE_PATH)) return null
  const encoded = pathname.slice(CACHE_PATH.length).split('/')[0]
  return encoded ? decodeURIComponent(encoded) : null
}

function resolvedAssetUrl(path: string, base: string): string {
  return new URL(path, new URL(base, pageBase())).href
}

export function registerModelAssets(modelId: string, paths: readonly string[]): void {
  for (const path of paths) {
    if (!path) continue
    assetOwners.set(path, modelId)
  }
}

export function createModelLibraryAssetResolver(base: string): AssetResolver {
  const inner = createHttpAssetResolver(base)

  async function resolve(asset: ModelAsset, options?: AssetRequestOptions): Promise<ArrayBuffer> {
    const owner = assetOwners.get(asset.path) ?? assetOwners.get(asset.id)
    const storage = cacheStorage()
    if (!owner || !storage) return inner.resolve(asset, options)

    const assetUrl = resolvedAssetUrl(asset.path, base)
    const key = cacheKey(owner, assetUrl)

    let cache: Cache | null = null
    try {
      cache = await storage.open(CACHE_NAME)
      const cached = await cache.match(key)
      if (cached) return cached.arrayBuffer()
    } catch {
      cache = null
    }

    const fresh = await inner.resolve(asset, options)
    if (cache) {
      try {
        await cache.put(
          key,
          new Response(fresh.slice(0), {
            headers: {
              'content-type': asset.mimeType ?? 'application/octet-stream',
              'x-lil-tart-bytes': String(fresh.byteLength),
              'x-lil-tart-model-id': owner,
              'x-lil-tart-asset-url': assetUrl,
            },
          }),
        )
      } catch {
        // Persistent storage is optional; the fresh model can still run.
      }
    }
    return fresh
  }

  return {
    resolve,
    stream: async (asset, options) => {
      const value = await resolve(asset, options)
      return new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(new Uint8Array(value))
          controller.close()
        },
      })
    },
  }
}

export async function listStoredModels(): Promise<StoredModelInfo[]> {
  const storage = cacheStorage()
  if (!storage) return []

  try {
    const cache = await storage.open(CACHE_NAME)
    const requests = await cache.keys()
    const models = new Map<string, StoredModelInfo>()

    for (const request of requests) {
      const modelId = modelIdFromRequest(request)
      if (!modelId) continue
      const response = await cache.match(request)
      const bytes = Number(response?.headers.get('x-lil-tart-bytes') ?? 0)
      const current = models.get(modelId) ?? { modelId, bytes: 0, assets: 0 }
      current.bytes += Number.isFinite(bytes) ? bytes : 0
      current.assets += 1
      models.set(modelId, current)
    }

    return [...models.values()].sort((a, b) => a.modelId < b.modelId ? -1 : a.modelId > b.modelId ? 1 : 0)
  } catch {
    return []
  }
}

export async function removeStoredModel(modelId: string): Promise<void> {
  const storage = cacheStorage()
  if (!storage) return

  try {
    const cache = await storage.open(CACHE_NAME)
    const requests = await cache.keys()
    await Promise.all(
      requests
        .filter((request) => modelIdFromRequest(request) === modelId)
        .map((request) => cache.delete(request)),
    )
  } catch {
    // Storage may be unavailable or blocked.
  }
}

export async function clearStoredModels(): Promise<void> {
  const storage = cacheStorage()
  if (!storage) return
  try {
    await storage.delete(CACHE_NAME)
  } catch {
    // Storage may be unavailable or blocked.
  }
}
