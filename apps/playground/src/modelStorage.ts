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

function normalizeBase(base: string): string {
  return new URL(base, pageBase()).href
}

// Cache keys embed the resolved absolute URL, so changing a resolver's base orphans
// everything written under the previous base. Entries written under another base can
// never be read again, so they must not be counted as stored.
//
// Membership is reachability: would the current base resolve this entry's asset path to
// the URL the entry was stored under? Comparing base strings is not equivalent, in both
// directions. A nested base is too loose: https://host/ would claim an entry cached at
// https://host/v1/model.tflite that the resolver will now request as
// https://host/model.tflite. And a base whose last segment is a file rather than a
// directory is too strict: https://host/ and https://host/v1 both resolve the path
// "v1/model.tflite" to https://host/v1/model.tflite, so a plain string comparison would
// call a still-reachable entry orphaned and offer to delete live bytes.
// Entries written before the path was recorded fall back to base equality, and before
// that to a structural prefix comparison.
function belongsToBase(
  assetUrl: string | null | undefined,
  base: string,
  recordedBase?: string | null,
  recordedPath?: string | null,
): boolean {
  if (!assetUrl) return false
  if (recordedPath) {
    try {
      return new URL(recordedPath, normalizeBase(base)).href === assetUrl
    } catch {
      return false
    }
  }
  if (recordedBase) return recordedBase === normalizeBase(base)
  try {
    const target = new URL(base, pageBase())
    const actual = new URL(assetUrl, pageBase())
    if (actual.origin !== target.origin) return false
    if (actual.pathname === target.pathname) return true
    const prefix = target.pathname.endsWith('/') ? target.pathname : `${target.pathname}/`
    return actual.pathname.startsWith(prefix)
  } catch {
    return false
  }
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
              'x-lil-tart-base': normalizeBase(base),
              'x-lil-tart-asset-path': asset.path,
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

export async function listStoredModels(base: string): Promise<StoredModelInfo[]> {
  return collectStoredModels(base, true)
}

// Entries written under a base other than the one in use. Listing them does not
// reclaim them: a caller that switches the base back can recover what is here.
// The exception is pruneSupersededModelCaches, which deletes the same set of
// entries, but only where the base is a compile-time constant and therefore can
// never be switched back to.
export async function listOrphanedModels(base: string): Promise<StoredModelInfo[]> {
  return collectStoredModels(base, false)
}

async function collectStoredModels(base: string, keepMatching: boolean): Promise<StoredModelInfo[]> {
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
      const matches = belongsToBase(
        response?.headers.get('x-lil-tart-asset-url'),
        base,
        response?.headers.get('x-lil-tart-base'),
        response?.headers.get('x-lil-tart-asset-path'),
      )
      if (matches !== keepMatching) continue
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

// Reclaims entries for one model that were written under a superseded base.
// Only called with the base the caller is actively using, so it cannot touch
// another panel's cache. Returns how many entries were removed.
export async function pruneSupersededModelCaches(modelId: string, base: string): Promise<number> {
  const storage = cacheStorage()
  if (!storage) return 0

  try {
    const cache = await storage.open(CACHE_NAME)
    const requests = await cache.keys()
    let removed = 0
    for (const request of requests) {
      if (modelIdFromRequest(request) !== modelId) continue
      const response = await cache.match(request)
      if (belongsToBase(response?.headers.get('x-lil-tart-asset-url'), base)) continue
      if (await cache.delete(request)) removed += 1
    }
    return removed
  } catch {
    // Storage may be unavailable or blocked.
    return 0
  }
}

// Base-scoped removal. A model id can hold entries under more than one base, so
// every delete must say which ones it means: otherwise a "reclaim the previous
// base" action also destroys the live copy.
export async function removeStoredModel(modelId: string, base: string): Promise<void> {
  await removeMatching(modelId, base, true)
}

export async function removeOrphanedModel(modelId: string, base: string): Promise<void> {
  await removeMatching(modelId, base, false)
}

async function removeMatching(modelId: string, base: string, keepMatching: boolean): Promise<void> {
  const storage = cacheStorage()
  if (!storage) return

  try {
    const cache = await storage.open(CACHE_NAME)
    const requests = await cache.keys()
    for (const request of requests) {
      if (modelIdFromRequest(request) !== modelId) continue
      const response = await cache.match(request)
      const matches = belongsToBase(
        response?.headers.get('x-lil-tart-asset-url'),
        base,
        response?.headers.get('x-lil-tart-base'),
        response?.headers.get('x-lil-tart-asset-path'),
      )
      if (matches !== keepMatching) continue
      await cache.delete(request)
    }
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
