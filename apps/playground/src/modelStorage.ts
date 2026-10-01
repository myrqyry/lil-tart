import {
  createHttpAssetResolver,
  createManifestVerifyingAssetResolver,
  type AssetIntegrityVerificationOptions,
  type AssetRequestOptions,
  type AssetResolver,
  type ModelAsset,
  type ModelManifest,
} from '@litert-playground/inference-core'

const CACHE_NAME = 'lil-tart-model-library-v1'
const CACHE_PATH = '/__lil_tart_model_cache__/'
const assetOwners = new Map<string, string>()

export interface StoredModelInfo {
  modelId: string
  bytes: number
  assets: number
  unverified?: boolean
}

export interface ModelLibraryIntegrityOptions {
  manifest: ModelManifest
  verificationOptions?: AssetIntegrityVerificationOptions
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

// Membership follows the resolver's exact URL semantics, including absolute,
// root-relative and slash-less paths. Old entries may lack the original path;
// recover it from registered assets when possible, otherwise preserve the bytes
// as unverified rather than guessing that they are safe to delete.
function belongsToBase(response: Response | undefined, modelId: string, base: string): boolean | null {
  const assetUrl = response?.headers.get('x-lil-tart-asset-url')
  if (!assetUrl) return null
  try {
    const target = normalizeBase(base)
    const path = response?.headers.get('x-lil-tart-asset-path')
    if (path) return resolvedAssetUrl(path, target) === assetUrl

    const recordedBase = response?.headers.get('x-lil-tart-base')
    const candidates = [...assetOwners].filter(([, owner]) => owner === modelId)
    // A currently registered path resolving to this key proves reachability even
    // when the cache predates path/base metadata.
    if (candidates.some(([candidate]) => resolvedAssetUrl(candidate, target) === assetUrl)) return true
    if (recordedBase) {
      if (normalizeBase(recordedBase) === target) return true
      if (candidates.some(([candidate]) => resolvedAssetUrl(candidate, recordedBase) === assetUrl)) return false
    }
    return null
  } catch {
    return null
  }
}

export function registerModelAssets(modelId: string, paths: readonly string[]): void {
  for (const path of paths) {
    if (!path) continue
    assetOwners.set(path, modelId)
  }
}

export function createModelLibraryAssetResolver(
  base: string,
  integrity?: ModelLibraryIntegrityOptions,
): AssetResolver {
  const inner = createHttpAssetResolver(base)
  const verifiedSource = integrity
    ? createManifestVerifyingAssetResolver(
        integrity.manifest,
        inner,
        integrity.verificationOptions,
      )
    : inner

  async function verifyCached(asset: ModelAsset, value: ArrayBuffer): Promise<ArrayBuffer> {
    if (!integrity) return value
    return createManifestVerifyingAssetResolver(
      integrity.manifest,
      { resolve: async () => value },
      integrity.verificationOptions,
    ).resolve(asset)
  }

  async function resolve(asset: ModelAsset, options?: AssetRequestOptions): Promise<ArrayBuffer> {
    const owner = assetOwners.get(asset.path) ?? assetOwners.get(asset.id)
    const storage = cacheStorage()
    if (!owner || !storage) return verifiedSource.resolve(asset, options)

    const assetUrl = resolvedAssetUrl(asset.path, base)
    const key = cacheKey(owner, assetUrl)

    let cache: Cache | null = null
    try {
      cache = await storage.open(CACHE_NAME)
      const cached = await cache.match(key)
      if (cached) {
        try {
          return await verifyCached(asset, await cached.arrayBuffer())
        } catch {
          // A bad persisted entry must not poison every retry. Remove it and
          // fall through to a fresh verified fetch.
          await cache.delete(key)
        }
      }
    } catch {
      cache = null
    }

    // Integrity-aware callers verify fresh bytes before they are persisted.
    const fresh = await verifiedSource.resolve(asset, options)
    if (cache) {
      try {
        await cache.put(
          key,
          new Response(fresh, {
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

// Lists unreachable and unverified legacy entries without deleting them. The
// free-text ModelRunner leaves them for deliberate removal. The fixed-base LFM
// panel also calls pruneSupersededModelCaches after loading; that removes only
// proven orphans for its model. Unverified entries are retained in both paths.
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
      const matches = belongsToBase(response, modelId, base)
      if (keepMatching ? matches !== true : matches === true) continue
      const bytes = Number(response?.headers.get('x-lil-tart-bytes') ?? 0)
      const current = models.get(modelId) ?? { modelId, bytes: 0, assets: 0 }
      current.bytes += Number.isFinite(bytes) ? bytes : 0
      current.assets += 1
      if (matches === null) current.unverified = true
      models.set(modelId, current)
    }

    return [...models.values()].sort((a, b) => a.modelId < b.modelId ? -1 : a.modelId > b.modelId ? 1 : 0)
  } catch {
    return []
  }
}

// Reclaims proven orphans for one model relative to the supplied base. This is
// currently called only by the fixed-base LFM panel. Another panel using the same
// model id at a different base would share this deletion scope. Unknown legacy
// entries are never pruned. Returns how many entries were removed.
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
      if (belongsToBase(response, modelId, base) !== false) continue
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
      const matches = belongsToBase(response, modelId, base)
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
