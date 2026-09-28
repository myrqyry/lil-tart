import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  createModelLibraryAssetResolver,
  registerModelAssets,
  listOrphanedModels,
  listStoredModels,
  pruneSupersededModelCaches,
  removeOrphanedModel,
  removeStoredModel,
} from './modelStorage'

const CACHE_PATH = '/__lil_tart_model_cache__/'
const HF = 'https://huggingface.co/'

interface FakeEntry {
  url: string
  bytes: number
  assetUrl: string
  base?: string
  path?: string
  body?: ArrayBuffer
}

// One shared store, so deletions persist across open() calls the way real
// Cache Storage behaves.
let store = new Map<string, FakeEntry>()

function cacheKeyUrl(modelId: string, assetUrl: string): string {
  return `https://app.test${CACHE_PATH}${encodeURIComponent(modelId)}/${encodeURIComponent(assetUrl)}`
}

function seed(list: FakeEntry[]): void {
  store = new Map(list.map((entry) => [entry.url, entry]))
}

function cache() {
  return {
    keys: async () => [...store.keys()].map((url) => new Request(url)),
    match: async (request: Request) => {
      const entry = store.get(request.url)
      if (!entry) return undefined
      return new Response(entry.body ?? new Uint8Array(entry.bytes), {
        headers: {
          'x-lil-tart-bytes': String(entry.bytes),
          'x-lil-tart-asset-url': entry.assetUrl,
          ...(entry.base ? { 'x-lil-tart-base': entry.base } : {}),
          ...(entry.path ? { 'x-lil-tart-asset-path': entry.path } : {}),
        },
      })
    },
    put: async (request: Request, response: Response) => {
      const body = await response.arrayBuffer()
      store.set(request.url, {
        url: request.url,
        bytes: body.byteLength,
        body,
        assetUrl: response.headers.get('x-lil-tart-asset-url')!,
        base: response.headers.get('x-lil-tart-base') ?? undefined,
        path: response.headers.get('x-lil-tart-asset-path') ?? undefined,
      })
    },
    delete: async (request: Request) => store.delete(request.url),
  }
}

beforeEach(() => {
  seed([])
  const storage = {
    open: async () => cache(),
    delete: async () => true,
  }
  ;(globalThis as { caches?: unknown }).caches = storage
  ;(globalThis as { document?: unknown }).document = { baseURI: 'https://app.test/' }
})

afterEach(() => {
  vi.unstubAllGlobals()
  delete (globalThis as { caches?: unknown }).caches
  delete (globalThis as { document?: unknown }).document
})

describe('listStoredModels', () => {
  it('counts only entries recorded against the requested base', async () => {
    const modelId = 'lfm2.5-encoder-230m'
    seed([
      {
        url: cacheKeyUrl(modelId, `${HF}litert-community/encoder/resolve/main/model.tflite`),
        bytes: 500,
        assetUrl: `${HF}litert-community/encoder/resolve/main/model.tflite`,
        path: 'litert-community/encoder/resolve/main/model.tflite',
      },
      {
        // A copy keyed on the old app origin is unreachable from the HF base.
        url: cacheKeyUrl(modelId, 'https://app.test/litert-community/encoder/resolve/main/model.tflite'),
        bytes: 500,
        assetUrl: 'https://app.test/litert-community/encoder/resolve/main/model.tflite',
        path: 'litert-community/encoder/resolve/main/model.tflite',
      },
    ])

    const stored = await listStoredModels(HF)

    // One usable model worth 500 bytes, not a phantom 1000.
    expect(stored).toEqual([{ modelId, bytes: 500, assets: 1 }])
  })

  it('reports nothing when every entry belongs to a different base', async () => {
    seed([
      {
        url: cacheKeyUrl('qwen3-tts', 'https://app.test/models/qwen3-tts/model.tflite'),
        bytes: 900,
        assetUrl: 'https://app.test/models/qwen3-tts/model.tflite',
        path: 'models/qwen3-tts/model.tflite',
      },
    ])

    expect(await listStoredModels(HF)).toEqual([])
  })

  it('does not match a base that is only a string prefix of another origin', async () => {
    seed([
      {
        url: cacheKeyUrl('model', 'https://huggingface.co.evil.test/litert-community/model.tflite'),
        bytes: 100,
        assetUrl: 'https://huggingface.co.evil.test/litert-community/model.tflite',
        path: 'litert-community/model.tflite',
      },
    ])

    expect(await listStoredModels(HF)).toEqual([])
  })

  // A base typed without a trailing slash must not claim a sibling path that merely
  // shares its leading characters. This is reachable: the model-server base is
  // free-text user input in the playground runtime panel.
  it('does not claim a sibling path when the base has no trailing slash', async () => {
    seed([
      {
        url: cacheKeyUrl('model', 'https://huggingface.co/litert-community-evil/model.tflite'),
        bytes: 100,
        assetUrl: 'https://huggingface.co/litert-community-evil/model.tflite',
        path: 'litert-community/model.tflite',
      },
    ])

    expect(await listStoredModels('https://huggingface.co/litert-community')).toEqual([])
    expect(await listOrphanedModels('https://huggingface.co/litert-community')).toEqual([
      { modelId: 'model', bytes: 100, assets: 1 },
    ])
  })

  it('resolves a manifest path against a base without a trailing slash', async () => {
    seed([
      {
        url: cacheKeyUrl('model', 'https://huggingface.co/litert-community/model.tflite'),
        bytes: 100,
        assetUrl: 'https://huggingface.co/litert-community/model.tflite',
        path: 'litert-community/model.tflite',
      },
    ])

    expect(await listStoredModels('https://huggingface.co/litert-community')).toEqual([
      { modelId: 'model', bytes: 100, assets: 1 },
    ])
  })

  // A path prefix is too loose once bases are nested. Cached at /v1/, the entry is
  // unreachable once the base is the root, because the resolver now asks for
  // /model.tflite. Counting it would report bytes as stored that get re-downloaded.
  it('orphans a nested-base entry once the base widens to its root', async () => {
    seed([
      {
        url: cacheKeyUrl('model', 'https://host/v1/model.tflite'),
        bytes: 900,
        assetUrl: 'https://host/v1/model.tflite',
        path: 'model.tflite',
        base: 'https://host/v1/',
      },
    ])

    expect(await listStoredModels('https://host/')).toEqual([])
    expect(await listOrphanedModels('https://host/')).toEqual([
      { modelId: 'model', bytes: 900, assets: 1 },
    ])
    // Still recoverable by switching the base back.
    expect(await listStoredModels('https://host/v1/')).toEqual([
      { modelId: 'model', bytes: 900, assets: 1 },
    ])
  })

  // https://host/ and https://host/v1 both resolve "v1/model.tflite" to the same absolute
  // URL, because a base whose last segment is a file is not a directory. Base-string
  // equality would call this live entry orphaned and offer to delete usable bytes.
  it('keeps an entry reachable through a base that resolves the same URL', async () => {
    const assetUrl = 'https://huggingface.co/litert-community/model.tflite'
    seed([
      {
        url: cacheKeyUrl('model', assetUrl),
        bytes: 900,
        assetUrl,
        base: 'https://huggingface.co/',
        path: 'litert-community/model.tflite',
      },
    ])

    expect(await listStoredModels('https://huggingface.co/litert-community')).toEqual([
      { modelId: 'model', bytes: 900, assets: 1 },
    ])
    expect(await listOrphanedModels('https://huggingface.co/litert-community')).toEqual([])
  })

  it('still orphans an entry the current base resolves to a different URL', async () => {
    const assetUrl = 'https://huggingface.co/litert-community/model.tflite'
    seed([
      {
        url: cacheKeyUrl('model', assetUrl),
        bytes: 900,
        assetUrl,
        base: 'https://huggingface.co/',
        path: 'litert-community/model.tflite',
      },
    ])

    // "model.tflite" under this base is /litert-community/model.tflite, but the
    // recorded path resolves elsewhere once the base is nested a level deeper.
    expect(await listStoredModels('https://huggingface.co/other/')).toEqual([])
  })

  it('treats bases that normalize to the same URL as the same base', async () => {
    seed([
      {
        url: cacheKeyUrl('model', 'https://host/model.tflite'),
        bytes: 100,
        assetUrl: 'https://host/model.tflite',
        base: 'https://host/',
      },
    ])

    expect(await listStoredModels('https://host')).toEqual([{ modelId: 'model', bytes: 100, assets: 1 }])
  })
})

describe('listOrphanedModels', () => {
  it('separates entries written under a superseded base without deleting them', async () => {
    const modelId = 'lfm2.5-encoder-230m'
    const superseded = 'https://app.test/litert-community/encoder/resolve/main/model.tflite'
    const current = `${HF}litert-community/encoder/resolve/main/model.tflite`
    seed([
      { url: cacheKeyUrl(modelId, current), bytes: 500, assetUrl: current, path: 'litert-community/encoder/resolve/main/model.tflite' },
      { url: cacheKeyUrl(modelId, superseded), bytes: 500, assetUrl: superseded, path: 'litert-community/encoder/resolve/main/model.tflite' },
    ])

    expect(await listStoredModels(HF)).toEqual([{ modelId, bytes: 500, assets: 1 }])
    expect(await listOrphanedModels(HF)).toEqual([{ modelId, bytes: 500, assets: 1 }])

    // Listing must not reclaim: switching the base back has to recover the entries.
    expect(await listOrphanedModels(HF)).toHaveLength(1)
    expect(await listStoredModels('https://app.test/')).toEqual([
      { modelId, bytes: 500, assets: 1 },
    ])
  })
})

describe('pruneSupersededModelCaches', () => {
  it('reclaims only the named model entries written under another base', async () => {
    const target = 'lfm2.5-encoder-230m'
    const other = 'qwen3-tts'
    const superseded = 'https://app.test/litert-community/encoder/resolve/main/model.tflite'
    const current = `${HF}litert-community/encoder/resolve/main/model.tflite`
    seed([
      { url: cacheKeyUrl(target, superseded), bytes: 500, assetUrl: superseded, path: 'litert-community/encoder/resolve/main/model.tflite' },
      { url: cacheKeyUrl(target, current), bytes: 500, assetUrl: current, path: 'litert-community/encoder/resolve/main/model.tflite' },
      { url: cacheKeyUrl(other, superseded), bytes: 700, assetUrl: superseded, path: 'litert-community/encoder/resolve/main/model.tflite' },
    ])

    expect(await pruneSupersededModelCaches(target, HF)).toBe(1)
    expect(await pruneSupersededModelCaches(target, HF)).toBe(0)

    const remaining = await listStoredModels(HF)
    expect(remaining).toEqual([{ modelId: target, bytes: 500, assets: 1 }])
  })
})

describe('base-scoped removal', () => {
  // One model id can hold entries under two bases at once, so an unqualified
  // delete from a "previous base" affordance would take the live copy with it.
  const modelId = 'lfm2.5-encoder-230m'
  const superseded = 'https://app.test/litert-community/encoder/resolve/main/model.tflite'
  const current = `${HF}litert-community/encoder/resolve/main/model.tflite`

  function seedBothBases() {
    seed([
      { url: cacheKeyUrl(modelId, current), bytes: 500, assetUrl: current, path: 'litert-community/encoder/resolve/main/model.tflite' },
      { url: cacheKeyUrl(modelId, superseded), bytes: 500, assetUrl: superseded, path: 'litert-community/encoder/resolve/main/model.tflite' },
    ])
  }

  it('reclaims only the previous-base bytes when removing an orphan', async () => {
    seedBothBases()

    await removeOrphanedModel(modelId, HF)

    expect(await listStoredModels(HF)).toEqual([{ modelId, bytes: 500, assets: 1 }])
    expect(await listOrphanedModels(HF)).toEqual([])
  })

  it('leaves the previous-base bytes alone when removing the live copy', async () => {
    seedBothBases()

    await removeStoredModel(modelId, HF)

    expect(await listStoredModels(HF)).toEqual([])
    expect(await listOrphanedModels(HF)).toEqual([{ modelId, bytes: 500, assets: 1 }])
  })

  it('never removes another model that shares the superseded base', async () => {
    seed([
      { url: cacheKeyUrl(modelId, superseded), bytes: 500, assetUrl: superseded, path: 'litert-community/encoder/resolve/main/model.tflite' },
      { url: cacheKeyUrl('qwen3-tts', superseded), bytes: 700, assetUrl: superseded, path: 'litert-community/encoder/resolve/main/model.tflite' },
    ])

    await removeOrphanedModel(modelId, HF)

    expect(await listOrphanedModels(HF)).toEqual([{ modelId: 'qwen3-tts', bytes: 700, assets: 1 }])
  })
})


describe('resolved asset membership', () => {
  it.each([
    ['absolute path', 'https://cdn.test/model.tflite', 'https://old.test/', 'https://new.test/', 'https://cdn.test/model.tflite'],
    ['root-relative path', '/model.tflite', 'https://host/v1/', 'https://host/v2/', 'https://host/model.tflite'],
    ['equivalent base URLs', 'litert-community/model.tflite', HF, `${HF}litert-community`, `${HF}litert-community/model.tflite`],
  ])('preserves a live %s across listing, orphan removal and pruning', async (_name, path, base, currentBase, assetUrl) => {
    seed([{ url: cacheKeyUrl('membership', assetUrl), bytes: 100, assetUrl, base, path }])
    expect(await listStoredModels(currentBase)).toEqual([{ modelId: 'membership', bytes: 100, assets: 1 }])
    expect(await listOrphanedModels(currentBase)).toEqual([])
    expect(await pruneSupersededModelCaches('membership', currentBase)).toBe(0)
    await removeOrphanedModel('membership', currentBase)
    expect(store.size).toBe(1)
    await removeStoredModel('membership', currentBase)
    expect(store.size).toBe(0)
  })

  it('prunes the same nested-base orphan that listing reports, without touching the live copy', async () => {
    seed([
      { url: cacheKeyUrl('membership', 'https://host/v1/model.tflite'), bytes: 100, assetUrl: 'https://host/v1/model.tflite', base: 'https://host/v1/', path: 'model.tflite' },
      { url: cacheKeyUrl('membership', 'https://host/model.tflite'), bytes: 200, assetUrl: 'https://host/model.tflite', base: 'https://host/', path: 'model.tflite' },
    ])
    expect(await listOrphanedModels('https://host/')).toEqual([{ modelId: 'membership', bytes: 100, assets: 1 }])
    expect(await pruneSupersededModelCaches('membership', 'https://host/')).toBe(1)
    expect(await listStoredModels('https://host/')).toEqual([{ modelId: 'membership', bytes: 200, assets: 1 }])
  })

  it('recovers legacy membership from registered original paths', async () => {
    const path = 'https://cdn.test/legacy.tflite'
    registerModelAssets('legacy-absolute', [path])
    seed([{ url: cacheKeyUrl('legacy-absolute', path), bytes: 100, assetUrl: path, base: 'https://old.test/' }])
    expect(await listOrphanedModels('https://new.test/')).toEqual([])
    await removeOrphanedModel('legacy-absolute', 'https://new.test/')
    expect(await pruneSupersededModelCaches('legacy-absolute', 'https://new.test/')).toBe(0)
    expect(store.size).toBe(1)
  })

  it('does not delete anything when the current base is invalid', async () => {
    const assetUrl = `${HF}model.tflite`
    seed([{ url: cacheKeyUrl('invalid-base', assetUrl), bytes: 100, assetUrl, base: HF, path: 'model.tflite' }])
    expect(await listOrphanedModels('http://[')).toEqual([
      { modelId: 'invalid-base', bytes: 100, assets: 1, unverified: true },
    ])
    await removeOrphanedModel('invalid-base', 'http://[')
    await removeStoredModel('invalid-base', 'http://[')
    expect(await pruneSupersededModelCaches('invalid-base', 'http://[')).toBe(0)
    expect(store.size).toBe(1)
  })
})


describe('legacy cache safety and resolver metadata', () => {
  it.each([undefined, 'https://old.test/'])('retains unidentified legacy bytes (recorded base: %s)', async (base) => {
    const assetUrl = 'https://cdn.test/unknown.tflite'
    seed([{ url: cacheKeyUrl('unknown-legacy', assetUrl), bytes: 100, assetUrl, base }])
    expect(await listStoredModels(HF)).toEqual([])
    expect(await listOrphanedModels(HF)).toEqual([
      { modelId: 'unknown-legacy', bytes: 100, assets: 1, unverified: true },
    ])
    await removeOrphanedModel('unknown-legacy', HF)
    await removeStoredModel('unknown-legacy', HF)
    expect(await pruneSupersededModelCaches('unknown-legacy', HF)).toBe(0)
    expect(store.size).toBe(1)
  })

  it('records original paths and reuses the cached bytes across equivalent resolver bases', async () => {
    const asset = { id: 'model', path: 'litert-community/roundtrip.tflite' }
    registerModelAssets('roundtrip', [asset.path])
    const fetchMock = vi.fn().mockResolvedValue(new Response(new Uint8Array([1, 2, 3])))
    vi.stubGlobal('fetch', fetchMock)
    const first = await createModelLibraryAssetResolver(HF).resolve(asset)
    expect([...store.values()][0].path).toBe(asset.path)
    const second = await createModelLibraryAssetResolver(`${HF}litert-community`).resolve(asset)
    expect(new Uint8Array(second)).toEqual(new Uint8Array(first))
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(await listStoredModels(`${HF}litert-community`)).toEqual([
      { modelId: 'roundtrip', bytes: 3, assets: 1 },
    ])
  })
})
