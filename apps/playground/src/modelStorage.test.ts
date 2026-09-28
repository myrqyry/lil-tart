import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import {
  listOrphanedModels,
  listStoredModels,
  pruneSupersededModelCaches,
} from './modelStorage'

const CACHE_PATH = '/__lil_tart_model_cache__/'
const HF = 'https://huggingface.co/'

interface FakeEntry {
  url: string
  bytes: number
  assetUrl: string
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
      return {
        headers: {
          get: (name: string) =>
            name === 'x-lil-tart-bytes'
              ? String(entry.bytes)
              : name === 'x-lil-tart-asset-url'
                ? entry.assetUrl
                : null,
        },
      }
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
      },
      {
        // Written before the base moved to Hugging Face. Keyed on the app origin,
        // so the resolver can never read it again.
        url: cacheKeyUrl(modelId, 'https://app.test/litert-community/encoder/resolve/main/model.tflite'),
        bytes: 500,
        assetUrl: 'https://app.test/litert-community/encoder/resolve/main/model.tflite',
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
      },
    ])

    expect(await listStoredModels('https://huggingface.co/litert-community')).toEqual([])
    expect(await listOrphanedModels('https://huggingface.co/litert-community')).toEqual([
      { modelId: 'model', bytes: 100, assets: 1 },
    ])
  })

  it('claims the base path itself and its descendants without a trailing slash', async () => {
    seed([
      {
        url: cacheKeyUrl('model', 'https://huggingface.co/litert-community/model.tflite'),
        bytes: 100,
        assetUrl: 'https://huggingface.co/litert-community/model.tflite',
      },
    ])

    expect(await listStoredModels('https://huggingface.co/litert-community')).toEqual([
      { modelId: 'model', bytes: 100, assets: 1 },
    ])
  })
})

describe('listOrphanedModels', () => {
  it('separates entries written under a superseded base without deleting them', async () => {
    const modelId = 'lfm2.5-encoder-230m'
    const superseded = 'https://app.test/litert-community/encoder/resolve/main/model.tflite'
    const current = `${HF}litert-community/encoder/resolve/main/model.tflite`
    seed([
      { url: cacheKeyUrl(modelId, current), bytes: 500, assetUrl: current },
      { url: cacheKeyUrl(modelId, superseded), bytes: 500, assetUrl: superseded },
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
      { url: cacheKeyUrl(target, superseded), bytes: 500, assetUrl: superseded },
      { url: cacheKeyUrl(target, current), bytes: 500, assetUrl: current },
      { url: cacheKeyUrl(other, superseded), bytes: 700, assetUrl: superseded },
    ])

    expect(await pruneSupersededModelCaches(target, HF)).toBe(1)
    expect(await pruneSupersededModelCaches(target, HF)).toBe(0)

    const remaining = await listStoredModels(HF)
    expect(remaining).toEqual([{ modelId: target, bytes: 500, assets: 1 }])
  })
})
