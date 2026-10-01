import { describe, expect, it, vi } from 'vitest'
import { createManifestVerifyingAssetResolver } from './manifest-resolver'
import type { AssetResolver, ModelManifest } from '../types'

const bytes = new Uint8Array([1, 2, 3]).buffer

const manifest: ModelManifest = {
  modelId: 'test-model',
  name: 'Test model',
  version: '1',
  capabilities: ['image-classification'],
  backends: { wasm: true },
  memory: { downloadBytes: 3, residentBytes: 3 },
  assets: [{ id: 'model', path: 'model.bin' }],
}

function resolver(overrides: Partial<AssetResolver> = {}): AssetResolver {
  return {
    resolve: vi.fn().mockResolvedValue(bytes),
    ...overrides,
  }
}

async function readStream(stream: ReadableStream<Uint8Array>): Promise<ArrayBuffer> {
  const reader = stream.getReader()
  const chunks: Uint8Array[] = []
  let total = 0
  while (true) {
    const next = await reader.read()
    if (next.done) break
    chunks.push(next.value)
    total += next.value.byteLength
  }
  const result = new Uint8Array(total)
  let offset = 0
  for (const chunk of chunks) {
    result.set(chunk, offset)
    offset += chunk.byteLength
  }
  return result.buffer
}

describe('manifest-verifying asset resolver', () => {
  it('accepts a matching declared byte count', async () => {
    const inner = resolver()
    const verifying = createManifestVerifyingAssetResolver(
      { ...manifest, assets: [{ id: 'model', path: 'model.bin', bytes: 3 }] },
      inner,
    )

    await expect(verifying.resolve({ id: 'model', path: 'model.bin' })).resolves.toEqual(bytes)
  })

  it('rejects a mismatched declared byte count', async () => {
    const verifying = createManifestVerifyingAssetResolver(
      { ...manifest, assets: [{ id: 'model', path: 'model.bin', bytes: 4 }] },
      resolver(),
    )

    await expect(verifying.resolve({ id: 'model', path: 'model.bin' })).rejects.toMatchObject({
      code: 'ASSET_INTEGRITY_FAILED',
      asset: 'model',
    })
  })

  it('accepts a matching declared SHA-256', async () => {
    const verifying = createManifestVerifyingAssetResolver(
      {
        ...manifest,
        assets: [{ id: 'model', path: 'model.bin', sha256: '039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81' }],
      },
      resolver(),
    )

    await expect(verifying.resolve({ id: 'model', path: 'model.bin' })).resolves.toEqual(bytes)
  })

  it('rejects a mismatched declared SHA-256', async () => {
    const verifying = createManifestVerifyingAssetResolver(
      { ...manifest, assets: [{ id: 'model', path: 'model.bin', sha256: '00'.repeat(32) }] },
      resolver(),
    )

    await expect(verifying.resolve({ id: 'model', path: 'model.bin' })).rejects.toMatchObject({
      code: 'ASSET_INTEGRITY_FAILED',
      asset: 'model',
    })
  })

  it('applies manifest integrity facts when a runtime requests an asset by exact path', async () => {
    const verifying = createManifestVerifyingAssetResolver(
      { ...manifest, assets: [{ id: 'semantic-model-id', path: 'model.bin', bytes: 4 }] },
      resolver(),
    )

    await expect(verifying.resolve({ id: 'model.bin', path: 'model.bin' })).rejects.toMatchObject({
      code: 'ASSET_INTEGRITY_FAILED',
      asset: 'semantic-model-id',
    })
  })

  it('does not apply one semantic ID\'s integrity facts to a different path', async () => {
    const verifying = createManifestVerifyingAssetResolver(
      {
        ...manifest,
        assets: [{ id: 'voice', path: 'voices/demo_speaker.npy', bytes: 4 }],
      },
      resolver(),
    )

    await expect(
      verifying.resolve({ id: 'voice', path: 'voices/custom_speaker.npy', optional: true }),
    ).resolves.toEqual(bytes)
  })

  it('can skip SHA-256 above an explicit in-memory ceiling while preserving byte checks', async () => {
    const verifying = createManifestVerifyingAssetResolver(
      {
        ...manifest,
        assets: [{ id: 'model', path: 'model.bin', bytes: 3, sha256: '00'.repeat(32) }],
      },
      resolver(),
      { maxSha256Bytes: 2 },
    )

    await expect(verifying.resolve({ id: 'model', path: 'model.bin' })).resolves.toEqual(bytes)
  })

  it('keeps streamed large assets streaming when SHA-256 is above the ceiling', async () => {
    const inner = resolver({
      stream: vi.fn().mockResolvedValue(
        new ReadableStream({
          start(controller) {
            controller.enqueue(new Uint8Array([1]))
            controller.enqueue(new Uint8Array([2, 3]))
            controller.close()
          },
        }),
      ),
    })
    const verifying = createManifestVerifyingAssetResolver(
      {
        ...manifest,
        assets: [{ id: 'model', path: 'model.bin', bytes: 3, sha256: '00'.repeat(32) }],
      },
      inner,
      { maxSha256Bytes: 2 },
    )

    await expect(readStream(await verifying.stream!({ id: 'model', path: 'model.bin' }))).resolves.toEqual(bytes)
  })

  it('rejects streamed byte-length mismatches even when large-asset SHA-256 is skipped', async () => {
    const inner = resolver({
      stream: vi.fn().mockResolvedValue(
        new ReadableStream({
          start(controller) {
            controller.enqueue(new Uint8Array([1, 2]))
            controller.close()
          },
        }),
      ),
    })
    const verifying = createManifestVerifyingAssetResolver(
      {
        ...manifest,
        assets: [{ id: 'model', path: 'model.bin', bytes: 3, sha256: '00'.repeat(32) }],
      },
      inner,
      { maxSha256Bytes: 2 },
    )

    await expect(readStream(await verifying.stream!({ id: 'model', path: 'model.bin' })))
      .rejects.toMatchObject({ code: 'ASSET_INTEGRITY_FAILED', asset: 'model' })
  })

  it('does not require absent verification facts', async () => {
    await expect(
      createManifestVerifyingAssetResolver(manifest, resolver()).resolve({ id: 'model', path: 'model.bin' }),
    ).resolves.toEqual(bytes)
  })

  it('includes asset ID and expected and actual values in failures', async () => {
    const verifying = createManifestVerifyingAssetResolver(
      { ...manifest, assets: [{ id: 'model', path: 'model.bin', bytes: 4 }] },
      resolver(),
    )

    await expect(verifying.resolve({ id: 'model', path: 'model.bin' })).rejects.toThrow(
      'model: expected 4 bytes, received 3',
    )
  })

  it('forwards progress and abort options to the inner resolver', async () => {
    const inner = resolver()
    const signal = new AbortController().signal
    const onProgress = vi.fn()
    const verifying = createManifestVerifyingAssetResolver(manifest, inner)

    await verifying.resolve({ id: 'model', path: 'model.bin' }, { signal, onProgress })

    expect(inner.resolve).toHaveBeenCalledWith({ id: 'model', path: 'model.bin' }, { signal, onProgress })
  })

  it('validates streamed data after the complete buffer is available', async () => {
    const inner = resolver({
      stream: vi.fn().mockResolvedValue(
        new ReadableStream({
          start(controller) {
            controller.enqueue(new Uint8Array([1]))
            controller.enqueue(new Uint8Array([2, 3]))
            controller.close()
          },
        }),
      ),
    })
    const verifying = createManifestVerifyingAssetResolver(
      { ...manifest, assets: [{ id: 'model', path: 'model.bin', bytes: 3 }] },
      inner,
    )

    await expect(readStream(await verifying.stream!({ id: 'model', path: 'model.bin' }))).resolves.toEqual(bytes)
    expect(inner.stream).toHaveBeenCalled()
  })
})
